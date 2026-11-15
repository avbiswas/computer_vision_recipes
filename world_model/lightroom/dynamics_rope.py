"""World model v2: strided context + 3D RoPE (+ anchor), causal in time, MaskGIT in each frame.

Context = frames at these offsets before the target (covers ~3 s at 10 fps):
    OFFSETS = 32, 16, 8, 4, 3, 2, 1, [0 = target]
plus an "anchor": an older real frame (layout reference), placed at its true time offset.

Positions (no learned position embeddings):
  * spatial attention:  2D RoPE over (row, col) of the 16x16 token grid
  * temporal attention: 1D RoPE over the REAL time offset of each frame (-32 ... 0, anchor -t)
Rotary embeddings come from the `rotary-embedding-torch` library.

The layer parameter names match nn.TransformerEncoderLayer exactly, so weights from the
earlier models (dynamics.WorldModel) load straight in; only the old learned position /
frame-slot embeddings are dropped.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from dynamics import N_CODES, MASK, TOKENS, N_ACTIONS, NO_ACTION

OFFSETS = (32, 16, 8, 4, 3, 2, 1, 0)   # last one is the frame being predicted
GRID = 16


class RoPEAttention(nn.Module):
    """Multi-head attention with rotary positions. Same parameter names/shapes as
    nn.MultiheadAttention (in_proj_weight, in_proj_bias, out_proj) for weight reuse."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x, freqs, causal=False):
        B, L, d = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        if freqs is not None:                                              # (None = ablation without RoPE)
            q, k = apply_rotary_emb(freqs, q), apply_rotary_emb(freqs, k)  # rotate by position
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, d))


class RoPELayer(nn.Module):
    """Pre-norm transformer layer; parameter names match nn.TransformerEncoderLayer."""

    def __init__(self, d, heads):
        super().__init__()
        self.self_attn = RoPEAttention(d, heads)
        self.linear1, self.linear2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, freqs, causal=False):
        x = x + self.self_attn(self.norm1(x), freqs, causal)
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class STBlockRoPE(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.spatial, self.temporal = RoPELayer(d, heads), RoPELayer(d, heads)

    def forward(self, x, s_freqs, t_freqs):  # x: (B, F, N, d)
        B, Fr, N, d = x.shape
        x = self.spatial(x.reshape(B * Fr, N, d), s_freqs).view(B, Fr, N, d)
        xt = x.transpose(1, 2).reshape(B * N, Fr, d)
        tf = None if t_freqs is None else t_freqs[:, None].expand(B, N, Fr, -1).reshape(B * N, 1, Fr, -1)
        xt = self.temporal(xt, tf, causal=True)                                # frames are time-ordered
        return xt.view(B, N, Fr, d).transpose(1, 2)


class WorldModelRoPE(nn.Module):
    rope = True

    def __init__(self, d=384, layers=8, heads=8, offsets=OFFSETS, use_rope=True):
        super().__init__()
        self.offsets, self.use_rope = tuple(offsets), use_rope
        self.K = len(self.offsets) - 1                # context slots (for code that asks for wm.K)
        hd = d // heads
        self.tok = nn.Embedding(N_CODES + 1, d)
        self.act = nn.Embedding(N_ACTIONS + 1, d)
        self.anchor_emb = nn.Parameter(torch.randn(1, d) * 0.02)   # marks the anchor frame
        self.rot_space = RotaryEmbedding(dim=hd // 2)  # half the head dim for rows, half for cols
        self.rot_time = RotaryEmbedding(dim=hd)
        if not use_rope:  # ablation: learned absolute positions (per grid cell + per window slot)
            self.pos = nn.Parameter(torch.randn(TOKENS, d) * 0.02)
            self.frame = nn.Parameter(torch.randn(len(self.offsets), d) * 0.02)
        self.blocks = nn.ModuleList([STBlockRoPE(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CODES)
        r = torch.arange(GRID).repeat_interleave(GRID).float()
        c = torch.arange(GRID).repeat(GRID).float()
        self.register_buffer("rows", r, persistent=False)
        self.register_buffer("cols", c, persistent=False)

    def forward_seq(self, seq, frame_actions, times, anchor, anchor_time):
        """
        seq:           (B, F, 256) code ids (MASK where hidden), time-ordered, last = target
        frame_actions: (B, F)      action that produced each frame (NO_ACTION if unknown)
        times:         (B, F)      time offset of each frame relative to the target (<= 0)
        anchor:        (B, 256)    code ids of an older real frame
        anchor_time:   (B,)        its time offset (<= the oldest context frame)
        returns logits (B, F, 256, N_CODES)
        """
        B = seq.shape[0]
        none = torch.full((B, 1), NO_ACTION, device=seq.device, dtype=frame_actions.dtype)
        seq = torch.cat([anchor[:, None], seq], 1)
        fa = torch.cat([none, frame_actions], 1)
        tt = torch.cat([anchor_time[:, None].float(), times.float()], 1)
        x = self.tok(seq) + self.act(fa)[:, :, None]
        x = torch.cat([x[:, :1] + self.anchor_emb, x[:, 1:]], 1)                # mark the anchor
        if self.use_rope:
            s_freqs = torch.cat([self.rot_space(self.rows), self.rot_space(self.cols)], -1)  # (256, hd)
            t_freqs = self.rot_time(tt)                                                       # (B, F+1, hd)
        else:
            x = x + self.pos
            x = torch.cat([x[:, :1], x[:, 1:] + self.frame[:, None]], 1)
            s_freqs = t_freqs = None
        for blk in self.blocks:
            x = blk(x, s_freqs, t_freqs)
        return self.head(self.norm(x[:, 1:]))


def window_indices(t, offsets=OFFSETS):
    """Frame indices for a target at index t (tensor (B,)): clamped at 0 for episode starts."""
    off = torch.tensor(offsets, device=t.device)
    return (t[:, None] - off).clamp(min=0)                                    # (B, F)


def frame_actions_for(idx, actions_of):
    """Action that produced each window frame, only where the previous window slot is exactly
    the previous frame (the dense part); NO_ACTION for strided / duplicated frames.
    actions_of(i) -> action taken at frame i (moving i -> i+1)."""
    prev_ok = torch.zeros_like(idx, dtype=torch.bool)
    prev_ok[:, 1:] = (idx[:, 1:] - idx[:, :-1] == 1)
    acts = actions_of((idx - 1).clamp(min=0))
    return torch.where(prev_ok & (idx >= 1), acts, torch.full_like(acts, NO_ACTION)), prev_ok


def load_from_old(model, state):
    """Warm-start from a dynamics.WorldModel checkpoint (drops learned pos/frame embeddings,
    unless this is the no-RoPE ablation, which keeps pos and the newest frame-slot vectors)."""
    if model.use_rope:
        state = {k: v for k, v in state.items() if k not in ("pos", "frame")}
    else:
        state = {**state, "frame": state["frame"][-len(model.offsets):]}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return [m for m in missing if "rot_" not in m], unexpected


@torch.no_grad()
def predict_next_rope(model, seq_ctx, frame_actions, times, anchor, anchor_time, passes=1, temperature=0.0):
    """seq_ctx: (B, F-1, 256) clean context; predicts the target slot (1 pass or MaskGIT)."""
    import math
    B = seq_ctx.shape[0]
    target = torch.full((B, TOKENS), MASK, device=seq_ctx.device, dtype=seq_ctx.dtype)
    for i in range(passes):
        logits = model.forward_seq(torch.cat([seq_ctx, target[:, None]], 1), frame_actions, times,
                                   anchor, anchor_time)[:, -1].float()
        probs = F.softmax(logits / max(temperature, 1e-6), -1) if temperature > 0 else F.softmax(logits, -1)
        guess = torch.multinomial(probs.flatten(0, 1), 1).view(B, TOKENS) if temperature > 0 else logits.argmax(-1)
        conf = probs.gather(-1, guess[..., None]).squeeze(-1)
        unknown = target == MASK
        guess = torch.where(unknown, guess, target)
        conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))
        n_hidden = int(TOKENS * math.cos(math.pi / 2 * (i + 1) / passes))
        if n_hidden == 0:
            return guess
        target = guess.scatter(1, conf.argsort(1)[:, :n_hidden], MASK)
    return target

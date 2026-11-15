"""Stage B - the world model: given the last K frames (as tokens) and the actions,
predict the tokens of the next frame.

Input sequence to the transformer (K = context length, 256 tokens per frame):

  [ frame t-K+1 | frame t-K+2 | ... | frame t | NEXT FRAME (masked) ]
     each token = code embedding + position-in-frame + which-frame + action that led to that frame

Training is MaskGIT-style: hide a random fraction of the next frame's tokens
(half the time: all of them) and predict the hidden ones with cross-entropy.
At play time we can fill the frame in 1 pass (fast) or N refinement passes
(sharper, picks one consistent future when the world is random - the fireflies).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CODES = 512           # VQ-VAE codebook size
MASK = N_CODES          # extra token id meaning "unknown, please predict"
TOKENS = 256            # 16 x 16 tokens per frame
N_ACTIONS = 5           # stay / forward / backward / turn left / turn right
NO_ACTION = N_ACTIONS   # used for the oldest context frame (its action is unknown)


class STBlock(nn.Module):
    """Spatio-temporal block (as in Genie), built from two stock PyTorch transformer layers.
    Cheaper than one layer attending over all frames x tokens at once.
      1. spatial:  the 256 tokens of each frame attend to each other        (where things are)
      2. temporal: each grid cell attends to the same cell in the other frames (how it changes)"""

    def __init__(self, d, heads):
        super().__init__()
        layer = lambda: nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.0,
                                                   activation="gelu", batch_first=True, norm_first=True)
        self.spatial, self.temporal = layer(), layer()

    def forward(self, x):  # x: (B, F frames, N tokens, d)
        B, Fr, N, d = x.shape
        x = self.spatial(x.reshape(B * Fr, N, d)).view(B, Fr, N, d)
        x = self.temporal(x.transpose(1, 2).reshape(B * N, Fr, d))
        return x.view(B, N, Fr, d).transpose(1, 2)


class WorldModel(nn.Module):
    def __init__(self, K=4, d=256, layers=6, heads=8, anchor=False):
        super().__init__()
        self.K, self.use_anchor = K, anchor
        self.tok = nn.Embedding(N_CODES + 1, d)
        self.pos = nn.Parameter(torch.randn(TOKENS, d) * 0.02)      # where in the 16x16 grid
        self.frame = nn.Parameter(torch.randn(K + 1, d) * 0.02)     # which frame in the window
        self.act = nn.Embedding(N_ACTIONS + 1, d)
        if anchor:  # an extra "reference" frame from the start of the episode (boxes never move)
            self.anchor_emb = nn.Parameter(torch.randn(1, d) * 0.02)
        self.blocks = nn.Sequential(*[STBlock(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CODES)

    def forward(self, ctx, actions, target, anchor=None):
        """
        ctx:     (B, K, 256)  code ids of the K past frames (oldest first)
        actions: (B, K)       actions[:, j] moves frame j -> frame j+1 (last one -> target)
        target:  (B, 256)     code ids of the next frame, with MASK where unknown
        anchor:  (B, 256)     optional: code ids of an earlier real frame (if use_anchor)
        returns logits (B, 256, N_CODES) for the next frame
        """
        B, K, _ = ctx.shape
        seq = torch.cat([ctx, target[:, None]], 1)                         # (B, K+1, 256)
        none = torch.full((B, 1), NO_ACTION, device=ctx.device, dtype=actions.dtype)
        frame_actions = torch.cat([none, actions], 1)                      # (B, K+1)
        frame_emb = self.frame
        if self.use_anchor:                                                # prepend the anchor frame
            seq = torch.cat([anchor[:, None], seq], 1)
            frame_actions = torch.cat([none, frame_actions], 1)
            frame_emb = torch.cat([self.anchor_emb, frame_emb], 0)
        x = (self.tok(seq) + self.pos + frame_emb[:, None]
             + self.act(frame_actions)[:, :, None])                        # (B, frames, 256, d)
        x = self.blocks(x)                                                 # (B, K+1, 256, d)
        return self.head(self.norm(x[:, -1]))                              # next frame only


def mask_target(target, full_mask_prob=0.5):
    """MaskGIT training mask. Returns (masked target, bool mask of hidden positions)."""
    B, N = target.shape
    u = torch.rand(B, device=target.device)
    ratio = torch.cos(u * math.pi / 2)                       # cosine schedule, mostly high ratios
    ratio = torch.where(torch.rand(B, device=target.device) < full_mask_prob, 1.0, ratio)
    n_hide = (ratio * N).ceil().clamp(min=1)
    rank = torch.rand(B, N, device=target.device).argsort(1).argsort(1)  # random order
    hidden = rank < n_hide[:, None]
    return target.masked_fill(hidden, MASK), hidden


def corrupt_context(ctx, max_p=0.1):
    """Anti-drift: replace a few context tokens with random codes, so the model learns
    to cope with its own imperfect predictions at play time."""
    p = torch.rand(ctx.shape[0], 1, 1, device=ctx.device) * max_p
    noise = torch.randint_like(ctx, N_CODES)
    return torch.where(torch.rand_like(ctx, dtype=torch.float) < p, noise, ctx)


@torch.no_grad()
def predict_next(model, ctx, actions, passes=1, temperature=1.0, anchor=None):
    """Fill in the next frame. passes=1: every token in one shot (argmax if temperature=0).
    passes>1: MaskGIT - commit the most confident tokens each pass, re-predict the rest."""
    B = ctx.shape[0]
    target = torch.full((B, TOKENS), MASK, device=ctx.device, dtype=ctx.dtype)
    for i in range(passes):
        logits = model(ctx, actions, target, anchor).float()
        if temperature > 0:
            probs = F.softmax(logits / temperature, -1)
            guess = torch.multinomial(probs.flatten(0, 1), 1).view(B, TOKENS)
        else:
            probs = F.softmax(logits, -1)
            guess = logits.argmax(-1)
        conf = probs.gather(-1, guess[..., None]).squeeze(-1)
        unknown = target == MASK
        guess = torch.where(unknown, guess, target)                # keep already-committed tokens
        conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))
        n_still_hidden = int(TOKENS * math.cos(math.pi / 2 * (i + 1) / passes))
        if n_still_hidden == 0:
            return guess
        hide = conf.argsort(1)[:, :n_still_hidden]                 # least confident stay hidden
        target = guess.scatter(1, hide, MASK)
    return target


def load_resized(model, state):
    """Load weights from a model with a different context length K (or without an anchor):
    the old frame embeddings fill the newest slots, older slots copy the oldest one."""
    old, new = state["frame"], model.frame.data
    if old.shape != new.shape:
        n = min(len(old), len(new))
        resized = old[:1].repeat(len(new), 1)
        resized[-n:] = old[-n:]
        state = {**state, "frame": resized}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing

"""Train world model v2 (dynamics_rope.WorldModelRoPE): strided context + 3D RoPE + anchor,
causal in time, MaskGIT in every frame, loss on the dense recent frames.

  python train_rope.py --cache cache_long --d 384 --layers 8 --init runs/gpu4/best.pt --out runs/rope29
"""
import argparse
import contextlib
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from dynamics import mask_target, corrupt_context, MASK
from dynamics_rope import WorldModelRoPE, OFFSETS, window_indices, frame_actions_for, load_from_old
from vqvae import VQVAE, DEVICE
from progress import ProgressDreams

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_long")
    ap.add_argument("--out", default="runs/rope")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init", default=None, help="warm start from an old dynamics.WorldModel checkpoint")
    ap.add_argument("--change_w", type=float, default=2.0)
    ap.add_argument("--lamp_w", type=float, default=8.0)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--clean_p", type=float, default=0.5)
    ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
    ap.add_argument("--snapshot_every", type=int, default=1000)
    ap.add_argument("--offsets", default=",".join(map(str, OFFSETS)), help="context offsets, last must be 0")
    ap.add_argument("--no_rope", action="store_true", help="ablation: learned positions instead of RoPE")
    a = ap.parse_args()
    a.rope = True
    offsets = tuple(int(x) for x in a.offsets.split(","))
    a.use_rope = not a.no_rope
    os.makedirs(f"{a.out}/ckpts", exist_ok=True)

    dev = DEVICE if DEVICE == "cuda" else "cpu"
    tokens = torch.from_numpy(np.load(f"{a.cache}/tokens.npy")).to(dev)          # int16 (N, T+1, 256)
    actions = torch.from_numpy(np.load(f"{a.cache}/actions.npy").astype(np.int64)).to(dev)
    lamp = torch.from_numpy(np.load(f"{a.cache}/lamp.npy")).to(dev)              # (N, T+1, 2)
    N, T1, _ = tokens.shape
    n_train = int(N * 0.95)
    c = (torch.arange(16, device=dev) + 0.5) / 16
    gy, gx = torch.meshgrid(c, c, indexing="ij")
    grid = torch.stack([gx.flatten(), gy.flatten()], -1)                            # (256, 2)
    g = torch.Generator().manual_seed(0)

    def batch(lo, hi, B):
        ep = torch.randint(lo, hi, (B,), generator=g).to(dev)
        t = torch.randint(1, T1, (B,), generator=g).to(dev)                         # target frame index
        idx = window_indices(t, offsets)                                            # (B, F)
        seq = tokens[ep[:, None], idx].long()                                       # (B, F, 256)
        fa, dense = frame_actions_for(idx, lambda i: actions[ep[:, None], i])
        times = (idx - t[:, None]).float()
        a_max = (t - offsets[0]).clamp(min=0)                                        # anchor <= oldest frame
        a_idx = (torch.rand(B, generator=g).to(dev) * (a_max + 1)).long()
        anchor = tokens[ep, a_idx].long()
        near = (lamp[ep[:, None], idx][:, :, None] - grid).norm(dim=-1) < 0.12        # (B, F, 256)
        sup = dense & (idx >= 1)                                                     # frames we train on
        return [x.to(DEVICE) for x in (seq, fa, times, anchor, (a_idx - t).float(), near, sup)]

    model = WorldModelRoPE(d=a.d, layers=a.layers, offsets=offsets, use_rope=a.use_rope).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.03)
    print(f"RoPE world model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"offsets {offsets}, rope {a.use_rope}, data {tuple(tokens.shape)} on {DEVICE}", flush=True)
    ckpt, first = f"{a.out}/world_model.pt", 1
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); first = ck["step"] + 1
        print(f"resumed from step {ck['step']}", flush=True)
    elif a.init:
        missing, unexpected = load_from_old(model, torch.load(a.init, map_location=DEVICE)["model"])
        print(f"warm start from {a.init}; new: {missing}; dropped: {unexpected}", flush=True)
    vq = VQVAE().to(DEVICE).eval()
    vq.load_state_dict(torch.load(a.vqvae, map_location=DEVICE))
    progress = ProgressDreams(vq, f"{a.out}/progress", K=8)
    if first == 1:
        progress.snapshot(model, 0)

    t0 = time.time()
    for step in range(first, a.steps + 1):
        seq, fa, times, anchor, a_time, near, sup = batch(0, n_train, a.batch)
        B, Fr, _ = seq.shape
        inp, hidden = mask_target(seq.reshape(B * Fr, 256))
        inp, hidden = inp.view(B, Fr, 256), hidden.view(B, Fr, 256)
        clean = torch.rand(B, device=DEVICE) < a.clean_p                          # inference-like samples
        hidden[clean, :-1] = False
        inp = torch.where(hidden, torch.full_like(seq, MASK), corrupt_context(seq, a.noise))
        with amp():
            logits = model.forward_seq(inp, fa, times, anchor, a_time).float()     # (B, F, 256, 512)
        lossmask = hidden & sup[:, :, None]
        per_tok = F.cross_entropy(logits[lossmask], seq[lossmask], reduction="none")
        prev = torch.cat([seq[:, :1], seq[:, :-1]], 1)
        w = 1 + a.change_w * (seq != prev)[lossmask].float() + a.lamp_w * near[lossmask].float()
        loss = (w * per_tok).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad(), amp():  # held-out: clean context, predict the target in ONE pass
                seq, fa, times, anchor, a_time, *_ = batch(n_train, N, 64)
                inp = seq.clone(); inp[:, -1] = MASK
                pred = model.forward_seq(inp, fa, times, anchor, a_time)[:, -1].argmax(-1)
                acc = (pred == seq[:, -1]).float().mean().item()
                copy = (seq[:, -2] == seq[:, -1]).float().mean().item()
            model.train()
            print(f"step {step:6d}  loss {loss.item():.3f}  val 1-pass acc {acc:.3f} (copy {copy:.3f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
            torch.save({"model": model.state_dict(), "args": vars(a), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step}, ckpt)
            torch.save({"model": model.state_dict(), "args": vars(a), "step": step},
                       f"{a.out}/ckpts/wm_{step:06d}.pt")
        if step % a.snapshot_every == 0 or step == a.steps:
            e = progress.snapshot(model, step)
            print(f"           dream error  1-10: {e[0]:.1f}  11-30: {e[1]:.1f}  31+: {e[2]:.1f}  "
                  f"(motion {e[3]:.2f}, real {e[4]:.2f})", flush=True)


if __name__ == "__main__":
    main()

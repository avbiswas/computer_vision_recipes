"""Train the world model on token sequences (teacher forcing: the context is always
real recorded frames; the model never sees its own predictions during training).

usage: python train_dynamics.py --K 4 --steps 12000 --out runs/dyn_k4
"""
import argparse
import contextlib
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from dynamics import WorldModel, mask_target, corrupt_context, predict_next, load_resized
from vqvae import VQVAE, DEVICE
from progress import ProgressDreams

# On NVIDIA GPUs: TF32 matmuls + bf16 autocast (roughly 2-3x faster, same results in practice)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/dyn_k4")
    ap.add_argument("--K", type=int, default=4, help="context frames")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6, help="ST blocks (each = 2 stock transformer layers)")
    ap.add_argument("--change_w", type=float, default=0.0,
                    help="extra loss weight on tokens that differ from the last frame (fights 'copy' collapse)")
    ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
    ap.add_argument("--snapshot_every", type=int, default=500, help="dream videos during training")
    ap.add_argument("--self_p", type=float, default=0.0,
                    help="fraction of batches whose newest context frames are the model's OWN predictions")
    ap.add_argument("--self_max", type=int, default=3, help="max self-predicted context frames")
    ap.add_argument("--noise", type=float, default=0.1, help="max fraction of context tokens randomised")
    ap.add_argument("--init", default=None, help="start from these weights with a fresh LR schedule")
    ap.add_argument("--anchor", action="store_true", help="add an earlier real frame as a fixed reference")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data_dev = DEVICE if DEVICE == "cuda" else "cpu"  # on a big GPU, keep all tokens in GPU memory
    tokens = torch.from_numpy(np.load(f"{a.cache}/tokens.npy").astype(np.int64)).to(data_dev)  # (N, T+1, 256)
    actions = torch.from_numpy(np.load(f"{a.cache}/actions.npy").astype(np.int64)).to(data_dev)  # (N, T)
    N, T1, _ = tokens.shape
    n_train = int(N * 0.95)  # same held-out episodes as the VQ-VAE
    g = torch.Generator().manual_seed(0)

    def batch(lo, hi, B, R=0):
        """K real context frames + target. With R > 0 the window is R frames longer and the
        caller rolls the model forward R steps, so the last R context frames are its own."""
        ep = torch.randint(lo, hi, (B,), generator=g).to(data_dev)
        t = torch.randint(0, T1 - a.K - R, (B,), generator=g).to(data_dev)  # window start
        idx = t[:, None] + torch.arange(a.K + R + 1, device=data_dev)
        seq = tokens[ep[:, None], idx].to(DEVICE)                   # (B, K+R+1, 256)
        act = actions[ep[:, None], idx[:, :-1]].to(DEVICE)          # (B, K+R)
        # anchor: a random earlier real frame (at or before the window start)
        a_idx = (torch.rand(B, generator=g).to(data_dev) * (t + 1)).long()
        batch.anchor = tokens[ep, a_idx].to(DEVICE) if a.anchor else None
        return seq, act

    @torch.no_grad()
    def self_rollout(seq, act, R):
        """Exposure-bias fix: start from K real frames, let the model predict R frames
        (exactly as in a dream), and return that imperfect context + the REAL next frame."""
        ctx = seq[:, :a.K]
        model.eval()
        for r in range(R):
            with amp():
                pred = predict_next(model, ctx, act[:, r:r + a.K], passes=1, temperature=0,
                                    anchor=batch.anchor)
            ctx = torch.cat([ctx[:, 1:], pred[:, None]], 1)
        model.train()
        return ctx, act[:, R:R + a.K], seq[:, a.K + R]

    model = WorldModel(K=a.K, d=a.d, layers=a.layers, anchor=a.anchor).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.03)
    print(f"world model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
          f"K={a.K} on {DEVICE}", flush=True)
    if a.init:
        missing = load_resized(model, torch.load(a.init, map_location=DEVICE)["model"])
        print(f"initialised from {a.init}  (new params: {missing})", flush=True)
    ckpt, first = f"{a.out}/world_model.pt", 1
    if os.path.exists(ckpt):  # resume after a crash / restart
        ck = torch.load(ckpt, map_location=DEVICE)
        if "opt" in ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); first = ck["step"] + 1
            print(f"resumed from step {ck['step']}", flush=True)
    vq = VQVAE().to(DEVICE).eval()
    vq.load_state_dict(torch.load(a.vqvae, map_location=DEVICE))
    progress = ProgressDreams(vq, f"{a.out}/progress", K=a.K)
    if first == 1:
        progress.snapshot(model, 0)  # the untrained model's "dream" = noise
    t0 = time.time()
    for step in range(first, a.steps + 1):
        R = int(torch.randint(1, a.self_max + 1, (1,), generator=g)) \
            if torch.rand(1, generator=g).item() < a.self_p else 0
        seq, act = batch(0, n_train, a.batch, R)
        if R:
            ctx, act, tgt = self_rollout(seq, act, R)
        else:
            ctx, act, tgt = seq[:, :a.K], act, seq[:, a.K]
        inp, hidden = mask_target(tgt)
        with amp():
            logits = model(corrupt_context(ctx, a.noise), act, inp, batch.anchor).float()
        per_token = F.cross_entropy(logits[hidden], tgt[hidden], reduction="none")
        weight = 1 + a.change_w * (tgt != ctx[:, -1])[hidden].float()  # moving parts count more
        loss = (weight * per_token).sum() / weight.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():  # held-out: predict a fully hidden next frame in ONE pass
                seq, act = batch(n_train, N, 64)
                ctx, tgt = seq[:, :a.K], seq[:, a.K]
                full = torch.full_like(tgt, 512)
                pred = model(ctx, act, full, batch.anchor).argmax(-1)
                acc = (pred == tgt).float().mean().item()
                copy = (ctx[:, -1] == tgt).float().mean().item()  # baseline: "nothing changes"
            model.train()
            print(f"step {step:6d}  loss {loss.item():.3f}  val 1-pass token acc {acc:.3f} "
                  f"(copy-last-frame baseline {copy:.3f})  {time.time() - t0:.0f}s", flush=True)
        if step % a.snapshot_every == 0 or step == a.steps:
            err = progress.snapshot(model, step)
            print(f"           dream pixel error vs real  steps 1-10: {err[0]:.1f}  11-30: {err[1]:.1f}  "
                  f"31-120: {err[2]:.1f}  (dream motion {err[3]:.2f}, real {err[4]:.2f})", flush=True)
        if step % 500 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "args": vars(a), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step}, ckpt)
            os.makedirs(f"{a.out}/ckpts", exist_ok=True)  # keep every checkpoint (model only)
            torch.save({"model": model.state_dict(), "args": vars(a), "step": step},
                       f"{a.out}/ckpts/wm_{step:06d}.pt")


if __name__ == "__main__":
    main()

"""Diagnose: one-step prediction (always from REAL past frames) vs. free-running dream.

If one-step is good but the dream melts -> errors compound (drift).
If one-step is already bad -> the model hasn't learned the dynamics yet.

  python eval_onestep.py --wm runs/dyn_k4/world_model.pt
"""
import argparse
import numpy as np
import torch
from PIL import Image
from vqvae import VQVAE, DEVICE
from dream import load
from dynamics import predict_next

ap = argparse.ArgumentParser()
ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
ap.add_argument("--wm", default="runs/dyn_k4/world_model.pt")
ap.add_argument("--out", default="runs/dyn_k4/onestep.png")
a = ap.parse_args()
vq, wm = load(a.vqvae, a.wm)
K = wm.K
tokens = torch.from_numpy(np.load("cache/tokens.npy")[-40:].astype(np.int64)).to(DEVICE)  # held-out
actions = torch.from_numpy(np.load("cache/actions.npy")[-40:].astype(np.int64)).to(DEVICE)
frames = np.load("cache/frames.npy", mmap_mode="r")[-40:]

err_pred, err_copy, err_recon, tiles = [], [], [], []
for ep in range(0, 40, 4):
    for t in range(K, 60, 7):
        ctx, act = tokens[ep, t - K:t][None], actions[ep, t - K:t][None]
        pred = predict_next(wm, ctx, act, passes=1, temperature=0)
        img = VQVAE.to_image(vq.decode(pred.view(1, 16, 16)))[0].cpu().numpy().astype(float)
        rec = VQVAE.to_image(vq.decode(tokens[ep, t].view(1, 16, 16)))[0].cpu().numpy().astype(float)
        real, prev = frames[ep, t].astype(float), frames[ep, t - 1].astype(float)
        err_pred.append(np.abs(img - real).mean())
        err_copy.append(np.abs(prev - real).mean())
        err_recon.append(np.abs(rec - real).mean())
        if len(tiles) < 6 and t == K + 14:
            tiles.append(np.concatenate([frames[ep, t], img.astype(np.uint8)], 0))
print(f"mean abs pixel error (0-255) for the NEXT frame, from real context:")
print(f"  world model 1-step : {np.mean(err_pred):.2f}")
print(f"  copy last frame    : {np.mean(err_copy):.2f}")
print(f"  VQ-VAE ceiling     : {np.mean(err_recon):.2f}   (perfect token prediction)")
Image.fromarray(np.concatenate(tiles, 1)).resize((6 * 256, 512), Image.NEAREST).save(a.out)

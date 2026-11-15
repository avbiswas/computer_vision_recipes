"""Run the trained VQ-VAE encoder over every recorded frame once.

  cache/tokens.npy  int16 (N_episodes, T+1, 256)   (16x16 code ids per frame, flattened)

The dynamics model then trains on these integers only - it never sees pixels.

usage: python tokenize_frames.py
"""
import argparse
import numpy as np
import torch
from vqvae import VQVAE, DEVICE

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
a = ap.parse_args()

model = VQVAE().to(DEVICE).eval()
model.load_state_dict(torch.load(a.vqvae, map_location=DEVICE))
frames = np.load(f"{a.cache}/frames.npy", mmap_mode="r")
N, T1 = frames.shape[:2]
tokens = np.zeros((N, T1, 256), np.int16)
for i in range(0, N, 16):  # 16 episodes (~1000 frames) per batch
    x = torch.from_numpy(np.ascontiguousarray(frames[i:i + 16])).to(DEVICE)
    ids = model.encode(VQVAE.to_input(x.flatten(0, 1)))
    tokens[i:i + 16] = ids.view(-1, T1, 256).cpu().numpy()
    if i % 400 == 0:
        print(f"{i}/{N}", flush=True)
np.save(f"{a.cache}/tokens.npy", tokens)
print("saved", tokens.shape, "codes used:", len(np.unique(tokens)))

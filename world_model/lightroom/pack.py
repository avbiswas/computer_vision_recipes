"""Pack the recorded episodes into two big arrays (memory-mapped .npy files).

  cache/frames.npy   uint8 (N_episodes, T+1, 64, 64, 3)
  cache/actions.npy  int8  (N_episodes, T)

usage: python pack.py --data data --out cache
"""
import argparse
import glob
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="cache")
a = ap.parse_args()

files = sorted(glob.glob(f"{a.data}/ep_*.npz"))
first = np.load(files[0])
os.makedirs(a.out, exist_ok=True)
frames = np.lib.format.open_memmap(f"{a.out}/frames.npy", "w+", np.uint8,
                                   (len(files),) + first["frames"].shape)
actions = np.zeros((len(files),) + first["actions"].shape, np.int8)
for i, f in enumerate(files):
    ep = np.load(f)
    frames[i] = ep["frames"]
    actions[i] = ep["actions"]
frames.flush()
np.save(f"{a.out}/actions.npy", actions)
print(f"packed {len(files)} episodes -> {a.out}  frames {frames.shape}")

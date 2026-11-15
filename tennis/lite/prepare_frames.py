"""
Step 1 - Shrink every frame once, before training.

Training reads 3 frames per sample. Decoding three 1280x720 JPEGs every time is slow,
so we resize every frame to 512x288 grayscale ONCE and store them all in one big
uint8 array on disk (np.memmap). Reading a frame is then just an array lookup.

Output: data/frames_512x288.npy      shape (num_frames, 288, 512),    same order as data/tennis.jsonl
        data/frames_512x288_rgb.npy  shape (num_frames, 288, 512, 3)  with --color
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # .../tennis
IMAGES = ROOT / "raw/TrackNet/Dataset"
W, H = 512, 288


def load_small_gray(file):
    img = cv2.imread(str(IMAGES / file), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def load_small_rgb(file):
    img = cv2.cvtColor(cv2.imread(str(IMAGES / file)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", action="store_true", help="keep R,G,B (3x bigger file)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(ROOT / "data/tennis.jsonl")]
    name, shape, load = (("frames_512x288_rgb.npy", (len(rows), H, W, 3), load_small_rgb) if args.color
                         else ("frames_512x288.npy", (len(rows), H, W), load_small_gray))
    frames = np.lib.format.open_memmap(ROOT / "data" / name, mode="w+", dtype=np.uint8, shape=shape)
    with Pool(8) as pool:
        for i, img in enumerate(pool.imap(load, [r["file"] for r in rows], chunksize=64)):
            frames[i] = img
    frames.flush()
    print(f"saved {len(rows)} frames -> data/{name}")

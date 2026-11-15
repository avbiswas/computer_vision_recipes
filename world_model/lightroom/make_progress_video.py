"""Stitch training snapshots into one "watch it learn" MP4.

  python make_progress_video.py runs/dyn_k4/progress --seconds 2 --scale 4

Each snapshot plays for --seconds with a "training step N" label;
layout per row: real room | dream.
"""
import argparse
import glob
import re
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--seconds", type=float, default=2.0)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--scale", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()

files = sorted(glob.glob(f"{a.folder}/step_*.npy"))
out = a.out or f"{a.folder}/watch_it_learn.mp4"
n = int(a.seconds * a.fps)
frames = []
for f in files:
    step = int(re.findall(r"step_(\d+)", f)[0])
    arr = np.load(f)                                    # (scenes, T, 64, 130, 3)
    rows = np.concatenate(list(arr), 1)                 # (T, 64*scenes, 130, 3)
    for t in range(min(n, len(rows))):
        img = Image.fromarray(rows[t]).resize((rows.shape[2] * a.scale, rows.shape[1] * a.scale),
                                              Image.NEAREST)
        canvas = Image.new("RGB", (img.width, img.height + 40), (12, 13, 17))
        canvas.paste(img, (0, 40))
        d = ImageDraw.Draw(canvas)
        d.text((10, 12), f"training step {step:,}", fill=(242, 184, 75))
        d.text((img.width // 4 - 20, 12), "real", fill=(200, 200, 200))
        d.text((3 * img.width // 4 - 20, 12), "dream", fill=(200, 200, 200))
        frames.append(np.asarray(canvas))

h, w = frames[0].shape[:2]
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-", "-pix_fmt", "yuv420p",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], stdin=subprocess.PIPE)
for f in frames:
    ff.stdin.write(f.tobytes())
ff.stdin.close()
ff.wait()
print(f"wrote {out}  ({len(files)} snapshots, {len(frames) / a.fps:.0f}s)")

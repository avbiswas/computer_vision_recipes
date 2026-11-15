"""High-res VQ-VAE snapshots for video production (fixed held-out frames every time).

For one checkpoint, writes runs/vqvae/snapshots/step_XXXXXX/:
  grid.png            8 frames: real (top) vs reconstruction (bottom), 512 px per frame
  frames/real_i.png, frames/recon_i.png   each frame alone at 512 px (lossless, nearest)
  episode_j.mp4       held-out episode: real | reconstruction + arrow-key indicator

  python snapshot_vq.py --step 4000          # snapshot the current checkpoint once
  python snapshot_vq.py --watch              # snapshot every new checkpoint until training ends
"""
import argparse
import os
import re
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from vqvae import VQVAE, DEVICE
from videoutil import draw_keys, upscale, write_mp4, font, BG

SCALE = 8  # 64 px -> 512 px


def fixed_samples(frames, actions):
    n_train = int(len(frames) * 0.95)
    rng = np.random.default_rng(123)
    eps = rng.choice(np.arange(n_train, len(frames)), 8, replace=False)
    stills = np.stack([frames[e, rng.integers(frames.shape[1])] for e in eps])
    videos = [(np.asarray(frames[e]), np.asarray(actions[e])) for e in eps[:2]]
    return stills, videos


@torch.no_grad()
def recon(model, x):
    out = []
    for i in range(0, len(x), 64):
        t = VQVAE.to_input(torch.from_numpy(np.ascontiguousarray(x[i:i + 64])).to(DEVICE))
        out.append(VQVAE.to_image(model.decode(model.encode(t))).cpu().numpy())
    return np.concatenate(out)


def snapshot(ckpt, step, stills, videos, root):
    model = VQVAE().to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    out = f"{root}/step_{step:06d}"
    os.makedirs(f"{out}/frames", exist_ok=True)
    rs = recon(model, stills)
    S, pad, top = 64 * SCALE, 16, 70
    grid = Image.new("RGB", (8 * S + 9 * pad, 2 * S + 3 * pad + top), BG)
    d = ImageDraw.Draw(grid)
    d.text((pad, 20), f"VQ-VAE v2  ·  training step {step:,}  ·  top: real   bottom: reconstruction (256 codes)",
           fill=(242, 184, 75), font=font(30))
    for i in range(8):
        a, b = upscale(stills[i], SCALE), upscale(rs[i], SCALE)
        a.save(f"{out}/frames/real_{i}.png"), b.save(f"{out}/frames/recon_{i}.png")
        grid.paste(a, (pad + i * (S + pad), top + pad))
        grid.paste(b, (pad + i * (S + pad), top + 2 * pad + S))
    grid.save(f"{out}/grid.png")

    for j, (fr, act) in enumerate(videos):
        rv = recon(model, fr)
        video = []
        for t in range(len(fr)):
            a = 0 if t == 0 else int(act[t - 1])  # key that produced this frame
            canvas = Image.new("RGB", (2 * S + 3 * pad, S + 2 * pad + 170), BG)
            dd = ImageDraw.Draw(canvas)
            canvas.paste(upscale(fr[t], SCALE), (pad, pad + 50))
            canvas.paste(upscale(rv[t], SCALE), (2 * pad + S, pad + 50))
            dd.text((pad + S // 2, pad + 22), "real", fill=(200, 200, 200), font=font(28), anchor="mm")
            dd.text((2 * pad + S + S // 2, pad + 22), f"reconstruction · step {step:,}",
                    fill=(200, 200, 200), font=font(28), anchor="mm")
            keys = draw_keys(a, size=44)
            canvas.paste(keys, ((canvas.width - keys.width) // 2, S + pad + 60))
            video.append(np.asarray(canvas))
        write_mp4(video, f"{out}/episode_{j}.mp4", fps=10)
    print(f"snapshot step {step} -> {out}", flush=True)


def last_logged_step(log):
    steps = re.findall(r"step\s+(\d+)", open(log).read()) if os.path.exists(log) else []
    return int(steps[-1]) if steps else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vqvae/vqvae.pt")
    ap.add_argument("--log", default="logs/vqvae.log")
    ap.add_argument("--out", default="runs/vqvae/snapshots")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--done", default="runs/vqvae/DONE")
    a = ap.parse_args()
    frames = np.load("cache/frames.npy", mmap_mode="r")
    actions = np.load("cache/actions.npy")
    stills, videos = fixed_samples(frames, actions)
    if not a.watch:
        snapshot(a.ckpt, a.step if a.step is not None else last_logged_step(a.log), stills, videos, a.out)
    else:
        seen = os.path.getmtime(a.ckpt) if os.path.exists(a.ckpt) else 0
        while not os.path.exists(a.done):
            time.sleep(20)
            m = os.path.getmtime(a.ckpt) if os.path.exists(a.ckpt) else 0
            if m != seen:
                time.sleep(5)  # let the save finish
                seen = m
                snapshot(a.ckpt, round(last_logged_step(a.log) / 500) * 500, stills, videos, a.out)
        snapshot(a.ckpt, last_logged_step(a.log), stills, videos, a.out)  # final

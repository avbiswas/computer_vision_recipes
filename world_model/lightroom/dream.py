"""Run the real room and the dream side by side with the SAME lamp actions.

  python dream.py rollout --steps 200 --passes 1   # writes runs/dream/*.gif + PSNR per step
  python dream.py play                             # interactive: arrows move the lamp in both

The dream starts from K real frames (to fill its context window); after that it
only ever sees its own predictions.
"""
import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
from env import FireflyRoom, DIR_NAMES
from record import lamp_policy
from vqvae import VQVAE, DEVICE
from dynamics import WorldModel, predict_next


def load(vq_path, wm_path):
    vq = VQVAE().to(DEVICE).eval()
    vq.load_state_dict(torch.load(vq_path, map_location=DEVICE))
    ck = torch.load(wm_path, map_location=DEVICE)
    wm = WorldModel(K=ck["args"]["K"], d=ck["args"]["d"], layers=ck["args"]["layers"]).to(DEVICE).eval()
    wm.load_state_dict(ck["model"])
    return vq, wm


class Dream:
    """Keeps the sliding window of the last K frames (tokens) and actions."""

    def __init__(self, vq, wm, real_frames, real_actions, passes=1, temperature=1.0):
        self.vq, self.wm, self.passes, self.temperature = vq, wm, passes, temperature
        x = VQVAE.to_input(torch.from_numpy(np.stack(real_frames)).to(DEVICE))
        self.ctx = vq.encode(x).flatten(1)[None]                     # (1, K, 256)
        # actions[j] moved frame j -> j+1; the last slot is filled with the next action at step()
        self.act = torch.tensor([list(real_actions) + [0]], device=DEVICE)  # (1, K)

    def step(self, action):
        self.act[0, -1] = action
        nxt = predict_next(self.wm, self.ctx, self.act, self.passes, self.temperature)  # (1, 256)
        self.ctx = torch.cat([self.ctx[:, 1:], nxt[:, None]], 1)    # slide the window
        self.act = torch.cat([self.act[:, 1:], torch.zeros_like(self.act[:, :1])], 1)
        return VQVAE.to_image(self.vq.decode(nxt.view(1, 16, 16)))[0].cpu().numpy()


def start(env, K, rng):
    """Reset the real room and play K-1 real steps to fill the dream's context."""
    frames, acts = [env.reset()], []
    pol = lamp_policy(env, rng)
    for _ in range(K - 1):
        acts.append(next(pol))
        frames.append(env.step(acts[-1]))
    return frames, acts, pol


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2) / 255 ** 2
    return 10 * np.log10(1 / max(mse, 1e-10))


def rollout(a, vq, wm):
    os.makedirs(a.out, exist_ok=True)
    K = wm.K
    for ep in range(a.episodes):
        env = FireflyRoom(res=64, seed=10_000 + ep)  # seeds never used in recording
        rng = np.random.default_rng(10_000 + ep)
        frames, acts, pol = start(env, K, rng)
        dream = Dream(vq, wm, frames, acts, a.passes, a.temperature)
        strip, scores, t0 = [], [], time.time()
        for t in range(a.steps):
            act = next(pol)
            real, fake = env.step(act), dream.step(act)
            scores.append(psnr(real, fake))
            gap = np.zeros((64, 2, 3), np.uint8)
            strip.append(np.concatenate([real, gap, fake], 1))
        fps = a.steps / (time.time() - t0)
        imgs = [Image.fromarray(f).resize((130 * 4, 64 * 4), Image.NEAREST) for f in strip]
        name = f"{a.out}/ep{ep}_p{a.passes}_K{K}.gif"
        imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        np.save(name.replace(".gif", "_psnr.npy"), np.array(scores))
        s = np.array(scores)
        print(f"{name}: PSNR step1 {s[0]:.1f}  steps1-10 {s[:10].mean():.1f}  "
              f"steps 100+ {s[100:].mean() if len(s) > 100 else float('nan'):.1f}  "
              f"dream fps {fps:.1f} (incl. real env)", flush=True)


def play(a, vq, wm):
    import matplotlib.pyplot as plt
    env, rng = FireflyRoom(res=64), np.random.default_rng()
    frames, acts, _ = start(env, wm.K, rng)
    dream = Dream(vq, wm, frames, acts, a.passes, a.temperature)
    held = {"d": 0}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1, im2 = ax1.imshow(frames[-1]), ax2.imshow(frames[-1])
    ax1.set_title("real room"), ax2.set_title("dream (world model)")
    ax1.axis("off"), ax2.axis("off")

    def press(e):
        if e.key in DIR_NAMES:
            held["d"] = DIR_NAMES.index(e.key)

    def release(e):
        if e.key in DIR_NAMES:
            held["d"] = 0

    def tick():
        im1.set_data(env.step(held["d"]))
        im2.set_data(dream.step(held["d"]))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", press)
    fig.canvas.mpl_connect("key_release_event", release)
    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rollout", "play"])
    ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
    ap.add_argument("--wm", default="runs/dyn_k4/world_model.pt")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="runs/dream")
    a = ap.parse_args()
    vq, wm = load(a.vqvae, a.wm)
    rollout(a, vq, wm) if a.mode == "rollout" else play(a, vq, wm)

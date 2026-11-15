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
    if ck["args"].get("rope"):
        from dynamics_rope import WorldModelRoPE
        offs = tuple(int(x) for x in ck["args"].get("offsets", "32,16,8,4,3,2,1,0").split(","))
        wm = WorldModelRoPE(d=ck["args"]["d"], layers=ck["args"]["layers"], offsets=offs,
                            use_rope=ck["args"].get("use_rope", True)).to(DEVICE).eval()
        wm.load_state_dict(ck["model"])
        return vq, wm
    wm = WorldModel(K=ck["args"]["K"], d=ck["args"]["d"], layers=ck["args"]["layers"], causal=ck["args"].get("causal", False),
                    anchor=ck["args"].get("anchor", False)).to(DEVICE).eval()
    wm.load_state_dict(ck["model"])
    return vq, wm


class Dream:
    """Keeps the sliding window of the last K frames (tokens) and actions."""

    def __new__(cls, vq, wm, *args, **kwargs):  # RoPE models keep a longer, strided history
        if getattr(wm, "rope", False):
            return DreamRoPE(vq, wm, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, vq, wm, real_frames, real_actions, passes=1, temperature=1.0):
        self.vq, self.wm, self.passes, self.temperature = vq, wm, passes, temperature
        x = VQVAE.to_input(torch.from_numpy(np.stack(real_frames)).to(DEVICE))
        self.ctx = vq.encode(x).flatten(1)[None]                     # (1, K, 256)
        self.anchor = self.ctx[:, 0].clone()                         # first real frame, kept forever
        # actions[j] moved frame j -> j+1; the last slot is filled with the next action at step()
        self.act = torch.tensor([list(real_actions) + [0]], device=DEVICE)  # (1, K)

    def step(self, action):
        self.act[0, -1] = action
        nxt = predict_next(self.wm, self.ctx, self.act, self.passes, self.temperature,
                           anchor=self.anchor if self.wm.use_anchor else None)  # (1, 256)
        self.ctx = torch.cat([self.ctx[:, 1:], nxt[:, None]], 1)    # slide the window
        self.act = torch.cat([self.act[:, 1:], torch.zeros_like(self.act[:, :1])], 1)
        return VQVAE.to_image(self.vq.decode(nxt.view(1, 16, 16)))[0].cpu().numpy()


class DreamRoPE:
    """Dream for the strided-context RoPE model: keeps the full token history (trimmed to what
    the strided window needs) and the first frame as the anchor."""

    def __init__(self, vq, wm, real_frames, real_actions, passes=1, temperature=0.0):
        self.vq, self.wm, self.passes, self.temperature, self.offsets = vq, wm, passes, temperature, wm.offsets
        x = VQVAE.to_input(torch.from_numpy(np.stack(real_frames)).to(DEVICE))
        self.hist = list(vq.encode(x).flatten(1))                    # token frames, oldest first
        self.acts = list(real_actions)                               # acts[i] moved frame i -> i+1
        self.anchor = self.hist[0]
        self.t0 = 0                                                  # absolute index of hist[0]... kept simple

    def step(self, action):
        from dynamics_rope import predict_next_rope
        self.acts.append(int(action))
        t = len(self.hist)                                           # index of the frame to predict
        idx = [max(t - o, 0) for o in self.offsets]
        fa = []
        for j, i in enumerate(idx):
            ok = j > 0 and i >= 1 and idx[j - 1] == i - 1
            fa.append(self.acts[i - 1] if ok else 5)                 # 5 = NO_ACTION
        ctx = torch.stack([self.hist[i] for i in idx[:-1]])[None]
        times = torch.tensor([[i - t for i in idx]], device=DEVICE)
        anchor_time = torch.tensor([-min(t, 250)], device=DEVICE)
        nxt = predict_next_rope(self.wm, ctx, torch.tensor([fa], device=DEVICE), times,
                                self.anchor[None], anchor_time, self.passes, self.temperature)[0]
        self.hist.append(nxt)
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

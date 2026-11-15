""""Watch it learn" videos: during training, dream the SAME fixed scenes with the SAME
lamp moves at every snapshot, so snapshots are directly comparable.

  runs/<run>/progress/step_000500.npy   uint8 (scenes, T, 64, 130, 3)  real | dream
  runs/<run>/progress/step_000500.mp4   high-res: real | dream | pressed key, per scene

make_progress_video.py stitches them into one labelled video.
"""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from env import FireflyRoom
from videoutil import draw_keys, upscale, write_mp4, font, BG
from record import lamp_policy
from dream import Dream


class ProgressDreams:
    def __init__(self, vq, out, K, scenes=3, steps=120, seed=20_000):
        self.vq, self.out, self.K = vq, out, K
        os.makedirs(out, exist_ok=True)
        # record the fixed test scenes once (real frames + the actions used)
        self.scenes = []
        for s in range(scenes):
            env, rng = FireflyRoom(res=64, seed=seed + s), np.random.default_rng(seed + s)
            pol = lamp_policy(env, rng)
            frames, acts = [env.reset()], []
            for _ in range(K - 1 + steps):
                acts.append(next(pol))
                frames.append(env.step(acts[-1]))
            self.scenes.append((frames, acts))

    @torch.no_grad()
    def snapshot(self, model, step):
        was_training = model.training
        model.eval()
        K, all_strips = self.K, []
        for frames, acts in self.scenes:
            # 1-pass, argmax: deterministic, so changes between snapshots come from learning only
            dream = Dream(self.vq, model, frames[:K], acts[:K - 1], passes=1, temperature=0)
            strip = []
            for t, act in enumerate(acts[K - 1:]):
                real, fake = frames[K + t], dream.step(act)
                strip.append(np.concatenate([real, np.zeros((64, 2, 3), np.uint8), fake], 1))
            all_strips.append(np.stack(strip))
        arr = np.stack(all_strips)                                       # (scenes, T, 64, 130, 3)
        np.save(f"{self.out}/step_{step:06d}.npy", arr)
        np.save(f"{self.out}/actions.npy", np.array([acts[K - 1:] for _, acts in self.scenes]))
        self.render_mp4(arr, step)
        model.train(was_training)
        real, fake = arr[:, :, :, :64].astype(float), arr[:, :, :, 66:].astype(float)
        e = np.abs(real - fake).mean((0, 2, 3, 4))                       # per rollout step
        motion = lambda v: np.abs(np.diff(v, axis=1)).mean()
        return e[:10].mean(), e[10:30].mean(), e[30:].mean(), motion(fake), motion(real)

    def render_mp4(self, arr, step, scale=6):
        """High-res video: each scene is a row 'real | dream | pressed key'."""
        S, pad = 64 * scale, 12
        keys_w = draw_keys(0, size=36).width
        W = 2 * S + keys_w + 4 * pad
        H = 84 + len(arr) * (S + pad) + pad
        video = []
        for t in range(arr.shape[1]):
            canvas = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(canvas)
            d.text((pad, 18), f"world model · training step {step:,}", fill=(242, 184, 75), font=font(26))
            d.text((pad + S // 2, 68), "real", fill=(200, 200, 200), font=font(20), anchor="mm")
            d.text((2 * pad + S + S // 2, 68), "dream", fill=(200, 200, 200), font=font(20), anchor="mm")
            for r, (scene, (_, acts)) in enumerate(zip(arr, self.scenes)):
                y = 84 + r * (S + pad)
                canvas.paste(upscale(scene[t, :, :64], scale), (pad, y))
                canvas.paste(upscale(scene[t, :, 66:], scale), (2 * pad + S, y))
                keys = draw_keys(int(acts[self.K - 1 + t]), size=36)
                canvas.paste(keys, (3 * pad + 2 * S, y + (S - keys.height) // 2))
            video.append(np.asarray(canvas))
        write_mp4(video, f"{self.out}/step_{step:06d}.mp4", fps=10)

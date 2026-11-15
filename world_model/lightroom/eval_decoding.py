"""Compare dream decoding settings on the same fixed scenes (no training involved).

  python eval_decoding.py --wm runs/dyn_k4/ckpts/wm_006500.pt
Prints rollout pixel error vs the real room (lower = better) per setting.
"""
import argparse
import numpy as np
import torch
from dream import load, Dream
from env import FireflyRoom
from record import lamp_policy

ap = argparse.ArgumentParser()
ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
ap.add_argument("--wm", required=True)
ap.add_argument("--scenes", type=int, default=6)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--fast", action="store_true", help="1-pass only")
a = ap.parse_args()
vq, wm = load(a.vqvae, a.wm)
K = wm.K
scenes = []
for s in range(a.scenes):
    env, rng = FireflyRoom(res=64, seed=30_000 + s), np.random.default_rng(30_000 + s)
    pol = lamp_policy(env, rng)
    frames, acts = [env.reset()], []
    for _ in range(K - 1 + a.steps):
        acts.append(next(pol))
        frames.append(env.step(acts[-1]))
    scenes.append((np.stack(frames).astype(float), acts))

def run(passes, temp):
    torch.manual_seed(0)
    errs = []
    for frames, acts in scenes:
        d = Dream(vq, wm, list(frames[:K].astype(np.uint8)), acts[:K - 1], passes, temp)
        fake = np.stack([d.step(x) for x in acts[K - 1:]]).astype(float)
        errs.append(np.abs(fake - frames[K:]).mean((1, 2, 3)))
    e = np.mean(errs, 0)
    return e[:10].mean(), e[10:30].mean(), e[30:].mean()

frozen = np.mean([np.abs(f[K:] - f[K - 1]).mean((1, 2, 3)) for f, _ in scenes], 0)
print(f"{'frozen reference':22s} 1-10 {frozen[:10].mean():5.1f}  11-30 {frozen[10:30].mean():5.1f}  31+ {frozen[30:].mean():5.1f}")
for passes, temp in ([(1, 0)] if a.fast else [(1, 0), (4, 0), (8, 0), (8, 0.5), (8, 1.0)]):
    e = run(passes, temp)
    print(f"passes {passes} temp {temp:<4}        1-10 {e[0]:5.1f}  11-30 {e[1]:5.1f}  31+ {e[2]:5.1f}", flush=True)

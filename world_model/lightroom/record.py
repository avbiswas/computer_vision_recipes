"""Record episodes from FireflyRoom.

The world runs itself (fireflies wander + bounce); a scripted "user" drives the
flashlight with tank controls (see lamp_policy).

Each episode -> data/ep_XXXXX.npz:
  frames    uint8 (T+1, H, W, 3)   frames[0] = reset frame
  actions   int8  (T,)             actions[t]: frames[t] -> frames[t+1]
  lamp      float32 (T+1, 2)       ground-truth state, for analysis only
  fly_pos   float32 (T+1, F, 2)    (e.g. building the wall-bounce test set)
  fly_head  float32 (T+1, F)
  fly_names, n_boxes

usage: python record.py --episodes 4000 --steps 64 --res 64 --out data
"""
import argparse
import os
import numpy as np
from env import FireflyRoom


def lamp_policy(env: FireflyRoom, rng):
    """Scripted 'user' for the flashlight. Chunks of one held action (2-10 steps):
    30% stay (world evolves on its own), 35% forward, 10% backward, 25% turning.
    A chunk ends early if the flashlight bumps into a wall or box."""
    while True:
        a = int(rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.35, 0.10, 0.125, 0.125]))
        for _ in range(int(rng.integers(2, 11))):
            yield a
            if env.last_blocked:
                break


def record_episode(env, rng, steps):
    frames, acts, states = [env.reset()], [], [env.state()]
    pol = lamp_policy(env, rng)
    for _ in range(steps):
        a = next(pol)
        acts.append(a)
        frames.append(env.step(a))
        states.append(env.state())
    return dict(frames=np.stack(frames), actions=np.array(acts, np.int8),
                lamp=np.stack([s["lamp"] for s in states]).astype(np.float32),
                heading=np.array([s["heading"] for s in states], np.float32),
                fly_pos=np.stack([s["fly_pos"] for s in states]).astype(np.float32),
                fly_head=np.stack([s["fly_head"] for s in states]).astype(np.float32),
                fly_names=np.array(env.fly_names), n_boxes=env.n_boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first episode index (for parallel workers)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed = a.seed * 100_003 + a.start  # different stream per worker
    rng = np.random.default_rng(seed)
    env = FireflyRoom(res=a.res, seed=seed + 1)
    for ep in range(a.start, a.start + a.episodes):
        np.savez_compressed(f"{a.out}/ep_{ep:05d}.npz", **record_episode(env, rng, a.steps))
        if ep % 100 == 0:
            print(f"episode {ep}", flush=True)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()

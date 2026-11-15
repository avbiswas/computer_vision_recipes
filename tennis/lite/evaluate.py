"""
Step 7 - How good is the ball tracking, and WHERE does it fail?

Runs a model on the test match (game10, never seen in training) and splits the ball
results by difficulty. A prediction counts as "found" if it is within 10 px (in 1280x720)
of the labelled ball.

Usage:  python lite/evaluate.py                                   (v2 model)
        python lite/evaluate.py --weights runs/lite_v1/last.pt     (compare another model)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import TennisFrames
from decode import decode
from model import TennisNetLite

ROOT = Path(__file__).resolve().parents[1]
FOUND_PX = 10


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def report(name, frames, preds, rows):
    """found = right place, wrong = a ball somewhere else, missed = no ball predicted."""
    if not frames:
        return
    found = [i for i in frames if preds[i] and distance(preds[i], rows[i]["ball"]["xy"]) <= FOUND_PX]
    wrong = [i for i in frames if preds[i] and i not in set(found)]
    n = len(frames)
    error = np.median([distance(preds[i], rows[i]["ball"]["xy"]) for i in found]) if found else float("nan")
    print(f"{name:30s} {n:5d} frames   found {len(found) / n:6.1%}   wrong place {len(wrong) / n:5.1%}"
          f"   missed {(n - len(found) - len(wrong)) / n:6.1%}   median error {error:.1f}px")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/lite_v2/best.pt"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load(args.weights, map_location="cpu")
    color = weights["stem.0.weight"].shape[1] == 9
    model = TennisNetLite(pretrained=False, color=color)
    model.load_state_dict(weights)
    model = model.to(device).eval()

    dataset = TennisFrames(args.split, color=color)
    rows = [dataset.rows[i] for i in dataset.indices]

    # 1. predict the ball on every frame (in batches of 32)
    preds = []
    for start in range(0, len(dataset), 32):
        x = torch.stack([dataset[n][0] for n in range(start, min(start + 32, len(dataset)))]).to(device)
        heatmaps, sizes = model(x)
        for h, s in zip(heatmaps.cpu().numpy(), sizes.cpu().numpy()):
            preds.append(decode(h, s)[0])

    # 2. how fast is the true ball moving? (pixels per frame, vs. the previous frame of the same clip)
    speed = [None] * len(rows)
    for i in range(1, len(rows)):
        before, now = rows[i - 1], rows[i]
        if before["clip"] == now["clip"] and before["ball"]["xy"] and now["ball"]["xy"]:
            speed[i] = distance(before["ball"]["xy"], now["ball"]["xy"])

    # 3. results per case
    has_ball = [i for i, r in enumerate(rows) if r["ball"]["xy"]]
    ball = lambda i: rows[i]["ball"]
    print(f"model {args.weights}  |  {args.split} split\n")
    report("all frames with a ball", has_ball, preds, rows)
    report("clearly visible", [i for i in has_ball if ball(i)["vis"] == 1], preds, rows)
    report("hard to see (faint/blurred)", [i for i in has_ball if ball(i)["vis"] == 2], preds, rows)
    report("racket-hit frame", [i for i in has_ball if ball(i)["status"] == 1], preds, rows)
    report("bounce frame", [i for i in has_ball if ball(i)["status"] == 2], preds, rows)
    report("far half (ball is small)", [i for i in has_ball if ball(i)["xy"][1] < 300], preds, rows)
    report("near half", [i for i in has_ball if ball(i)["xy"][1] >= 300], preds, rows)
    report("slow ball (<10 px/frame)", [i for i in has_ball if speed[i] is not None and speed[i] < 10], preds, rows)
    report("fast ball (>26 px/frame)", [i for i in has_ball if speed[i] is not None and speed[i] > 26], preds, rows)

    no_ball = [i for i, r in enumerate(rows) if not r["ball"]["xy"]]
    if no_ball:
        print(f"\nframes without a ball: {len(no_ball)}, false alarms {sum(preds[i] is not None for i in no_ball) / len(no_ball):.1%}")
    true_pos = sum(1 for i in has_ball if preds[i] and distance(preds[i], ball(i)["xy"]) <= FOUND_PX)
    precision = true_pos / max(1, sum(p is not None for p in preds))
    recall = true_pos / max(1, len(has_ball))
    print(f"overall: precision {precision:.3f}  recall {recall:.3f}  F1 {2 * precision * recall / (precision + recall):.3f}")


if __name__ == "__main__":
    main()

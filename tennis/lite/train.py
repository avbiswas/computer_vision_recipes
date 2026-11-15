"""
Step 5 - Train.

One network, three tasks, one loss:
    loss = ball_heatmap_loss + court_heatmap_loss + player_heatmap_loss + 0.1 * player_size_loss

Heatmaps are 99.9% background, so we use the "focal" loss from CenterNet:
it down-weights the easy background pixels and focuses on the few pixels near the objects.

Usage:  python lite/train.py --epochs 1 --frac 0.25      (quick check)
        python lite/train.py                              (real run: 20 full epochs, hard frames boosted)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data import BALL, COURT, PLAYER, TennisFrames, sample_weights
from decode import evaluate
from model import TennisNetLite

ROOT = Path(__file__).resolve().parents[1]


def focal_loss(pred, target):
    """CenterNet focal loss. pred/target: heatmaps in 0..1, target peaks are exactly 1."""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    is_peak = target.eq(1).float()
    peak_loss = -torch.log(pred) * (1 - pred) ** 2 * is_peak
    background_loss = -torch.log(1 - pred) * pred ** 2 * (1 - target) ** 4 * (1 - is_peak)
    return (peak_loss.sum() + background_loss.sum()) / is_peak.sum().clamp(min=1)


def compute_loss(heatmaps, sizes, true_heatmaps, true_sizes, court_ok):
    ball = focal_loss(heatmaps[:, BALL], true_heatmaps[:, BALL])
    players = focal_loss(heatmaps[:, PLAYER], true_heatmaps[:, PLAYER])
    ok = court_ok.bool()                                    # only learn the court where its label is reliable
    court = focal_loss(heatmaps[ok][:, COURT], true_heatmaps[ok][:, COURT]) if ok.any() else heatmaps.sum() * 0
    has_size = (true_sizes[:, :1] > 0).float()              # size is only defined at player centres
    size = (torch.abs(sizes - true_sizes) * has_size).sum() / has_size.sum().clamp(min=1)
    total = ball + court + players + 0.1 * size
    return total, {"ball": ball.item(), "court": court.item(), "players": players.item(), "size": size.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--frac", type=float, default=1.0, help="random fraction of train frames per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--boost", type=float, default=3.0, help="how much more often hard frames are shown (1 = no boost)")
    ap.add_argument("--color", action="store_true", help="3 colour frames (9 channels) instead of 3 grayscale")
    ap.add_argument("--name", default="lite_v2")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = TennisFrames("train", augment=True, color=args.color)
    val_set = TennisFrames("val", color=args.color)
    weights, n_hard = sample_weights(train_set, args.boost)
    print(f"{n_hard} hard frames of {len(train_set)}, shown {args.boost:g}x more often "
          f"-> {float(weights[weights > 1].sum() / weights.sum()):.0%} of what the model sees")
    model = TennisNetLite(color=args.color).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = int(len(train_set) * args.frac) // args.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * steps_per_epoch)

    for epoch in range(args.epochs):
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=steps_per_epoch * args.batch)
        loader = torch.utils.data.DataLoader(train_set, args.batch, sampler=sampler, num_workers=6, drop_last=True,
                                             persistent_workers=False)
        start = time.time()
        for step, (x, true_heatmaps, true_sizes, court_ok) in enumerate(loader):
            heatmaps, sizes = model(x.to(device))
            loss, parts = compute_loss(heatmaps, sizes, true_heatmaps.to(device), true_sizes.to(device), court_ok.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                speed = (step + 1) * args.batch / (time.time() - start)
                print(f"epoch {epoch} step {step}/{steps_per_epoch}  loss {loss.item():.3f}  "
                      + "  ".join(f"{k} {v:.3f}" for k, v in parts.items()) + f"  ({speed:.0f} img/s)", flush=True)

        metrics = evaluate(model, val_set, device, max_frames=400)
        print(f"== epoch {epoch} val: {metrics}  ({time.time() - start:.0f}s)", flush=True)
        torch.save(model.state_dict(), out_dir / "last.pt")
        with open(out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")


if __name__ == "__main__":
    main()

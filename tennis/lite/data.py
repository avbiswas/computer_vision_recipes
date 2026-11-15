"""
Step 2 - The dataset: turns one labelled frame into (input, targets).

INPUT   3 grayscale frames [t-2, t-1, t] stacked like the R,G,B channels of one image.
        Motion of the ball shows up as a difference between the channels.
        color=True: the same 3 frames in colour -> 9 channels (R,G,B of t-2, then t-1, then t).

TARGETS everything is a heatmap at half the input resolution (256 x 144):
        channel 0      ball            a small Gaussian blob at the ball (empty if not visible)
        channels 1-14  court keypoints one blob per court point
        channel 15     player centres  one blob per player
        + "size" map   player (width, height), only read at the player centres
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
IN_W, IN_H = 512, 288          # network input
STRIDE = 2                     # heatmaps are 2x smaller than the input
OUT_W, OUT_H = IN_W // STRIDE, IN_H // STRIDE
SCALE = IN_W / 1280            # labels are in 1280x720 pixels -> input pixels

BALL, COURT, PLAYER = 0, slice(1, 15), 15
NUM_HEATMAPS = 16
SIZE_NORM = 32.0               # player sizes are predicted in units of 32 heatmap pixels

META = json.load(open(ROOT / "data/meta.json"))
COURT_FLIP = META["court_kp_flip"]   # left<->right index swap for a horizontal flip


# ---------------------------------------------------------------- heatmap drawing
def draw_gaussian(heatmap, x, y, sigma):
    """Add a Gaussian blob with peak exactly 1.0 at the (rounded) pixel x, y."""
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < heatmap.shape[1] and 0 <= cy < heatmap.shape[0]):
        return False
    r = int(3 * sigma)
    ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
    blob = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    x0, x1 = max(0, cx - r), min(heatmap.shape[1], cx + r + 1)
    y0, y1 = max(0, cy - r), min(heatmap.shape[0], cy + r + 1)
    patch = blob[y0 - cy + r:y1 - cy + r, x0 - cx + r:x1 - cx + r]
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)
    return True


# ---------------------------------------------------------------- augmentation
def random_transform():
    """One random geometric transform (2x3 matrix) + flip flag, shared by all 3 frames and all labels."""
    flip = random.random() < 0.5
    angle = random.uniform(-3, 3)                      # degrees
    scale = random.uniform(0.85, 1.15)
    shift = (random.uniform(-0.08, 0.08) * IN_W, random.uniform(-0.08, 0.08) * IN_H)
    M = cv2.getRotationMatrix2D((IN_W / 2, IN_H / 2), angle, scale)
    M[:, 2] += shift
    if flip:                                           # mirror: x -> W - x
        M = np.array([[-1, 0, IN_W], [0, 1, 0]]) @ np.vstack([M, [0, 0, 1]])
    return M, flip


def apply_to_points(M, pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def random_photometric(frames):
    """Brightness / contrast / blur / noise - identical for the 3 frames."""
    frames = frames.astype(np.float32)
    frames = frames * random.uniform(0.7, 1.3) + random.uniform(-25, 25)
    if frames.ndim == 4:                               # colour: slightly tint R,G,B (other courts, other cameras)
        frames = frames * np.random.uniform(0.9, 1.1, 3)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        frames = np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in frames])
    if random.random() < 0.3:
        frames = frames + np.random.normal(0, 4, frames.shape)
    return np.clip(frames, 0, 255)


# ---------------------------------------------------------------- dataset
def sample_weights(dataset, boost=3.0):
    """How often each frame is picked during training.
    "Hard" frames - a racket hit (and 2 frames either side) or a faint/blurred ball -
    are rare (~19% of train frames), so we show them `boost` times more often."""
    rows = dataset.rows
    near_hit = set()
    for i, r in enumerate(rows):
        if r["ball"]["status"] == 1:
            near_hit.update(j for j in range(i - 2, i + 3) if 0 <= j < len(rows) and rows[j]["clip"] == r["clip"])
    hard = [i in near_hit or rows[i]["ball"]["vis"] == 2 for i in dataset.indices]
    return torch.tensor([boost if h else 1.0 for h in hard]), sum(hard)


class TennisFrames(torch.utils.data.Dataset):
    def __init__(self, split, augment=False, color=False):
        self.rows = [json.loads(line) for line in open(ROOT / "data/tennis.jsonl")]
        self.frames = np.load(ROOT / ("data/frames_512x288_rgb.npy" if color else "data/frames_512x288.npy"), mmap_mode="r")
        self.indices = [i for i, r in enumerate(self.rows) if r["split"] == split]
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def previous(self, i, steps):
        """Index of the frame `steps` earlier in the same clip (or the first frame of the clip)."""
        for _ in range(steps):
            if i > 0 and self.rows[i - 1]["clip"] == self.rows[i]["clip"]:
                i -= 1
        return i

    def __getitem__(self, n):
        i = self.indices[n]
        row = self.rows[i]
        gap = random.choice([1, 2]) if self.augment else 1          # temporal augmentation
        frames = self.frames[[self.previous(i, 2 * gap), self.previous(i, gap), i]].astype(np.float32)

        # labels in input pixels
        ball = [p * SCALE for p in row["ball"]["xy"]] if row["ball"]["xy"] else None
        boxes = [[v * SCALE for v in p["box"]] for p in row["players"]]
        court_ok = row["court"]["ok"]
        court = np.array(row["court"]["kps"]) * SCALE if court_ok else None

        if self.augment:
            M, flip = random_transform()
            frames = np.stack([cv2.warpAffine(f, M, (IN_W, IN_H), borderMode=cv2.BORDER_REFLECT) for f in frames])
            frames = random_photometric(frames)
            if ball:
                ball = apply_to_points(M, ball)[0]
            boxes = [self.transform_box(M, b) for b in boxes]
            if court is not None:
                court = apply_to_points(M, court)
                if flip:
                    court = court[COURT_FLIP]                # left points become right points

        heatmaps = np.zeros((NUM_HEATMAPS, OUT_H, OUT_W), np.float32)
        sizes = np.zeros((2, OUT_H, OUT_W), np.float32)
        if ball is not None:
            draw_gaussian(heatmaps[BALL], ball[0] / STRIDE, ball[1] / STRIDE, sigma=1.5)
        if court is not None:
            for k, (x, y) in enumerate(court):
                draw_gaussian(heatmaps[1 + k], x / STRIDE, y / STRIDE, sigma=2.0)
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
            if draw_gaussian(heatmaps[PLAYER], cx, cy, sigma=2.5):
                sizes[:, int(round(cy)), int(round(cx))] = [(x2 - x1) / STRIDE / SIZE_NORM,
                                                            (y2 - y1) / STRIDE / SIZE_NORM]

        if frames.ndim == 4:                                         # colour (3, H, W, RGB) -> (9, H, W)
            frames = frames.transpose(0, 3, 1, 2).reshape(9, IN_H, IN_W)
        x = torch.from_numpy(frames / 255.0).float()                 # (3 or 9, 288, 512)
        return x, torch.from_numpy(heatmaps), torch.from_numpy(sizes), torch.tensor(float(court_ok))

    @staticmethod
    def transform_box(M, box):
        x1, y1, x2, y2 = box
        corners = apply_to_points(M, [[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        return [*corners.min(0), *corners.max(0)]


if __name__ == "__main__":
    # quick visual check: python lite/data.py  -> writes lite/check_sample.jpg
    ds = TennisFrames("train", augment=True)
    x, hm, sizes, court_ok = ds[1234]
    img = cv2.cvtColor((x[2].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = cv2.resize(hm.numpy().max(0), (IN_W, IN_H))           # all heatmaps in one picture
    img[..., 2] = np.maximum(img[..., 2], (overlay * 255).astype(np.uint8))
    cv2.imwrite(str(ROOT / "lite/check_sample.jpg"), img)
    print("input", tuple(x.shape), "heatmaps", tuple(hm.shape), "court_ok", bool(court_ok), "train samples", len(ds))

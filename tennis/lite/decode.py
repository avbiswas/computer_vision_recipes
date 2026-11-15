"""
Step 4 - Turn heatmaps back into answers, and score them.

decode()   heatmaps -> ball (x, y) | court 14 x (x, y) | 2 player boxes
           all returned in ORIGINAL 1280x720 pixels.
evaluate() runs the model on a split and prints simple, explainable metrics.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data import BALL, PLAYER, SCALE, SIZE_NORM, STRIDE

TO_FULL_RES = STRIDE / SCALE      # heatmap pixel -> 1280x720 pixel  (= 5.0)


def refine(heatmap, x, y):
    """Sub-pixel position: average the 3x3 neighbourhood of pixel (x, y), weighted by heat."""
    h, w = heatmap.shape
    y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
    patch = heatmap[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((xs * patch).sum() / patch.sum()), float((ys * patch).sum() / patch.sum())


def peak(heatmap):
    """Sub-pixel location and height of the highest point of one heatmap."""
    y, x = divmod(int(heatmap.argmax()), heatmap.shape[1])
    return (*refine(heatmap, x, y), float(heatmap[y, x]))


def decode(heatmaps, sizes, ball_threshold=0.5, player_threshold=0.3):
    """heatmaps (16, H, W) and sizes (2, H, W) for ONE image, as numpy arrays."""
    bx, by, score = peak(heatmaps[BALL])
    ball = (bx * TO_FULL_RES, by * TO_FULL_RES) if score > ball_threshold else None

    court = [peak(heatmaps[1 + k])[:2] for k in range(14)]
    court = [(x * TO_FULL_RES, y * TO_FULL_RES) for x, y in court]

    # players: local maxima of the centre heatmap (a 3x3 max-pool trick = simple non-max suppression)
    hm = torch.from_numpy(heatmaps[PLAYER])[None, None]
    is_peak = (hm == F.max_pool2d(hm, 3, stride=1, padding=1)) & (hm > player_threshold)
    ys, xs = np.nonzero(is_peak[0, 0].numpy())
    best = np.argsort(-heatmaps[PLAYER][ys, xs])[:2]            # the two strongest peaks
    players = []
    for y, x in zip(ys[best], xs[best]):
        w, h = sizes[:, y, x] * SIZE_NORM
        cx, cy = refine(heatmaps[PLAYER], x, y)                   # sub-pixel centre (avoids 5px jumps)
        players.append([(cx - w / 2) * TO_FULL_RES, (cy - h / 2) * TO_FULL_RES,
                        (cx + w / 2) * TO_FULL_RES, (cy + h / 2) * TO_FULL_RES])
    return ball, court, players


def vertical_speed(points):
    """Average vertical speed (pixels per frame, + = moving down the screen) over a list of
    (frame, x, y). Uses the first and last point we have, so missing frames don't matter."""
    if len(points) < 2:
        return None
    (t0, _, y0), (t1, _, y1) = points[0], points[-1]
    return (y1 - y0) / (t1 - t0)


def near_player(point, players, margin=40):
    """Is the point inside (or within `margin` pixels of) any player's box?"""
    return any(x1 - margin <= point[0] <= x2 + margin and y1 - margin <= point[1] <= y2 + margin
               for x1, y1, x2, y2 in players)


def find_hit(track, players, half=4, min_speed=2.0):
    """Racket hit = the ball's vertical direction flips (coming towards a player -> going away)
    while the ball is next to that player.

    track:   the last 2*half+1 ball positions (x, y), oldest first, None where the ball wasn't seen.
    players: the player boxes at the middle frame.
    Returns the ball position at the middle frame, or None.
    (Looking `half` frames ahead means hits are reported `half` frames late - fine for a demo.)"""
    track = list(track)
    if len(track) < 2 * half + 1 or track[half] is None:
        return None
    seen = [(t, *p) for t, p in enumerate(track) if p is not None]
    before = vertical_speed([s for s in seen if s[0] <= half])
    after = vertical_speed([s for s in seen if s[0] >= half])
    if before is None or after is None:
        return None
    flipped = before * after < 0                                        # falling <-> rising
    fast_enough = abs(before) > min_speed and abs(after) > min_speed    # ignore tiny noisy wiggles
    if flipped and fast_enough and near_player(track[half], players):
        return track[half]
    return None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


@torch.no_grad()
def evaluate(model, dataset, device, max_frames=None, ball_hit_px=10):
    """Simple metrics in 1280x720 pixels:
       ball    precision / recall: a hit is a predicted ball within `ball_hit_px` of the true ball
       court   median distance between predicted and true keypoints
       players fraction of true players matched by a predicted box with IoU > 0.5"""
    model.eval()
    tp = fp = fn = 0
    court_err, player_hits, player_total = [], 0, 0
    order = range(len(dataset)) if not max_frames else np.linspace(0, len(dataset) - 1, max_frames).astype(int)
    for n in order:
        x, *_ = dataset[n]
        row = dataset.rows[dataset.indices[n]]
        heatmaps, sizes = model(x[None].to(device))
        ball, court, players = decode(heatmaps[0].cpu().numpy(), sizes[0].cpu().numpy())

        true_ball = row["ball"]["xy"]
        if ball and true_ball and np.hypot(ball[0] - true_ball[0], ball[1] - true_ball[1]) <= ball_hit_px:
            tp += 1
        else:
            fp += ball is not None
            fn += true_ball is not None

        if row["court"]["ok"]:
            court_err += [np.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(court, row["court"]["kps"])]

        for true_box in (p["box"] for p in row["players"]):
            player_total += 1
            player_hits += any(iou(true_box, b) > 0.5 for b in players)

    model.train()
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"ball_precision": round(precision, 3), "ball_recall": round(recall, 3),
            "ball_f1": round(2 * precision * recall / max(1e-9, precision + recall), 3),
            "court_median_px": round(float(np.median(court_err)), 1) if court_err else None,
            "players_found": round(player_hits / max(1, player_total), 3)}

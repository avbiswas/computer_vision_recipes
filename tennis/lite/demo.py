"""
Step 6 - Demo video.

For every frame:  keep the last 3 frames -> run TennisNet-lite -> decode ball, court, players.

  --mode boxes    draw player boxes, the ball (with a short trail) and the court lines on the video
  --mode minimap  put a top-down court on the right; players/ball are mapped onto it with a homography
  --mode both     (default) both at once

Usage:  python lite/demo.py --clip game10/Clip1
        python lite/demo.py --clip game10            (every clip of a match, back to back)
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from data import IN_H, IN_W
from decode import decode, find_hit
from model import TennisNetLite

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "raw/TrackNet/Dataset"

# ------------------------------------------------------------------ the court, seen from above
# Coordinates of a real court in "reference units" (same ones the court detector uses).
# The 14 keypoints, in the SAME ORDER as the model's 14 court heatmaps:
COURT_KEYPOINTS = np.float32([
    (286, 561), (1379, 561), (286, 2935), (1379, 2935),      # outer corners (doubles)
    (423, 561), (423, 2935), (1242, 561), (1242, 2935),      # singles sidelines ends
    (423, 1110), (1242, 1110), (423, 2386), (1242, 2386),    # service-line corners
    (832, 1110), (832, 2386)])                               # centre service line ends
COURT_LINES = [((286, 561), (1379, 561)), ((286, 2935), (1379, 2935)), ((286, 1748), (1379, 1748)),  # baselines, net
               ((286, 561), (286, 2935)), ((1379, 561), (1379, 2935)),                               # doubles sidelines
               ((423, 561), (423, 2935)), ((1242, 561), (1242, 2935)),                               # singles sidelines
               ((423, 1110), (1242, 1110)), ((423, 2386), (1242, 2386)),                             # service lines
               ((832, 1110), (832, 2386))]                                                           # centre line
COURT_SIZE = (1665, 3496)            # whole reference canvas incl. a margin around the court (w, h)

GREEN, PINK, YELLOW, WHITE, RED = (80, 220, 80), (200, 80, 255), (0, 230, 255), (255, 255, 255), (40, 40, 255)
HIT_WINDOW = 9                         # frames of ball history used to spot a racket hit (see decode.find_hit)
SHOW_HIT_FRAMES = 45                   # how long a HIT marker stays visible (1.5 s)


def draw_minimap_background(height):
    scale = height / COURT_SIZE[1]
    w = int(COURT_SIZE[0] * scale)
    img = np.full((height, w, 3), (60, 110, 60), np.uint8)               # green surround
    court = np.int32(np.array([[286, 561], [1379, 561], [1379, 2935], [286, 2935]]) * scale)
    cv2.fillPoly(img, [court], (140, 90, 40))                            # blue court
    for a, b in COURT_LINES:
        cv2.line(img, tuple(np.int32(np.array(a) * scale)), tuple(np.int32(np.array(b) * scale)), WHITE, 2)
    return img, scale


def to_court(H_image_to_court, point):
    """Map one image point (x, y) onto the top-down court."""
    p = cv2.perspectiveTransform(np.float32([[point]]), H_image_to_court)[0, 0]
    return float(p[0]), float(p[1])


# ------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="game10/Clip1", help="a clip (game10/Clip1) or a whole match (game10)")
    ap.add_argument("--weights", default=str(ROOT / "runs/lite_v2/best.pt"))
    ap.add_argument("--mode", choices=["boxes", "minimap", "both"], default="both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load(args.weights, map_location="cpu")
    color = weights["stem.0.weight"].shape[1] == 9          # a colour model takes 9 input channels
    model = TennisNetLite(pretrained=False, color=color)
    model.load_state_dict(weights)
    model = model.to(device).eval()

    files = sorted((IMAGES / args.clip).glob("**/*.jpg"), key=lambda p: (p.parent.name, p.name))
    first = cv2.imread(str(files[0]))
    H, W = first.shape[:2]
    minimap_bg, mm_scale = draw_minimap_background(H)
    draw_boxes = args.mode in ("boxes", "both")
    draw_map = args.mode in ("minimap", "both")
    out_w = W + (minimap_bg.shape[1] if draw_map else 0)
    out_path = args.out or str(ROOT / f"runs/demo_{args.clip.replace('/', '_')}_{args.mode}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (out_w, H))

    last3 = deque(maxlen=3)              # the 3-frame window the model needs
    ball_trail = deque(maxlen=8)         # recent ball positions (image pixels), for drawing
    ball_history = deque(maxlen=HIT_WINDOW)     # recent ball positions, for hit detection
    player_history = deque(maxlen=HIT_WINDOW)   # recent player boxes, for hit detection
    hits = []                            # (frame number, ball point, feet of the player who hit)
    map_trail = deque(maxlen=8)          # recent ball positions (minimap pixels)
    smooth_court = None                  # camera is fixed: average the court points over time
    clip_name = None

    for frame_no, f in enumerate(files):
        if f.parent.name != clip_name:   # new clip: forget the past
            clip_name, smooth_court, smooth_players = f.parent.name, None, {}
            last3.clear(); ball_trail.clear(); map_trail.clear()
            ball_history.clear(); player_history.clear(); hits.clear()

        frame = cv2.imread(str(f))
        small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB if color else cv2.COLOR_BGR2GRAY)
        small = cv2.resize(small, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        last3.append(small)
        while len(last3) < 3:            # start of a clip: repeat the first frame
            last3.appendleft(small)

        x = np.stack(last3)                                  # (3, H, W) gray or (3, H, W, RGB) colour
        if color:
            x = x.transpose(0, 3, 1, 2).reshape(9, IN_H, IN_W)
        x = torch.from_numpy(x / 255.0).float()[None].to(device)
        with torch.no_grad():
            heatmaps, sizes = model(x)
        ball, court, players = decode(heatmaps[0].cpu().numpy(), sizes[0].cpu().numpy())

        court = np.float32(court)
        smooth_court = court if smooth_court is None else 0.9 * smooth_court + 0.1 * court
        H_court_to_image, _ = cv2.findHomography(COURT_KEYPOINTS, smooth_court, cv2.RANSAC, 5.0)

        ball_trail.append(ball)                                     # None when the ball was not seen
        ball_history.append(ball)
        player_history.append(players)

        # racket hit: decided a few frames late, once we have seen where the ball went next
        middle = HIT_WINDOW // 2
        if len(ball_history) == HIT_WINDOW and (not hits or frame_no - hits[-1][0] > 10):
            hit = find_hit(ball_history, player_history[middle])
            if hit:
                hitter = min(player_history[middle], key=lambda b: np.hypot((b[0] + b[2]) / 2 - hit[0], (b[1] + b[3]) / 2 - hit[1]))
                hits.append((frame_no - middle, hit, ((hitter[0] + hitter[2]) / 2, hitter[3])))

        # ---------------- mode 1: draw on the video
        if draw_boxes:
            if H_court_to_image is not None:
                for a, b in COURT_LINES:
                    pa, pb = cv2.perspectiveTransform(np.float32([[a, b]]), H_court_to_image)[0]
                    cv2.line(frame, tuple(np.int32(pa)), tuple(np.int32(pb)), (255, 200, 0), 1, cv2.LINE_AA)
            for x1, y1, x2, y2 in players:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
            trail = [p for p in ball_trail if p is not None]
            for i in range(1, len(trail)):
                cv2.line(frame, tuple(np.int32(trail[i - 1])), tuple(np.int32(trail[i])), YELLOW, 2, cv2.LINE_AA)
            if ball:
                cv2.circle(frame, tuple(np.int32(ball)), 8, YELLOW, 2, cv2.LINE_AA)
            for when, point, _ in hits:
                if frame_no - when < SHOW_HIT_FRAMES:
                    p = tuple(np.int32(point))
                    cv2.circle(frame, p, 16, RED, 3, cv2.LINE_AA)
                    cv2.putText(frame, "HIT", (p[0] + 18, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)

        # ---------------- mode 2: top-down minimap on the right
        if draw_map:
            minimap = minimap_bg.copy()
            if H_court_to_image is not None:
                H_image_to_court = np.linalg.inv(H_court_to_image)
                for x1, y1, x2, y2 in players:
                    feet = ((x1 + x2) / 2, y2)                       # players stand on the ground: use their feet
                    pos = np.array(to_court(H_image_to_court, feet))
                    side = "far" if pos[1] < 1748 else "near"         # which half of the court (net is at y=1748)
                    # smoothing: move only 25% of the way to the new position each frame
                    smooth_players[side] = pos if side not in smooth_players else 0.75 * smooth_players[side] + 0.25 * pos
                    cx, cy = smooth_players[side]
                    cv2.circle(minimap, (int(cx * mm_scale), int(cy * mm_scale)), 9, PINK, -1, cv2.LINE_AA)
                # the ball is usually in the air, so this is its "shadow" on the ground - approximate
                map_trail.append(to_court(H_image_to_court, ball) if ball else None)
                pts = [(int(p[0] * mm_scale), int(p[1] * mm_scale)) for p in map_trail if p is not None]
                for i in range(1, len(pts)):
                    cv2.line(minimap, pts[i - 1], pts[i], YELLOW, 1, cv2.LINE_AA)
                if ball and pts:
                    cv2.circle(minimap, pts[-1], 5, YELLOW, -1, cv2.LINE_AA)
                for when, _, feet in hits:      # the ball is in the air at a hit, so mark the hitter's feet
                    if frame_no - when < SHOW_HIT_FRAMES:
                        cx, cy = to_court(H_image_to_court, feet)
                        cv2.circle(minimap, (int(cx * mm_scale), int(cy * mm_scale)), 14, RED, 3, cv2.LINE_AA)
            frame = np.hstack([frame, minimap])

        writer.write(frame)

    writer.release()
    print(f"wrote {len(files)} frames -> {out_path}")


if __name__ == "__main__":
    main()

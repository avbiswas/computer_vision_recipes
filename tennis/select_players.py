"""Pick the two players from raw Grounding DINO person detections and export.

Heuristic for the fixed broadcast view (behind-the-baseline camera):
  - drop boxes whose centre is in the outer 20% left/right (line judges, umpire, ball kids)
  - near player: best-scoring box with centre y > 200
  - far player:  best-scoring remaining box with bottom y >= 130 and centre y < 300
                 (spectators in the stands sit higher, bottoms above ~125px)
Outputs:
  labels/players.jsonl   one line per frame: {"file", "players":[{"box","score","role"}]}
  labels/players_coco.json  COCO detections (category "player"), boxes as x,y,w,h
  labels/preview/*.jpg   a few frames with boxes drawn
"""
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
DATA = ROOT / "raw/TrackNet/Dataset"
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "labels/players_raw.jsonl"))
ap.add_argument("--out", default=str(ROOT / "labels"))
ap.add_argument("--min_score", type=float, default=0.3)
ap.add_argument("--static_frac", type=float, default=0.75)
ap.add_argument("--static_iou", type=float, default=0.7)
ap.add_argument("--edge", type=float, default=0.05)
ap.add_argument("--previews", type=int, default=12)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def iou(a_, b_):
    ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
    iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
    i = ix * iy
    u = (a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - i
    return i / u if u else 0


def static_boxes(clip_rows):
    """Boxes seen at (almost) the same place in >= static_frac of a clip's frames: not players."""
    n = len(clip_rows)
    if n < 10:
        return []
    step = max(1, n // 60)  # subsample for speed
    sub = clip_rows[::step]
    stat = []
    for d in sub:
        for x in d["dets"]:
            b = x["box"]
            if any(iou(b, s) > a.static_iou for s in stat):
                continue
            hits = sum(any(iou(b, y["box"]) > a.static_iou for y in e["dets"]) for e in sub)
            if hits >= a.static_frac * len(sub):
                stat.append(b)
    return stat


def pick(d, stat):
    W, H = d["w"], d["h"]
    c = []
    for x in d["dets"]:
        b = x["box"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x["score"] < a.min_score or not (a.edge * W < cx < (1 - a.edge) * W):
            continue
        if any(iou(b, s) > a.static_iou for s in stat):
            continue
        c.append((x, cy))
    c.sort(key=lambda t: -t[0]["score"])
    top2 = sorted(c[:2], key=lambda t: -t[1])  # lower in image = near
    res = []
    for t, role in zip(top2, ["near", "far"] if len(top2) == 2 else
                       (["near"] if top2 and top2[0][1] > 0.45 * H else ["far"])):
        res.append({"box": t[0]["box"], "score": t[0]["score"], "role": role})
    return res


rows = [json.loads(l) for l in open(a.raw)]
clips = {}
for d in rows:
    clips.setdefault(d["file"].rsplit("/", 1)[0], []).append(d)
STAT = {k: static_boxes(sorted(v, key=lambda d: d["file"])) for k, v in clips.items()}
print("static boxes per clip:", {k: len(v) for k, v in list(STAT.items())[:5]}, "...")
coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "player"}]}
stats = {0: 0, 1: 0, 2: 0}
with open(out / "players.jsonl", "w") as fh:
    for i, d in enumerate(rows):
        ps = pick(d, STAT[d["file"].rsplit("/", 1)[0]])
        stats[len(ps)] += 1
        fh.write(json.dumps({"file": d["file"], "players": ps}) + "\n")
        coco["images"].append({"id": i, "file_name": d["file"], "width": d["w"], "height": d["h"]})
        for p in ps:
            x1, y1, x2, y2 = p["box"]
            coco["annotations"].append({
                "id": len(coco["annotations"]), "image_id": i, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0, "score": p["score"], "attributes": {"role": p["role"]},
            })
json.dump(coco, open(out / "players_coco.json", "w"))
print(f"{len(rows)} frames | 2 players: {stats[2]}  1: {stats[1]}  0: {stats[0]}")

pv = out / "preview"
pv.mkdir(exist_ok=True)
random.seed(0)
for d in random.sample(rows, min(a.previews, len(rows))):
    im = Image.open(DATA / d["file"]).convert("RGB")
    dr = ImageDraw.Draw(im)
    for x in d["dets"]:
        dr.rectangle(x["box"], outline=(120, 120, 120), width=1)
    for p in pick(d, STAT[d["file"].rsplit("/", 1)[0]]):
        dr.rectangle(p["box"], outline=(0, 255, 0) if p["role"] == "near" else (255, 0, 255), width=3)
        dr.text((p["box"][0], p["box"][1] - 12), f'{p["role"]} {p["score"]:.2f}', fill=(255, 255, 0))
    im.save(pv / d["file"].replace("/", "_"))

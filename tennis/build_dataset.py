"""Merge ball (TrackNet CSV), players (Grounding DINO) and court (TennisCourtDetector) labels.

Writes:
  data/tennis.jsonl   one line per frame, sorted by game/clip/frame:
    {"file": "game1/Clip1/0000.jpg", "game": "game1", "clip": "game1/Clip1", "idx": 0,
     "split": "train"|"val"|"test", "w": 1280, "h": 720,
     "ball": {"xy": [x,y] | null, "vis": 0-3, "status": 0 flying|1 hit|2 bounce|null},
     "players": [{"role": "near"|"far", "box": [x1,y1,x2,y2], "score": s}],
     "court": {"ok": bool, "kps": [[x,y]]*14 | null, "H": 3x3 ref->image | null, "err": px}}
  data/meta.json      keypoint names, horizontal-flip index map, splits, label docs.
Images stay in raw/TrackNet/Dataset (paths are relative to it).
"""
import csv, json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "raw/TrackNet/Dataset"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

SPLITS = {"val": ["game9"], "test": ["game10"]}  # everything else is train; split by match, never by frame
split_of = lambda g: next((s for s, gs in SPLITS.items() if g in gs), "train")

KP_NAMES = ["far_base_L", "far_base_R", "near_base_L", "near_base_R",
            "far_singles_L", "near_singles_L", "far_singles_R", "near_singles_R",
            "far_service_L", "far_service_R", "near_service_L", "near_service_R",
            "far_service_T", "near_service_T"]
KP_FLIP = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]  # index swap under horizontal flip


def load(p):
    return {(d := json.loads(l))["file"]: d for l in open(p)}


players = load(ROOT / "labels/players.jsonl")
court = load(ROOT / "labels/court.jsonl")
W, H = 1280, 720

rows = []
for lab in sorted(DATA.glob("*/*/Label.csv")):
    clip = lab.parent.relative_to(DATA).as_posix()
    game = clip.split("/")[0]
    for r in csv.DictReader(open(lab)):
        f = f"{clip}/{r['file name']}"
        vis = int(r["visibility"])
        xy = [float(r["x-coordinate"]), float(r["y-coordinate"])] if vis and r["x-coordinate"] else None
        st = r["status"].strip()
        c = court.get(f, {})
        rows.append({
            "file": f, "game": game, "clip": clip, "idx": int(Path(f).stem), "split": split_of(game),
            "w": W, "h": H,
            "ball": {"xy": xy, "vis": vis, "status": int(float(st)) if st else None},
            "players": [{k: p[k] for k in ("role", "box", "score")} for p in players.get(f, {}).get("players", [])],
            "court": {"ok": bool(c.get("ok")), "kps": c.get("kps"), "H": c.get("H"), "err": c.get("err")},
        })

missing = [r["file"] for r in rows if not (DATA / r["file"]).exists()]
assert not missing, f"{len(missing)} label rows without images, e.g. {missing[:3]}"
rows.sort(key=lambda r: (r["game"], r["clip"], r["idx"]))
with open(OUT / "tennis.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

stats = {}
for s in ("train", "val", "test"):
    rs = [r for r in rows if r["split"] == s]
    stats[s] = {"frames": len(rs), "clips": len({r["clip"] for r in rs}),
                "ball_visible": sum(r["ball"]["xy"] is not None for r in rs),
                "bounces": sum(r["ball"]["status"] == 2 for r in rs),
                "two_players": sum(len(r["players"]) == 2 for r in rs),
                "court_ok": sum(r["court"]["ok"] for r in rs)}
json.dump({"image_root": "raw/TrackNet/Dataset", "size": [W, H], "splits": SPLITS | {"train": "rest"},
           "court_kp_names": KP_NAMES, "court_kp_flip": KP_FLIP,
           "ball_vis": {"0": "not in frame", "1": "visible", "2": "hard to see (blur)", "3": "occluded"},
           "ball_status": {"0": "flying", "1": "hit", "2": "bounce"},
           "sources": {"ball": "TrackNet Label.csv (human)", "players": "Grounding DINO tiny + select_players.py",
                       "court": "TennisCourtDetector + per-clip RANSAC (+ view-warp for game4)"},
           "stats": stats}, open(OUT / "meta.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

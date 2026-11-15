"""Robust court homography per clip (fixed broadcast camera within a clip).

Pools the network's raw keypoints (kps_net) from every frame of a clip and fits one
reference->image homography with RANSAC. Per frame we then check how well the clip H
matches that frame's own detections; if the camera moved (median residual large) we
fall back to a per-frame RANSAC fit.

Writes labels/court.jsonl, one line per frame:
  {"file", "ok": usable label?, "kps": [[x,y]]*14, "H": 3x3 (court reference -> image), "src": "clip"|"frame"|"none", "err": px}
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party/TennisCourtDetector"))
from homography import refer_kps  # (14,1,2) reference court points

REF = refer_kps[:, 0, :].astype(np.float32)
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "labels/court_raw.jsonl"))
ap.add_argument("--out", default=str(ROOT / "labels/court.jsonl"))
ap.add_argument("--ransac_px", type=float, default=4.0)
ap.add_argument("--ok_px", type=float, default=10.0, help="err above this = unreliable label")
ap.add_argument("--moved_px", type=float, default=8.0, help="median residual above this = camera moved")
a = ap.parse_args()


def corr(rows):
    src, dst = [], []
    for r in rows:
        for i, p in enumerate(r["kps_net"]):
            if p is not None:
                src.append(REF[i]); dst.append(p)
    return np.float32(src), np.float32(dst)


def resid(H, r):
    proj = cv2.perspectiveTransform(refer_kps, H)[:, 0, :]
    d = [np.hypot(*(proj[i] - p)) for i, p in enumerate(r["kps_net"]) if p is not None]
    return float(np.median(d)) if d else None, proj


rows = [json.loads(l) for l in open(a.raw)]
clips = defaultdict(list)
for r in rows:
    clips[r["file"].rsplit("/", 1)[0]].append(r)

stats = defaultdict(int)
with open(a.out, "w") as fh:
    for clip, rs in sorted(clips.items()):
        s, d = corr(rs)
        Hc = cv2.findHomography(s, d, cv2.RANSAC, a.ransac_px)[0] if len(s) >= 4 else None
        for r in sorted(rs, key=lambda r: r["file"]):
            H, src, err, kps = None, "none", None, None
            if Hc is not None:
                err, proj = resid(Hc, r)
                if err is not None and err <= a.moved_px:
                    H, src, kps = Hc, "clip", proj
            if H is None:  # camera moved or no clip fit: per-frame robust fit
                fs, fd = corr([r])
                if len(fs) >= 6:
                    Hf = cv2.findHomography(fs, fd, cv2.RANSAC, a.ransac_px)[0]
                    if Hf is not None:
                        e, proj = resid(Hf, r)
                        H, src, err, kps = Hf, "frame", e, proj
            stats[src] += 1
            fh.write(json.dumps({
                "file": r["file"], "src": src, "ok": bool(H is not None and err is not None and err <= a.ok_px), "err": None if err is None else round(err, 2),
                "kps": None if kps is None else [[round(float(x), 1), round(float(y), 1)] for x, y in kps],
                "H": None if H is None else np.round(H, 6).tolist(),
            }) + "\n")
print(dict(stats), "frames;", len(clips), "clips")

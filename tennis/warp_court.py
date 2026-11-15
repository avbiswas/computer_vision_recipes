"""Test-time view normalisation for the court detector.

For frames where the detector struggles (e.g. game4's low camera), warp the frame so the
court sits where it does in a typical broadcast view, run the detector, then map the
keypoints back with the inverse warp.

  W = H_canon @ inv(H_rough)       image(game4) -> image(canonical view)
  H_rough: best existing fit for the clip (lowest err frame in labels/court.jsonl)
  H_canon: a good clip's court->image homography (default game1/Clip1)

Writes the same format as court_raw.jsonl (kps_net in ORIGINAL image coords),
so fit_court_clips.py can be run on it.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch

cv2.setNumThreads(0)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party/TennisCourtDetector"))
from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps

DATA = ROOT / "raw/TrackNet/Dataset"
ap = argparse.ArgumentParser()
ap.add_argument("--games", default="game4")
ap.add_argument("--fits", default=str(ROOT / "labels/court.jsonl"), help="rough fits to warp from")
ap.add_argument("--canon", default="game1/Clip1")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--refine", action="store_true", help="classical line refinement (in warped image)")
ap.add_argument("--out", default=str(ROOT / "labels/court_raw_warp.jsonl"))
a = ap.parse_args()

fits = [json.loads(l) for l in open(a.fits)]
canon_fits = [json.loads(l) for l in open(ROOT / "labels/court.jsonl")]
H_canon = np.array(next(f["H"] for f in canon_fits if f["file"].startswith(a.canon + "/") and f["src"] == "clip"))
best = {}
for f in fits:
    c = f["file"].rsplit("/", 1)[0]
    if f["H"] is not None and (c not in best or f["err"] < best[c]["err"]):
        best[c] = f

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = BallTrackerNet(out_channels=15)
model.load_state_dict(torch.load(ROOT / "models/court_model.pt", map_location="cpu"))
model = model.to(dev).eval()

games = a.games.split(",")
files = sorted(p.relative_to(DATA).as_posix() for g in games for p in (DATA / g).glob("*/*.jpg"))[:: a.stride]
print(len(files), "frames")
with open(a.out, "w") as fh, torch.inference_mode():
    for f in files:
        clip = f.rsplit("/", 1)[0]
        W = H_canon @ np.linalg.inv(np.array(best[clip]["H"]))
        im = cv2.imread(str(DATA / f))
        wim = cv2.warpPerspective(im, W, (im.shape[1], im.shape[0]))
        x = cv2.resize(wim, (640, 360)).astype(np.float32) / 255.0
        p = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)))[0].cpu().numpy()
        Wi = np.linalg.inv(W)
        pts = []
        for k in range(14):
            u, v = postprocess((p[k] * 255).astype(np.uint8), low_thresh=170, max_radius=25)
            if u is None or v is None:
                pts.append(None); continue
            if a.refine and k not in (8, 9, 12):
                try:
                    u, v = refine_kps(wim, int(v), int(u))
                except Exception:
                    pass
            q = cv2.perspectiveTransform(np.float32([[[u, v]]]), Wi)[0, 0]
            pts.append([round(float(q[0]), 1), round(float(q[1]), 1)])
        fh.write(json.dumps({"file": f, "kps_net": pts}) + "\n")

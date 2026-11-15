"""Label 14 tennis-court keypoints on every frame with the pretrained TennisCourtDetector.

Writes labels/court_raw.jsonl (resumable), one line per frame:
  {"file", "kps_net": [[x,y]|null]*14,   # network + refine_kps
   "kps": [[x,y]]*14 | null,              # after homography fit (fills occluded points)
   "H": 3x3 court-reference -> image | null, "fit_err": mean px distance net vs fitted}
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
cv2.setNumThreads(0)

ROOT = Path(__file__).parent
TCD = ROOT / "third_party/TennisCourtDetector"
sys.path.insert(0, str(TCD))
from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

DATA = ROOT / "raw/TrackNet/Dataset"
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(ROOT / "models/court_model.pt"))
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--out", default=str(ROOT / "labels/court_raw.jsonl"))
a = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = BallTrackerNet(out_channels=15)
model.load_state_dict(torch.load(a.weights, map_location="cpu"))
model = model.to(dev).eval()

files = sorted(p.relative_to(DATA).as_posix() for p in DATA.glob("*/*/*.jpg"))[:: a.stride]
if a.limit:
    files = files[: a.limit]
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["file"] for l in out.open()} if out.exists() else set()
todo = [f for f in files if f not in done]
print(f"{len(files)} frames, {len(done)} done, {len(todo)} to go on {dev}", flush=True)


class DS(torch.utils.data.Dataset):
    def __len__(self): return len(todo)
    def __getitem__(self, i):
        im = cv2.imread(str(DATA / todo[i]))
        x = cv2.resize(im, (640, 360)).astype(np.float32) / 255.0
        return todo[i], torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), im


dl = torch.utils.data.DataLoader(DS(), batch_size=a.batch, num_workers=4, collate_fn=lambda b: list(zip(*b)),
                                 multiprocessing_context="fork", prefetch_factor=4)
t0 = time.time(); n = 0
with out.open("a") as fh, torch.inference_mode():
    for fs, xs, ims in dl:
        pred = torch.sigmoid(model(torch.stack(xs).to(dev))).cpu().numpy()
        for f, p, im in zip(fs, pred, ims):
            pts = []
            for k in range(14):
                x, y = postprocess((p[k] * 255).astype(np.uint8), low_thresh=170, max_radius=25)
                if x is not None and y is not None and k not in (8, 9, 12):
                    try:
                        x, y = refine_kps(im, int(y), int(x))
                    except Exception:
                        pass
                pts.append((None, None) if x is None else (float(x), float(y)))
            H = get_trans_matrix(pts)
            kps, err = None, None
            if H is not None:
                kps = [[round(float(u), 1), round(float(v), 1)] for u, v in cv2.perspectiveTransform(refer_kps, H)[:, 0]]
                d = [np.hypot(pts[i][0] - kps[i][0], pts[i][1] - kps[i][1]) for i in range(14) if pts[i][0] is not None]
                err = round(float(np.mean(d)), 2) if d else None
            fh.write(json.dumps({
                "file": f,
                "kps_net": [None if q[0] is None else [round(q[0], 1), round(q[1], 1)] for q in pts],
                "kps": kps, "H": None if H is None else np.round(H, 6).tolist(), "fit_err": err,
            }) + "\n")
        fh.flush(); n += len(fs)
        if n % 400 < a.batch or n == len(todo):
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {n/el:.2f} fps  eta {(len(todo)-n)/(n/el)/60:.1f} min", flush=True)

"""Local, free auto-labelling of tennis players with Grounding DINO.

Writes one JSON line per frame to labels/players_raw.jsonl (resumable):
  {"file": "game1/Clip1/0000.jpg", "w":1280, "h":720,
   "dets": [{"box":[x1,y1,x2,y2], "score":0.71, "label":"person"}, ...]}
Then run select_players.py to pick the two players and export COCO.
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = Path(__file__).parent
DATA = ROOT / "raw/TrackNet/Dataset"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
ap.add_argument("--prompt", default="a person.")
ap.add_argument("--thresh", type=float, default=0.25)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="only first N frames (testing)")
ap.add_argument("--stride", type=int, default=1, help="every Nth frame")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--short", type=int, default=0, help="resize shortest edge (default model 800)")
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--bench", type=int, default=0, help="time N frames, write nothing")
ap.add_argument("--bench_out", default="")
ap.add_argument("--out", default=str(ROOT / "labels/players_raw.jsonl"))
a = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
proc = AutoProcessor.from_pretrained(a.model)
if a.short:
    proc.image_processor.size = {"shortest_edge": a.short, "longest_edge": int(a.short * 1333 / 800)}
model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to(dev).eval()
if a.fp16:
    model = model.half()

files = sorted(p.relative_to(DATA).as_posix() for p in DATA.glob("*/*/*.jpg"))
files = files[:: a.stride]
if a.limit:
    files = files[: a.limit]

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    done = {json.loads(l)["file"] for l in out.open()}
todo = [f for f in files if f not in done]
print(f"{len(files)} frames, {len(done)} done, {len(todo)} to go on {dev}", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, fs): self.fs = fs
    def __len__(self): return len(self.fs)
    def __getitem__(self, i):
        im = Image.open(DATA / self.fs[i]).convert("RGB")
        x = proc.image_processor(images=im, return_tensors="pt")
        return self.fs[i], im.size, x["pixel_values"][0], x["pixel_mask"][0]


def collate(b):
    fs, sizes, pv, pm = zip(*b)
    H = max(p.shape[1] for p in pv); W = max(p.shape[2] for p in pv)
    PV = torch.zeros(len(b), 3, H, W); PM = torch.zeros(len(b), H, W, dtype=torch.long)
    for i, (p, m) in enumerate(zip(pv, pm)):
        PV[i, :, : p.shape[1], : p.shape[2]] = p; PM[i, : m.shape[0], : m.shape[1]] = m
    return list(fs), list(sizes), PV, PM


if a.bench:
    todo = files[:: max(1, len(files) // a.bench)][: a.bench]
txt = proc.tokenizer([a.prompt] * a.batch, return_tensors="pt").to(dev)
dl = torch.utils.data.DataLoader(DS(todo), batch_size=a.batch, num_workers=a.workers,
                                 collate_fn=collate, persistent_workers=a.workers > 0,
                                 multiprocessing_context="fork" if a.workers else None, prefetch_factor=4 if a.workers else None)
fh = open(a.bench_out if a.bench and a.bench_out else (os.devnull if a.bench else out), "w" if a.bench else "a")
t0 = time.time(); n = 0
with torch.inference_mode():
    for fs, sizes, PV, PM in dl:
        k = len(fs)
        PV = PV.to(dev, dtype=torch.float16 if a.fp16 else torch.float32); PM = PM.to(dev)
        o = model(pixel_values=PV, pixel_mask=PM, input_ids=txt.input_ids[:k],
                  attention_mask=txt.attention_mask[:k], token_type_ids=txt.token_type_ids[:k])
        o.logits = o.logits.float(); o.pred_boxes = o.pred_boxes.float()
        res = proc.post_process_grounded_object_detection(
            o, txt.input_ids[:k], threshold=a.thresh, text_threshold=a.thresh,
            target_sizes=[(h, w) for (w, h) in sizes])
        for f, (w, h), r in zip(fs, sizes, res):
            labels = r.get("text_labels", r.get("labels"))
            dets = [{"box": [round(v, 1) for v in b.tolist()], "score": round(float(s_), 3), "label": str(l)}
                    for b, s_, l in zip(r["boxes"], r["scores"], labels)]
            fh.write(json.dumps({"file": f, "w": w, "h": h, "dets": dets}) + "\n")
        fh.flush(); n += k
        if n % 200 < a.batch or n == len(todo):
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {n/el:.2f} fps  eta {(len(todo)-n)/(n/el)/60:.1f} min", flush=True)
print(f"done {n} in {time.time()-t0:.0f}s ({n/(time.time()-t0):.2f} fps)")

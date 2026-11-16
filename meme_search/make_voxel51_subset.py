"""Small text-free meme set (~100) for Voxel51: show an agent writing captions + user queries.
python make_voxel51_subset.py [--n 100]
Picks test-split memes with clean inpainting (moderate text area), round-robin over topics.
Writes voxel51_100/:
  images/                 text-removed templates (out/clean)
  metadata.csv            topic, title, original OCR text, MemeCap captions (for comparing against the agent)
  text_slots.coco.json    where the text used to be (COCO detections, label "text_slot")
"""
import argparse, csv, json, random, shutil
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
meta = json.load(open("labels/memes_full.json"))
dst = Path(f"voxel51_{a.n}"); (dst / "images").mkdir(parents=True, exist_ok=True)

by = {}
for f, m in meta.items():
    lay = Path("out/layout") / f.replace(".png", ".json")
    if m["split"] != "test" or not lay.exists() or not (Path("out/clean") / f).exists(): continue
    L = json.load(open(lay))
    if not (0.02 <= L["mask_frac"] <= 0.30) or not L["text"]: continue  # skip no-text / mostly-text memes
    by.setdefault(m["topic"], []).append((f, L))
random.seed(0)
for v in by.values(): random.shuffle(v)
pick = []
while len(pick) < a.n and any(by.values()):
    for t in sorted(by):
        if by[t] and len(pick) < a.n: pick.append(by[t].pop())

rows, images, anns = [], [], []
for i, (f, L) in enumerate(pick, 1):
    shutil.copy2(Path("out/clean") / f, dst / "images" / f)
    m, (w, h) = meta[f], L["size"]
    images.append({"id": i, "file_name": f, "width": w, "height": h})
    for t in L["text"]:
        x0, y0, x1, y1 = t["bbox"]
        anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                     "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
    rows.append({"filepath": f, "topic": m["topic"], "title": m["title"],
                 "original_text": " / ".join(t["text"] for t in L["text"]), "n_text_lines": len(L["text"]),
                 "memecap_img_caption": m["img_captions"][0], "memecap_meme_caption": m["meme_captions"][0]})
json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "text_slot"}]},
          open(dst / "text_slots.coco.json", "w"))
with open(dst / "metadata.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(dst, len(rows), {t: sum(r["topic"] == t for r in rows) for t in sorted(by)})

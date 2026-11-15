"""Evaluate one or more layout models on the held-out DocLayNet test subset.

python evaluate.py runs/y11s_doclaynet/weights/best.pt models/baseline/yolov11s-doclaynet.pt
Writes results/eval.json and a markdown table to results/eval.md.
"""
import json, sys
from pathlib import Path
from ultralytics import YOLO

CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
           "Picture", "Section-header", "Table", "Text", "Title"]
out = Path("results"); out.mkdir(exist_ok=True)
res = {}
for w in sys.argv[1:]:
    m = YOLO(w)
    r = m.val(data="data/doclaynet/data.yaml", split="test", imgsz=1024, batch=8, device="mps",
              plots=False, verbose=False, project="runs/eval", name=Path(w).stem, exist_ok=True)
    per = {CLASSES[c]: round(float(r.box.maps[c]), 4) for c in range(len(CLASSES))}
    res[w] = {"mAP50-95": round(float(r.box.map), 4), "mAP50": round(float(r.box.map50), 4), "per_class_mAP50-95": per}
    print(w, res[w]["mAP50-95"], res[w]["mAP50"], flush=True)
json.dump(res, open(out / "eval.json", "w"), indent=2)
names = list(res)
lines = ["| Class | " + " | ".join(Path(n).parent.parent.name if "runs/" in n else Path(n).stem for n in names) + " |",
         "|---|" + "---|" * len(names)]
lines.append("| **mAP50-95 (all)** | " + " | ".join(f"**{res[n]['mAP50-95']:.3f}**" for n in names) + " |")
lines.append("| mAP50 (all) | " + " | ".join(f"{res[n]['mAP50']:.3f}" for n in names) + " |")
for c in CLASSES:
    lines.append(f"| {c} | " + " | ".join(f"{res[n]['per_class_mAP50-95'][c]:.3f}" for n in names) + " |")
(out / "eval.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

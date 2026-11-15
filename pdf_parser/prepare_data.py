"""Download a diverse slice of DocLayNet v1.2 and convert it to YOLO format.

Shards in the HF repo are grouped by document category, so we take shards spread
across the whole range to cover all six categories. Each parquet is deleted after
conversion to keep disk usage low.
"""
import io, json, os, random, sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "docling-project/DocLayNet-v1.2"
CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
           "Picture", "Section-header", "Table", "Text", "Title"]  # DocLayNet ids 1..11
OUT = Path("data/doclaynet")
DEMO = Path("data/demo_pdfs")

PLAN = {
    "train": ([f"data/train-{i:05d}-of-00072.parquet" for i in range(0, 72, 8)], 1),
    "val": ([f"data/validation-{i:05d}-of-00007.parquet" for i in (0, 3, 6)], 3),  # keep every 3rd row
    "test": ([f"data/test-{i:05d}-of-00006.parquet" for i in (0, 2, 4)], 1),
}


def convert(split, shard, stride, stats):
    path = hf_hub_download(REPO, shard, repo_type="dataset", local_dir="raw")
    pf = pq.ParquetFile(path)
    img_dir, lbl_dir = OUT / "images" / split, OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image", "bboxes", "category_id", "metadata"] + (["pdf"] if split == "test" else [])
    row = 0
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=cols).to_pylist():
            row += 1
            if (row - 1) % stride:
                continue
            m = r["metadata"]
            name = f"{m['doc_category']}_{m['page_hash'][:16]}"
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            W, H = img.size
            img.save(img_dir / f"{name}.jpg", quality=90)
            lines = []
            for (x, y, w, h), c in zip(r["bboxes"], r["category_id"]):
                if w <= 0 or h <= 0:
                    continue
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"{c-1} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} {min(w/W,1):.6f} {min(h/H,1):.6f}")
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            stats.setdefault(split, {}).setdefault(m["doc_category"], 0)
            stats[split][m["doc_category"]] += 1
            # keep a few original PDFs from the test split for the end-to-end demo
            if split == "test" and r.get("pdf") and random.random() < 0.03:
                DEMO.mkdir(parents=True, exist_ok=True)
                (DEMO / f"{name}.pdf").write_bytes(r["pdf"])
    os.remove(path)


if __name__ == "__main__":
    random.seed(0)
    stats = {}
    for split, (shards, stride) in PLAN.items():
        for s in shards:
            convert(split, s, stride, stats)
            print(split, s, json.dumps(stats[split]), flush=True)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    json.dump(stats, open(OUT / "stats.json", "w"), indent=2)
    print("done", json.dumps(stats))

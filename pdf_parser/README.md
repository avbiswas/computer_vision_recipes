# PDF → clean data

Layout detector (YOLO11) + reading order + Granite-Docling 258M VLM for tables, formulas and scanned pages → markdown + JSON + layout overlay.

## Run
```
.venv/bin/python parse_pdf.py file.pdf --weights models/baseline/yolov11s-doclaynet.pt [--pages 0,1] [--no-vlm]
.venv/bin/python make_demo.py --weights <w> --out <dir>    # 4-panel before/after images
.venv/bin/python evaluate.py <weights...>                  # results/eval.md
.venv/bin/python train.py --model models/yolo11s.pt --epochs 8 --name y11s_doclaynet
```

## Our fine-tune vs public baseline (DocLayNet test subset, 2,500 pages, mAP50-95)
Ours: COCO YOLO11s, 9,635 train pages (~1/8 of DocLayNet, no patents), 8 epochs, 1024px, M2 Max MPS, **2h56m**.
Baseline: hantian/yolo-doclaynet yolov11s, full DocLayNet (~69k pages).

| | Ours | Baseline |
|---|---|---|
| mAP50-95 | 0.608 | 0.744 |
| mAP50 | 0.789 | 0.906 |
| Title | 0.471 | 0.802 |
| Picture | 0.526 | 0.787 |
| Table | 0.696 | 0.819 |
| Text | 0.786 | 0.852 |
| Page-footer | **0.603** | 0.569 |

Full table: results/eval.md. Val mAP was still rising each epoch (0.36 → 0.58), so we're under-trained, not saturated.

## Demos
`demo_baseline/` and `demo_finetuned/`: Transformer table, ResNet 2-column pages, scanned page (no text layer). Each takes ~75 s/page with the VLM on MPS.

## Known issues
- Merged table cells are flattened (OTSL lcel/ucel → empty cells).
- Some captions are detected as Text by our model.
- Reading order is a 2-column heuristic.
- Patents are absent from our train slice.
- Rare classes (Title, Footnote) are weak with 1/8 of the data.

## Next
- Train longer or on more shards (title/picture gap).
- Use the baseline detector as the default in the pipeline.
- LoRA the VLM on table crops.
- Add Voxel51 later: embeddings of failures, auto-label checks.

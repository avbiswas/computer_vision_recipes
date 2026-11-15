# Meme search

Type a situation, get a blank meme template back with the text slots pre-filled.

```
query ──CLIP──> best text-free template ──> slots (from OCR boxes) ──> pre-filled, editable meme
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `baseline.py` | Zero-shot CLIP retrieval on the original MemeCap memes |
| 1 | `remove_text.py` | Florence-2 OCR (text + boxes) → mask → LaMa inpaint. Writes `out/clean/` (templates) and `out/layout/` (text boxes, font size, colors) |
| 2 | `train_clip.py` | Fine-tune CLIP on clean template ↔ MemeCap captions; evaluates text→meme retrieval |
| 3 | `search.py` | Query → top-k templates, groups OCR lines into slots, pre-fills them (full query in the main slot, most important words found by CLIP leave-one-out in the others) |
| — | `make_labels.py` | Zero-shot topic labels + metadata export (CSV / FiftyOne JSON) for dataset browsing |

Data: [MemeCap](https://github.com/eujhwang/meme-cap) in `raw/` (6,416 images; 5,823 trainval, 559 test).
LaMa weights: `models/big-lama.pt` from `huggingface.co/fashn-ai/LaMa`.

```bash
python remove_text.py --n 0 --shard 0/6      # run 6 shards in parallel on a GPU (~30 min on an A100)
python train_clip.py --epochs 10 --out runs/clip_b16
python search.py "my code finally compiled after 6 hours of debugging" --ckpt runs/clip_b16/best.pt
```

## Browser demo (transformers.js)

```bash
python web/export_web.py                       # fine-tuned text encoder -> ONNX, template embeddings, slots, thumbnails
python3 -m http.server 8000 -d web             # then open http://127.0.0.1:8000
```
Only the CLIP text encoder runs in the browser (254 MB fp32 ONNX; the int8 version is 64 MB but drifts to cosine ~0.96, so the demo uses fp32).
Template image embeddings are precomputed. Search ~0.2 s, including leave-one-out word importance.

## Results (ViT-B-16, 559 test memes, text→image Recall@10)

| | meaning | title | literal |
|---|---|---|---|
| Zero-shot, original memes (text visible) | 80.3 | 36.9 | 62.9 |
| Zero-shot, text removed | 25.2 | 17.0 | 80.5 |
| Fine-tuned, text removed (best val epoch = 3) | **34.8** | 17.5 | **93.2** |

- Most of the zero-shot "meaning" score came from CLIP reading the printed text: removing it drops R@10 from 80 to 25.
- Removing the text *helps* literal (visual) queries: 63 → 80 zero-shot.
- Fine-tuning recovers part of the meaning gap (25 → 35) and pushes literal retrieval to 93. It overfits after ~3 epochs (5.3k training images); checkpoint is picked on a 500-meme val split.
- Ceiling: many MemeCap meaning captions describe the specific joke that was in the (now removed) text, so they cannot be fully recovered from pixels.

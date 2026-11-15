# Tennis tracking: ball, players and court with one tiny model

Broadcast tennis video in → ball, both players and the 14 court keypoints out, plus a top-down minimap and racket-hit markers.
One **3.45M-parameter** network does all three tasks at **~5 ms/frame** (≈200 fps) on an Apple M2 Max.

```
video frame ─┐
frame t-1   ─┼─► TennisNet-lite ─► ball heatmap ───────────► ball (x, y) ─► trail + HIT detection
frame t-2   ─┘   (ResNet-18 cut    14 court heatmaps ───────► court points ─► homography ─► minimap
                  + U-Net decoder) player heatmap + w,h ────► 2 player boxes ─► feet on the minimap
```

The interesting part is **how the training data was made**: the public dataset only labels the ball,
so players and court were **auto-labelled for free, locally**, by two bigger models (Grounding DINO and a
pretrained court detector). The small model then learns all three tasks from those labels. This is distillation:
slow, big models make the labels once, and a small fast model learns to copy them.

## Results (v2 model, test match `game10`, never seen in training)

| Ball | |
|---|---|
| Precision / recall / F1 | 0.997 / 0.876 / **0.932** |
| Median position error | 1.6 px (in 1280×720) |
| Clearly visible ball found | 91% |
| Faint / blurred ball found | 57% |
| Ball at a racket hit found | 56% |
| Far half of court / near half | 84% / 93% |

Court keypoints: ~1.3 px median error. Players: >99% matched (IoU > 0.5). Both on the validation match.
Racket-hit markers (in the demo): ~90% precision, ~54% recall, reported 4 frames late.

The model almost never hallucinates a ball; it fails by **missing** small, faint balls (far court, at the racket).
Showing hard frames 3× more often (v2) helped hits a little (48% → 56%) but not faint/far balls, which points to
input resolution as the next lever (a far ball is ~2 px on the 256×144 output heatmap).

## Quick start

```bash
bash tennis/setup.sh           # venv + TrackNet dataset + court detector (~3 GB download)
cd tennis
bash label_all.sh              # auto-label players + court, build the dataset  (~1-2 h)
.venv/bin/python lite/train.py                     # train v2 (~6 min/epoch on M2 Max; stop early, see below)
.venv/bin/python lite/evaluate.py                  # the results table above
.venv/bin/python lite/demo.py --clip game10/Clip1  # render runs/demo_game10_Clip1_both.mp4
```

The demo writes `mp4v`. For a file that plays everywhere: `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 out.mp4`.

Already have trained weights? Put them at `runs/lite_v2/best.pt`: the demo then only needs `setup.sh`
(raw frames), no labelling or training. `evaluate.py` additionally needs the dataset from `label_all.sh`.

Training notes:
- `train.py` defaults: 20 epochs, batch 16, hard frames boosted 3×. v2 was stopped after 10 of 20 epochs (it had plateaued on the test match) and that checkpoint was saved as `best.pt`.
  `train.py` itself only writes `last.pt` each epoch, so copy the one you want.
- Training uses ~6 data-loader workers; if your machine runs out of memory, lower `--batch` and `num_workers` in `train.py`.
- `--color` (3 RGB frames = 9 channels) is implemented but was **not** trained to completion; it needs `prepare_frames.py --color` (8.8 GB cache).

## The pipeline, file by file

**1. Auto-labelling (top-level scripts, run by `label_all.sh`)**

| Script | What it does |
|---|---|
| `label_players.py` | Grounding DINO tiny (`"a person."`) on every frame → all people |
| `select_players.py` | Keeps the 2 players: drops boxes that never move (line judges, crowd), boxes at the image edge, low scores |
| `label_court.py` | Pretrained [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) → 14 keypoints per frame |
| `fit_court_clips.py` | The camera is fixed within a clip, so fit **one** homography per clip with RANSAC (robust to bad frames) |
| `warp_court.py` + `merge_court.py` | game4 has a low camera the detector fails on: warp each frame to look like a normal broadcast view, detect, warp back |
| `build_dataset.py` | Merges ball (TrackNet CSVs) + players + court into `data/tennis.jsonl`; split by match (val = game9, test = game10) |

**2. The small model (`lite/`, numbered in the order you should read them)**

| File | Step |
|---|---|
| `prepare_frames.py` | 1. Resize every frame once to 512×288 grayscale and keep them in one memory-mapped array |
| `data.py` | 2. Input = 3 consecutive grayscale frames as the 3 "colour" channels (motion becomes visible). Targets = Gaussian heatmaps. Augmentation: flip/rotate/scale + brightness/blur |
| `model.py` | 3. ResNet-18 stem + layers 1–3 (layer4 dropped: 8M of its 11M weights), U-Net decoder back to 1/2 resolution, one head with 16 heatmaps + player w/h |
| `decode.py` | 4. Heatmaps → points (sub-pixel peak), player boxes (local maxima), racket-hit rule |
| `train.py` | 5. CenterNet focal loss on every heatmap, L1 on sizes; court loss only where the court label is reliable |
| `demo.py` | 6. Render video: boxes/ball/court lines on the frame and/or a top-down minimap on the right |
| `evaluate.py` | 7. Ball accuracy split by difficulty (the table above) |

Key ideas worth explaining:
- **Heatmaps instead of boxes** for tiny things: predicting "where is the peak" is much easier than regressing coordinates.
- **Time as channels**: a single frame can't tell a ball from a white dot; three frames can (it moves).
- **Homography**: 14 court points ↔ a known court diagram gives a 3×3 matrix that maps any ground pixel onto the minimap. Players are mapped by their **feet** (bottom of the box), since only points on the ground map correctly.
- **Racket hit** = the ball's vertical direction flips while it is next to a player (`decode.find_hit`).

## Data and licences

- Ball labels and frames: the TrackNet tennis dataset (Huang et al., 2019), via the download link in [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet).
- Court detector: [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector), cloned at a pinned commit with a two-line NumPy 2 fix (`court_detector.patch`). That repo has no licence file, so its code and weights are downloaded from the author, not redistributed here.
- Player boxes: [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) via 🤗 transformers.
- Nothing large is committed: `raw/`, `labels/`, `data/`, `models/`, `runs/`, `third_party/` are all git-ignored and regenerated by the scripts above.

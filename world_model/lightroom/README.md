# World model: a playable dream of a room

A tiny room with boxes, a flashlight you drive, and fireflies that fly around on their own.
We record it, then train a model to *dream* it: given the last few frames and your key press,
it generates the next frame. After a short warm-up from real frames, every pixel you see is generated.

```
real room ──record──> frames + actions ──VQ-VAE──> 16x16 tokens ──world model──> next frame tokens ──decode──> pixels
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| Environment | `env.py` | `FireflyRoom`: 2–3 fixed boxes, a flashlight cone (tank controls), 1–2 fireflies that wander and bounce off walls. 2D ray casting for shadows |
| 1. Record | `record.py`, `record_all.sh` | A scripted player drives the flashlight; saves frames (64x64) + actions per episode |
| 2. Pack | `pack.py` | Packs episodes into memory-mapped arrays in `cache/` |
| 3. Tokenizer | `vqvae.py` | VQ-VAE: 64x64 frame ↔ 16x16 grid of codebook ids (K=512). Pixel + edge losses, fireflies weighted extra |
| 4. Tokenize | `tokenize_frames.py` | Encodes every frame once; the world model only ever sees integers |
| 5. World model | `dynamics_rope.py`, `train_rope.py` | Genie-style transformer: spatial + temporal attention, 3D RoPE, causal in time, MaskGIT prediction of the next frame (final model, 9.8M params) |
| 6. Play | `play_live.py`, `play_dream.py`, `dream.py` | Drive the flashlight inside the dream, live or side by side with the real room |

`dynamics.py` / `train_dynamics.py` are the first version (learned position embeddings, 4-frame context), kept for comparison.

## Quick start

```bash
# 1. record + pack (256-step episodes for the RoPE model)
python record.py --episodes 4000 --steps 256 --out data_long
python pack.py --data data_long --out cache_long

# 2. tokenizer, then tokenize
python vqvae.py --out runs/vqvae_ft --gan_start 999999 --fly_w 20
python tokenize_frames.py --cache cache_long --vqvae runs/vqvae_ft/vqvae.pt

# 3. world model
python train_rope.py --cache cache_long --d 256 --layers 6 --batch 64 --steps 10000 --out runs/rope1
# (the released model was warm-started from a first-version run with --init)

# 4. play inside the dream
python play_live.py --wm runs/rope1/ckpts/wm_010000.pt
python play_dream.py --wm runs/rope1/ckpts/wm_010000.pt     # real room vs. dream, same key presses
```

`record_all.sh` runs recording in 8 parallel workers; `train_vq*.sh` / `train_wm*.sh` are the shell recipes used for the earlier runs.

Keys in the dream: arrows drive the flashlight, `space` re-syncs from a real frame, `n` new room, `c` records an MP4, `q` quits.

## What mattered

- **No GAN in the tokenizer.** It went unstable; plain L1 + edge loss with fireflies weighted 20x kept them sharp.
- **Train on its own mistakes.** Pure teacher forcing made dreams melt after a few frames; feeding back the model's own predictions (plus token noise) during training fixed the drift.
- **3D RoPE + dense recent frames.** Rotary positions over (row, col, time) were the biggest win; spreading the context over older frames did not help.
- **1-pass decoding** beat multi-pass MaskGIT refinement for rollout quality.

Other tools: `eval_decoding.py` / `eval_onestep.py` (dream error vs. the real room), `progress.py` + `make_progress_video.py` ("watch it learn" videos), `render_showcase*.py` (the showcase videos).

#!/bin/sh
# Full pipeline after recording: tokenizer -> tokens -> world model -> sample dreams.
# Safe to re-run after a restart: finished stages are skipped, the world model resumes.
set -e
mkdir -p logs
[ -f runs/vqvae/DONE ] || { python3 vqvae.py --steps 10000 > logs/vqvae.log 2>&1 && touch runs/vqvae/DONE; }
[ -f cache/tokens.npy ] || python3 tokenize_frames.py > logs/tokenize.log 2>&1
python3 train_dynamics.py --K 4 --out runs/dyn_k4 >> logs/dyn_k4.log 2>&1
python3 dream.py rollout --passes 1 --episodes 3  > logs/dream.log 2>&1
python3 dream.py rollout --passes 8 --episodes 3 >> logs/dream.log 2>&1
echo "all done"

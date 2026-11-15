#!/bin/sh
# World model: tokenize with the frozen fine-tuned VQ-VAE, then train (resumable).
set -e
mkdir -p logs
[ -f cache/tokens.npy ] || python3 tokenize_frames.py > logs/tokenize.log 2>&1
python3 train_dynamics.py --K 4 --out runs/dyn_k4 --change_w 4 --self_p 0.75 --self_max 6 --noise 0.3 >> logs/dyn_k4.log 2>&1
python3 dream.py rollout --passes 1 --episodes 3  > logs/dream.log 2>&1
python3 dream.py rollout --passes 8 --episodes 3 >> logs/dream.log 2>&1
echo "all done"

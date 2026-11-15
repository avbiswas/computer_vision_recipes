#!/bin/sh
# VQ-VAE v2 + high-res snapshots of every checkpoint. Re-run after a restart to resume.
mkdir -p logs runs/vqvae
python3 snapshot_vq.py --watch > logs/snapshot_vq.log 2>&1 &
python3 vqvae.py >> logs/vqvae.log 2>&1 && touch runs/vqvae/DONE
wait

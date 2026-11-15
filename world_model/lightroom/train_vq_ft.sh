#!/bin/sh
# Fine-tune VQ-VAE v2 without the GAN critic, with firefly pixels weighted 20x.
mkdir -p logs runs/vqvae_ft
python3 snapshot_vq.py --watch --ckpt runs/vqvae_ft/vqvae.pt --log logs/vqvae_ft.log \
  --out runs/vqvae_ft/snapshots --done runs/vqvae_ft/DONE > logs/snapshot_vq_ft.log 2>&1 &
python3 vqvae.py --init runs/vqvae/vqvae_step17000_gan.pt --out runs/vqvae_ft --steps 6000 \
  --lr 1.5e-4 --gan_start 999999 --fly_w 20 >> logs/vqvae_ft.log 2>&1 && touch runs/vqvae_ft/DONE
wait

#!/bin/sh
# Continue from the best checkpoint (step 9000) with a fresh, lower LR schedule (resumable).
mkdir -p logs
python3 train_dynamics.py --K 4 --out runs/dyn_k4b --init runs/dyn_k4/best.pt --steps 4000 --lr 1.5e-4 \
  --change_w 4 --self_p 0.6 --self_max 4 --noise 0.3 >> logs/dyn_k4b.log 2>&1

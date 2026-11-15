#!/bin/sh
# Pull dream videos (+ best checkpoints on request) from the GPU box every 5 minutes.
#   ./sync_from_gpu.sh            videos only, loop
#   CKPTS=1 ./sync_from_gpu.sh    also checkpoints
KEY="$HOME/.ssh/primeintellect_ed25519"; HOST=ubuntu@137.23.55.195
REMOTE=computer_vision_recipes/world_model/lightroom/runs
while true; do
  rsync -az -e "ssh -i $KEY -o BatchMode=yes" --include='*/' --include='*.mp4' --include='*.log' \
    ${CKPTS:+--include='*.pt'} --exclude='*' "$HOST:$REMOTE/" runs/cloud/
  rsync -az -e "ssh -i $KEY -o BatchMode=yes" "$HOST:computer_vision_recipes/world_model/lightroom/logs/" runs/cloud/logs/
  [ -n "$ONCE" ] && break
  sleep 300
done

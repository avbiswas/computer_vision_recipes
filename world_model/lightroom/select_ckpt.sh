#!/bin/sh
# Pick the best world-model checkpoint: 12-scene, 60-step dream test (1-pass) on each.
for c in runs/dyn_k4/ckpts/wm_0[0-9]*.pt; do
  s=$(basename $c .pt | sed 's/wm_0*//')
  [ "$s" -ge 8000 ] || continue
  printf "%-6s " $s
  python3 eval_decoding.py --wm $c --scenes 12 --fast 2>/dev/null | sed -n 2p | sed 's/passes 1 temp 0 *//'
done

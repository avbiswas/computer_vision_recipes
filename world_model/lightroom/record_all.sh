#!/bin/sh
# Record 4000 episodes with 8 parallel workers (500 each) into data/
mkdir -p data logs
for w in 0 1 2 3 4 5 6 7; do
  python3 record.py --episodes 500 --steps 64 --start $((w*500)) --out data > logs/record_$w.log 2>&1 &
done
wait
echo "recorded $(ls data | wc -l) episodes"

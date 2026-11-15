"""Replace the game4 rows of labels/court.jsonl with the fixed ones from warp_court.py.

game4 is filmed from an unusually low camera and the court detector fails on it.
label_all.sh re-labels it with warp_court.py (two passes) -> labels/court_g4_p2.jsonl,
and this script swaps those rows into labels/court.jsonl.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
fixed = {json.loads(line)["file"]: line for line in open(ROOT / "labels/court_g4_p2.jsonl")}
lines = [fixed.get(json.loads(line)["file"], line) for line in open(ROOT / "labels/court.jsonl")]
open(ROOT / "labels/court.jsonl", "w").writelines(lines)

game4 = [json.loads(line) for line in lines if line.startswith('{"file": "game4/')]
print(f"game4 court labels usable: {sum(r['ok'] for r in game4)} / {len(game4)}")

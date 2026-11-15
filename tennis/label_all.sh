#!/bin/bash
# Auto-label every TrackNet frame (players + court) and build the training dataset.
# About 1-2 hours on an Apple Silicon laptop. Every step writes files under labels/ or data/.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

# --- players: Grounding DINO "a person." on every frame, then keep the two players
$PY label_players.py --batch 8 --workers 6 --fp16 --short 512      # -> labels/players_raw.jsonl
$PY select_players.py                                                # -> labels/players.jsonl (+ previews)

# --- court: pretrained TennisCourtDetector on every frame, then one robust homography per clip
$PY label_court.py                                                   # -> labels/court_raw.jsonl
$PY fit_court_clips.py                                               # -> labels/court.jsonl

# --- court fix for game4 (low camera): warp to a normal broadcast view, detect, warp back. Two passes.
$PY warp_court.py --out labels/court_raw_g4_p1.jsonl
$PY fit_court_clips.py --raw labels/court_raw_g4_p1.jsonl --out labels/court_g4_p1.jsonl
$PY warp_court.py --refine --fits labels/court_g4_p1.jsonl --out labels/court_raw_g4_p2.jsonl
$PY fit_court_clips.py --raw labels/court_raw_g4_p2.jsonl --out labels/court_g4_p2.jsonl
$PY merge_court.py                                                   # game4 rows -> labels/court.jsonl

# --- merge ball + players + court into one file, then cache small frames for fast training
$PY build_dataset.py                                                 # -> data/tennis.jsonl, data/meta.json
$PY lite/prepare_frames.py                                           # -> data/frames_512x288.npy
echo "dataset ready. Next: .venv/bin/python lite/train.py"

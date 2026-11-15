#!/bin/bash
# One-time setup for tennis/: Python env, TrackNet dataset, TennisCourtDetector code + weights.
# Run from anywhere:  bash tennis/setup.sh
set -e
cd "$(dirname "$0")"

# 1. Python environment
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/pip install -q -r requirements.txt

# 2. TrackNet tennis dataset (10 matches, ~20k frames, ball labels) -> raw/TrackNet/Dataset
#    Source: https://github.com/yastrebksv/TrackNet (Google Drive folder from its README)
if [ ! -d raw/TrackNet/Dataset ]; then
    mkdir -p raw && (cd raw && ../.venv/bin/gdown --folder https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut)
    (cd raw/TrackNet && unzip -q Dataset.zip && rm Dataset.zip)
fi

# 3. TennisCourtDetector (pretrained court-keypoint model, used only to auto-label the court)
#    We pin the commit and apply two tiny NumPy-2 fixes (court_detector.patch).
#    The repo has no license file, so we download its weights from the author's link instead of shipping them.
if [ ! -d third_party/TennisCourtDetector ]; then
    git clone -q https://github.com/yastrebksv/TennisCourtDetector third_party/TennisCourtDetector
    (cd third_party/TennisCourtDetector && git checkout -q e5cd4f1ce26b15361700d3d89e068cbf0e82749e && git apply ../../court_detector.patch)
fi
if [ ! -f models/court_model.pt ]; then
    mkdir -p models && .venv/bin/gdown -q 1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG -O models/court_model.pt
fi

echo "setup done. Next: bash label_all.sh  (or see README.md)"

#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/cavadalab/Documents/scsv/fungitastic2026_2/.venv/bin/python"
MODELS=(
    # "facebook/dinov3-vits16-pretrain-lvd1689m"
    # "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    # "facebook/dinov3-vitb16-pretrain-lvd1689m"
    # "facebook/dinov3-vitl16-pretrain-lvd1689m"
    # "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    "facebook/dinov3-vit7b16-pretrain-lvd1689m"
)

for CURRENT_MODEL_NAME in "${MODELS[@]}"; do
    # image size -> 224
    gbatch --gpus 1 --time 2:00:00 --name dinov3-train MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=train "$PYTHON" dinov3.py
    gbatch --gpus 1 --time 2:00:00 --name dinov3-val MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=val "$PYTHON" dinov3.py
    gbatch --gpus 1 --time 2:00:00 --name dinov3-test MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=test "$PYTHON" dinov3.py

    # image size -> 448
    gbatch --gpus 1 --time 2:00:00 --name dinov3-train MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=train "$PYTHON" dinov3.py
    gbatch --gpus 1 --time 2:00:00 --name dinov3-val MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=val "$PYTHON" dinov3.py
    gbatch --gpus 1 --time 2:00:00 --name dinov3-test MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=test "$PYTHON" dinov3.py
done

#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/cavadalab/Documents/scsv/fungitastic2026_2/.venv/bin/python"
MODELS=(
    # "dinov3-vits16-pretrain-lvd1689m"
    # "dinov3-vits16plus-pretrain-lvd1689m"
    # "dinov3-vitb16-pretrain-lvd1689m"
    # "dinov3-vitl16-pretrain-lvd1689m"
    # "dinov3-vith16plus-pretrain-lvd1689m"
    "dinov3-vit7b16-pretrain-lvd1689m"
)

for CURRENT_MODEL_NAME in "${MODELS[@]}"; do
    # image size -> 224
    gbatch --gpus 1 --time 2:00:00 --name classify-mlp BACKBONE="$CURRENT_MODEL_NAME" IMAGE_SIZE=224 "$PYTHON" mlp_cls_ablation.py

    # image size -> 448
    gbatch --gpus 1 --time 2:00:00 --name classify-mlp BACKBONE="$CURRENT_MODEL_NAME" IMAGE_SIZE=448 "$PYTHON" mlp_cls_ablation.py
    
done

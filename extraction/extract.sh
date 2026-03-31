#!/usr/bin/env bash
set -euo pipefail

BACKGROUNDS=(
    # "normal"
    "crop"
    "crop_black"
    # "masked_black"
    # "masked_blurred"
)

# DINOV2 MODELS
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
    for BACKGROUND in "${BACKGROUNDS[@]}"; do
        gbatch --gpus 1 --time 2:00:00 --name dinov3-train-300-$BACKGROUND MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=train BACKGROUND=$BACKGROUND "$PYTHON" dinov3.py
        gbatch --gpus 1 --time 2:00:00 --name dinov3-train-720-$BACKGROUND MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=train BACKGROUND=$BACKGROUND "$PYTHON" dinov3.py
    done
    # gbatch --gpus 1 --time 2:00:00 --name dinov3-val-300-$BACKGROUND MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=val "$PYTHON" dinov3.py
    # gbatch --gpus 1 --time 2:00:00 --name dinov3-test-300-$BACKGROUND MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=test "$PYTHON" dinov3.py
    # gbatch --gpus 1 --time 2:00:00 --name dinov3-val-300-$BACKGROUND MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=val "$PYTHON" dinov3.py
    # gbatch --gpus 1 --time 2:00:00 --name dinov3-test-720-$BACKGROUND MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=test "$PYTHON" dinov3.py
done


# # DINOV2 MODELS
# MODELS=(
#     "facebook/dinov2-with-registers-small"
#     "facebook/dinov2-with-registers-base"
#     "facebook/dinov2-with-registers-large"
#     "facebook/dinov2-with-registers-giant"
# )

# for CURRENT_MODEL_NAME in "${MODELS[@]}"; do    
#     gbatch --gpus 1 --time 2:00:00 --name dinov2-train MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=train "$PYTHON" dinov2.py
#     gbatch --gpus 1 --time 2:00:00 --name dinov2-val MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=val "$PYTHON" dinov2.py
#     gbatch --gpus 1 --time 2:00:00 --name dinov2-test MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=test "$PYTHON" dinov2.py
#     gbatch --gpus 1 --time 2:00:00 --name dinov2-train MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=train "$PYTHON" dinov2.py
#     gbatch --gpus 1 --time 2:00:00 --name dinov2-val MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=val "$PYTHON" dinov2.py
#     gbatch --gpus 1 --time 2:00:00 --name dinov2-test MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=test "$PYTHON" dinov2.py
# done


# # INTERNVIT MODELS
# MODELS=(
#     "OpenGVLab/InternViT-300M-448px-V2_5"
#     "OpenGVLab/InternViT-6B-448px-V2_5"
# )

# for CURRENT_MODEL_NAME in "${MODELS[@]}"; do    
#     gbatch --gpus 1 --time 2:00:00 --name internvit-train MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=train "$PYTHON" internvit.py
#     gbatch --gpus 1 --time 2:00:00 --name internvit-val MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=val "$PYTHON" internvit.py
#     gbatch --gpus 1 --time 2:00:00 --name internvit-test MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=300 SPLIT=test "$PYTHON" internvitvit
#     gbatch --gpus 1 --time 2:00:00 --name internvit-train MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=train "$PYTHON" internvit.py
#     gbatch --gpus 1 --time 2:00:00 --name internvit-val MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=val "$PYTHON" internvit.py
#     gbatch --gpus 1 --time 2:00:00 --name internvit-test MODEL_NAME="$CURRENT_MODEL_NAME" DATASET_SIZE=720 SPLIT=test "$PYTHON" internvit.py
# done
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
DATASET="${DATASET:-fungi_tastic}"
DATASET_SIZE="${DATASET_SIZE:-720}"
GENERIC_PROMPT="${GENERIC_PROMPT:-mushroom}"
COMPUTE_IOU="${COMPUTE_IOU:-0}"

cd "$ROOT_DIR"

# gbatch --gpus 1 --time 2:00:00 --name sam3-species-train DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=species SPLIT=train "$PYTHON" segmentation/sam3_masks.py
# gbatch --gpus 1 --time 2:00:00 --name sam3-species-val DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=species SPLIT=val "$PYTHON" segmentation/sam3_masks.py
# gbatch --gpus 1 --time 2:00:00 --name sam3-species-test DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=species SPLIT=test "$PYTHON" segmentation/sam3_masks.py

# gbatch --gpus 1 --time 2:00:00 --name sam3-generic-train DATASET="$DATASET" DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=generic GENERIC_PROMPT="$GENERIC_PROMPT" SPLIT=train "$PYTHON" segmentation/sam3_masks.py
# gbatch --gpus 1 --time 2:00:00 --name sam3-generic-val DATASET="$DATASET" DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=generic GENERIC_PROMPT="$GENERIC_PROMPT" SPLIT=val "$PYTHON" segmentation/sam3_masks.py
# gbatch --gpus 1 --time 2:00:00 --name sam3-generic-test DATASET="$DATASET" DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=generic GENERIC_PROMPT="$GENERIC_PROMPT" SPLIT=test "$PYTHON" segmentation/sam3_masks.py

## Segment the 2000 classes
gbatch --gpus 1 --time 2:00:00 --name sam3-generic-train DATASET="$DATASET" DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=generic GENERIC_PROMPT="$GENERIC_PROMPT" SPLIT=train "$PYTHON" segmentation/sam3_full.py
# gbatch --gpus 1 --time 2:00:00 --name sam3-generic-val DATASET="$DATASET" DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=generic GENERIC_PROMPT="$GENERIC_PROMPT" SPLIT=val "$PYTHON" segmentation/sam3_full.py
# gbatch --gpus 1 --time 2:00:00 --name sam3-generic-test DATASET="$DATASET" DATASET_SIZE="$DATASET_SIZE" COMPUTE_IOU="$COMPUTE_IOU" PROMPT_MODE=generic GENERIC_PROMPT="$GENERIC_PROMPT" SPLIT=test "$PYTHON" segmentation/sam3_full.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import os
import sys
from tqdm import tqdm
from typing import Any, Sequence

from PIL import Image
from torch.utils.data import DataLoader

from utils import load_shards, seed_everything, balance_data, remap_labels, load_masks

SEED = 7

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 448)

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")

DATA_SUBSET = os.environ.get("DATA_SUBSET", "all")
SPLIT = os.environ.get("SPLIT", "test")
DATASET_SIZE = os.environ.get("DATASET_SIZE", "720")
TASK = os.environ.get("TASK", "closed")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
PIN_MEMORY = DEVICE == "cuda"
POLYGON_CLOSE_KERNEL_SIZE = 3

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.mask_fungi import MaskFungiTastic
from dataset.utils.mask_vis import get_image_shape, resize_mask_to_image

def collate_batch(
    batch: list[tuple[Image.Image, np.ndarray, int | None, str, list[str]]]
) -> tuple[list[Image.Image], list[np.ndarray], list[int | None], list[str]]:
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch] # this is not present
    labels = [item[2] for item in batch]
    file_paths = [item[3] for item in batch]
    return images, masks, labels, file_paths

# load the dataset
dataset = MaskFungiTastic(
    root=str(DATASET_ROOT),
    data_subset=DATA_SUBSET,
    split=SPLIT,
    size=DATASET_SIZE,
    task=TASK,
    transform=None,
    seg_task="binary",
)

dataloader_kwargs: dict[str, Any] = {
    "batch_size": BATCH_SIZE,
    "shuffle": False,
    "num_workers": NUM_WORKERS,
    "collate_fn": collate_batch,
    "pin_memory": PIN_MEMORY,
}
if NUM_WORKERS > 0:
    dataloader_kwargs["persistent_workers"] = True
    dataloader_kwargs["prefetch_factor"] = 2
dataloader = DataLoader(dataset, **dataloader_kwargs)


def compute_metric(masks):

    class_aware_iou = {}

    progress = tqdm(dataloader, desc="sam3", unit="batch")

    for _, _, labels, file_paths in progress:
        label = labels[0]
        file_path = file_paths[0]
        image_id = file_path.split("/")[-1].replace(".JPG", ".txt")
        gt_mask = masks[image_id]["gt_mask"]
        sam_mask = masks[image_id]["sam_mask"]

        intersection = np.logical_and(gt_mask, sam_mask).sum()
        union = np.logical_or(gt_mask, sam_mask).sum()

        if class_aware_iou.get(label) is None:
            class_aware_iou[label] = {"intersection": 0, "union": 0}

        class_aware_iou[label]["intersection"] += intersection
        class_aware_iou[label]["union"] += union

    macro_iou = 0

    for label, metrics in class_aware_iou.items():
        iou = metrics["intersection"] / metrics["union"] if metrics["union"] > 0 else 0
        class_aware_iou[label]["iou"] = iou
        macro_iou += iou
    
    macro_iou /= len(class_aware_iou)
    return macro_iou, class_aware_iou

if __name__ == "__main__":

    max_samples = 100
    num_seeds = [7, 42, 123, 2024, 9999]
    experiment_name = "prototype"

    sam_masks_general = load_masks(Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/test/720/FungiTastic/test/720p"))
    sam_masks_specific = load_masks(Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_specific_mushroom_200/all/test/720/FungiTastic/test/720p"))

    computed_iou_general, _ = compute_metric(sam_masks_general)
    print(f"Macro IoU - GENERAL: {computed_iou_general:.4f}")
    
    computed_iou_specific, _ = compute_metric(sam_masks_specific)
    print(f"Macro IoU - SPECIFIC: {computed_iou_general:.4f}")



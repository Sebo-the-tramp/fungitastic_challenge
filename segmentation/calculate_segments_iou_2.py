import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")
OUTPUT_ROOT = PROJECT_ROOT / "data_processed"
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "512"))
SEED = 0
DTYPE = torch.bfloat16

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.mask_fungi import MaskFungiTastic
from dataset.utils.mask_vis import get_image_shape, resize_mask_to_image

SPLIT = os.environ.get("SPLIT", "val")
DEFAULT_BATCH_SIZE = 8
MODEL_LOAD_DTYPE = DTYPE
FEATURE_DTYPE = DTYPE
DATA_SUBSET = "all"
DATASET_SIZE = os.environ.get("DATASET_SIZE", "300")
IMAGE_SIZE = 224 if DATASET_SIZE == "300" else 448
TASK = "closed"
SEG_TASK = "binary"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"

from utils import rich_table_to_latex


def collate_batch(batch):
    images = [item[0] for item in batch]
    # from /home/cavadalab/Documents/scsv/fungitastic2026/FungiTastic/dataset/mask_fungi.py
    masks = [torch.from_numpy(resize_mask_to_image(item[1], get_image_shape(item[0]))).unsqueeze(0) for item in batch]    
    labels = [item[2] for item in batch]
    file_paths = [item[3] for item in batch]
    return images, masks, labels, file_paths


def get_dataloader(split):
    dataset = MaskFungiTastic(
        root=str(DATASET_ROOT),
        split=split,
        size=DATASET_SIZE,
        task=TASK,
        data_subset=DATA_SUBSET,
        transform=None,
        seg_task=SEG_TASK,
        workers=8,
    )

    dataloader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "collate_fn": collate_batch,
        }

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader

def polygon_to_mask(polygons, img_width, img_height):
    """
    Converts a list of polygons into a single binary mask.
    segments: List of polygons, where each polygon is a list of (x, y) tuples.
    """
    # 1. Initialize an empty black mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 2. Prepare the list of scaled polygons for OpenCV
    all_polygons_scaled = []

    for polygon in polygons:
        # Scale the normalized (0-1) coordinates to actual pixel values
        pixel_coords = np.array([[x * img_width, y * img_height] for x, y in polygon], dtype=np.int32)
        all_polygons_scaled.append(pixel_coords)

    # 3. Fill all polygons with white (255)    
    if all_polygons_scaled:
        cv2.fillPoly(mask, all_polygons_scaled, 255)

    return mask

def read_segments(file_name, path_segment):
    segment_path = os.path.join(path_segment, file_name.replace(".jpg", ".png"))
    segments = []
    class_ids = []

    if os.path.exists(segment_path):
        with open(segment_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(" ")
                class_id = int(parts[0])
                points = parts[1:]
                polygon = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
                segments.append(polygon)
                class_ids.append(class_id)
        return segments, class_ids
    else:
        return None, None


def compute_iou(image_name, images, masks, path_segment):

    segments, _ = read_segments(image_name + ".txt", path_segment)

    if segments is not None:
        mask_sam = polygon_to_mask(segments, images[0].size[0], images[0].size[1])
        mask_sam_tensor = torch.from_numpy(mask_sam).unsqueeze(0)
        intersection = np.logical_and(masks[0][0].numpy() > 0, mask_sam_tensor[0].numpy() > 0).sum()
        union = np.logical_or(masks[0][0].numpy() > 0, mask_sam_tensor[0].numpy() > 0).sum()
        iou = intersection / union if union > 0 else 0
        return iou, 1 if len(segments) > 0 else 0

    else:
        print(f"No segments found for {image_name}")
        return 0.0, 0

# Limit to the first 5 batches

table = Table(title=f"IoU on different split images")
table.add_column("Mode")
table.add_column("Split", justify="right")
table.add_column("Mean IoU", justify="right")
table.add_column("Non-empty", justify="right")
table.add_column("Total Images", justify="right")

raw_data = []

for split in ["train", "val", "test"]:

    PATH_SEGMENT_GENERIC = f"/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/{split}/720/FungiTastic/{split}/720p"
    PATH_SEGMENT_SPECIFIC = f"/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_specific_200/all/{split}/720/FungiTastic/{split}/720p"

    dataloader = get_dataloader(split)
    limit = len(dataloader)
    print(f"Computing IoU for the first {limit} batches...")

    visualize_masks_gt_sam_flag = False
    avg_iou_specific = 0
    avg_iou_generic = 0
    images_with_masks_specific = 0
    images_with_masks_generic = 0

    for i, (images, masks, labels, file_paths) in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):    
        if i < limit:
            image_name = Path(file_paths[0]).stem

            iou_generic, mask_generic_present = compute_iou(image_name, images, masks, PATH_SEGMENT_GENERIC)
            iou_specific, mask_specific_present = compute_iou(image_name, images, masks, PATH_SEGMENT_SPECIFIC)

            avg_iou_generic += iou_generic
            avg_iou_specific += iou_specific
            images_with_masks_generic += mask_generic_present
            images_with_masks_specific += mask_specific_present

        else:
            break

    raw_data.append({
        "mode": "Generic",
        "split": split,
        "avg_iou": avg_iou_generic / limit,
        "images_with_masks": images_with_masks_generic,
        "total_images": limit,
    })
    raw_data.append({
        "mode": "Specific",
        "split": split,
        "avg_iou": avg_iou_specific / limit,
        "images_with_masks": images_with_masks_specific,
        "total_images": limit,
    })

sorted_data = sorted(raw_data, key=lambda x: (x["mode"]))
for entry in sorted_data:
    table.add_row(
        entry["mode"],
        entry["split"],
        f"{entry['avg_iou']:.4f}",
        str(entry["images_with_masks"]),
        str(entry["total_images"]),
    )

Console().print(table)

latex_output = rich_table_to_latex(table)
with open(OUTPUT_ROOT / f"iou_results_{OUTPUT_NAME}.tex", "w") as f:
    f.write(latex_output)
print(f"LaTeX table saved to {OUTPUT_ROOT / f'iou_results_{OUTPUT_NAME}.tex'}")
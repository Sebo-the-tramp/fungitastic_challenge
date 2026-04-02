import os
import sys
import json
import random
import cv2

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from PIL import Image
from torch.utils.data import DataLoader


import numpy as np
import torch

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "results" / "runs"
EXPERIMENTS_PATH = ROOT / "dashboard" / "experiments.json"
RESULTS_PATH = ROOT / "dashboard" / "results.json"

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")

DATA_SUBSET = os.environ.get("DATA_SUBSET", "all")
SPLIT = os.environ.get("SPLIT", "test")
DATASET_SIZE = os.environ.get("DATASET_SIZE", "720")
TASK = os.environ.get("TASK", "closed")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
PIN_MEMORY = DEVICE == "cuda"
POLYGON_CLOSE_KERNEL_SIZE = 3

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.mask_fungi import MaskFungiTastic
from dataset.utils.mask_vis import get_image_shape, resize_mask_to_image

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_shards(data_path: Path) -> dict[str, torch.Tensor | None]:
    labels = []
    cls_tokens = []
    register_tokens = []
    mean_pooled_patch_tokens = []
    mean_pooled_gt_masked_patch_tokens = []
    mean_pooled_sam_masked_patch_tokens = []
    patch_features = []
    file_paths = []
    has_register_tokens = None

    for shard_path in tqdm(sorted(data_path.glob("*.pt")), desc=f"Loading {data_path.name}", unit="shard"):
        shard_data = torch.load(shard_path, map_location="cpu")
        labels.extend(shard_data["labels"])
        file_paths.extend(shard_data["file_paths"])
        cls_tokens.append(shard_data["cls_token"].float())
        shard_register_tokens = shard_data["register_tokens"]
        if has_register_tokens is None:
            has_register_tokens = shard_register_tokens is not None
        assert has_register_tokens == (shard_register_tokens is not None)
        if shard_register_tokens is not None:
            register_tokens.append(shard_register_tokens.float())
        mean_pooled_patch_tokens.append(shard_data["mean_pooled_patch_tokens"].float())
        mean_pooled_gt_masked_patch_tokens.append(shard_data["mean_pooled_gt_masked_patch_tokens"].float())
        mean_pooled_sam_masked_patch_tokens.append(shard_data["mean_pooled_sam_masked_patch_tokens"].float())
        if "patch_features" in shard_data and shard_data["patch_features"] is not None:
            patch_features.append(shard_data["patch_features"].float())

    return {
        "labels": torch.tensor(labels),
        "file_paths": file_paths,
        "cls_tokens": torch.cat(cls_tokens),
        "register_tokens": torch.cat(register_tokens).to("cpu") if register_tokens else None,
        "mean_pooled_patch_tokens": torch.cat(mean_pooled_patch_tokens).to("cpu"),
        "mean_pooled_gt_masked_patch_tokens": torch.cat(mean_pooled_gt_masked_patch_tokens).to("cpu"),
        "mean_pooled_sam_masked_patch_tokens": torch.cat(mean_pooled_sam_masked_patch_tokens).to("cpu"),
        "patch_features": torch.cat(patch_features) if patch_features else None,
    }

def collate_batch(
    batch: list[tuple[Image.Image, np.ndarray, int | None, str, list[str]]]
) -> tuple[list[Image.Image], list[np.ndarray], list[int | None], list[str]]:
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch] # this is not present
    labels = [item[2] for item in batch]
    file_paths = [item[3] for item in batch]
    return images, masks, labels, file_paths

def read_segments(segment_path):
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
        print(f"Segment not found for {segment_path}")
        return None, None
    


def polygon_to_mask(
    polygons: Sequence[Sequence[tuple[float, float]]],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    contours = []
    for polygon in polygons:
        coords = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(np.rint(coords[:, 0] * (img_width - 1)), 0, img_width - 1)
        coords[:, 1] = np.clip(np.rint(coords[:, 1] * (img_height - 1)), 0, img_height - 1)
        coords = coords.astype(np.int32)
        keep = np.ones(len(coords), dtype=bool)
        keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
        coords = coords[keep]
        if len(coords) > 1 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        contour = coords.reshape(-1, 1, 2)
        if cv2.contourArea(contour) == 0:
            contour = cv2.convexHull(contour)
        if len(contour) >= 3:
            contours.append(contour)

    if contours:
        cv2.fillPoly(mask, contours, 255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (POLYGON_CLOSE_KERNEL_SIZE, POLYGON_CLOSE_KERNEL_SIZE),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        padded = np.pad(mask, 1, constant_values=0)
        flooded = padded.copy()
        flood_mask = np.zeros((img_height + 4, img_width + 4), dtype=np.uint8)
        cv2.floodFill(flooded, flood_mask, (0, 0), 255)
        mask = cv2.bitwise_or(mask, cv2.bitwise_not(flooded)[1:-1, 1:-1])

    return mask


def load_masks(mask_base_path: Path) -> torch.Tensor:

    if "specific" in str(mask_base_path):
        print("Loading specific SAM masks...")
        cache_path = "/home/cavadalab/Documents/scsv/fungitastic2026_2/classification_paper/cache/sam_masks_specific.pt"
    elif "generic" in str(mask_base_path):
        print("Loading generic SAM masks...")
        cache_path = "/home/cavadalab/Documents/scsv/fungitastic2026_2/classification_paper/cache/sam_masks_general.pt"
    else:
        assert False, f"Unknown mask type in path: {mask_base_path}"

    if os.path.exists(cache_path):
        print(f"Loading cached SAM masks from {cache_path}")
        return torch.load(cache_path)

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
    masks = {}

    progress = tqdm(dataloader, desc="sam3", unit="batch")

    not_found_count = 0

    for images, gt_masks, _, file_paths in progress:
        for image, gt_mask, file_path in zip(images, gt_masks, file_paths):
            file_name = file_path.split("/")[-1].replace(".JPG", ".txt")
            mask_path = mask_base_path / file_name
            segments, _ = read_segments(mask_path)
            if segments is not None:
                mask_sam = polygon_to_mask(segments, image.size[0], image.size[1])
                print(f"Image shape for {file_name}: {image.size}")
                sam_mask_tensor = torch.from_numpy(mask_sam).unsqueeze(0)
                gt_mask_tensor = torch.from_numpy(resize_mask_to_image(gt_mask, (image.size[1], image.size[0]))).unsqueeze(0)

                assert sam_mask_tensor.shape == gt_mask_tensor.shape, f"Shape mismatch for {file_name}: SAM mask shape {sam_mask_tensor.shape}, GT mask shape {gt_mask_tensor.shape}"

                masks[file_name] = {
                    "sam_mask": sam_mask_tensor,
                    "gt_mask": gt_mask_tensor,
                }

            else:
                not_found_count += 1
                print(f"Mask not found for {file_name}. Total not found so far: {not_found_count}")

    # Cache the SAM masks for future runs
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(masks, cache_path)

    return masks


def balance_data(data: dict[str, torch.Tensor | None], seed: int, samples_per_class: int) -> dict[str, torch.Tensor | None]:

    # set random seed for reproducibility
    random.seed(seed)

    labels = data["labels"]
    unique_labels = torch.unique(labels)
    balanced_indices = []

    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        if len(label_indices) > samples_per_class:
            sampled_indices = torch.tensor(random.sample(label_indices.tolist(), samples_per_class))
        else:
            sampled_indices = label_indices
        balanced_indices.append(sampled_indices)

    balanced_indices = torch.cat(balanced_indices)
    # balanced_data = {key: (value[balanced_indices] if value is not None else None) for key, value in data.items()} # -> before

    balanced_data = {
        key: (
            # Check if it's a list (like your paths)
            [value[i] for i in balanced_indices.tolist()] if isinstance(value, list) 
            # Otherwise, assume it's a tensor/array that supports direct indexing
            else value[balanced_indices] if value is not None 
            else None
        ) 
        for key, value in data.items()
    }

    return balanced_data


def filter_data(data, classes_to_exclude):
    found_in_set_mask = torch.isin(data['labels'], classes_to_exclude)
    keep_mask = ~found_in_set_mask

    filtered_data = {}
    target_length = len(data['labels'])

    for key, value in data.items():
        if value is not None and hasattr(value, '__len__') and len(value) == target_length:
            # Apply the boolean keep_mask
            filtered_data[key] = value[keep_mask]
        else:
            # Pass through mismatched/None values
            filtered_data[key] = value

    return filtered_data


def remap_labels(y_train: torch.Tensor, y_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train_labels = torch.unique(y_train, sorted=True)
    assert torch.isin(y_test, train_labels).all().item()
    label_map = {int(label): idx for idx, label in enumerate(train_labels.tolist())}
    y_train_mapped = torch.tensor([label_map[int(label)] for label in y_train.tolist()])
    y_test_mapped = torch.tensor([label_map[int(label)] for label in y_test.tolist()])
    return y_train_mapped, y_test_mapped


def remap_labels_val(y_train: torch.Tensor, y_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train_labels = torch.unique(y_train, sorted=True)
    assert torch.isin(y_test, train_labels).all().item()
    label_map = {int(label): idx for idx, label in enumerate(train_labels.tolist())}
    y_train_mapped = torch.tensor([label_map[int(label)] for label in y_train.tolist()])    
    y_test_mapped = torch.tensor([label_map[int(label)] for label in y_test.tolist()])
    return y_train_mapped, y_test_mapped

def refresh_dashboard_results() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = json.loads(EXPERIMENTS_PATH.read_text())["experiments"]
    runs = [json.loads(path.read_text()) for path in sorted(RUNS_ROOT.glob("*/*.json"))]
    planned_items = sum(len(experiment.get("items", [])) for experiment in experiments)
    completed_items = 0
    best_test_acc = None
    axis_best: dict[str, dict[str, dict[str, Any]]] = {}
    experiment_rows: dict[str, list[dict[str, Any]]] = {}

    for run in runs:
        experiment_rows.setdefault(str(run["experiment_id"]), []).extend(
            [{**row, "timestamp": run["timestamp"]} for row in run["rows"]]
        )
        for row in run["rows"]:
            for axis_name, axis_value in row["axes"].items():
                candidate = {
                    "value": str(axis_value),
                    "item_id": str(row["item_id"]),
                    "experiment_id": str(run["experiment_id"]),
                    "test_acc": float(row["metrics"]["test_acc"]),
                    "timestamp": str(run["timestamp"]),
                }
                current = axis_best.setdefault(str(axis_name), {}).get(str(axis_value))
                if current is None or candidate["test_acc"] > current["test_acc"]:
                    axis_best[str(axis_name)][str(axis_value)] = candidate

    experiment_summaries = []
    for experiment in experiments:
        rows = experiment_rows.get(str(experiment["id"]), [])
        items = []
        best = None

        for item in experiment.get("items", []):
            item_rows = [row for row in rows if str(row["item_id"]) == str(item["name"])]
            if item_rows:
                completed_items += 1
                best_row = max(item_rows, key=lambda row: (float(row["metrics"]["test_acc"]), str(row["timestamp"])))
                latest_row = max(item_rows, key=lambda row: str(row["timestamp"]))
                best = best_row if best is None or float(best_row["metrics"]["test_acc"]) > float(best["metrics"]["test_acc"]) else best
                best_test_acc = float(best_row["metrics"]["test_acc"]) if best_test_acc is None else max(best_test_acc, float(best_row["metrics"]["test_acc"]))
                items.append(
                    {
                        "name": str(item["name"]),
                        "status": "done",
                        "best_metrics": dict(best_row["metrics"]),
                        "latest_metrics": dict(latest_row["metrics"]),
                        "best_meta": dict(best_row.get("meta", {})),
                        "latest_meta": dict(latest_row.get("meta", {})),
                        "best_timestamp": str(best_row["timestamp"]),
                        "latest_timestamp": str(latest_row["timestamp"]),
                    }
                )
                continue

            items.append({"name": str(item["name"]), "status": str(item["status"])})

        experiment_summaries.append(
            {
                "id": str(experiment["id"]),
                "num_runs": len({str(row["timestamp"]) for row in rows}),
                "completed_items": sum(1 for item in items if item["status"] == "done"),
                "total_items": len(items),
                "latest_timestamp": max((str(row["timestamp"]) for row in rows), default=None),
                "best": None if best is None else {
                    "item_id": str(best["item_id"]),
                    "test_acc": float(best["metrics"]["test_acc"]),
                },
                "items": items,
            }
        )

    data = {
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "overview": {
            "total_runs": len(runs),
            "planned_experiments": len(experiments),
            "planned_items": planned_items,
            "completed_items": completed_items,
            "remaining_items": planned_items - completed_items,
            "best_test_acc": best_test_acc,
        },
        "axes": [
            {"name": axis_name, "values": [values[value] for value in sorted(values)]}
            for axis_name, values in sorted(axis_best.items())
        ],
        "experiments": experiment_summaries,
    }
    RESULTS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def save_run(
    experiment_id: str,
    script_path: str,
    axes: dict[str, str | int | float],
    rows: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    run_dir = RUNS_ROOT / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_path = run_dir / f"{timestamp.replace(':', '-')}.json"
    assert not run_path.exists()

    payload = {
        "run_id": f"{experiment_id}_{timestamp}",
        "experiment_id": experiment_id,
        "script": script_path,
        "timestamp": timestamp,
        "meta": {} if meta is None else dict(meta),
        "rows": [
            {
                "item_id": str(row["item_id"]),
                "axes": {**axes, **dict(row.get("axes", {}))},
                "metrics": {key: float(value) for key, value in dict(row["metrics"]).items()},
                "meta": dict(row.get("meta", {})),
            }
            for row in rows
        ],
        "notes": notes,
    }
    run_path.write_text(json.dumps(payload, indent=2) + "\n")
    refresh_dashboard_results()
    return run_path



def compute_metrics_final(data_raw, num_classes=None):

    """
    Computes Image-level and Pixel-level metrics from data_raw.
    """
    # --- 1. Image-Level Metrics ---
    # Overall Accuracy: (Total Correct Images) / (Total Images)
    correct_images = sum(1 for d in data_raw if d['pred_class'] == d['gt_class'])
    overall_img_acc = correct_images / len(data_raw)

    # Macro Accuracy (Image-level): Mean of per-class image accuracies
    img_accs_per_class = []
    for cid in range(num_classes):
        class_samples = [d for d in data_raw if d['gt_class'] == cid]
        if len(class_samples) > 0:
            correct = sum(1 for d in class_samples if d['pred_class'] == cid)
            img_accs_per_class.append(correct / len(class_samples))
    
    macro_img_acc = np.mean(img_accs_per_class)

    # --- 2. Pixel-Level Metrics (IoU & Macro Accuracy) ---
    pixel_iou_per_class = []
    pixel_acc_per_class = []

    for cid in range(num_classes):
        # Samples where this class IS the Ground Truth
        gt_class_data = [d for d in data_raw if d['gt_class'] == cid]
        # Samples where we PREDICTED this class (but it might be wrong)
        pred_class_data = [d for d in data_raw if d['pred_class'] == cid]

        if len(gt_class_data) == 0 and len(pred_class_data) == 0:
            continue

        # Intersection: Pixels in SAM mask ONLY if we predicted the right class label
        # (This assumes the whole mask is assigned the predicted class)
        intersection = sum(d['pixel_in'] for d in gt_class_data if d['pred_class'] == cid)

        # Union: (Total GT area) + (Total area we claimed was this class) - Intersection
        total_gt_area = sum(d['total_pixels'] for d in gt_class_data)
        
        # Predicted area is the SAM mask size for every image we labeled as 'cid'
        # Note: we need the SAM mask size. Since pixel_in + pixel_out = total_pixels 
        # in your current loop only for the GT class, we'll use a slightly different logic:
        total_pred_area = sum((d['pixel_in'] + d['pixel_out']) for d in pred_class_data)

        union = total_gt_area + total_pred_area - intersection
        
        # Per-class Pixel Accuracy (Macro)
        if total_gt_area > 0:
            pixel_acc_per_class.append(intersection / total_gt_area)
            
        # Per-class IoU
        if union > 0:
            pixel_iou_per_class.append(intersection / (union + 1e-10))

    return {
        'overall_img_acc': overall_img_acc,
        'macro_img_acc': macro_img_acc,
        'macro_pixel_acc': np.mean(pixel_acc_per_class),
        'mIoU': np.mean(pixel_iou_per_class)
    }


def compute_metrics_final_fast(data_raw, num_classes=None):

    """
    Computes Image-level and Pixel-level metrics from data_raw.
    """
    if isinstance(data_raw, dict):
        gt_class = torch.as_tensor(data_raw["gt_class"], dtype=torch.long)
        pred_class = torch.as_tensor(data_raw["pred_class"], dtype=torch.long)
        total_pixels = torch.as_tensor(data_raw["total_pixels"], dtype=torch.float64)
        pixel_in = torch.as_tensor(data_raw["pixel_in"], dtype=torch.float64)
        pixel_out = torch.as_tensor(data_raw["pixel_out"], dtype=torch.float64)
    else:
        gt_class = torch.tensor([d["gt_class"] for d in data_raw], dtype=torch.long)
        pred_class = torch.tensor([d["pred_class"] for d in data_raw], dtype=torch.long)
        total_pixels = torch.tensor([d["total_pixels"] for d in data_raw], dtype=torch.float64)
        pixel_in = torch.tensor([d["pixel_in"] for d in data_raw], dtype=torch.float64)
        pixel_out = torch.tensor([d["pixel_out"] for d in data_raw], dtype=torch.float64)

    if num_classes is None:
        num_classes = int(torch.cat([gt_class, pred_class]).max().item()) + 1

    correct_mask = pred_class == gt_class
    overall_img_acc = correct_mask.to(torch.float64).mean().item()

    gt_count = torch.bincount(gt_class, minlength=num_classes)
    correct_count = torch.bincount(gt_class[correct_mask], minlength=num_classes)
    valid_gt = gt_count > 0
    macro_img_acc = (correct_count[valid_gt].to(torch.float64) / gt_count[valid_gt]).mean().item()

    intersection = torch.bincount(gt_class[correct_mask], weights=pixel_in[correct_mask], minlength=num_classes)
    total_gt_area = torch.bincount(gt_class, weights=total_pixels, minlength=num_classes)
    total_pred_area = torch.bincount(pred_class, weights=pixel_in + pixel_out, minlength=num_classes)
    union = total_gt_area + total_pred_area - intersection

    macro_pixel_acc = (intersection[valid_gt] / total_gt_area[valid_gt]).mean().item()
    valid_union = union > 0
    miou = (intersection[valid_union] / (union[valid_union] + 1e-10)).mean().item()

    return {
        "overall_img_acc": overall_img_acc,
        "macro_img_acc": macro_img_acc,
        "macro_pixel_acc": macro_pixel_acc,
        "mIoU": miou,
    }

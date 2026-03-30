import os
import sys
import json
import random
import cv2

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
SPLIT = os.environ.get("SPLIT", "train")
DATASET_SIZE = os.environ.get("DATASET_SIZE", "720")
TASK = os.environ.get("TASK", "closed")
PROMPT_MODE = os.environ.get("PROMPT_MODE", "species")
GENERIC_PROMPT = os.environ.get("GENERIC_PROMPT", "mushroom")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
PIN_MEMORY = DEVICE == "cuda"

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.fungi import FungiTastic

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
    has_register_tokens = None

    for shard_path in tqdm(sorted(data_path.glob("*.pt")), desc=f"Loading {data_path.name}", unit="shard"):
        shard_data = torch.load(shard_path, map_location="cpu")
        labels.extend(shard_data["labels"])
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
    # masks = [item[1] for item in batch] # this is not present
    labels = [item[1] for item in batch]
    file_paths = [item[2] for item in batch]
    return images, [], labels, file_paths

def read_segments(file_path):
    segment_path = os.path.join(file_path)
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
        print(f"Segment not found for {file_path}")
        return None
    


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
    # cv2.fillPoly can take a list of arrays directly
    if all_polygons_scaled:
        cv2.fillPoly(mask, all_polygons_scaled, 255)
    
    return mask


def load_masks(mask_path: Path) -> torch.Tensor:

    dataset = FungiTastic(
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

    for images, gt_masks, _, file_paths in progress:        
        for gt_mask, file_path in zip(gt_masks, file_paths):
            file_name = file_path.split("/")[-1].replace(".JPG", ".txt")
            mask_path = mask_path / file_name
            segments, class_ids = read_segments(mask_path)
            if segments is not None:

                mask_sam = polygon_to_mask(segments, images[0].size[0], images[0].size[1])
                mask_sam_tensor = torch.from_numpy(mask_sam).unsqueeze(0)
                masks[file_name] = mask_sam_tensor

    return masks


def balance_data(data: dict[str, torch.Tensor | None], seed: int, samples_per_class: int) -> dict[str, torch.Tensor | None]:
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
    balanced_data = {key: (value[balanced_indices] if value is not None else None) for key, value in data.items()}
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

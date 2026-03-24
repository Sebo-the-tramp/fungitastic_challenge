import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")
MODEL_NAME = "facebook/sam3"
DATA_SUBSET = "all"
SPLIT = "train"
DATASET_SIZE = "720"
TASK = "closed"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "sam3_yolo" / DATA_SUBSET / SPLIT / DATASET_SIZE
BATCH_SIZE = 4
NUM_WORKERS = 8
THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
COMPUTE_IOU = True
MIN_MASK_AREA = 256
POLYGON_EPSILON_RATIO = 0.002
MAX_CONTOURS_PER_MASK = 4
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
PIN_MEMORY = DEVICE == "cuda"

assert hasattr(transformers, "Sam3Model"), "Install a transformers version that exposes Sam3Model."
assert hasattr(transformers, "Sam3Processor"), "Install a transformers version that exposes Sam3Processor."

Sam3Model = transformers.Sam3Model
Sam3Processor = transformers.Sam3Processor

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.mask_fungi import MaskFungiTastic
from dataset.utils.mask_vis import get_image_shape, resize_mask_to_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(
    batch: list[tuple[Image.Image, np.ndarray, int | None, str, list[str]]]
) -> tuple[list[Image.Image], list[np.ndarray], list[int | None], list[str]]:
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    file_paths = [item[3] for item in batch]
    return images, masks, labels, file_paths


def confirm_override(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.rglob("*.txt")):
        answer = input(f"{path} already contains labels. Override files? [y/N]: ").strip().lower()
        assert answer == "y"


def build_label_maps(dataset: MaskFungiTastic) -> tuple[dict[int, int], dict[int, str], list[str]]:
    category_ids = sorted(category_id for category_id in dataset.category_id2label if category_id >= 0)
    category_to_yolo = {category_id: index for index, category_id in enumerate(category_ids)}
    category_to_name = {category_id: dataset.category_id2label[category_id] for category_id in category_ids}
    class_names = [category_to_name[category_id] for category_id in category_ids]
    return category_to_yolo, category_to_name, class_names


def save_metadata(class_names: list[str]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "classes.txt").write_text("\n".join(class_names) + "\n")
    metadata = {
        "model_name": MODEL_NAME,
        "data_subset": DATA_SUBSET,
        "split": SPLIT,
        "dataset_size": DATASET_SIZE,
        "task": TASK,
        "prompt_source": "dataset_species_name",
        "num_classes": len(class_names),
        "compute_iou": COMPUTE_IOU,
        "threshold": THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "min_mask_area": MIN_MASK_AREA,
        "polygon_epsilon_ratio": POLYGON_EPSILON_RATIO,
        "max_contours_per_mask": MAX_CONTOURS_PER_MASK,
    }
    (OUTPUT_ROOT / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def image_path_to_label_path(image_path: str) -> Path:
    relative_path = Path(image_path).relative_to(DATASET_ROOT)
    return OUTPUT_ROOT / relative_path.with_suffix(".txt")


def move_to_device(batch: Any) -> Any:
    if torch.is_tensor(batch):
        if PIN_MEMORY and batch.device.type == "cpu":
            batch = batch.pin_memory()
        kwargs: dict[str, Any] = {"device": DEVICE, "non_blocking": PIN_MEMORY}
        if batch.is_floating_point():
            kwargs["dtype"] = MODEL_DTYPE
        return batch.to(**kwargs)
    if isinstance(batch, dict):
        return {key: move_to_device(value) for key, value in batch.items()}
    return batch


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def mask_to_yolo_segments(mask: np.ndarray, width: int, height: int) -> list[str]:
    uint8_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS_PER_MASK]
    segments: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_MASK_AREA:
            continue
        epsilon = POLYGON_EPSILON_RATIO * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if polygon.shape[0] < 3:
            continue
        normalized = polygon.astype(np.float32)
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        segments.append(coords)
    return segments


def result_to_binary_mask(result: dict[str, Any] | None, height: int, width: int) -> np.ndarray:
    if result is None or len(result["masks"]) == 0:
        return np.zeros((height, width), dtype=bool)
    masks = np.stack([to_numpy(mask).astype(bool) for mask in result["masks"]], axis=0)
    return masks.any(axis=0)


def result_to_yolo_rows(result: dict[str, Any] | None, class_id: int, width: int, height: int) -> list[str]:
    if result is None or len(result["masks"]) == 0:
        return []
    scores = to_numpy(result["scores"]).astype(np.float32)
    order = np.argsort(-scores)
    rows: list[str] = []
    for index in order.tolist():
        mask = to_numpy(result["masks"][index]).astype(np.uint8)
        for segment in mask_to_yolo_segments(mask, width, height):
            rows.append(f"{class_id} {segment}")
    return rows


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64)
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(intersection / union)


def main() -> None:
    seed_everything(SEED)
    confirm_override(OUTPUT_ROOT)

    dataset = MaskFungiTastic(
        root=str(DATASET_ROOT),
        data_subset=DATA_SUBSET,
        split=SPLIT,
        size=DATASET_SIZE,
        task=TASK,
        transform=None,
        seg_task="binary",
    )
    category_to_yolo, category_to_name, class_names = build_label_maps(dataset)
    save_metadata(class_names)

    processor = Sam3Processor.from_pretrained(MODEL_NAME)
    model = Sam3Model.from_pretrained(MODEL_NAME, torch_dtype=MODEL_DTYPE).to(DEVICE).eval()

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

    total_rows = 0
    empty_files = 0
    iou_sum = 0.0
    iou_count = 0

    with torch.inference_mode():
        progress = tqdm(dataloader, desc="sam3", unit="batch")
        for images, masks, labels, file_paths in progress:
            prompts = [category_to_name[int(label)] for label in labels]
            class_ids = [category_to_yolo[int(label)] for label in labels]
            raw_inputs = processor(
                images=images,
                text=prompts,
                return_tensors="pt",
            )
            target_sizes = raw_inputs["original_sizes"].tolist()
            inputs = move_to_device(dict(raw_inputs))
            outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=THRESHOLD,
                mask_threshold=MASK_THRESHOLD,
                target_sizes=target_sizes,
            )

            for image, gt_mask, file_path, class_id, result in zip(images, masks, file_paths, class_ids, results):
                label_path = image_path_to_label_path(file_path)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                width, height = image.size
                rows = result_to_yolo_rows(result, class_id, width, height)
                label_path.write_text("\n".join(rows) + ("\n" if rows else ""))
                total_rows += len(rows)
                empty_files += int(len(rows) == 0)
                if COMPUTE_IOU:
                    gt_mask = resize_mask_to_image(gt_mask, get_image_shape(image))
                    pred_mask = result_to_binary_mask(result, gt_mask.shape[0], gt_mask.shape[1])
                    iou_sum += compute_iou(pred_mask, gt_mask)
                    iou_count += 1
            if COMPUTE_IOU and iou_count > 0:
                progress.set_postfix(mean_iou=f"{iou_sum / iou_count:.4f}", images=iou_count)

    print(f"saved {len(dataset)} labels")
    print(f"saved {total_rows} polygons")
    print(f"empty labels {empty_files}")
    if COMPUTE_IOU and iou_count > 0:
        print(f"mean_iou {iou_sum / iou_count:.6f} on {iou_count} images")


if __name__ == "__main__":
    main()

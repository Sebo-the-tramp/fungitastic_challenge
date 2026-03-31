from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")
METADATA_PATH = DATASET_ROOT / "metadata" / "FungiTastic" / "FungiTastic-Train.csv"
MASKS_PATH = DATASET_ROOT / "masks" / "FungiTastic-Mini-TrainMasks.parquet"
SEGMENT_ROOTS = {
    "Specific": PROJECT_ROOT / "data_processed" / "sam3_yolo_specific_200" / "all" / "train" / "720" / "FungiTastic" / "train" / "720p",
    "Generic": PROJECT_ROOT / "data_processed" / "sam3_yolo_generic_mushroom_200" / "all" / "train" / "720" / "FungiTastic" / "train" / "720p",
}


def load_rows() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH, usecols=["filename"])
    masks = pd.read_parquet(MASKS_PATH)[["file_name", "height", "width", "rle"]]
    masks = masks.groupby("file_name", sort=False).agg({"rle": list, "height": "first", "width": "first"}).reset_index()
    return meta.merge(masks, left_on="filename", right_on="file_name", how="inner")[["filename", "height", "width", "rle"]]


def decode_rle(rle: np.ndarray, height: int, width: int) -> np.ndarray:
    counts = np.asarray(rle[:-4], dtype=np.int64)
    return np.repeat(np.arange(counts.size) % 2, counts).reshape(height, width).astype(bool)


def load_pred_mask(path: Path, height: int, width: int) -> tuple[np.ndarray, bool]:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not path.exists():
        return mask.astype(bool), False
    polygons = []
    for line in path.read_text().splitlines():
        values = [float(value) for value in line.split()[1:]]
        if not values:
            continue
        polygon = np.asarray(list(zip(values[::2], values[1::2])), dtype=np.float32)
        polygon[:, 0] *= width
        polygon[:, 1] *= height
        polygons.append(np.rint(polygon).astype(np.int32))
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool), bool(polygons)


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64) / union)


def main() -> None:
    assert METADATA_PATH.exists()
    assert MASKS_PATH.exists()
    for root in SEGMENT_ROOTS.values():
        assert root.exists(), root
    rows = load_rows()
    sums = {name: 0.0 for name in SEGMENT_ROOTS}
    non_empty = {name: 0 for name in SEGMENT_ROOTS}
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="IoU"):
        gt_mask = decode_rle(row.rle[-1], row.height, row.width)
        file_name = Path(row.filename).with_suffix(".txt").name
        for name, root in SEGMENT_ROOTS.items():
            pred_mask, has_polygons = load_pred_mask(root / file_name, row.height, row.width)
            sums[name] += iou(pred_mask, gt_mask)
            non_empty[name] += int(has_polygons)
    table = Table(title=f"IoU on {len(rows)} images")
    table.add_column("Mode")
    table.add_column("Mean IoU", justify="right")
    table.add_column("Non-empty", justify="right")
    for name in SEGMENT_ROOTS:
        table.add_row(name, f"{sums[name] / len(rows):.4f}", str(non_empty[name]))
    Console().print(table)


if __name__ == "__main__":
    main()

import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from prototype_experiment_common import (
    build_class_indices,
    build_mask_cache,
    build_prediction_outputs,
    build_prototypes,
    confirm_override_paths,
    make_experiment_dir,
    remap_labels_fast,
    sample_balanced_indices,
    squared_euclidean_distances,
)
from utils import compute_metrics_final_fast, load_masks, load_shards, seed_everything

SEED = 7
PCA_DIM = 1024
PCA_EPS = 1e-6
MAX_SAMPLES = 200
NUM_SEEDS = 20

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 448)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
MASKS_PATH = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/test/720/FungiTastic/test/720p")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def pca_whiten_l2_features(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    pca_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_features = train_features.float()
    test_features = test_features.float()
    pca_dim = min(pca_dim, train_features.shape[0] - 1, train_features.shape[1])
    assert pca_dim > 0
    train_mean = train_features.mean(dim=0, keepdim=True)
    train_centered = train_features - train_mean
    test_centered = test_features - train_mean
    _, singular_values, basis = torch.pca_lowrank(train_centered, q=pca_dim, center=False)
    basis = basis[:, :pca_dim]
    scale = singular_values[:pca_dim] / (train_features.shape[0] - 1) ** 0.5
    train_features = F.normalize(train_centered @ basis / scale.clamp_min(PCA_EPS), dim=1)
    test_features = F.normalize(test_centered @ basis / scale.clamp_min(PCA_EPS), dim=1)
    return train_features, test_features


@torch.no_grad()
def prototype_method_pca_white_l2(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    test_file_names: list[str],
    total_pixels: torch.Tensor,
    pixel_in: torch.Tensor,
    pixel_out: torch.Tensor,
    num_classes: int,
    pca_dim: int,
) -> tuple[list[dict[str, int | str]], dict[str, torch.Tensor]]:
    train_features, test_features = pca_whiten_l2_features(train_features, test_features, pca_dim=pca_dim)
    prototypes = F.normalize(build_prototypes(train_features, train_labels, num_classes), dim=1)
    pred_class = squared_euclidean_distances(test_features, prototypes).argmin(dim=1)
    return build_prediction_outputs(
        pred_class=pred_class,
        test_labels=test_labels,
        test_file_names=test_file_names,
        total_pixels=total_pixels,
        pixel_in=pixel_in,
        pixel_out=pixel_out,
    )


def run_sweep(
    min_samples: int = 1,
    max_samples: int | None = None,
    seeds: list[int] = [],
    experiment_name: str = "",
    masks: dict[str, dict[str, torch.Tensor]] = {},
    pca_dim: int = PCA_DIM,
    save_csv: bool = True,
) -> list[dict[str, int | float]]:
    experiment_dir = make_experiment_dir(CLASSIFICATION_RESULTS_DIR, experiment_name)
    if save_csv:
        confirm_override_paths([
            path
            for seed in seeds
            for path in (experiment_dir / f"{seed}_computed.csv", experiment_dir / f"{seed}_raw.csv")
        ])

    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)
    train_labels, test_labels = remap_labels_fast(train_data["labels"], test_data["labels"])
    num_classes = int(train_labels.max().item()) + 1
    train_class_indices = build_class_indices(train_labels, num_classes)
    test_class_indices = build_class_indices(test_labels, num_classes)
    test_file_names_all, total_pixels_all, pixel_in_all, pixel_out_all = build_mask_cache(test_data["file_paths"], masks)
    train_features_all = train_data["cls_tokens"].to(DEVICE)
    train_labels_all = train_labels.to(DEVICE)
    test_features_all = test_data["cls_tokens"].to(DEVICE)
    max_samples = max_samples or max(len(class_indices) for class_indices in train_class_indices)
    all_results = []

    for seed in tqdm(seeds, desc="Seeds", position=0):
        seed_everything(seed)
        computed_path = experiment_dir / f"{seed}_computed.csv"
        raw_path = experiment_dir / f"{seed}_raw.csv"
        computed_file = open(computed_path, "w", newline="") if save_csv else None
        raw_file = open(raw_path, "w", newline="") if save_csv else None
        computed_writer = None
        raw_writer = None

        for samples_per_class in tqdm(range(min_samples, max_samples + 1), desc=f"Samples seed={seed}", position=1, leave=False):
            train_indices = sample_balanced_indices(train_class_indices, seed=seed, samples_per_class=samples_per_class)
            test_indices = sample_balanced_indices(test_class_indices, seed=seed, samples_per_class=samples_per_class)
            raw_data, metrics_data = prototype_method_pca_white_l2(
                train_features=train_features_all[train_indices.to(DEVICE)],
                train_labels=train_labels_all[train_indices.to(DEVICE)],
                test_features=test_features_all[test_indices.to(DEVICE)],
                test_labels=test_labels[test_indices],
                test_file_names=[test_file_names_all[i] for i in test_indices.tolist()],
                total_pixels=total_pixels_all[test_indices],
                pixel_in=pixel_in_all[test_indices],
                pixel_out=pixel_out_all[test_indices],
                num_classes=num_classes,
                pca_dim=pca_dim,
            )
            metrics = compute_metrics_final_fast(metrics_data, num_classes=num_classes)
            result_computed = {
                "samples_per_class": samples_per_class,
                "accuracy_euclidean": metrics["macro_img_acc"],
                "accuracy_euclidean_overall": metrics["overall_img_acc"],
                "mIoU": metrics["mIoU"],
            }
            raw_rows = [{"sample_per_class": samples_per_class, **row} for row in raw_data]
            all_results.append({"seed": seed, **result_computed})

            if save_csv:
                if computed_writer is None:
                    computed_writer = csv.DictWriter(computed_file, fieldnames=result_computed.keys())
                    computed_writer.writeheader()
                if raw_writer is None:
                    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_rows[0].keys())
                    raw_writer.writeheader()
                computed_writer.writerow(result_computed)
                raw_writer.writerows(raw_rows)

        if save_csv:
            computed_file.close()
            raw_file.close()

    return all_results


if __name__ == "__main__":
    np.random.seed(SEED)
    num_seeds = list(map(int, np.random.randint(0, 10000, size=NUM_SEEDS)))
    experiment_name = f"prototype_pca_white_l2_{PCA_DIM}"
    masks = load_masks(MASKS_PATH)
    run_sweep(
        min_samples=1,
        max_samples=MAX_SAMPLES,
        seeds=num_seeds,
        experiment_name=experiment_name,
        masks=masks,
        pca_dim=PCA_DIM,
        save_csv=True,
    )

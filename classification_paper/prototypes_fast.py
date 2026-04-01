import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch
import os
import csv
import random
from tqdm import tqdm

from utils import load_shards, seed_everything, load_masks, compute_metrics_final_fast

SEED = 7

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 448)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def remap_labels_fast(train_labels: torch.Tensor, test_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    label_values = torch.unique(train_labels, sorted=True)
    train_labels_mapped = torch.searchsorted(label_values, train_labels)
    test_labels_mapped = torch.searchsorted(label_values, test_labels)
    assert torch.equal(label_values[test_labels_mapped], test_labels)
    return train_labels_mapped, test_labels_mapped


def build_class_indices(labels: torch.Tensor, num_classes: int) -> list[list[int]]:
    return [torch.where(labels == class_id)[0].tolist() for class_id in range(num_classes)]


def sample_balanced_indices(class_indices: list[list[int]], seed: int, samples_per_class: int) -> torch.Tensor:
    random.seed(seed)
    balanced_indices = []
    for label_indices in class_indices:
        if len(label_indices) > samples_per_class:
            balanced_indices.extend(random.sample(label_indices, samples_per_class))
        else:
            balanced_indices.extend(label_indices)
    return torch.tensor(balanced_indices, dtype=torch.long)


def build_mask_cache(
    file_paths: list[str],
    masks: dict[str, dict[str, torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    file_names = [file_path.split("/")[-1] for file_path in file_paths]
    base_names = [file_name.split(".")[0] for file_name in file_names]
    total_pixels = []
    pixel_in = []
    pixel_out = []

    for file_name in file_names:
        mask_data = masks[file_name.replace(".JPG", ".txt")]
        gt_mask = mask_data["gt_mask"]
        sam_mask = mask_data["sam_mask"]
        total = int(gt_mask.sum().item())
        inside = int((gt_mask & sam_mask).sum().item())
        total_pixels.append(total)
        pixel_in.append(inside)
        pixel_out.append(total - inside)

    return (
        base_names,
        torch.tensor(total_pixels, dtype=torch.long),
        torch.tensor(pixel_in, dtype=torch.long),
        torch.tensor(pixel_out, dtype=torch.long),
    )


@torch.no_grad()
def prototype_method(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_feature_norms: torch.Tensor,
    test_labels: torch.Tensor,
    test_file_names: list[str],
    total_pixels: torch.Tensor,
    pixel_in: torch.Tensor,
    pixel_out: torch.Tensor,
    num_classes: int,
) -> tuple[list[dict[str, int | str]], dict[str, torch.Tensor]]:
    feature_dim = train_features.size(1)
    prototypes = torch.zeros(num_classes, feature_dim, dtype=train_features.dtype, device=train_features.device)
    prototypes.index_add_(0, train_labels, train_features)
    prototypes = prototypes / torch.bincount(train_labels, minlength=num_classes).unsqueeze(1)

    distances = test_feature_norms.unsqueeze(1) + prototypes.square().sum(dim=1).unsqueeze(0)
    distances = distances - 2 * test_features @ prototypes.T
    pred_class = distances.argmin(dim=1).cpu()

    raw_data = []
    test_labels_cpu = test_labels.cpu()
    for i in range(len(test_labels_cpu)):
        raw_data.append({
            "index": i,
            "file_name": test_file_names[i],
            "total_pixels": int(total_pixels[i].item()),
            "pixel_in": int(pixel_in[i].item()),
            "pixel_out": int(pixel_out[i].item()),
            "gt_class": int(test_labels_cpu[i].item()),
            "pred_class": int(pred_class[i].item()),
        })

    return raw_data, {
        "gt_class": test_labels_cpu,
        "pred_class": pred_class,
        "total_pixels": total_pixels,
        "pixel_in": pixel_in,
        "pixel_out": pixel_out,
    }


def run_sweep(min_samples=1, max_samples=None, seeds=[], experiment_name="", masks={}, save_csv=True):
    all_results = []

    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)

    train_labels, test_labels = remap_labels_fast(train_data["labels"], test_data["labels"])
    train_data["labels"] = train_labels
    test_data["labels"] = test_labels

    num_classes = train_labels.max().item() + 1
    train_class_indices = build_class_indices(train_labels, num_classes)
    test_class_indices = build_class_indices(test_labels, num_classes)
    test_file_names_all, total_pixels_all, pixel_in_all, pixel_out_all = build_mask_cache(test_data["file_paths"], masks)

    train_features_all = train_data["cls_tokens"].to(DEVICE)
    train_labels_all = train_labels.to(DEVICE)
    test_features_all = test_data["cls_tokens"].to(DEVICE)
    test_feature_norms_all = test_features_all.square().sum(dim=1)

    for seed in tqdm(seeds, desc="Seeds", position=0):
        seed_everything(seed)
        seed_results = []
        csv_path_prefix = f"./results/{experiment_name}/{seed}"
        os.makedirs("/".join(csv_path_prefix.split("/")[:-1]), exist_ok=True)

        computed_file = None
        raw_file = None
        computed_writer = None
        raw_writer = None

        if save_csv:
            computed_file = open(f"{csv_path_prefix}_computed.csv", "w", newline="")
            raw_file = open(f"{csv_path_prefix}_raw.csv", "w", newline="")

        for samples_per_class in tqdm(
            range(min_samples, max_samples + 1),
            desc=f"Samples seed={seed}",
            position=1,
            leave=False,
        ):
            train_indices = sample_balanced_indices(train_class_indices, seed=seed, samples_per_class=samples_per_class)
            test_indices = sample_balanced_indices(test_class_indices, seed=seed, samples_per_class=samples_per_class)
            train_indices_device = train_indices.to(DEVICE)
            test_indices_device = test_indices.to(DEVICE)

            raw_data, metrics_data = prototype_method(
                train_features=train_features_all[train_indices_device],
                train_labels=train_labels_all[train_indices_device],
                test_features=test_features_all[test_indices_device],
                test_feature_norms=test_feature_norms_all[test_indices_device],
                test_labels=test_labels[test_indices],
                test_file_names=[test_file_names_all[i] for i in test_indices.tolist()],
                total_pixels=total_pixels_all[test_indices],
                pixel_in=pixel_in_all[test_indices],
                pixel_out=pixel_out_all[test_indices],
                num_classes=num_classes,
            )
            metrics = compute_metrics_final_fast(metrics_data, num_classes=num_classes)

            result_computed = {
                "samples_per_class": samples_per_class,
                "accuracy_euclidean": metrics["macro_img_acc"],
                "accuracy_euclidean_overall": metrics["overall_img_acc"],
                "mIoU": metrics["mIoU"],
                
            }
            raw_rows = [{"sample_per_class": samples_per_class, **data} for data in raw_data]

            seed_results.append(result_computed)
            all_results.append({"seed": seed, **result_computed})

            if save_csv:
                if computed_writer is None:
                    computed_writer = csv.DictWriter(computed_file, fieldnames=result_computed.keys())
                    computed_writer.writeheader()
                computed_writer.writerow(result_computed)

                if raw_writer is None:
                    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_rows[0].keys())
                    raw_writer.writeheader()
                raw_writer.writerows(raw_rows)

        if save_csv:
            computed_file.close()
            raw_file.close()

        # plot_sweep(seed_results, save_path=f"{csv_path_prefix}_plot_accuracy.png", metric="accuracy_euclidean", save_only=True)
        # plot_sweep(seed_results, save_path=f"{csv_path_prefix}_plot_miou.png", metric="mIoU", save_only=True)

    return all_results


def plot_sweep(results, save_path="sweep_samples_per_class_plot.png", metric="accuracy_euclidean", save_only=False):
    x = [r["samples_per_class"] for r in results]
    plt.figure(figsize=(10, 7))
    plt.plot(x, [r[metric] for r in results], label="Euclidean")
    plt.xlabel("Samples per Class")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Samples per Class")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}", dpi=300)
    if not save_only:
        plt.show()
    plt.close()


if __name__ == "__main__":

    max_samples = 200
    num_seeds = [7, 42, 123, 2024, 9999][:1]
    # np.random.seed(42)
    # num_seeds = list(map(int, np.random.randint(0, 10000, size=20)))
    experiment_name = "prototype"

    masks = load_masks(Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/test/720/FungiTastic/test/720p"))
    results = run_sweep(1, max_samples, seeds=num_seeds, experiment_name=experiment_name, masks=masks, save_csv=True)

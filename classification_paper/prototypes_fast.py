import csv
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

SEED = 7
BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", "224")
DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
MODEL_TRAIN = ROOT / f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train"
MODEL_TEST = ROOT / f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test"
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "sweep_samples_per_class_results.csv"
PLOT_PATH = OUTPUT_DIR / "sweep_samples_per_class_plot.png"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
CONSOLE = Console()

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cls_data(data_path: Path) -> dict[str, torch.Tensor]:
    labels: list[int] = []
    cls_tokens: list[torch.Tensor] = []
    for shard_path in tqdm(sorted(data_path.glob("*.pt")), desc=f"Loading {data_path.name}", unit="shard"):
        shard_data = torch.load(shard_path, map_location="cpu")
        labels.extend(shard_data["labels"])
        cls_tokens.append(shard_data["cls_token"].float())
    return {"labels": torch.tensor(labels, dtype=torch.long), "cls_tokens": torch.cat(cls_tokens, dim=0)}


def get_max_samples_per_class(labels: torch.Tensor) -> int:
    return int(torch.bincount(labels).max().item())


def split_by_class(features: torch.Tensor, labels: torch.Tensor, class_labels: torch.Tensor, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    grouped: list[torch.Tensor] = []
    for label in class_labels.tolist():
        indices = torch.where(labels == label)[0]
        if indices.numel() == 0:
            grouped.append(torch.empty((0, features.size(1)), dtype=features.dtype, device=DEVICE))
            continue
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        grouped.append(features[shuffled].to(DEVICE))
    return grouped


def squared_distances(x: torch.Tensor, y: torch.Tensor, x_norms: torch.Tensor | None = None) -> torch.Tensor:
    if x_norms is None:
        x_norms = x.square().sum(dim=1)
    y_norms = y.square().sum(dim=1)
    return x_norms[:, None] - 2 * (x @ y.T) + y_norms[None, :]


@torch.inference_mode()
def prototype_method(
    train_data: dict[str, torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[float, float]:
    train_labels = train_data["labels"]
    test_labels = test_data["labels"]
    class_labels = torch.unique(train_labels, sorted=True)
    assert torch.isin(test_labels, class_labels).all().item()
    train_targets = torch.searchsorted(class_labels, train_labels).to(DEVICE)
    test_targets = torch.searchsorted(class_labels, test_labels).to(DEVICE)
    train_features = train_data["cls_tokens"].to(DEVICE)
    test_features = test_data["cls_tokens"].to(DEVICE)
    sums = torch.zeros((class_labels.numel(), train_features.size(1)), dtype=train_features.dtype, device=DEVICE)
    sums.index_add_(0, train_targets, train_features)
    counts = torch.bincount(train_targets, minlength=class_labels.numel()).to(train_features.dtype)
    prototypes = sums / counts[:, None]
    predictions = squared_distances(test_features, prototypes).argmin(dim=1)
    correct = predictions == test_targets
    per_class_total = torch.bincount(test_targets, minlength=class_labels.numel())
    per_class_correct = torch.bincount(test_targets[correct], minlength=class_labels.numel())
    overall_acc = correct.float().mean().item()
    macc = (per_class_correct[per_class_total > 0].float() / per_class_total[per_class_total > 0].float()).mean().item()
    return overall_acc, macc


def build_test_eval_order(test_by_class: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_count = max(features.size(0) for features in test_by_class)
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    prefix_lengths: list[int] = []
    total = 0
    for rank in range(max_count):
        rank_features = [features[rank : rank + 1] for features in test_by_class if rank < features.size(0)]
        rank_targets = [class_id for class_id, features in enumerate(test_by_class) if rank < features.size(0)]
        if rank_features:
            feature_chunks.append(torch.cat(rank_features, dim=0))
            target_chunks.append(torch.tensor(rank_targets, dtype=torch.long, device=DEVICE))
            total += len(rank_targets)
        prefix_lengths.append(total)
    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
        torch.tensor(prefix_lengths, dtype=torch.long),
    )


def confirm_override(path: Path) -> None:
    if path.exists():
        assert input(f"{path} exists. Override? [y/N] ").strip().lower() == "y"


def save_results(results: list[dict[str, float | int]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    confirm_override(csv_path)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def plot_sweep(results: list[dict[str, float | int]], plot_path: Path, save_only: bool = False) -> None:
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    confirm_override(plot_path)
    x = [int(row["samples_per_class"]) for row in results]
    y = [float(row["accuracy_euclidean"]) for row in results]
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, label="Euclidean")
    plt.xlabel("Samples per Class")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Samples per Class")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    if not save_only:
        plt.show()
    plt.close()


def print_summary(results: list[dict[str, float | int]]) -> None:
    best = max(results, key=lambda row: float(row["accuracy_euclidean"]))
    table = Table(title="Prototype Sweep")
    table.add_column("best_k", justify="right")
    table.add_column("best_mAcc", justify="right")
    table.add_row(str(best["samples_per_class"]), f"{float(best['accuracy_euclidean']):.4f}")
    CONSOLE.print(table)


@torch.inference_mode()
def run_sweep(
    min_samples: int = 1,
    max_samples: int | None = None,
    save_csv: bool = True,
    save_plot: bool = HAS_MATPLOTLIB,
) -> list[dict[str, float | int]]:
    seed_everything(SEED)
    train_data = load_cls_data(MODEL_TRAIN)
    test_data = load_cls_data(MODEL_TEST)
    class_labels = torch.unique(train_data["labels"], sorted=True)
    assert torch.isin(test_data["labels"], class_labels).all().item()
    max_train_samples = get_max_samples_per_class(train_data["labels"])
    max_samples = max_train_samples if max_samples is None else min(max_samples, max_train_samples)
    assert 1 <= min_samples <= max_samples

    train_by_class = split_by_class(train_data["cls_tokens"], train_data["labels"], class_labels, SEED)
    test_by_class = split_by_class(test_data["cls_tokens"], test_data["labels"], class_labels, SEED)
    del train_data, test_data

    test_features, test_targets, prefix_lengths = build_test_eval_order(test_by_class)
    test_norms = test_features.square().sum(dim=1)
    prototype_sums = torch.zeros((len(train_by_class), train_by_class[0].size(1)), dtype=train_by_class[0].dtype, device=DEVICE)
    prototype_counts = torch.zeros(len(train_by_class), dtype=train_by_class[0].dtype, device=DEVICE)
    max_test_rank = prefix_lengths.numel()
    results: list[dict[str, float | int]] = []

    for samples_per_class in tqdm(range(1, max_samples + 1), desc="Sweep", unit="k"):
        update_ids = [class_id for class_id, features in enumerate(train_by_class) if samples_per_class <= features.size(0)]
        if update_ids:
            new_samples = torch.stack([train_by_class[class_id][samples_per_class - 1] for class_id in update_ids], dim=0)
            update_ids_tensor = torch.tensor(update_ids, dtype=torch.long, device=DEVICE)
            prototype_sums[update_ids_tensor] += new_samples
            prototype_counts[update_ids_tensor] += 1
        if samples_per_class < min_samples:
            continue
        prototypes = prototype_sums / prototype_counts[:, None]
        test_limit = int(prefix_lengths[min(samples_per_class, max_test_rank) - 1].item())
        predictions = squared_distances(test_features[:test_limit], prototypes, test_norms[:test_limit]).argmin(dim=1)
        correct = predictions == test_targets[:test_limit]
        per_class_total = torch.bincount(test_targets[:test_limit], minlength=prototypes.size(0))
        per_class_correct = torch.bincount(test_targets[:test_limit][correct], minlength=prototypes.size(0))
        macc = (per_class_correct[per_class_total > 0].float() / per_class_total[per_class_total > 0].float()).mean().item()
        results.append({"samples_per_class": samples_per_class, "accuracy_euclidean": macc})

    if save_csv:
        save_results(results, CSV_PATH)
    if save_plot:
        plot_sweep(results, PLOT_PATH, save_only=True)
    print_summary(results)
    return results


if __name__ == "__main__":
    run_sweep(save_plot=HAS_MATPLOTLIB)

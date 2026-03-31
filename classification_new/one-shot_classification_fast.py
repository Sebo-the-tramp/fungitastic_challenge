import os

from pathlib import Path
from time import perf_counter

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MPLCONFIGDIR = RESULTS_DIR / ".matplotlib"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import torch

from tqdm import tqdm

from utils import filter_data, load_shards, seed_everything

SEED = 7
MAX_SAMPLES = 500
SEED_ITERATIONS = 20
BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", "224")
FEATURE_KEY = "cls_tokens"
PLOT_PATH = RESULTS_DIR / "all_metrics_variance_plot.png"
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")

TensorDict = dict[str, torch.Tensor]
ResultDict = dict[int, dict[int, dict[str, float]]]


def prepare_data() -> tuple[TensorDict, TensorDict, list[torch.Tensor], list[torch.Tensor]]:
    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)

    train_labels = train_data["labels"]
    test_labels = test_data["labels"]
    train_only_labels = torch.tensor(
        sorted(set(train_labels.unique().tolist()) - set(test_labels.unique().tolist())),
        dtype=train_labels.dtype,
    )
    train_data = filter_data(train_data, train_only_labels)
    train_labels = train_data["labels"]
    shared_labels = torch.unique(train_labels, sorted=True)
    assert torch.isin(test_labels, shared_labels).all().item()

    train_data = {**train_data, "labels": torch.searchsorted(shared_labels, train_labels)}
    test_data = {**test_data, "labels": torch.searchsorted(shared_labels, test_labels)}

    print(f"Classes removed from train because they are absent in test: {len(train_only_labels)}")
    return train_data, test_data, build_class_indices(train_data["labels"]), build_class_indices(test_data["labels"])


def build_class_indices(labels: torch.Tensor) -> list[torch.Tensor]:
    counts = torch.bincount(labels)
    return list(torch.argsort(labels).split(counts.tolist()))


def build_index_cache(class_indices: list[torch.Tensor]) -> dict[int, dict[int, torch.Tensor]]:
    cache = {samples: {} for samples in range(1, MAX_SAMPLES + 1)}

    for seed_offset in range(SEED_ITERATIONS):
        seed = SEED + seed_offset
        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = [indices[torch.randperm(indices.numel(), generator=generator)] for indices in class_indices]

        for samples in range(1, MAX_SAMPLES + 1):
            cache[samples][seed] = torch.cat([indices[:samples] for indices in shuffled_indices])

    return cache


def macro_scores(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> dict[str, float]:
    confusion = torch.bincount(
        y_true * num_classes + y_pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).float()
    true_positives = confusion.diag()
    predicted_positives = confusion.sum(dim=0)
    actual_positives = confusion.sum(dim=1)
    zeros = torch.zeros_like(true_positives)
    precision = torch.where(predicted_positives > 0, true_positives / predicted_positives, zeros)
    recall = torch.where(actual_positives > 0, true_positives / actual_positives, zeros)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), zeros)

    return {
        "accuracy": float((y_pred == y_true).float().mean().item()),
        "balanced_accuracy": float(recall.mean().item()),
        "precision": float(precision.mean().item()),
        "f1_score": float(f1.mean().item()),
    }


@torch.inference_mode()
def prototype_method(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    prototypes = torch.zeros((num_classes, train_features.size(1)), dtype=train_features.dtype)
    prototypes.scatter_add_(0, train_labels.unsqueeze(1).expand_as(train_features), train_features)
    counts = torch.bincount(train_labels, minlength=num_classes).clamp_min_(1).to(train_features.dtype).unsqueeze(1)
    prototypes /= counts

    train_norms = (prototypes * prototypes).sum(dim=1)
    test_norms = (test_features * test_features).sum(dim=1, keepdim=True)
    distances = test_norms + train_norms.unsqueeze(0) - 2 * (test_features @ prototypes.T)
    predicted_labels = distances.argmin(dim=1)
    return macro_scores(test_labels, predicted_labels, num_classes)


def graph_results_full(results: ResultDict) -> None:
    import matplotlib.pyplot as plt

    samples = sorted(results)
    metrics = ("accuracy", "balanced_accuracy", "precision", "f1_score")
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision": "Precision",
        "f1_score": "F1 Score",
    }
    colors = {
        "accuracy": "blue",
        "balanced_accuracy": "red",
        "precision": "green",
        "f1_score": "orange",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for metric in metrics:
        values = np.array(
            [[results[samples_per_class][seed][metric] for seed in sorted(results[samples_per_class])] for samples_per_class in samples],
            dtype=np.float32,
        )
        means = values.mean(axis=1)
        stds = values.std(axis=1)
        ax.plot(samples, means, label=labels[metric], color=colors[metric])
        ax.fill_between(samples, means - stds, means + stds, color=colors[metric], alpha=0.1)

    ax.set_title("Performance Metrics vs. Samples Per Class")
    ax.set_xlabel("Samples per Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    seed_everything(SEED)

    prepare_start = perf_counter()
    train_data, test_data, train_class_indices, test_class_indices = prepare_data()
    prepare_time = perf_counter() - prepare_start

    cache_start = perf_counter()
    train_index_cache = build_index_cache(train_class_indices)
    test_index_cache = build_index_cache(test_class_indices)
    cache_time = perf_counter() - cache_start

    train_features = train_data[FEATURE_KEY]
    train_labels = train_data["labels"]
    test_features = test_data[FEATURE_KEY]
    test_labels = test_data["labels"]
    num_classes = int(train_labels.max().item()) + 1
    results: ResultDict = {}

    eval_start = perf_counter()
    for samples_per_class in tqdm(range(1, MAX_SAMPLES + 1), desc="Samples per class"):
        results[samples_per_class] = {}

        for seed_offset in tqdm(range(SEED_ITERATIONS), desc="Seeds", leave=False):
            seed = SEED + seed_offset
            train_indices = train_index_cache[samples_per_class][seed]
            test_indices = test_index_cache[samples_per_class][seed]
            results[samples_per_class][seed] = prototype_method(
                train_features[train_indices],
                train_labels[train_indices],
                test_features[test_indices],
                test_labels[test_indices],
                num_classes,
            )

    eval_time = perf_counter() - eval_start

    plot_start = perf_counter()
    graph_results_full(results)
    plot_time = perf_counter() - plot_start
    total_time = perf_counter() - prepare_start

    print(f"prepare_time={prepare_time:.2f}s")
    print(f"cache_time={cache_time:.2f}s")
    print(f"eval_time={eval_time:.2f}s")
    print(f"plot_time={plot_time:.2f}s")
    print(f"total_time={total_time:.2f}s")
    print(f"plot_path={PLOT_PATH}")


if __name__ == "__main__":
    main()

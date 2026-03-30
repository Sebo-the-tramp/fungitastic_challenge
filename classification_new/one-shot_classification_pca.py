import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from utils import balance_data, load_shards, remap_labels, seed_everything

SEED = 7
PCA_DIM = 128
BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 224)
FEATURE_KEY = "cls_tokens"
MAX_SAMPLES = 100
SEED_ITERATIONS = 1
METRICS = ("accuracy", "precision", "recall", "f1_score")
PLOT_METRICS = ("accuracy",)
MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"
PARTIAL_RESULTS_PATH = CLASSIFICATION_RESULTS_DIR / "one_shot_classification_pca_partial.csv"
PLOT_PATH = CLASSIFICATION_RESULTS_DIR / "one_shot_classification_pca_online.png"

TensorDict = dict[str, torch.Tensor | None]
ResultDict = dict[int, dict[int, dict[str, float]]]


def prepare_outputs() -> None:
    CLASSIFICATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in (PARTIAL_RESULTS_PATH, PLOT_PATH):
        if path.exists():
            answer = input(f"{path} exists. Overwrite? [y/N] ").strip().lower()
            assert answer == "y"
            path.unlink()

    with PARTIAL_RESULTS_PATH.open("w", newline="") as file:
        csv.writer(file).writerow(["samples_per_class", "seed", *METRICS])


def prepare_data() -> tuple[TensorDict, TensorDict]:
    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)
    return (
        {**train_data, "labels": train_data["labels"]},
        {**test_data, "labels": test_data["labels"]},
    )


def balance_full_dataset(
    train_data: TensorDict,
    test_data: TensorDict,
    seed: int = SEED,
    samples_per_class: int = 5,
) -> tuple[TensorDict, TensorDict]:
    train_data_balanced = balance_data(train_data, seed=seed, samples_per_class=samples_per_class)
    test_data_balanced = balance_data(test_data, seed=seed, samples_per_class=samples_per_class)
    train_labels, test_labels = remap_labels(train_data_balanced["labels"], test_data_balanced["labels"])
    return (
        {**train_data_balanced, "labels": train_labels},
        {**test_data_balanced, "labels": test_labels},
    )


@torch.inference_mode()
def prototype_method_pca(
    train_data: TensorDict,
    test_data: TensorDict,
) -> dict[str, float]:
    train_labels = train_data["labels"]
    test_labels = test_data["labels"]
    train_features = train_data[FEATURE_KEY]
    test_features = test_data[FEATURE_KEY]
    assert PCA_DIM <= min(train_features.shape)
    pca = PCA(n_components=PCA_DIM, whiten=True, svd_solver="full")
    train_features = torch.from_numpy(pca.fit_transform(train_features.cpu().float().numpy())).to(dtype=torch.float32, device=train_features.device)
    test_features = torch.from_numpy(pca.transform(test_features.cpu().float().numpy())).to(dtype=torch.float32, device=test_features.device)
    num_classes = train_labels.max().item() + 1
    out = torch.zeros(num_classes, train_features.size(1), dtype=train_features.dtype, device=train_features.device)
    index = train_labels.unsqueeze(1).expand_as(train_features)
    out.scatter_reduce_(dim=0, index=index, src=train_features, reduce="mean", include_self=False)
    euclidean_distances = torch.cdist(test_features, out)
    predicted_labels = euclidean_distances.argmin(dim=1)
    accuracy = (predicted_labels == test_labels).float().mean().item()
    y_true = test_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def append_partial_result(samples_per_class: int, seed: int, result: dict[str, float]) -> None:
    with PARTIAL_RESULTS_PATH.open("a", newline="") as file:
        csv.writer(file).writerow([samples_per_class, seed, *[result[metric] for metric in METRICS]])


def load_partial_results() -> ResultDict:
    results: ResultDict = {}

    with PARTIAL_RESULTS_PATH.open() as file:
        for row in csv.DictReader(file):
            samples_per_class = int(row["samples_per_class"])
            seed = int(row["seed"])
            results.setdefault(samples_per_class, {})[seed] = {metric: float(row[metric]) for metric in METRICS}

    return results


def plot_partial_results() -> None:
    results = load_partial_results()
    x = sorted(results)
    means = {metric: [] for metric in PLOT_METRICS}
    stds = {metric: [] for metric in PLOT_METRICS}
    colors = {"accuracy": "blue", "precision": "green", "recall": "red", "f1_score": "purple"}
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1 Score"}

    for samples_per_class in x:
        for metric in PLOT_METRICS:
            values = [results[samples_per_class][seed][metric] for seed in results[samples_per_class]]
            means[metric].append(np.mean(values))
            stds[metric].append(np.std(values))

    plt.figure(figsize=(10, 7))

    for metric in PLOT_METRICS:
        metric_means = np.array(means[metric])
        metric_stds = np.array(stds[metric])
        plt.plot(x, metric_means, label=labels[metric], color=colors[metric])
        plt.fill_between(x, metric_means - metric_stds, metric_means + metric_stds, color=colors[metric], alpha=0.1)

    plt.title("Performance Metrics vs. Samples per Class")
    plt.xlabel("Samples per Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    seed_everything(SEED)
    prepare_outputs()
    train_data, test_data = prepare_data()

    for samples_per_class in tqdm(range(1, MAX_SAMPLES + 1), desc="Samples per class"):
        for seed_offset in tqdm(range(SEED_ITERATIONS), desc="Seeds", leave=False):
            random_seed = SEED + seed_offset
            train_data_balanced, test_data_balanced = balance_full_dataset(train_data, test_data, seed=random_seed, samples_per_class=samples_per_class)
            run_result = prototype_method_pca(train_data_balanced, test_data_balanced)
            append_partial_result(samples_per_class, random_seed, run_result)
            plot_partial_results()

    print(PARTIAL_RESULTS_PATH)
    print(PLOT_PATH)


if __name__ == "__main__":
    main()

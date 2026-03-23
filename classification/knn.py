import torch
import torch.nn.functional as F

from pathlib import Path
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from utils import load_shards, remap_labels, save_run, seed_everything

SEED = 7
SCRIPT_PATH = "classification/knn.py"
BACKBONE = "dinov3-vit7b16-pretrain-lvd1689m"
IMAGE_SIZE = 224
BACKGROUND_AUG = "normal"
FINAL_LAYER_CLASSIFIER_METHOD = "knn"
EXPERIMENT_ID = f"{BACKBONE}_{BACKGROUND_AUG}_{IMAGE_SIZE}_{FINAL_LAYER_CLASSIFIER_METHOD}"
K_MIN = 1
K_MAX = 10
TEST_CHUNK_SIZE = 256
TRAIN_CHUNK_SIZE = 4096
MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_{BACKGROUND_AUG}_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_{BACKGROUND_AUG}_{IMAGE_SIZE}/test")
MODEL_VAL = Path(f"facebook/{BACKBONE}/bfloat16_{BACKGROUND_AUG}_{IMAGE_SIZE}/val")
EXPERIMENTS: list[tuple[str, tuple[str, ...]]] = [
    ("cls", ("cls_tokens",)),
    ("patch", ("mean_pooled_patch_tokens",)),
    ("masked", ("mean_pooled_masked_patch_tokens",)),
    ("cls+patch", ("cls_tokens", "mean_pooled_patch_tokens")),
    ("cls+masked", ("cls_tokens", "mean_pooled_masked_patch_tokens")),
]

ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CONSOLE = Console()


def knn_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    assert train_features.shape[0] >= K_MAX
    num_classes = int(train_labels.max().item()) + 1
    predictions = []

    for test_start in tqdm(range(0, len(test_features), TEST_CHUNK_SIZE), desc="KNN", unit="chunk"):
        test_chunk = F.normalize(test_features[test_start:test_start + TEST_CHUNK_SIZE].to(device), p=2, dim=1)
        best_scores = torch.full((test_chunk.shape[0], K_MAX), -torch.inf, device=device)
        best_labels = torch.zeros((test_chunk.shape[0], K_MAX), dtype=train_labels.dtype, device=device)

        for train_start in range(0, len(train_features), TRAIN_CHUNK_SIZE):
            train_chunk = F.normalize(train_features[train_start:train_start + TRAIN_CHUNK_SIZE].to(device), p=2, dim=1)
            label_chunk = train_labels[train_start:train_start + TRAIN_CHUNK_SIZE].to(device)
            chunk_scores = test_chunk @ train_chunk.T
            chunk_scores, chunk_indices = chunk_scores.topk(min(K_MAX, train_chunk.shape[0]), dim=1)
            chunk_labels = label_chunk[chunk_indices]
            all_scores = torch.cat([best_scores, chunk_scores], dim=1)
            all_labels = torch.cat([best_labels, chunk_labels], dim=1)
            best_scores, best_indices = all_scores.topk(K_MAX, dim=1)
            best_labels = all_labels.gather(1, best_indices)

        votes = F.one_hot(best_labels, num_classes=num_classes).cumsum(dim=1)
        predictions.append(votes.argmax(dim=2).cpu())

    return torch.cat(predictions)


def sweep_knn(
    X_train_image_features: torch.Tensor,
    y_train: torch.Tensor,
    X_test_image_features: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    y_train_mapped, y_test_mapped = remap_labels(y_train, y_test)
    y_pred = knn_predictions(X_train_image_features, y_train_mapped, X_test_image_features, device)
    return (y_pred == y_test_mapped.unsqueeze(1)).float().mean(dim=0)


def build_features(data: dict[str, torch.Tensor], feature_names: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([data[name].float() for name in feature_names], dim=1)


def run_experiment(
    name: str,
    feature_names: tuple[str, ...],
    full_train_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    full_train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> dict[str, str | list[float]]:
    X_full_train = build_features(full_train_features, feature_names)
    X_test = build_features(test_features, feature_names)
    test_accuracies = sweep_knn(X_full_train, full_train_labels, X_test, test_labels, device)

    return {
        "name": name,
        "features": "+".join(feature_names),
        "test_curve": test_accuracies.tolist(),
    }


def print_results(results: list[dict[str, str | list[float]]]) -> None:
    table = Table(title=f"Cosine k-NN Results ({K_MIN}-{K_MAX})")
    table.add_column("Experiment")
    table.add_column("k", justify="right")
    table.add_column("Test", justify="right")

    for result in results:
        for k, test_acc in enumerate(result["test_curve"], start=K_MIN):
            table.add_row(str(result["name"]), str(k), f"{float(test_acc):.4f}")

    CONSOLE.print(table)


def save_results(
    results: list[dict[str, str | list[float]]],
    run_meta: dict[str, int | float | str],
) -> Path:
    rows = []
    for result in results:
        for k, test_acc in enumerate(result["test_curve"], start=K_MIN):
            rows.append(
                {
                    "item_id": f"{result['name']}@k={k}",
                    "status": "done",
                    "axes": {
                        "modality": str(result["name"]),
                        "k": k,
                    },
                    "meta": {
                        "features": str(result["features"]),
                    },
                    "metrics": {
                        "test_acc": float(test_acc),
                    },
                }
            )

    return save_run(
        experiment_id=EXPERIMENT_ID,
        script_path=SCRIPT_PATH,
        axes={
            "backbone": BACKBONE,
            "image_size": IMAGE_SIZE,
            "background_aug": BACKGROUND_AUG,
        },
        rows=rows,
        meta=run_meta,
    )


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = ROOT / MODEL_TRAIN
    val_path = ROOT / MODEL_VAL
    test_path = ROOT / MODEL_TEST

    train_data = load_shards(train_path)
    val_data = load_shards(val_path)
    test_data = load_shards(test_path)

    full_train_features = {
        "cls_tokens": torch.cat([train_data["cls_tokens"], val_data["cls_tokens"]]).float(),
        "mean_pooled_patch_tokens": torch.cat([
            train_data["mean_pooled_patch_tokens"],
            val_data["mean_pooled_patch_tokens"],
        ]).float(),
        "mean_pooled_masked_patch_tokens": torch.cat([
            train_data["mean_pooled_masked_patch_tokens"],
            val_data["mean_pooled_masked_patch_tokens"],
        ]).float(),
    }
    test_features = {
        "cls_tokens": test_data["cls_tokens"].float(),
        "mean_pooled_patch_tokens": test_data["mean_pooled_patch_tokens"].float(),
        "mean_pooled_masked_patch_tokens": test_data["mean_pooled_masked_patch_tokens"].float(),
    }

    train_labels = train_data["labels"]
    val_labels = val_data["labels"]
    full_train_labels = torch.cat([train_labels, val_labels])
    test_labels = test_data["labels"]
    run_meta = {
        "seed": SEED,
        "device": str(device),
        "k_min": K_MIN,
        "k_max": K_MAX,
        "test_chunk_size": TEST_CHUNK_SIZE,
        "train_chunk_size": TRAIN_CHUNK_SIZE,
        "num_train_samples": int(len(train_labels)),
        "num_val_samples": int(len(val_labels)),
        "num_full_train_samples": int(len(full_train_labels)),
        "num_test_samples": int(len(test_labels)),
        "model_train": str(MODEL_TRAIN),
        "model_val": str(MODEL_VAL),
        "model_test": str(MODEL_TEST),
    }

    results = []
    for name, feature_names in EXPERIMENTS:
        results.append(
            run_experiment(
                name=name,
                feature_names=feature_names,
                full_train_features=full_train_features,
                test_features=test_features,
                full_train_labels=full_train_labels,
                test_labels=test_labels,
                device=device,
            )
        )

    print_results(results)
    run_path = save_results(results, run_meta)
    CONSOLE.print(f"Saved run to {run_path}")


if __name__ == "__main__":
    main()

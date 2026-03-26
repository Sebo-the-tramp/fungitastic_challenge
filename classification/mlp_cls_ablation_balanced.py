import os
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from rich.table import Table
from rich.console import Console
from typing import Any

from utils import load_shards, save_run, seed_everything, balance_data

 
SEED = 7
SCRIPT_PATH = "classification/mlp_cls_ablation_balanced.py"
BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 224)
BACKGROUND_AUG = "normal"
FINAL_LAYER_CLASSIFIER_METHOD = "mlp_balanced"
EXPERIMENT_ID = f"{BACKBONE}_{BACKGROUND_AUG}_{IMAGE_SIZE}_{FINAL_LAYER_CLASSIFIER_METHOD}"

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_VAL = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/val")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 30
MIN_DELTA = 1e-4
VAL_FRACTION = 0.15
PROJECTION_DIM = 256
HIDDEN_DIM = 512
NUM_REGISTER_TOKENS = 4
MAX_CONFUSION_MATRIX_CLASSES_TO_PRINT = 20
CONFUSION_MATRIX_IMAGE_SUFFIX = ".png"
CONFUSION_MATRIX_IMAGE_DPI = 220
CONFUSION_MATRIX_RESULTS_DIR = CLASSIFICATION_RESULTS_DIR / EXPERIMENT_ID

REGISTER_FEATURE_NAMES = tuple(f"register_tokens_{index}" for index in range(NUM_REGISTER_TOKENS))
BASE_EXPERIMENTS: list[tuple[str, tuple[str, ...]]] = [
    ("cls", ("cls_tokens",)),
    # ("patch", ("mean_pooled_patch_tokens",)),
    # ("gt_masked", ("mean_pooled_gt_masked_patch_tokens",)),
    # ("sam_masked", ("mean_pooled_sam_masked_patch_tokens",)),
    # ("cls+patch", ("cls_tokens", "mean_pooled_patch_tokens")),
    # ("cls+gt_masked", ("cls_tokens", "mean_pooled_gt_masked_patch_tokens")),
    # ("cls+sam_masked", ("cls_tokens", "mean_pooled_sam_masked_patch_tokens")),
]
REGISTER_EXPERIMENTS: list[tuple[str, tuple[str, ...]]] = [
#     (f"cls+register_{index}", ("cls_tokens", feature_name))
#     for index, feature_name in enumerate(REGISTER_FEATURE_NAMES)
# ] + [
#     (f"register_{index}", (feature_name))
#     for index, feature_name in enumerate(REGISTER_FEATURE_NAMES)
# ] + [
    # ("cls+gt_masked+register_3", ("cls_tokens", "mean_pooled_gt_masked_patch_tokens", "register_tokens_3")),
    # ("cls+sam_masked+register_3", ("cls_tokens", "mean_pooled_sam_masked_patch_tokens", "register_tokens_3")),
]

CONSOLE = Console()


class BranchMLP(torch.nn.Module):
    def __init__(self, input_dims: list[int], projection_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, projection_dim),
                torch.nn.ReLU(),
            )
            for input_dim in input_dims
        ])
        self.fusion = torch.nn.Linear(len(input_dims) * projection_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        encoded = [projection(branch) for projection, branch in zip(self.projections, branches)]
        fused = torch.cat(encoded, dim=1)
        hidden = torch.relu(self.fusion(fused))
        return self.output(hidden)


def remap_labels_from_reference(
    y_reference: torch.Tensor,
    *y_target_sets: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    reference_labels = torch.unique(y_reference, sorted=True)
    label_map = {int(label): idx for idx, label in enumerate(reference_labels.tolist())}

    for y_target in y_target_sets:
        assert torch.isin(y_target, reference_labels).all().item()

    mapped = []
    for labels in (y_reference, *y_target_sets):
        mapped.append(
            torch.tensor([label_map[int(label)] for label in labels.tolist()], dtype=torch.long)
        )

    return tuple(mapped)


def make_stratified_split_indices(y: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    train_indices = []
    val_indices = []

    for label in torch.unique(y, sorted=True):
        label_indices = torch.nonzero(y == label, as_tuple=False).squeeze(1)
        label_indices = label_indices[torch.randperm(len(label_indices))]

        if len(label_indices) == 1:
            train_indices.append(label_indices)
            continue

        num_val = max(1, int(round(len(label_indices) * val_fraction)))
        num_val = min(num_val, len(label_indices) - 1)
        val_indices.append(label_indices[:num_val])
        train_indices.append(label_indices[num_val:])

    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices)
    train_indices = train_indices[torch.randperm(len(train_indices))]
    val_indices = val_indices[torch.randperm(len(val_indices))]
    return train_indices, val_indices


def subset_data(data: dict[str, torch.Tensor | None], indices: torch.Tensor) -> dict[str, torch.Tensor | None]:
    return {key: (value[indices] if value is not None else None) for key, value in data.items()}


def prepare_branches(
    features: dict[str, torch.Tensor],
    names: tuple[str, ...],
    indices: torch.Tensor | None,
    device: torch.device,
) -> list[torch.Tensor]:
    branches = []
    # print(f"Preparing branches: {names} with indices: {indices.shape if indices is not None else None} on device: {device}")
    for name in names:
        branch = features[name]
        if indices is not None:
            branch = branch[indices]
        branches.append(branch.to(device))
    return branches


def evaluate_split(
    model: BranchMLP,
    branches: list[torch.Tensor],
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(branches)
        loss = criterion(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def evaluate_split_metrics(
    model: BranchMLP,
    branches: list[torch.Tensor],
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(branches)
        loss = criterion(logits, labels).item()

    labels_cpu = labels.detach().cpu()
    predictions = logits.argmax(dim=1).detach().cpu()
    num_classes = logits.shape[1]
    confusion_matrix = torch.bincount(
        labels_cpu * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = confusion_matrix.diag().float()
    predicted_support = confusion_matrix.sum(dim=0).float()
    true_support = confusion_matrix.sum(dim=1).float()
    precision_per_class = true_positive / predicted_support.clamp_min(1.0)
    recall_per_class = true_positive / true_support.clamp_min(1.0)
    valid_classes = (predicted_support + true_support) > 0

    return {
        "loss": loss,
        "accuracy": (predictions == labels_cpu).float().mean().item(),
        "precision": precision_per_class[valid_classes].mean().item(),
        "recall": recall_per_class[valid_classes].mean().item(),
        "confusion_matrix": confusion_matrix.tolist(),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "support_per_class": true_support.to(torch.int64).tolist(),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_experiment(
    name: str,
    feature_names: tuple[str, ...],
    train_features: dict[str, torch.Tensor],
    val_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(SEED)

    X_train = prepare_branches(train_features, feature_names, None, device)
    X_val = prepare_branches(val_features, feature_names, None, device)
    X_test = prepare_branches(test_features, feature_names, None, device)
    y_train = train_labels.to(device)
    y_val = val_labels.to(device)
    y_test = test_labels.to(device)

    model = BranchMLP(
        input_dims=[branch.shape[1] for branch in X_train],
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=int(train_labels.max().item()) + 1,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in tqdm(range(MAX_EPOCHS), desc=name, unit="epoch"):
        model.train()
        permutation = torch.randperm(len(y_train), device=device)
        num_batches = (len(y_train) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_index in range(num_batches):
            start = batch_index * BATCH_SIZE
            end = min((batch_index + 1) * BATCH_SIZE, len(y_train))
            batch_indices = permutation[start:end]
            batch_branches = [branch[batch_indices] for branch in X_train]
            batch_labels = y_train[batch_indices]

            logits = model(batch_branches)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, _ = evaluate_split(model, X_val, y_val, criterion)
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    train_metrics = evaluate_split_metrics(model, X_train, y_train, criterion)
    val_metrics = evaluate_split_metrics(model, X_val, y_val, criterion)
    test_metrics = evaluate_split_metrics(model, X_test, y_test, criterion)

    return {
        "name": name,
        "features": "+".join(feature_names),
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "train_acc": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "val_acc": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "test_acc": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_precision_per_class": test_metrics["precision_per_class"],
        "test_recall_per_class": test_metrics["recall_per_class"],
        "test_support_per_class": test_metrics["support_per_class"],
    }


def print_results(results: list[dict[str, Any]]) -> None:
    table = Table(title="MLP CLS Ablation Balanced")
    table.add_column("Experiment")
    table.add_column("Features")
    table.add_column("Params", justify="right")
    table.add_column("Best Epoch", justify="right")
    table.add_column("Train Acc", justify="right")
    table.add_column("Val Acc", justify="right")
    table.add_column("Test Acc", justify="right")
    table.add_column("Test Prec", justify="right")
    table.add_column("Test Rec", justify="right")

    for result in sorted(results, key=lambda item: float(item["test_acc"]), reverse=True):
        table.add_row(
            str(result["name"]),
            str(result["features"]),
            f"{int(result['params']):,}",
            str(result["best_epoch"]),
            f"{float(result['train_acc']):.4f}",
            f"{float(result['val_acc']):.4f}",
            f"{float(result['test_acc']):.4f}",
            f"{float(result['test_precision']):.4f}",
            f"{float(result['test_recall']):.4f}",
        )

    CONSOLE.print(table)


def print_best_confusion_matrix(results: list[dict[str, Any]], label_ids: list[int]) -> None:
    best_result = max(results, key=lambda item: float(item["test_acc"]))
    confusion_matrix = list(best_result["test_confusion_matrix"])
    num_classes = len(confusion_matrix)
    if num_classes > MAX_CONFUSION_MATRIX_CLASSES_TO_PRINT:
        CONSOLE.print(
            f"Best test confusion matrix for {best_result['name']} not printed because num_classes={num_classes}. Stored in the run JSON with label_ids."
        )
        return

    table = Table(title=f"Test Confusion Matrix: {best_result['name']}")
    table.add_column("true\\pred")
    for label_id in label_ids:
        table.add_column(str(label_id), justify="right")
    for label_id, row in zip(label_ids, confusion_matrix):
        table.add_row(str(label_id), *[str(int(value)) for value in row])
    CONSOLE.print(table)


def prompt_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    return input(f"{path.name} exists. Overwrite? [y/N]: ").strip().lower() == "y"


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "+"} else "_" for character in value)


def save_confusion_matrix_plot(
    name: str,
    confusion_matrix: list[list[int]],
    label_ids: list[int],
    output_path: Path,
) -> None:
    num_classes = len(label_ids)
    assert num_classes == len(confusion_matrix)
    figure_size = max(8.0, num_classes * (0.55 if num_classes <= 20 else 0.35 if num_classes <= 40 else 0.24))
    tick_size = 12 if num_classes <= 20 else 9 if num_classes <= 40 else 7
    value_size = 10 if num_classes <= 20 else 0
    max_value = max(max(row) for row in confusion_matrix)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(figure_size + 2.0, figure_size))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{name} test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(num_classes), label_ids, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), label_ids)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_aspect("equal")

    if value_size > 0:
        threshold = max_value * 0.5
        for row_index, row in enumerate(confusion_matrix):
            for column_index, value in enumerate(row):
                ax.text(
                    column_index,
                    row_index,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=value_size,
                    color="white" if value > threshold else "black",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=CONFUSION_MATRIX_IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_images(results: list[dict[str, Any]], label_ids: list[int], run_path: Path) -> list[Path]:
    output_dir = CONFUSION_MATRIX_RESULTS_DIR / run_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for result in results:
        output_path = output_dir / f"{safe_filename(str(result['name']))}_test_confusion_matrix{CONFUSION_MATRIX_IMAGE_SUFFIX}"
        if not prompt_overwrite(output_path):
            continue
        save_confusion_matrix_plot(
            name=str(result["name"]),
            confusion_matrix=list(result["test_confusion_matrix"]),
            label_ids=label_ids,
            output_path=output_path,
        )
        saved_paths.append(output_path)
    return saved_paths


def save_results(results: list[dict[str, Any]], run_meta: dict[str, Any]) -> Path:
    rows = []
    for result in results:
        rows.append(
            {
                "item_id": str(result["name"]),
                "status": "done",
                "axes": {
                    "modality": str(result["name"]),
                },
                "meta": {
                    "features": str(result["features"]),
                    "params": int(result["params"]),
                    "best_epoch": int(result["best_epoch"]),
                    "test_confusion_matrix": result["test_confusion_matrix"],
                    "test_precision_per_class": result["test_precision_per_class"],
                    "test_recall_per_class": result["test_recall_per_class"],
                    "test_support_per_class": result["test_support_per_class"],
                },
                "metrics": {
                    "train_acc": float(result["train_acc"]),
                    "train_precision": float(result["train_precision"]),
                    "train_recall": float(result["train_recall"]),
                    "val_acc": float(result["val_acc"]),
                    "val_precision": float(result["val_precision"]),
                    "val_recall": float(result["val_recall"]),
                    "test_acc": float(result["test_acc"]),
                    "test_precision": float(result["test_precision"]),
                    "test_recall": float(result["test_recall"]),
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

    train_data = load_shards(ROOT / MODEL_TRAIN)
    val_data = load_shards(ROOT / MODEL_VAL)
    test_data = load_shards(ROOT / MODEL_TEST)

    # here we try to balance
    samples_per_class = 2
    train_data = balance_data(train_data, seed=SEED, samples_per_class=samples_per_class)
    train_label_ids = torch.unique(train_data["labels"], sorted=True)
    val_supported_mask = torch.isin(val_data["labels"], train_label_ids)
    filtered_val_labels = torch.unique(val_data["labels"][~val_supported_mask], sorted=True).tolist()
    num_filtered_val_samples = int((~val_supported_mask).sum().item())
    val_data = subset_data(val_data, torch.nonzero(val_supported_mask, as_tuple=False).squeeze(1))
    assert torch.isin(test_data["labels"], train_label_ids).all().item()
    if filtered_val_labels:
        CONSOLE.print(
            f"Filtered {num_filtered_val_samples} validation samples with labels not present in train: {filtered_val_labels}"
        )

    train_features = {
        "cls_tokens": train_data["cls_tokens"].float(),
        "mean_pooled_patch_tokens": train_data["mean_pooled_patch_tokens"].float(),
        "mean_pooled_gt_masked_patch_tokens": train_data["mean_pooled_gt_masked_patch_tokens"].float(),
        "mean_pooled_sam_masked_patch_tokens": train_data["mean_pooled_sam_masked_patch_tokens"].float(),
    }
    val_features = {
        "cls_tokens": val_data["cls_tokens"].float(),
        "mean_pooled_patch_tokens": val_data["mean_pooled_patch_tokens"].float(),
        "mean_pooled_gt_masked_patch_tokens": val_data["mean_pooled_gt_masked_patch_tokens"].float(),
        "mean_pooled_sam_masked_patch_tokens": val_data["mean_pooled_sam_masked_patch_tokens"].float(),
    }
    test_features = {
        "cls_tokens": test_data["cls_tokens"].float(),
        "mean_pooled_patch_tokens": test_data["mean_pooled_patch_tokens"].float(),
        "mean_pooled_gt_masked_patch_tokens": test_data["mean_pooled_gt_masked_patch_tokens"].float(),
        "mean_pooled_sam_masked_patch_tokens": test_data["mean_pooled_sam_masked_patch_tokens"].float(),
    }
    experiments = BASE_EXPERIMENTS
    register_meta: dict[str, int] = {}
    if train_data["register_tokens"] is not None:
        assert val_data["register_tokens"] is not None
        assert test_data["register_tokens"] is not None
        train_register_tokens = train_data["register_tokens"].float()
        val_register_tokens = val_data["register_tokens"].float()
        test_register_tokens = test_data["register_tokens"].float()
        train_features |= {
            name: train_register_tokens[:, index, :]
            for index, name in enumerate(REGISTER_FEATURE_NAMES)
        }
        val_features |= {
            name: val_register_tokens[:, index, :]
            for index, name in enumerate(REGISTER_FEATURE_NAMES)
        }
        test_features |= {
            name: test_register_tokens[:, index, :]
            for index, name in enumerate(REGISTER_FEATURE_NAMES)
        }
        experiments = [*BASE_EXPERIMENTS, *REGISTER_EXPERIMENTS]
        register_meta = {
            "num_register_tokens": int(train_register_tokens.shape[1]),
            "register_feature_dim": int(train_register_tokens.shape[2]),
        }

    label_ids = torch.unique(train_data["labels"], sorted=True).tolist()
    train_labels, val_labels, test_labels = remap_labels_from_reference(
        train_data["labels"],
        val_data["labels"],
        test_data["labels"],
    )
    run_meta = {
        "seed": SEED,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "samples_per_class": samples_per_class,
        "val_split": "original_val",
        "num_filtered_val_samples": num_filtered_val_samples,
        "filtered_val_labels": filtered_val_labels,
        "projection_dim": PROJECTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": int(train_labels.max().item()) + 1,
        "num_train_samples": int(len(train_labels)),
        "num_val_samples": int(len(val_labels)),
        "num_test_samples": int(len(test_labels)),
        "label_ids": label_ids,
        "model_train": str(MODEL_TRAIN),
        "model_val": str(MODEL_VAL),
        "model_test": str(MODEL_TEST),
        **register_meta,
    }

    results = []
    for name, feature_names in experiments:
        results.append(
            run_experiment(
                name=name,
                feature_names=feature_names,
                train_features=train_features,
                val_features=val_features,
                test_features=test_features,
                train_labels=train_labels,
                val_labels=val_labels,
                test_labels=test_labels,
                device=device,
            )
        )

    print_results(results)
    print_best_confusion_matrix(results, label_ids)
    run_path = save_results(results, run_meta)
    image_paths = save_confusion_matrix_images(results, label_ids, run_path)
    CONSOLE.print(f"Saved run to {run_path}")
    if image_paths:
        CONSOLE.print(f"Saved {len(image_paths)} confusion matrix images to {CONFUSION_MATRIX_RESULTS_DIR / run_path.stem}")


if __name__ == "__main__":
    main()

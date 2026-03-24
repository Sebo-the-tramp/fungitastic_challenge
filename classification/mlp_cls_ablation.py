import os
import torch

from tqdm import tqdm
from pathlib import Path
from rich.table import Table
from rich.console import Console

from utils import load_shards, save_run, seed_everything

 
SEED = 7
SCRIPT_PATH = "classification/mlp_cls_ablation.py"
BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 224)
BACKGROUND_AUG = "normal"
FINAL_LAYER_CLASSIFIER_METHOD = "mlp"
EXPERIMENT_ID = f"{BACKBONE}_{BACKGROUND_AUG}_{IMAGE_SIZE}_{FINAL_LAYER_CLASSIFIER_METHOD}"

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_VAL = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/val")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")

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

REGISTER_FEATURE_NAMES = tuple(f"register_tokens_{index}" for index in range(NUM_REGISTER_TOKENS))
BASE_EXPERIMENTS: list[tuple[str, tuple[str, ...]]] = [
    ("cls", ("cls_tokens",)),
    ("patch", ("mean_pooled_patch_tokens",)),
    ("masked", ("mean_pooled_masked_patch_tokens",)),
    ("cls+patch", ("cls_tokens", "mean_pooled_patch_tokens")),
    ("cls+masked", ("cls_tokens", "mean_pooled_masked_patch_tokens")),
]
REGISTER_EXPERIMENTS: list[tuple[str, tuple[str, ...]]] = [
    (f"cls+register_{index}", ("cls_tokens", feature_name))
    for index, feature_name in enumerate(REGISTER_FEATURE_NAMES)
] + [
    ("cls+masked+register_3", ("cls_tokens", "mean_pooled_masked_patch_tokens", "register_tokens_3")),
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


def prepare_branches(
    features: dict[str, torch.Tensor],
    names: tuple[str, ...],
    indices: torch.Tensor | None,
    device: torch.device,
) -> list[torch.Tensor]:
    branches = []
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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_experiment(
    name: str,
    feature_names: tuple[str, ...],
    train_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int | str]:
    seed_everything(SEED)

    X_train = prepare_branches(train_features, feature_names, train_indices, device)
    X_val = prepare_branches(train_features, feature_names, val_indices, device)
    X_test = prepare_branches(test_features, feature_names, None, device)
    y_train = train_labels[train_indices].to(device)
    y_val = train_labels[val_indices].to(device)
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
    _, train_accuracy = evaluate_split(model, X_train, y_train, criterion)
    _, val_accuracy = evaluate_split(model, X_val, y_val, criterion)
    _, test_accuracy = evaluate_split(model, X_test, y_test, criterion)

    return {
        "name": name,
        "features": "+".join(feature_names),
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "train_acc": train_accuracy,
        "val_acc": val_accuracy,
        "test_acc": test_accuracy,
    }


def print_results(results: list[dict[str, float | int | str]]) -> None:
    table = Table(title="MLP CLS Ablation")
    table.add_column("Experiment")
    table.add_column("Features")
    table.add_column("Params", justify="right")
    table.add_column("Best Epoch", justify="right")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Test", justify="right")

    for result in sorted(results, key=lambda item: float(item["test_acc"]), reverse=True):
        table.add_row(
            str(result["name"]),
            str(result["features"]),
            f"{int(result['params']):,}",
            str(result["best_epoch"]),
            f"{float(result['train_acc']):.4f}",
            f"{float(result['val_acc']):.4f}",
            f"{float(result['test_acc']):.4f}",
        )

    CONSOLE.print(table)


def save_results(results: list[dict[str, float | int | str]], run_meta: dict[str, int | float | str]) -> Path:
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
                },
                "metrics": {
                    "train_acc": float(result["train_acc"]),
                    "val_acc": float(result["val_acc"]),
                    "test_acc": float(result["test_acc"]),
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

    train_features = {
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
    experiments = BASE_EXPERIMENTS
    register_meta: dict[str, int] = {}
    if train_data["register_tokens"] is not None:
        assert val_data["register_tokens"] is not None
        assert test_data["register_tokens"] is not None
        train_register_tokens = torch.cat([train_data["register_tokens"], val_data["register_tokens"]]).float()
        test_register_tokens = test_data["register_tokens"].float()
        train_features |= {
            name: train_register_tokens[:, index, :]
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

    train_labels_raw = torch.cat([train_data["labels"], val_data["labels"]])
    train_labels, test_labels = remap_labels_from_reference(train_labels_raw, test_data["labels"])
    train_indices, val_indices = make_stratified_split_indices(train_labels, VAL_FRACTION)
    run_meta = {
        "seed": SEED,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "val_fraction": VAL_FRACTION,
        "projection_dim": PROJECTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": int(train_labels.max().item()) + 1,
        "num_train_samples": int(len(train_indices)),
        "num_val_samples": int(len(val_indices)),
        "num_test_samples": int(len(test_labels)),
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
                test_features=test_features,
                train_labels=train_labels,
                test_labels=test_labels,
                train_indices=train_indices,
                val_indices=val_indices,
                device=device,
            )
        )

    print_results(results)
    run_path = save_results(results, run_meta)
    CONSOLE.print(f"Saved run to {run_path}")


if __name__ == "__main__":
    main()

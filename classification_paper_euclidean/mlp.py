import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from utils import compute_metrics_final_fast, load_masks, load_shards, seed_everything

SEED = 7
SEED_GENERATOR = 42
MIN_SAMPLES = 1
MAX_SAMPLES = 200
NUM_SEEDS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4
VAL_FRACTION = 0.2
HIDDEN_DIM = 256
SAVE_NETWORKS = False

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "448"))
DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
MASKS_PATH = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/test/720/FungiTastic/test/720p")
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "networks"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "mlp"

CONSOLE = Console()


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


def make_stratified_split_indices(labels: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    train_indices = []
    val_indices = []

    for label in torch.unique(labels, sorted=True):
        label_indices = torch.where(labels == label)[0]
        label_indices = label_indices[torch.randperm(len(label_indices))]
        if len(label_indices) == 1:
            train_indices.append(label_indices)
            continue
        num_val = max(1, int(round(len(label_indices) * val_fraction)))
        num_val = min(num_val, len(label_indices) - 1)
        val_indices.append(label_indices[:num_val])
        train_indices.append(label_indices[num_val:])

    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices) if val_indices else torch.empty(0, dtype=torch.long)
    train_indices = train_indices[torch.randperm(len(train_indices))]
    if len(val_indices) > 0:
        val_indices = val_indices[torch.randperm(len(val_indices))]
    return train_indices, val_indices


def build_model(input_dim: int, num_classes: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, num_classes),
    )


def normalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std


def evaluate_split(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    if len(labels) == 0:
        return float("nan"), float("nan")
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_or_load_mlp(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    checkpoint_path: Path,
    seed: int,
    samples_per_class: int,
    save_networks: bool,
    overwrite_networks: bool,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, int, float, float, bool]:
    input_dim = int(train_features.shape[1])

    if save_networks and checkpoint_path.exists() and not overwrite_networks:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["backbone"] == BACKBONE
        assert checkpoint["image_size"] == IMAGE_SIZE
        assert checkpoint["input_dim"] == input_dim
        assert checkpoint["num_classes"] == num_classes
        assert checkpoint["hidden_dim"] == HIDDEN_DIM
        model = build_model(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        mean = checkpoint["mean"].to(DEVICE)
        std = checkpoint["std"].to(DEVICE)
        return (
            model,
            mean,
            std,
            int(checkpoint["best_epoch"]),
            float(checkpoint["train_acc"]),
            float(checkpoint["val_acc"]),
            True,
        )

    seed_everything(seed)
    split_train_indices, split_val_indices = make_stratified_split_indices(train_labels.cpu(), VAL_FRACTION)
    split_train_indices_device = split_train_indices.to(DEVICE)
    split_val_indices_device = split_val_indices.to(DEVICE)

    mean = train_features[split_train_indices_device].mean(dim=0, keepdim=True)
    std = train_features[split_train_indices_device].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    X_train = normalize_features(train_features[split_train_indices_device], mean, std)
    y_train = train_labels[split_train_indices_device]
    X_val = normalize_features(train_features[split_val_indices_device], mean, std)
    y_val = train_labels[split_val_indices_device]

    model = build_model(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = clone_state_dict(model)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"Train seed={seed} n={samples_per_class}", leave=False):
        model.train()
        permutation = torch.randperm(len(y_train), device=DEVICE)
        for start in range(0, len(y_train), BATCH_SIZE):
            batch_indices = permutation[start:start + BATCH_SIZE]
            logits = model(X_train[batch_indices])
            loss = criterion(logits, y_train[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_loss = evaluate_split(model, X_val, y_val, criterion)[0] if len(y_val) > 0 else evaluate_split(model, X_train, y_train, criterion)[0]
        if current_loss < best_loss - MIN_DELTA:
            best_loss = current_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    train_acc = evaluate_split(model, X_train, y_train, criterion)[1]
    val_acc = evaluate_split(model, X_val, y_val, criterion)[1]

    if save_networks:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seed": seed,
                "samples_per_class": samples_per_class,
                "backbone": BACKBONE,
                "image_size": IMAGE_SIZE,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "hidden_dim": HIDDEN_DIM,
                "best_epoch": best_epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "mean": mean.detach().cpu(),
                "std": std.detach().cpu(),
                "model_state_dict": clone_state_dict(model),
            },
            checkpoint_path,
        )
    return model, mean, std, best_epoch, train_acc, val_acc, False


@torch.no_grad()
def mlp_method(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    test_file_names: list[str],
    total_pixels: torch.Tensor,
    pixel_in: torch.Tensor,
    pixel_out: torch.Tensor,
) -> tuple[list[dict[str, int | str]], dict[str, torch.Tensor]]:
    pred_class = model(normalize_features(test_features, mean, std)).argmax(dim=1).cpu()
    test_labels_cpu = test_labels.cpu()
    raw_data = []

    for i in range(len(test_labels_cpu)):
        raw_data.append(
            {
                "index": i,
                "file_name": test_file_names[i],
                "total_pixels": int(total_pixels[i].item()),
                "pixel_in": int(pixel_in[i].item()),
                "pixel_out": int(pixel_out[i].item()),
                "gt_class": int(test_labels_cpu[i].item()),
                "pred_class": int(pred_class[i].item()),
            }
        )

    return raw_data, {
        "gt_class": test_labels_cpu,
        "pred_class": pred_class,
        "total_pixels": total_pixels,
        "pixel_in": pixel_in,
        "pixel_out": pixel_out,
    }


def confirm_overwrite(path: Path) -> None:
    if not path.exists():
        return
    answer = input(f"{path} exists. Overwrite it? [y/N]: ").strip().lower()
    assert answer == "y", f"Refusing to overwrite {path}"


def ask_overwrite_networks() -> bool:
    answer = input("Overwrite existing cached MLP checkpoints if found? [y/N]: ").strip().lower()
    return answer == "y"


def ask_save_networks() -> bool:
    default_prompt = "Y/n" if SAVE_NETWORKS else "y/N"
    answer = input(f"Save trained MLP checkpoints? [{default_prompt}]: ").strip().lower()
    if not answer:
        return SAVE_NETWORKS
    return answer == "y"


def seed_computed_csv_path(seed: int) -> Path:
    return RESULTS_DIR / f"{seed}_computed.csv"


def print_summary(results: list[dict[str, int | float]]) -> None:
    table = Table(title="MLP Sweep")
    table.add_column("Samples", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("MLP", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("mIoU", justify="right")
    table.add_column("Epoch", justify="right")
    table.add_column("Cache", justify="right")

    for row in sorted(results, key=lambda item: float(item["accuracy_mlp"]), reverse=True)[:10]:
        table.add_row(
            str(int(row["samples_per_class"])),
            str(int(row["seed"])),
            f"{float(row['accuracy_mlp']):.4f}",
            f"{float(row['accuracy_mlp_overall']):.4f}",
            f"{float(row['mIoU']):.4f}",
            str(int(row["best_epoch"])),
            str(int(row["used_cached_weights"])),
        )

    CONSOLE.print(table)


def run_sweep(
    min_samples: int,
    max_samples: int,
    seeds: list[int],
    masks: dict[str, dict[str, torch.Tensor]],
    save_networks: bool,
    overwrite_networks: bool,
) -> list[dict[str, int | float]]:
    all_results = []

    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)

    train_labels, test_labels = remap_labels_fast(train_data["labels"], test_data["labels"])
    num_classes = int(train_labels.max().item()) + 1
    train_class_indices = build_class_indices(train_labels, num_classes)
    test_class_indices = build_class_indices(test_labels, num_classes)
    test_file_names_all, total_pixels_all, pixel_in_all, pixel_out_all = build_mask_cache(test_data["file_paths"], masks)

    train_features_all = train_data["cls_tokens"].float().to(DEVICE)
    train_labels_all = train_labels.to(DEVICE)
    test_features_all = test_data["cls_tokens"].float().to(DEVICE)
    test_labels_all = test_labels.to(DEVICE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for seed in tqdm(seeds, desc="Seeds", position=0):
        computed_csv_path = seed_computed_csv_path(seed)
        confirm_overwrite(computed_csv_path)
        with open(computed_csv_path, "w", newline="") as computed_file:
            writer = None
            for samples_per_class in tqdm(range(min_samples, max_samples + 1), desc=f"Samples seed={seed}", position=1, leave=False):
                train_indices = sample_balanced_indices(train_class_indices, seed=seed, samples_per_class=samples_per_class)
                test_indices = sample_balanced_indices(test_class_indices, seed=seed, samples_per_class=samples_per_class)
                train_indices_device = train_indices.to(DEVICE)
                test_indices_device = test_indices.to(DEVICE)
                checkpoint_path = CACHE_DIR / str(samples_per_class) / f"{seed}.pth"

                model, mean, std, best_epoch, train_acc, val_acc, used_cached_weights = train_or_load_mlp(
                    train_features=train_features_all[train_indices_device],
                    train_labels=train_labels_all[train_indices_device],
                    num_classes=num_classes,
                    checkpoint_path=checkpoint_path,
                    seed=seed,
                    samples_per_class=samples_per_class,
                    save_networks=save_networks,
                    overwrite_networks=overwrite_networks,
                )
                _, metrics_data = mlp_method(
                    model=model,
                    mean=mean,
                    std=std,
                    test_features=test_features_all[test_indices_device],
                    test_labels=test_labels_all[test_indices_device],
                    test_file_names=[test_file_names_all[i] for i in test_indices.tolist()],
                    total_pixels=total_pixels_all[test_indices],
                    pixel_in=pixel_in_all[test_indices],
                    pixel_out=pixel_out_all[test_indices],
                )
                metrics = compute_metrics_final_fast(metrics_data, num_classes=num_classes)

                result = {
                    "samples_per_class": samples_per_class,
                    "accuracy_euclidean": metrics["macro_img_acc"],
                    "accuracy_euclidean_overall": metrics["overall_img_acc"],
                    "accuracy_mlp": metrics["macro_img_acc"],
                    "accuracy_mlp_overall": metrics["overall_img_acc"],
                    "mIoU": metrics["mIoU"],
                    "best_epoch": best_epoch,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "used_cached_weights": int(used_cached_weights),
                }
                all_results.append({"seed": seed, **result})

                if writer is None:
                    writer = csv.DictWriter(computed_file, fieldnames=result.keys())
                    writer.writeheader()
                writer.writerow(result)

    return all_results


def main() -> None:
    seed_everything(SEED)
    save_networks = ask_save_networks()
    overwrite_networks = ask_overwrite_networks() if save_networks else False
    num_seeds = list(map(int, np.random.default_rng(SEED_GENERATOR).integers(0, 10000, size=NUM_SEEDS)))
    masks = load_masks(MASKS_PATH)
    results = run_sweep(
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        seeds=num_seeds,
        masks=masks,
        save_networks=save_networks,
        overwrite_networks=overwrite_networks,
    )
    print_summary(results)


if __name__ == "__main__":
    main()

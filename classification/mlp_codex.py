import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from utils import load_shards, seed_everything

SEED = 7
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
IMAGE_SIZE = 448
MODEL_ROOT = Path(f"facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_{IMAGE_SIZE}")
TRAIN_PATH = ROOT / MODEL_ROOT / "train"
VAL_PATH = ROOT / MODEL_ROOT / "val"
TEST_PATH = ROOT / MODEL_ROOT / "test"
MODEL_SPECS: list[tuple[str, int, int]] = [
    ("cls_masked", 60, 7),
    ("cls_masked", 60, 13),
    ("cls_masked", 60, 19),
    ("cls", 60, 7),
    ("cls", 60, 13),
]
BATCH_SIZE = 256
HIDDEN_DIM = 1024
DROPOUT = 0.2
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PIN_MEMORY = True
CONSOLE = Console()


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def remap_labels_from_reference(
    y_reference: torch.Tensor,
    *y_target_sets: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    reference_labels = torch.unique(y_reference, sorted=True)
    label_map = {int(label): idx for idx, label in enumerate(reference_labels.tolist())}
    mapped = []
    for labels in (y_reference, *y_target_sets):
        mapped.append(torch.tensor([label_map[int(label)] for label in labels.tolist()], dtype=torch.long))
    return tuple(mapped)


def normalize_feature(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def build_feature_bank(data: dict[str, torch.Tensor], feature_name: str) -> torch.Tensor:
    cls = normalize_feature(data["cls_tokens"])
    masked = normalize_feature(data["mean_pooled_masked_patch_tokens"])
    if feature_name == "cls":
        return cls
    if feature_name == "cls_masked":
        return torch.cat([cls, masked], dim=1)
    assert False, feature_name


def make_loader(features: torch.Tensor, labels: torch.Tensor, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.bincount(labels).float().pow(-0.5)[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY)


def evaluate_logits(
    model: MLP,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    loader = DataLoader(TensorDataset(features, labels), batch_size=1024, pin_memory=PIN_MEMORY)
    logits = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device, non_blocking=True)).cpu()
            logits.append(out)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return torch.cat(logits), correct / total


def train_one_model(
    feature_name: str,
    epochs: int,
    seed: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    seed_everything(seed)
    model = MLP(train_features.shape[1], int(train_labels.max().item()) + 1).to(device)
    loader = make_loader(train_features, train_labels, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    epoch_bar = tqdm(range(epochs), desc=f"{feature_name} s{seed}", unit="epoch", leave=False)

    for _ in epoch_bar:
        model.train()
        for xb, yb in loader:
            logits = model(xb.to(device, non_blocking=True))
            loss = F.cross_entropy(logits, yb.to(device, non_blocking=True), label_smoothing=LABEL_SMOOTHING)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return evaluate_logits(model, test_features, test_labels, device)


def print_data_table(train_labels: torch.Tensor, test_labels: torch.Tensor, device: torch.device) -> None:
    table = Table(title="MLP Codex Setup")
    table.add_column("item")
    table.add_column("value", justify="right")
    table.add_row("device", str(device))
    table.add_row("train+val samples", str(len(train_labels)))
    table.add_row("test samples", str(len(test_labels)))
    table.add_row("num classes", str(int(train_labels.max().item()) + 1))
    table.add_row("ensemble models", str(len(MODEL_SPECS)))
    CONSOLE.print(table)


def print_results(rows: list[tuple[str, int, int, float, float]]) -> None:
    table = Table(title="MLP Codex Results")
    table.add_column("feature")
    table.add_column("epochs", justify="right")
    table.add_column("seed", justify="right")
    table.add_column("single acc", justify="right")
    table.add_column("ensemble acc", justify="right")
    for feature_name, epochs, seed, single_acc, ensemble_acc in rows:
        table.add_row(feature_name, str(epochs), str(seed), f"{single_acc:.4f}", f"{ensemble_acc:.4f}")
    CONSOLE.print(table)


def main() -> None:
    seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_shards(TRAIN_PATH)
    val_data = load_shards(VAL_PATH)
    test_data = load_shards(TEST_PATH)

    full_train = {
        "cls_tokens": torch.cat([train_data["cls_tokens"], val_data["cls_tokens"]]),
        "mean_pooled_masked_patch_tokens": torch.cat(
            [train_data["mean_pooled_masked_patch_tokens"], val_data["mean_pooled_masked_patch_tokens"]]
        ),
    }
    train_labels = torch.cat([train_data["labels"], val_data["labels"]])
    train_labels, test_labels = remap_labels_from_reference(train_labels, test_data["labels"])

    feature_cache = {name: build_feature_bank(full_train, name) for name in {spec[0] for spec in MODEL_SPECS}}
    test_cache = {
        name: build_feature_bank(test_data, name)
        for name in {spec[0] for spec in MODEL_SPECS}
    }

    print_data_table(train_labels, test_labels, device)

    rows: list[tuple[str, int, int, float, float]] = []
    ensemble_logits: list[torch.Tensor] = []
    for feature_name, epochs, seed in MODEL_SPECS:
        logits, single_acc = train_one_model(
            feature_name=feature_name,
            epochs=epochs,
            seed=seed,
            train_features=feature_cache[feature_name],
            train_labels=train_labels,
            test_features=test_cache[feature_name],
            test_labels=test_labels,
            device=device,
        )
        ensemble_logits.append(logits)
        ensemble_pred = torch.stack(ensemble_logits).mean(dim=0).argmax(dim=1)
        ensemble_acc = (ensemble_pred == test_labels).float().mean().item()
        rows.append((feature_name, epochs, seed, single_acc, ensemble_acc))

    print_results(rows)


if __name__ == "__main__":
    main()

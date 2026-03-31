import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch

from tqdm import tqdm
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch.nn.functional as F

from utils import load_shards, seed_everything, balance_data, filter_data, remap_labels

    
SEED = 7

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 224)

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_VAL = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/val")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/fpelosin/projects/papers/fungi/fungitastic_challenge/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def prepare_data() -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:
    train_data = load_shards(ROOT / MODEL_TRAIN)
    val_data = load_shards(ROOT / MODEL_VAL)
    test_data = load_shards(ROOT / MODEL_TEST)    


    # filtering out classes in train/val not in test
    unique_labels_train = train_data['labels'].unique()
    unique_labels_val = val_data['labels'].unique()
    unique_labels_test = test_data['labels'].unique()

    train_union_val_unique = torch.cat([unique_labels_train, unique_labels_val], dim=0).unique()
    classes_in_train_val_not_in_test = set(train_union_val_unique.tolist()) - set(unique_labels_test.tolist())
    classes_in_train_val_not_in_test_tensor = torch.tensor(list(classes_in_train_val_not_in_test))
    print(f"Number of classes in test set not in train/val: {len(classes_in_train_val_not_in_test)}")

    # filter first
    train_data_filtered = filter_data(train_data, classes_in_train_val_not_in_test_tensor)
    val_data_filtered = filter_data(val_data, classes_in_train_val_not_in_test_tensor)

    # balance then
    samples_per_class = 500
    train_data_balanced = balance_data(train_data_filtered, seed=SEED, samples_per_class=samples_per_class)
    val_data_balanced = balance_data(val_data_filtered, seed=SEED, samples_per_class=samples_per_class)   
    test_data_balanced = balance_data(test_data, seed=SEED, samples_per_class=samples_per_class)

    train_labels, val_labels, test_labels = remap_labels(train_data_balanced['labels'], val_data_balanced['labels'], test_data_balanced['labels'])

    return (
        {**train_data_balanced, "labels": train_labels},
        {**val_data_balanced, "labels": val_labels},
        {**test_data_balanced, "labels": test_labels},
    )


def prototype_method(train_data: dict[str, torch.Tensor | None], 
     test_data: dict[str, torch.Tensor | None]) -> None:


    train_labels = train_data['labels']
    test_labels = test_data['labels']

    train_features = train_data['cls_tokens']    
    test_features = test_data['cls_tokens']

    num_classes = train_labels.max().item() + 1
    feature_dim = train_features.size(1)

    # Compute mean and std for prototype methods
    mean = torch.zeros(num_classes, feature_dim, dtype=train_features.dtype, device=train_features.device)
    std = torch.zeros(num_classes, feature_dim, dtype=train_features.dtype, device=train_features.device)
    for class_id in range(num_classes):
        class_features = train_features[train_labels == class_id]
        if len(class_features) > 0:
            mean[class_id] = class_features.mean(dim=0)
            std[class_id] = class_features.std(dim=0)

    # Prototype (mean) distances
    euclidean_distances = torch.cdist(test_features, mean)
    cosine_distances = 1 - F.cosine_similarity(
        test_features.unsqueeze(1),   # [N_test, 1, D]
        mean.unsqueeze(0),            # [1, C, D]
        dim=-1
    )
    # Diagonal Mahalanobis
    eps = 1e-6
    var = std.pow(2) + eps  # [C, D]
    diff = test_features.unsqueeze(1) - mean.unsqueeze(0)   # [N_test, C, D]
    mahalanobis_distances = torch.sqrt((diff.pow(2) / var.unsqueeze(0)).sum(dim=-1))  # [N_test, C]

    predicted_euclidean_labels = euclidean_distances.argmin(dim=1)
    predicted_cosine_labels = cosine_distances.argmin(dim=1)
    predicted_mahalanobis_labels = mahalanobis_distances.argmin(dim=1)

    accuracy_euclidean = (predicted_euclidean_labels == test_labels).float().mean().item()
    accuracy_cosine = (predicted_cosine_labels == test_labels).float().mean().item()
    accuracy_mahalanobis = (predicted_mahalanobis_labels == test_labels).float().mean().item()

    # # --- Memory Bank Approach ---
    # # For each class, collect its 5 training samples
    # samples_per_class = 5
    # device = train_features.device
    # memory_bank = torch.zeros(num_classes, samples_per_class, feature_dim, dtype=train_features.dtype, device=device)
    # for class_id in range(num_classes):
    #     class_features = train_features[train_labels == class_id]
    #     n = min(samples_per_class, class_features.shape[0])
    #     memory_bank[class_id, :n] = class_features[:n]

    # # For each test feature, compute distance to all 5 samples per class
    # N_test = test_features.shape[0]
    # # Euclidean: [N_test, num_classes, samples_per_class]
    # test_expanded = test_features.unsqueeze(1).unsqueeze(2)  # [N_test, 1, 1, D]
    # memory_expanded = memory_bank.unsqueeze(0)               # [1, num_classes, samples_per_class, D]
    # diff = test_expanded - memory_expanded                   # [N_test, num_classes, samples_per_class, D]
    # euclidean_mb = torch.norm(diff, dim=-1)                  # [N_test, num_classes, samples_per_class]
    # min_euclidean_mb, _ = euclidean_mb.min(dim=2)            # [N_test, num_classes]

    # # Cosine: [N_test, num_classes, samples_per_class]
    # test_norm = F.normalize(test_features, dim=-1).unsqueeze(1).unsqueeze(2)  # [N_test, 1, 1, D]
    # memory_norm = F.normalize(memory_bank, dim=-1).unsqueeze(0)               # [1, num_classes, samples_per_class, D]
    # cosine_sim = (test_norm * memory_norm).sum(dim=-1)                        # [N_test, num_classes, samples_per_class]
    # cosine_mb = 1 - cosine_sim                                                # cosine distance
    # min_cosine_mb, _ = cosine_mb.min(dim=2)                                   # [N_test, num_classes]

    # predicted_mb_euclidean = min_euclidean_mb.argmin(dim=1)
    # predicted_mb_cosine = min_cosine_mb.argmin(dim=1)

    # accuracy_mb_euclidean = (predicted_mb_euclidean == test_labels).float().mean().item()
    # accuracy_mb_cosine = (predicted_mb_cosine == test_labels).float().mean().item()
    accuracy_mb_euclidean = 0.0  # Placeholder since we are not computing it
    accuracy_mb_cosine = 0.0     # Placeholder since we are not computing it

    return (accuracy_euclidean, accuracy_cosine, accuracy_mahalanobis, accuracy_mb_euclidean, accuracy_mb_cosine)


def main() -> None:
    import colorcet as cc
    from matplotlib.colors import ListedColormap

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = Table(title="One-Shot Classification Results")
    table.add_column("Method", justify="left", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right", style="magenta")

    train_data, val_data, test_data = prepare_data()

    train_features = train_data["cls_tokens"].cpu().numpy()
    train_labels = train_data["labels"].cpu().numpy()

    print("Running t-SNE on train features...")
    # tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    # train_features_2d = tsne.fit_transform(train_features)

    # num_classes = len(np.unique(train_labels))
    # palette = cc.glasbey[:num_classes]
    # cmap = ListedColormap(palette)

    # plt.figure(figsize=(12, 10))
    # plt.scatter(
    #     train_features_2d[:, 0],
    #     train_features_2d[:, 1],
    #     c=train_labels,
    #     cmap=cmap,
    #     s=20,
    #     alpha=0.8,
    #     vmin=0,
    #     vmax=num_classes - 1,
    # )

    # for class_id in range(num_classes):
    #     class_points = train_features_2d[train_labels == class_id]
    #     if len(class_points) > 0:
    #         prototype_2d = class_points.mean(axis=0)
    #         plt.scatter(
    #             prototype_2d[0],
    #             prototype_2d[1],
    #             marker="+",
    #             s=180,
    #             c=[palette[class_id]],
    #             linewidths=2.5,
    #         )

    # plt.title("t-SNE of Train Features (Filtered & Balanced) with Prototypes")
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.tight_layout()
    # plt.savefig("tsne_train_data_filtered.png", dpi=300)
    # plt.close()

    (
        accuracy_euclidean,
        accuracy_cosine,
        accuracy_mahalanobis,
        accuracy_mb_euclidean,
        accuracy_mb_cosine,
    ) = prototype_method(train_data, val_data, test_data)

    print(f"Prototype method accuracy (Euclidean): {accuracy_euclidean:.4f}")
    print(f"Prototype method accuracy (Cosine): {accuracy_cosine:.4f}")
    print(f"Prototype method accuracy (Mahalanobis): {accuracy_mahalanobis:.4f}")
    print(f"Memory Bank accuracy (Euclidean): {accuracy_mb_euclidean:.4f}")
    print(f"Memory Bank accuracy (Cosine): {accuracy_mb_cosine:.4f}")

    table.add_row("Prototype Method (Euclidean)", f"{accuracy_euclidean:.4f}")
    table.add_row("Prototype Method (Cosine)", f"{accuracy_cosine:.4f}")
    table.add_row("Prototype Method (Mahalanobis)", f"{accuracy_mahalanobis:.4f}")
    table.add_row("Memory Bank (Euclidean)", f"{accuracy_mb_euclidean:.4f}")
    table.add_row("Memory Bank (Cosine)", f"{accuracy_mb_cosine:.4f}")

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
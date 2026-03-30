import os
import torch
import numpy as np
import plotext as pltx
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

from utils import load_shards, seed_everything, balance_data, remap_labels, graph_results

SEED = 7

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 224)

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def prepare_data(train_path:str, test_path:str) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:
    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)

    return (
        {**train_data, "labels": train_data['labels']},
        {**test_data, "labels": test_data['labels']},
    )


def balance_full_dataset(train_data: dict[str, torch.Tensor | None],
    test_data: dict[str, torch.Tensor | None], 
    seed: int = SEED, 
    samples_per_class: int = 5) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:

    train_data_balanced = balance_data(train_data, seed=seed, samples_per_class=samples_per_class)    
    test_data_balanced = balance_data(test_data, seed=seed, samples_per_class=samples_per_class)

    train_labels, test_labels = remap_labels(train_data_balanced['labels'], test_data_balanced['labels'])

    return (
        {**train_data_balanced, "labels": train_labels},
        {**test_data_balanced, "labels": test_labels},
    )


def prototype_method(train_data: dict[str, torch.Tensor | None], 
    test_data: dict[str, torch.Tensor | None]) -> None:

    # 1. Extract features and labels from the data
    train_labels = train_data['labels']
    test_labels = test_data['labels']

    train_features = train_data['cls_tokens']
    test_features = test_data['cls_tokens']

    # 2. Setup output tensor and indices
    num_classes = train_labels.max().item() + 1
    out = torch.zeros(num_classes, train_features.size(1), dtype=train_features.dtype, device=train_features.device)

    # Expand labels so the shape matches the features tensor: [N, Features]
    index = train_labels.unsqueeze(1).expand_as(train_features) 

    # 3. Compute the grouped mean
    out.scatter_reduce_(dim=0, index=index, src=train_features, reduce="mean", include_self=False)

    euclidean_distances = torch.cdist(test_features, out)
    predicted_labels = euclidean_distances.argmin(dim=1)
    accuracy = (predicted_labels == test_labels).float().mean().item()
    
    y_true = test_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


def main() -> None:
    seed_everything(SEED)

    results = {}
    MAX_SAMPLES = 100
    SEED_ITERATIONS = 1

    train_data, test_data = prepare_data()

    for i in tqdm(range(MAX_SAMPLES), desc="Samples per class"):
        results[i] = {}

        for j in tqdm(range(SEED_ITERATIONS), desc="Seeds", leave=False):
            random_seed = SEED + j

            train_data_balanced, test_data_balanced = balance_full_dataset(train_data, test_data, seed=random_seed, samples_per_class=i+1)
            run_result = prototype_method(train_data_balanced, test_data_balanced)
            results[i][random_seed] = run_result # Store only accuracy for now, can store full dict if you want to graph all metrics later

    graph_results(results, metrics=['accuracy'])


if __name__ == "__main__":
    main()
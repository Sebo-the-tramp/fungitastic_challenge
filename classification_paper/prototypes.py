import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch
import os
import csv

from sklearn.metrics import precision_recall_fscore_support
from utils import load_shards, seed_everything, balance_data, remap_labels, load_masks

SEED = 7

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 448)

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"


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
    
    accuracies_per_class = {}

    for class_id in range(num_classes):
        if(test_labels == class_id).sum().item() == 0:
            # print(f"Class {class_id}: No samples in test set, skipping.")
            continue
        class_features = test_features[test_labels == class_id]
        dist_euc = torch.cdist(class_features, mean, p=2)
        min_dist_euc = dist_euc.argmin(dim=1)
        acc_euc = (min_dist_euc == class_id).float().mean().item()
        # print(f"Class {class_id}: Euclidean Accuracy = {acc_euc:.4f}")

        accuracies_per_class[class_id] = {
            'euclidean': acc_euc
        }

    overall_acc_euc = (torch.cdist(test_features, mean, p=2).argmin(dim=1) == test_labels).float().mean().item()
    mAcc_euc = np.mean([v['euclidean'] for v in accuracies_per_class.values()])

    return overall_acc_euc, mAcc_euc


def run_sweep(min_samples=1, max_samples=None, seeds=[], experiment_name = "", save_csv=True):
    results = []

    # Prepare data with current samples_per_class
    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)


    for seed in seeds:
        seed_everything(seed)
        csv_path = f'./results/{experiment_name}/{seed}.csv'

        for samples_per_class in range(min_samples, max_samples + 1):
            
            train_data_balanced = balance_data(train_data, seed=seed, samples_per_class=samples_per_class)
            test_data_balanced = balance_data(test_data, seed=seed, samples_per_class=samples_per_class)

            train_labels, test_labels = remap_labels(train_data_balanced['labels'], test_data_balanced['labels'])
            train_data_balanced['labels'] = train_labels
            test_data_balanced['labels'] = test_labels

            # Run prototype method
            global_acc_euc, mAcc  = prototype_method(train_data_balanced, test_data_balanced)

            result = {
                'samples_per_class': samples_per_class,
                'accuracy_euclidean': mAcc,
                'accuracy_euclidean_overall': global_acc_euc,
            }
            results.append(result)
            # Save intermediate results
            os.makedirs('./results', exist_ok=True)
            if save_csv:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result.keys())
                    writer.writeheader()
                    writer.writerows(results)

    return results

def plot_sweep(results, save_only=False):
    x = [r['samples_per_class'] for r in results]
    plt.figure(figsize=(10, 7))
    plt.plot(x, [r['accuracy_euclidean'] for r in results], label='Euclidean')
    plt.xlabel('Samples per Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Samples per Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sweep_samples_per_class_plot.png', dpi=300)
    if not save_only:
        plt.show()
    plt.close()

if __name__ == "__main__":

    max_samples = 100
    num_seeds = [7, 42, 123, 2024, 9999]
    experiment_name = "prototype"

    masks = load_masks(Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/test/720/FungiTastic/test/720p"))
    results = run_sweep(1, max_samples, seeds=num_seeds, experiment_name=experiment_name)

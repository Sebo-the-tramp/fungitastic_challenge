import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch
import os
import csv

from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from utils import load_shards, seed_everything, balance_data, remap_labels

SEED = 7
PCA_DIM = 128
FEATURE_KEY = "cls_tokens"

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 448)

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"

TensorDict = dict[str, torch.Tensor | None]


def get_max_samples_per_class():
    train_data = load_shards(ROOT / MODEL_TRAIN)
    labels = train_data['labels'].cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    return counts.max()


@torch.inference_mode()
def prototype_method_pca(train_data: TensorDict, test_data: TensorDict) -> tuple[float, float]:
    train_labels = train_data['labels']
    test_labels = test_data['labels']
    train_features = train_data[FEATURE_KEY]
    test_features = test_data[FEATURE_KEY]

    pca_dim = min(PCA_DIM, train_features.shape[0], train_features.shape[1])
    pca = PCA(n_components=pca_dim, whiten=True, svd_solver='full')
    train_features = torch.from_numpy(pca.fit_transform(train_features.cpu().float().numpy())).to(
        device=train_features.device,
        dtype=torch.float32,
    )
    test_features = torch.from_numpy(pca.transform(test_features.cpu().float().numpy())).to(
        device=test_features.device,
        dtype=torch.float32,
    )

    num_classes = int(train_labels.max().item()) + 1
    feature_dim = train_features.size(1)
    mean = torch.zeros(num_classes, feature_dim, dtype=train_features.dtype, device=train_features.device)
    for class_id in range(num_classes):
        mean[class_id] = train_features[train_labels == class_id].mean(dim=0)

    predicted_labels = torch.cdist(test_features, mean, p=2).argmin(dim=1)
    accuracies_per_class = {}
    for class_id in range(num_classes):
        if(test_labels == class_id).sum().item() == 0:
            continue
        acc_euc = (predicted_labels[test_labels == class_id] == class_id).float().mean().item()
        accuracies_per_class[class_id] = {
            'euclidean': acc_euc
        }

    overall_acc_euc = (predicted_labels == test_labels).float().mean().item()
    mAcc_euc = float(np.mean([v['euclidean'] for v in accuracies_per_class.values()]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels.cpu().numpy(),
        predicted_labels.cpu().numpy(),
        average='macro',
        zero_division=0
    )

    return overall_acc_euc, mAcc_euc


def prototype_method(train_data: TensorDict, test_data: TensorDict) -> tuple[float, float]:
    return prototype_method_pca(train_data, test_data)


def run_sweep(min_samples=1, max_samples=None, save_csv=True):
    seed_everything(SEED)
    if max_samples is None:
        max_samples = get_max_samples_per_class()
    results = []
    csv_path = 'sweep_samples_per_class_results.csv'
    

    # Prepare data with current samples_per_class
    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)

    for samples_per_class in range(min_samples, max_samples + 1):
        
        train_data_balanced = balance_data(train_data, seed=SEED, samples_per_class=samples_per_class)
        test_data_balanced = balance_data(test_data, seed=SEED, samples_per_class=samples_per_class)

        train_labels, test_labels = remap_labels(train_data_balanced['labels'], test_data_balanced['labels'])
        train_data_balanced['labels'] = train_labels
        test_data_balanced['labels'] = test_labels
        global_acc_euc, mAcc  = prototype_method_pca(train_data_balanced, test_data_balanced)
        result = {
            'samples_per_class': samples_per_class,
            'accuracy_euclidean': mAcc,
        }
        print(result)
        results.append(result)
        # Save intermediate results
        if save_csv:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                writer.writeheader()
                writer.writerows(results)
        # Plot intermediate results
        plot_sweep(results, save_only=True)
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
    max_samples = get_max_samples_per_class()
    results = run_sweep(1, max_samples)
    plot_sweep(results)

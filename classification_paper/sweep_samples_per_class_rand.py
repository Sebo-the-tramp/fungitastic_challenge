import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch
import os

from utils import load_shards, seed_everything, balance_data, filter_data, remap_labels
from oneshot_classification import prototype_method, SEED, BACKBONE, IMAGE_SIZE, ROOT, MODEL_TRAIN, MODEL_VAL, MODEL_TEST

def get_max_samples_per_class():
    train_data = load_shards(ROOT / MODEL_TRAIN)
    labels = train_data['labels'].cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    return counts.max()

def run_sweep_rand(min_samples=1, max_samples=None, target_dim=128, save_csv=True):
    seed_everything(SEED)
    if max_samples is None:
        max_samples = get_max_samples_per_class()
    results = []
    csv_path = f'sweep_samples_per_class_rand_{target_dim}.csv'
    for samples_per_class in range(min_samples, max_samples + 1):
        print(f"Running for samples_per_class = {samples_per_class}, target_dim = {target_dim}")
        train_data = load_shards(ROOT / MODEL_TRAIN)
        val_data = load_shards(ROOT / MODEL_VAL)
        test_data = load_shards(ROOT / MODEL_TEST)
        unique_labels_train = train_data['labels'].unique()
        unique_labels_val = val_data['labels'].unique()
        unique_labels_test = test_data['labels'].unique()
        train_union_val_unique = torch.cat([unique_labels_train, unique_labels_val], dim=0).unique()
        classes_in_train_val_not_in_test = set(train_union_val_unique.tolist()) - set(unique_labels_test.tolist())
        classes_in_train_val_not_in_test_tensor = torch.tensor(list(classes_in_train_val_not_in_test))
        train_data_filtered = filter_data(train_data, classes_in_train_val_not_in_test_tensor)
        val_data_filtered = filter_data(val_data, classes_in_train_val_not_in_test_tensor)
        train_data_balanced = balance_data(train_data_filtered, seed=SEED, samples_per_class=samples_per_class)
        val_data_balanced = balance_data(val_data_filtered, seed=SEED, samples_per_class=samples_per_class)
        test_data_balanced = balance_data(test_data, seed=SEED, samples_per_class=samples_per_class)
        train_labels, val_labels, test_labels = remap_labels(train_data_balanced['labels'], val_data_balanced['labels'], test_data_balanced['labels'])
        train_data_balanced['labels'] = train_labels
        val_data_balanced['labels'] = val_labels
        test_data_balanced['labels'] = test_labels
        # Dimensionality reduction using random projection
        feature_dim = train_data_balanced['cls_tokens'].shape[1]
        rand_matrix = np.random.randn(feature_dim, target_dim) / np.sqrt(target_dim)
        for d in [train_data_balanced, val_data_balanced, test_data_balanced]:
            d['cls_tokens'] = torch.from_numpy(d['cls_tokens'].cpu().numpy() @ rand_matrix).to(torch.float32)
        acc_euc, acc_cos, acc_mah, _, _ = prototype_method(train_data_balanced, val_data_balanced, test_data_balanced)
        result = {
            'samples_per_class': samples_per_class,
            'accuracy_euclidean': acc_euc,
            'accuracy_cosine': acc_cos,
            'accuracy_mahalanobis': acc_mah
        }
        results.append(result)
        # Save intermediate results
        if save_csv:
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                writer.writeheader()
                writer.writerows(results)
        # Plot intermediate results
        plot_sweep(results, save_only=True, suffix=f'_rand_{target_dim}')
    return results

def plot_sweep(results, save_only=False, suffix=''):
    x = [r['samples_per_class'] for r in results]
    plt.figure(figsize=(10, 7))
    plt.plot(x, [r['accuracy_euclidean'] for r in results], label='Euclidean')
    plt.plot(x, [r['accuracy_cosine'] for r in results], label='Cosine')
    plt.plot(x, [r['accuracy_mahalanobis'] for r in results], label='Mahalanobis')
    plt.xlabel('Samples per Class')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Samples per Class (Random Projection{suffix})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'sweep_samples_per_class_plot{suffix}.png', dpi=300)
    if not save_only:
        plt.show()
    plt.close()

if __name__ == "__main__":
    max_samples = get_max_samples_per_class()
    target_dim = 128  # You can change this as needed
    results = run_sweep_rand(1, max_samples, target_dim=target_dim)
    plot_sweep(results, suffix=f'_rand_{target_dim}')

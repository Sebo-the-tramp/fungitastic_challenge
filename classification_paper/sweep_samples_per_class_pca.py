
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch
import os
import csv
from sklearn.decomposition import PCA

from utils import load_shards, seed_everything, balance_data, filter_data, remap_labels
from oneshot_classification import prototype_method, SEED, BACKBONE, IMAGE_SIZE, ROOT, MODEL_TRAIN, MODEL_VAL, MODEL_TEST

def get_max_samples_per_class():
    train_data = load_shards(ROOT / MODEL_TRAIN)
    labels = train_data['labels'].cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    return counts.max()

def run_sweep(min_samples=1, max_samples=None, save_csv=True):
    seed_everything(SEED)
    if max_samples is None:
        max_samples = get_max_samples_per_class()
    results = []
    csv_path = 'sweep_samples_per_class_pca_results.csv'
    for samples_per_class in range(min_samples, max_samples + 1):
        print(f"Running for samples_per_class = {samples_per_class} (with PCA)")
        # Prepare data with current samples_per_class
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

        # --- PCA transformation ---
        # Stack all features for PCA fit (train only)
        train_features = train_data_balanced['cls_tokens'].cpu().numpy()
        val_features = val_data_balanced['cls_tokens'].cpu().numpy()
        test_features = test_data_balanced['cls_tokens'].cpu().numpy()
        # Choose n_components (e.g., 128 or min(train_features.shape[1], train_features.shape[0]))
        n_components = min(128, train_features.shape[1], train_features.shape[0])
        pca = PCA(n_components=n_components, random_state=SEED)
        pca.fit(train_features)
        train_pca = torch.tensor(pca.transform(train_features)).float()
        val_pca = torch.tensor(pca.transform(val_features)).float()
        test_pca = torch.tensor(pca.transform(test_features)).float()
        train_data_balanced['cls_tokens'] = train_pca
        val_data_balanced['cls_tokens'] = val_pca
        test_data_balanced['cls_tokens'] = test_pca

        # Run prototype method on PCA features
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
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                writer.writeheader()
                writer.writerows(results)
        # Plot intermediate results
        plot_sweep(results, save_only=True, filename='sweep_samples_per_class_pca_plot.png')
    return results

def plot_sweep(results, save_only=False, filename='sweep_samples_per_class_pca_plot.png'):
    x = [r['samples_per_class'] for r in results]
    plt.figure(figsize=(10, 7))
    plt.plot(x, [r['accuracy_euclidean'] for r in results], label='Euclidean (PCA)')
    plt.plot(x, [r['accuracy_cosine'] for r in results], label='Cosine (PCA)')
    plt.plot(x, [r['accuracy_mahalanobis'] for r in results], label='Mahalanobis (PCA)')
    plt.xlabel('Samples per Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Samples per Class (PCA)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if not save_only:
        plt.show()
    plt.close()

if __name__ == "__main__":
    max_samples = get_max_samples_per_class()
    results = run_sweep(1, max_samples)
    plot_sweep(results, filename='sweep_samples_per_class_pca_plot.png')

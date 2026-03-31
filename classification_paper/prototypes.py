import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.table import Table
from rich.console import Console
import torch
import os
import csv
from tqdm import tqdm

from utils import load_shards, seed_everything, balance_data, remap_labels, load_masks, compute_metrics_final

SEED = 7

BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 448)

MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_normal_{IMAGE_SIZE}/test")
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
CLASSIFICATION_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def prototype_method(train_data: dict[str, torch.Tensor | None], 
    test_data: dict[str, torch.Tensor | None], masks) -> None:

    train_labels = train_data['labels']
    test_labels = test_data['labels']

    train_features = train_data['cls_tokens']
    test_features = test_data['cls_tokens']

    test_image_file_names = [file_path.split("/")[-1] for file_path in test_data['file_paths']]

    gt_masks_test = [masks.get(file_name.replace(".JPG", ".txt"))['gt_mask'] for file_name in test_image_file_names]
    sam_masks_test = [masks.get(file_name.replace(".JPG", ".txt"))['sam_mask'] for file_name in test_image_file_names]

    num_classes = train_labels.max().item() + 1
    feature_dim = train_features.size(1)

    prototypes = torch.zeros(num_classes, feature_dim, dtype=train_features.dtype, device=train_features.device)
    for class_id in range(num_classes):
        class_features = train_features[train_labels == class_id]
        if len(class_features) > 0:
            prototypes[class_id] = class_features.mean(dim=0)

    data_raw = []
    for i in range(len(test_labels)):

        total_pixels = gt_masks_test[i].sum().item()
        pixel_in = (gt_masks_test[i] & sam_masks_test[i]).sum().item()
        pixel_out = total_pixels - pixel_in
        gt_class = test_labels[i].item()
        pred_class = torch.cdist(test_features[i:i+1], prototypes, p=2).argmin(dim=1).item()

        data_raw.append({
            'index': i,
            'file_name': test_image_file_names[i].split('.')[0],
            'total_pixels': total_pixels,
            'pixel_in': pixel_in,
            'pixel_out': pixel_out,
            'gt_class': gt_class,
            'pred_class': pred_class
        })

    # overall_acc_euc = (torch.cdist(test_features, prototypes, p=2).argmin(dim=1) == test_labels).float().mean().item()
    # class_accuracies = [
    #     (torch.cdist(test_features[test_labels == class_id], prototypes, p=2).argmin(dim=1) == class_id).float().mean().item() 
    #     for class_id in range(num_classes)
    #     if (test_labels == class_id).any()  # This is the "Gatekeeper"
    # ]
    # mAcc_euc = np.mean(class_accuracies) if class_accuracies else 0.0

    # print(f"Overall Accuracy (Euclidean): {overall_acc_euc:.4f}")
    # print(f"Mean Accuracy (Euclidean): {mAcc_euc:.4f}")

    # print(f"+++++++++++++++++++++++++++++")

    # metrics = compute_metrics_final(data_raw, num_classes=num_classes)

    # for metric_name, metric_value in metrics.items():
    #     print(f"{metric_name}: {metric_value:.4f}")

    return data_raw


def run_sweep(min_samples=1, max_samples=None, seeds=[], experiment_name = "", masks={}, save_csv=True):

    # Prepare data with current samples_per_class
    train_data = load_shards(ROOT / MODEL_TRAIN)
    test_data = load_shards(ROOT / MODEL_TEST)

    for seed in tqdm(seeds, desc="Seeds", position=0):
        seed_everything(seed)

        results_raw = []
        results_computed = []

        csv_path_prefix = f'./results/{experiment_name}/{seed}'

        for samples_per_class in tqdm(
            range(min_samples, max_samples + 1),
            desc=f"Samples seed={seed}",
            position=1,
            leave=False,
        ):

            train_data_balanced = balance_data(train_data, seed=seed, samples_per_class=samples_per_class)
            test_data_balanced = balance_data(test_data, seed=seed, samples_per_class=samples_per_class)

            train_labels, test_labels = remap_labels(train_data_balanced['labels'], test_data_balanced['labels'])
            train_data_balanced['labels'] = train_labels
            test_data_balanced['labels'] = test_labels

            # Run prototype method
            raw_data = prototype_method(train_data_balanced, test_data_balanced, masks=masks)
            metrics = compute_metrics_final(raw_data, num_classes=train_labels.max().item() + 1)
            
            result_computed = {
                'samples_per_class': samples_per_class,
                'accuracy_euclidean': metrics["macro_img_acc"],
                'accuracy_euclidean_overall': metrics["overall_img_acc"]
            }
            results_computed.append(result_computed)

            results_raw.append([{"sample_per_class": samples_per_class, **data} for data in raw_data])

            # Save intermediate results
            os.makedirs("/".join(csv_path_prefix.split("/")[:-1]), exist_ok=True)            
            if save_csv:
                csv_path_computed = f"{csv_path_prefix}_computed.csv"
                csv_path_raw = f"{csv_path_prefix}_raw.csv"

                with open(csv_path_raw, 'a', newline='') as f:
                    last_row = results_raw[-1]
                    for row_dict in last_row:
                        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                        if f.tell() == 0:  # Check if file is empty to write header
                            writer.writeheader()
                        writer.writerow(row_dict)

                with open(csv_path_computed, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result_computed.keys())
                    if f.tell() == 0:  # Check if file is empty to write header
                        writer.writeheader()
                    writer.writerow(result_computed)



                # with open(csv_path_computed, 'w', newline='') as f:
                #     writer = csv.DictWriter(f, fieldnames=result_computed.keys())
                #     writer.writeheader()
                #     writer.writerows(results_computed)

                # csv_path_raw = f"{csv_path_prefix}_raw.csv"
                # with open(csv_path_raw, 'w', newline='') as f:
                #     writer = csv.DictWriter(f, fieldnames=results_raw[0][0].keys())
                #     writer.writeheader()
                #     for batch in results_raw[-1]:
                #         for line in batch:
                #             writer.writerows(line)
        
        csv_path_plot = f"{csv_path_prefix}_plot.png"
        plot_sweep(results_computed, save_path=f"{csv_path_plot}", save_only=True)

    return results

def plot_sweep(results, save_path="sweep_samples_per_class_plot.png", save_only=False):
    x = [r['samples_per_class'] for r in results]
    plt.figure(figsize=(10, 7))
    plt.plot(x, [r['accuracy_euclidean'] for r in results], label='Euclidean')
    plt.xlabel('Samples per Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Samples per Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}', dpi=300)
    if not save_only:
        plt.show()
    plt.close()

if __name__ == "__main__":

    max_samples = 100
    num_seeds = [7, 42, 123, 2024, 9999]
    experiment_name = "prototype"

    masks = load_masks(Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed/sam3_yolo_generic_mushroom_200/all/test/720/FungiTastic/test/720p"))
    results = run_sweep(1, max_samples, seeds=num_seeds, experiment_name=experiment_name, masks=masks, save_csv=True)

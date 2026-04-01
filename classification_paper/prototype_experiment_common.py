from pathlib import Path
import random

import torch


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


@torch.no_grad()
def build_prototypes(train_features: torch.Tensor, train_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    prototypes = torch.zeros(
        num_classes,
        train_features.shape[1],
        dtype=train_features.dtype,
        device=train_features.device,
    )
    prototypes.index_add_(0, train_labels, train_features)
    return prototypes / torch.bincount(train_labels, minlength=num_classes).unsqueeze(1)


def squared_euclidean_distances(test_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return (
        test_features.square().sum(dim=1, keepdim=True)
        + prototypes.square().sum(dim=1).unsqueeze(0)
        - 2 * test_features @ prototypes.T
    )


def build_prediction_outputs(
    pred_class: torch.Tensor,
    test_labels: torch.Tensor,
    test_file_names: list[str],
    total_pixels: torch.Tensor,
    pixel_in: torch.Tensor,
    pixel_out: torch.Tensor,
) -> tuple[list[dict[str, int | str]], dict[str, torch.Tensor]]:
    pred_class = pred_class.cpu()
    test_labels_cpu = test_labels.cpu()
    raw_data = []
    for i in range(len(test_labels_cpu)):
        raw_data.append({
            "index": i,
            "file_name": test_file_names[i],
            "total_pixels": int(total_pixels[i].item()),
            "pixel_in": int(pixel_in[i].item()),
            "pixel_out": int(pixel_out[i].item()),
            "gt_class": int(test_labels_cpu[i].item()),
            "pred_class": int(pred_class[i].item()),
        })
    return raw_data, {
        "gt_class": test_labels_cpu,
        "pred_class": pred_class,
        "total_pixels": total_pixels,
        "pixel_in": pixel_in,
        "pixel_out": pixel_out,
    }


def confirm_override_paths(paths: list[Path]) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return
    print("Existing output files:")
    for path in existing_paths:
        print(path)
    assert input("Override these files? [y/N]: ").strip().lower() == "y"


def make_experiment_dir(results_dir: Path, experiment_name: str) -> Path:
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def format_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")

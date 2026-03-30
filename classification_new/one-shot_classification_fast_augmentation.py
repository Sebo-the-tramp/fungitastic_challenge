import os

from collections import Counter
from pathlib import Path
from time import perf_counter

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MPLCONFIGDIR = RESULTS_DIR / ".matplotlib"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import torch

from tqdm import tqdm

from utils import seed_everything

SEED = int(os.environ.get("SEED", "7"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "50"))
SEED_ITERATIONS = int(os.environ.get("SEED_ITERATIONS", "5"))
SKIP_PLOTS = os.environ.get("SKIP_PLOTS", "0") == "1"
BACKBONE = os.environ.get("BACKBONE", "dinov3-vit7b16-pretrain-lvd1689m")
FEATURE_KEY = "cls_tokens"
BASE_VARIANT = os.environ.get("BASE_VARIANT", "bfloat16_normal_224")
CROP_VARIANTS = (
    "bfloat16_crop_224",
    "bfloat16_crop_black_224",
    "bfloat16_crop_448",
    "bfloat16_crop_black_448",
)

DEFAULT_EXTRA_VARIANTS = {
    "bfloat16_normal_224": ",".join(CROP_VARIANTS),
    "bfloat16_normal_448": ",".join(CROP_VARIANTS),
    "bfloat16_crop_224": "bfloat16_crop_black_224,bfloat16_crop_448,bfloat16_crop_black_448",
    "bfloat16_crop_black_224": "bfloat16_crop_224,bfloat16_crop_448,bfloat16_crop_black_448",
    "bfloat16_crop_448": "bfloat16_crop_224,bfloat16_crop_black_224,bfloat16_crop_black_448",
    "bfloat16_crop_black_448": "bfloat16_crop_224,bfloat16_crop_black_224,bfloat16_crop_448",

    # "bfloat16_normal_224": "bfloat16_masked_black_224,bfloat16_masked_blurred_224,bfloat16_masked_black_448,bfloat16_masked_blurred_448",
    # "bfloat16_normal_448": "bfloat16_masked_black_224,bfloat16_masked_blurred_224,bfloat16_masked_black_448,bfloat16_masked_blurred_448",    

    "bfloat16_masked_black_224": "bfloat16_masked_black_448,bfloat16_masked_blurred_224,bfloat16_masked_blurred_448",
    "bfloat16_masked_black_448": "bfloat16_masked_black_224,bfloat16_masked_blurred_224,bfloat16_masked_blurred_448",
    "bfloat16_masked_blurred_224": "bfloat16_masked_black_224,bfloat16_masked_black_448,bfloat16_masked_blurred_448",
    "bfloat16_masked_blurred_448": "bfloat16_masked_black_224,bfloat16_masked_black_448,bfloat16_masked_blurred_224",
}
EXTRA_VARIANTS = tuple(
    variant
    for variant in os.environ.get("EXTRA_VARIANTS", DEFAULT_EXTRA_VARIANTS.get(BASE_VARIANT, "")).split(",")
    if variant
)
TRAIN_VARIANTS = (BASE_VARIANT, *EXTRA_VARIANTS)
RUN_TAG = f"{BACKBONE}_{BASE_VARIANT}"
MAIN_PLOT_PATH = RESULTS_DIR / f"{RUN_TAG}_accuracy_baseline_vs_augmented.png"
DIAGNOSTIC_PLOT_PATH = RESULTS_DIR / f"{RUN_TAG}_accuracy_each_variant.png"
ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")
MODEL_ROOT = ROOT / "facebook" / BACKBONE
FULL_LABELS = {
    "bfloat16_normal_224": "Normal 224",
    "bfloat16_normal_448": "Normal 448",
    "bfloat16_crop_224": "Crop 224",
    "bfloat16_crop_black_224": "Crop Black 224",
    "bfloat16_crop_448": "Crop 448",
    "bfloat16_crop_black_448": "Crop Black 448",
    "bfloat16_masked_black_224": "Masked Black 224",
    "bfloat16_masked_black_448": "Masked Black 448",
    "bfloat16_masked_blurred_224": "Masked Blurred 224",
    "bfloat16_masked_blurred_448": "Masked Blurred 448",
}
SHORT_LABELS = {
    "bfloat16_normal_224": "224",
    "bfloat16_normal_448": "448",
    "bfloat16_crop_224": "Crop 224",
    "bfloat16_crop_black_224": "Crop Black 224",
    "bfloat16_crop_448": "Crop 448",
    "bfloat16_crop_black_448": "Crop Black 448",
    "bfloat16_masked_black_224": "Black 224",
    "bfloat16_masked_black_448": "Black 448",
    "bfloat16_masked_blurred_224": "Blurred 224",
    "bfloat16_masked_blurred_448": "Blurred 448",
}
VARIANT_COLORS = {
    "bfloat16_normal_224": "#111111",
    "bfloat16_normal_448": "#2563eb",
    "bfloat16_crop_224": "#ea580c",
    "bfloat16_crop_black_224": "#c2410c",
    "bfloat16_crop_448": "#dc2626",
    "bfloat16_crop_black_448": "#991b1b",
    "bfloat16_masked_black_224": "#059669",
    "bfloat16_masked_black_448": "#0f766e",
    "bfloat16_masked_blurred_224": "#7c3aed",
    "bfloat16_masked_blurred_448": "#a855f7",
}


def combined_label(variants: tuple[str, ...]) -> str:
    return " + ".join(SHORT_LABELS[variant] for variant in variants)


MAIN_CURVES = (
    (BASE_VARIANT, FULL_LABELS[BASE_VARIANT], (BASE_VARIANT,), VARIANT_COLORS[BASE_VARIANT]),
    ("augmented", combined_label(TRAIN_VARIANTS), TRAIN_VARIANTS, "#c2410c"),
)
DIAGNOSTIC_CURVES = tuple(
    (variant, FULL_LABELS[variant], (variant,), VARIANT_COLORS[variant]) for variant in TRAIN_VARIANTS
) + (("augmented", combined_label(TRAIN_VARIANTS), TRAIN_VARIANTS, "#c2410c"),)

TensorDict = dict[str, torch.Tensor]
FeatureMap = dict[str, torch.Tensor]
MetricDict = dict[str, float]
CurveResults = dict[str, dict[int, dict[int, MetricDict]]]


def load_feature_shards(data_path: Path) -> tuple[TensorDict, list[str]]:
    labels = []
    features = []
    file_ids = []
    shard_paths = sorted(data_path.glob("*.pt"))
    assert shard_paths, data_path

    for shard_path in tqdm(shard_paths, desc=f"Loading {data_path.parent.name}/{data_path.name}", unit="shard"):
        shard_data = torch.load(shard_path, map_location="cpu")
        labels.extend(shard_data["labels"])
        file_ids.extend(Path(file_path).name for file_path in shard_data["file_paths"])
        features.append(shard_data["cls_token"].float())

    return {"labels": torch.tensor(labels), FEATURE_KEY: torch.cat(features)}, file_ids


def align_train_variant(
    base_labels: torch.Tensor,
    base_file_ids: list[str],
    variant: str,
    variant_data: TensorDict,
    variant_file_ids: list[str],
) -> TensorDict:
    assert Counter(variant_file_ids) == Counter(base_file_ids)
    if variant_file_ids == base_file_ids:
        print(f"Aligned train split: {variant}")
        return {**variant_data, "labels": base_labels}

    variant_index = {file_id: idx for idx, file_id in enumerate(variant_file_ids)}
    order = torch.tensor([variant_index[file_id] for file_id in base_file_ids])
    aligned_data = {key: value[order] for key, value in variant_data.items()}
    aligned_data["labels"] = base_labels
    print(f"Aligned train split: {variant} (reordered)")
    return aligned_data


def filter_rows(data: TensorDict, keep_mask: torch.Tensor) -> TensorDict:
    return {key: value[keep_mask] for key, value in data.items()}


def build_class_indices(labels: torch.Tensor) -> list[torch.Tensor]:
    counts = torch.bincount(labels)
    return list(torch.argsort(labels).split(counts.tolist()))


def build_index_cache(class_indices: list[torch.Tensor], max_samples: int) -> dict[int, dict[int, torch.Tensor]]:
    cache = {samples: {} for samples in range(1, max_samples + 1)}

    for seed_offset in range(SEED_ITERATIONS):
        seed = SEED + seed_offset
        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = [indices[torch.randperm(indices.numel(), generator=generator)] for indices in class_indices]

        for samples in range(1, max_samples + 1):
            cache[samples][seed] = torch.cat([indices[:samples] for indices in shuffled_indices])

    return cache


def prepare_data() -> tuple[FeatureMap, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], int]:
    train_payloads = {variant: load_feature_shards(MODEL_ROOT / variant / "train") for variant in TRAIN_VARIANTS}
    test_data, _ = load_feature_shards(MODEL_ROOT / BASE_VARIANT / "test")

    train_datas = {variant: payload[0] for variant, payload in train_payloads.items()}
    train_file_ids = {variant: payload[1] for variant, payload in train_payloads.items()}
    base_train_labels = train_datas[BASE_VARIANT]["labels"]
    base_train_file_ids = train_file_ids[BASE_VARIANT]

    for variant in EXTRA_VARIANTS:
        train_datas[variant] = align_train_variant(
            base_train_labels,
            base_train_file_ids,
            variant,
            train_datas[variant],
            train_file_ids[variant],
        )

    shared_labels = torch.unique(base_train_labels, sorted=True)
    train_datas = {
        variant: {**train_data, "labels": torch.searchsorted(shared_labels, train_data["labels"])}
        for variant, train_data in train_datas.items()
    }
    test_data = {**test_data, "labels": torch.searchsorted(shared_labels, test_data["labels"])}

    train_labels = train_datas[BASE_VARIANT]["labels"]
    train_class_indices = build_class_indices(train_labels)
    # max_samples = min(MAX_SAMPLES, min(indices.numel() for indices in train_class_indices))
    max_samples = MAX_SAMPLES

    print(f"Classes: {len(train_class_indices)}")
    print(f"Train variants: {', '.join(TRAIN_VARIANTS)}")
    print(f"Max balanced samples per class: {max_samples}")
    return (
        {variant: train_data[FEATURE_KEY] for variant, train_data in train_datas.items()},
        train_labels,
        test_data[FEATURE_KEY],
        test_data["labels"],
        train_class_indices,
        max_samples,
    )


@torch.inference_mode()
def prototype_accuracy(
    train_feature_sets: list[torch.Tensor],
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
) -> MetricDict:
    feature_dim = train_feature_sets[0].size(1)
    prototypes = torch.zeros((num_classes, feature_dim), dtype=train_feature_sets[0].dtype)
    index = train_labels.unsqueeze(1).expand(-1, feature_dim)

    for train_features in train_feature_sets:
        prototypes.scatter_add_(0, index, train_features)

    counts = torch.bincount(train_labels, minlength=num_classes).to(train_feature_sets[0].dtype).unsqueeze(1)
    prototypes /= counts * len(train_feature_sets)

    train_norms = (prototypes * prototypes).sum(dim=1)
    test_norms = (test_features * test_features).sum(dim=1, keepdim=True)
    distances = test_norms + train_norms.unsqueeze(0) - 2 * (test_features @ prototypes.T)
    predicted_labels = distances.argmin(dim=1)

    confusion = torch.bincount(
        test_labels * num_classes + predicted_labels,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).float()
    per_class_accuracy = confusion.diag() / confusion.sum(dim=1).clamp_min_(1)
    return {
        "accuracy": float((predicted_labels == test_labels).float().mean().item()),
        "macro_accuracy": float(per_class_accuracy.mean().item()),
    }


def evaluate_curves(
    train_feature_map: FeatureMap,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    train_class_indices: list[torch.Tensor],
    max_samples: int,
) -> CurveResults:
    results: CurveResults = {key: {} for key, _, _, _ in DIAGNOSTIC_CURVES}
    train_index_cache = build_index_cache(train_class_indices, max_samples)
    num_classes = int(train_labels.max().item()) + 1

    for samples_per_class in tqdm(range(1, max_samples + 1), desc="Samples per class"):
        for key in results:
            results[key][samples_per_class] = {}

        for seed_offset in tqdm(range(SEED_ITERATIONS), desc="Seeds", leave=False):
            seed = SEED + seed_offset
            train_indices = train_index_cache[samples_per_class][seed]
            selected_train_labels = train_labels[train_indices]

            for key, _, variants, _ in DIAGNOSTIC_CURVES:
                train_feature_sets = [train_feature_map[variant][train_indices] for variant in variants]
                results[key][samples_per_class][seed] = prototype_accuracy(
                    train_feature_sets,
                    selected_train_labels,
                    test_features,
                    test_labels,
                    num_classes,
                )

    return results


def graph_results(results: CurveResults, curves: tuple[tuple[str, str, tuple[str, ...], str], ...], plot_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    anchor_key = curves[0][0]
    samples = sorted(results[anchor_key])
    fig, ax = plt.subplots(figsize=(10, 7))

    for key, label, _, color in curves:
        values = np.array(
            [[results[key][samples_per_class][seed]["accuracy"] for seed in sorted(results[key][samples_per_class])] for samples_per_class in samples],
            dtype=np.float32,
        )
        means = values.mean(axis=1)
        stds = values.std(axis=1)
        ax.plot(samples, means, label=label, color=color, linewidth=2.2)
        ax.fill_between(samples, means - stds, means + stds, color=color, alpha=0.12)

    ax.set_title(title)
    ax.set_xlabel("Samples per Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def confirm_override(path: Path) -> None:
    if path.exists():
        answer = input(f"{path} exists. Override? [y/N]: ").strip().lower()
        assert answer == "y"


def print_summary(results: CurveResults) -> None:
    anchor_key = DIAGNOSTIC_CURVES[0][0]
    last_samples = max(results[anchor_key])

    for key, label, _, _ in DIAGNOSTIC_CURVES:
        mean_accuracy = np.mean([row["accuracy"] for row in results[key][last_samples].values()])
        mean_macro_accuracy = np.mean([row["macro_accuracy"] for row in results[key][last_samples].values()])
        print(f"{label} accuracy@{last_samples}={mean_accuracy:.4f} macro_accuracy@{last_samples}={mean_macro_accuracy:.4f}")


def main() -> None:
    seed_everything(SEED)
    if not SKIP_PLOTS:
        confirm_override(MAIN_PLOT_PATH)
        confirm_override(DIAGNOSTIC_PLOT_PATH)

    prepare_start = perf_counter()
    train_feature_map, train_labels, test_features, test_labels, train_class_indices, max_samples = prepare_data()
    prepare_time = perf_counter() - prepare_start

    eval_start = perf_counter()
    results = evaluate_curves(
        train_feature_map,
        train_labels,
        test_features,
        test_labels,
        train_class_indices,
        max_samples,
    )
    eval_time = perf_counter() - eval_start

    plot_start = perf_counter()
    if not SKIP_PLOTS:
        graph_results(results, MAIN_CURVES, MAIN_PLOT_PATH, "Accuracy vs. Samples per Class")
        graph_results(results, DIAGNOSTIC_CURVES, DIAGNOSTIC_PLOT_PATH, "Accuracy by Train Variant")
    plot_time = perf_counter() - plot_start
    total_time = perf_counter() - prepare_start

    print(f"prepare_time={prepare_time:.2f}s")
    print(f"eval_time={eval_time:.2f}s")
    print(f"plot_time={plot_time:.2f}s")
    print(f"total_time={total_time:.2f}s")
    print(f"main_plot_path={MAIN_PLOT_PATH}")
    print(f"diagnostic_plot_path={DIAGNOSTIC_PLOT_PATH}")
    print(f"skip_plots={SKIP_PLOTS}")
    print_summary(results)


if __name__ == "__main__":
    main()

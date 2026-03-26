from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "runs"
OUTPUT_DIR = ROOT / "paper" / "media" / "teaser"
EXPERIMENT_SUFFIX = "_mlp"
OUTPUT_STEM = "mlp_absolute_test_acc_sorted"
MATPLOTLIB_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
IMAGE_SIZE = 448
MODALITY_ORDER = [
    "cls",
    "patch",
    "gt_masked",
    "sam_masked",
    "cls+patch",
    "cls+gt_masked",
    "cls+sam_masked",
    # "cls+register_tokens_3",
    # "cls+gt_masked+register_tokens_3",
]
MODALITY_ALIASES = {
    "cls+register_3": "cls+register_tokens_3",
    "cls+masked+register_3": "cls+masked+register_tokens_3",
    "cls+gt_masked+register_3": "cls+gt_masked+register_tokens_3",
    "cls+sam_masked+register_3": "cls+sam_masked+register_tokens_3",
}
COLORS = {
    "cls": "#1f77b4",
    "patch": "#ff7f0e",
    "gt_masked": "#2ca02c",
    "sam_masked": "#17becf",
    "cls+patch": "#9467bd",
    "cls+gt_masked": "#d62728",
    "cls+sam_masked": "#e377c2",
    "cls+register_tokens_3": "#8c564b",
    "cls+masked+register_tokens_3": "#bcbd22",
    "cls+gt_masked+register_tokens_3": "#7f7f7f",
    "cls+sam_masked+register_tokens_3": "#aec7e8",
}
MATPLOTLIB_MARKERS = {
    "cls": "o",
    "patch": "s",
    "gt_masked": "D",
    "sam_masked": "D",
    "cls+patch": "^",
    "cls+gt_masked": "X",
    "cls+sam_masked": "X",
    "cls+register_tokens_3": "P",
    "cls+masked+register_tokens_3": "*",
    "cls+gt_masked+register_tokens_3": "*",
    "cls+sam_masked+register_tokens_3": "*",
}
TITLE_SIZE = 60
LABEL_SIZE = 48
TICK_SIZE = 38
LEGEND_SIZE = 54
LINE_WIDTH = 4.2
MARKER_SIZE = 18
MATPLOTLIB_FIGSIZE = (32, 34)
BACKBONE_LABELS = {
    "vit7b16": "ViT-7B",
    "vits16": "ViT-S",
    "vits16plus": "ViT-S+",
    "vitb16": "ViT-B",
    "vitl16": "ViT-L",
    "vith16plus": "ViT-H+",
}


def latest_run_path(experiment_dir: Path) -> Path:
    json_paths = sorted(experiment_dir.glob("*.json"))
    assert json_paths, f"Missing json files in {experiment_dir}"
    return json_paths[-1]


def short_backbone(backbone: str) -> str:
    return backbone.replace("dinov3-", "").replace("-pretrain-lvd1689m", "")


def display_backbone(backbone: str) -> str:
    return BACKBONE_LABELS.get(short_backbone(backbone), f"DINOv3 {short_backbone(backbone)}")


def prompt_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    return input(f"{path.name} exists. Overwrite? [y/N]: ").strip().lower() == "y"


def canonical_modality(item: str) -> str:
    return MODALITY_ALIASES.get(item, item)


def has_gt_modality(item: str) -> bool:
    return any(part.startswith("gt_") for part in canonical_modality(item).split("+"))


def load_records() -> list[dict[str, Any]]:
    experiment_dirs = sorted(path for path in RUNS_DIR.iterdir() if path.is_dir() and path.name.endswith(EXPERIMENT_SUFFIX))
    assert experiment_dirs, f"No {EXPERIMENT_SUFFIX} experiments in {RUNS_DIR}"
    records: list[dict[str, Any]] = []
    for experiment_dir in experiment_dirs:
        payload = json.loads(latest_run_path(experiment_dir).read_text())
        experiment_id = str(payload["experiment_id"])
        for row in payload["rows"]:
            backbone = str(row["axes"]["backbone"])
            image_size = int(row["axes"]["image_size"])
            if image_size != IMAGE_SIZE:
                continue
            records.append(
                {
                    "experiment_id": experiment_id,
                    "backbone": backbone,
                    "backbone_short": short_backbone(backbone),
                    "backbone_label": display_backbone(backbone),
                    "image_size": image_size,
                    "item_id": canonical_modality(str(row["item_id"])),
                    "test_acc": float(row["metrics"]["test_acc"]),
                }
            )
    return records


def experiment_ids(records: list[dict[str, Any]]) -> list[str]:
    return sorted({record["experiment_id"] for record in records})


def ordered_items(records: list[dict[str, Any]], experiments: list[str]) -> list[str]:
    items_per_experiment = {
        experiment_id: {record["item_id"] for record in records if record["experiment_id"] == experiment_id}
        for experiment_id in experiments
    }
    missing_items = {
        experiment_id: [item for item in MODALITY_ORDER if item not in items_per_experiment[experiment_id]]
        for experiment_id in experiments
    }
    missing_items = {experiment_id: items for experiment_id, items in missing_items.items() if items}
    assert not missing_items, f"Missing modalities in latest runs: {missing_items}"
    return MODALITY_ORDER


def ordered_experiments(
    record_map: dict[tuple[str, str], dict[str, Any]], experiments: list[str], items: list[str]
) -> list[str]:
    return sorted(
        experiments,
        key=lambda experiment_id: (
            -max(record_map[(experiment_id, item)]["test_acc"] for item in items),
            record_map[(experiment_id, "cls")]["backbone_short"],
        ),
    )


def build_plot_data(
    record_map: dict[tuple[str, str], dict[str, Any]], experiments: list[str]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    labels = []
    for experiment_id in experiments:
        record = record_map[(experiment_id, "cls")]
        labels.append(f'{record["backbone_label"]} ({record["image_size"]} px)')
    return record_map, labels


def make_matplotlib(
    record_map: dict[tuple[str, str], dict[str, Any]], experiments: list[str], items: list[str], labels: list[str]
) -> tuple[plt.Figure, plt.Axes]:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )
    fig, ax = plt.subplots(figsize=MATPLOTLIB_FIGSIZE)
    x = list(range(len(experiments)))
    for item in items:
        y = [record_map[(experiment_id, item)]["test_acc"] * 100.0 for experiment_id in experiments]
        ax.plot(
            x,
            y,
            marker=MATPLOTLIB_MARKERS[item],
            linestyle="--" if has_gt_modality(item) else "-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=2.5,
            markeredgecolor="#ffffff",
            label=item,
            color=COLORS[item],
        )
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    # ax.set_xlabel("DINOv3 backbone / input resolution sorted from lowest to highest mean test accuracy")
    # ax.set_title("MLP modality comparison across DINOv3 backbones and resolutions")
    ax.legend(
        ncols=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        markerscale=2.6,
        handlelength=2.0,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    ax.grid(True, axis="y", color="#d1d5db", linewidth=1.0)
    ax.grid(False, axis="x")
    fig.subplots_adjust(bottom=0.42, top=0.95, left=0.11, right=0.99)
    return fig, ax


def main() -> None:
    assert RUNS_DIR.exists(), f"Missing {RUNS_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    should_save = prompt_overwrite(MATPLOTLIB_PATH)
    if not should_save:
        print("Skipped output")
        return
    records = load_records()
    record_map = {(record["experiment_id"], record["item_id"]): record for record in records}
    experiments = experiment_ids(records)
    items = ordered_items(records, experiments)
    experiments = ordered_experiments(record_map, experiments, items)
    record_map, matplotlib_labels = build_plot_data(record_map, experiments)

    matplotlib_fig, _ = make_matplotlib(record_map, experiments, items, matplotlib_labels)
    matplotlib_fig.savefig(MATPLOTLIB_PATH, dpi=220, bbox_inches="tight", pad_inches=0.25)
    print(f"Saved {MATPLOTLIB_PATH}")
    plt.close(matplotlib_fig)


if __name__ == "__main__":
    main()

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
MODALITY_ORDER = ["cls", "patch", "masked", "cls+patch", "cls+masked"]
COLORS = {
    "cls": "#0f172a",
    "patch": "#c2410c",
    "masked": "#15803d",
    "cls+patch": "#1d4ed8",
    "cls+masked": "#7c3aed",
}
MATPLOTLIB_MARKERS = {
    "cls": "o",
    "patch": "s",
    "masked": "D",
    "cls+patch": "^",
    "cls+masked": "X",
}
TITLE_SIZE = 30
LABEL_SIZE = 24
TICK_SIZE = 19
LEGEND_SIZE = 18
LINE_WIDTH = 3.2
MARKER_SIZE = 12
MATPLOTLIB_FIGSIZE = (20, 10.5)
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
            records.append(
                {
                    "experiment_id": experiment_id,
                    "backbone": backbone,
                    "backbone_short": short_backbone(backbone),
                    "backbone_label": display_backbone(backbone),
                    "image_size": image_size,
                    "item_id": str(row["item_id"]),
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
    common_items = set.intersection(*(items_per_experiment[experiment_id] for experiment_id in experiments))
    items = [item for item in MODALITY_ORDER if item in common_items]
    assert items, "No shared modalities across experiments"
    return items


def ordered_experiments(
    record_map: dict[tuple[str, str], dict[str, Any]], experiments: list[str], items: list[str]
) -> list[str]:
    return sorted(
        experiments,
        key=lambda experiment_id: (
            sum(record_map[(experiment_id, item)]["test_acc"] for item in items) / len(items),
            record_map[(experiment_id, "cls")]["image_size"],
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
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=1.5,
            markeredgecolor="#ffffff",
            label=item,
            color=COLORS[item],
        )
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    # ax.set_xlabel("DINOv3 backbone / input resolution sorted from lowest to highest mean test accuracy")
    # ax.set_title("MLP modality comparison across DINOv3 backbones and resolutions")
    ax.legend(ncols=len(items), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(True, axis="y", color="#d1d5db", linewidth=1.0)
    ax.grid(False, axis="x")
    fig.subplots_adjust(bottom=0.25, top=0.84)
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
    matplotlib_fig.savefig(MATPLOTLIB_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved {MATPLOTLIB_PATH}")
    plt.close(matplotlib_fig)


if __name__ == "__main__":
    main()

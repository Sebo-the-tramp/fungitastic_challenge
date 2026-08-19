from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import StrMethodFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_PAPER_DIR = ROOT / "classification_paper"
MODEL_RESULTS_PREFIX = "results_"
RESULT_SUBDIR = "prototype_pca_white_cosine_512"
BASELINE_DIR = CLASSIFICATION_PAPER_DIR / "results" / RESULT_SUBDIR
BASELINE_LABEL = "dinov3-vit7b16-pretrain-lvd1689m"
OUTPUT_DIR = CLASSIFICATION_PAPER_DIR / "results_backbone_comparison"
PANEL_FIGSIZE = (12, 9)
DPI = 300
STYLE = "seaborn-v0_8-whitegrid"
LINE_WIDTH = 3.0
ALPHA = 0.18
MIOU_MAX_PERCENT = 65.0
MIOU_HIDDEN_GRIDLINE_PERCENT = 60.0
TITLE_FONT_SIZE = 40
COUNT_LINE_COLOR = "#374151"
COUNT_FILL_COLOR = "#9ca3af"
COUNT_ALPHA = 0.22
LEGEND_FACE_COLOR = "#ffffff"
LEGEND_EDGE_COLOR = "#d1d5db"
LEGEND_ALPHA = 0.95
COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#1f78b4",
    "#b15928",
    "#17becf",
]
REFERENCE_LINES = {
    "mIoU": [
        ("SPECIFIC (oracle class)", 0.5522, "#8f5e3c"),
    ],
}
PLOTS = [
    (("mIoU",), "plot_miou_backbones.png", "mIoU (%)", "mIoU across runs and DINO variants"),
    (("accuracy_cosine", "accuracy_euclidean"), "plot_accuracy_backbones.png", "Accuracy (%)", "Accuracy across runs and DINO variants"),
]


def prompt_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    return input(f"{path.name} exists. Overwrite? [y/N]: ").strip().lower() == "y"


def has_expected_header(folder: Path) -> bool:
    paths = sorted(folder.glob("*_computed.csv"), key=lambda path: int(path.stem.split("_", 1)[0]))
    if not paths:
        return False
    with paths[0].open(newline="") as handle:
        first_line = handle.readline().strip()
    return first_line.startswith("samples_per_class,")


def model_folders() -> list[tuple[str, Path]]:
    CLASSIFICATION_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    folders = [(BASELINE_LABEL, BASELINE_DIR)] if has_expected_header(BASELINE_DIR) else []
    folders += [
        (results_dir.name.removeprefix(MODEL_RESULTS_PREFIX), results_dir / RESULT_SUBDIR)
        for results_dir in sorted(CLASSIFICATION_PAPER_DIR.glob(f"{MODEL_RESULTS_PREFIX}*"))
        if (results_dir / RESULT_SUBDIR).is_dir() and has_expected_header(results_dir / RESULT_SUBDIR)
    ]
    folders = sorted(folders, key=lambda item: item[0])
    assert folders, f"No model folders with *_computed.csv in {CLASSIFICATION_PAPER_DIR}"
    return folders


def computed_csv_paths(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("*_computed.csv"), key=lambda path: int(path.stem.split("_", 1)[0]))
    assert paths, f"Missing *_computed.csv in {folder}"
    return paths


def metric_names(folder: Path) -> list[str]:
    with computed_csv_paths(folder)[0].open(newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert fieldnames is not None, f"Missing header in computed csv under {folder}"
    return fieldnames


def resolve_metric(folder: Path, metrics: tuple[str, ...]) -> str | None:
    names = metric_names(folder)
    for metric in metrics:
        if metric in names:
            return metric
    return None


def load_run(csv_path: Path, metric: str) -> dict[int, float]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["samples_per_class"]): float(row[metric]) for row in rows}


def aggregate_folder(folder: Path, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    runs = [run for run in (load_run(path, metric) for path in computed_csv_paths(folder)) if run]
    assert runs, f"No non-empty *_computed.csv in {folder}"
    x = np.array(sorted({sample for run in runs for sample in run}), dtype=np.int64)
    values = np.full((len(runs), len(x)), np.nan, dtype=np.float64)
    for run_index, run in enumerate(runs):
        for x_index, sample in enumerate(x):
            if int(sample) in run:
                values[run_index, x_index] = run[int(sample)] * 100.0
    return x, np.nanmean(values, axis=0), np.nanstd(values, axis=0), values.shape[0]


def folder_styles(folders: list[tuple[str, Path]]) -> dict[str, dict[str, str]]:
    return {
        label: {"color": COLORS[index % len(COLORS)], "linestyle": "--" if label.startswith("dinov2-") else "-"}
        for index, (label, _) in enumerate(folders)
    }


def reference_raw_csv_path(folders: list[tuple[str, Path]]) -> Path:
    for _, folder in folders:
        paths = sorted(folder.glob("*_raw.csv"), key=lambda path: int(path.stem.split("_", 1)[0]))
        if paths:
            return paths[0]
    raise AssertionError(f"No *_raw.csv found in {CLASSIFICATION_PAPER_DIR}")


def classes_with_at_least_x_images(folders: list[tuple[str, Path]]) -> tuple[np.ndarray, np.ndarray]:
    max_sample = 0
    class_counts: dict[int, int] = {}
    with reference_raw_csv_path(folders).open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample = int(row["sample_per_class"])
            if sample > max_sample:
                max_sample = sample
                class_counts = {}
            if sample == max_sample:
                label = int(row["gt_class"])
                class_counts[label] = class_counts.get(label, 0) + 1
    assert max_sample > 0, f"No sample_per_class values found in raw csv under {CLASSIFICATION_PAPER_DIR}"
    counts = np.array(list(class_counts.values()), dtype=np.int64)
    samples = np.arange(1, max_sample + 1, dtype=np.int64)
    totals = (counts[:, None] >= samples[None, :]).sum(axis=0)
    return samples, totals


def sample_ticks(samples: np.ndarray) -> list[int]:
    max_sample = int(samples.max())
    ticks: list[int] = []
    base = 1
    while base <= max_sample:
        ticks.extend([base, 2 * base, 5 * base])
        base *= 10
    ticks = sorted({tick for tick in ticks if tick <= max_sample})
    return ticks if ticks[-1] == max_sample else ticks + [max_sample]


def miou_y_ticks(max_value: float) -> list[float]:
    top_tick = int(np.ceil(max_value / 10.0) * 10.0)
    return [1.0] + [float(tick) for tick in range(10, top_tick + 1, 10) if tick < max_value]


def apply_style() -> None:
    plt.style.use(STYLE)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )


def add_reference_lines(ax: plt.Axes, metric: str) -> None:
    for name, value, color in REFERENCE_LINES.get(metric, []):
        y = value * 100.0
        ax.axhline(y, color=color, linewidth=2.0, linestyle=":")
        label = f"{name} {value * 100.0:.2f}%" if metric.startswith("accuracy_") else f"{name} {value:.4f}"
        ax.annotate(
            label,
            xy=(1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-8, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=color,
            fontsize=12,
        )


def plot_metric_panel(
    ax: plt.Axes,
    folders: list[tuple[str, Path]],
    styles: dict[str, dict[str, str]],
    metrics: tuple[str, ...],
    ylabel: str,
    title: str,
    show_legend: bool,
) -> None:
    is_miou = metrics[0] == "mIoU"
    folder_metrics = [(label, folder, resolve_metric(folder, metrics)) for label, folder in folders]
    metric_folders = [(label, folder, metric) for label, folder, metric in folder_metrics if metric is not None]
    assert metric_folders, f"No result folders with metrics {metrics} in {CLASSIFICATION_PAPER_DIR}"
    max_value = 0.0
    for label, folder, metric in metric_folders:
        assert metric is not None
        x, mean, std, _ = aggregate_folder(folder, metric)
        style = styles[label]
        ax.plot(x, mean, color=style["color"], linewidth=LINE_WIDTH, linestyle=style["linestyle"], label=label)
        lower = np.maximum(mean - std, 1e-3) if is_miou else mean - std
        ax.fill_between(x, lower, mean + std, color=style["color"], alpha=ALPHA)
        if is_miou:
            max_value = max(max_value, float(np.max(mean + std)))
    add_reference_lines(ax, metrics[0])
    if is_miou:
        ticks = miou_y_ticks(MIOU_MAX_PERCENT)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0, top=MIOU_MAX_PERCENT)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.grid(False, axis="y", which="both")
        for tick in ticks:
            if not np.isclose(tick, MIOU_HIDDEN_GRIDLINE_PERCENT):
                ax.axhline(tick, color="#d1d5db", linewidth=1.0, zorder=0)
    else:
        ax.set_ylim(bottom=0)
        if metrics[0].startswith("accuracy_"):
            for tick in ax.get_yticks():
                if not np.isclose(tick, 10.0):
                    ax.axhline(tick, color="#d1d5db", linewidth=1.0, zorder=0)
        else:
            ax.grid(True, axis="y", color="#d1d5db", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", labelbottom=False)
    if show_legend:
        ax.legend(
            frameon=True,
            facecolor=LEGEND_FACE_COLOR,
            edgecolor=LEGEND_EDGE_COLOR,
            framealpha=LEGEND_ALPHA,
            ncols=2,
            loc="lower right",
        )


def plot_classes_with_at_least_x(ax: plt.Axes, samples: np.ndarray, totals: np.ndarray) -> None:
    ax.step(samples, totals, where="post", color=COUNT_LINE_COLOR, linewidth=2.6)
    ax.fill_between(samples, totals, step="post", color=COUNT_FILL_COLOR, alpha=COUNT_ALPHA)
    ax.set_xscale("log")
    ax.set_xlabel("x images per class")
    ax.set_ylabel("Classes with >= x")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, axis="y", color="#d1d5db", linewidth=1.0)
    ax.grid(False, axis="x")
    ticks = sample_ticks(samples)
    ax.set_xticks(ticks, [str(tick) for tick in ticks])


def make_panel_figure(
    folders: list[tuple[str, Path]],
    styles: dict[str, dict[str, str]],
    samples: np.ndarray,
    class_counts: np.ndarray,
    metrics: tuple[str, ...],
    ylabel: str,
    title: str,
) -> plt.Figure:
    apply_style()
    fig = plt.figure(figsize=PANEL_FIGSIZE, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(4.2, 1.35), hspace=0.06)
    ax_metric = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1], sharex=ax_metric)
    ax_metric.set_xscale("log")
    ax_bar.set_xscale("log")
    plot_metric_panel(ax_metric, folders, styles, metrics, ylabel, title, show_legend=True)
    plot_classes_with_at_least_x(ax_bar, samples, class_counts)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = model_folders()
    styles = folder_styles(folders)
    samples, class_counts = classes_with_at_least_x_images(folders)
    for metrics, output_name, ylabel, title in PLOTS:
        output_path = OUTPUT_DIR / output_name
        if not prompt_overwrite(output_path):
            print(f"Skipped {output_path}")
            continue
        fig = make_panel_figure(folders, styles, samples, class_counts, metrics, ylabel, title)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, StrMethodFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "classification_paper" / "results"
PANEL_FIGSIZE = (12, 9)
DPI = 300
STYLE = "seaborn-v0_8-whitegrid"
LINE_WIDTH = 3.0
ALPHA = 0.18
TITLE_FONT_SIZE = 40
COUNT_LINE_COLOR = "#374151"
COUNT_FILL_COLOR = "#9ca3af"
COUNT_ALPHA = 0.22
XTICK_STEP = 20
LEGEND_Y_OFFSET_PX = 50
RANDOM_LINE_COLOR = "#6b7280"
RANDOM_BASELINE_DENOM = 194
RANDOM_BASELINE_VALUE = 1.0 / RANDOM_BASELINE_DENOM
MIOU_Y_MIN = 5.0
MIOU_Y_MAX = 65.0
MIOU_Y_TICKS = [10.0, 20.0, 30.0, 40.0, 50.0]
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
SPECIAL_STYLES = {
    "mlp": {"color": "#c44e52", "linestyle": "--"},
}
SWEEP_PREFIX = "prototype_pca_white_cosine_"
SWEEP_DIM_MIN = 128
SWEEP_DIM_MAX = 2056
REFERENCE_LINES = {
    "mIoU": [
        ("SPECIFIC (oracle class)", 0.5522, "#8f5e3c"),
    ],
}
PLOTS = [
    (("mIoU",), "plot_miou_runs_pca_white_cosine_sweep.png", "mIoU (%)", "mIoU"),
    (
        ("accuracy_cosine", "accuracy_euclidean"),
        "plot_accuracy_runs_pca_white_cosine_sweep.png",
        "Accuracy (%)",
        "mAcc",
    ),
]


def prompt_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    return input(f"{path.name} exists. Overwrite? [y/N]: ").strip().lower() == "y"


def computed_csv_paths(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("*_computed.csv"), key=lambda path: int(path.stem.split("_", 1)[0]))
    assert paths, f"Missing *_computed.csv in {folder}"
    return paths


def has_expected_header(folder: Path) -> bool:
    paths = sorted(folder.glob("*_computed.csv"), key=lambda path: int(path.stem.split("_", 1)[0]))
    if not paths:
        return False
    with paths[0].open(newline="") as handle:
        first_line = handle.readline().strip()
    return first_line.startswith("samples_per_class,")


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


def folder_dim(folder: Path) -> int | None:
    if not folder.name.startswith(SWEEP_PREFIX):
        return None
    dim_text = folder.name.removeprefix(SWEEP_PREFIX)
    if not dim_text.isdigit():
        return None
    dim = int(dim_text)
    if dim < SWEEP_DIM_MIN or dim > SWEEP_DIM_MAX:
        return None
    return dim


def result_folders() -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folder_dims = [(folder, folder_dim(folder)) for folder in RESULTS_DIR.iterdir() if folder.is_dir()]
    folders = [
        folder
        for folder, dim in folder_dims
        if dim is not None and has_expected_header(folder)
    ]
    assert folders, (
        f"No result folders matching {SWEEP_PREFIX}<dim> in [{SWEEP_DIM_MIN}, {SWEEP_DIM_MAX}] "
        f"with *_computed.csv in {RESULTS_DIR}"
    )
    return sorted(folders, key=lambda folder: int(folder.name.removeprefix(SWEEP_PREFIX)))


def folder_styles(folders: list[Path]) -> dict[str, dict[str, str]]:
    styles: dict[str, dict[str, str]] = {}
    color_index = 0
    for folder in folders:
        style = SPECIAL_STYLES.get(folder.name)
        if style:
            styles[folder.name] = style
            continue
        styles[folder.name] = {"color": COLORS[color_index % len(COLORS)], "linestyle": "-"}
        color_index += 1
    return styles


def reference_raw_csv_path(folders: list[Path]) -> Path:
    for folder in folders:
        paths = sorted(folder.glob("*_raw.csv"), key=lambda path: int(path.stem.split("_", 1)[0]))
        if paths:
            return paths[0]
    raise AssertionError(f"No *_raw.csv found in {RESULTS_DIR}")


def classes_with_at_least_x_images(folders: list[Path]) -> tuple[np.ndarray, np.ndarray]:
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
    assert max_sample > 0, f"No sample_per_class values found in raw csv under {RESULTS_DIR}"
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


def add_reference_lines(ax: plt.Axes, folders: list[Path], metric: str) -> None:
    lines = list(REFERENCE_LINES.get(metric, []))
    if metric.startswith("accuracy_"):
        lines.append((f"Random (1/{RANDOM_BASELINE_DENOM})", RANDOM_BASELINE_VALUE, RANDOM_LINE_COLOR))
    for name, value, color in lines:
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
    folders: list[Path],
    styles: dict[str, dict[str, str]],
    metrics: tuple[str, ...],
    ylabel: str,
    title: str,
    show_legend: bool,
) -> None:
    is_miou = metrics[0] == "mIoU"
    folder_metrics = {folder: resolve_metric(folder, metrics) for folder in folders}
    metric_folders = [folder for folder, metric in folder_metrics.items() if metric is not None]
    assert metric_folders, f"No result folders with metrics {metrics} in {RESULTS_DIR}"
    upper_max = 0.0
    for folder in metric_folders:
        metric = folder_metrics[folder]
        assert metric is not None
        x, mean, std, _ = aggregate_folder(folder, metric)
        style = styles[folder.name]
        color = style["color"]
        linestyle = style["linestyle"]
        label = folder.name
        ax.plot(x, mean, color=color, linewidth=LINE_WIDTH, linestyle=linestyle, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=ALPHA)
        upper_max = max(upper_max, float(np.nanmax(mean + std)))

    for _, value, _ in REFERENCE_LINES.get(metrics[0], []):
        upper_max = max(upper_max, value * 100.0)
    if metrics[0].startswith("accuracy_"):
        upper_max = max(upper_max, RANDOM_BASELINE_VALUE * 100.0)

    y_max = max(10.0, float(np.ceil(upper_max / 10.0) * 10.0))
    add_reference_lines(ax, folders, metrics[0])
    if is_miou:
        ax.set_yscale("log")
        ax.set_ylim(bottom=MIOU_Y_MIN, top=MIOU_Y_MAX)
        ax.yaxis.set_major_locator(FixedLocator(MIOU_Y_TICKS))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    else:
        ax.set_yscale("linear")
        ax.set_ylim(0.0, y_max)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.grid(False, axis="y")
    if is_miou:
        for y in MIOU_Y_TICKS:
            ax.axhline(y=y, color="#d1d5db", linewidth=1.0, zorder=0)
    else:
        for y in range(10, int(y_max) + 1, 10):
            ax.axhline(y=float(y), color="#d1d5db", linewidth=1.0, zorder=0)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", labelbottom=False)
    if show_legend:
        if metrics[0].startswith("accuracy_"):
            offset = mtransforms.ScaledTranslation(0.0, LEGEND_Y_OFFSET_PX / ax.figure.dpi, ax.figure.dpi_scale_trans)
            ax.legend(
                frameon=True,
                ncols=2,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.0),
                bbox_transform=ax.transAxes + offset,
                facecolor="white",
                edgecolor="#d1d5db",
                framealpha=0.95,
            )
        else:
            ax.legend(frameon=True, ncols=2, loc="lower right", facecolor="white", edgecolor="#d1d5db", framealpha=0.95)


def plot_classes_with_at_least_x(ax: plt.Axes, samples: np.ndarray, totals: np.ndarray) -> None:
    ax.step(samples, totals, where="post", color=COUNT_LINE_COLOR, linewidth=2.6)
    ax.fill_between(samples, totals, step="post", color=COUNT_FILL_COLOR, alpha=COUNT_ALPHA)
    ax.set_xlabel("x images per class")
    ax.set_ylabel("Classes with >= x")
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, axis="y", color="#d1d5db", linewidth=1.0)
    ax.grid(False, axis="x")
    ticks = sample_ticks(samples)
    ax.set_xticks(ticks, [str(tick) for tick in ticks])


def make_panel_figure(
    folders: list[Path],
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folders = result_folders()
    styles = folder_styles(folders)
    samples, class_counts = classes_with_at_least_x_images(folders)
    for metrics, output_name, ylabel, title in PLOTS:
        output_path = RESULTS_DIR / output_name
        if not prompt_overwrite(output_path):
            print(f"Skipped {output_path}")
            continue
        fig = make_panel_figure(folders, styles, samples, class_counts, metrics, ylabel, title)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

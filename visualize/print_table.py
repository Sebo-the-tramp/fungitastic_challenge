import csv
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TABLE_PATH = ROOT / "visualize" / "example_table.ltx"
RESULTS_DIR = ROOT / "classification_paper" / "results"
TABLE_PATH = ROOT / "visualize" / "table.ltx"
TABLE_STD_PATH = ROOT / "visualize" / "table_std.ltx"

METHODS = [
    "prototype_normalized_cosine",
    "prototype_pca_cosine_1024",
    "prototype_pca_white_cosine_1024",
]
METHOD_LABELS = {
    "prototype_normalized_cosine": "Norm. cosine",
    "prototype_pca_cosine_1024": "PCA cosine",
    "prototype_pca_white_cosine_1024": "PCA white cosine",
}
K_VALUES = [5, 10, 20, 50, 100, 200]
ACCURACY_COLUMN = "accuracy_cosine"
MIOU_COLUMN = "mIoU"
DECIMALS = 2
TABLE_SIZE = "\\small"
TAB_COLSEP = "3pt"
MEAN_CAPTION = "Prototype results reported as mean across available seeds for different samples per class $k$."
MEAN_LABEL = "tab:prototype_results_mean"
STD_CAPTION = "Prototype results reported as mean\\pm std across available seeds for different samples per class $k$."
STD_LABEL = "tab:prototype_results_mean_std"

CONSOLE = Console()

MetricStats = dict[int, tuple[float, float]]
MethodSummary = dict[str, MetricStats]


def confirm_overwrite(path: Path) -> None:
    if not path.exists():
        return
    answer = input(f"{path} exists. Overwrite it? [y/N]: ").strip().lower()
    assert answer == "y", f"Refusing to overwrite {path}"


def read_computed_csv(path: Path) -> dict[int, dict[str, float]]:
    with open(path, newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            sample = int(row["samples_per_class"])
            rows[sample] = {key: float(value) for key, value in row.items() if key != "samples_per_class"}
    return rows


def summarize(values: list[float]) -> tuple[float, float]:
    assert values, "Cannot summarize an empty list"
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def summarize_method(method: str) -> MethodSummary:
    method_dir = RESULTS_DIR / method
    assert method_dir.exists(), f"Missing results directory: {method_dir}"
    csv_paths = sorted(method_dir.glob("*_computed.csv"))
    assert csv_paths, f"No computed CSV files found in {method_dir}"

    accuracy_values = {k: [] for k in K_VALUES}
    miou_values = {k: [] for k in K_VALUES}

    for csv_path in tqdm(csv_paths, desc=method, leave=False):
        rows = read_computed_csv(csv_path)
        for k in K_VALUES:
            if k not in rows:
                continue
            accuracy_values[k].append(rows[k][ACCURACY_COLUMN])
            miou_values[k].append(rows[k][MIOU_COLUMN])

    return {
        "mAcc": {k: summarize(accuracy_values[k]) for k in K_VALUES},
        "mIoU": {k: summarize(miou_values[k]) for k in K_VALUES},
    }


def format_preview(stats: tuple[float, float]) -> str:
    mean, std = stats
    return f"{mean:.{DECIMALS}f}"


def format_cell(stats: tuple[float, float], show_std: bool) -> str:
    mean, std = stats
    if show_std:
        return f"${mean:.{DECIMALS}f}\\pm{std:.{DECIMALS}f}$"
    return f"${mean:.{DECIMALS}f}$"


def load_template_parts(caption: str, label: str) -> tuple[list[str], list[str]]:
    lines = EXAMPLE_TABLE_PATH.read_text().splitlines()
    midrule_index = lines.index("\\midrule")
    bottomrule_index = lines.index("\\bottomrule")
    footer = []
    for line in lines[bottomrule_index:]:
        if line.startswith("\\vspace{"):
            continue
        if line.startswith("\\caption{"):
            footer.append(f"\\caption{{{caption}}}")
            continue
        if line.startswith("\\label{"):
            footer.append(f"\\label{{{label}}}")
            continue
        footer.append(line)
    return lines[:midrule_index + 1], footer


def build_latex_table(results: dict[str, MethodSummary], show_std: bool, caption: str, label: str) -> str:
    header, footer = load_template_parts(caption=caption, label=label)
    if TABLE_SIZE:
        header = [*header[:2], TABLE_SIZE, *header[2:]]
    if TAB_COLSEP:
        header = [*header[:3], f"\\setlength{{\\tabcolsep}}{{{TAB_COLSEP}}}", *header[3:]]
    body = []
    for method in METHODS:
        row = [METHOD_LABELS[method]]
        row.extend(format_cell(results[method]["mAcc"][k], show_std=show_std) for k in K_VALUES)
        row.extend(format_cell(results[method]["mIoU"][k], show_std=show_std) for k in K_VALUES)
        body.append("&".join(row) + r"\\")
    return "\n".join([*header, *body, *footer]) + "\n"


def print_preview(results: dict[str, MethodSummary]) -> None:
    table = Table(title="Prototype Results")
    table.add_column("Method")
    for k in K_VALUES:
        table.add_column(f"mAcc@{k}", justify="right")
    for k in K_VALUES:
        table.add_column(f"mIoU@{k}", justify="right")

    for method in METHODS:
        row = [METHOD_LABELS[method]]
        row.extend(format_preview(results[method]["mAcc"][k]) for k in K_VALUES)
        row.extend(format_preview(results[method]["mIoU"][k]) for k in K_VALUES)
        table.add_row(*row)

    CONSOLE.print(table)


def main() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {method: summarize_method(method) for method in METHODS}
    latex_table = build_latex_table(results, show_std=False, caption=MEAN_CAPTION, label=MEAN_LABEL)
    latex_table_std = build_latex_table(results, show_std=True, caption=STD_CAPTION, label=STD_LABEL)
    print_preview(results)
    print("\n% table.ltx\n", end="")
    print(latex_table, end="")
    print("\n% table_std.ltx\n", end="")
    print(latex_table_std, end="")
    confirm_overwrite(TABLE_PATH)
    confirm_overwrite(TABLE_STD_PATH)
    TABLE_PATH.write_text(latex_table)
    TABLE_STD_PATH.write_text(latex_table_std)
    CONSOLE.print(f"Saved {TABLE_PATH}")
    CONSOLE.print(f"Saved {TABLE_STD_PATH}")


if __name__ == "__main__":
    main()

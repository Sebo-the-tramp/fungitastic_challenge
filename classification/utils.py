import json
import random

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "results" / "runs"
EXPERIMENTS_PATH = ROOT / "dashboard" / "experiments.json"
RESULTS_PATH = ROOT / "dashboard" / "results.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_shards(data_path: Path) -> dict[str, torch.Tensor | None]:
    labels = []
    cls_tokens = []
    register_tokens = []
    mean_pooled_patch_tokens = []
    mean_pooled_masked_patch_tokens = []
    patch_features = []
    has_register_tokens = None

    for shard_path in tqdm(sorted(data_path.glob("*.pt")), desc=f"Loading {data_path.name}", unit="shard"):
        shard_data = torch.load(shard_path, map_location="cpu")
        labels.extend(shard_data["labels"])
        cls_tokens.append(shard_data["cls_token"].float())
        shard_register_tokens = shard_data["register_tokens"]
        if has_register_tokens is None:
            has_register_tokens = shard_register_tokens is not None
        assert has_register_tokens == (shard_register_tokens is not None)
        if shard_register_tokens is not None:
            register_tokens.append(shard_register_tokens.flatten(1).float())
        mean_pooled_patch_tokens.append(shard_data["mean_pooled_patch_tokens"].float())
        mean_pooled_masked_patch_tokens.append(shard_data["mean_pooled_masked_patch_tokens"].float())
        if "patch_features" in shard_data and shard_data["patch_features"] is not None:
            patch_features.append(shard_data["patch_features"].float())

    return {
        "labels": torch.tensor(labels),
        "cls_tokens": torch.cat(cls_tokens),
        "register_tokens": torch.cat(register_tokens).to("cpu") if register_tokens else None,
        "mean_pooled_patch_tokens": torch.cat(mean_pooled_patch_tokens).to("cpu"),
        "mean_pooled_masked_patch_tokens": torch.cat(mean_pooled_masked_patch_tokens).to("cpu"),
        "patch_features": torch.cat(patch_features) if patch_features else None,
    }


def remap_labels(y_train: torch.Tensor, y_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train_labels = torch.unique(y_train, sorted=True)
    assert torch.isin(y_test, train_labels).all().item()
    label_map = {int(label): idx for idx, label in enumerate(train_labels.tolist())}
    y_train_mapped = torch.tensor([label_map[int(label)] for label in y_train.tolist()])
    y_test_mapped = torch.tensor([label_map[int(label)] for label in y_test.tolist()])
    return y_train_mapped, y_test_mapped


def refresh_dashboard_results() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = json.loads(EXPERIMENTS_PATH.read_text())["experiments"]
    runs = [json.loads(path.read_text()) for path in sorted(RUNS_ROOT.glob("*/*.json"))]
    planned_items = sum(len(experiment.get("items", [])) for experiment in experiments)
    completed_items = 0
    best_test_acc = None
    axis_best: dict[str, dict[str, dict[str, Any]]] = {}
    experiment_rows: dict[str, list[dict[str, Any]]] = {}

    for run in runs:
        experiment_rows.setdefault(str(run["experiment_id"]), []).extend(
            [{**row, "timestamp": run["timestamp"]} for row in run["rows"]]
        )
        for row in run["rows"]:
            for axis_name, axis_value in row["axes"].items():
                candidate = {
                    "value": str(axis_value),
                    "item_id": str(row["item_id"]),
                    "experiment_id": str(run["experiment_id"]),
                    "test_acc": float(row["metrics"]["test_acc"]),
                    "timestamp": str(run["timestamp"]),
                }
                current = axis_best.setdefault(str(axis_name), {}).get(str(axis_value))
                if current is None or candidate["test_acc"] > current["test_acc"]:
                    axis_best[str(axis_name)][str(axis_value)] = candidate

    experiment_summaries = []
    for experiment in experiments:
        rows = experiment_rows.get(str(experiment["id"]), [])
        items = []
        best = None

        for item in experiment.get("items", []):
            item_rows = [row for row in rows if str(row["item_id"]) == str(item["name"])]
            if item_rows:
                completed_items += 1
                best_row = max(item_rows, key=lambda row: (float(row["metrics"]["test_acc"]), str(row["timestamp"])))
                latest_row = max(item_rows, key=lambda row: str(row["timestamp"]))
                best = best_row if best is None or float(best_row["metrics"]["test_acc"]) > float(best["metrics"]["test_acc"]) else best
                best_test_acc = float(best_row["metrics"]["test_acc"]) if best_test_acc is None else max(best_test_acc, float(best_row["metrics"]["test_acc"]))
                items.append(
                    {
                        "name": str(item["name"]),
                        "status": "done",
                        "best_metrics": dict(best_row["metrics"]),
                        "latest_metrics": dict(latest_row["metrics"]),
                        "best_meta": dict(best_row.get("meta", {})),
                        "latest_meta": dict(latest_row.get("meta", {})),
                        "best_timestamp": str(best_row["timestamp"]),
                        "latest_timestamp": str(latest_row["timestamp"]),
                    }
                )
                continue

            items.append({"name": str(item["name"]), "status": str(item["status"])})

        experiment_summaries.append(
            {
                "id": str(experiment["id"]),
                "num_runs": len({str(row["timestamp"]) for row in rows}),
                "completed_items": sum(1 for item in items if item["status"] == "done"),
                "total_items": len(items),
                "latest_timestamp": max((str(row["timestamp"]) for row in rows), default=None),
                "best": None if best is None else {
                    "item_id": str(best["item_id"]),
                    "test_acc": float(best["metrics"]["test_acc"]),
                },
                "items": items,
            }
        )

    data = {
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "overview": {
            "total_runs": len(runs),
            "planned_experiments": len(experiments),
            "planned_items": planned_items,
            "completed_items": completed_items,
            "remaining_items": planned_items - completed_items,
            "best_test_acc": best_test_acc,
        },
        "axes": [
            {"name": axis_name, "values": [values[value] for value in sorted(values)]}
            for axis_name, values in sorted(axis_best.items())
        ],
        "experiments": experiment_summaries,
    }
    RESULTS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def save_run(
    experiment_id: str,
    script_path: str,
    axes: dict[str, str | int | float],
    rows: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    run_dir = RUNS_ROOT / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_path = run_dir / f"{timestamp.replace(':', '-')}.json"
    assert not run_path.exists()

    payload = {
        "run_id": f"{experiment_id}_{timestamp}",
        "experiment_id": experiment_id,
        "script": script_path,
        "timestamp": timestamp,
        "meta": {} if meta is None else dict(meta),
        "rows": [
            {
                "item_id": str(row["item_id"]),
                "axes": {**axes, **dict(row.get("axes", {}))},
                "metrics": {key: float(value) for key, value in dict(row["metrics"]).items()},
                "meta": dict(row.get("meta", {})),
            }
            for row in rows
        ],
        "notes": notes,
    }
    run_path.write_text(json.dumps(payload, indent=2) + "\n")
    refresh_dashboard_results()
    return run_path

#!/usr/bin/env python
"""Rebuild the damage_focal_tversky_v2 campaign summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "damage_focal_tversky_v2.csv"
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions" / "damage_focal_tversky_v2"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "predictions" / "damage_focal_tversky_v2_summary.csv"

REFERENCES = [
    {
        "name": "U-Net + TTA d4",
        "f1_damaged": 0.631300,
        "iou_damaged": 0.461240,
        "mean_iou": 0.681574,
    },
    {
        "name": "dlong100_hist1000_attention_safe_sqrt4_focal_tversky",
        "f1_damaged": 0.678801,
        "iou_damaged": 0.513776,
        "mean_iou": 0.707285,
    },
]

FIELDNAMES = [
    "experiment",
    "model",
    "split",
    "epochs",
    "loss",
    "sampler",
    "damage_sampling_alpha",
    "seed",
    "base_channels",
    "time_limit",
    "status",
    "history_epochs",
    "checkpoint",
    "metrics_json",
    "pixel_accuracy",
    "mean_iou",
    "iou_background",
    "iou_no_damage",
    "iou_damaged",
    "precision_damaged",
    "recall_damaged",
    "f1_damaged",
    "average_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip(): str(value).strip() for key, value in row.items() if key}


def read_config(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [clean_row(row) for row in csv.DictReader(f)]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def metric_value(block: dict[str, Any], key: str, class_index: int | None = None) -> float | None:
    if class_index is None:
        value = block.get(key)
    else:
        values = block.get(key)
        value = values[class_index] if isinstance(values, list) and len(values) > class_index else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_history_epochs(path: Path) -> int:
    data = load_json(path)
    if data is None:
        return 0
    history = data if isinstance(data, list) else data.get("history") if isinstance(data, dict) else None
    return len(history) if isinstance(history, list) else 0


def extract_metrics(payload: dict[str, Any] | None) -> dict[str, float | None]:
    if payload is None:
        return {
            "pixel_accuracy": None,
            "mean_iou": None,
            "iou_background": None,
            "iou_no_damage": None,
            "iou_damaged": None,
            "precision_damaged": None,
            "recall_damaged": None,
            "f1_damaged": None,
            "average_loss": None,
        }
    summary = payload.get("summary_row")
    if isinstance(summary, dict):
        return {
            "pixel_accuracy": metric_value(summary, "pixel_accuracy"),
            "mean_iou": metric_value(summary, "mean_iou"),
            "iou_background": metric_value(summary, "iou_background"),
            "iou_no_damage": metric_value(summary, "iou_no_damage"),
            "iou_damaged": metric_value(summary, "iou_damaged"),
            "precision_damaged": metric_value(summary, "precision_damaged"),
            "recall_damaged": metric_value(summary, "recall_damaged"),
            "f1_damaged": metric_value(summary, "f1_damaged"),
            "average_loss": metric_value(summary, "average_loss"),
        }
    return {
        "pixel_accuracy": metric_value(payload, "pixel_accuracy"),
        "mean_iou": metric_value(payload, "mean_iou"),
        "iou_background": metric_value(payload, "iou_per_class", 0),
        "iou_no_damage": metric_value(payload, "iou_per_class", 1),
        "iou_damaged": metric_value(payload, "iou_per_class", 2),
        "precision_damaged": metric_value(payload, "precision_per_class", 2),
        "recall_damaged": metric_value(payload, "recall_per_class", 2),
        "f1_damaged": metric_value(payload, "f1_per_class", 2),
        "average_loss": metric_value(payload, "average_loss"),
    }


def status_for(expected_epochs: int, history_epochs: int, metrics_json: Path, checkpoint: Path) -> str:
    if history_epochs >= expected_epochs and metrics_json.exists():
        return "complete"
    if history_epochs >= expected_epochs and checkpoint.exists():
        return "evaluate_only"
    if history_epochs > 0:
        return "partial"
    return "missing"


def build_row(config_row: dict[str, str], predictions_dir: Path, checkpoints_dir: Path) -> dict[str, Any]:
    experiment = config_row["experiment"]
    metrics_json = predictions_dir / f"{experiment}_test_metrics.json"
    checkpoint = checkpoints_dir / experiment / "best_damage_arch.pt"
    history = checkpoints_dir / experiment / "metrics_history.json"
    expected_epochs = int(config_row["epochs"])
    history_epochs = load_history_epochs(history)
    payload = load_json(metrics_json)
    metrics = extract_metrics(payload)
    row = {
        "experiment": experiment,
        "model": config_row["model"],
        "split": config_row["split"],
        "epochs": expected_epochs,
        "loss": config_row["loss"],
        "sampler": config_row["sampler"],
        "damage_sampling_alpha": config_row["damage_sampling_alpha"],
        "seed": config_row["seed"],
        "base_channels": config_row["base_channels"],
        "time_limit": config_row["time_limit"],
        "status": status_for(expected_epochs, history_epochs, metrics_json, checkpoint),
        "history_epochs": history_epochs,
        "checkpoint": str(checkpoint),
        "metrics_json": str(metrics_json),
        **metrics,
    }
    return {key: row.get(key) for key in FIELDNAMES}


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("f1_damaged") or -1.0),
        float(row.get("iou_damaged") or -1.0),
        float(row.get("mean_iou") or -1.0),
    )


def finite_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def has_ranking_metrics(row: dict[str, Any]) -> bool:
    return all(
        finite_metric(row, key) is not None
        for key in ("f1_damaged", "iou_damaged", "mean_iou")
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("References:")
    for ref in REFERENCES:
        print(
            f"- {ref['name']}: "
            f"F1 damaged={ref['f1_damaged']:.6f}, "
            f"IoU damaged={ref['iou_damaged']:.6f}, "
            f"mean IoU={ref['mean_iou']:.6f}"
        )
    print()
    rows_with_metrics = [row for row in rows if has_ranking_metrics(row)]
    print(f"Total rows: {len(rows)}")
    print(f"Runs with metrics: {len(rows_with_metrics)}")
    print(f"Runs without metrics: {len(rows) - len(rows_with_metrics)}")
    print()
    print("Top runs by F1 damaged, IoU damaged, mean IoU:")
    ranked = sorted(rows_with_metrics, key=sort_key, reverse=True)
    if not ranked:
        print("No runs with numeric f1_damaged, iou_damaged, and mean_iou yet.")
        return
    for index, row in enumerate(ranked[:20], start=1):
        print(
            f"{index:02d}. {row['experiment']} | "
            f"F1={fmt(row['f1_damaged'])} "
            f"IoU_dmg={fmt(row['iou_damaged'])} "
            f"mIoU={fmt(row['mean_iou'])} "
            f"precision={fmt(row['precision_damaged'])} "
            f"recall={fmt(row['recall_damaged'])}"
        )


def main() -> None:
    args = parse_args()
    config_rows = read_config(args.config)
    rows = [build_row(row, args.predictions_dir, args.checkpoints_dir) for row in config_rows]
    rows = sorted(rows, key=lambda row: (row["status"], row["experiment"]))
    write_csv(args.output, rows)
    print(f"Wrote {args.output} ({len(rows)} row(s))")
    print_summary(rows)


if __name__ == "__main__":
    main()

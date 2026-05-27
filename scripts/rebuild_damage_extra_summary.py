"""Rebuild the extra damage sweep summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDNAMES = [
    "experiment",
    "split",
    "image_size",
    "batch_size",
    "loss",
    "class_weights",
    "lr",
    "epochs",
    "augment_mode",
    "sampler",
    "damage_sampling_alpha",
    "test_pixel_accuracy",
    "test_mean_iou",
    "test_iou_background",
    "test_iou_no_damage",
    "test_iou_damaged",
    "test_precision_damaged",
    "test_recall_damaged",
    "test_f1_damaged",
    "checkpoint",
    "test_metrics_json",
]


class SummaryError(Exception):
    """Raised when the damage extra summary cannot be rebuilt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the extra 1024 damage sweep summary CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/damage_extra_sweep_v1.csv"),
        help="Config CSV defining the sweep rows.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory containing evaluation metrics JSON files.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory containing experiment checkpoint folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions/unet_1024_damage_extra_sweep_v1_summary.csv"),
        help="Summary CSV path to write.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> list[dict[str, str]]:
    if not config_path.exists():
        raise SummaryError(f"Config CSV does not exist: {config_path}")
    try:
        with config_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except OSError as exc:
        raise SummaryError(f"Could not read config CSV '{config_path}': {exc}") from exc
    if not rows:
        raise SummaryError(f"Config CSV is empty: {config_path}")
    return rows


def load_metrics(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
    except OSError as exc:
        raise SummaryError(f"Could not read metrics JSON '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SummaryError(f"Could not parse metrics JSON '{path}': {exc}") from exc
    if not isinstance(metrics, dict):
        raise SummaryError(f"Metrics JSON must contain an object: {path}")
    return metrics


def class_value(metrics: dict[str, object], key: str, index: int) -> object:
    values = metrics.get(key, [])
    if not isinstance(values, list) or len(values) <= index:
        return None
    return values[index]


def normalize_class_weights(value: str) -> str:
    return value.replace(";", " ").strip()


def build_row(
    config: dict[str, str],
    predictions_dir: Path,
    checkpoints_dir: Path,
) -> dict[str, object] | None:
    experiment = config["experiment"]
    metrics_json = predictions_dir / f"{experiment}_test_metrics.json"
    if not metrics_json.exists():
        print(f"Skipping missing metrics: {metrics_json}")
        return None

    metrics = load_metrics(metrics_json)
    checkpoint = checkpoints_dir / experiment / "best_unet.pt"
    return {
        "experiment": experiment,
        "split": config["split"],
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "loss": config["loss"],
        "class_weights": normalize_class_weights(config["class_weights"]),
        "lr": config["lr"],
        "epochs": config["epochs"],
        "augment_mode": config["augment_mode"],
        "sampler": config["sampler"],
        "damage_sampling_alpha": config["damage_sampling_alpha"],
        "test_pixel_accuracy": metrics.get("pixel_accuracy"),
        "test_mean_iou": metrics.get("mean_iou"),
        "test_iou_background": class_value(metrics, "iou_per_class", 0),
        "test_iou_no_damage": class_value(metrics, "iou_per_class", 1),
        "test_iou_damaged": class_value(metrics, "iou_per_class", 2),
        "test_precision_damaged": class_value(metrics, "precision_per_class", 2),
        "test_recall_damaged": class_value(metrics, "recall_per_class", 2),
        "test_f1_damaged": class_value(metrics, "f1_per_class", 2),
        "checkpoint": str(checkpoint),
        "test_metrics_json": str(metrics_json),
    }


def metric_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        metric_float(row, "test_iou_damaged"),
        metric_float(row, "test_f1_damaged"),
        metric_float(row, "test_mean_iou"),
    )


def rebuild_summary(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser()
    predictions_dir = args.predictions_dir.expanduser()
    checkpoints_dir = args.checkpoints_dir.expanduser()
    rows = [
        row
        for config in load_config(config_path)
        if (row := build_row(config, predictions_dir, checkpoints_dir)) is not None
    ]
    rows.sort(key=sort_key, reverse=True)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary CSV: {output_path}")
    print(f"Rows: {len(rows)}")
    if rows:
        best = rows[0]
        print(
            "Best row: "
            f"{best['experiment']} | "
            f"IoU damaged={best['test_iou_damaged']} | "
            f"F1 damaged={best['test_f1_damaged']} | "
            f"mean IoU={best['test_mean_iou']}"
        )


def main() -> int:
    args = parse_args()
    try:
        rebuild_summary(args)
    except SummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

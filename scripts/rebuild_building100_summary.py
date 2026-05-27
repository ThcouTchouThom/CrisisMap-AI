"""Rebuild the Building100 Rorqual sweep summary from metrics JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


THRESHOLDS = ["0.3", "0.4", "0.5", "0.6"]
THRESHOLD_FIELD_METRICS = [
    "pixel_accuracy",
    "mean_iou",
    "background_iou",
    "building_iou",
    "building_precision",
    "building_recall",
    "building_f1",
    "object_precision",
    "object_recall",
]


class SummaryError(Exception):
    """Raised when the Building100 summary cannot be rebuilt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the Building100 sweep summary CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/building100_sweep_v1.csv"),
        help="Sweep configuration CSV.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory containing *_building_test_metrics.json files.",
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
        default=Path("outputs/predictions/building100_sweep_v1_summary.csv"),
        help="Summary CSV path to write.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SummaryError(f"Config CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except OSError as exc:
        raise SummaryError(f"Could not read JSON '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SummaryError(f"Could not parse JSON '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise SummaryError(f"JSON must contain an object: {path}")
    return payload


def threshold_key(raw: str) -> str:
    return raw.replace(".", "p")


def get_metric(metrics_by_threshold: dict[str, object], threshold: str, metric: str) -> object:
    block = metrics_by_threshold.get(threshold, {})
    if not isinstance(block, dict):
        return None
    if metric == "object_precision":
        object_metrics = block.get("object_metrics", {})
        return object_metrics.get("object_precision") if isinstance(object_metrics, dict) else None
    if metric == "object_recall":
        object_metrics = block.get("object_metrics", {})
        return object_metrics.get("object_recall") if isinstance(object_metrics, dict) else None
    return block.get(metric)


def float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_threshold(
    metrics_by_threshold: dict[str, object],
    metric: str,
) -> tuple[str | None, float | None]:
    best_key = None
    best_value = None
    for threshold in THRESHOLDS:
        value = float_or_none(get_metric(metrics_by_threshold, threshold, metric))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_key = threshold
            best_value = value
    return best_key, best_value


def build_fieldnames() -> list[str]:
    fields = [
        "experiment",
        "model",
        "train_csv",
        "val_csv",
        "test_csv",
        "input_mode",
        "loss",
        "augment_mode",
        "sampler",
        "sampler_alpha",
        "lr",
        "image_size",
        "batch_size",
        "epochs",
        "checkpoint",
        "metrics_path",
        "best_val_building_iou",
        "best_val_building_f1",
        "best_val_building_recall",
        "best_val_building_precision",
        "best_threshold_by_building_iou",
        "best_threshold_building_iou",
        "best_threshold_by_f1",
        "best_threshold_f1",
        "best_threshold_by_recall",
        "best_threshold_recall",
    ]
    for threshold in THRESHOLDS:
        prefix = f"test_t{threshold_key(threshold)}"
        fields.extend(f"{prefix}_{metric}" for metric in THRESHOLD_FIELD_METRICS)
    return fields


def build_row(
    config_row: dict[str, str],
    predictions_dir: Path,
    checkpoints_dir: Path,
) -> dict[str, object] | None:
    experiment = config_row["experiment"]
    metrics_path = predictions_dir / f"{experiment}_building_test_metrics.json"
    if not metrics_path.exists():
        print(f"Skipping missing metrics: {metrics_path}")
        return None

    checkpoint_dir = checkpoints_dir / experiment
    checkpoint = checkpoint_dir / "best_building.pt"
    best_val_path = checkpoint_dir / "best_val_metrics.json"
    best_val = read_json(best_val_path) if best_val_path.exists() else {}
    metrics_payload = read_json(metrics_path)
    metrics_by_threshold = metrics_payload.get("metrics_by_threshold", {})
    if not isinstance(metrics_by_threshold, dict):
        metrics_by_threshold = {}

    best_iou_threshold, best_iou = best_threshold(metrics_by_threshold, "building_iou")
    best_f1_threshold, best_f1 = best_threshold(metrics_by_threshold, "building_f1")
    best_recall_threshold, best_recall = best_threshold(metrics_by_threshold, "building_recall")

    row: dict[str, object] = {
        "experiment": experiment,
        "model": config_row["model"],
        "train_csv": config_row["train_csv"],
        "val_csv": config_row["val_csv"],
        "test_csv": config_row["test_csv"],
        "input_mode": config_row["input_mode"],
        "loss": config_row["loss"],
        "augment_mode": config_row["augment_mode"],
        "sampler": config_row["sampler"],
        "sampler_alpha": config_row["sampler_alpha"],
        "lr": config_row["lr"],
        "image_size": config_row["image_size"],
        "batch_size": config_row["batch_size"],
        "epochs": config_row["epochs"],
        "checkpoint": str(checkpoint),
        "metrics_path": str(metrics_path),
        "best_val_building_iou": best_val.get("building_iou"),
        "best_val_building_f1": best_val.get("building_f1"),
        "best_val_building_recall": best_val.get("building_recall"),
        "best_val_building_precision": best_val.get("building_precision"),
        "best_threshold_by_building_iou": best_iou_threshold,
        "best_threshold_building_iou": best_iou,
        "best_threshold_by_f1": best_f1_threshold,
        "best_threshold_f1": best_f1,
        "best_threshold_by_recall": best_recall_threshold,
        "best_threshold_recall": best_recall,
    }
    for threshold in THRESHOLDS:
        prefix = f"test_t{threshold_key(threshold)}"
        for metric in THRESHOLD_FIELD_METRICS:
            row[f"{prefix}_{metric}"] = get_metric(metrics_by_threshold, threshold, metric)
    return row


def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        float_or_none(row.get("best_threshold_building_iou")) or -1.0,
        float_or_none(row.get("best_threshold_f1")) or -1.0,
        float_or_none(row.get("best_threshold_recall")) or -1.0,
    )


def rebuild_summary(args: argparse.Namespace) -> None:
    config_rows = read_csv(args.config)
    predictions_dir = args.predictions_dir.expanduser()
    checkpoints_dir = args.checkpoints_dir.expanduser()
    rows = [
        row
        for config_row in config_rows
        if (row := build_row(config_row, predictions_dir, checkpoints_dir)) is not None
    ]
    rows.sort(key=sort_key, reverse=True)

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=build_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV: {output}")
    print(f"Rows: {len(rows)}")


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

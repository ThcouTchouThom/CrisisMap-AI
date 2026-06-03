"""Rebuild the targeted long damage finalists campaign summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("configs/damage_finalists_long_v1.csv")
DEFAULT_PREDICTIONS_DIR = Path("outputs/predictions")
DEFAULT_CHECKPOINTS_DIR = Path("outputs/checkpoints")
DEFAULT_OUTPUT = DEFAULT_PREDICTIONS_DIR / "damage_finalists_long_v1_summary.csv"
UNET_CHAMPION_TTA_D4 = {
    "label": "U-Net champion TTA d4",
    "mean_iou": 0.681574,
    "iou_damaged": 0.461240,
    "f1_damaged": 0.631300,
}
METRIC_FIELDS = [
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


class SummaryError(Exception):
    """Raised when the finalists summary cannot be rebuilt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the long damage finalists summary CSV.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clean_key(key: object) -> str:
    return str(key).strip().lstrip("\ufeff") if key is not None else ""


def clean_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def clean_row(row: dict[object, object]) -> dict[str, str]:
    return {clean_key(key): clean_value(value) for key, value in row.items()}


def read_config(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SummaryError(f"Config CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def get_required(row: dict[str, str], key: str, path: Path, row_index: int) -> str:
    value = row.get(key, "")
    if value:
        return value
    raise SummaryError(
        f"Missing required field '{key}' in {path}, row {row_index}. "
        f"Available keys: {list(row)}. Row: {row}"
    )


def load_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read {path}: {exc}", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


def metric_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_metric_value(values: object, index: int) -> float | None:
    if isinstance(values, list) and len(values) > index:
        return metric_float(values[index])
    return None


def extract_metrics(payload: dict[str, object]) -> dict[str, object]:
    summary_row = payload.get("summary_row")
    if isinstance(summary_row, dict):
        return {field: summary_row.get(field) for field in METRIC_FIELDS}
    return {
        "pixel_accuracy": metric_float(payload.get("pixel_accuracy")),
        "mean_iou": metric_float(payload.get("mean_iou")),
        "iou_background": list_metric_value(payload.get("iou_per_class"), 0),
        "iou_no_damage": list_metric_value(payload.get("iou_per_class"), 1),
        "iou_damaged": list_metric_value(payload.get("iou_per_class"), 2),
        "precision_damaged": list_metric_value(payload.get("precision_per_class"), 2),
        "recall_damaged": list_metric_value(payload.get("recall_per_class"), 2),
        "f1_damaged": list_metric_value(payload.get("f1_per_class"), 2),
        "average_loss": metric_float(payload.get("average_loss")),
    }


def build_row(
    config_row: dict[str, str],
    config_path: Path,
    row_index: int,
    predictions_dir: Path,
    checkpoints_dir: Path,
) -> dict[str, object] | None:
    experiment = get_required(config_row, "experiment", config_path, row_index)
    metrics_path = predictions_dir / f"{experiment}_test_metrics.json"
    if not metrics_path.exists():
        print(f"Skipping missing metrics: {metrics_path}")
        return None
    payload = load_json(metrics_path)
    if payload is None:
        return None

    row: dict[str, object] = {
        "experiment": experiment,
        "model": get_required(config_row, "model", config_path, row_index),
        "split": get_required(config_row, "split", config_path, row_index),
        "image_size": get_required(config_row, "image_size", config_path, row_index),
        "batch_size": get_required(config_row, "batch_size", config_path, row_index),
        "epochs": get_required(config_row, "epochs", config_path, row_index),
        "loss": get_required(config_row, "loss", config_path, row_index),
        "class_weights": get_required(config_row, "class_weights", config_path, row_index),
        "lr": get_required(config_row, "lr", config_path, row_index),
        "augment_mode": get_required(config_row, "augment_mode", config_path, row_index),
        "sampler": get_required(config_row, "sampler", config_path, row_index),
        "damage_sampling_alpha": get_required(
            config_row,
            "damage_sampling_alpha",
            config_path,
            row_index,
        ),
        "checkpoint": str(checkpoints_dir / experiment / "best_damage_arch.pt"),
        "metrics_json": str(metrics_path),
    }
    row.update(extract_metrics(payload))
    return row


def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        metric_float(row.get("f1_damaged")) or -1.0,
        metric_float(row.get("iou_damaged")) or -1.0,
        metric_float(row.get("mean_iou")) or -1.0,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "split",
        "image_size",
        "batch_size",
        "epochs",
        "loss",
        "class_weights",
        "lr",
        "augment_mode",
        "sampler",
        "damage_sampling_alpha",
        *METRIC_FIELDS,
        "checkpoint",
        "metrics_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("CrisisMap AI - Long Damage Finalists Summary")
    print("=" * 45)
    print(f"Rows with metrics: {len(rows)}")
    print(
        f"Reference {UNET_CHAMPION_TTA_D4['label']}: "
        f"mean IoU={UNET_CHAMPION_TTA_D4['mean_iou']:.6f}, "
        f"IoU damaged={UNET_CHAMPION_TTA_D4['iou_damaged']:.6f}, "
        f"F1 damaged={UNET_CHAMPION_TTA_D4['f1_damaged']:.6f}"
    )
    for index, row in enumerate(rows, start=1):
        print(
            f"{index}. {row.get('experiment')} | "
            f"mean IoU={format_metric(row.get('mean_iou'))} | "
            f"IoU damaged={format_metric(row.get('iou_damaged'))} | "
            f"F1 damaged={format_metric(row.get('f1_damaged'))}"
        )


def format_metric(value: object) -> str:
    parsed = metric_float(value)
    return "nan" if parsed is None else f"{parsed:.6f}"


def main() -> int:
    args = parse_args()
    try:
        config_rows = read_config(args.config)
        rows = [
            row
            for index, config_row in enumerate(config_rows, start=2)
            if (
                row := build_row(
                    config_row,
                    args.config,
                    index,
                    args.predictions_dir,
                    args.checkpoints_dir,
                )
            )
            is not None
        ]
        rows.sort(key=sort_key, reverse=True)
        write_csv(args.output, rows)
    except SummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_summary(rows)
    print(f"Saved summary CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

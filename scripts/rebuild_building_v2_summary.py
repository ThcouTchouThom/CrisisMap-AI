#!/usr/bin/env python
"""Rebuild the Building v2 campaign summary from compact metrics files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/building_v2_sweep.csv")
DEFAULT_OUTPUT = Path("outputs/predictions/building_v2_sweep_summary.csv")
DEFAULT_PRED_DIR = Path("outputs/predictions/building_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Building v2 summary CSV.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_config(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def metrics_by_threshold(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = payload.get("metrics_by_threshold")
    if isinstance(metrics, dict):
        return {
            str(key): value
            for key, value in metrics.items()
            if isinstance(value, dict)
        }
    metric = payload.get("metrics")
    if isinstance(metric, dict):
        return {"default": metric}
    return {}


def best_threshold(metrics: dict[str, dict[str, Any]], key: str) -> tuple[str | None, float | None]:
    best_key = None
    best_value = None
    for threshold, values in metrics.items():
        try:
            value = float(values.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if best_value is None or value > best_value:
            best_key = threshold
            best_value = value
    return best_key, best_value


def make_row(config: dict[str, str], pred_dir: Path) -> dict[str, Any]:
    experiment = config["experiment"]
    metrics_path = pred_dir / f"{experiment}_test_metrics.json"
    payload = read_json(metrics_path)
    base = {
        "experiment": experiment,
        "model": config.get("model"),
        "train_mode": config.get("train_mode"),
        "crop_size": config.get("crop_size"),
        "loss": config.get("loss"),
        "augment_mode": config.get("augment_mode"),
        "sampler": config.get("sampler"),
        "sampler_alpha": config.get("sampler_alpha"),
        "rare_building_crop_alpha": config.get("rare_building_crop_alpha"),
        "lr": config.get("lr"),
        "epochs": config.get("epochs"),
        "checkpoint": f"outputs/checkpoints/{experiment}/best_building.pt",
        "test_metrics_json": str(metrics_path),
        "complete": bool(payload),
    }
    if payload is None:
        return base
    metrics = metrics_by_threshold(payload)
    iou_threshold, iou_value = best_threshold(metrics, "building_iou")
    f1_threshold, f1_value = best_threshold(metrics, "building_f1")
    recall_threshold, recall_value = best_threshold(metrics, "building_recall")
    preferred = metrics.get(iou_threshold or "", {})
    object_metrics = preferred.get("object_metrics", {})
    if not isinstance(object_metrics, dict):
        object_metrics = {}
    return {
        **base,
        "best_iou_threshold": iou_threshold,
        "best_building_iou": iou_value,
        "best_f1_threshold": f1_threshold,
        "best_building_f1": f1_value,
        "best_recall_threshold": recall_threshold,
        "best_building_recall": recall_value,
        "building_precision_at_best_iou": preferred.get("building_precision"),
        "building_recall_at_best_iou": preferred.get("building_recall"),
        "building_f1_at_best_iou": preferred.get("building_f1"),
        "mean_iou_at_best_iou": preferred.get("mean_iou"),
        "object_precision_at_best_iou": object_metrics.get("object_precision"),
        "object_recall_at_best_iou": object_metrics.get("object_recall"),
    }


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    def value(name: str) -> float:
        try:
            return float(row.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return (
        value("best_building_iou"),
        value("best_building_f1"),
        value("best_building_recall"),
    )


def main() -> None:
    args = parse_args()
    configs = read_config(args.config)
    rows = [make_row(config, args.pred_dir) for config in configs]
    rows.sort(key=sort_key, reverse=True)
    fieldnames = [
        "experiment",
        "model",
        "train_mode",
        "crop_size",
        "loss",
        "augment_mode",
        "sampler",
        "sampler_alpha",
        "rare_building_crop_alpha",
        "lr",
        "epochs",
        "complete",
        "best_iou_threshold",
        "best_building_iou",
        "best_f1_threshold",
        "best_building_f1",
        "best_recall_threshold",
        "best_building_recall",
        "building_precision_at_best_iou",
        "building_recall_at_best_iou",
        "building_f1_at_best_iou",
        "mean_iou_at_best_iou",
        "object_precision_at_best_iou",
        "object_recall_at_best_iou",
        "checkpoint",
        "test_metrics_json",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    complete_count = sum(1 for row in rows if row.get("complete"))
    print(f"Wrote {args.output} with {complete_count}/{len(rows)} completed rows.")


if __name__ == "__main__":
    main()

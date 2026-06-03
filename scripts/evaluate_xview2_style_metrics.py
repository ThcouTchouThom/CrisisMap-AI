#!/usr/bin/env python
"""Compute mask-based xView2-style metrics from exported PNG masks.

This script evaluates the files produced by scripts/export_xview2_format.py.
It implements a lightweight xView2-style mask score:

score = 0.3 * localization_f1 + 0.7 * damage_f1

For current 3-class CrisisMap AI outputs, the score is reported as
binary_damage_xview2_like_score because the project collapses xView2 damage
levels into a single damaged class. It is not comparable to the official
5-class xView2 score or leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


THREE_CLASS_DAMAGE_CLASSES = [2]
FIVE_CLASS_DAMAGE_CLASSES = [1, 2, 3, 4]


class XView2MetricError(Exception):
    """Raised when exported xView2-style masks cannot be evaluated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exported xView2-style localization and damage masks."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--target-mode", choices=["3-class", "5-class"], default="3-class")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--per-sample-csv", type=Path, default=None)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise XView2MetricError(f"Missing mask file: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mask.shape != (1024, 1024):
        raise XView2MetricError(f"Expected 1024x1024 mask, got {mask.shape}: {path}")
    return mask


def discover_indices(input_dir: Path, prefix: str) -> list[str]:
    pattern = f"{prefix}_localization_*_prediction.png"
    indices = []
    for path in sorted(input_dir.glob(pattern)):
        name = path.name
        start = len(f"{prefix}_localization_")
        end = -len("_prediction.png")
        indices.append(name[start:end])
    if not indices:
        raise XView2MetricError(f"No prediction files found with pattern: {pattern}")
    return indices


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def binary_counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    return tp, fp, fn


def class_counts(pred: np.ndarray, target: np.ndarray, class_id: int) -> tuple[int, int, int]:
    pred_class = pred == class_id
    target_class = target == class_id
    tp = int(np.logical_and(pred_class, target_class).sum())
    fp = int(np.logical_and(pred_class, ~target_class).sum())
    fn = int(np.logical_and(~pred_class, target_class).sum())
    return tp, fp, fn


def evaluate_sample(
    input_dir: Path,
    prefix: str,
    index: str,
    damage_classes: list[int],
) -> dict[str, Any]:
    loc_pred = load_mask(input_dir / f"{prefix}_localization_{index}_prediction.png")
    loc_target = load_mask(input_dir / f"{prefix}_localization_{index}_target.png")
    damage_pred = load_mask(input_dir / f"{prefix}_damage_{index}_prediction.png")
    damage_target = load_mask(input_dir / f"{prefix}_damage_{index}_target.png")

    loc_tp, loc_fp, loc_fn = binary_counts(loc_pred > 0, loc_target > 0)
    _, _, loc_f1 = f1_from_counts(loc_tp, loc_fp, loc_fn)

    class_metrics = {}
    damage_f1_values = []
    for class_id in damage_classes:
        tp, fp, fn = class_counts(damage_pred, damage_target, class_id)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        class_metrics[str(class_id)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        damage_f1_values.append(f1)
    damage_f1 = float(np.mean(damage_f1_values)) if damage_f1_values else 0.0
    return {
        "index": index,
        "localization_tp": loc_tp,
        "localization_fp": loc_fp,
        "localization_fn": loc_fn,
        "localization_f1": loc_f1,
        "damage_f1": damage_f1,
        "damage_class_metrics": class_metrics,
    }


def aggregate_metrics(samples: list[dict[str, Any]], damage_classes: list[int]) -> dict[str, Any]:
    loc_tp = sum(int(row["localization_tp"]) for row in samples)
    loc_fp = sum(int(row["localization_fp"]) for row in samples)
    loc_fn = sum(int(row["localization_fn"]) for row in samples)
    loc_precision, loc_recall, loc_f1 = f1_from_counts(loc_tp, loc_fp, loc_fn)

    per_class = {}
    damage_f1_values = []
    for class_id in damage_classes:
        key = str(class_id)
        tp = sum(int(row["damage_class_metrics"][key]["tp"]) for row in samples)
        fp = sum(int(row["damage_class_metrics"][key]["fp"]) for row in samples)
        fn = sum(int(row["damage_class_metrics"][key]["fn"]) for row in samples)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        per_class[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        damage_f1_values.append(f1)

    damage_f1 = float(np.mean(damage_f1_values)) if damage_f1_values else 0.0
    weighted_score = 0.3 * loc_f1 + 0.7 * damage_f1
    return {
        "num_samples": len(samples),
        "localization_precision": loc_precision,
        "localization_recall": loc_recall,
        "localization_f1": loc_f1,
        "damage_f1": damage_f1,
        "damage_classes": damage_classes,
        "damage_per_class": per_class,
        "weighted_score": weighted_score,
    }


def write_summary_csv(path: Path, target_mode: str, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_name = (
        "official_style_weighted_score"
        if target_mode == "5-class"
        else "binary_damage_xview2_like_score"
    )
    row = {
        "target_mode": target_mode,
        "num_samples": summary["num_samples"],
        "localization_precision": summary["localization_precision"],
        "localization_recall": summary["localization_recall"],
        "localization_f1": summary["localization_f1"],
        "damage_f1": summary["damage_f1"],
        score_name: summary["weighted_score"],
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_per_sample_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "localization_f1", "damage_f1"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    damage_classes = (
        FIVE_CLASS_DAMAGE_CLASSES if args.target_mode == "5-class" else THREE_CLASS_DAMAGE_CLASSES
    )
    indices = discover_indices(input_dir, args.prefix)
    samples = [
        evaluate_sample(input_dir, args.prefix, index, damage_classes)
        for index in indices
    ]
    summary = aggregate_metrics(samples, damage_classes)
    score_name = (
        "official_style_weighted_score"
        if args.target_mode == "5-class"
        else "binary_damage_xview2_like_score"
    )
    payload = {
        "format": "mask_based_xview2_style_metrics",
        "target_mode": args.target_mode,
        "important_note": (
            "The 3-class score is xView2-like only. It is not comparable to the "
            "official 5-class xView2 score or leaderboard."
            if args.target_mode == "3-class"
            else "This is an official-style weighted mask score, not the original polygon scorer."
        ),
        "score_formula": "0.3 * localization_f1 + 0.7 * damage_f1",
        score_name: summary["weighted_score"],
        "summary": summary,
        "samples": samples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.output_csv:
        write_summary_csv(args.output_csv, args.target_mode, summary)
    if args.per_sample_csv:
        write_per_sample_csv(args.per_sample_csv, samples)

    print(f"Evaluated {summary['num_samples']} samples from {input_dir}")
    print(f"localization_f1 = {summary['localization_f1']:.6f}")
    print(f"damage_f1 = {summary['damage_f1']:.6f}")
    print(f"{score_name} = {summary['weighted_score']:.6f}")
    if args.target_mode == "3-class":
        print("NOTE: 3-class score is xView2-like only, not official xView2.")


if __name__ == "__main__":
    main()

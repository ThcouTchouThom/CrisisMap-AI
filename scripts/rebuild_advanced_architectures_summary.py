#!/usr/bin/env python
"""Rebuild summaries for advanced damage architecture campaigns.

The advanced evaluators do not all serialize metrics with the exact same JSON
shape. This script normalizes the known formats into one global CSV and one CSV
per campaign group.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

GROUPS = {
    "multihead_damage": {
        "directory": PREDICTIONS_DIR / "multihead_damage",
        "fallback_patterns": ["multihead_damage*.json", "mh_*.json"],
        "summary": PREDICTIONS_DIR / "multihead_damage_summary.csv",
    },
    "xview2_strong_baseline_v2": {
        "directory": PREDICTIONS_DIR / "xview2_strong_baseline_v2",
        "fallback_patterns": ["xview2_strong_baseline_v2*.json", "xv2*_v2*.json"],
        "summary": PREDICTIONS_DIR / "xview2_strong_baseline_v2_summary.csv",
    },
    "multitemporal_fusion_v2": {
        "directory": PREDICTIONS_DIR / "multitemporal_fusion_v2",
        "fallback_patterns": ["multitemporal_fusion_v2*.json", "mtf*_v2*.json"],
        "summary": PREDICTIONS_DIR / "multitemporal_fusion_v2_summary.csv",
    },
}

GLOBAL_SUMMARY = PREDICTIONS_DIR / "advanced_architectures_summary.csv"
DAMAGED_CLASS_ALIASES = {"damaged", "damage", "destroyed_or_damaged", "class_2", "2"}

REFERENCES = [
    {
        "name": "U-Net + TTA d4",
        "f1_damaged": 0.631300,
        "iou_damaged": 0.461240,
        "mean_iou": 0.681574,
    },
    {
        "name": "Actuel champion damage finalist",
        "f1_damaged": 0.678801,
        "iou_damaged": 0.513776,
        "mean_iou": 0.707285,
    },
]

FIELDNAMES = [
    "group",
    "experiment",
    "json_path",
    "model",
    "target_mode",
    "label_mode",
    "image_size",
    "num_samples",
    "split_csv",
    "checkpoint",
    "mean_iou",
    "iou_damaged",
    "precision_damaged",
    "recall_damaged",
    "f1_damaged",
    "pixel_accuracy",
    "constrained_mean_iou",
    "constrained_iou_damaged",
    "constrained_f1_damaged",
    "building_iou",
    "building_f1",
    "building_precision",
    "building_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--output", type=Path, default=GLOBAL_SUMMARY)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"WARNING: ignoring non-object JSON: {path}")
        return None
    return data


def as_number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def normalized_dict(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {normalize_key(k): v for k, v in data.items()}


def get_first(data: dict[str, Any], keys: Iterable[str]) -> Any:
    norm = normalized_dict(data)
    for key in keys:
        normalized = normalize_key(key)
        if normalized in norm:
            return norm[normalized]
    return None


def direct_metric(block: dict[str, Any], metric: str) -> float | int | None:
    candidates = [
        metric,
        f"test_{metric}",
        f"damage_{metric}",
        f"test_damage_{metric}",
    ]
    return as_number(get_first(block, candidates))


def damaged_index_from_labels(labels: Any) -> int:
    if isinstance(labels, dict):
        for key, value in labels.items():
            if normalize_key(value) in DAMAGED_CLASS_ALIASES:
                try:
                    return int(key)
                except (TypeError, ValueError):
                    continue
    if isinstance(labels, list):
        for index, value in enumerate(labels):
            if normalize_key(value) in DAMAGED_CLASS_ALIASES:
                return index
    return 2


def per_class_metric(block: dict[str, Any], metric: str) -> float | int | None:
    labels = (
        get_first(block, ["class_labels", "labels", "classes", "class_names"])
        or ["background", "no_damage", "damaged"]
    )
    damaged_index = damaged_index_from_labels(labels)
    candidates = [
        f"{metric}_per_class",
        f"per_class_{metric}",
        f"{metric}s_per_class",
        f"class_{metric}",
    ]
    values = get_first(block, candidates)
    if isinstance(values, list) and 0 <= damaged_index < len(values):
        return as_number(values[damaged_index])
    if isinstance(values, dict):
        norm = normalized_dict(values)
        for key in (
            str(damaged_index),
            "damaged",
            "damage",
            "class_2",
            "2",
        ):
            if normalize_key(key) in norm:
                return as_number(norm[normalize_key(key)])
    return None


def find_nested_metric(data: Any, metric: str, max_depth: int = 4) -> float | int | None:
    """Find the first direct metric in nested dictionaries/lists."""
    if max_depth < 0:
        return None
    if isinstance(data, dict):
        value = direct_metric(data, metric)
        if value is not None:
            return value
        for child in data.values():
            found = find_nested_metric(child, metric, max_depth - 1)
            if found is not None:
                return found
    elif isinstance(data, list):
        for child in data:
            found = find_nested_metric(child, metric, max_depth - 1)
            if found is not None:
                return found
    return None


def extract_metric(data: dict[str, Any], metric: str, blocks: list[dict[str, Any]]) -> float | int | None:
    for block in blocks:
        value = direct_metric(block, metric)
        if value is not None:
            return value
    per_class_name = metric.removesuffix("_damaged")
    for block in blocks:
        value = per_class_metric(block, per_class_name)
        if value is not None:
            return value
    return find_nested_metric(data, metric)


def metric_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for key in (
        "summary_row",
        "metrics",
        "damage_metrics",
        "test_metrics",
        "summary_metrics",
        "best_val_metrics",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            blocks.append(value)
            if isinstance(value.get("damage_metrics"), dict):
                blocks.append(value["damage_metrics"])
            if isinstance(value.get("summary_metrics"), dict):
                blocks.append(value["summary_metrics"])
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in ("damage_metrics", "constrained_damage_metrics"):
            if isinstance(metrics.get(key), dict):
                blocks.append(metrics[key])
    blocks.append(payload)
    return blocks


def infer_experiment(path: Path, payload: dict[str, Any]) -> str:
    for block in (payload.get("summary_row"), payload.get("config"), payload):
        if isinstance(block, dict):
            value = get_first(block, ["experiment", "experiment_name", "run_name", "name"])
            if value:
                return str(value)
    return path.stem


def collect_paths(group_name: str, group: dict[str, Any], predictions_dir: Path) -> list[Path]:
    directory = group["directory"]
    if not directory.is_absolute():
        directory = predictions_dir / directory
    paths: set[Path] = set()
    if directory.exists():
        paths.update(p for p in directory.glob("*.json") if p.is_file())
    for pattern in group["fallback_patterns"]:
        paths.update(p for p in predictions_dir.glob(pattern) if p.is_file())
    return sorted(paths)


def row_from_payload(group_name: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    blocks = metric_blocks(payload)
    summary = payload.get("summary_row") if isinstance(payload.get("summary_row"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    constrained = metrics.get("constrained_damage_metrics") if isinstance(metrics.get("constrained_damage_metrics"), dict) else {}
    building = metrics.get("building_metrics") if isinstance(metrics.get("building_metrics"), dict) else {}
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}

    row = {
        "group": group_name,
        "experiment": infer_experiment(path, payload),
        "json_path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "model": get_first(summary, ["model"]) or get_first(config, ["model"]),
        "target_mode": get_first(summary, ["target_mode"]) or get_first(config, ["target_mode"]),
        "label_mode": get_first(summary, ["label_mode"]) or get_first(config, ["label_mode"]),
        "image_size": get_first(summary, ["image_size"]) or get_first(config, ["image_size"]),
        "num_samples": payload.get("num_samples"),
        "split_csv": payload.get("split_csv"),
        "checkpoint": payload.get("checkpoint"),
        "mean_iou": extract_metric(payload, "mean_iou", blocks),
        "iou_damaged": extract_metric(payload, "iou_damaged", blocks),
        "precision_damaged": extract_metric(payload, "precision_damaged", blocks),
        "recall_damaged": extract_metric(payload, "recall_damaged", blocks),
        "f1_damaged": extract_metric(payload, "f1_damaged", blocks),
        "pixel_accuracy": extract_metric(payload, "pixel_accuracy", blocks),
        "constrained_mean_iou": direct_metric(constrained, "mean_iou")
        or as_number(get_first(summary, ["constrained_mean_iou"])),
        "constrained_iou_damaged": direct_metric(constrained, "iou_damaged")
        or as_number(get_first(summary, ["constrained_iou_damaged"])),
        "constrained_f1_damaged": direct_metric(constrained, "f1_damaged")
        or as_number(get_first(summary, ["constrained_f1_damaged"])),
        "building_iou": direct_metric(building, "building_iou")
        or as_number(get_first(summary, ["building_iou"])),
        "building_f1": direct_metric(building, "building_f1")
        or as_number(get_first(summary, ["building_f1"])),
        "building_precision": direct_metric(building, "building_precision")
        or as_number(get_first(summary, ["building_precision"])),
        "building_recall": direct_metric(building, "building_recall")
        or as_number(get_first(summary, ["building_recall"])),
    }
    return {key: row.get(key) for key in FIELDNAMES}


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("f1_damaged") or -1.0),
        float(row.get("iou_damaged") or -1.0),
        float(row.get("mean_iou") or -1.0),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_metric(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return "NA"
    return f"{float(number):.6f}"


def print_references() -> None:
    print("References:")
    for ref in REFERENCES:
        print(
            f"- {ref['name']}: "
            f"F1 damaged={ref['f1_damaged']:.6f}, "
            f"IoU damaged={ref['iou_damaged']:.6f}, "
            f"mean IoU={ref['mean_iou']:.6f}"
        )
    print()


def print_rankings(rows: list[dict[str, Any]]) -> None:
    print_references()
    if not rows:
        print("No advanced architecture JSON metrics found.")
        return
    ranked = sorted(rows, key=sort_key, reverse=True)
    print("Top advanced architecture runs ranked by F1 damaged, IoU damaged, mean IoU:")
    for index, row in enumerate(ranked[:20], start=1):
        print(
            f"{index:02d}. [{row.get('group')}] {row.get('experiment')} | "
            f"F1={format_metric(row.get('f1_damaged'))} "
            f"IoU_dmg={format_metric(row.get('iou_damaged'))} "
            f"mIoU={format_metric(row.get('mean_iou'))} "
            f"precision={format_metric(row.get('precision_damaged'))} "
            f"recall={format_metric(row.get('recall_damaged'))}"
        )


def main() -> None:
    args = parse_args()
    predictions_dir = args.predictions_dir
    all_rows: list[dict[str, Any]] = []

    for group_name, group in GROUPS.items():
        group = dict(group)
        group["directory"] = predictions_dir / group["directory"].name
        group["summary"] = predictions_dir / group["summary"].name
        rows: list[dict[str, Any]] = []
        paths = collect_paths(group_name, group, predictions_dir)
        print(f"{group_name}: found {len(paths)} JSON file(s)")
        for path in paths:
            payload = load_json(path)
            if payload is None:
                continue
            rows.append(row_from_payload(group_name, path, payload))
        rows = sorted(rows, key=sort_key, reverse=True)
        write_csv(group["summary"], rows)
        print(f"Wrote {group['summary']} ({len(rows)} row(s))")
        all_rows.extend(rows)

    all_rows = sorted(all_rows, key=sort_key, reverse=True)
    write_csv(args.output, all_rows)
    print(f"Wrote {args.output} ({len(all_rows)} row(s))")
    print()
    print_rankings(all_rows)


if __name__ == "__main__":
    main()

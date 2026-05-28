"""Rebuild a compact summary for damage architecture experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_OUTPUT = DEFAULT_PREDICTIONS_DIR / "damage_arch_summary.csv"
UNET_CHAMPION_RAW = {
    "label": "U-Net champion raw",
    "mean_iou": 0.676606,
    "iou_damaged": 0.446431,
    "f1_damaged": 0.617286,
}
UNET_CHAMPION_TTA_D4 = {
    "label": "U-Net champion TTA d4",
    "mean_iou": 0.681574,
    "iou_damaged": 0.461240,
    "f1_damaged": 0.631300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild damage architecture summary CSV.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Experiment filename prefix, for example damage_arch_v1 or damage_arch_v2.",
    )
    return parser.parse_args()


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


def load_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read {path}: {exc}", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


def row_from_payload(path: Path, payload: dict[str, object]) -> dict[str, object]:
    summary_row = payload.get("summary_row")
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}

    row: dict[str, object] = {
        "experiment": path.name.removesuffix("_test_metrics.json"),
        "metrics_json": str(path),
        "rank_f1_iou": "",
        "rank_iou_f1": "",
    }
    if isinstance(summary_row, dict):
        row.update(summary_row)
    else:
        row.update(
            {
                "model": config.get("model"),
                "checkpoint": config.get("checkpoint"),
                "split_csv": config.get("split_csv"),
                "image_size": config.get("image_size"),
                "batch_size": config.get("batch_size"),
                "target_mode": config.get("target_mode"),
                "pixel_accuracy": metric_float(payload.get("pixel_accuracy")),
                "mean_iou": metric_float(payload.get("mean_iou")),
                "iou_background": list_metric_value(payload.get("iou_per_class"), 0),
                "iou_no_damage": list_metric_value(payload.get("iou_per_class"), 1),
                "iou_damaged": list_metric_value(payload.get("iou_per_class"), 2),
                "precision_damaged": list_metric_value(
                    payload.get("precision_per_class"),
                    2,
                ),
                "recall_damaged": list_metric_value(payload.get("recall_per_class"), 2),
                "f1_damaged": list_metric_value(payload.get("f1_per_class"), 2),
            }
        )

    row["model"] = row.get("model") or config.get("model")
    row["checkpoint"] = row.get("checkpoint") or config.get("checkpoint")
    row["split_csv"] = row.get("split_csv") or config.get("split_csv")
    row["image_size"] = row.get("image_size") or config.get("image_size")
    row["batch_size"] = row.get("batch_size") or config.get("batch_size")
    row["target_mode"] = row.get("target_mode") or config.get("target_mode")
    return row


def discover_metric_files(predictions_dir: Path, prefixes: list[str]) -> list[Path]:
    paths: list[Path] = []
    for prefix in prefixes:
        paths.extend(predictions_dir.glob(f"{prefix}*_test_metrics.json"))
    return sorted(set(paths))


def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        metric_float(row.get("f1_damaged")) or -1.0,
        metric_float(row.get("iou_damaged")) or -1.0,
        metric_float(row.get("mean_iou")) or -1.0,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_f1_iou",
        "rank_iou_f1",
        "experiment",
        "model",
        "split_csv",
        "image_size",
        "batch_size",
        "target_mode",
        "pixel_accuracy",
        "mean_iou",
        "iou_background",
        "iou_no_damage",
        "iou_damaged",
        "precision_damaged",
        "recall_damaged",
        "f1_damaged",
        "average_loss",
        "checkpoint",
        "metrics_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("CrisisMap AI - Damage Architecture Summary")
    print("=" * 43)
    print(f"Rows found: {len(rows)}")
    print()
    print("References")
    for reference in [UNET_CHAMPION_RAW, UNET_CHAMPION_TTA_D4]:
        print(
            f"- {reference['label']}: "
            f"mean IoU={reference['mean_iou']:.6f}, "
            f"IoU damaged={reference['iou_damaged']:.6f}, "
            f"F1 damaged={reference['f1_damaged']:.6f}"
        )
    print()
    print("Top architecture rows by F1 damaged")
    for row in rows[:10]:
        print(
            f"{row.get('rank_f1_iou')}. {row.get('experiment')} | "
            f"model={row.get('model')} | "
            f"mean IoU={format_metric(row.get('mean_iou'))} | "
            f"IoU damaged={format_metric(row.get('iou_damaged'))} | "
            f"F1 damaged={format_metric(row.get('f1_damaged'))}"
        )


def format_metric(value: object) -> str:
    parsed = metric_float(value)
    return "nan" if parsed is None else f"{parsed:.6f}"


def main() -> int:
    args = parse_args()
    prefixes = args.prefix or ["damage_arch_v1", "damage_arch_v2"]
    metric_files = discover_metric_files(args.predictions_dir, prefixes)
    rows: list[dict[str, object]] = []
    for path in metric_files:
        payload = load_json(path)
        if payload is None:
            continue
        rows.append(row_from_payload(path, payload))

    rows_by_f1 = sorted(rows, key=sort_key, reverse=True)
    rows_by_iou = sorted(
        rows,
        key=lambda row: (
            metric_float(row.get("iou_damaged")) or -1.0,
            metric_float(row.get("f1_damaged")) or -1.0,
            metric_float(row.get("mean_iou")) or -1.0,
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows_by_f1, start=1):
        row["rank_f1_iou"] = rank
    for rank, row in enumerate(rows_by_iou, start=1):
        row["rank_iou_f1"] = rank

    write_csv(args.output, rows_by_f1)
    print_summary(rows_by_f1)
    print()
    print(f"Saved summary CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

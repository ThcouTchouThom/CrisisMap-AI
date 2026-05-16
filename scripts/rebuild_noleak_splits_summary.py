"""Rebuild the no-leak 1024 split-sweep summary from metrics JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_SPLITS = [
    "splits_noleak_full_train",
    "splits_noleak_full_balanced_damage",
    "splits_noleak_match_hist1500",
    "splits_noleak_match_hist_all",
    "splits_noleak_dmg001_v2",
    "splits_noleak_damage_heavy",
    "splits_noleak_disaster_stratified_150",
    "splits_noleak_disaster_stratified_200",
    "splits_noleak_building_rich_002",
    "splits_noleak_building_rich_003",
]
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
    """Raised when the summary cannot be rebuilt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the no-leak split-sweep summary CSV."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Directory containing *_test_metrics.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions/unet_1024_noleak_splits_100epochs_summary.csv"),
        help="Summary CSV path to write.",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--image-size", default="1024")
    parser.add_argument("--batch-size", default="2")
    parser.add_argument("--loss", default="ce-dice")
    parser.add_argument("--class-weights", default="0.05 1.0 4.0")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--augment-mode", default="none")
    parser.add_argument("--sampler", default="none")
    return parser.parse_args()


def class_value(metrics: dict[str, object], key: str, index: int) -> object:
    values = metrics.get(key, [])
    if not isinstance(values, list) or len(values) <= index:
        return None
    return values[index]


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


def metrics_path_for_split(args: argparse.Namespace, split: str) -> Path:
    experiment = (
        f"unet_{args.image_size}_{split}_{args.epochs}epochs_"
        f"aug-{args.augment_mode}_sampler-{args.sampler}"
    )
    return args.predictions_dir / f"{experiment}_test_metrics.json"


def split_from_metrics_path(args: argparse.Namespace, path: Path) -> str | None:
    prefix = f"unet_{args.image_size}_"
    suffix = f"_{args.epochs}epochs_aug-{args.augment_mode}_sampler-{args.sampler}_test_metrics.json"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def build_row(args: argparse.Namespace, split: str, metrics_path: Path) -> dict[str, object]:
    experiment = metrics_path.name.removesuffix("_test_metrics.json")
    metrics = load_metrics(metrics_path)
    checkpoint_path = Path("outputs/checkpoints") / experiment / "best_unet.pt"
    return {
        "experiment": experiment,
        "split": split,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "class_weights": args.class_weights,
        "lr": args.lr,
        "epochs": args.epochs,
        "augment_mode": args.augment_mode,
        "sampler": args.sampler,
        "test_pixel_accuracy": metrics.get("pixel_accuracy"),
        "test_mean_iou": metrics.get("mean_iou"),
        "test_iou_background": class_value(metrics, "iou_per_class", 0),
        "test_iou_no_damage": class_value(metrics, "iou_per_class", 1),
        "test_iou_damaged": class_value(metrics, "iou_per_class", 2),
        "test_precision_damaged": class_value(metrics, "precision_per_class", 2),
        "test_recall_damaged": class_value(metrics, "recall_per_class", 2),
        "test_f1_damaged": class_value(metrics, "f1_per_class", 2),
        "checkpoint": str(checkpoint_path),
        "test_metrics_json": str(metrics_path),
    }


def sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(row["test_iou_damaged"] if row["test_iou_damaged"] is not None else -1),
        float(row["test_f1_damaged"] if row["test_f1_damaged"] is not None else -1),
        float(row["test_mean_iou"] if row["test_mean_iou"] is not None else -1),
    )


def rebuild_summary(args: argparse.Namespace) -> None:
    predictions_dir = args.predictions_dir.expanduser()
    if not predictions_dir.exists():
        raise SummaryError(f"Predictions directory does not exist: {predictions_dir}")

    splits = args.splits if args.splits is not None else DEFAULT_SPLITS
    pattern = (
        f"unet_{args.image_size}_splits_noleak_*_{args.epochs}epochs_"
        f"aug-{args.augment_mode}_sampler-{args.sampler}_test_metrics.json"
    )
    discovered = {}
    for metrics_path in predictions_dir.glob(pattern):
        split = split_from_metrics_path(args, metrics_path)
        if split is not None:
            discovered[split] = metrics_path

    rows = []
    for split in splits:
        metrics_path = discovered.get(split, metrics_path_for_split(args, split))
        if not metrics_path.exists():
            print(f"Skipping missing metrics: {metrics_path}")
            continue
        rows.append(build_row(args, split, metrics_path))

    rows.sort(key=sort_key, reverse=True)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary CSV: {output_path}")
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

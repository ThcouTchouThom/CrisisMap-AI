"""Rebuild the no-leak augmentation/sampler campaign summary from metrics JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_SPLITS = [
    "splits_noleak_match_hist1000",
    "splits_noleak_building_rich_002",
    "splits_noleak_match_hist_all",
    "splits_noleak_dmg001_v2",
]
DEFAULT_VARIANTS = [
    ("none", "none", "4"),
    ("safe", "none", "4"),
    ("damage-aware", "none", "4"),
    ("none", "damage-simple", "4"),
    ("safe", "damage-simple", "4"),
    ("damage-aware", "damage-simple", "4"),
    ("safe", "damage-sqrt", "4"),
    ("damage-aware", "damage-sqrt", "4"),
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
    "augment_prob",
    "damage_augment_threshold",
    "sampler",
    "damage_sampling_alpha",
    "high_damage_threshold",
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
    """Raised when the augmentation/sampler summary cannot be rebuilt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the no-leak augmentation/sampler summary CSV."
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
        default=Path(
            "outputs/predictions/unet_1024_noleak_aug_sampler_100epochs_summary.csv"
        ),
        help="Summary CSV path to write.",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--image-size", default="1024")
    parser.add_argument("--batch-size", default="2")
    parser.add_argument("--loss", default="ce-dice")
    parser.add_argument("--class-weights", default="0.05 1.0 4.0")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--augment-prob", default="0.5")
    parser.add_argument("--damage-augment-threshold", default="0.001")
    parser.add_argument("--high-damage-threshold", default="0.06")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional variants as augment|sampler|alpha tokens.",
    )
    return parser.parse_args()


def split_alias(split: str) -> str:
    return split.removeprefix("splits_noleak_")


def alpha_label(alpha: str) -> str:
    try:
        value = float(alpha)
    except ValueError:
        return alpha.replace(".", "p")
    if value.is_integer():
        return str(int(value))
    return alpha.replace(".", "p")


def experiment_name(
    image_size: str,
    split: str,
    epochs: str,
    augment_mode: str,
    sampler: str,
    alpha: str,
) -> str:
    name = (
        f"unet_{image_size}_aug_noleak_{split_alias(split)}_{epochs}epochs_"
        f"aug-{augment_mode}_sampler-{sampler}"
    )
    if sampler == "damage-sqrt":
        name = f"{name}-alpha{alpha_label(alpha)}"
    return name


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


def build_row(
    args: argparse.Namespace,
    split: str,
    augment_mode: str,
    sampler: str,
    alpha: str,
    available_metrics: dict[str, Path],
) -> dict[str, object] | None:
    experiment = experiment_name(
        args.image_size,
        split,
        args.epochs,
        augment_mode,
        sampler,
        alpha,
    )
    metrics_name = f"{experiment}_test_metrics.json"
    metrics_path = available_metrics.get(metrics_name, args.predictions_dir / metrics_name)
    if not metrics_path.exists():
        print(f"Skipping missing metrics: {metrics_path}")
        return None

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
        "augment_mode": augment_mode,
        "augment_prob": args.augment_prob,
        "damage_augment_threshold": args.damage_augment_threshold,
        "sampler": sampler,
        "damage_sampling_alpha": alpha,
        "high_damage_threshold": args.high_damage_threshold,
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
    args.predictions_dir = predictions_dir

    available_metrics = {
        path.name: path
        for path in predictions_dir.glob("unet_*_aug_noleak_*_test_metrics.json")
    }
    splits = args.splits if args.splits is not None else DEFAULT_SPLITS
    variants = parse_variants(args.variants) if args.variants is not None else DEFAULT_VARIANTS
    rows = []
    for split in splits:
        for augment_mode, sampler, alpha in variants:
            row = build_row(args, split, augment_mode, sampler, alpha, available_metrics)
            if row is not None:
                rows.append(row)

    rows.sort(key=sort_key, reverse=True)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary CSV: {output_path}")
    print(f"Rows: {len(rows)}")


def parse_variants(tokens: list[str]) -> list[tuple[str, str, str]]:
    variants = []
    for token in tokens:
        parts = token.split("|")
        if len(parts) != 3:
            raise SummaryError(
                "Invalid --variants token. Expected augment|sampler|alpha, "
                f"got: {token}"
            )
        variants.append((parts[0], parts[1], parts[2]))
    return variants


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

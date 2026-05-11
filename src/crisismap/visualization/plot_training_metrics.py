"""Plot training curves from a CrisisMap AI metrics_history.json file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED_FIELDS = {
    "epoch",
    "train_loss",
    "val_loss",
    "val_mean_iou",
    "val_iou_per_class",
}


class PlotMetricsError(Exception):
    """Raised when training metrics cannot be plotted safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot U-Net training metrics.")
    parser.add_argument(
        "--metrics",
        required=True,
        type=Path,
        help="Path to metrics_history.json produced by train_unet.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where PNG figures will be saved.",
    )
    return parser.parse_args()


def load_history(metrics_path: Path) -> list[dict[str, object]]:
    metrics_path = metrics_path.expanduser().resolve()
    if not metrics_path.exists():
        raise PlotMetricsError(f"Metrics file does not exist: {metrics_path}")
    if not metrics_path.is_file():
        raise PlotMetricsError(f"Metrics path is not a file: {metrics_path}")

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
    except json.JSONDecodeError as exc:
        raise PlotMetricsError(f"Could not parse metrics JSON: {exc}") from exc
    except OSError as exc:
        raise PlotMetricsError(f"Could not read metrics file '{metrics_path}': {exc}") from exc

    if not isinstance(history, list):
        raise PlotMetricsError("Metrics history JSON must contain a list of epoch records.")
    if not history:
        raise PlotMetricsError("Metrics history is empty.")

    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise PlotMetricsError(f"Metrics row {index} is not an object.")
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise PlotMetricsError(
                f"Metrics row {index} is missing required field(s): {', '.join(missing)}"
            )
        if not isinstance(row["val_iou_per_class"], list):
            raise PlotMetricsError(f"Metrics row {index} has invalid val_iou_per_class.")

    return history


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if "raw" in {part.lower() for part in output_dir.parts}:
        raise PlotMetricsError("Refusing to write figures inside a raw data directory.")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PlotMetricsError(f"Could not create output directory '{output_dir}': {exc}") from exc
    if not output_dir.is_dir():
        raise PlotMetricsError(f"Output path is not a directory: {output_dir}")
    return output_dir


def numeric_series(history: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for index, row in enumerate(history):
        try:
            values.append(float(row[key]))
        except (TypeError, ValueError) as exc:
            raise PlotMetricsError(
                f"Metric '{key}' in row {index} is not numeric: {row[key]}"
            ) from exc
    return values


def epochs(history: list[dict[str, object]]) -> list[int]:
    values = []
    for index, row in enumerate(history):
        try:
            values.append(int(row["epoch"]))
        except (TypeError, ValueError) as exc:
            raise PlotMetricsError(
                f"Metric 'epoch' in row {index} is not an integer: {row['epoch']}"
            ) from exc
    return values


def iou_per_class(history: list[dict[str, object]]) -> list[list[float | None]]:
    class_count = max(len(row["val_iou_per_class"]) for row in history)
    curves: list[list[float | None]] = [[] for _ in range(class_count)]

    for row_index, row in enumerate(history):
        values = row["val_iou_per_class"]
        for class_index in range(class_count):
            value = values[class_index] if class_index < len(values) else None
            if value is None:
                curves[class_index].append(None)
                continue
            try:
                curves[class_index].append(float(value))
            except (TypeError, ValueError) as exc:
                raise PlotMetricsError(
                    "val_iou_per_class contains a non-numeric value "
                    f"at row {row_index}, class {class_index}: {value}"
                ) from exc

    return curves


def save_train_vs_val_loss(history: list[dict[str, object]], output_dir: Path) -> Path:
    epoch_values = epochs(history)
    train_loss = numeric_series(history, "train_loss")
    val_loss = numeric_series(history, "val_loss")

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epoch_values, train_loss, marker="o", label="Train loss")
    axis.plot(epoch_values, val_loss, marker="o", label="Validation loss")
    axis.set_title("U-Net Training vs Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(True, alpha=0.3)
    axis.legend()

    path = output_dir / "train_vs_val_loss.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_mean_iou_curve(history: list[dict[str, object]], output_dir: Path) -> Path:
    epoch_values = epochs(history)
    mean_iou = numeric_series(history, "val_mean_iou")

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epoch_values, mean_iou, marker="o", color="#2f6f9f", label="Mean IoU")
    axis.set_title("Validation Mean IoU")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean IoU")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    path = output_dir / "mean_iou_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_iou_per_class_curve(history: list[dict[str, object]], output_dir: Path) -> Path:
    epoch_values = epochs(history)
    curves = iou_per_class(history)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for class_index, values in enumerate(curves):
        axis.plot(
            epoch_values,
            values,
            marker="o",
            label=f"Class {class_index}",
        )

    axis.set_title("Validation IoU Per Class")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("IoU")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    path = output_dir / "iou_per_class_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_all(history: list[dict[str, object]], output_dir: Path) -> list[Path]:
    return [
        save_train_vs_val_loss(history, output_dir),
        save_mean_iou_curve(history, output_dir),
        save_iou_per_class_curve(history, output_dir),
    ]


def main() -> int:
    args = parse_args()
    try:
        history = load_history(args.metrics)
        output_dir = prepare_output_dir(args.output_dir)
        paths = plot_all(history, output_dir)
    except PlotMetricsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Saved training metric figures:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

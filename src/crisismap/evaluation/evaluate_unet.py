"""Evaluate a trained U-Net checkpoint on an xBD/xView2 split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.models.unet import UNet  # noqa: E402


CLASS_LABELS = {
    "3-class": ["background", "no damage", "damaged"],
    "5-class": [
        "background",
        "no damage",
        "minor damage",
        "major damage",
        "destroyed",
    ],
}


class EvaluationError(Exception):
    """Raised when model evaluation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
        help="Path to test_pairs.csv or val_pairs.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to best_unet.pt.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where metrics JSON will be saved.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--target-mode",
        choices=sorted(CLASS_LABELS),
        default="3-class",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device string, for example cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise EvaluationError("--image-size must be a positive integer.")
    if args.batch_size <= 0:
        raise EvaluationError("--batch-size must be a positive integer.")
    if args.num_workers < 0:
        raise EvaluationError("--num-workers must be non-negative.")
    if args.base_channels <= 0:
        raise EvaluationError("--base-channels must be a positive integer.")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise EvaluationError("CUDA device was requested, but CUDA is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(args: argparse.Namespace) -> XBDPairDataset:
    return XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
    )


def make_loader(
    dataset: XBDPairDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_model(args: argparse.Namespace, device: torch.device) -> UNet:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise EvaluationError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise EvaluationError(f"Checkpoint path is not a file: {checkpoint_path}")

    num_classes = len(CLASS_LABELS[args.target_mode])
    model = UNet(
        in_channels=6,
        num_classes=num_classes,
        base_channels=args.base_channels,
    ).to(device)

    try:
        checkpoint = load_checkpoint_file(checkpoint_path, device)
    except OSError as exc:
        raise EvaluationError(f"Could not read checkpoint '{checkpoint_path}': {exc}") from exc
    except RuntimeError as exc:
        raise EvaluationError(
            f"Could not load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise EvaluationError(
            "Checkpoint weights do not match this UNet configuration. "
            "Check --target-mode and --base-channels."
        ) from exc

    model.eval()
    return model


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise EvaluationError("Checkpoint is not a state_dict or model checkpoint dict.")

    cleaned = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise EvaluationError("Checkpoint does not contain any tensor weights.")
    return cleaned


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        confusion += confusion_matrix(preds, targets, num_classes)
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    metrics = metrics_from_confusion(confusion)
    metrics["average_loss"] = total_loss / max(total_samples, 1)
    return metrics


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    valid = (targets >= 0) & (targets < num_classes)
    indices = targets[valid] * num_classes + preds[valid]
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    matrix = confusion.detach().cpu().numpy().astype(np.float64)
    true_positive = np.diag(matrix)
    actual_total = matrix.sum(axis=1)
    predicted_total = matrix.sum(axis=0)
    total_pixels = matrix.sum()

    iou_denominator = actual_total + predicted_total - true_positive
    precision_denominator = predicted_total
    recall_denominator = actual_total

    iou = safe_divide(true_positive, iou_denominator)
    precision = safe_divide(true_positive, precision_denominator)
    recall = safe_divide(true_positive, recall_denominator)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)

    pixel_accuracy = float(true_positive.sum() / total_pixels) if total_pixels else 0.0
    mean_iou = float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "iou_per_class": array_to_optional_list(iou),
        "precision_per_class": array_to_optional_list(precision),
        "recall_per_class": array_to_optional_list(recall),
        "f1_per_class": array_to_optional_list(f1),
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "total_pixels": int(total_pixels),
    }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def array_to_optional_list(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values.tolist()]


def build_output_payload(
    args: argparse.Namespace,
    metrics: dict[str, object],
    dataset_size: int,
    device: torch.device,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "config": {
            "root": str(args.root),
            "split_csv": str(args.split_csv),
            "checkpoint": str(args.checkpoint),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "target_mode": args.target_mode,
            "base_channels": args.base_channels,
            "device": str(device),
        },
        "dataset_size": dataset_size,
        "num_classes": len(CLASS_LABELS[args.target_mode]),
        "class_labels": CLASS_LABELS[args.target_mode],
        "elapsed_seconds": elapsed_seconds,
        **metrics,
    }


def write_metrics_json(output: Path, payload: dict[str, object]) -> None:
    output = output.expanduser().resolve()
    if not output.parent.exists():
        raise EvaluationError(f"Output parent folder does not exist: {output.parent}")
    if not output.parent.is_dir():
        raise EvaluationError(f"Output parent path is not a directory: {output.parent}")

    try:
        with output.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
    except OSError as exc:
        raise EvaluationError(f"Could not write metrics JSON '{output}': {exc}") from exc


def print_summary(payload: dict[str, object]) -> None:
    print("CrisisMap AI - U-Net Evaluation")
    print("=" * 34)
    print(f"Device: {payload['config']['device']}")
    print(f"Dataset size: {payload['dataset_size']}")
    print(f"Average loss: {payload['average_loss']:.6f}")
    print(f"Pixel accuracy: {payload['pixel_accuracy']:.6f}")
    print(f"Mean IoU: {payload['mean_iou']:.6f}")
    print()

    print("Per-class metrics")
    labels = payload["class_labels"]
    for class_id, label in enumerate(labels):
        iou = format_metric(payload["iou_per_class"][class_id])
        precision = format_metric(payload["precision_per_class"][class_id])
        recall = format_metric(payload["recall_per_class"][class_id])
        f1 = format_metric(payload["f1_per_class"][class_id])
        print(
            f"  {class_id} = {label}: "
            f"IoU={iou}, precision={precision}, recall={recall}, F1={f1}"
        )

    print()
    print("Confusion matrix")
    for row in payload["confusion_matrix"]:
        print("  " + " ".join(str(value) for value in row))


def format_metric(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6f}"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        model = load_model(args, device)
        criterion = torch.nn.CrossEntropyLoss()

        started_at = time.time()
        metrics = evaluate(
            model,
            loader,
            criterion,
            device,
            num_classes=len(CLASS_LABELS[args.target_mode]),
        )
        payload = build_output_payload(
            args,
            metrics,
            dataset_size=len(dataset),
            device=device,
            elapsed_seconds=time.time() - started_at,
        )
        write_metrics_json(args.output, payload)
        print_summary(payload)
        print()
        print(f"Saved metrics JSON: {args.output.expanduser().resolve()}")
    except (EvaluationError, XBDDatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

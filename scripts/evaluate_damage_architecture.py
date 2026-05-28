"""Evaluate Axis 2 damage architecture checkpoints on xBD/xView2 splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import (  # noqa: E402
    CLASS_LABELS,
    confusion_matrix,
    metrics_from_confusion,
)
from crisismap.models.damage_model_factory import (  # noqa: E402
    DamageModelFactoryError,
    create_damage_model,
    damage_model_metadata,
)


CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [0, 170, 80],
        [220, 40, 40],
    ],
    dtype=np.uint8,
)


class DamageArchitectureEvaluationError(Exception):
    """Raised when damage architecture evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a damage architecture checkpoint.",
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-mode", choices=sorted(CLASS_LABELS), default="3-class")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-examples-dir", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_mode != "3-class":
        raise DamageArchitectureEvaluationError(
            "Axis 2 architecture evaluation currently expects 3-class targets."
        )
    for name in ["image_size", "batch_size", "base_channels"]:
        if int(getattr(args, name)) <= 0:
            raise DamageArchitectureEvaluationError(
                f"--{name.replace('_', '-')} must be positive."
            )
    if args.num_workers < 0:
        raise DamageArchitectureEvaluationError("--num-workers must be non-negative.")
    if args.num_examples < 0:
        raise DamageArchitectureEvaluationError("--num-examples must be non-negative.")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise DamageArchitectureEvaluationError(
                "CUDA device was requested, but CUDA is not available."
            )
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        raise DamageArchitectureEvaluationError(
            "Checkpoint is not a state_dict or model checkpoint dict."
        )

    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[str(key).removeprefix("module.")] = value
    if not cleaned:
        raise DamageArchitectureEvaluationError("Checkpoint contains no tensor weights.")
    return cleaned


def load_dataset(args: argparse.Namespace) -> XBDPairDataset:
    return XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
        augment_mode="none",
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


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise DamageArchitectureEvaluationError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise DamageArchitectureEvaluationError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    model = create_damage_model(
        args.model,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        in_channels=6,
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    try:
        model.load_state_dict(extract_state_dict(checkpoint))
    except RuntimeError as exc:
        raise DamageArchitectureEvaluationError(
            "Checkpoint weights do not match the requested architecture. "
            "Check --model, --target-mode, and --base-channels."
        ) from exc
    model.eval()
    return model


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    save_examples_dir: Path | None,
    num_examples: int,
) -> tuple[dict[str, object], int]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = 0.0
    total_samples = 0
    examples_saved = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        confusion += confusion_matrix(preds, targets, num_classes)

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

        if save_examples_dir is not None and examples_saved < num_examples:
            examples_saved = save_batch_examples(
                batch=batch,
                preds=preds.detach().cpu(),
                save_dir=save_examples_dir,
                start_index=examples_saved,
                max_examples=num_examples,
            )

    metrics = metrics_from_confusion(confusion)
    metrics["average_loss"] = total_loss / max(total_samples, 1)
    return metrics, examples_saved


def save_batch_examples(
    batch: dict[str, object],
    preds: torch.Tensor,
    save_dir: Path,
    start_index: int,
    max_examples: int,
) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    targets = batch["target"].detach().cpu()
    pair_ids = batch.get("pair_id", [])
    saved = start_index
    for item_index in range(images.shape[0]):
        if saved >= max_examples:
            break
        pair_id = (
            str(pair_ids[item_index])
            if isinstance(pair_ids, (list, tuple)) and item_index < len(pair_ids)
            else f"sample_{saved:03d}"
        )
        save_example_figure(
            image_tensor=images[item_index],
            target=targets[item_index],
            pred=preds[item_index],
            output_path=save_dir / f"{saved:03d}_{sanitize_filename(pair_id)}.png",
        )
        saved += 1
    return saved


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def tensor_rgb(image: torch.Tensor, start_channel: int) -> np.ndarray:
    rgb = image[start_channel : start_channel + 3].permute(1, 2, 0).numpy()
    return np.clip(rgb, 0.0, 1.0)


def colorize_mask(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.numpy().astype(np.int64)
    mask_np = np.clip(mask_np, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[mask_np]


def overlay_prediction(post_image: np.ndarray, pred: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    colors = colorize_mask(pred).astype(np.float32) / 255.0
    pred_np = pred.numpy()
    overlay = post_image.copy()
    building = pred_np > 0
    overlay[building] = (1.0 - alpha) * overlay[building] + alpha * colors[building]
    return np.clip(overlay, 0.0, 1.0)


def save_example_figure(
    image_tensor: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    output_path: Path,
) -> None:
    pre = tensor_rgb(image_tensor, 0)
    post = tensor_rgb(image_tensor, 3)
    panels = [
        ("Pre-disaster", pre),
        ("Post-disaster", post),
        ("Ground truth", colorize_mask(target)),
        ("Prediction", colorize_mask(pred)),
        ("Overlay", overlay_prediction(post, pred)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def metric_value(metrics: dict[str, object], key: str, class_index: int | None = None) -> float | None:
    if class_index is None:
        value = metrics.get(key)
    else:
        values = metrics.get(key, [])
        value = values[class_index] if isinstance(values, list) and len(values) > class_index else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_row(args: argparse.Namespace, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "split_csv": str(args.split_csv),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "target_mode": args.target_mode,
        "pixel_accuracy": metric_value(metrics, "pixel_accuracy"),
        "mean_iou": metric_value(metrics, "mean_iou"),
        "iou_background": metric_value(metrics, "iou_per_class", 0),
        "iou_no_damage": metric_value(metrics, "iou_per_class", 1),
        "iou_damaged": metric_value(metrics, "iou_per_class", 2),
        "precision_damaged": metric_value(metrics, "precision_per_class", 2),
        "recall_damaged": metric_value(metrics, "recall_per_class", 2),
        "f1_damaged": metric_value(metrics, "f1_per_class", 2),
        "average_loss": metric_value(metrics, "average_loss"),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def print_summary(row: dict[str, object]) -> None:
    print("CrisisMap AI - Damage Architecture Evaluation")
    print("=" * 47)
    print(f"Model: {row['model']}")
    print(f"Mean IoU: {format_metric(row['mean_iou'])}")
    print(f"IoU damaged: {format_metric(row['iou_damaged'])}")
    print(f"Precision damaged: {format_metric(row['precision_damaged'])}")
    print(f"Recall damaged: {format_metric(row['recall_damaged'])}")
    print(f"F1 damaged: {format_metric(row['f1_damaged'])}")


def format_metric(value: object) -> str:
    return "nan" if value is None else f"{float(value):.6f}"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        model = load_model(args, device)
        started_at = time.time()
        metrics, examples_saved = evaluate(
            model=model,
            loader=loader,
            device=device,
            num_classes=len(CLASS_LABELS[args.target_mode]),
            use_amp=use_amp,
            save_examples_dir=args.save_examples_dir,
            num_examples=args.num_examples,
        )
        row = build_row(args, metrics)
        payload = {
            "config": {
                "root": str(args.root),
                "split_csv": str(args.split_csv),
                "checkpoint": str(args.checkpoint),
                "model": args.model,
                "model_metadata": damage_model_metadata(args.model),
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "target_mode": args.target_mode,
                "device": str(device),
                "amp": use_amp,
                "num_workers": args.num_workers,
            },
            "dataset_size": len(dataset),
            "class_labels": CLASS_LABELS[args.target_mode],
            "elapsed_seconds": time.time() - started_at,
            "examples_saved": examples_saved,
            **metrics,
            "summary_row": row,
        }
        write_json(args.output_json, payload)
        write_csv(args.output_csv, row)
        print_summary(row)
        print()
        print(f"Saved JSON: {args.output_json}")
        print(f"Saved CSV: {args.output_csv}")
        if args.save_examples_dir is not None:
            print(f"Saved examples: {args.save_examples_dir}")
    except (
        DamageArchitectureEvaluationError,
        DamageModelFactoryError,
        XBDDatasetError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

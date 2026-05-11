"""Train a baseline U-Net on xBD/xView2 split CSVs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.models.unet import UNet  # noqa: E402


TARGET_MODES = {"3-class", "5-class"}


class TrainingError(Exception):
    """Raised when training cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline U-Net for xBD.")
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        type=Path,
        help="Path to train_pairs.csv.",
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        type=Path,
        help="Path to val_pairs.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for checkpoints and metrics_history.json.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_MODES),
        default="3-class",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device string, for example cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--image-size": args.image_size,
        "--batch-size": args.batch_size,
        "--epochs": args.epochs,
        "--base-channels": args.base_channels,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise TrainingError(f"{name} must be a positive integer.")
    if args.lr <= 0:
        raise TrainingError("--lr must be positive.")
    if args.num_workers < 0:
        raise TrainingError("--num-workers must be non-negative.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise TrainingError("--max-train-samples must be a positive integer.")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise TrainingError("--max-val-samples must be a positive integer.")


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if "raw" in {part.lower() for part in output_dir.parts}:
        raise TrainingError("Refusing to write checkpoints inside a raw data directory.")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrainingError(f"Could not create output directory '{output_dir}': {exc}") from exc
    if not output_dir.is_dir():
        raise TrainingError(f"Output path is not a directory: {output_dir}")
    return output_dir


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingError("CUDA device was requested, but CUDA is not available.")
        return device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(
    root: Path,
    split_csv: Path,
    image_size: int,
    target_mode: str,
    max_samples: int | None,
) -> torch.utils.data.Dataset:
    dataset = XBDPairDataset(
        root=root,
        split_csv=split_csv,
        image_size=image_size,
        target_mode=target_mode,
    )
    if max_samples is None:
        return dataset
    return Subset(dataset, range(min(max_samples, len(dataset))))


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def class_weights(target_mode: str, device: torch.device) -> torch.Tensor | None:
    if target_mode == "3-class":
        return torch.tensor([0.2, 1.0, 2.0], dtype=torch.float32, device=device)
    return None


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

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

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(total_samples, 1)
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
    confusion_np = confusion.detach().cpu().numpy().astype(np.float64)
    correct = np.trace(confusion_np)
    total = confusion_np.sum()
    pixel_accuracy = float(correct / total) if total else 0.0

    intersections = np.diag(confusion_np)
    unions = confusion_np.sum(axis=1) + confusion_np.sum(axis=0) - intersections
    iou_per_class = np.divide(
        intersections,
        unions,
        out=np.full_like(intersections, np.nan, dtype=np.float64),
        where=unions > 0,
    )
    mean_iou = (
        float(np.nanmean(iou_per_class))
        if not np.all(np.isnan(iou_per_class))
        else 0.0
    )

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "iou_per_class": [
            None if np.isnan(value) else float(value) for value in iou_per_class.tolist()
        ],
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, object],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def write_metrics_history(path: Path, history: list[dict[str, object]]) -> None:
    serializable_history = json.loads(json.dumps(history, default=str))
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=2)


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def print_epoch_summary(epoch_metrics: dict[str, object]) -> None:
    iou_values = epoch_metrics["val_iou_per_class"]
    iou_text = ", ".join(
        "nan" if value is None else f"{value:.4f}" for value in iou_values
    )
    print(
        f"Epoch {epoch_metrics['epoch']:03d} | "
        f"train loss {epoch_metrics['train_loss']:.4f} | "
        f"val loss {epoch_metrics['val_loss']:.4f} | "
        f"pixel acc {epoch_metrics['val_pixel_accuracy']:.4f} | "
        f"mean IoU {epoch_metrics['val_mean_iou']:.4f} | "
        f"IoU/class [{iou_text}]"
    )


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = prepare_output_dir(args.output_dir)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    num_classes = 3 if args.target_mode == "3-class" else 5
    train_dataset = make_dataset(
        args.root,
        args.train_csv,
        args.image_size,
        args.target_mode,
        args.max_train_samples,
    )
    val_dataset = make_dataset(
        args.root,
        args.val_csv,
        args.image_size,
        args.target_mode,
        args.max_val_samples,
    )
    if len(train_dataset) == 0:
        raise TrainingError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise TrainingError("Validation dataset is empty.")

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
    )

    model = UNet(
        in_channels=6,
        num_classes=num_classes,
        base_channels=args.base_channels,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(args.target_mode, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Device: {device}")
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Model parameters: {parameter_count(model):,}")

    best_mean_iou = -1.0
    history: list[dict[str, object]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            use_amp,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_iou_per_class": val_metrics["iou_per_class"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_metrics)
        print_epoch_summary(epoch_metrics)

        if epoch_metrics["val_mean_iou"] > best_mean_iou:
            best_mean_iou = float(epoch_metrics["val_mean_iou"])
            save_checkpoint(
                output_dir / "best_unet.pt",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                args,
            )

        save_checkpoint(
            output_dir / "last_unet.pt",
            model,
            optimizer,
            epoch,
            epoch_metrics,
            args,
        )
        write_metrics_history(output_dir / "metrics_history.json", history)

    print(f"Training complete in {(time.time() - start_time) / 60.0:.2f} minutes.")
    print(f"Best validation mean IoU: {best_mean_iou:.4f}")
    print(f"Outputs saved to: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (TrainingError, XBDDatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

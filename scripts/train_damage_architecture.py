"""Train Axis 2 damage segmentation architectures without changing train_unet.py."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError  # noqa: E402
from crisismap.models.damage_model_factory import (  # noqa: E402
    DamageModelFactoryError,
    create_damage_model,
    damage_model_metadata,
    supported_damage_model_names,
)
from crisismap.training.train_unet import (  # noqa: E402
    AUGMENT_MODES,
    LOSS_CHOICES,
    SAMPLER_MODES,
    TARGET_MODES,
    TrainingError,
    best_mean_iou_from_checkpoint,
    best_mean_iou_from_history,
    build_criterion,
    clean_state_dict,
    evaluate,
    load_checkpoint_file,
    make_dataset,
    make_loader,
    make_train_sampler,
    parameter_count,
    print_epoch_summary,
    read_metrics_history,
    train_one_epoch,
    write_metrics_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stronger damage segmentation architectures.",
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Architecture name. Supported: "
            + ", ".join(supported_damage_model_names())
        ),
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss", choices=sorted(LOSS_CHOICES), default="ce-dice")
    parser.add_argument(
        "--class-weights",
        nargs=3,
        type=float,
        default=None,
        metavar=("W_BACKGROUND", "W_NO_DAMAGE", "W_DAMAGED"),
    )
    parser.add_argument("--target-mode", choices=sorted(TARGET_MODES), default="3-class")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--augment-mode", choices=sorted(AUGMENT_MODES), default="safe")
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--damage-augment-threshold", type=float, default=0.001)
    parser.add_argument("--sampler", choices=sorted(SAMPLER_MODES), default="damage-sqrt")
    parser.add_argument("--damage-sampling-alpha", type=float, default=4.0)
    parser.add_argument("--high-damage-threshold", type=float, default=0.06)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_mode != "3-class":
        raise TrainingError("Axis 2 architecture training currently expects 3-class targets.")
    for name in ["image_size", "batch_size", "epochs", "base_channels"]:
        if int(getattr(args, name)) <= 0:
            raise TrainingError(f"--{name.replace('_', '-')} must be positive.")
    if args.lr <= 0:
        raise TrainingError("--lr must be positive.")
    if args.num_workers < 0:
        raise TrainingError("--num-workers must be non-negative.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise TrainingError("--max-train-samples must be positive.")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise TrainingError("--max-val-samples must be positive.")
    if not 0.0 <= args.augment_prob <= 1.0:
        raise TrainingError("--augment-prob must be between 0 and 1.")
    if args.damage_augment_threshold < 0.0:
        raise TrainingError("--damage-augment-threshold must be non-negative.")
    if args.damage_sampling_alpha < 0.0:
        raise TrainingError("--damage-sampling-alpha must be non-negative.")
    if args.high_damage_threshold < 0.0:
        raise TrainingError("--high-damage-threshold must be non-negative.")


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if "raw" in {part.lower() for part in output_dir.parts}:
        raise TrainingError("Refusing to write checkpoints inside a raw data directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
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


def load_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise TrainingError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise TrainingError(f"Resume checkpoint path is not a file: {checkpoint_path}")

    checkpoint = load_checkpoint_file(checkpoint_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = int(checkpoint.get("epoch", 0))
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
            except (RuntimeError, ValueError, KeyError) as exc:
                print(
                    "WARNING: Could not load optimizer_state_dict; "
                    f"resuming model weights only. Details: {exc}",
                    file=sys.stderr,
                )
    else:
        state_dict = checkpoint
        epoch = 0

    model.load_state_dict(clean_state_dict(state_dict))
    return epoch + 1


def save_arch_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, object],
    args: argparse.Namespace,
    loss_config: dict[str, object],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_name": args.model,
        "model_metadata": damage_model_metadata(args.model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(args),
        "loss_config": loss_config,
    }
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    output_dir = prepare_output_dir(args.output_dir)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    num_classes = 3

    train_dataset = make_dataset(
        args.root,
        args.train_csv,
        args.image_size,
        args.target_mode,
        args.max_train_samples,
        augment_mode=args.augment_mode,
        augment_prob=args.augment_prob,
        damage_augment_threshold=args.damage_augment_threshold,
    )
    val_dataset = make_dataset(
        args.root,
        args.val_csv,
        args.image_size,
        args.target_mode,
        args.max_val_samples,
        augment_mode="none",
        augment_prob=0.0,
        damage_augment_threshold=args.damage_augment_threshold,
    )
    if len(train_dataset) == 0:
        raise TrainingError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise TrainingError("Validation dataset is empty.")

    train_sampler = make_train_sampler(
        train_dataset,
        args.sampler,
        args.damage_sampling_alpha,
        args.high_damage_threshold,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=train_sampler is None,
        device=device,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
    )

    model = create_damage_model(
        args.model,
        num_classes=num_classes,
        in_channels=6,
        base_channels=args.base_channels,
    ).to(device)
    criterion, loss_config = build_criterion(
        loss_name=args.loss,
        target_mode=args.target_mode,
        num_classes=num_classes,
        class_weights_arg=args.class_weights,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    history: list[dict[str, object]] = []
    if args.resume_checkpoint is not None:
        start_epoch = load_resume_checkpoint(args.resume_checkpoint, model, optimizer, device)
        history = [
            item
            for item in read_metrics_history(output_dir / "metrics_history.json")
            if int(item.get("epoch", 0) or 0) < start_epoch
        ]

    metadata = damage_model_metadata(args.model)
    print(f"Device: {device}")
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"Model: {metadata['canonical_model']}")
    print(f"Model family: {metadata['family']}")
    print(f"Model description: {metadata['description']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Model parameters: {parameter_count(model):,}")
    print(f"Loss: {loss_config['loss']}")
    print(f"Class weights: {loss_config['class_weights'] or 'none'}")
    print(f"Augment mode: {args.augment_mode}")
    print(f"Augment probability: {args.augment_prob}")
    print(f"Sampler: {args.sampler}")
    print(f"Damage sampling alpha: {args.damage_sampling_alpha}")
    print(f"High damage threshold: {args.high_damage_threshold}")
    if args.resume_checkpoint is not None:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
        print(f"Starting epoch: {start_epoch}")

    best_mean_iou = -1.0
    if args.resume_checkpoint is not None:
        best_mean_iou = max(
            best_mean_iou_from_history(history),
            best_mean_iou_from_checkpoint(output_dir / "best_damage_arch.pt", device),
        )

    started_at = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
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
            "model": metadata["canonical_model"],
            "loss": loss_config["loss"],
            "class_weights": loss_config["class_weights"],
            "augment_mode": args.augment_mode,
            "augment_prob": args.augment_prob,
            "sampler": args.sampler,
            "damage_sampling_alpha": args.damage_sampling_alpha,
        }
        history.append(epoch_metrics)
        print_epoch_summary(epoch_metrics)

        if epoch_metrics["val_mean_iou"] > best_mean_iou:
            best_mean_iou = float(epoch_metrics["val_mean_iou"])
            save_arch_checkpoint(
                output_dir / "best_damage_arch.pt",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                args,
                loss_config,
            )

        save_arch_checkpoint(
            output_dir / "last_damage_arch.pt",
            model,
            optimizer,
            epoch,
            epoch_metrics,
            args,
            loss_config,
        )
        write_metrics_history(output_dir / "metrics_history.json", history)

    print(f"Training complete in {(time.time() - started_at) / 60.0:.2f} minutes.")
    print(f"Best validation mean IoU: {best_mean_iou:.4f}")
    print(f"Outputs saved to: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (
        TrainingError,
        XBDDatasetError,
        DamageModelFactoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

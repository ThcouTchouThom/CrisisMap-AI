#!/usr/bin/env python
"""Train Multi-Temporal Fusion damage segmentation models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crisismap.evaluation.evaluate_unet import CLASS_LABELS  # noqa: E402
from crisismap.models.multitemporal_fusion import (  # noqa: E402
    MultiTemporalFusionError,
    create_multitemporal_fusion_model,
    supported_multitemporal_fusion_models,
)
from train_xview2_strong_baseline import (  # noqa: E402
    LABEL_MODES,
    TRAIN_MODES,
    StrongBaselineTrainWrapper,
    XView2StrongBaselineTrainingError,
    compute_loss,
    damage_channels,
    evaluate,
    make_dataset,
    make_loader,
    metric_key,
    resolve_device,
    train_one_epoch,
    write_history,
)


class MultiTemporalFusionTrainingError(Exception):
    """Raised when MTF training cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Temporal Fusion damage models.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=supported_multitemporal_fusion_models())
    parser.add_argument("--target-mode", choices=["3-class", "5-class"], default="3-class")
    parser.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="3-class")
    parser.add_argument("--train-mode", choices=sorted(TRAIN_MODES), default="crop512")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--rare-damage-crop-prob", type=float, default=0.75)
    parser.add_argument("--rare-damage-crop-alpha", type=float, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--loss", choices=["focal-soft-dice", "ce-dice", "focal-dice", "focal-tversky"], default="focal-soft-dice")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-tversky-alpha", type=float, default=0.3)
    parser.add_argument("--focal-tversky-beta", type=float, default=0.7)
    parser.add_argument("--focal-tversky-gamma", type=float, default=0.75)
    parser.add_argument("--localization-loss-weight", type=float, default=0.3)
    parser.add_argument("--damage-loss-weight", type=float, default=1.0)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--damage-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.label_mode in {"3-class", "5-class"} and args.label_mode != args.target_mode:
        raise MultiTemporalFusionTrainingError(
            "--label-mode 3-class/5-class must match --target-mode."
        )
    if args.label_mode == "multilabel-building-damage" and args.target_mode != "3-class":
        raise MultiTemporalFusionTrainingError(
            "multilabel-building-damage currently expects --target-mode 3-class."
        )
    for name in ["image_size", "crop_size", "batch_size", "epochs"]:
        if int(getattr(args, name)) <= 0:
            raise MultiTemporalFusionTrainingError(f"--{name.replace('_', '-')} must be positive.")
    if args.train_mode != "full1024" and args.crop_size >= args.image_size:
        raise MultiTemporalFusionTrainingError("--crop-size must be smaller than --image-size.")
    if not 0.0 <= args.rare_damage_crop_prob <= 1.0:
        raise MultiTemporalFusionTrainingError("--rare-damage-crop-prob must be between 0 and 1.")
    if args.rare_damage_crop_alpha is not None and args.rare_damage_crop_alpha < 0.0:
        raise MultiTemporalFusionTrainingError("--rare-damage-crop-alpha must be non-negative.")
    if args.lr <= 0.0:
        raise MultiTemporalFusionTrainingError("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise MultiTemporalFusionTrainingError("--weight-decay must be non-negative.")
    if args.num_workers < 0:
        raise MultiTemporalFusionTrainingError("--num-workers must be non-negative.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise MultiTemporalFusionTrainingError("--max-train-samples must be positive.")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise MultiTemporalFusionTrainingError("--max-val-samples must be positive.")
    if args.focal_gamma <= 0.0:
        raise MultiTemporalFusionTrainingError("--focal-gamma must be positive.")
    if args.focal_tversky_alpha < 0.0 or args.focal_tversky_beta < 0.0:
        raise MultiTemporalFusionTrainingError("--focal-tversky-alpha/beta must be non-negative.")
    if args.focal_tversky_gamma <= 0.0:
        raise MultiTemporalFusionTrainingError("--focal-tversky-gamma must be positive.")
    if args.localization_loss_weight < 0.0 or args.damage_loss_weight <= 0.0:
        raise MultiTemporalFusionTrainingError("Loss weights must be non-negative/positive.")


def prepare_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if "raw" in {part.lower() for part in path.parts}:
        raise MultiTemporalFusionTrainingError("Refusing to write inside a raw data directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    best_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "best_metrics": best_metrics,
        "config": {
            "model": args.model,
            "target_mode": args.target_mode,
            "label_mode": args.label_mode,
            "train_mode": args.train_mode,
            "image_size": args.image_size,
            "crop_size": args.crop_size,
            "rare_damage_crop_prob": args.rare_damage_crop_prob,
            "rare_damage_crop_alpha": args.rare_damage_crop_alpha,
            "damage_channels": damage_channels(args.label_mode, args.target_mode),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "focal_tversky_alpha": args.focal_tversky_alpha,
            "focal_tversky_beta": args.focal_tversky_beta,
            "focal_tversky_gamma": args.focal_tversky_gamma,
            "building_threshold": args.building_threshold,
            "damage_threshold": args.damage_threshold,
        },
    }


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = prepare_output_dir(args.output_dir)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Target mode: {args.target_mode}")
    print(f"Label mode: {args.label_mode}")
    print(f"Train mode: {args.train_mode}")
    print(f"Loss: {args.loss}")
    print(f"Rare damage crop alpha: {args.rare_damage_crop_alpha}")
    print(f"AMP: {bool(args.amp and device.type == 'cuda')}")

    train_base = make_dataset(
        args.root,
        args.train_csv,
        args.image_size,
        args.target_mode,
        args.max_train_samples,
    )
    crop_size = args.crop_size if args.train_mode != "full1024" else None
    train_dataset = StrongBaselineTrainWrapper(
        train_base,
        crop_size=crop_size,
        augment=args.augment,
        rare_damage_crop_prob=args.rare_damage_crop_prob,
        rare_damage_crop_alpha=args.rare_damage_crop_alpha,
    )
    val_dataset = make_dataset(
        args.root,
        args.val_csv,
        args.image_size,
        args.target_mode,
        args.max_val_samples,
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, device, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, device, shuffle=False)

    try:
        model = create_multitemporal_fusion_model(
            args.model,
            damage_channels=damage_channels(args.label_mode, args.target_mode),
        ).to(device)
    except MultiTemporalFusionError as exc:
        raise SystemExit(str(exc)) from exc
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = output_dir / "metrics_history.json"
    best_metrics_path = output_dir / "best_val_metrics.json"
    best_checkpoint_path = output_dir / "best_multitemporal_fusion.pt"
    last_checkpoint_path = output_dir / "last_multitemporal_fusion.pt"

    history = read_history(history_path)
    start_epoch = 1
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0, -1.0)
    if history:
        for row in history:
            metrics = row.get("val_metrics", {})
            key = metric_key(metrics)
            if key > best_key:
                best_key = key
                best_metrics = metrics
    if args.resume_checkpoint:
        checkpoint = load_checkpoint_file(args.resume_checkpoint, device)
        if not isinstance(checkpoint, dict):
            raise MultiTemporalFusionTrainingError("Resume checkpoint must be a checkpoint dict.")
        model.load_state_dict(checkpoint["model_state_dict"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (KeyError, ValueError, RuntimeError) as exc:
            print(f"WARNING: Could not load optimizer state; resuming model-only: {exc}")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}")

    started_at = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = evaluate(model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        current_key = metric_key(val_metrics)
        improved = current_key > best_key
        if improved:
            best_key = current_key
            best_metrics = val_metrics
            torch.save(
                checkpoint_payload(model, optimizer, epoch, args, val_metrics, best_metrics),
                best_checkpoint_path,
            )
            with best_metrics_path.open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)
        torch.save(
            checkpoint_payload(model, optimizer, epoch, args, val_metrics, best_metrics),
            last_checkpoint_path,
        )
        write_history(history_path, history)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_mean_iou={float(val_metrics.get('mean_iou', 0.0)):.4f} "
            f"val_iou_damaged={float(val_metrics.get('iou_damaged', 0.0)):.4f} "
            f"val_f1_damaged={float(val_metrics.get('f1_damaged', 0.0)):.4f} "
            f"{'BEST' if improved else ''}"
        )

    summary = {
        "config": checkpoint_payload(model, optimizer, args.epochs, args, {}, best_metrics)["config"],
        "epochs": args.epochs,
        "best_val_metrics": best_metrics,
        "elapsed_seconds": time.time() - started_at,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "metrics_history": str(history_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training complete. Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    except XView2StrongBaselineTrainingError as exc:
        raise SystemExit(str(exc)) from exc

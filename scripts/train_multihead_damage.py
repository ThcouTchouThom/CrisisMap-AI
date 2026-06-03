#!/usr/bin/env python
"""Train Axis 3 multi-head building localization + damage models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crisismap.evaluation.evaluate_unet import CLASS_LABELS, confusion_matrix, metrics_from_confusion  # noqa: E402
from crisismap.models.multihead_damage import (  # noqa: E402
    MultiHeadDamageError,
    create_multihead_damage_model,
    supported_multihead_damage_models,
)
from train_xview2_strong_baseline import (  # noqa: E402
    TRAIN_MODES,
    StrongBaselineTrainWrapper,
    XView2StrongBaselineTrainingError,
    autocast_context,
    binary_bce_dice_loss,
    binary_focal_dice_loss,
    binary_focal_tversky_loss,
    make_dataset,
    make_loader,
    multiclass_ce_dice_loss,
    multiclass_focal_dice_loss,
    multiclass_focal_tversky_loss,
    resolve_device,
    write_history,
)


LABEL_MODES = {
    "3-class",
    "building-damage-2class",
    "multilabel-building-damage",
    "5-class",
}
BUILDING_LOSSES = {"bce-dice", "focal-tversky"}
DAMAGE_LOSSES = {"ce-dice", "focal-dice", "focal-tversky", "masked-ce", "masked-focal"}


class MultiHeadDamageTrainingError(Exception):
    """Raised when multi-head damage training cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Axis 3 multi-head damage models.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=supported_multihead_damage_models())
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
    parser.add_argument("--building-loss", choices=sorted(BUILDING_LOSSES), default="bce-dice")
    parser.add_argument("--damage-loss", choices=sorted(DAMAGE_LOSSES), default="ce-dice")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    parser.add_argument("--lambda-building", type=float, default=0.3)
    parser.add_argument("--lambda-damage", type=float, default=1.0)
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--damage-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.label_mode in {"3-class", "5-class"} and args.label_mode != args.target_mode:
        raise MultiHeadDamageTrainingError("--label-mode 3-class/5-class must match --target-mode.")
    if args.label_mode in {"building-damage-2class", "multilabel-building-damage"} and args.target_mode != "3-class":
        raise MultiHeadDamageTrainingError(f"{args.label_mode} currently expects --target-mode 3-class.")
    for name in ["image_size", "crop_size", "batch_size", "epochs"]:
        if int(getattr(args, name)) <= 0:
            raise MultiHeadDamageTrainingError(f"--{name.replace('_', '-')} must be positive.")
    if args.train_mode != "full1024" and args.crop_size >= args.image_size:
        raise MultiHeadDamageTrainingError("--crop-size must be smaller than --image-size.")
    if not 0.0 <= args.rare_damage_crop_prob <= 1.0:
        raise MultiHeadDamageTrainingError("--rare-damage-crop-prob must be between 0 and 1.")
    if args.rare_damage_crop_alpha is not None and args.rare_damage_crop_alpha < 0.0:
        raise MultiHeadDamageTrainingError("--rare-damage-crop-alpha must be non-negative.")
    if args.lr <= 0.0:
        raise MultiHeadDamageTrainingError("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise MultiHeadDamageTrainingError("--weight-decay must be non-negative.")
    if args.num_workers < 0:
        raise MultiHeadDamageTrainingError("--num-workers must be non-negative.")
    if args.lambda_building < 0.0 or args.lambda_damage <= 0.0:
        raise MultiHeadDamageTrainingError("Loss weights must be non-negative/positive.")
    for name in ["building_threshold", "damage_threshold"]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise MultiHeadDamageTrainingError(f"--{name.replace('_', '-')} must be 0..1.")


def damage_channels(label_mode: str, target_mode: str) -> int:
    if label_mode == "building-damage-2class":
        return 2
    if label_mode == "multilabel-building-damage":
        return 1
    return len(CLASS_LABELS[target_mode])


def prepare_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if "raw" in {part.lower() for part in path.parts}:
        raise MultiHeadDamageTrainingError("Refusing to write inside a raw data directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def building_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    building_targets = (targets > 0).float()
    if args.building_loss == "focal-tversky":
        return binary_focal_tversky_loss(
            outputs["building_logits"],
            building_targets,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.tversky_gamma,
        )
    return binary_bce_dice_loss(outputs["building_logits"], building_targets)


def masked_building_damage_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    building_mask = targets > 0
    if not torch.any(building_mask):
        return logits.sum() * 0.0
    masked_logits = logits.permute(0, 2, 3, 1)[building_mask]
    masked_targets = (targets[building_mask] > 1).long()
    if args.damage_loss in {"masked-focal", "focal-dice"}:
        ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")
        pt = torch.exp(-ce)
        return torch.mean(torch.pow(1.0 - pt, args.focal_gamma) * ce)
    return F.cross_entropy(masked_logits, masked_targets)


def damage_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    logits = outputs["damage_logits"]
    if args.label_mode == "building-damage-2class":
        return masked_building_damage_loss(logits, targets, args)
    if args.label_mode == "multilabel-building-damage":
        damage_targets = (targets > 1).float()
        if args.damage_loss == "focal-tversky":
            return binary_focal_tversky_loss(
                logits,
                damage_targets,
                alpha=args.tversky_alpha,
                beta=args.tversky_beta,
                gamma=args.tversky_gamma,
            )
        if args.damage_loss in {"focal-dice", "masked-focal"}:
            return binary_focal_dice_loss(logits, damage_targets, gamma=args.focal_gamma)
        return binary_bce_dice_loss(logits, damage_targets)
    if args.damage_loss == "focal-tversky":
        return multiclass_focal_tversky_loss(
            logits,
            targets,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.tversky_gamma,
        )
    if args.damage_loss in {"focal-dice", "masked-focal"}:
        return multiclass_focal_dice_loss(logits, targets, gamma=args.focal_gamma)
    return multiclass_ce_dice_loss(logits, targets)


def compute_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    loc_loss = building_loss(outputs, targets, args)
    dmg_loss = damage_loss(outputs, targets, args)
    return args.lambda_building * loc_loss + args.lambda_damage * dmg_loss


def predict_damage_mask(outputs: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    if args.label_mode == "building-damage-2class":
        building = torch.sigmoid(outputs["building_logits"].squeeze(1)) >= args.building_threshold
        damage_class = outputs["damage_logits"].argmax(dim=1)
        pred = torch.zeros_like(damage_class, dtype=torch.long)
        pred[building] = 1
        pred[building & (damage_class == 1)] = 2
        return pred
    if args.label_mode == "multilabel-building-damage":
        building = torch.sigmoid(outputs["building_logits"].squeeze(1)) >= args.building_threshold
        damaged = torch.sigmoid(outputs["damage_logits"].squeeze(1)) >= args.damage_threshold
        pred = torch.zeros_like(building, dtype=torch.long)
        pred[building] = 1
        pred[building & damaged] = 2
        return pred
    return outputs["damage_logits"].argmax(dim=1)


def building_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    total = tp + tn + fp + fn
    building_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "pixel_accuracy": (tp + tn) / max(total, 1),
        "mean_iou": (building_iou + background_iou) / 2.0,
        "background_iou": background_iou,
        "building_iou": building_iou,
        "building_precision": precision,
        "building_recall": recall,
        "building_f1": f1,
        "building_tp": tp,
        "building_tn": tn,
        "building_fp": fp,
        "building_fn": fn,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    num_classes = len(CLASS_LABELS[args.target_mode])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    building_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        with autocast_context(device, args.amp):
            outputs = model(images)
            loss = compute_loss(outputs, targets, args)
        preds = predict_damage_mask(outputs, args)
        confusion += confusion_matrix(preds, targets, num_classes)
        building_pred = torch.sigmoid(outputs["building_logits"].squeeze(1)) >= args.building_threshold
        building_target = targets > 0
        building_counts["tp"] += int(torch.count_nonzero(building_pred & building_target).item())
        building_counts["tn"] += int(torch.count_nonzero((~building_pred) & (~building_target)).item())
        building_counts["fp"] += int(torch.count_nonzero(building_pred & (~building_target)).item())
        building_counts["fn"] += int(torch.count_nonzero((~building_pred) & building_target).item())
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    damage_metrics = metrics_from_confusion(confusion)
    return {
        "average_loss": total_loss / max(total_samples, 1),
        "damage_metrics": damage_metrics,
        "building_metrics": building_metrics_from_counts(**building_counts),
        "summary_metrics": {
            **damage_metrics,
            "building_iou": building_metrics_from_counts(**building_counts)["building_iou"],
            "building_f1": building_metrics_from_counts(**building_counts)["building_f1"],
        },
    }


def metric_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    damage = metrics.get("damage_metrics", {})
    building = metrics.get("building_metrics", {})
    return (
        float(damage.get("f1_damaged", 0.0)),
        float(damage.get("iou_damaged", 0.0)),
        float(building.get("building_iou", 0.0)),
    )


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
            "building_loss": args.building_loss,
            "damage_loss": args.damage_loss,
            "lambda_building": args.lambda_building,
            "lambda_damage": args.lambda_damage,
            "building_threshold": args.building_threshold,
            "damage_threshold": args.damage_threshold,
        },
    }


def train_one_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            outputs = model(images)
            loss = compute_loss(outputs, targets, args)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(loader)} loss={float(loss.detach().item()):.4f}")
    return total_loss / max(total_samples, 1)


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
    print(f"Building loss: {args.building_loss}")
    print(f"Damage loss: {args.damage_loss}")
    print(f"AMP: {bool(args.amp and device.type == 'cuda')}")

    train_base = make_dataset(args.root, args.train_csv, args.image_size, args.target_mode, args.max_train_samples)
    crop_size = args.crop_size if args.train_mode != "full1024" else None
    train_dataset = StrongBaselineTrainWrapper(
        train_base,
        crop_size=crop_size,
        augment=args.augment,
        rare_damage_crop_prob=args.rare_damage_crop_prob,
        rare_damage_crop_alpha=args.rare_damage_crop_alpha,
    )
    val_dataset = make_dataset(args.root, args.val_csv, args.image_size, args.target_mode, args.max_val_samples)
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, device, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, device, shuffle=False)

    try:
        model = create_multihead_damage_model(
            args.model,
            damage_channels=damage_channels(args.label_mode, args.target_mode),
        ).to(device)
    except MultiHeadDamageError as exc:
        raise SystemExit(str(exc)) from exc
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = output_dir / "metrics_history.json"
    best_metrics_path = output_dir / "best_val_metrics.json"
    best_checkpoint_path = output_dir / "best_multihead_damage.pt"
    last_checkpoint_path = output_dir / "last_multihead_damage.pt"

    history = read_history(history_path)
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0, -1.0)
    if history:
        for row in history:
            metrics = row.get("val_metrics", {})
            key = metric_key(metrics)
            if key > best_key:
                best_key = key
                best_metrics = metrics

    start_epoch = 1
    if args.resume_checkpoint:
        checkpoint = load_checkpoint_file(args.resume_checkpoint, device)
        if not isinstance(checkpoint, dict):
            raise MultiHeadDamageTrainingError("Resume checkpoint must be a checkpoint dict.")
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
            torch.save(checkpoint_payload(model, optimizer, epoch, args, val_metrics, best_metrics), best_checkpoint_path)
            with best_metrics_path.open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)
        torch.save(checkpoint_payload(model, optimizer, epoch, args, val_metrics, best_metrics), last_checkpoint_path)
        write_history(history_path, history)
        damage = val_metrics["damage_metrics"]
        building = val_metrics["building_metrics"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_iou_damaged={float(damage.get('iou_damaged', 0.0)):.4f} "
            f"val_f1_damaged={float(damage.get('f1_damaged', 0.0)):.4f} "
            f"val_building_iou={float(building.get('building_iou', 0.0)):.4f} "
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

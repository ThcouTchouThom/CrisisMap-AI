#!/usr/bin/env python
"""Train xView2 strong-baseline-inspired damage models.

This script is intentionally separate from the existing U-Net and Siamese
training entrypoints. It trains a shared-encoder ResNet U-Net with two heads:

* building localization binary head
* damage head, either multiclass or binary damaged/not-damaged
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


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
from crisismap.models.xview2_strong_baseline import (  # noqa: E402
    XView2StrongBaselineError,
    create_xview2_strong_baseline_model,
    supported_xview2_strong_baseline_models,
)


LABEL_MODES = {"3-class", "5-class", "multilabel-building-damage"}
TRAIN_MODES = {"full1024", "crop512", "crop608"}
LOSS_MODES = {"focal-soft-dice", "ce-dice", "focal-dice", "focal-tversky"}
DAMAGE_CLASSES = {2, 3, 4}


class XView2StrongBaselineTrainingError(Exception):
    """Raised when strong-baseline training cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train xView2 strong-baseline-inspired damage segmentation models."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=supported_xview2_strong_baseline_models())
    parser.add_argument("--target-mode", choices=["3-class", "5-class"], default="3-class")
    parser.add_argument("--label-mode", choices=sorted(LABEL_MODES), default="3-class")
    parser.add_argument("--train-mode", choices=sorted(TRAIN_MODES), default="full1024")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--rare-damage-crop-prob", type=float, default=0.65)
    parser.add_argument("--rare-damage-crop-alpha", type=float, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--loss", choices=sorted(LOSS_MODES), default="focal-soft-dice")
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.label_mode in {"3-class", "5-class"} and args.label_mode != args.target_mode:
        raise XView2StrongBaselineTrainingError(
            "--label-mode 3-class/5-class must match --target-mode."
        )
    if args.label_mode == "multilabel-building-damage" and args.target_mode != "3-class":
        raise XView2StrongBaselineTrainingError(
            "multilabel-building-damage currently expects --target-mode 3-class."
        )
    for name in ["image_size", "crop_size", "batch_size", "epochs"]:
        if int(getattr(args, name)) <= 0:
            raise XView2StrongBaselineTrainingError(f"--{name.replace('_', '-')} must be positive.")
    if args.train_mode != "full1024" and args.crop_size >= args.image_size:
        raise XView2StrongBaselineTrainingError("--crop-size must be smaller than --image-size.")
    if not 0.0 <= args.rare_damage_crop_prob <= 1.0:
        raise XView2StrongBaselineTrainingError("--rare-damage-crop-prob must be between 0 and 1.")
    if args.rare_damage_crop_alpha is not None and args.rare_damage_crop_alpha < 0.0:
        raise XView2StrongBaselineTrainingError("--rare-damage-crop-alpha must be non-negative.")
    if args.lr <= 0.0:
        raise XView2StrongBaselineTrainingError("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise XView2StrongBaselineTrainingError("--weight-decay must be non-negative.")
    if args.num_workers < 0:
        raise XView2StrongBaselineTrainingError("--num-workers must be non-negative.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise XView2StrongBaselineTrainingError("--max-train-samples must be positive.")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise XView2StrongBaselineTrainingError("--max-val-samples must be positive.")
    if args.focal_gamma <= 0.0:
        raise XView2StrongBaselineTrainingError("--focal-gamma must be positive.")
    if args.focal_tversky_alpha < 0.0 or args.focal_tversky_beta < 0.0:
        raise XView2StrongBaselineTrainingError("--focal-tversky-alpha/beta must be non-negative.")
    if args.focal_tversky_gamma <= 0.0:
        raise XView2StrongBaselineTrainingError("--focal-tversky-gamma must be positive.")
    if args.localization_loss_weight < 0.0 or args.damage_loss_weight <= 0.0:
        raise XView2StrongBaselineTrainingError("Loss weights must be non-negative/positive.")
    for name in ["building_threshold", "damage_threshold"]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise XView2StrongBaselineTrainingError(f"--{name.replace('_', '-')} must be 0..1.")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise XView2StrongBaselineTrainingError("CUDA was requested, but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if "raw" in {part.lower() for part in path.parts}:
        raise XView2StrongBaselineTrainingError("Refusing to write inside a raw data directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


class StrongBaselineTrainWrapper(torch.utils.data.Dataset):
    """Apply train-only geometric augmentation and optional rare-damage crops."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        crop_size: int | None,
        augment: bool,
        rare_damage_crop_prob: float,
        rare_damage_crop_alpha: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.crop_size = crop_size
        self.augment = augment
        self.rare_damage_crop_prob = rare_damage_crop_prob
        self.rare_damage_crop_alpha = rare_damage_crop_alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        image = sample["image"]
        target = sample["target"]
        if self.crop_size is not None:
            image, target = self.random_crop(image, target)
        if self.augment:
            image, target = self.geometric_augment(image, target)
        sample["image"] = image.contiguous()
        sample["target"] = target.contiguous()
        return sample

    def random_crop(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape
        crop = int(self.crop_size or h)
        if crop >= h or crop >= w:
            return image, target

        damage_pixels = (target > 1).nonzero(as_tuple=False)
        rare_crop_prob = self.rare_damage_crop_prob
        if self.rare_damage_crop_alpha is not None:
            rare_crop_prob = self.rare_damage_crop_alpha / (1.0 + self.rare_damage_crop_alpha)
        if damage_pixels.numel() and torch.rand(()) < rare_crop_prob:
            choice = damage_pixels[torch.randint(0, damage_pixels.shape[0], (1,)).item()]
            cy, cx = int(choice[0]), int(choice[1])
            top = min(max(cy - crop // 2, 0), h - crop)
            left = min(max(cx - crop // 2, 0), w - crop)
        else:
            top = int(torch.randint(0, h - crop + 1, (1,)).item())
            left = int(torch.randint(0, w - crop + 1, (1,)).item())
        return image[:, top : top + crop, left : left + crop], target[top : top + crop, left : left + crop]

    def geometric_augment(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-2,))
            target = torch.flip(target, dims=(-2,))
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            image = torch.rot90(image, k=k, dims=(-2, -1))
            target = torch.rot90(target, k=k, dims=(-2, -1))
        return image, target


def make_dataset(
    root: Path,
    split_csv: Path,
    image_size: int,
    target_mode: str,
    max_samples: int | None,
) -> torch.utils.data.Dataset:
    try:
        dataset = XBDPairDataset(
            root=root,
            split_csv=split_csv,
            image_size=image_size,
            target_mode=target_mode,
            augment_mode="none",
        )
    except XBDDatasetError as exc:
        raise XView2StrongBaselineTrainingError(str(exc)) from exc
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def damage_channels(label_mode: str, target_mode: str) -> int:
    if label_mode == "multilabel-building-damage":
        return 1
    return len(CLASS_LABELS[target_mode])


def binary_focal_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    logits = logits.squeeze(1)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    focal = torch.mean(torch.pow(1.0 - pt, gamma) * bce)
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets)
    denominator = torch.sum(probs) + torch.sum(targets)
    dice = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    return focal + dice


def binary_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    logits = logits.squeeze(1)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets)
    denominator = torch.sum(probs) + torch.sum(targets)
    dice = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    return bce + dice


def binary_focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    logits = logits.squeeze(1)
    targets = targets.float()
    probs = torch.sigmoid(logits)
    tp = torch.sum(probs * targets)
    fp = torch.sum(probs * (1.0 - targets))
    fn = torch.sum((1.0 - probs) * targets)
    tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    return torch.pow(1.0 - tversky, gamma)


def multiclass_focal_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal = torch.mean(torch.pow(1.0 - pt, gamma) * ce)
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probs.dtype)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
    dice = torch.mean(1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon))
    return focal + dice


def multiclass_ce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets)
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probs.dtype)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
    dice = torch.mean(1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon))
    return ce + dice


def multiclass_focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probs.dtype)
    dims = (0, 2, 3)
    tp = torch.sum(probs * one_hot, dim=dims)
    fp = torch.sum(probs * (1.0 - one_hot), dim=dims)
    fn = torch.sum((1.0 - probs) * one_hot, dim=dims)
    tversky = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    return torch.mean(torch.pow(1.0 - tversky, gamma))


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    label_mode: str,
    gamma: float,
    loss_mode: str,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    localization_weight: float,
    damage_weight: float,
) -> torch.Tensor:
    building_targets = (targets > 0).float()
    if loss_mode == "ce-dice":
        localization_loss = binary_bce_dice_loss(outputs["building_logits"], building_targets)
    elif loss_mode == "focal-tversky":
        localization_loss = binary_focal_tversky_loss(
            outputs["building_logits"],
            building_targets,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
        )
    else:
        localization_loss = binary_focal_dice_loss(
            outputs["building_logits"],
            building_targets,
            gamma=gamma,
        )
    if label_mode == "multilabel-building-damage":
        damage_targets = (targets > 1).float()
        if loss_mode == "ce-dice":
            damage_loss = binary_bce_dice_loss(outputs["damage_logits"], damage_targets)
        elif loss_mode == "focal-tversky":
            damage_loss = binary_focal_tversky_loss(
                outputs["damage_logits"],
                damage_targets,
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=tversky_gamma,
            )
        else:
            damage_loss = binary_focal_dice_loss(
                outputs["damage_logits"],
                damage_targets,
                gamma=gamma,
            )
    else:
        if loss_mode == "ce-dice":
            damage_loss = multiclass_ce_dice_loss(outputs["damage_logits"], targets)
        elif loss_mode == "focal-tversky":
            damage_loss = multiclass_focal_tversky_loss(
                outputs["damage_logits"],
                targets,
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=tversky_gamma,
            )
        else:
            damage_loss = multiclass_focal_dice_loss(
                outputs["damage_logits"],
                targets,
                gamma=gamma,
            )
    return localization_weight * localization_loss + damage_weight * damage_loss


def predict_damage_mask(
    outputs: dict[str, torch.Tensor],
    label_mode: str,
    building_threshold: float,
    damage_threshold: float,
) -> torch.Tensor:
    if label_mode == "multilabel-building-damage":
        building = torch.sigmoid(outputs["building_logits"].squeeze(1)) >= building_threshold
        damaged = torch.sigmoid(outputs["damage_logits"].squeeze(1)) >= damage_threshold
        pred = torch.zeros_like(building, dtype=torch.long)
        pred[building] = 1
        pred[building & damaged] = 2
        return pred
    return outputs["damage_logits"].argmax(dim=1)


def autocast_context(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            outputs = model(images)
            loss = compute_loss(
                outputs,
                targets,
                args.label_mode,
                args.focal_gamma,
                args.loss,
                args.focal_tversky_alpha,
                args.focal_tversky_beta,
                args.focal_tversky_gamma,
                args.localization_loss_weight,
                args.damage_loss_weight,
            )
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(loader)} loss={float(loss.detach().item()):.4f}")
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    num_classes = len(CLASS_LABELS[args.target_mode])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        with autocast_context(device, args.amp):
            outputs = model(images)
            loss = compute_loss(
                outputs,
                targets,
                args.label_mode,
                args.focal_gamma,
                args.loss,
                args.focal_tversky_alpha,
                args.focal_tversky_beta,
                args.focal_tversky_gamma,
                args.localization_loss_weight,
                args.damage_loss_weight,
            )
        preds = predict_damage_mask(
            outputs,
            args.label_mode,
            args.building_threshold,
            args.damage_threshold,
        )
        confusion += confusion_matrix(preds, targets, num_classes)
        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    metrics = metrics_from_confusion(confusion)
    metrics["average_loss"] = total_loss / max(total_samples, 1)
    return metrics


def metric_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("f1_damaged", 0.0)),
        float(metrics.get("iou_damaged", 0.0)),
        float(metrics.get("mean_iou", 0.0)),
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


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
        model = create_xview2_strong_baseline_model(
            args.model,
            damage_channels=damage_channels(args.label_mode, args.target_mode),
        ).to(device)
    except XView2StrongBaselineError as exc:
        raise SystemExit(str(exc)) from exc
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = output_dir / "metrics_history.json"
    best_metrics_path = output_dir / "best_val_metrics.json"
    best_checkpoint_path = output_dir / "best_xview2_strong.pt"
    last_checkpoint_path = output_dir / "last_xview2_strong.pt"

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0, -1.0)
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
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
    main()

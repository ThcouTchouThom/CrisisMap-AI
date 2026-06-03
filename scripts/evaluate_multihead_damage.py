#!/usr/bin/env python
"""Evaluate Axis 3 multi-head building + damage checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import CLASS_LABELS, confusion_matrix, metrics_from_confusion  # noqa: E402
from crisismap.models.multihead_damage import (  # noqa: E402
    MultiHeadDamageError,
    create_multihead_damage_model,
    supported_multihead_damage_models,
)
from train_multihead_damage import (  # noqa: E402
    LABEL_MODES,
    building_metrics_from_counts,
    damage_channels,
    predict_damage_mask,
)
from train_xview2_strong_baseline import autocast_context  # noqa: E402


class MultiHeadDamageEvaluationError(Exception):
    """Raised when multi-head evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Axis 3 multi-head checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--model", choices=supported_multihead_damage_models(), default=None)
    parser.add_argument("--target-mode", choices=["3-class", "5-class"], default=None)
    parser.add_argument("--label-mode", choices=sorted(LABEL_MODES), default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--building-threshold", type=float, default=None)
    parser.add_argument("--damage-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise MultiHeadDamageEvaluationError("CUDA was requested, but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> object:
    if not path.exists():
        raise MultiHeadDamageEvaluationError(f"Checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise MultiHeadDamageEvaluationError("Checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise MultiHeadDamageEvaluationError("Checkpoint state_dict is empty.")
    return cleaned


def checkpoint_config(checkpoint: object) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return dict(checkpoint["config"])
    return {}


def pick_arg(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def clip_to_building(preds: torch.Tensor, building_mask: torch.Tensor) -> torch.Tensor:
    clipped = preds.clone()
    clipped[~building_mask] = 0
    return clipped


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args_proxy: argparse.Namespace,
    target_mode: str,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    num_classes = len(CLASS_LABELS[target_mode])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    constrained_confusion = torch.zeros_like(confusion)
    building_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        with autocast_context(device, amp):
            outputs = model(images)
        preds = predict_damage_mask(outputs, args_proxy)
        building_pred = torch.sigmoid(outputs["building_logits"].squeeze(1)) >= args_proxy.building_threshold
        building_target = targets > 0
        constrained = clip_to_building(preds, building_pred)
        confusion += confusion_matrix(preds, targets, num_classes)
        constrained_confusion += confusion_matrix(constrained, targets, num_classes)
        building_counts["tp"] += int(torch.count_nonzero(building_pred & building_target).item())
        building_counts["tn"] += int(torch.count_nonzero((~building_pred) & (~building_target)).item())
        building_counts["fp"] += int(torch.count_nonzero(building_pred & (~building_target)).item())
        building_counts["fn"] += int(torch.count_nonzero((~building_pred) & building_target).item())
    return {
        "damage_metrics": metrics_from_confusion(confusion),
        "constrained_damage_metrics": metrics_from_confusion(constrained_confusion),
        "building_metrics": building_metrics_from_counts(**building_counts),
    }


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative.")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint_config(checkpoint)
    model_name = pick_arg(args, config, "model", None)
    target_mode = pick_arg(args, config, "target_mode", "3-class")
    label_mode = pick_arg(args, config, "label_mode", target_mode)
    image_size = int(pick_arg(args, config, "image_size", 1024))
    building_threshold = float(pick_arg(args, config, "building_threshold", 0.5))
    damage_threshold = float(pick_arg(args, config, "damage_threshold", 0.5))

    if model_name is None:
        raise SystemExit("--model is required when checkpoint config does not include it.")
    if target_mode not in CLASS_LABELS:
        raise SystemExit(f"Unsupported target mode: {target_mode}")
    if label_mode not in LABEL_MODES:
        raise SystemExit(f"Unsupported label mode: {label_mode}")

    try:
        dataset = XBDPairDataset(
            root=args.root,
            split_csv=args.split_csv,
            image_size=image_size,
            target_mode=target_mode,
            augment_mode="none",
        )
    except XBDDatasetError as exc:
        raise SystemExit(str(exc)) from exc
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    try:
        model = create_multihead_damage_model(
            model_name,
            damage_channels=damage_channels(label_mode, target_mode),
        ).to(device)
    except MultiHeadDamageError as exc:
        raise SystemExit(str(exc)) from exc
    model.load_state_dict(clean_state_dict(checkpoint))

    args_proxy = argparse.Namespace(
        label_mode=label_mode,
        building_threshold=building_threshold,
        damage_threshold=damage_threshold,
    )
    started_at = time.time()
    metrics = evaluate(model, loader, device, args_proxy, target_mode, args.amp)
    damage = metrics["damage_metrics"]
    constrained = metrics["constrained_damage_metrics"]
    building = metrics["building_metrics"]
    row = {
        "model": model_name,
        "target_mode": target_mode,
        "label_mode": label_mode,
        "image_size": image_size,
        "building_threshold": building_threshold,
        "damage_threshold": damage_threshold,
        "test_mean_iou": damage.get("mean_iou"),
        "test_iou_damaged": damage.get("iou_damaged"),
        "test_f1_damaged": damage.get("f1_damaged"),
        "constrained_mean_iou": constrained.get("mean_iou"),
        "constrained_iou_damaged": constrained.get("iou_damaged"),
        "constrained_f1_damaged": constrained.get("f1_damaged"),
        "building_iou": building.get("building_iou"),
        "building_f1": building.get("building_f1"),
        "building_precision": building.get("building_precision"),
        "building_recall": building.get("building_recall"),
    }
    if target_mode == "3-class":
        row["binary_damage_xview2_like_score"] = (
            0.3 * float(building.get("building_f1", 0.0))
            + 0.7 * float(damage.get("f1_damaged", 0.0))
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "root": str(args.root),
        "split_csv": str(args.split_csv),
        "num_samples": len(dataset),
        "elapsed_seconds": time.time() - started_at,
        "config": {
            "model": model_name,
            "target_mode": target_mode,
            "label_mode": label_mode,
            "image_size": image_size,
            "building_threshold": building_threshold,
            "damage_threshold": damage_threshold,
        },
        "metrics": metrics,
        "summary_row": row,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if args.output_csv:
        write_csv(args.output_csv, row)
    print(
        f"mean_iou={float(damage.get('mean_iou', 0.0)):.6f} "
        f"iou_damaged={float(damage.get('iou_damaged', 0.0)):.6f} "
        f"f1_damaged={float(damage.get('f1_damaged', 0.0)):.6f} "
        f"building_iou={float(building.get('building_iou', 0.0)):.6f}"
    )


if __name__ == "__main__":
    main()

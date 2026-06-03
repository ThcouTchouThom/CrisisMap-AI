#!/usr/bin/env python
"""Evaluate Multi-Temporal Fusion checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from crisismap.evaluation.evaluate_unet import (  # noqa: E402
    CLASS_LABELS,
    confusion_matrix,
    metrics_from_confusion,
)
from crisismap.models.multitemporal_fusion import (  # noqa: E402
    MultiTemporalFusionError,
    create_multitemporal_fusion_model,
    supported_multitemporal_fusion_models,
)
from train_xview2_strong_baseline import LABEL_MODES, damage_channels, predict_damage_mask  # noqa: E402


class MultiTemporalFusionEvaluationError(Exception):
    """Raised when evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Multi-Temporal Fusion checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--model", choices=supported_multitemporal_fusion_models(), default=None)
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
            raise MultiTemporalFusionEvaluationError("CUDA was requested, but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> object:
    if not path.exists():
        raise MultiTemporalFusionEvaluationError(f"Checkpoint does not exist: {path}")
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
        raise MultiTemporalFusionEvaluationError("Checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise MultiTemporalFusionEvaluationError("Checkpoint state_dict is empty.")
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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_mode: str,
    target_mode: str,
    building_threshold: float,
    damage_threshold: float,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    num_classes = len(CLASS_LABELS[target_mode])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
        else:
            outputs = model(images)
        preds = predict_damage_mask(outputs, label_mode, building_threshold, damage_threshold)
        confusion += confusion_matrix(preds, targets, num_classes)
    return metrics_from_confusion(confusion)


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
        model = create_multitemporal_fusion_model(
            model_name,
            damage_channels=damage_channels(label_mode, target_mode),
        ).to(device)
    except MultiTemporalFusionError as exc:
        raise SystemExit(str(exc)) from exc
    model.load_state_dict(clean_state_dict(checkpoint))
    metrics = evaluate(
        model,
        loader,
        device,
        label_mode,
        target_mode,
        building_threshold,
        damage_threshold,
        args.amp,
    )

    summary_row = {
        "model": model_name,
        "target_mode": target_mode,
        "label_mode": label_mode,
        "image_size": image_size,
        **metrics,
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "root": str(args.root),
        "split_csv": str(args.split_csv),
        "num_samples": len(dataset),
        "config": {
            "model": model_name,
            "target_mode": target_mode,
            "label_mode": label_mode,
            "image_size": image_size,
            "building_threshold": building_threshold,
            "damage_threshold": damage_threshold,
        },
        "metrics": metrics,
        "summary_row": summary_row,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if args.output_csv:
        write_csv(args.output_csv, summary_row)
    print(
        f"mean_iou={float(metrics.get('mean_iou', 0.0)):.6f} "
        f"iou_damaged={float(metrics.get('iou_damaged', 0.0)):.6f} "
        f"f1_damaged={float(metrics.get('f1_damaged', 0.0)):.6f}"
    )


if __name__ == "__main__":
    main()

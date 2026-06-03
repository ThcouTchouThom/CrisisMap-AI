#!/usr/bin/env python
"""Evaluate xView2 strong-baseline-inspired checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


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


class XView2StrongBaselineEvaluationError(Exception):
    """Raised when evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an xView2 strong-baseline checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--model", choices=supported_xview2_strong_baseline_models(), default=None)
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
            raise XView2StrongBaselineEvaluationError("CUDA was requested, but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> object:
    if not path.exists():
        raise XView2StrongBaselineEvaluationError(f"Checkpoint does not exist: {path}")
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
        raise XView2StrongBaselineEvaluationError("Checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise XView2StrongBaselineEvaluationError("Checkpoint state_dict is empty.")
    return cleaned


def config_from_checkpoint(checkpoint: object) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return dict(checkpoint["config"])
    return {}


def pick_arg(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def damage_channels(label_mode: str, target_mode: str) -> int:
    if label_mode == "multilabel-building-damage":
        return 1
    return len(CLASS_LABELS[target_mode])


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
        with autocast_context(device, amp):
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
    config = config_from_checkpoint(checkpoint)
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
        model = create_xview2_strong_baseline_model(
            model_name,
            damage_channels=damage_channels(label_mode, target_mode),
        ).to(device)
    except XView2StrongBaselineError as exc:
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
        "summary_row": {
            "model": model_name,
            "target_mode": target_mode,
            "label_mode": label_mode,
            "image_size": image_size,
            **metrics,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if args.output_csv:
        write_csv(args.output_csv, payload["summary_row"])

    print(
        f"mean_iou={float(metrics.get('mean_iou', 0.0)):.6f} "
        f"iou_damaged={float(metrics.get('iou_damaged', 0.0)):.6f} "
        f"f1_damaged={float(metrics.get('f1_damaged', 0.0)):.6f}"
    )


if __name__ == "__main__":
    main()

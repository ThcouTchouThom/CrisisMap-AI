#!/usr/bin/env python
"""Evaluate downstream building-constrained damage metrics for Axis 3 models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crisismap.data.xbd_dataset import XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import CLASS_LABELS, confusion_matrix, metrics_from_confusion  # noqa: E402
from crisismap.models.multihead_damage import create_multihead_damage_model  # noqa: E402
from train_multihead_damage import LABEL_MODES, damage_channels, predict_damage_mask  # noqa: E402
from train_xview2_strong_baseline import autocast_context  # noqa: E402


MODES = [
    "raw",
    "predicted_building_clip",
    "predicted_building_component_majority",
    "oracle_building_clip",
    "oracle_building_component_majority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Axis 3 downstream post-processing.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--target-mode", choices=["3-class", "5-class"], default=None)
    parser.add_argument("--label-mode", choices=sorted(LABEL_MODES), default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    parser.add_argument("--damage-threshold", type=float, default=None)
    parser.add_argument("--component-connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_config(checkpoint: object) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return dict(checkpoint["config"])
    return {}


def clean_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def label_connected_components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    mask = mask.astype(bool, copy=False)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    if connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            queue: deque[tuple[int, int]] = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        queue.append((ny, nx))
    return labels, current


def component_majority_single(raw_pred: np.ndarray, building_mask: np.ndarray, connectivity: int) -> np.ndarray:
    labels, component_count = label_connected_components(building_mask, connectivity)
    output = np.zeros_like(raw_pred, dtype=np.int64)
    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        values = raw_pred[component_mask]
        no_damage = int(np.count_nonzero(values == 1))
        damaged = int(np.count_nonzero(values == 2))
        output[component_mask] = 2 if damaged > no_damage else 1
    return output


def component_majority_batch(raw_preds: torch.Tensor, building_masks: torch.Tensor, connectivity: int, device: torch.device) -> torch.Tensor:
    outputs = []
    for pred, mask in zip(raw_preds.detach().cpu().numpy(), building_masks.detach().cpu().numpy()):
        outputs.append(component_majority_single(pred.astype(np.int64), mask.astype(bool), connectivity))
    return torch.from_numpy(np.stack(outputs, axis=0)).to(device=device, dtype=torch.long)


def apply_clip(preds: torch.Tensor, building_mask: torch.Tensor) -> torch.Tensor:
    out = preds.clone()
    out[~building_mask] = 0
    return out


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint_config(checkpoint)
    model_name = args.model or config.get("model")
    target_mode = args.target_mode or config.get("target_mode", "3-class")
    label_mode = args.label_mode or config.get("label_mode", target_mode)
    image_size = int(args.image_size or config.get("image_size", 1024))
    damage_threshold = float(args.damage_threshold or config.get("damage_threshold", 0.5))
    if model_name is None:
        raise RuntimeError("--model is required when checkpoint config does not include it.")
    dataset = XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=image_size,
        target_mode=target_mode,
        augment_mode="none",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = create_multihead_damage_model(
        model_name,
        damage_channels=damage_channels(label_mode, target_mode),
    ).to(device)
    model.load_state_dict(clean_state_dict(checkpoint))
    model.eval()
    num_classes = len(CLASS_LABELS[target_mode])
    confusions: dict[str, torch.Tensor] = {}
    confusions["raw"] = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    confusions["oracle_building_clip"] = torch.zeros_like(confusions["raw"])
    confusions["oracle_building_component_majority"] = torch.zeros_like(confusions["raw"])
    for threshold in args.thresholds:
        confusions[f"predicted_building_clip@{threshold:g}"] = torch.zeros_like(confusions["raw"])
        confusions[f"predicted_building_component_majority@{threshold:g}"] = torch.zeros_like(confusions["raw"])

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()
        with autocast_context(device, args.amp):
            outputs = model(images)
        proxy = argparse.Namespace(
            label_mode=label_mode,
            building_threshold=0.5,
            damage_threshold=damage_threshold,
        )
        raw = predict_damage_mask(outputs, proxy)
        confusions["raw"] += confusion_matrix(raw, targets, num_classes)
        oracle_mask = targets > 0
        confusions["oracle_building_clip"] += confusion_matrix(apply_clip(raw, oracle_mask), targets, num_classes)
        oracle_component = component_majority_batch(raw, oracle_mask, args.component_connectivity, device)
        confusions["oracle_building_component_majority"] += confusion_matrix(oracle_component, targets, num_classes)
        building_prob = torch.sigmoid(outputs["building_logits"].squeeze(1))
        for threshold in args.thresholds:
            mask = building_prob >= threshold
            confusions[f"predicted_building_clip@{threshold:g}"] += confusion_matrix(apply_clip(raw, mask), targets, num_classes)
            component = component_majority_batch(raw, mask, args.component_connectivity, device)
            confusions[f"predicted_building_component_majority@{threshold:g}"] += confusion_matrix(component, targets, num_classes)

    rows = []
    metrics_by_mode = {}
    for mode, confusion in confusions.items():
        metrics = metrics_from_confusion(confusion)
        metrics_by_mode[mode] = metrics
        rows.append({"mode": mode, **metrics})
    return {
        "config": {
            "checkpoint": str(args.checkpoint),
            "root": str(args.root),
            "split_csv": str(args.split_csv),
            "model": model_name,
            "target_mode": target_mode,
            "label_mode": label_mode,
            "image_size": image_size,
            "thresholds": args.thresholds,
            "damage_threshold": damage_threshold,
            "component_connectivity": args.component_connectivity,
            "note": "Oracle modes use target>0 only as analysis upper bounds.",
        },
        "num_samples": len(dataset),
        "metrics_by_mode": metrics_by_mode,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "pixel_accuracy", "mean_iou", "iou_background", "iou_no_damage", "iou_damaged", "precision_damaged", "recall_damaged", "f1_damaged"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(args.output_csv, payload["rows"])
    best = max(payload["rows"], key=lambda row: (float(row.get("f1_damaged", 0.0)), float(row.get("iou_damaged", 0.0))))
    print(
        f"best_mode={best['mode']} "
        f"iou_damaged={float(best.get('iou_damaged', 0.0)):.6f} "
        f"f1_damaged={float(best.get('f1_damaged', 0.0)):.6f}"
    )


if __name__ == "__main__":
    main()

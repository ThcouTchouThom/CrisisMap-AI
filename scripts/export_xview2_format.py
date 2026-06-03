#!/usr/bin/env python
"""Export damage predictions and targets as xView2-style PNG masks.

The current CrisisMap AI damage task is a simplified 3-class semantic
segmentation problem. In that mode, exported damage masks use:

0 = background
1 = no_damage building
2 = damaged building

This is intentionally xView2-like, but it is not directly comparable to the
official 5-class xView2 scoring protocol. The script also supports future
5-class checkpoints with:

0 = background, 1 = no_damage, 2 = minor, 3 = major, 4 = destroyed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import (  # noqa: E402
    CLASS_LABELS,
    extract_state_dict,
    load_checkpoint_file,
)
from crisismap.models.unet import UNet  # noqa: E402


TTA_MODES = {"none", "flips", "rot90", "d4"}


class XView2ExportError(Exception):
    """Raised when xView2-style export cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export U-Net damage predictions and targets as xView2-style PNG masks."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-mode", choices=["3-class", "5-class"], default="3-class")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tta-mode", choices=sorted(TTA_MODES), default="none")
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV path. Defaults to <output-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional metadata JSON path. Defaults to <output-dir>/metadata.json.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ["image_size", "batch_size", "base_channels"]:
        if int(getattr(args, name)) <= 0:
            raise XView2ExportError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise XView2ExportError("--num-workers must be non-negative.")
    if args.start_index < 0:
        raise XView2ExportError("--start-index must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise XView2ExportError("--max-samples must be positive when provided.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise XView2ExportError("CUDA was requested, but CUDA is not available.")
    return device


def load_model(args: argparse.Namespace, device: torch.device) -> UNet:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise XView2ExportError(f"Checkpoint does not exist: {checkpoint_path}")
    num_classes = len(CLASS_LABELS[args.target_mode])
    model = UNet(in_channels=6, num_classes=num_classes, base_channels=args.base_channels).to(device)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    try:
        model.load_state_dict(extract_state_dict(checkpoint))
    except RuntimeError as exc:
        raise XView2ExportError(
            "Checkpoint weights do not match this U-Net configuration. "
            "Check --target-mode and --base-channels."
        ) from exc
    model.eval()
    return model


def tta_ops(mode: str) -> list[tuple[int, bool, bool]]:
    if mode == "none":
        return [(0, False, False)]
    if mode == "flips":
        return [(0, False, False), (0, True, False), (0, False, True), (0, True, True)]
    if mode == "rot90":
        return [(k, False, False) for k in range(4)]
    if mode == "d4":
        return [(k, False, False) for k in range(4)] + [(k, True, False) for k in range(4)]
    raise XView2ExportError(f"Unsupported TTA mode: {mode}")


def apply_op(tensor: torch.Tensor, op: tuple[int, bool, bool]) -> torch.Tensor:
    rotations, flip_h, flip_v = op
    output = torch.rot90(tensor, k=rotations, dims=(-2, -1)) if rotations else tensor
    if flip_h:
        output = torch.flip(output, dims=(-1,))
    if flip_v:
        output = torch.flip(output, dims=(-2,))
    return output


def invert_op(tensor: torch.Tensor, op: tuple[int, bool, bool]) -> torch.Tensor:
    rotations, flip_h, flip_v = op
    output = tensor
    if flip_v:
        output = torch.flip(output, dims=(-2,))
    if flip_h:
        output = torch.flip(output, dims=(-1,))
    if rotations:
        output = torch.rot90(output, k=-rotations, dims=(-2, -1))
    return output


def autocast_context(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


@torch.no_grad()
def predict_logits_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    mode: str,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    ops = tta_ops(mode)
    for op in ops:
        view = apply_op(images, op)
        with autocast_context(device, amp):
            logits = model(view)
        logits = invert_op(logits, op).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    if logits_sum is None:
        raise XView2ExportError(f"No TTA operations for mode: {mode}")
    return logits_sum / float(len(ops))


def mask_to_localization(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8, copy=False)


def sanitize_damage_mask(mask: np.ndarray, target_mode: str) -> np.ndarray:
    max_class = 4 if target_mode == "5-class" else 2
    clipped = np.clip(mask.astype(np.int64, copy=False), 0, max_class)
    return clipped.astype(np.uint8, copy=False)


def save_uint8_png(path: Path, mask: np.ndarray) -> None:
    if mask.shape != (1024, 1024):
        image = Image.fromarray(mask.astype(np.uint8), mode="L")
        image = image.resize((1024, 1024), Image.NEAREST)
        mask = np.asarray(image, dtype=np.uint8)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def write_masks(
    output_dir: Path,
    prefix: str,
    index: int,
    prediction: np.ndarray,
    target: np.ndarray,
    target_mode: str,
) -> dict[str, str]:
    sample_id = f"{index:05d}"
    pred_damage = sanitize_damage_mask(prediction, target_mode)
    target_damage = sanitize_damage_mask(target, target_mode)
    pred_loc = mask_to_localization(pred_damage)
    target_loc = mask_to_localization(target_damage)

    paths = {
        "localization_prediction": f"{prefix}_localization_{sample_id}_prediction.png",
        "damage_prediction": f"{prefix}_damage_{sample_id}_prediction.png",
        "localization_target": f"{prefix}_localization_{sample_id}_target.png",
        "damage_target": f"{prefix}_damage_{sample_id}_target.png",
    }
    save_uint8_png(output_dir / paths["localization_prediction"], pred_loc)
    save_uint8_png(output_dir / paths["damage_prediction"], pred_damage)
    save_uint8_png(output_dir / paths["localization_target"], target_loc)
    save_uint8_png(output_dir / paths["damage_target"], target_damage)
    return paths


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or output_dir / "manifest.csv"
    metadata_path = args.metadata_json or output_dir / "metadata.json"

    try:
        dataset = XBDPairDataset(
            root=args.root,
            split_csv=args.split_csv,
            image_size=args.image_size,
            target_mode=args.target_mode,
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
    model = load_model(args, device)

    rows: list[dict[str, Any]] = []
    exported = 0
    next_index = args.start_index

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].cpu().numpy()
        logits = predict_logits_tta(model, images, args.tta_mode, device, args.amp)
        predictions = logits.argmax(dim=1).cpu().numpy()
        pair_ids = batch.get("pair_id")

        for i in range(predictions.shape[0]):
            if args.max_samples is not None and exported >= args.max_samples:
                break
            paths = write_masks(
                output_dir=output_dir,
                prefix=args.prefix,
                index=next_index,
                prediction=predictions[i],
                target=targets[i],
                target_mode=args.target_mode,
            )
            rows.append(
                {
                    "index": f"{next_index:05d}",
                    "pair_id": pair_ids[i] if pair_ids is not None else "",
                    **paths,
                }
            )
            exported += 1
            next_index += 1
        if args.max_samples is not None and exported >= args.max_samples:
            break

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "index",
            "pair_id",
            "localization_prediction",
            "damage_prediction",
            "localization_target",
            "damage_target",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "format": "xview2_style_mask_export",
        "important_note": (
            "3-class exports are xView2-like only and are not comparable to the "
            "official 5-class xView2 leaderboard/scoring protocol."
        ),
        "target_mode": args.target_mode,
        "class_mapping": {
            "3-class": {"0": "background", "1": "no_damage", "2": "damaged"},
            "5-class": {
                "0": "background",
                "1": "no_damage",
                "2": "minor",
                "3": "major",
                "4": "destroyed",
            },
        }[args.target_mode],
        "checkpoint": str(args.checkpoint),
        "root": str(args.root),
        "split_csv": str(args.split_csv),
        "image_size_input": args.image_size,
        "image_size_exported": 1024,
        "tta_mode": args.tta_mode,
        "num_exported": exported,
        "manifest": str(manifest_path),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {exported} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate damage post-processing with predicted and oracle building masks.

This is an evaluation-only script. It compares raw damage predictions against
building-mask post-processing modes using one or more building segmentation
checkpoints. Predicted-building modes never use ground truth building masks;
oracle modes use them only as an upper-bound reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset
from crisismap.evaluation.evaluate_unet import (
    CLASS_LABELS,
    confusion_matrix,
    extract_state_dict,
    load_checkpoint_file,
    metrics_from_confusion,
)
from crisismap.models.damage_model_factory import (
    DamageModelFactoryError,
    create_damage_model,
    supported_damage_model_names,
)
from train_building_segmentation import (
    MODEL_CHOICES as BUILDING_MODEL_CHOICES,
    BuildingTrainingError,
    build_model as build_building_model,
    clean_state_dict,
    input_channels as building_input_channels,
    normalize_logits as normalize_building_logits,
)


TTA_MODES = {"none", "flips", "rot90", "d4"}
ENSEMBLE_MODES = {"average_prob", "union", "intersection", "majority"}
POSTPROCESS_MODES = {
    "predicted_building_clip",
    "predicted_building_component_majority",
    "oracle_building_clip",
    "oracle_building_component_majority",
}
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MASK_COLORS = np.array(
    [
        [0, 0, 0],
        [0, 190, 90],
        [220, 40, 40],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate damage predictions with predicted/oracle building-mask post-processing."
    )
    parser.add_argument("--damage-checkpoint", required=True, type=Path)
    parser.add_argument("--building-checkpoint", required=True, nargs="+", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--target-mode", default="3-class", choices=sorted(CLASS_LABELS))
    parser.add_argument(
        "--damage-model",
        default="unet",
        choices=sorted(set(supported_damage_model_names() + ["unet"])),
        help="Damage architecture used by --damage-checkpoint.",
    )
    parser.add_argument("--damage-base-channels", type=int, default=32)
    parser.add_argument("--damage-tta", default="d4", choices=sorted(TTA_MODES))
    parser.add_argument(
        "--building-model",
        required=True,
        nargs="+",
        choices=sorted(BUILDING_MODEL_CHOICES),
        help="One model name per building checkpoint, or one name reused for all checkpoints.",
    )
    parser.add_argument(
        "--building-input-mode",
        nargs="+",
        default=["pre"],
        choices=["pre", "post", "pre-post"],
        help="One input mode per building checkpoint, or one mode reused for all checkpoints.",
    )
    parser.add_argument("--building-tta", default="none", choices=sorted(TTA_MODES))
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--ensemble-modes",
        nargs="+",
        default=["average_prob"],
        choices=sorted(ENSEMBLE_MODES),
    )
    parser.add_argument(
        "--postprocess-modes",
        nargs="+",
        default=[
            "predicted_building_clip",
            "predicted_building_component_majority",
            "oracle_building_clip",
            "oracle_building_component_majority",
        ],
        choices=sorted(POSTPROCESS_MODES),
    )
    parser.add_argument("--component-connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--save-examples-dir", type=Path)
    parser.add_argument("--num-examples", type=int, default=8)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def expand_per_checkpoint(values: list[str], n: int, label: str) -> list[str]:
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{label} must contain either 1 value or {n} values.")
    return values


def checkpoint_model_hint(checkpoint: object) -> str | None:
    if not isinstance(checkpoint, dict):
        return None
    for key in ("model_name", "actual_model", "model"):
        value = checkpoint.get(key)
        if isinstance(value, str) and value:
            return value
    config = checkpoint.get("config")
    if isinstance(config, dict):
        for key in ("model", "model_name", "actual_model"):
            value = config.get(key)
            if isinstance(value, str) and value:
                return value
    metadata = checkpoint.get("model_metadata")
    if isinstance(metadata, dict):
        for key in ("requested_model", "canonical_model"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def load_damage_model(
    checkpoint_path: Path,
    model_name: str,
    target_mode: str,
    base_channels: int,
    device: torch.device,
) -> nn.Module:
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    state_dict = extract_state_dict(checkpoint)
    num_classes = len(CLASS_LABELS[target_mode])
    try:
        model = create_damage_model(
            model_name,
            num_classes=num_classes,
            in_channels=6,
            base_channels=base_channels,
        ).to(device)
    except DamageModelFactoryError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        hint = checkpoint_model_hint(checkpoint)
        hint_text = f" Checkpoint metadata suggests model '{hint}'." if hint else ""
        raise RuntimeError(
            "Damage checkpoint weights do not match the requested architecture. "
            f"Requested --damage-model='{model_name}', --target-mode='{target_mode}', "
            f"--damage-base-channels={base_channels}, checkpoint='{checkpoint_path}'."
            f"{hint_text}"
        ) from exc
    model.eval()
    return model


def load_building_model(
    checkpoint_path: Path,
    model_name: str,
    input_mode: str,
    device: torch.device,
) -> nn.Module:
    try:
        model = build_building_model(
            model_name=model_name,
            in_channels=building_input_channels(input_mode),
            device=device,
        )
    except BuildingTrainingError as exc:
        raise RuntimeError(str(exc)) from exc

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()
    return model


def tta_ops(mode: str) -> list[tuple[str, int | None]]:
    if mode == "none":
        return [("identity", None)]
    if mode == "flips":
        return [("identity", None), ("hflip", None), ("vflip", None), ("hvflip", None)]
    if mode == "rot90":
        return [("rot90", k) for k in range(4)]
    if mode == "d4":
        return [("rot90", k) for k in range(4)] + [("rot90_hflip", k) for k in range(4)]
    raise ValueError(f"Unsupported TTA mode: {mode}")


def apply_op(x: torch.Tensor, op: tuple[str, int | None]) -> torch.Tensor:
    name, k = op
    if name == "identity":
        return x
    if name == "hflip":
        return torch.flip(x, dims=(-1,))
    if name == "vflip":
        return torch.flip(x, dims=(-2,))
    if name == "hvflip":
        return torch.flip(x, dims=(-2, -1))
    if name == "rot90":
        return torch.rot90(x, k=int(k or 0), dims=(-2, -1))
    if name == "rot90_hflip":
        return torch.flip(torch.rot90(x, k=int(k or 0), dims=(-2, -1)), dims=(-1,))
    raise ValueError(f"Unsupported TTA operation: {name}")


def invert_op(x: torch.Tensor, op: tuple[str, int | None]) -> torch.Tensor:
    name, k = op
    if name == "identity":
        return x
    if name == "hflip":
        return torch.flip(x, dims=(-1,))
    if name == "vflip":
        return torch.flip(x, dims=(-2,))
    if name == "hvflip":
        return torch.flip(x, dims=(-2, -1))
    if name == "rot90":
        return torch.rot90(x, k=-int(k or 0), dims=(-2, -1))
    if name == "rot90_hflip":
        return torch.rot90(torch.flip(x, dims=(-1,)), k=-int(k or 0), dims=(-2, -1))
    raise ValueError(f"Unsupported TTA operation: {name}")


def select_building_input(images: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "pre":
        return images[:, :3]
    if input_mode == "post":
        return images[:, 3:]
    if input_mode == "pre-post":
        return images
    raise ValueError(f"Unsupported building input mode: {input_mode}")


@torch.no_grad()
def predict_damage_logits_tta(
    model: nn.Module,
    images: torch.Tensor,
    mode: str,
    amp_enabled: bool,
    device_type: str,
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    context = torch.autocast(device_type=device_type, enabled=amp_enabled) if amp_enabled else nullcontext()
    for op in tta_ops(mode):
        x_aug = apply_op(images, op)
        with context:
            logits = model(x_aug)
        logits = invert_op(logits.float(), op)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return logits_sum / len(tta_ops(mode))


@torch.no_grad()
def predict_building_probability_tta(
    model: nn.Module,
    images: torch.Tensor,
    input_mode: str,
    mode: str,
    amp_enabled: bool,
    device_type: str,
) -> torch.Tensor:
    x = select_building_input(images, input_mode)
    logits_sum: torch.Tensor | None = None
    context = torch.autocast(device_type=device_type, enabled=amp_enabled) if amp_enabled else nullcontext()
    for op in tta_ops(mode):
        x_aug = apply_op(x, op)
        with context:
            logits = normalize_building_logits(model(x_aug))
        logits = invert_op(logits.float(), op)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return torch.sigmoid(logits_sum / len(tta_ops(mode))).squeeze(1)


def label_connected_components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    mask_bool = mask.astype(bool)
    labels = np.zeros(mask_bool.shape, dtype=np.int32)
    if not mask_bool.any():
        return labels, 0

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    h, w = mask_bool.shape
    current = 0
    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            queue: deque[tuple[int, int]] = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        queue.append((ny, nx))
    return labels, current


def component_majority_single(pred: np.ndarray, building_mask: np.ndarray, connectivity: int) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.int64)
    labels, count = label_connected_components(building_mask, connectivity)
    for comp_id in range(1, count + 1):
        comp = labels == comp_id
        no_damage = int(np.logical_and(pred == 1, comp).sum())
        damaged = int(np.logical_and(pred == 2, comp).sum())
        out[comp] = 2 if damaged > no_damage else 1
    return out


def component_majority_batch(
    raw_preds: torch.Tensor,
    building_masks: torch.Tensor,
    connectivity: int,
) -> torch.Tensor:
    outputs = []
    for pred, mask in zip(raw_preds.cpu().numpy(), building_masks.cpu().numpy()):
        outputs.append(component_majority_single(pred.astype(np.int64), mask.astype(bool), connectivity))
    return torch.as_tensor(np.stack(outputs, axis=0), dtype=torch.long, device=raw_preds.device)


def ensemble_mask(probabilities: list[torch.Tensor], threshold: float, mode: str) -> torch.Tensor:
    stack = torch.stack(probabilities, dim=0)
    if mode == "average_prob":
        return stack.mean(dim=0) >= threshold
    binary = stack >= threshold
    if mode == "union":
        return binary.any(dim=0)
    if mode == "intersection":
        return binary.all(dim=0)
    if mode == "majority":
        votes_needed = (binary.shape[0] + 1) // 2
        return binary.sum(dim=0) >= votes_needed
    raise ValueError(f"Unsupported ensemble mode: {mode}")


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask.astype(np.int64), 0, len(MASK_COLORS) - 1)
    return MASK_COLORS[clipped]


def image_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = mask_to_rgb(mask).astype(np.float32) / 255.0
    active = mask > 0
    out = image.copy()
    out[active] = (1 - alpha) * out[active] + alpha * rgb[active]
    return np.clip(out, 0.0, 1.0)


def save_examples(
    save_dir: Path,
    batch: dict[str, Any],
    targets: torch.Tensor,
    raw_preds: torch.Tensor,
    sample_building_mask: torch.Tensor | None,
    sample_processed_pred: torch.Tensor | None,
    start_index: int,
    max_examples: int,
) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    targets_cpu = targets.detach().cpu().numpy()
    raw_cpu = raw_preds.detach().cpu().numpy()
    bmask_cpu = sample_building_mask.detach().cpu().numpy() if sample_building_mask is not None else None
    processed_cpu = (
        sample_processed_pred.detach().cpu().numpy() if sample_processed_pred is not None else None
    )

    saved = 0
    for i in range(images.shape[0]):
        if start_index + saved >= max_examples:
            break
        pre = image_to_numpy(images[i, :3])
        post = image_to_numpy(images[i, 3:])
        raw_pred = raw_cpu[i]
        gt = targets_cpu[i]
        panels = [
            ("Pre", pre),
            ("Post", post),
            ("Ground truth", mask_to_rgb(gt)),
            ("Raw damage", mask_to_rgb(raw_pred)),
        ]
        if bmask_cpu is not None:
            panels.append(("Pred building", bmask_cpu[i].astype(np.float32)))
        if processed_cpu is not None:
            panels.append(("Post-processed", mask_to_rgb(processed_cpu[i])))
            panels.append(("Overlay", overlay_mask(post, processed_cpu[i])))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, image) in zip(axes, panels):
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        pair_ids = batch.get("pair_id")
        pair_id = pair_ids[i] if pair_ids is not None else f"sample_{start_index + saved:04d}"
        fig.tight_layout()
        fig.savefig(save_dir / f"{pair_id}_damage_building_postprocess.png", dpi=120)
        plt.close(fig)
        saved += 1
    return saved


def row_from_metrics(
    mode: str,
    threshold: str,
    ensemble_mode: str,
    metrics: dict[str, float],
    damage_tta: str,
    building_tta: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "threshold": threshold,
        "building_ensemble_mode": ensemble_mode,
        "damage_tta": damage_tta,
        "building_tta": building_tta,
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "mean_iou": float(metrics["mean_iou"]),
        "iou_background": float(metrics["iou_background"]),
        "iou_no_damage": float(metrics["iou_no_damage"]),
        "iou_damaged": float(metrics["iou_damaged"]),
        "precision_damaged": float(metrics["precision_damaged"]),
        "recall_damaged": float(metrics["recall_damaged"]),
        "f1_damaged": float(metrics["f1_damaged"]),
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    device = resolve_device(args.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = bool(args.amp and device.type == "cuda")

    n_building = len(args.building_checkpoint)
    building_models = expand_per_checkpoint(args.building_model, n_building, "--building-model")
    building_input_modes = expand_per_checkpoint(
        args.building_input_mode, n_building, "--building-input-mode"
    )

    print(f"Device: {device}")
    print(f"Damage model: {args.damage_model}")
    print(f"Damage checkpoint: {args.damage_checkpoint}")
    print(f"Damage TTA: {args.damage_tta}")
    print(f"Building TTA: {args.building_tta}")
    print(f"Building checkpoints: {n_building}")

    try:
        damage_model = load_damage_model(
            args.damage_checkpoint,
            args.damage_model,
            args.target_mode,
            args.damage_base_channels,
            device,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    building_nets = [
        load_building_model(path, model_name, input_mode, device)
        for path, model_name, input_mode in zip(
            args.building_checkpoint, building_models, building_input_modes
        )
    ]

    try:
        dataset = XBDPairDataset(
            root=args.root,
            pairs_csv=args.split_csv,
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

    confusion_by_key: dict[tuple[str, str, str], torch.Tensor] = {
        ("raw", "raw", "none"): torch.zeros((3, 3), dtype=torch.int64),
    }
    if "oracle_building_clip" in args.postprocess_modes:
        confusion_by_key[("oracle_building_clip", "oracle", "oracle")] = torch.zeros(
            (3, 3), dtype=torch.int64
        )
    if "oracle_building_component_majority" in args.postprocess_modes:
        confusion_by_key[("oracle_building_component_majority", "oracle", "oracle")] = torch.zeros(
            (3, 3), dtype=torch.int64
        )
    for ensemble_mode in args.ensemble_modes:
        for threshold in args.thresholds:
            threshold_key = f"{threshold:.3f}"
            if "predicted_building_clip" in args.postprocess_modes:
                confusion_by_key[("predicted_building_clip", threshold_key, ensemble_mode)] = (
                    torch.zeros((3, 3), dtype=torch.int64)
                )
            if "predicted_building_component_majority" in args.postprocess_modes:
                confusion_by_key[
                    ("predicted_building_component_majority", threshold_key, ensemble_mode)
                ] = torch.zeros((3, 3), dtype=torch.int64)

    saved_examples = 0
    sample_mask: torch.Tensor | None = None
    sample_processed: torch.Tensor | None = None

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()

        damage_logits = predict_damage_logits_tta(
            damage_model, images, args.damage_tta, amp_enabled, device_type
        )
        raw_preds = damage_logits.argmax(dim=1)
        confusion_by_key[("raw", "raw", "none")] += confusion_matrix(
            raw_preds.cpu(), targets.cpu(), 3
        )

        building_gt = targets > 0
        if "oracle_building_clip" in args.postprocess_modes:
            oracle_clip = raw_preds.clone()
            oracle_clip[~building_gt] = 0
            confusion_by_key[("oracle_building_clip", "oracle", "oracle")] += confusion_matrix(
                oracle_clip.cpu(), targets.cpu(), 3
            )
        if "oracle_building_component_majority" in args.postprocess_modes:
            oracle_component = component_majority_batch(
                raw_preds, building_gt, args.component_connectivity
            )
            confusion_by_key[("oracle_building_component_majority", "oracle", "oracle")] += (
                confusion_matrix(oracle_component.cpu(), targets.cpu(), 3)
            )

        building_probs = [
            predict_building_probability_tta(
                model, images, input_mode, args.building_tta, amp_enabled, device_type
            )
            for model, input_mode in zip(building_nets, building_input_modes)
        ]

        for ensemble_mode in args.ensemble_modes:
            for threshold in args.thresholds:
                threshold_key = f"{threshold:.3f}"
                building_mask = ensemble_mask(building_probs, threshold, ensemble_mode)
                if "predicted_building_clip" in args.postprocess_modes:
                    clipped = raw_preds.clone()
                    clipped[~building_mask] = 0
                    confusion_by_key[
                        ("predicted_building_clip", threshold_key, ensemble_mode)
                    ] += confusion_matrix(clipped.cpu(), targets.cpu(), 3)
                if "predicted_building_component_majority" in args.postprocess_modes:
                    component_pred = component_majority_batch(
                        raw_preds, building_mask, args.component_connectivity
                    )
                    confusion_by_key[
                        ("predicted_building_component_majority", threshold_key, ensemble_mode)
                    ] += confusion_matrix(component_pred.cpu(), targets.cpu(), 3)

                if sample_mask is None:
                    sample_mask = building_mask.detach().cpu()
                    if "predicted_building_component_majority" in args.postprocess_modes:
                        sample_processed = component_majority_batch(
                            raw_preds, building_mask, args.component_connectivity
                        ).detach().cpu()
                    elif "predicted_building_clip" in args.postprocess_modes:
                        sample_processed = raw_preds.clone()
                        sample_processed[~building_mask] = 0
                        sample_processed = sample_processed.detach().cpu()

        if args.save_examples_dir and saved_examples < args.num_examples:
            saved_examples += save_examples(
                args.save_examples_dir,
                batch,
                targets,
                raw_preds,
                sample_mask,
                sample_processed,
                saved_examples,
                args.num_examples,
            )
            sample_mask = None
            sample_processed = None

        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx}/{len(loader)} batches")

    rows = []
    raw_metrics: dict[str, float] | None = None
    for (mode, threshold_key, ensemble_mode), conf in sorted(confusion_by_key.items()):
        metrics = metrics_from_confusion(conf)
        if mode == "raw":
            raw_metrics = metrics
        rows.append(
            row_from_metrics(
                mode=mode,
                threshold=threshold_key,
                ensemble_mode=ensemble_mode,
                metrics=metrics,
                damage_tta=args.damage_tta,
                building_tta=args.building_tta,
            )
        )

    rows.sort(
        key=lambda row: (
            float(row["f1_damaged"]),
            float(row["iou_damaged"]),
            float(row["mean_iou"]),
        ),
        reverse=True,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "damage_checkpoint": str(args.damage_checkpoint),
            "damage_model": args.damage_model,
            "damage_tta": args.damage_tta,
            "building_checkpoints": [str(path) for path in args.building_checkpoint],
            "building_models": building_models,
            "building_input_modes": building_input_modes,
            "building_tta": args.building_tta,
            "thresholds": args.thresholds,
            "ensemble_modes": args.ensemble_modes,
            "postprocess_modes": args.postprocess_modes,
            "component_connectivity": args.component_connectivity,
            "root": str(args.root),
            "split_csv": str(args.split_csv),
            "image_size": args.image_size,
            "target_mode": args.target_mode,
            "num_samples": len(dataset),
        },
        "elapsed_seconds": time.time() - start_time,
        "rows": rows,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "mode",
            "threshold",
            "building_ensemble_mode",
            "damage_tta",
            "building_tta",
            "pixel_accuracy",
            "mean_iou",
            "iou_background",
            "iou_no_damage",
            "iou_damaged",
            "precision_damaged",
            "recall_damaged",
            "f1_damaged",
        ]
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\nBest downstream rows")
    for row in rows[:8]:
        print(
            f"{row['mode']} threshold={row['threshold']} ensemble={row['building_ensemble_mode']} "
            f"mIoU={float(row['mean_iou']):.4f} damaged IoU={float(row['iou_damaged']):.4f} "
            f"F1={float(row['f1_damaged']):.4f} precision={float(row['precision_damaged']):.4f} "
            f"recall={float(row['recall_damaged']):.4f}"
        )

    if raw_metrics is not None:
        print("\nDeltas vs raw")
        for row in rows[:8]:
            if row["mode"] == "raw":
                continue
            print(
                f"{row['mode']} threshold={row['threshold']} ensemble={row['building_ensemble_mode']} "
                f"delta_mIoU={float(row['mean_iou']) - raw_metrics['mean_iou']:+.4f} "
                f"delta_damaged_IoU={float(row['iou_damaged']) - raw_metrics['iou_damaged']:+.4f} "
                f"delta_F1={float(row['f1_damaged']) - raw_metrics['f1_damaged']:+.4f}"
            )


if __name__ == "__main__":
    main()

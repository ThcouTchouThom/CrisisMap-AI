"""Evaluate logit ensembles between the U-Net champion and an Axis 2 model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

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
from crisismap.models.damage_model_factory import (  # noqa: E402
    DamageModelFactoryError,
    create_damage_model,
    damage_model_metadata,
)
from crisismap.models.unet import UNet  # noqa: E402

from evaluate_damage_tta import apply_op, invert_op, tta_ops  # noqa: E402


class DamageEnsembleError(Exception):
    """Raised when ensemble evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net + damage architecture logit ensembles.",
    )
    parser.add_argument("--unet-checkpoint", required=True, type=Path)
    parser.add_argument("--arch-checkpoint", required=True, type=Path)
    parser.add_argument("--arch-model", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--split-role", choices=["validation", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-mode", choices=["3-class"], default="3-class")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--arch-base-channels", type=int, default=32)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="U-Net logit weights. Architecture weight is 1 - weight.",
    )
    parser.add_argument(
        "--tta-modes",
        nargs="+",
        choices=["none", "d4"],
        default=["none"],
    )
    parser.add_argument(
        "--damage-biases",
        nargs="+",
        type=float,
        default=[0.0],
        help="Optional additive bias for class 2 damaged logits.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise DamageEnsembleError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise DamageEnsembleError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise DamageEnsembleError("--num-workers must be non-negative.")
    if args.base_channels <= 0 or args.arch_base_channels <= 0:
        raise DamageEnsembleError("--base-channels values must be positive.")
    for weight in args.weights:
        if not 0.0 <= weight <= 1.0:
            raise DamageEnsembleError("--weights must be between 0 and 1.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DamageEnsembleError("CUDA was requested, but CUDA is not available.")
    return device


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise DamageEnsembleError("Checkpoint is not a state_dict or checkpoint dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[str(key).removeprefix("module.")] = value
    if not cleaned:
        raise DamageEnsembleError("Checkpoint contains no tensor weights.")
    return cleaned


def load_unet(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = UNet(
        in_channels=6,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = load_checkpoint_file(args.unet_checkpoint, device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


def load_arch(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = create_damage_model(
        args.arch_model,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        in_channels=6,
        base_channels=args.arch_base_channels,
    ).to(device)
    checkpoint = load_checkpoint_file(args.arch_checkpoint, device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


def make_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
        augment_mode="none",
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def predict_logits_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    mode: str,
    use_amp: bool,
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    for op in tta_ops(mode):
        view = apply_op(images, op)
        with autocast_context(use_amp):
            logits = model(view)
        logits = invert_op(logits, op).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    if logits_sum is None:
        raise DamageEnsembleError(f"No TTA operations for mode: {mode}")
    return logits_sum / float(len(tta_ops(mode)))


@torch.no_grad()
def evaluate(
    unet_model: torch.nn.Module,
    arch_model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_classes = len(CLASS_LABELS[args.target_mode])

    for tta_mode in args.tta_modes:
        for unet_weight in args.weights:
            for damage_bias in args.damage_biases:
                confusion = torch.zeros(
                    (num_classes, num_classes),
                    dtype=torch.int64,
                    device=device,
                )
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    targets = batch["target"].to(device, non_blocking=True)
                    unet_logits = predict_logits_tta(
                        unet_model,
                        images,
                        tta_mode,
                        use_amp,
                    )
                    arch_logits = predict_logits_tta(
                        arch_model,
                        images,
                        tta_mode,
                        use_amp,
                    )
                    logits = unet_weight * unet_logits + (1.0 - unet_weight) * arch_logits
                    logits[:, 2] = logits[:, 2] + float(damage_bias)
                    preds = torch.argmax(logits, dim=1)
                    confusion += confusion_matrix(preds, targets, num_classes)

                metrics = metrics_from_confusion(confusion)
                rows.append(build_row(args, metrics, tta_mode, unet_weight, damage_bias))
    return rows


def metric_value(metrics: dict[str, object], key: str, class_index: int | None = None) -> float | None:
    if class_index is None:
        value = metrics.get(key)
    else:
        values = metrics.get(key, [])
        value = values[class_index] if isinstance(values, list) and len(values) > class_index else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_row(
    args: argparse.Namespace,
    metrics: dict[str, object],
    tta_mode: str,
    unet_weight: float,
    damage_bias: float,
) -> dict[str, object]:
    return {
        "split_role": args.split_role,
        "tta_mode": tta_mode,
        "unet_weight": unet_weight,
        "arch_weight": 1.0 - unet_weight,
        "damage_bias": damage_bias,
        "arch_model": args.arch_model,
        "pixel_accuracy": metric_value(metrics, "pixel_accuracy"),
        "mean_iou": metric_value(metrics, "mean_iou"),
        "iou_background": metric_value(metrics, "iou_per_class", 0),
        "iou_no_damage": metric_value(metrics, "iou_per_class", 1),
        "iou_damaged": metric_value(metrics, "iou_per_class", 2),
        "precision_damaged": metric_value(metrics, "precision_per_class", 2),
        "recall_damaged": metric_value(metrics, "recall_per_class", 2),
        "f1_damaged": metric_value(metrics, "f1_per_class", 2),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]], split_role: str, has_bias_grid: bool) -> None:
    print("CrisisMap AI - Damage Ensemble Evaluation")
    print("=" * 45)
    if has_bias_grid:
        print(
            "WARNING: damage-bias grids must be selected on validation before "
            "final test reporting."
        )
        print(f"Current split role: {split_role}")
        print()
    best_iou = max(rows, key=lambda row: float(row.get("iou_damaged") or -1.0), default=None)
    best_f1 = max(rows, key=lambda row: float(row.get("f1_damaged") or -1.0), default=None)
    if best_iou:
        print(
            "Best by IoU damaged: "
            f"tta={best_iou['tta_mode']}, w_unet={best_iou['unet_weight']}, "
            f"bias={best_iou['damage_bias']}, "
            f"IoU={format_metric(best_iou['iou_damaged'])}, "
            f"F1={format_metric(best_iou['f1_damaged'])}"
        )
    if best_f1:
        print(
            "Best by F1 damaged: "
            f"tta={best_f1['tta_mode']}, w_unet={best_f1['unet_weight']}, "
            f"bias={best_f1['damage_bias']}, "
            f"IoU={format_metric(best_f1['iou_damaged'])}, "
            f"F1={format_metric(best_f1['f1_damaged'])}"
        )


def format_metric(value: object) -> str:
    return "nan" if value is None else f"{float(value):.6f}"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")
        loader = make_loader(args, device)
        unet_model = load_unet(args, device)
        arch_model = load_arch(args, device)
        started_at = time.time()
        rows = evaluate(unet_model, arch_model, loader, args, device, use_amp)
        payload = {
            "config": {
                "unet_checkpoint": str(args.unet_checkpoint),
                "arch_checkpoint": str(args.arch_checkpoint),
                "arch_model": args.arch_model,
                "arch_model_metadata": damage_model_metadata(args.arch_model),
                "root": str(args.root),
                "split_csv": str(args.split_csv),
                "split_role": args.split_role,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "target_mode": args.target_mode,
                "device": str(device),
                "amp": use_amp,
                "num_workers": args.num_workers,
                "weights": args.weights,
                "damage_biases": args.damage_biases,
                "tta_modes": args.tta_modes,
            },
            "elapsed_seconds": time.time() - started_at,
            "rows": rows,
        }
        write_json(args.output_json, payload)
        write_csv(args.output_csv, rows)
        print_summary(
            rows,
            split_role=args.split_role,
            has_bias_grid=len(args.damage_biases) > 1 or args.damage_biases != [0.0],
        )
        print()
        print(f"Saved JSON: {args.output_json}")
        print(f"Saved CSV: {args.output_csv}")
    except (
        DamageEnsembleError,
        DamageModelFactoryError,
        XBDDatasetError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

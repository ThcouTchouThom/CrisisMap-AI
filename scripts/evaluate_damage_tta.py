"""Evaluate a damage segmentation checkpoint with test-time augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import (  # noqa: E402
    CLASS_LABELS,
    confusion_matrix,
    extract_state_dict,
    load_checkpoint_file,
    metrics_from_confusion,
)
from crisismap.models.unet import UNet  # noqa: E402


TTA_MODES = {"none", "flips", "rot90", "d4"}
CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [35, 170, 80],
        [220, 45, 45],
    ],
    dtype=np.uint8,
)
CLASS_NAMES = ["background", "no damage", "damaged"]


class TTAEvaluationError(Exception):
    """Raised when TTA evaluation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a damage U-Net checkpoint with TTA.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-mode", choices=["3-class"], default="3-class")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--tta-modes",
        nargs="+",
        choices=sorted(TTA_MODES),
        default=["none"],
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--save-examples-dir", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise TTAEvaluationError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise TTAEvaluationError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise TTAEvaluationError("--num-workers must be non-negative.")
    if args.num_examples < 0:
        raise TTAEvaluationError("--num-examples must be non-negative.")
    if args.base_channels <= 0:
        raise TTAEvaluationError("--base-channels must be positive.")
    if not args.tta_modes:
        raise TTAEvaluationError("--tta-modes must contain at least one mode.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TTAEvaluationError("CUDA was requested, but CUDA is not available.")
    return device


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def load_dataset(args: argparse.Namespace) -> XBDPairDataset:
    return XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
        augment_mode="none",
    )


def make_loader(
    dataset: XBDPairDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_model(args: argparse.Namespace, device: torch.device) -> UNet:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise TTAEvaluationError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise TTAEvaluationError(f"Checkpoint path is not a file: {checkpoint_path}")

    model = UNet(
        in_channels=6,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    try:
        model.load_state_dict(extract_state_dict(checkpoint))
    except RuntimeError as exc:
        raise TTAEvaluationError(
            "Checkpoint weights do not match the local U-Net configuration. "
            "Check --target-mode and --base-channels."
        ) from exc
    model.eval()
    return model


def tta_ops(mode: str) -> list[tuple[int, bool, bool]]:
    if mode == "none":
        return [(0, False, False)]
    if mode == "flips":
        return [
            (0, False, False),
            (0, True, False),
            (0, False, True),
            (0, True, True),
        ]
    if mode == "rot90":
        return [(k, False, False) for k in range(4)]
    if mode == "d4":
        return [(k, False, False) for k in range(4)] + [(k, True, False) for k in range(4)]
    raise TTAEvaluationError(f"Unsupported TTA mode: {mode}")


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


@torch.no_grad()
def predict_logits_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    mode: str,
    use_amp: bool,
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    ops = tta_ops(mode)
    for op in ops:
        view = apply_op(images, op)
        with autocast_context(use_amp):
            logits = model(view)
        logits = invert_op(logits, op).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    if logits_sum is None:
        raise TTAEvaluationError(f"No TTA operations for mode: {mode}")
    return logits_sum / float(len(ops))


@torch.no_grad()
def evaluate_modes(
    model: torch.nn.Module,
    loader: DataLoader,
    modes: list[str],
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    save_examples_dir: Path | None,
    num_examples: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    confusion_by_mode = {
        mode: torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        for mode in modes
    }
    examples_saved = 0
    started_at = time.time()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        predictions: dict[str, torch.Tensor] = {}

        for mode in modes:
            logits = predict_logits_tta(model, images, mode, use_amp)
            preds = torch.argmax(logits, dim=1)
            predictions[mode] = preds
            confusion_by_mode[mode] += confusion_matrix(preds, targets, num_classes)

        if save_examples_dir is not None and examples_saved < num_examples:
            examples_saved = save_batch_examples(
                batch=batch,
                predictions=predictions,
                save_dir=save_examples_dir,
                start_index=examples_saved,
                max_examples=num_examples,
            )

    metrics_by_mode = {
        mode: metrics_from_confusion(confusion)
        for mode, confusion in confusion_by_mode.items()
    }
    metadata = {
        "elapsed_seconds": time.time() - started_at,
        "examples_saved": examples_saved,
    }
    return metrics_by_mode, metadata


def save_batch_examples(
    batch: dict[str, object],
    predictions: dict[str, torch.Tensor],
    save_dir: Path,
    start_index: int,
    max_examples: int,
) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    targets = batch["target"].detach().cpu()
    pair_ids = batch.get("pair_id", [])
    raw_preds = predictions.get("none")
    if raw_preds is None:
        first_mode = next(iter(predictions))
        raw_preds = predictions[first_mode]
    raw_preds = raw_preds.detach().cpu()

    comparison_modes = [mode for mode in predictions if mode != "none"]
    if not comparison_modes:
        comparison_modes = ["none"]

    saved = start_index
    batch_size = images.shape[0]
    for item_index in range(batch_size):
        if saved >= max_examples:
            break
        pair_id = (
            str(pair_ids[item_index])
            if isinstance(pair_ids, (list, tuple)) and item_index < len(pair_ids)
            else f"sample_{saved:03d}"
        )
        for mode in comparison_modes:
            pred = predictions[mode].detach().cpu()[item_index]
            figure_path = save_dir / f"{saved:03d}_{sanitize_filename(pair_id)}_{mode}.png"
            save_example_figure(
                image_tensor=images[item_index],
                target=targets[item_index],
                raw_pred=raw_preds[item_index],
                tta_pred=pred,
                mode=mode,
                output_path=figure_path,
            )
        saved += 1
    return saved


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def tensor_rgb(image: torch.Tensor, start_channel: int) -> np.ndarray:
    rgb = image[start_channel : start_channel + 3].permute(1, 2, 0).numpy()
    return np.clip(rgb, 0.0, 1.0)


def colorize_mask(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.numpy().astype(np.int64)
    mask_np = np.clip(mask_np, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[mask_np]


def overlay_prediction(post_image: np.ndarray, pred: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    colors = colorize_mask(pred).astype(np.float32) / 255.0
    pred_np = pred.numpy()
    overlay = post_image.copy()
    building = pred_np > 0
    overlay[building] = (1.0 - alpha) * overlay[building] + alpha * colors[building]
    return np.clip(overlay, 0.0, 1.0)


def save_example_figure(
    image_tensor: torch.Tensor,
    target: torch.Tensor,
    raw_pred: torch.Tensor,
    tta_pred: torch.Tensor,
    mode: str,
    output_path: Path,
) -> None:
    pre = tensor_rgb(image_tensor, 0)
    post = tensor_rgb(image_tensor, 3)
    panels = [
        ("Pre-disaster", pre),
        ("Post-disaster", post),
        ("Ground truth", colorize_mask(target)),
        ("Raw prediction", colorize_mask(raw_pred)),
        (f"TTA prediction ({mode})", colorize_mask(tta_pred)),
        ("TTA overlay", overlay_prediction(post, tta_pred)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    legend = [
        Patch(facecolor=np.asarray(CLASS_COLORS[index]) / 255.0, label=f"{index}: {name}")
        for index, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


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


def build_rows(metrics_by_mode: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode, metrics in metrics_by_mode.items():
        rows.append(
            {
                "tta_mode": mode,
                "pixel_accuracy": metric_value(metrics, "pixel_accuracy"),
                "mean_iou": metric_value(metrics, "mean_iou"),
                "iou_background": metric_value(metrics, "iou_per_class", 0),
                "iou_no_damage": metric_value(metrics, "iou_per_class", 1),
                "iou_damaged": metric_value(metrics, "iou_per_class", 2),
                "precision_damaged": metric_value(metrics, "precision_per_class", 2),
                "recall_damaged": metric_value(metrics, "recall_per_class", 2),
                "f1_damaged": metric_value(metrics, "f1_per_class", 2),
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tta_mode",
        "pixel_accuracy",
        "mean_iou",
        "iou_background",
        "iou_no_damage",
        "iou_damaged",
        "precision_damaged",
        "recall_damaged",
        "f1_damaged",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    raw = next((row for row in rows if row["tta_mode"] == "none"), None)
    print("CrisisMap AI - Damage TTA Evaluation")
    print("=" * 39)
    for row in rows:
        print(
            f"{row['tta_mode']}: "
            f"mean IoU={format_metric(row['mean_iou'])}, "
            f"IoU damaged={format_metric(row['iou_damaged'])}, "
            f"precision damaged={format_metric(row['precision_damaged'])}, "
            f"recall damaged={format_metric(row['recall_damaged'])}, "
            f"F1 damaged={format_metric(row['f1_damaged'])}"
        )
    if raw is None:
        return
    print()
    print("Deltas vs none")
    for row in rows:
        if row["tta_mode"] == "none":
            continue
        delta_text = []
        for key in [
            "mean_iou",
            "iou_damaged",
            "precision_damaged",
            "recall_damaged",
            "f1_damaged",
        ]:
            delta = optional_delta(row.get(key), raw.get(key))
            delta_text.append(f"{key}={format_delta(delta)}")
        print(f"{row['tta_mode']}: " + ", ".join(delta_text))


def optional_delta(value: object, baseline: object) -> float | None:
    try:
        if value is None or baseline is None:
            return None
        return float(value) - float(baseline)
    except (TypeError, ValueError):
        return None


def format_metric(value: object) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.6f}"


def format_delta(value: float | None) -> str:
    return "nan" if value is None else f"{value:+.6f}"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        model = load_model(args, device)
        modes = list(dict.fromkeys(["none", *args.tta_modes]))

        metrics_by_mode, metadata = evaluate_modes(
            model=model,
            loader=loader,
            modes=modes,
            device=device,
            num_classes=len(CLASS_LABELS[args.target_mode]),
            use_amp=use_amp,
            save_examples_dir=args.save_examples_dir,
            num_examples=args.num_examples,
        )
        rows = build_rows(metrics_by_mode)
        payload = {
            "config": {
                "checkpoint": str(args.checkpoint),
                "root": str(args.root),
                "split_csv": str(args.split_csv),
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "target_mode": args.target_mode,
                "device": str(device),
                "amp": use_amp,
                "num_workers": args.num_workers,
                "tta_modes": modes,
                "base_channels": args.base_channels,
            },
            "dataset_size": len(dataset),
            "class_labels": CLASS_LABELS[args.target_mode],
            "metrics_by_mode": metrics_by_mode,
            "summary_rows": rows,
            **metadata,
        }
        write_json(args.output_json, payload)
        write_csv(args.output_csv, rows)
        print_summary(rows)
        print()
        print(f"Saved JSON: {args.output_json}")
        print(f"Saved CSV: {args.output_csv}")
        if args.save_examples_dir is not None:
            print(f"Saved examples: {args.save_examples_dir}")
    except (
        TTAEvaluationError,
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

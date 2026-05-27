"""Evaluate a binary building segmentation checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_building_segmentation import (  # noqa: E402
    MODEL_CHOICES,
    BuildingTrainingError,
    XBDDatasetError,
    build_model,
    clean_state_dict,
    input_channels,
    make_dataset,
    make_loader,
    metrics_from_counts,
    normalize_logits,
    resolve_device,
)


INPUT_MODES = {"pre", "post", "pre-post"}
MASK_COLORS = np.asarray([[0, 0, 0], [35, 170, 80]], dtype=np.uint8)
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


class BuildingEvaluationError(Exception):
    """Raised when building segmentation evaluation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a binary building segmentation checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="unetplusplus_effb3")
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), default="pre")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--target-mode",
        choices=["building-binary"],
        default="building-binary",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--save-examples-dir", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Probability thresholds to evaluate.",
    )
    parser.add_argument(
        "--object-iou-threshold",
        type=float,
        default=0.1,
        help="Minimum component IoU used for object-level precision/recall.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise BuildingEvaluationError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise BuildingEvaluationError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise BuildingEvaluationError("--num-workers must be non-negative.")
    if args.num_examples < 0:
        raise BuildingEvaluationError("--num-examples must be non-negative.")
    if not args.thresholds:
        raise BuildingEvaluationError("--thresholds must contain at least one value.")
    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise BuildingEvaluationError("All thresholds must be between 0 and 1.")
    if not 0.0 <= args.object_iou_threshold <= 1.0:
        raise BuildingEvaluationError("--object-iou-threshold must be between 0 and 1.")


def load_checkpoint(path: Path, device: torch.device) -> object:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise BuildingEvaluationError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise BuildingEvaluationError(f"Checkpoint path is not a file: {checkpoint_path}")
    try:
        try:
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=device)
    except (OSError, RuntimeError) as exc:
        raise BuildingEvaluationError(
            f"Could not load checkpoint '{checkpoint_path}': {exc}"
        ) from exc


def load_model_from_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict[str, object]]:
    model, actual_model = build_model(args.model, input_channels(args.input_mode), device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_actual_model": checkpoint.get("actual_model"),
            "checkpoint_config": checkpoint.get("config"),
        }
    else:
        state_dict = checkpoint
        metadata = {
            "checkpoint_epoch": None,
            "checkpoint_actual_model": None,
            "checkpoint_config": None,
        }

    try:
        model.load_state_dict(clean_state_dict(state_dict))
    except RuntimeError as exc:
        raise BuildingEvaluationError(
            "Checkpoint weights do not match the requested model/input mode. "
            "Check --model and --input-mode."
        ) from exc

    model.eval()
    return model, actual_model, metadata


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    save_examples_dir: Path | None,
    num_examples: int,
    input_mode: str,
    thresholds: list[float],
    object_iou_threshold: float,
) -> tuple[dict[str, dict[str, object]], int]:
    counts_by_threshold = {
        threshold_key(threshold): {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for threshold in thresholds
    }
    objects_by_threshold = {
        threshold_key(threshold): {
            "gt_objects": 0,
            "pred_objects": 0,
            "matched_gt_objects": 0,
            "matched_pred_objects": 0,
        }
        for threshold in thresholds
    }
    saved_examples = 0
    total_samples = 0

    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast_context(use_amp):
            logits = normalize_logits(model(images))
            probabilities = torch.sigmoid(logits).squeeze(1)

        target_bool = targets > 0
        probability_cpu = probabilities.detach().cpu().numpy()
        target_cpu = target_bool.detach().cpu().numpy()

        example_predictions = None
        for threshold in thresholds:
            key = threshold_key(threshold)
            predictions = probabilities >= threshold
            update_pixel_counts(counts_by_threshold[key], predictions, target_bool)

            pred_cpu = predictions.detach().cpu().numpy()
            for sample_index in range(pred_cpu.shape[0]):
                object_counts = object_metrics_for_sample(
                    pred_cpu[sample_index],
                    target_cpu[sample_index],
                    object_iou_threshold,
                )
                update_object_counts(objects_by_threshold[key], object_counts)

            if abs(threshold - 0.5) < 1e-9:
                example_predictions = predictions

        if example_predictions is None:
            nearest_threshold = min(thresholds, key=lambda value: abs(value - 0.5))
            example_predictions = probabilities >= nearest_threshold

        if save_examples_dir is not None and saved_examples < num_examples:
            saved_examples = save_batch_examples(
                output_dir=save_examples_dir,
                batch=batch,
                predictions=example_predictions,
                probabilities=probability_cpu,
                already_saved=saved_examples,
                max_examples=num_examples,
                input_mode=input_mode,
            )

        total_samples += int(images.shape[0])

    metrics_by_threshold = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        metrics = metrics_from_counts(counts_by_threshold[key])
        metrics["confusion_counts"] = counts_by_threshold[key]
        metrics["object_metrics"] = object_metrics_from_counts(objects_by_threshold[key])
        metrics_by_threshold[key] = metrics
    return metrics_by_threshold, total_samples


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def update_pixel_counts(
    counts: dict[str, int],
    predictions: torch.Tensor,
    target_bool: torch.Tensor,
) -> None:
    counts["tp"] += int(torch.count_nonzero(predictions & target_bool).item())
    counts["tn"] += int(torch.count_nonzero((~predictions) & (~target_bool)).item())
    counts["fp"] += int(torch.count_nonzero(predictions & (~target_bool)).item())
    counts["fn"] += int(torch.count_nonzero((~predictions) & target_bool).item())


def update_object_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def object_metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    return {
        **counts,
        "object_recall": safe_divide(
            float(counts["matched_gt_objects"]),
            float(counts["gt_objects"]),
        ),
        "object_precision": safe_divide(
            float(counts["matched_pred_objects"]),
            float(counts["pred_objects"]),
        ),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=np.uint8)
    try:
        import cv2  # type: ignore

        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        return labels.astype(np.int32, copy=False), int(num_labels - 1)
    except Exception:
        return connected_components_fallback(mask)


def connected_components_fallback(mask: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0
    for row in range(height):
        for col in range(width):
            if mask[row, col] == 0 or labels[row, col] != 0:
                continue
            current_label += 1
            stack = [(row, col)]
            labels[row, col] = current_label
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx] != 0
                            and labels[ny, nx] == 0
                        ):
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))
    return labels, current_label


def object_metrics_for_sample(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float,
) -> dict[str, int]:
    pred_labels, pred_count = connected_components(pred_mask)
    gt_labels, gt_count = connected_components(gt_mask)
    result = {
        "gt_objects": gt_count,
        "pred_objects": pred_count,
        "matched_gt_objects": 0,
        "matched_pred_objects": 0,
    }
    if gt_count == 0 or pred_count == 0:
        return result

    gt_areas = np.bincount(gt_labels.ravel(), minlength=gt_count + 1).astype(np.float64)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1).astype(np.float64)
    overlap_mask = (gt_labels > 0) & (pred_labels > 0)
    if not np.any(overlap_mask):
        return result

    combined = gt_labels[overlap_mask] * (pred_count + 1) + pred_labels[overlap_mask]
    pair_ids, intersections = np.unique(combined, return_counts=True)
    best_gt_iou: defaultdict[int, float] = defaultdict(float)
    best_pred_iou: defaultdict[int, float] = defaultdict(float)
    for pair_id, intersection in zip(pair_ids, intersections):
        gt_id = int(pair_id // (pred_count + 1))
        pred_id = int(pair_id % (pred_count + 1))
        union = gt_areas[gt_id] + pred_areas[pred_id] - float(intersection)
        iou = safe_divide(float(intersection), union)
        best_gt_iou[gt_id] = max(best_gt_iou[gt_id], iou)
        best_pred_iou[pred_id] = max(best_pred_iou[pred_id], iou)

    result["matched_gt_objects"] = sum(
        1 for gt_id in range(1, gt_count + 1) if best_gt_iou[gt_id] >= iou_threshold
    )
    result["matched_pred_objects"] = sum(
        1 for pred_id in range(1, pred_count + 1) if best_pred_iou[pred_id] >= iou_threshold
    )
    return result


def save_batch_examples(
    output_dir: Path,
    batch: dict[str, object],
    predictions: torch.Tensor,
    probabilities: np.ndarray,
    already_saved: int,
    max_examples: int,
    input_mode: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    targets = batch["target"].detach().cpu().numpy()
    preds = predictions.detach().cpu().numpy().astype(np.uint8)
    pair_ids = batch["pair_id"]

    for index in range(images.shape[0]):
        if already_saved >= max_examples:
            break
        save_example_figure(
            output_dir=output_dir,
            pair_id=str(pair_ids[index]),
            image_tensor=images[index],
            target=targets[index],
            prediction=preds[index],
            probability=probabilities[index],
            input_mode=input_mode,
        )
        already_saved += 1
    return already_saved


def save_example_figure(
    output_dir: Path,
    pair_id: str,
    image_tensor: torch.Tensor,
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    input_mode: str,
) -> None:
    input_images = split_input_images(image_tensor, input_mode)
    target_rgb = colorize_mask(target)
    prediction_rgb = colorize_mask(prediction)
    overlay = overlay_mask(input_images[-1], prediction)

    panels = []
    if input_mode == "pre-post":
        panels.extend([("Pre-disaster input", input_images[0]), ("Post-disaster input", input_images[1])])
    else:
        panels.append((f"{input_mode} input", input_images[0]))
    panels.extend(
        [
            ("Ground truth building mask", target_rgb),
            ("Predicted building mask", prediction_rgb),
            ("Building probability", probability),
            ("Prediction overlay", overlay),
        ]
    )

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (title, image) in zip(axes_array, panels):
        axis.imshow(image, cmap="viridis" if title == "Building probability" else None)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes_array[len(panels) :]:
        axis.axis("off")

    legend = [
        Patch(color=MASK_COLORS[0] / 255.0, label="0: background"),
        Patch(color=MASK_COLORS[1] / 255.0, label="1: building"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2)
    fig.suptitle(pair_id)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_dir / f"{safe_filename(pair_id)}_building_eval.png", dpi=150)
    plt.close(fig)


def split_input_images(image_tensor: torch.Tensor, input_mode: str) -> list[np.ndarray]:
    if input_mode == "pre-post":
        return [
            tensor_to_uint8_image(image_tensor[:3]),
            tensor_to_uint8_image(image_tensor[3:6]),
        ]
    return [tensor_to_uint8_image(image_tensor[:3])]


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    return MASK_COLORS[np.clip(mask, 0, 1)]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    building = mask > 0
    overlay[building] = (
        (1.0 - alpha) * overlay[building] + alpha * MASK_COLORS[1].astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".rstrip("0").rstrip(".")


def preferred_metrics(metrics_by_threshold: dict[str, dict[str, object]]) -> dict[str, object]:
    if "0.5" in metrics_by_threshold:
        return metrics_by_threshold["0.5"]
    return next(iter(metrics_by_threshold.values()))


def build_payload(
    args: argparse.Namespace,
    device: torch.device,
    actual_model: str,
    checkpoint_metadata: dict[str, object],
    dataset_size: int,
    evaluated_samples: int,
    elapsed_seconds: float,
    metrics_by_threshold: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "config": {
            "checkpoint": str(args.checkpoint),
            "root": str(args.root),
            "split_csv": str(args.split_csv),
            "model_requested": args.model,
            "model_actual": actual_model,
            "input_mode": args.input_mode,
            "image_size": args.image_size,
            "target_mode": args.target_mode,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
            "thresholds": args.thresholds,
            "object_iou_threshold": args.object_iou_threshold,
        },
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "elapsed_seconds": elapsed_seconds,
        "metrics": preferred_metrics(metrics_by_threshold),
        "metrics_by_threshold": metrics_by_threshold,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_csv(path: Path, payload: dict[str, object]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "split_csv",
        "model_actual",
        "input_mode",
        "image_size",
        "evaluated_samples",
        "threshold",
        "pixel_accuracy",
        "mean_iou",
        "background_iou",
        "building_iou",
        "building_precision",
        "building_recall",
        "building_f1",
        "object_recall",
        "object_precision",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for threshold, metrics in payload["metrics_by_threshold"].items():
            object_metrics = metrics.get("object_metrics", {})
            writer.writerow(
                {
                    "checkpoint": payload["config"]["checkpoint"],
                    "split_csv": payload["config"]["split_csv"],
                    "model_actual": payload["config"]["model_actual"],
                    "input_mode": payload["config"]["input_mode"],
                    "image_size": payload["config"]["image_size"],
                    "evaluated_samples": payload["evaluated_samples"],
                    "threshold": threshold,
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "mean_iou": metrics["mean_iou"],
                    "background_iou": metrics["background_iou"],
                    "building_iou": metrics["building_iou"],
                    "building_precision": metrics["building_precision"],
                    "building_recall": metrics["building_recall"],
                    "building_f1": metrics["building_f1"],
                    "object_recall": object_metrics.get("object_recall"),
                    "object_precision": object_metrics.get("object_precision"),
                }
            )


def print_summary(payload: dict[str, object]) -> None:
    print("CrisisMap AI - Building Segmentation Evaluation")
    print("=" * 52)
    print(f"Device: {payload['config']['device']}")
    print(f"Model: {payload['config']['model_actual']}")
    print(f"Input mode: {payload['config']['input_mode']}")
    print(f"Evaluated samples: {payload['evaluated_samples']}")
    print()
    print("threshold | building IoU | building F1 | precision | recall | obj P | obj R")
    for threshold, metrics in payload["metrics_by_threshold"].items():
        object_metrics = metrics["object_metrics"]
        print(
            f"{threshold:>9} | "
            f"{metrics['building_iou']:.6f} | "
            f"{metrics['building_f1']:.6f} | "
            f"{metrics['building_precision']:.6f} | "
            f"{metrics['building_recall']:.6f} | "
            f"{object_metrics['object_precision']:.6f} | "
            f"{object_metrics['object_recall']:.6f}"
        )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")

        dataset = make_dataset(
            args.root,
            args.split_csv,
            args.image_size,
            args.input_mode,
            args.target_mode,
            max_samples=None,
            augment_mode="none",
        )
        loader = make_loader(
            dataset,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            device=device,
        )
        model, actual_model, checkpoint_metadata = load_model_from_checkpoint(args, device)

        started_at = time.time()
        metrics_by_threshold, evaluated_samples = evaluate_model(
            model,
            loader,
            device,
            use_amp,
            args.save_examples_dir,
            args.num_examples,
            args.input_mode,
            args.thresholds,
            args.object_iou_threshold,
        )
        payload = build_payload(
            args=args,
            device=device,
            actual_model=actual_model,
            checkpoint_metadata=checkpoint_metadata,
            dataset_size=len(dataset),
            evaluated_samples=evaluated_samples,
            elapsed_seconds=time.time() - started_at,
            metrics_by_threshold=metrics_by_threshold,
        )
        write_json(args.output_json, payload)
        if args.output_csv is not None:
            write_csv(args.output_csv, payload)

        print_summary(payload)
        print()
        print(f"Saved JSON: {args.output_json.expanduser().resolve()}")
        if args.output_csv is not None:
            print(f"Saved CSV: {args.output_csv.expanduser().resolve()}")
        if args.save_examples_dir is not None:
            print(f"Saved examples in: {args.save_examples_dir.expanduser().resolve()}")
    except (
        BuildingEvaluationError,
        BuildingTrainingError,
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

"""Evaluate a binary building segmentation checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    BuildingTrainingError,
    clean_state_dict,
    input_channels,
    build_model,
    make_dataset,
    make_loader,
    metrics_from_counts,
    normalize_logits,
    resolve_device,
    XBDDatasetError,
)


INPUT_MODES = {"pre", "post", "pre-post"}
MODEL_CHOICES = {"unet", "unetplusplus_effb3"}
MASK_COLORS = np.asarray([[0, 0, 0], [35, 170, 80]], dtype=np.uint8)


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
) -> tuple[dict[str, object], int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    saved_examples = 0
    total_samples = 0

    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast_context(use_amp):
            logits = normalize_logits(model(images))
            probabilities = torch.sigmoid(logits).squeeze(1)

        predictions = probabilities >= 0.5
        target_bool = targets > 0

        counts["tp"] += int(torch.count_nonzero(predictions & target_bool).item())
        counts["tn"] += int(torch.count_nonzero((~predictions) & (~target_bool)).item())
        counts["fp"] += int(torch.count_nonzero(predictions & (~target_bool)).item())
        counts["fn"] += int(torch.count_nonzero((~predictions) & target_bool).item())

        if save_examples_dir is not None and saved_examples < num_examples:
            saved_examples = save_batch_examples(
                output_dir=save_examples_dir,
                batch=batch,
                predictions=predictions,
                already_saved=saved_examples,
                max_examples=num_examples,
                input_mode=input_mode,
            )

        total_samples += int(images.shape[0])

    metrics = metrics_from_counts(counts)
    metrics["confusion_counts"] = counts
    return metrics, total_samples


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    from contextlib import nullcontext

    return nullcontext()


def save_batch_examples(
    output_dir: Path,
    batch: dict[str, object],
    predictions: torch.Tensor,
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
            ("Prediction overlay", overlay),
        ]
    )

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (title, image) in zip(axes_array, panels):
        axis.imshow(image)
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


def build_payload(
    args: argparse.Namespace,
    device: torch.device,
    actual_model: str,
    checkpoint_metadata: dict[str, object],
    dataset_size: int,
    evaluated_samples: int,
    elapsed_seconds: float,
    metrics: dict[str, object],
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
        },
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "elapsed_seconds": elapsed_seconds,
        "metrics": metrics,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_csv(path: Path, payload: dict[str, object]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    fieldnames = [
        "checkpoint",
        "split_csv",
        "model_actual",
        "input_mode",
        "image_size",
        "evaluated_samples",
        "pixel_accuracy",
        "mean_iou",
        "background_iou",
        "building_iou",
        "building_precision",
        "building_recall",
        "building_f1",
    ]
    row = {
        "checkpoint": payload["config"]["checkpoint"],
        "split_csv": payload["config"]["split_csv"],
        "model_actual": payload["config"]["model_actual"],
        "input_mode": payload["config"]["input_mode"],
        "image_size": payload["config"]["image_size"],
        "evaluated_samples": payload["evaluated_samples"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "background_iou": metrics["background_iou"],
        "building_iou": metrics["building_iou"],
        "building_precision": metrics["building_precision"],
        "building_recall": metrics["building_recall"],
        "building_f1": metrics["building_f1"],
    }
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def print_summary(payload: dict[str, object]) -> None:
    metrics = payload["metrics"]
    print("CrisisMap AI - Building Segmentation Evaluation")
    print("=" * 52)
    print(f"Device: {payload['config']['device']}")
    print(f"Model: {payload['config']['model_actual']}")
    print(f"Input mode: {payload['config']['input_mode']}")
    print(f"Evaluated samples: {payload['evaluated_samples']}")
    print()
    print(f"Pixel accuracy: {metrics['pixel_accuracy']:.6f}")
    print(f"Mean IoU: {metrics['mean_iou']:.6f}")
    print(f"Background IoU: {metrics['background_iou']:.6f}")
    print(f"Building IoU: {metrics['building_iou']:.6f}")
    print(f"Building precision: {metrics['building_precision']:.6f}")
    print(f"Building recall: {metrics['building_recall']:.6f}")
    print(f"Building F1: {metrics['building_f1']:.6f}")


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
        metrics, evaluated_samples = evaluate_model(
            model,
            loader,
            device,
            use_amp,
            args.save_examples_dir,
            args.num_examples,
            args.input_mode,
        )
        payload = build_payload(
            args=args,
            device=device,
            actual_model=actual_model,
            checkpoint_metadata=checkpoint_metadata,
            dataset_size=len(dataset),
            evaluated_samples=evaluated_samples,
            elapsed_seconds=time.time() - started_at,
            metrics=metrics,
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

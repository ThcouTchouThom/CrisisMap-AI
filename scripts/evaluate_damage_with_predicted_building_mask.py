"""Compare damage predictions with predicted and oracle building masks.

This is an evaluation-only downstream experiment. It does not change training,
model weights, or existing outputs.
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
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import (  # noqa: E402
    CLASS_LABELS,
    confusion_matrix,
    extract_state_dict,
    load_checkpoint_file,
    metrics_from_confusion,
)
from crisismap.models.unet import UNet  # noqa: E402
from evaluate_damage_tta import (  # noqa: E402
    TTA_MODES,
    TTAEvaluationError,
    apply_op,
    invert_op,
    tta_ops,
)
from train_building_segmentation import (  # noqa: E402
    MODEL_CHOICES as BUILDING_MODEL_CHOICES,
    BuildingTrainingError,
    build_model as build_building_model,
    clean_state_dict as clean_building_state_dict,
    input_channels as building_input_channels,
    normalize_logits as normalize_building_logits,
)


CLASS_NAMES = ["background", "no damage", "damaged"]
CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [35, 170, 80],
        [220, 45, 45],
    ],
    dtype=np.uint8,
)
BUILDING_COLORS = np.asarray([[0, 0, 0], [35, 170, 80]], dtype=np.uint8)
PREDICTED_BUILDING_MODES = [
    "predicted_building_clip",
    "predicted_building_component_majority",
]
ORACLE_MODES = [
    "oracle_building_clip",
    "oracle_building_component_majority",
]
ALL_MODE_ORDER = [
    "raw",
    "predicted_building_clip",
    "predicted_building_component_majority",
    "oracle_building_clip",
    "oracle_building_component_majority",
]


class DownstreamEvaluationError(Exception):
    """Raised when downstream building-mask evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw damage predictions with predicted building mask and "
            "oracle building mask post-processing."
        )
    )
    parser.add_argument("--damage-checkpoint", required=True, type=Path)
    parser.add_argument("--building-checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--target-mode", choices=["3-class"], default="3-class")
    parser.add_argument("--damage-model", choices=["unet"], default="unet")
    parser.add_argument(
        "--damage-tta",
        choices=sorted(TTA_MODES),
        default="none",
        help="Test-time augmentation mode for the damage model only.",
    )
    parser.add_argument(
        "--building-model",
        choices=sorted(BUILDING_MODEL_CHOICES),
        default="unetplusplus_effb3",
    )
    parser.add_argument(
        "--building-input-mode",
        choices=["pre", "post", "pre-post"],
        default="pre",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6],
        help="Building probability thresholds for predicted-building modes.",
    )
    parser.add_argument("--component-connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--save-examples-dir", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--damage-base-channels", type=int, default=32)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise DownstreamEvaluationError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise DownstreamEvaluationError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise DownstreamEvaluationError("--num-workers must be non-negative.")
    if args.num_examples < 0:
        raise DownstreamEvaluationError("--num-examples must be non-negative.")
    if args.damage_base_channels <= 0:
        raise DownstreamEvaluationError("--damage-base-channels must be positive.")
    if not args.thresholds:
        raise DownstreamEvaluationError("--thresholds must contain at least one value.")
    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise DownstreamEvaluationError("All thresholds must be between 0 and 1.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DownstreamEvaluationError("CUDA was requested, but CUDA is not available.")
    return device


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


def load_damage_model(args: argparse.Namespace, device: torch.device) -> UNet:
    if args.damage_model != "unet":
        raise DownstreamEvaluationError(f"Unsupported damage model: {args.damage_model}")
    checkpoint_path = args.damage_checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise DownstreamEvaluationError(f"Damage checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise DownstreamEvaluationError(f"Damage checkpoint is not a file: {checkpoint_path}")

    model = UNet(
        in_channels=6,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        base_channels=args.damage_base_channels,
    ).to(device)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


def load_building_checkpoint(path: Path, device: torch.device) -> object:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise DownstreamEvaluationError(f"Building checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise DownstreamEvaluationError(f"Building checkpoint is not a file: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_building_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    try:
        model, _ = build_building_model(
            args.building_model,
            building_input_channels(args.building_input_mode),
            device,
        )
        checkpoint = load_building_checkpoint(args.building_checkpoint, device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(clean_building_state_dict(state_dict))
    except (BuildingTrainingError, RuntimeError, OSError) as exc:
        raise DownstreamEvaluationError(
            "Could not load building model/checkpoint. Check --building-model, "
            "--building-input-mode, and --building-checkpoint."
        ) from exc
    model.eval()
    return model


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def predict_damage_logits_tta(
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
        raise DownstreamEvaluationError(f"No TTA operations for mode: {mode}")
    return logits_sum / float(len(ops))


@torch.no_grad()
def evaluate_downstream(
    damage_model: UNet,
    building_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, dict[str, object]]], int]:
    num_classes = len(CLASS_LABELS[args.target_mode])
    confusions: dict[str, dict[str, torch.Tensor]] = {
        "raw": {"raw": empty_confusion(num_classes, device)},
        "oracle": {
            mode: empty_confusion(num_classes, device)
            for mode in ORACLE_MODES
        },
    }
    for threshold in args.thresholds:
        key = threshold_key(threshold)
        confusions[key] = {
            mode: empty_confusion(num_classes, device)
            for mode in PREDICTED_BUILDING_MODES
        }

    saved_examples = 0
    evaluated_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        use_amp = bool(args.amp and device.type == "cuda")

        damage_logits = predict_damage_logits_tta(
            damage_model,
            images,
            args.damage_tta,
            use_amp,
        )
        raw_preds = torch.argmax(damage_logits, dim=1)
        with autocast_context(use_amp):
            building_logits = normalize_building_logits(
                building_model(select_building_input(images, args.building_input_mode))
            )
            building_probs = torch.sigmoid(building_logits).squeeze(1)

        confusions["raw"]["raw"] += confusion_matrix(raw_preds, targets, num_classes)

        oracle_clip = apply_building_clip(raw_preds, targets > 0)
        oracle_component = component_majority_batch(
            raw_preds,
            targets > 0,
            connectivity=args.component_connectivity,
            device=device,
        )
        confusions["oracle"]["oracle_building_clip"] += confusion_matrix(
            oracle_clip,
            targets,
            num_classes,
        )
        confusions["oracle"]["oracle_building_component_majority"] += confusion_matrix(
            oracle_component,
            targets,
            num_classes,
        )

        predicted_examples: dict[str, dict[str, torch.Tensor]] = {}
        for threshold in args.thresholds:
            key = threshold_key(threshold)
            building_mask = building_probs >= threshold
            predicted_clip = apply_building_clip(raw_preds, building_mask)
            predicted_component = component_majority_batch(
                raw_preds,
                building_mask,
                connectivity=args.component_connectivity,
                device=device,
            )
            confusions[key]["predicted_building_clip"] += confusion_matrix(
                predicted_clip,
                targets,
                num_classes,
            )
            confusions[key]["predicted_building_component_majority"] += confusion_matrix(
                predicted_component,
                targets,
                num_classes,
            )
            if args.save_examples_dir is not None and saved_examples < args.num_examples:
                predicted_examples[key] = {
                    "building_mask": building_mask,
                    "clip": predicted_clip,
                    "component": predicted_component,
                }

        if args.save_examples_dir is not None and saved_examples < args.num_examples:
            saved_examples = save_batch_examples(
                output_dir=args.save_examples_dir,
                batch=batch,
                raw_preds=raw_preds,
                building_probs=building_probs,
                predicted_examples=predicted_examples,
                oracle_component=oracle_component,
                targets=targets,
                already_saved=saved_examples,
                max_examples=args.num_examples,
            )

        evaluated_samples += int(images.shape[0])

    return metrics_from_confusions(confusions), evaluated_samples


def empty_confusion(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)


def select_building_input(images: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "pre":
        return images[:, :3]
    if input_mode == "post":
        return images[:, 3:6]
    if input_mode == "pre-post":
        return images
    raise DownstreamEvaluationError(f"Unsupported building input mode: {input_mode}")


def apply_building_clip(preds: torch.Tensor, building_mask: torch.Tensor) -> torch.Tensor:
    clipped = preds.clone()
    clipped[~building_mask] = 0
    return clipped


def component_majority_batch(
    raw_preds: torch.Tensor,
    building_masks: torch.Tensor,
    connectivity: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for pred_tensor, mask_tensor in zip(raw_preds, building_masks):
        pred = pred_tensor.detach().cpu().numpy().astype(np.int16, copy=False)
        building_mask = mask_tensor.detach().cpu().numpy().astype(bool, copy=False)
        outputs.append(component_majority_single(pred, building_mask, connectivity))
    return torch.from_numpy(np.stack(outputs, axis=0)).to(device=device, dtype=torch.long)


def component_majority_single(
    raw_pred: np.ndarray,
    building_mask: np.ndarray,
    connectivity: int,
) -> np.ndarray:
    labels, component_count = label_connected_components(building_mask, connectivity)
    output = np.zeros_like(raw_pred, dtype=np.int64)
    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        component_preds = raw_pred[component_mask]
        no_damage_count = int(np.count_nonzero(component_preds == 1))
        damaged_count = int(np.count_nonzero(component_preds == 2))
        component_class = 2 if damaged_count > no_damage_count else 1
        output[component_mask] = component_class
    return output


def label_connected_components(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    structure = (
        np.ones((3, 3), dtype=np.uint8)
        if connectivity == 8
        else np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    )

    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)
    except ImportError:
        pass

    try:
        from skimage.measure import label as skimage_label  # type: ignore

        sk_connectivity = 2 if connectivity == 8 else 1
        labels = skimage_label(mask, connectivity=sk_connectivity, background=0)
        return labels.astype(np.int32, copy=False), int(labels.max())
    except ImportError:
        return fallback_connected_components(mask, connectivity)


def fallback_connected_components(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0
    neighbors = (
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if connectivity == 8
        else [(-1, 0), (0, -1), (0, 1), (1, 0)]
    )
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            current_label += 1
            labels[start_y, start_x] = current_label
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            while queue:
                y, x = queue.popleft()
                for dy, dx in neighbors:
                    next_y = y + dy
                    next_x = x + dx
                    if not (0 <= next_y < height and 0 <= next_x < width):
                        continue
                    if not mask[next_y, next_x] or labels[next_y, next_x] != 0:
                        continue
                    labels[next_y, next_x] = current_label
                    queue.append((next_y, next_x))
    return labels, current_label


def metrics_from_confusions(
    confusions: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for threshold, modes in confusions.items():
        result[threshold] = {}
        for mode, confusion in modes.items():
            result[threshold][mode] = add_named_metrics(metrics_from_confusion(confusion))
    return result


def add_named_metrics(metrics: dict[str, object]) -> dict[str, object]:
    enriched = dict(metrics)
    iou = enriched["iou_per_class"]
    precision = enriched["precision_per_class"]
    recall = enriched["recall_per_class"]
    f1 = enriched["f1_per_class"]
    enriched.update(
        {
            "iou_background": iou[0],
            "iou_no_damage": iou[1],
            "iou_damaged": iou[2],
            "precision_damaged": precision[2],
            "recall_damaged": recall[2],
            "f1_damaged": f1[2],
        }
    )
    return enriched


def threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".rstrip("0").rstrip(".")


def flatten_metrics(
    metrics_by_threshold: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(build_row("raw", "raw", metrics_by_threshold["raw"]["raw"]))
    for threshold in sorted(
        (key for key in metrics_by_threshold if key not in {"raw", "oracle"}),
        key=float,
    ):
        for mode in PREDICTED_BUILDING_MODES:
            rows.append(build_row(mode, threshold, metrics_by_threshold[threshold][mode]))
    for mode in ORACLE_MODES:
        rows.append(build_row(mode, "oracle", metrics_by_threshold["oracle"][mode]))
    return rows


def build_row(mode: str, threshold: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "mode": mode,
        "threshold": threshold,
        "damage_tta": None,
        "pixel_accuracy": metrics.get("pixel_accuracy"),
        "mean_iou": metrics.get("mean_iou"),
        "iou_background": metrics.get("iou_background"),
        "iou_no_damage": metrics.get("iou_no_damage"),
        "iou_damaged": metrics.get("iou_damaged"),
        "precision_damaged": metrics.get("precision_damaged"),
        "recall_damaged": metrics.get("recall_damaged"),
        "f1_damaged": metrics.get("f1_damaged"),
    }


def compute_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    raw = next(row for row in rows if row["mode"] == "raw")
    keys = ["mean_iou", "iou_damaged", "precision_damaged", "recall_damaged", "f1_damaged"]
    deltas = []
    for row in rows:
        delta_row = {"mode": row["mode"], "threshold": row["threshold"]}
        for key in keys:
            delta_row[key] = subtract_or_none(row.get(key), raw.get(key))
        deltas.append(delta_row)
    return deltas


def subtract_or_none(value: object, baseline: object) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def build_payload(
    args: argparse.Namespace,
    device: torch.device,
    dataset_size: int,
    evaluated_samples: int,
    elapsed_seconds: float,
    metrics_by_threshold: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    rows = flatten_metrics(metrics_by_threshold)
    for row in rows:
        row["damage_tta"] = args.damage_tta
    return {
        "config": {
            "damage_checkpoint": str(args.damage_checkpoint),
            "building_checkpoint": str(args.building_checkpoint),
            "root": str(args.root),
            "split_csv": str(args.split_csv),
            "image_size": args.image_size,
            "target_mode": args.target_mode,
            "damage_model": args.damage_model,
            "damage_tta": args.damage_tta,
            "building_model": args.building_model,
            "building_input_mode": args.building_input_mode,
            "thresholds": args.thresholds,
            "component_connectivity": args.component_connectivity,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
        },
        "notes": {
            "predicted_building_modes": (
                "Predicted-building modes use only the building model probability "
                "mask. They do not use the ground-truth building mask."
            ),
            "oracle_modes": (
                "Oracle modes use target > 0 at evaluation time only and are not "
                "production results."
            ),
            "empty_component_policy": (
                "If a predicted or oracle building component has no class 1 or 2 "
                "pixels in the raw damage prediction, it is filled as class 1 "
                "no-damage by default."
            ),
        },
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "elapsed_seconds": elapsed_seconds,
        "metrics_by_threshold": metrics_by_threshold,
        "rows": rows,
        "deltas_vs_raw": compute_deltas(rows),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "threshold",
        "damage_tta",
        "pixel_accuracy",
        "mean_iou",
        "iou_background",
        "iou_no_damage",
        "iou_damaged",
        "precision_damaged",
        "recall_damaged",
        "f1_damaged",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(payload: dict[str, object]) -> None:
    print("CrisisMap AI - Damage with Predicted Building Mask")
    print("=" * 56)
    print(f"Device: {payload['config']['device']}")
    print(f"Damage TTA: {payload['config']['damage_tta']}")
    print(f"Evaluated samples: {payload['evaluated_samples']}")
    print()
    print("Metrics")
    for row in payload["rows"]:
        print(
            f"  {row['mode']} @ {row['threshold']}: "
            f"mean IoU={format_metric(row['mean_iou'])}, "
            f"IoU damaged={format_metric(row['iou_damaged'])}, "
            f"precision damaged={format_metric(row['precision_damaged'])}, "
            f"recall damaged={format_metric(row['recall_damaged'])}, "
            f"F1 damaged={format_metric(row['f1_damaged'])}"
        )
    print()
    print("Deltas vs raw")
    for row in payload["deltas_vs_raw"]:
        if row["mode"] == "raw":
            continue
        print(
            f"  {row['mode']} @ {row['threshold']}: "
            f"mean IoU {format_signed(row['mean_iou'])}, "
            f"IoU damaged {format_signed(row['iou_damaged'])}, "
            f"precision damaged {format_signed(row['precision_damaged'])}, "
            f"recall damaged {format_signed(row['recall_damaged'])}, "
            f"F1 damaged {format_signed(row['f1_damaged'])}"
        )


def format_metric(value: object) -> str:
    return "nan" if value is None else f"{float(value):.6f}"


def format_signed(value: object) -> str:
    return "nan" if value is None else f"{float(value):+.6f}"


def save_batch_examples(
    output_dir: Path,
    batch: dict[str, object],
    raw_preds: torch.Tensor,
    building_probs: torch.Tensor,
    predicted_examples: dict[str, dict[str, torch.Tensor]],
    oracle_component: torch.Tensor,
    targets: torch.Tensor,
    already_saved: int,
    max_examples: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    pair_ids = batch["pair_id"]

    for batch_index in range(images.shape[0]):
        if already_saved >= max_examples:
            break
        save_example_figure(
            output_dir=output_dir,
            pair_id=str(pair_ids[batch_index]),
            image_tensor=images[batch_index],
            target=targets[batch_index].detach().cpu().numpy(),
            raw_pred=raw_preds[batch_index].detach().cpu().numpy(),
            building_prob=building_probs[batch_index].detach().cpu().numpy(),
            predicted_examples={
                threshold: {
                    name: tensor[batch_index].detach().cpu().numpy()
                    for name, tensor in values.items()
                }
                for threshold, values in predicted_examples.items()
            },
            oracle_component=oracle_component[batch_index].detach().cpu().numpy(),
        )
        already_saved += 1
    return already_saved


def save_example_figure(
    output_dir: Path,
    pair_id: str,
    image_tensor: torch.Tensor,
    target: np.ndarray,
    raw_pred: np.ndarray,
    building_prob: np.ndarray,
    predicted_examples: dict[str, dict[str, np.ndarray]],
    oracle_component: np.ndarray,
) -> None:
    pre_image = tensor_to_uint8_image(image_tensor[:3])
    post_image = tensor_to_uint8_image(image_tensor[3:6])
    thresholds = sorted(predicted_examples, key=float)

    rows = 3 + len(thresholds)
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4 * rows))
    axes_array = np.asarray(axes)

    first_row = [
        ("Pre image", pre_image),
        ("Post image", post_image),
        ("GT damage", colorize_damage_mask(target)),
        ("Raw damage prediction", colorize_damage_mask(raw_pred)),
    ]
    for axis, (title, image) in zip(axes_array[0], first_row):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    axes_array[1, 0].imshow(building_prob, cmap="viridis", vmin=0, vmax=1)
    axes_array[1, 0].set_title("Predicted building probability")
    axes_array[1, 0].axis("off")
    axes_array[1, 1].imshow(overlay_damage(post_image, raw_pred))
    axes_array[1, 1].set_title("Raw overlay")
    axes_array[1, 1].axis("off")
    axes_array[1, 2].imshow(colorize_damage_mask(oracle_component))
    axes_array[1, 2].set_title("Oracle component majority")
    axes_array[1, 2].axis("off")
    axes_array[1, 3].imshow(overlay_damage(post_image, oracle_component))
    axes_array[1, 3].set_title("Oracle component overlay")
    axes_array[1, 3].axis("off")

    for row_index, threshold in enumerate(thresholds, start=2):
        values = predicted_examples[threshold]
        building_mask = values["building_mask"]
        clip_pred = values["clip"]
        component_pred = values["component"]
        panels = [
            (f"Building mask t={threshold}", colorize_building_mask(building_mask)),
            (f"Pred building clip t={threshold}", colorize_damage_mask(clip_pred)),
            (f"Pred component majority t={threshold}", colorize_damage_mask(component_pred)),
            (f"Pred component overlay t={threshold}", overlay_damage(post_image, component_pred)),
        ]
        for axis, (title, image) in zip(axes_array[row_index], panels):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")

    final_row = axes_array[-1]
    for axis in final_row:
        axis.axis("off")
    final_row[0].legend(
        handles=[
            Patch(color=CLASS_COLORS[0] / 255.0, label="0 background"),
            Patch(color=CLASS_COLORS[1] / 255.0, label="1 no damage"),
            Patch(color=CLASS_COLORS[2] / 255.0, label="2 damaged"),
        ],
        loc="center",
    )

    fig.suptitle(pair_id)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(output_dir / f"{safe_filename(pair_id)}_damage_building_downstream.png", dpi=130)
    plt.close(fig)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def colorize_damage_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    return CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]


def colorize_building_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    return BUILDING_COLORS[np.clip(mask, 0, 1)]


def overlay_damage(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    for class_id in (1, 2):
        pixels = mask == class_id
        overlay[pixels] = (
            (1.0 - alpha) * overlay[pixels]
            + alpha * CLASS_COLORS[class_id].astype(np.float32)
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        damage_model = load_damage_model(args, device)
        building_model = load_building_model(args, device)

        started_at = time.time()
        metrics_by_threshold, evaluated_samples = evaluate_downstream(
            damage_model,
            building_model,
            loader,
            device,
            args,
        )
        payload = build_payload(
            args=args,
            device=device,
            dataset_size=len(dataset),
            evaluated_samples=evaluated_samples,
            elapsed_seconds=time.time() - started_at,
            metrics_by_threshold=metrics_by_threshold,
        )
        write_json(args.output_json, payload)
        write_csv(args.output_csv, payload["rows"])
        print_summary(payload)
        print()
        print(f"Saved JSON: {args.output_json.expanduser().resolve()}")
        print(f"Saved CSV: {args.output_csv.expanduser().resolve()}")
        if args.save_examples_dir is not None:
            print(f"Saved examples in: {args.save_examples_dir.expanduser().resolve()}")
    except (
        DownstreamEvaluationError,
        TTAEvaluationError,
        XBDDatasetError,
        BuildingTrainingError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

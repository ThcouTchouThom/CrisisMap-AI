"""Estimate oracle gains from a perfect building segmentation mask.

This is an evaluation-only experiment. It compares raw U-Net predictions with
two oracle post-processing modes that use the ground-truth building mask at
evaluation time. It does not change training, model weights, or existing
evaluation outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


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


ORACLE_MODES = [
    "raw",
    "oracle_building_clip",
    "oracle_building_component_majority",
]

CLASS_NAMES = ["background", "no damage", "damaged"]
CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [35, 170, 80],
        [220, 45, 45],
    ],
    dtype=np.uint8,
)


class OracleEvaluationError(Exception):
    """Raised when oracle evaluation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw U-Net predictions with oracle building-mask modes.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to best_unet.pt or another compatible U-Net checkpoint.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
        help="Path to the split CSV to evaluate.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--target-mode",
        choices=["3-class"],
        default="3-class",
        help="Oracle modes are currently defined for the 3-class formulation.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path where the oracle metrics JSON will be saved.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path with one metrics row per evaluation mode.",
    )
    parser.add_argument(
        "--save-examples-dir",
        type=Path,
        default=None,
        help="Optional directory where visual comparison PNGs will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick local smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Maximum number of visual examples to save.",
    )
    parser.add_argument(
        "--component-connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connected-component neighborhood for the oracle building mask.",
    )
    parser.add_argument(
        "--empty-component-policy",
        choices=["no_damage", "gt_majority"],
        default="no_damage",
        help=(
            "Fallback when a GT building component has no predicted class 1 or 2 "
            "pixels after clipping."
        ),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="UNet base channel count used by the checkpoint.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise OracleEvaluationError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise OracleEvaluationError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise OracleEvaluationError("--num-workers must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise OracleEvaluationError("--max-samples must be positive when provided.")
    if args.num_examples < 0:
        raise OracleEvaluationError("--num-examples must be non-negative.")
    if args.base_channels <= 0:
        raise OracleEvaluationError("--base-channels must be positive.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise OracleEvaluationError("CUDA was requested, but CUDA is not available.")
    return device


def load_dataset(args: argparse.Namespace) -> XBDPairDataset | Subset:
    dataset = XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
        augment_mode="none",
    )
    if args.max_samples is None:
        return dataset

    sample_count = min(args.max_samples, len(dataset))
    return Subset(dataset, range(sample_count))


def make_loader(
    dataset: XBDPairDataset | Subset,
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
        raise OracleEvaluationError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise OracleEvaluationError(f"Checkpoint path is not a file: {checkpoint_path}")

    model = UNet(
        in_channels=6,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        base_channels=args.base_channels,
    ).to(device)

    try:
        checkpoint = load_checkpoint_file(checkpoint_path, device)
        state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
    except (OSError, RuntimeError) as exc:
        raise OracleEvaluationError(
            "Could not load checkpoint. Check --target-mode and --base-channels."
        ) from exc

    model.eval()
    return model


@torch.no_grad()
def evaluate_oracle_modes(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, object]], int]:
    num_classes = len(CLASS_LABELS[args.target_mode])
    confusions = {
        mode: torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        for mode in ORACLE_MODES
    }
    saved_examples = 0
    evaluated_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        logits = model(images)
        raw_preds = torch.argmax(logits, dim=1)

        clipped_preds = oracle_building_clip(raw_preds, targets)
        component_preds = oracle_component_majority(
            raw_preds,
            targets,
            connectivity=args.component_connectivity,
            empty_component_policy=args.empty_component_policy,
            device=device,
        )

        predictions_by_mode = {
            "raw": raw_preds,
            "oracle_building_clip": clipped_preds,
            "oracle_building_component_majority": component_preds,
        }
        for mode, preds in predictions_by_mode.items():
            confusions[mode] += confusion_matrix(preds, targets, num_classes)

        if args.save_examples_dir is not None and saved_examples < args.num_examples:
            saved_examples = save_batch_examples(
                output_dir=args.save_examples_dir,
                batch=batch,
                raw_preds=raw_preds,
                clipped_preds=clipped_preds,
                component_preds=component_preds,
                targets=targets,
                already_saved=saved_examples,
                max_examples=args.num_examples,
            )

        evaluated_samples += int(images.shape[0])

    metrics_by_mode = {}
    for mode, confusion in confusions.items():
        metrics_by_mode[mode] = add_named_metrics(metrics_from_confusion(confusion))
    return metrics_by_mode, evaluated_samples


def oracle_building_clip(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    clipped = preds.clone()
    clipped[targets == 0] = 0
    return clipped


def oracle_component_majority(
    raw_preds: torch.Tensor,
    targets: torch.Tensor,
    connectivity: int,
    empty_component_policy: str,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for pred_tensor, target_tensor in zip(raw_preds, targets):
        pred = pred_tensor.detach().cpu().numpy().astype(np.int16, copy=False)
        target = target_tensor.detach().cpu().numpy().astype(np.int16, copy=False)
        outputs.append(
            oracle_component_majority_single(
                pred,
                target,
                connectivity=connectivity,
                empty_component_policy=empty_component_policy,
            )
        )
    return torch.from_numpy(np.stack(outputs, axis=0)).to(device=device, dtype=torch.long)


def oracle_component_majority_single(
    pred: np.ndarray,
    target: np.ndarray,
    connectivity: int,
    empty_component_policy: str,
) -> np.ndarray:
    building_mask = target > 0
    clipped = pred.copy()
    clipped[~building_mask] = 0

    labels, component_count = label_connected_components(building_mask, connectivity)
    output = np.zeros_like(pred, dtype=np.int64)

    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        component_preds = clipped[component_mask]
        no_damage_count = int(np.count_nonzero(component_preds == 1))
        damaged_count = int(np.count_nonzero(component_preds == 2))

        if damaged_count > no_damage_count:
            component_class = 2
        elif no_damage_count + damaged_count > 0:
            component_class = 1
        elif empty_component_policy == "gt_majority":
            component_targets = target[component_mask]
            component_class = 2 if np.count_nonzero(component_targets == 2) > np.count_nonzero(component_targets == 1) else 1
        else:
            component_class = 1

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


def save_batch_examples(
    output_dir: Path,
    batch: dict[str, object],
    raw_preds: torch.Tensor,
    clipped_preds: torch.Tensor,
    component_preds: torch.Tensor,
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
        pair_id = str(pair_ids[batch_index])
        image = images[batch_index]
        pre_image = tensor_to_uint8_image(image[:3])
        post_image = tensor_to_uint8_image(image[3:6])

        save_example_figure(
            output_dir=output_dir,
            pair_id=pair_id,
            pre_image=pre_image,
            post_image=post_image,
            target=targets[batch_index].detach().cpu().numpy(),
            raw_pred=raw_preds[batch_index].detach().cpu().numpy(),
            clipped_pred=clipped_preds[batch_index].detach().cpu().numpy(),
            component_pred=component_preds[batch_index].detach().cpu().numpy(),
        )
        already_saved += 1

    return already_saved


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def save_example_figure(
    output_dir: Path,
    pair_id: str,
    pre_image: np.ndarray,
    post_image: np.ndarray,
    target: np.ndarray,
    raw_pred: np.ndarray,
    clipped_pred: np.ndarray,
    component_pred: np.ndarray,
) -> None:
    panels = [
        ("Pre-disaster image", pre_image),
        ("Post-disaster image", post_image),
        ("Ground truth", colorize_mask(target)),
        ("Raw prediction", colorize_mask(raw_pred)),
        ("Oracle building clip", colorize_mask(clipped_pred)),
        ("Oracle component majority", colorize_mask(component_pred)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (title, image) in zip(axes.ravel(), panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    legend_handles = [
        Patch(color=CLASS_COLORS[index] / 255.0, label=f"{index}: {name}")
        for index, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3)
    fig.suptitle(pair_id)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path = output_dir / f"{safe_filename(pair_id)}_oracle_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    clipped = np.clip(mask, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[clipped]


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def build_payload(
    args: argparse.Namespace,
    device: torch.device,
    dataset_size: int,
    evaluated_samples: int,
    elapsed_seconds: float,
    metrics_by_mode: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "config": {
            "checkpoint": str(args.checkpoint),
            "root": str(args.root),
            "split_csv": str(args.split_csv),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "target_mode": args.target_mode,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "device": str(device),
            "base_channels": args.base_channels,
            "component_connectivity": args.component_connectivity,
            "empty_component_policy": args.empty_component_policy,
            "empty_component_policy_description": empty_component_policy_description(
                args.empty_component_policy
            ),
        },
        "assumption": (
            "xBD/xView2 building instances are generally associated with one damage "
            "level, so assigning one predicted damage class per ground-truth building "
            "component is a meaningful oracle approximation."
        ),
        "dataset_size": dataset_size,
        "evaluated_samples": evaluated_samples,
        "num_classes": len(CLASS_LABELS[args.target_mode]),
        "class_labels": CLASS_LABELS[args.target_mode],
        "elapsed_seconds": elapsed_seconds,
        "metrics_by_mode": metrics_by_mode,
        "deltas_vs_raw": {
            mode: compute_deltas(metrics_by_mode["raw"], metrics_by_mode[mode])
            for mode in ORACLE_MODES
            if mode != "raw"
        },
    }


def empty_component_policy_description(policy: str) -> str:
    if policy == "gt_majority":
        return (
            "If a ground-truth building component has no predicted class 1 or 2 "
            "pixels after clipping, use the dominant ground-truth building class."
        )
    return (
        "If a ground-truth building component has no predicted class 1 or 2 pixels "
        "after clipping, fill the component as class 1, no damage."
    )


def compute_deltas(
    raw_metrics: dict[str, object],
    candidate_metrics: dict[str, object],
) -> dict[str, float | None]:
    keys = [
        "mean_iou",
        "iou_damaged",
        "precision_damaged",
        "recall_damaged",
        "f1_damaged",
    ]
    deltas = {}
    for key in keys:
        raw_value = raw_metrics.get(key)
        candidate_value = candidate_metrics.get(key)
        if raw_value is None or candidate_value is None:
            deltas[key] = None
        else:
            deltas[key] = float(candidate_value) - float(raw_value)
    return deltas


def write_json(path: Path, payload: dict[str, object]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_csv(path: Path, metrics_by_mode: dict[str, dict[str, object]]) -> None:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
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
        for mode in ORACLE_MODES:
            row = {"mode": mode}
            row.update({key: metrics_by_mode[mode].get(key) for key in fieldnames[1:]})
            writer.writerow(row)


def print_summary(payload: dict[str, object]) -> None:
    print("CrisisMap AI - Oracle Building Mask Evaluation")
    print("=" * 49)
    print(f"Device: {payload['config']['device']}")
    print(f"Evaluated samples: {payload['evaluated_samples']}")
    print(f"Empty component policy: {payload['config']['empty_component_policy']}")
    print()

    print("Metrics by mode")
    for mode in ORACLE_MODES:
        metrics = payload["metrics_by_mode"][mode]
        print(
            f"  {mode}: "
            f"mean IoU={format_metric(metrics['mean_iou'])}, "
            f"IoU damaged={format_metric(metrics['iou_damaged'])}, "
            f"precision damaged={format_metric(metrics['precision_damaged'])}, "
            f"recall damaged={format_metric(metrics['recall_damaged'])}, "
            f"F1 damaged={format_metric(metrics['f1_damaged'])}"
        )

    print()
    print("Deltas vs raw")
    for mode, deltas in payload["deltas_vs_raw"].items():
        print(f"  {mode}:")
        print(
            "    "
            f"mean IoU {format_signed(deltas['mean_iou'])}, "
            f"IoU damaged {format_signed(deltas['iou_damaged'])}, "
            f"precision damaged {format_signed(deltas['precision_damaged'])}, "
            f"recall damaged {format_signed(deltas['recall_damaged'])}, "
            f"F1 damaged {format_signed(deltas['f1_damaged'])}"
        )


def format_metric(value: object) -> str:
    return "nan" if value is None else f"{float(value):.6f}"


def format_signed(value: object) -> str:
    return "nan" if value is None else f"{float(value):+.6f}"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        model = load_model(args, device)

        started_at = time.time()
        metrics_by_mode, evaluated_samples = evaluate_oracle_modes(
            model,
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
            metrics_by_mode=metrics_by_mode,
        )

        write_json(args.output_json, payload)
        if args.output_csv is not None:
            write_csv(args.output_csv, metrics_by_mode)

        print_summary(payload)
        print()
        print(f"Saved oracle metrics JSON: {args.output_json.expanduser().resolve()}")
        if args.output_csv is not None:
            print(f"Saved oracle metrics CSV: {args.output_csv.expanduser().resolve()}")
        if args.save_examples_dir is not None:
            print(f"Saved examples in: {args.save_examples_dir.expanduser().resolve()}")
    except (
        OracleEvaluationError,
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

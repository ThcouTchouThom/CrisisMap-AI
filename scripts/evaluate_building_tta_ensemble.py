"""Evaluate building segmentation with TTA and checkpoint ensembles."""

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
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from train_building_segmentation import (  # noqa: E402
    MODEL_CHOICES,
    BuildingTrainingError,
    build_model,
    clean_state_dict,
    input_channels,
    normalize_logits,
)


TTA_MODES = {"none", "flips", "rot90", "d4"}
ENSEMBLE_MODES = {"average_prob", "union", "intersection", "majority"}
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


class BuildingEnsembleEvaluationError(Exception):
    """Raised when building ensemble evaluation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary building segmentation with TTA and ensembles.",
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", nargs="+", required=True, type=Path)
    parser.add_argument("--model", nargs="+", required=True, choices=sorted(MODEL_CHOICES))
    parser.add_argument("--input-mode", nargs="+", default=["pre"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--target-mode", choices=["building-binary"], default="building-binary")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--tta-modes",
        nargs="+",
        choices=sorted(TTA_MODES),
        default=["none"],
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--ensemble-modes",
        nargs="+",
        choices=sorted(ENSEMBLE_MODES),
        default=["average_prob"],
    )
    parser.add_argument("--component-connectivity", choices=[4, 8], type=int, default=8)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--save-examples-dir", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.model) != len(args.checkpoint):
        raise BuildingEnsembleEvaluationError(
            "--model and --checkpoint must have the same length."
        )
    if len(args.input_mode) not in {1, len(args.model)}:
        raise BuildingEnsembleEvaluationError(
            "--input-mode must contain one value or one value per model."
        )
    for input_mode in args.input_mode:
        if input_mode not in {"pre", "post", "pre-post"}:
            raise BuildingEnsembleEvaluationError(f"Unsupported input mode: {input_mode}")
    if args.image_size <= 0 or args.batch_size <= 0:
        raise BuildingEnsembleEvaluationError("--image-size and --batch-size must be positive.")
    if args.num_workers < 0 or args.num_examples < 0:
        raise BuildingEnsembleEvaluationError("--num-workers/--num-examples must be non-negative.")
    if not args.thresholds:
        raise BuildingEnsembleEvaluationError("--thresholds must not be empty.")
    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise BuildingEnsembleEvaluationError("All thresholds must be between 0 and 1.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BuildingEnsembleEvaluationError("CUDA was requested but is unavailable.")
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


def load_building_model(
    model_name: str,
    checkpoint_path: Path,
    input_mode: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise BuildingEnsembleEvaluationError(f"Checkpoint not found: {checkpoint}")
    try:
        model, _ = build_model(model_name, input_channels(input_mode), device)
        payload = load_checkpoint_file(checkpoint, device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else payload
        model.load_state_dict(clean_state_dict(state_dict))
    except (BuildingTrainingError, RuntimeError, OSError) as exc:
        raise BuildingEnsembleEvaluationError(
            f"Could not load building model/checkpoint: {model_name} / {checkpoint}"
        ) from exc
    model.eval()
    return model


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


def select_input(images: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "pre":
        return images[:, :3]
    if input_mode == "post":
        return images[:, 3:6]
    if input_mode == "pre-post":
        return images
    raise BuildingEnsembleEvaluationError(f"Unsupported input mode: {input_mode}")


def tta_ops(mode: str) -> list[tuple[int, bool, bool]]:
    if mode == "none":
        return [(0, False, False)]
    if mode == "flips":
        return [(0, False, False), (0, True, False), (0, False, True), (0, True, True)]
    if mode == "rot90":
        return [(rotation, False, False) for rotation in range(4)]
    if mode == "d4":
        return [(rotation, False, False) for rotation in range(4)] + [
            (rotation, True, False) for rotation in range(4)
        ]
    raise BuildingEnsembleEvaluationError(f"Unsupported TTA mode: {mode}")


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
def predict_probability_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    input_mode: str,
    tta_mode: str,
    use_amp: bool,
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    selected = select_input(images, input_mode)
    ops = tta_ops(tta_mode)
    for op in ops:
        view = apply_op(selected, op)
        with autocast_context(use_amp):
            logits = normalize_logits(model(view))
        logits = invert_op(logits, op).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    if logits_sum is None:
        raise BuildingEnsembleEvaluationError(f"No TTA operations for mode: {tta_mode}")
    return torch.sigmoid(logits_sum / float(len(ops))).squeeze(1)


def ensemble_mask(
    probabilities: torch.Tensor,
    threshold: float,
    mode: str,
) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise BuildingEnsembleEvaluationError("Expected probability tensor [M,N,H,W].")
    binary = probabilities >= threshold
    if mode == "average_prob":
        return probabilities.mean(dim=0) >= threshold
    if mode == "union":
        return binary.any(dim=0)
    if mode == "intersection":
        return binary.all(dim=0)
    if mode == "majority":
        return binary.sum(dim=0) >= int(np.ceil(probabilities.shape[0] / 2.0))
    raise BuildingEnsembleEvaluationError(f"Unsupported ensemble mode: {mode}")


def building_target(targets: torch.Tensor) -> torch.Tensor:
    return (targets > 0).long()


def update_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    valid = (targets >= 0) & (targets < 2)
    indices = targets[valid] * 2 + preds[valid]
    counts = torch.bincount(indices, minlength=4).reshape(2, 2)
    confusion += counts.to(device=confusion.device, dtype=confusion.dtype)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    confusion_f = confusion.to(dtype=torch.float64)
    tp = torch.diag(confusion_f)
    total = float(confusion_f.sum().item())
    union = confusion_f.sum(dim=1) + confusion_f.sum(dim=0) - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1.0), torch.full_like(tp, float("nan")))
    precision = tp[1] / confusion_f[:, 1].sum().clamp_min(1.0)
    recall = tp[1] / confusion_f[1, :].sum().clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "pixel_accuracy": float(tp.sum().item() / total) if total else 0.0,
        "mean_iou": float(torch.nanmean(iou).item()) if not torch.isnan(iou).all() else 0.0,
        "background_iou": 0.0 if torch.isnan(iou[0]) else float(iou[0].item()),
        "building_iou": 0.0 if torch.isnan(iou[1]) else float(iou[1].item()),
        "building_precision": float(precision.item()),
        "building_recall": float(recall.item()),
        "building_f1": float(f1.item()),
    }


def label_connected_components(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage  # type: ignore

        structure = (
            np.ones((3, 3), dtype=np.uint8)
            if connectivity == 8
            else np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        )
        labels, count = ndimage.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)
    except ImportError:
        return label_connected_components_fallback(mask, connectivity)


def label_connected_components_fallback(
    mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    height, width = mask.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    current = 0
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            current += 1
            labels[row, col] = current
            queue: deque[tuple[int, int]] = deque([(row, col)])
            while queue:
                y, x = queue.popleft()
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
    return labels, current


def object_metrics_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    connectivity: int,
) -> tuple[int, int, int, int]:
    pred_matched = 0
    pred_total = 0
    gt_matched = 0
    gt_total = 0
    for pred_tensor, target_tensor in zip(preds, targets):
        pred_mask = pred_tensor.detach().cpu().numpy().astype(bool, copy=False)
        gt_mask = target_tensor.detach().cpu().numpy().astype(bool, copy=False)
        pred_labels, pred_count = label_connected_components(pred_mask, connectivity)
        gt_labels, gt_count = label_connected_components(gt_mask, connectivity)
        pred_total += pred_count
        gt_total += gt_count
        for component_id in range(1, pred_count + 1):
            pred_matched += int(np.any(gt_mask[pred_labels == component_id]))
        for component_id in range(1, gt_count + 1):
            gt_matched += int(np.any(pred_mask[gt_labels == component_id]))
    return pred_matched, pred_total, gt_matched, gt_total


@torch.no_grad()
def evaluate(
    models: list[torch.nn.Module],
    input_modes: list[str],
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> tuple[list[dict[str, object]], int]:
    accumulators: dict[tuple[str, str, str], dict[str, object]] = {}
    for tta_mode in args.tta_modes:
        for ensemble_mode in args.ensemble_modes:
            for threshold in args.thresholds:
                accumulators[(tta_mode, ensemble_mode, threshold_key(threshold))] = {
                    "confusion": torch.zeros((2, 2), dtype=torch.int64, device=device),
                    "pred_matched": 0,
                    "pred_total": 0,
                    "gt_matched": 0,
                    "gt_total": 0,
                    "threshold": threshold,
                }

    saved_examples = 0
    evaluated_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = building_target(batch["target"].to(device, non_blocking=True))
        evaluated_samples += int(images.shape[0])

        probabilities_by_tta: dict[str, torch.Tensor] = {}
        for tta_mode in args.tta_modes:
            model_probs = [
                predict_probability_tta(model, images, input_mode, tta_mode, use_amp)
                for model, input_mode in zip(models, input_modes)
            ]
            probabilities_by_tta[tta_mode] = torch.stack(model_probs, dim=0)

        first_example_saved = False
        for tta_mode, probabilities in probabilities_by_tta.items():
            for ensemble_mode in args.ensemble_modes:
                for threshold in args.thresholds:
                    key = (tta_mode, ensemble_mode, threshold_key(threshold))
                    preds_bool = ensemble_mask(probabilities, threshold, ensemble_mode)
                    preds = preds_bool.long()
                    accumulator = accumulators[key]
                    update_confusion(accumulator["confusion"], preds, targets)
                    pred_m, pred_t, gt_m, gt_t = object_metrics_batch(
                        preds,
                        targets,
                        args.component_connectivity,
                    )
                    accumulator["pred_matched"] += pred_m
                    accumulator["pred_total"] += pred_t
                    accumulator["gt_matched"] += gt_m
                    accumulator["gt_total"] += gt_t

                    if (
                        args.save_examples_dir is not None
                        and saved_examples < args.num_examples
                        and not first_example_saved
                    ):
                        save_batch_examples(
                            batch=batch,
                            preds=preds.detach().cpu(),
                            targets=targets.detach().cpu(),
                            save_dir=args.save_examples_dir,
                            start_index=saved_examples,
                            max_examples=args.num_examples,
                            title_suffix=f"{tta_mode}_{ensemble_mode}_t{threshold_key(threshold)}",
                        )
                        saved_examples += min(images.shape[0], args.num_examples - saved_examples)
                        first_example_saved = True

    rows = []
    for (tta_mode, ensemble_mode, threshold_name), accumulator in accumulators.items():
        metrics = metrics_from_confusion(accumulator["confusion"])
        pred_total = int(accumulator["pred_total"])
        gt_total = int(accumulator["gt_total"])
        metrics["object_precision"] = (
            float(accumulator["pred_matched"]) / pred_total if pred_total else 0.0
        )
        metrics["object_recall"] = float(accumulator["gt_matched"]) / gt_total if gt_total else 0.0
        rows.append(
            {
                "tta_mode": tta_mode,
                "ensemble_mode": ensemble_mode,
                "threshold": accumulator["threshold"],
                **metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["building_iou"]),
            float(row["building_f1"]),
            float(row["building_recall"]),
        ),
        reverse=True,
    )
    return rows, evaluated_samples


def threshold_key(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def save_batch_examples(
    batch: dict[str, object],
    preds: torch.Tensor,
    targets: torch.Tensor,
    save_dir: Path,
    start_index: int,
    max_examples: int,
    title_suffix: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    pair_ids = batch.get("pair_id", [])
    for offset in range(min(images.shape[0], max_examples - start_index)):
        index = start_index + offset
        pair_id = (
            str(pair_ids[offset])
            if isinstance(pair_ids, (list, tuple)) and offset < len(pair_ids)
            else f"sample_{index:03d}"
        )
        pre = tensor_rgb(images[offset, :3])
        post = tensor_rgb(images[offset, 3:6])
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        panels = [
            ("pre", pre),
            ("post", post),
            ("gt building", targets[offset].numpy() * 255),
            ("pred building", preds[offset].numpy() * 255),
        ]
        for axis, (title, image) in zip(axes, panels):
            axis.imshow(image, cmap=None if image.ndim == 3 else "gray", vmin=0, vmax=255)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(save_dir / f"{index:03d}_{safe_name(pair_id)}_{title_suffix}.png", dpi=120)
        plt.close(fig)


def tensor_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.numpy().transpose(1, 2, 0)
    return (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tta_mode",
        "ensemble_mode",
        "threshold",
        "pixel_accuracy",
        "mean_iou",
        "background_iou",
        "building_iou",
        "building_precision",
        "building_recall",
        "building_f1",
        "object_precision",
        "object_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")
        input_modes = args.input_mode if len(args.input_mode) > 1 else args.input_mode * len(args.model)
        models = [
            load_building_model(model_name, checkpoint, input_mode, device)
            for model_name, checkpoint, input_mode in zip(args.model, args.checkpoint, input_modes)
        ]
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        started = time.time()
        rows, evaluated_samples = evaluate(models, input_modes, loader, args, device, use_amp)
        payload = {
            "config": {
                "root": str(args.root),
                "split_csv": str(args.split_csv),
                "model": args.model,
                "checkpoint": [str(path) for path in args.checkpoint],
                "input_mode": input_modes,
                "image_size": args.image_size,
                "thresholds": args.thresholds,
                "tta_modes": args.tta_modes,
                "ensemble_modes": args.ensemble_modes,
                "device": str(device),
                "amp": use_amp,
            },
            "dataset_size": len(dataset),
            "evaluated_samples": evaluated_samples,
            "elapsed_seconds": time.time() - started,
            "rows": rows,
        }
        write_json(args.output_json, payload)
        write_csv(args.output_csv, rows)
    except (
        BuildingEnsembleEvaluationError,
        BuildingTrainingError,
        XBDDatasetError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")
    if rows:
        best = rows[0]
        print(
            "Best building row: "
            f"tta={best['tta_mode']} ensemble={best['ensemble_mode']} "
            f"threshold={best['threshold']} building_iou={best['building_iou']:.6f} "
            f"f1={best['building_f1']:.6f} recall={best['building_recall']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

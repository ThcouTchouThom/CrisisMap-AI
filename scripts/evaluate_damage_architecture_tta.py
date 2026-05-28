"""Evaluate Axis 2 damage architecture checkpoints with test-time augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError  # noqa: E402
from crisismap.evaluation.evaluate_unet import CLASS_LABELS  # noqa: E402
from crisismap.models.damage_model_factory import (  # noqa: E402
    DamageModelFactoryError,
    create_damage_model,
    damage_model_metadata,
)

from evaluate_damage_tta import (  # noqa: E402
    TTAEvaluationError,
    TTA_MODES,
    build_rows,
    evaluate_modes,
    load_checkpoint_file,
    load_dataset,
    make_loader,
    print_summary,
    resolve_device,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a damage architecture checkpoint with TTA.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model", required=True)
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


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TTAEvaluationError("Checkpoint is not a state_dict or checkpoint dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[str(key).removeprefix("module.")] = value
    if not cleaned:
        raise TTAEvaluationError("Checkpoint contains no tensor weights.")
    return cleaned


def load_arch_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise TTAEvaluationError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise TTAEvaluationError(f"Checkpoint path is not a file: {checkpoint_path}")

    model = create_damage_model(
        args.model,
        num_classes=len(CLASS_LABELS[args.target_mode]),
        in_channels=6,
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    try:
        model.load_state_dict(extract_state_dict(checkpoint))
    except RuntimeError as exc:
        raise TTAEvaluationError(
            "Checkpoint weights do not match the requested architecture. "
            "Check --model and --base-channels."
        ) from exc
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")
        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        model = load_arch_model(args, device)
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
                "model": args.model,
                "model_metadata": damage_model_metadata(args.model),
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

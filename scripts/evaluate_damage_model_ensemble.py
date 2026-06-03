"""Evaluate heterogeneous damage-model ensembles and building-mask constraints.

This script is evaluation-only. It does not train models or modify existing
checkpoints/results unless the caller explicitly writes new output files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
PROJECT_SCRIPTS = PROJECT_ROOT / "scripts"
for path in (PROJECT_SRC, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.evaluation.evaluate_unet import (  # noqa: E402
    CLASS_LABELS,
    EvaluationError,
    confusion_matrix,
    extract_state_dict,
    load_checkpoint_file,
    metrics_from_confusion,
)
from crisismap.models.damage_model_factory import (  # noqa: E402
    DamageModelFactoryError,
    create_damage_model,
)
from crisismap.models.unet import UNet  # noqa: E402
from evaluate_damage_tta import apply_op, invert_op, tta_ops  # noqa: E402


CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [35, 170, 80],
        [220, 45, 45],
    ],
    dtype=np.uint8,
)
FAMILY_CHOICES = {
    "local_unet_existing",
    "siamese_unet_attention",
    "siamese_unet_abs_signed",
    "multitemporal_fusion",
    "xview2_strong_baseline",
    "multihead_damage",
    "building_segmentation",
}
ROLE_CHOICES = {"damage", "building", "both"}
ENSEMBLE_MODE_ALIASES = {
    "weighted_average": "weighted_average_logits",
    "logits": "average_logits",
    "probabilities": "average_prob",
}
BUILDING_CONSTRAINTS = {
    "none",
    "predicted_building_clip",
    "predicted_building_component_majority",
    "building_ensemble_mask",
}


class EnsembleEvaluationError(Exception):
    """Raised when ensemble evaluation cannot continue safely."""


@dataclass(frozen=True)
class CandidateConfig:
    enabled: bool
    name: str
    role: str
    family: str
    model: str
    checkpoint: Path
    weight: float
    target_mode: str
    label_mode: str
    base_channels: int
    input_mode: str
    building_threshold: float
    notes: str


@dataclass
class LoadedCandidate:
    config: CandidateConfig
    model: torch.nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ensembles of heterogeneous damage models and optional "
            "building-mask constraints."
        )
    )
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=PROJECT_ROOT / "configs" / "damage_ensemble_candidates.csv",
    )
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
        choices=["none", "flips", "rot90", "d4"],
        default=["none"],
    )
    parser.add_argument(
        "--ensemble-modes",
        nargs="+",
        choices=[
            "average_logits",
            "average_prob",
            "weighted_average",
            "weighted_average_logits",
            "weighted_average_prob",
            "majority_vote",
        ],
        default=["average_logits", "average_prob", "weighted_average_logits"],
    )
    parser.add_argument(
        "--damage-biases",
        nargs="+",
        type=float,
        default=[0.0],
        help="Additive logit bias for class 2 damaged.",
    )
    parser.add_argument(
        "--building-constraints",
        nargs="+",
        choices=sorted(BUILDING_CONSTRAINTS),
        default=["none"],
    )
    parser.add_argument(
        "--building-thresholds",
        nargs="+",
        type=float,
        default=[0.5],
    )
    parser.add_argument("--component-connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--min-damage-models", type=int, default=1)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--ranked-csv", type=Path, default=None)
    parser.add_argument("--save-examples-dir", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise EnsembleEvaluationError("--image-size must be positive.")
    if args.batch_size <= 0:
        raise EnsembleEvaluationError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise EnsembleEvaluationError("--num-workers must be non-negative.")
    if args.min_damage_models <= 0:
        raise EnsembleEvaluationError("--min-damage-models must be positive.")
    if args.num_examples < 0:
        raise EnsembleEvaluationError("--num-examples must be non-negative.")
    if not args.tta_modes:
        raise EnsembleEvaluationError("--tta-modes must contain at least one mode.")
    if not args.ensemble_modes:
        raise EnsembleEvaluationError("--ensemble-modes must contain at least one mode.")
    if not args.damage_biases:
        raise EnsembleEvaluationError("--damage-biases must contain at least one value.")
    if not args.building_thresholds:
        raise EnsembleEvaluationError("--building-thresholds must contain at least one value.")
    for threshold in args.building_thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise EnsembleEvaluationError("Building thresholds must be between 0 and 1.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EnsembleEvaluationError("CUDA was requested, but CUDA is not available.")
    return device


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def read_candidate_csv(csv_path: Path) -> list[CandidateConfig]:
    csv_path = csv_path.expanduser()
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise EnsembleEvaluationError(f"Candidate CSV does not exist: {csv_path}")

    candidates: list[CandidateConfig] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EnsembleEvaluationError(f"Candidate CSV has no header: {csv_path}")
        reader.fieldnames = [clean_key(name) for name in reader.fieldnames]
        for row_index, raw_row in enumerate(reader, start=2):
            row = clean_row(raw_row)
            enabled = parse_bool(row.get("enabled", "1"))
            name = get_required(row, ["name", "experiment", "run_name"], csv_path, row_index)
            family = get_required(row, ["family", "model_family"], csv_path, row_index)
            role = row.get("role", "damage").strip() or "damage"
            model = row.get("model", family).strip() or family
            checkpoint = resolve_project_path(
                get_required(row, ["checkpoint", "checkpoint_path"], csv_path, row_index)
            )
            candidate = CandidateConfig(
                enabled=enabled,
                name=name,
                role=role,
                family=family,
                model=model,
                checkpoint=checkpoint,
                weight=parse_float(row.get("weight", "1.0"), 1.0),
                target_mode=row.get("target_mode", "3-class").strip() or "3-class",
                label_mode=row.get("label_mode", row.get("target_mode", "3-class")).strip()
                or "3-class",
                base_channels=parse_int(row.get("base_channels", "32"), 32),
                input_mode=row.get("input_mode", "pre").strip() or "pre",
                building_threshold=parse_float(row.get("building_threshold", "0.5"), 0.5),
                notes=row.get("notes", "").strip(),
            )
            validate_candidate(candidate, csv_path, row_index, row)
            candidates.append(candidate)
    return candidates


def clean_key(value: object) -> str:
    return str(value).replace("\ufeff", "").strip()


def clean_row(raw_row: dict[str, object]) -> dict[str, str]:
    return {clean_key(key): str(value).strip() for key, value in raw_row.items() if key is not None}


def get_required(
    row: dict[str, str],
    aliases: list[str],
    csv_path: Path,
    row_index: int,
) -> str:
    for alias in aliases:
        value = row.get(alias)
        if value is not None and value.strip():
            return value.strip()
    raise EnsembleEvaluationError(
        "Missing required candidate CSV field.\n"
        f"CSV: {csv_path}\n"
        f"Row index: {row_index}\n"
        f"Accepted aliases: {aliases}\n"
        f"Available keys: {sorted(row)}\n"
        f"Row: {row}"
    )


def parse_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "n", "disabled", "planned"}


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_candidate(
    candidate: CandidateConfig,
    csv_path: Path,
    row_index: int,
    row: dict[str, str],
) -> None:
    if candidate.role not in ROLE_CHOICES:
        raise EnsembleEvaluationError(
            f"Invalid role '{candidate.role}' in {csv_path} row {row_index}: {row}"
        )
    if candidate.family not in FAMILY_CHOICES:
        raise EnsembleEvaluationError(
            f"Invalid family '{candidate.family}' in {csv_path} row {row_index}. "
            f"Supported families: {sorted(FAMILY_CHOICES)}"
        )
    if candidate.weight < 0.0:
        raise EnsembleEvaluationError(
            f"Candidate weight must be non-negative in {csv_path} row {row_index}: {row}"
        )
    if candidate.base_channels <= 0:
        raise EnsembleEvaluationError(
            f"base_channels must be positive in {csv_path} row {row_index}: {row}"
        )
    if candidate.input_mode not in {"pre", "post", "pre-post"}:
        raise EnsembleEvaluationError(
            f"input_mode must be pre, post, or pre-post in {csv_path} row {row_index}: {row}"
        )


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


def load_candidates(
    candidates: list[CandidateConfig],
    device: torch.device,
) -> tuple[list[LoadedCandidate], list[str]]:
    loaded: list[LoadedCandidate] = []
    warnings: list[str] = []
    for candidate in candidates:
        if not candidate.enabled:
            warnings.append(f"Skipping disabled candidate: {candidate.name}")
            continue
        if not candidate.checkpoint.exists():
            warnings.append(f"Skipping missing checkpoint for {candidate.name}: {candidate.checkpoint}")
            continue
        if not candidate.checkpoint.is_file():
            warnings.append(f"Skipping non-file checkpoint for {candidate.name}: {candidate.checkpoint}")
            continue
        try:
            model = create_candidate_model(candidate, device)
            checkpoint = load_checkpoint_file(candidate.checkpoint, device)
            state_dict = extract_candidate_state_dict(candidate, checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            loaded.append(LoadedCandidate(config=candidate, model=model))
            print(f"Loaded candidate: {candidate.name} ({candidate.family}/{candidate.model})")
        except (
            RuntimeError,
            OSError,
            ImportError,
            DamageModelFactoryError,
            EvaluationError,
            EnsembleEvaluationError,
        ) as exc:
            warnings.append(f"Skipping candidate {candidate.name}: {exc}")
    return loaded, warnings


def create_candidate_model(candidate: CandidateConfig, device: torch.device) -> torch.nn.Module:
    damage_channels = damage_channels_for_label_mode(candidate.label_mode)
    if candidate.family == "local_unet_existing":
        model = UNet(
            in_channels=6,
            num_classes=3,
            base_channels=candidate.base_channels,
        )
    elif candidate.family in {"siamese_unet_attention", "siamese_unet_abs_signed"}:
        model = create_damage_model(
            candidate.model,
            num_classes=3,
            in_channels=6,
            base_channels=candidate.base_channels,
        )
    elif candidate.family == "multitemporal_fusion":
        from crisismap.models.multitemporal_fusion import create_multitemporal_fusion_model

        model = create_multitemporal_fusion_model(
            candidate.model,
            damage_channels=damage_channels,
        )
    elif candidate.family == "xview2_strong_baseline":
        from crisismap.models.xview2_strong_baseline import create_xview2_strong_baseline_model

        model = create_xview2_strong_baseline_model(
            candidate.model,
            damage_channels=damage_channels,
        )
    elif candidate.family == "multihead_damage":
        from crisismap.models.multihead_damage import create_multihead_damage_model

        model = create_multihead_damage_model(
            candidate.model,
            damage_channels=damage_channels,
        )
    elif candidate.family == "building_segmentation":
        from train_building_segmentation import build_model, input_channels

        model, _actual_model = build_model(
            candidate.model,
            input_channels(candidate.input_mode),
            torch.device("cpu"),
        )
    else:
        raise EnsembleEvaluationError(f"Unsupported family: {candidate.family}")
    return model.to(device)


def damage_channels_for_label_mode(label_mode: str) -> int:
    normalized = label_mode.strip().lower()
    if normalized in {"3-class", "3class", "current"}:
        return 3
    if normalized in {
        "damage2",
        "damage-2",
        "damage-2class",
        "building_damage2class",
        "building-damage-2class",
        "2-class",
        "2class",
    }:
        return 2
    if normalized in {"binary", "multilabel", "multilabel_building_damage"}:
        return 1
    if normalized in {"5-class", "5class", "xview2_5class"}:
        return 5
    return 3


def extract_candidate_state_dict(candidate: CandidateConfig, checkpoint: object) -> dict[str, torch.Tensor]:
    if candidate.family == "building_segmentation":
        from train_building_segmentation import clean_state_dict

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return clean_state_dict(checkpoint["model_state_dict"])
        return clean_state_dict(checkpoint)
    return extract_state_dict(checkpoint)


@torch.no_grad()
def evaluate_ensembles(
    loaded_candidates: list[LoadedCandidate],
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    num_classes = len(CLASS_LABELS[args.target_mode])
    combos = build_evaluation_combos(args, loaded_candidates)
    if not combos:
        raise EnsembleEvaluationError("No valid ensemble combinations to evaluate.")

    confusions = {
        combo_key(combo): torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        for combo in combos
    }
    warnings: list[str] = []
    examples_saved: list[dict[str, Any]] = []
    saved_count = 0
    first_combo_key = combo_key(combos[0]) if combos else None

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        for tta_mode in args.tta_modes:
            predictions = predict_all_candidates_tta(
                loaded_candidates,
                images,
                tta_mode,
                use_amp,
            )
            damage_predictions = [
                prediction
                for prediction in predictions
                if prediction["damage_logits"] is not None
                and prediction["candidate"].config.role in {"damage", "both"}
            ]
            building_predictions = [
                prediction
                for prediction in predictions
                if prediction["building_logits"] is not None
                and prediction["candidate"].config.role in {"building", "both"}
            ]
            if len(damage_predictions) < args.min_damage_models:
                continue

            for combo in combos:
                if combo["tta_mode"] != tta_mode:
                    continue
                if combo["needs_building"] and not building_predictions:
                    continue
                preds = apply_ensemble_combo(
                    combo,
                    damage_predictions,
                    building_predictions,
                    device,
                    args.component_connectivity,
                )
                confusions[combo_key(combo)] += confusion_matrix(preds, targets, num_classes)
                if (
                    args.save_examples_dir is not None
                    and saved_count < args.num_examples
                    and combo_key(combo) == first_combo_key
                ):
                    saved = save_batch_examples(
                        args.save_examples_dir,
                        batch,
                        preds.detach().cpu(),
                        combo,
                        saved_count,
                        args.num_examples,
                    )
                    saved_count += saved
                    if saved:
                        examples_saved.append({"combo": combo, "count": saved})

        print(f"Processed batch {batch_index}/{len(loader)}")

    rows = []
    for combo in combos:
        key = combo_key(combo)
        metrics = metrics_from_confusion(confusions[key])
        rows.append(build_row(combo, metrics, loaded_candidates))

    rows = sorted(
        rows,
        key=lambda row: (
            none_to_negative(row.get("test_f1_damaged")),
            none_to_negative(row.get("test_iou_damaged")),
            none_to_negative(row.get("test_mean_iou")),
        ),
        reverse=True,
    )
    return rows, warnings, examples_saved


def build_evaluation_combos(
    args: argparse.Namespace,
    loaded_candidates: list[LoadedCandidate],
) -> list[dict[str, Any]]:
    has_building = any(
        candidate.config.role in {"building", "both"}
        and provides_building_family(candidate.config.family)
        for candidate in loaded_candidates
    )
    combos: list[dict[str, Any]] = []
    for tta_mode in args.tta_modes:
        for raw_mode in args.ensemble_modes:
            ensemble_mode = ENSEMBLE_MODE_ALIASES.get(raw_mode, raw_mode)
            for damage_bias in args.damage_biases:
                for constraint in args.building_constraints:
                    if constraint == "none":
                        combos.append(
                            {
                                "tta_mode": tta_mode,
                                "ensemble_mode": ensemble_mode,
                                "damage_bias": float(damage_bias),
                                "building_constraint": "none",
                                "building_threshold": None,
                                "needs_building": False,
                            }
                        )
                        continue
                    if not has_building:
                        print(
                            f"WARNING: skipping building constraint '{constraint}' "
                            "because no loaded candidate provides building logits."
                        )
                        continue
                    for threshold in args.building_thresholds:
                        combos.append(
                            {
                                "tta_mode": tta_mode,
                                "ensemble_mode": ensemble_mode,
                                "damage_bias": float(damage_bias),
                                "building_constraint": constraint,
                                "building_threshold": float(threshold),
                                "needs_building": True,
                            }
                        )
    return combos


def provides_building_family(family: str) -> bool:
    return family in {
        "building_segmentation",
        "multitemporal_fusion",
        "xview2_strong_baseline",
        "multihead_damage",
    }


def combo_key(combo: dict[str, Any]) -> tuple[Any, ...]:
    return (
        combo["tta_mode"],
        combo["ensemble_mode"],
        combo["damage_bias"],
        combo["building_constraint"],
        combo["building_threshold"],
    )


@torch.no_grad()
def predict_all_candidates_tta(
    loaded_candidates: list[LoadedCandidate],
    images: torch.Tensor,
    tta_mode: str,
    use_amp: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for loaded in loaded_candidates:
        damage_sum: torch.Tensor | None = None
        building_sum: torch.Tensor | None = None
        ops = tta_ops(tta_mode)
        for op in ops:
            view = apply_op(images, op)
            model_input = select_candidate_input(view, loaded.config)
            with autocast_context(use_amp):
                output = loaded.model(model_input)
            damage_logits, building_logits = normalize_candidate_output(
                output,
                loaded.config,
            )
            if damage_logits is not None:
                damage_logits = invert_op(damage_logits, op).float()
                damage_sum = damage_logits if damage_sum is None else damage_sum + damage_logits
            if building_logits is not None:
                building_logits = invert_op(building_logits, op).float()
                building_sum = building_logits if building_sum is None else building_sum + building_logits

        results.append(
            {
                "candidate": loaded,
                "damage_logits": None if damage_sum is None else damage_sum / float(len(ops)),
                "building_logits": None if building_sum is None else building_sum / float(len(ops)),
            }
        )
    return results


def select_candidate_input(images: torch.Tensor, candidate: CandidateConfig) -> torch.Tensor:
    if candidate.family != "building_segmentation":
        return images
    if candidate.input_mode == "pre":
        return images[:, :3]
    if candidate.input_mode == "post":
        return images[:, 3:]
    return images


def normalize_candidate_output(
    output: torch.Tensor | dict[str, torch.Tensor],
    candidate: CandidateConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    building_logits: torch.Tensor | None = None
    if candidate.family == "building_segmentation":
        from train_building_segmentation import normalize_logits

        return None, normalize_logits(output)

    if isinstance(output, dict):
        damage_logits = output.get("damage_logits")
        building_logits = output.get("building_logits")
    else:
        damage_logits = output

    if damage_logits is None:
        return None, building_logits
    return damage_logits_to_three_class_logits(damage_logits, building_logits), building_logits


def damage_logits_to_three_class_logits(
    damage_logits: torch.Tensor,
    building_logits: torch.Tensor | None,
) -> torch.Tensor:
    if damage_logits.ndim != 4:
        raise EnsembleEvaluationError(f"Expected 4D damage logits, got {damage_logits.shape}")

    channels = damage_logits.shape[1]
    eps = 1e-6
    if channels == 3:
        return damage_logits
    if channels == 5:
        probs = torch.softmax(damage_logits, dim=1)
        collapsed = torch.stack(
            [
                probs[:, 0],
                probs[:, 1],
                probs[:, 2:].sum(dim=1),
            ],
            dim=1,
        )
        return torch.log(collapsed.clamp_min(eps))
    if building_logits is None:
        raise EnsembleEvaluationError(
            "Cannot convert 1/2-channel damage logits to 3-class logits without building logits."
        )

    building_prob = torch.sigmoid(building_logits)
    if building_prob.shape[1] != 1:
        building_prob = building_prob[:, :1]
    if channels == 2:
        damage_prob = torch.softmax(damage_logits, dim=1)
        background = 1.0 - building_prob[:, 0]
        no_damage = building_prob[:, 0] * damage_prob[:, 0]
        damaged = building_prob[:, 0] * damage_prob[:, 1]
        return torch.log(torch.stack([background, no_damage, damaged], dim=1).clamp_min(eps))
    if channels == 1:
        damaged_prob = torch.sigmoid(damage_logits[:, 0])
        background = 1.0 - building_prob[:, 0]
        no_damage = building_prob[:, 0] * (1.0 - damaged_prob)
        damaged = building_prob[:, 0] * damaged_prob
        return torch.log(torch.stack([background, no_damage, damaged], dim=1).clamp_min(eps))
    raise EnsembleEvaluationError(f"Unsupported damage channel count: {channels}")


def apply_ensemble_combo(
    combo: dict[str, Any],
    damage_predictions: list[dict[str, Any]],
    building_predictions: list[dict[str, Any]],
    device: torch.device,
    component_connectivity: int,
) -> torch.Tensor:
    logits_list = [prediction["damage_logits"] for prediction in damage_predictions]
    weights = torch.tensor(
        [prediction["candidate"].config.weight for prediction in damage_predictions],
        dtype=torch.float32,
        device=device,
    )
    preds = ensemble_damage_predictions(
        logits_list,
        weights,
        combo["ensemble_mode"],
        combo["damage_bias"],
    )

    constraint = combo["building_constraint"]
    if constraint == "none":
        return preds

    building_mask = ensemble_building_mask(
        building_predictions,
        combo["building_threshold"],
        device,
    )
    if constraint in {"predicted_building_clip", "building_ensemble_mask"}:
        clipped = preds.clone()
        clipped[~building_mask] = 0
        return clipped
    if constraint == "predicted_building_component_majority":
        return component_majority_batch(preds, building_mask, component_connectivity)
    raise EnsembleEvaluationError(f"Unsupported building constraint: {constraint}")


def ensemble_damage_predictions(
    logits_list: list[torch.Tensor],
    weights: torch.Tensor,
    ensemble_mode: str,
    damage_bias: float,
) -> torch.Tensor:
    if not logits_list:
        raise EnsembleEvaluationError("No damage logits available for ensemble.")
    biased_logits = [add_damage_bias(logits.float(), damage_bias) for logits in logits_list]
    if ensemble_mode == "majority_vote":
        votes = torch.stack([torch.argmax(logits, dim=1) for logits in biased_logits], dim=0)
        one_hot = F.one_hot(votes, num_classes=3).sum(dim=0)
        return torch.argmax(one_hot, dim=-1)

    if ensemble_mode in {"average_logits", "weighted_average_logits"}:
        stacked = torch.stack(biased_logits, dim=0)
        if ensemble_mode == "weighted_average_logits":
            normalized = normalize_weights(weights, stacked.shape[0]).view(-1, 1, 1, 1, 1)
            logits = (stacked * normalized).sum(dim=0)
        else:
            logits = stacked.mean(dim=0)
        return torch.argmax(logits, dim=1)

    if ensemble_mode in {"average_prob", "weighted_average_prob"}:
        probs = torch.stack([torch.softmax(logits, dim=1) for logits in biased_logits], dim=0)
        if ensemble_mode == "weighted_average_prob":
            normalized = normalize_weights(weights, probs.shape[0]).view(-1, 1, 1, 1, 1)
            averaged = (probs * normalized).sum(dim=0)
        else:
            averaged = probs.mean(dim=0)
        return torch.argmax(averaged, dim=1)

    raise EnsembleEvaluationError(f"Unsupported ensemble mode: {ensemble_mode}")


def add_damage_bias(logits: torch.Tensor, damage_bias: float) -> torch.Tensor:
    if damage_bias == 0.0:
        return logits
    biased = logits.clone()
    biased[:, 2] = biased[:, 2] + float(damage_bias)
    return biased


def normalize_weights(weights: torch.Tensor, expected_count: int) -> torch.Tensor:
    if weights.numel() != expected_count:
        return torch.full((expected_count,), 1.0 / expected_count, dtype=torch.float32, device=weights.device)
    clipped = torch.clamp(weights.float(), min=0.0)
    total = clipped.sum()
    if float(total.item()) <= 0.0:
        return torch.full_like(clipped, 1.0 / expected_count)
    return clipped / total


def ensemble_building_mask(
    building_predictions: list[dict[str, Any]],
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    if not building_predictions:
        raise EnsembleEvaluationError("Building constraint requested, but no building logits are available.")
    probs = []
    weights = []
    for prediction in building_predictions:
        logits = prediction["building_logits"]
        if logits is None:
            continue
        if logits.shape[1] != 1:
            logits = logits[:, :1]
        probs.append(torch.sigmoid(logits.float()))
        weights.append(prediction["candidate"].config.weight)
    if not probs:
        raise EnsembleEvaluationError("Building constraint requested, but no building logits are available.")
    stacked = torch.stack(probs, dim=0)
    weight_tensor = normalize_weights(
        torch.tensor(weights, dtype=torch.float32, device=device),
        stacked.shape[0],
    ).view(-1, 1, 1, 1, 1)
    averaged = (stacked * weight_tensor).sum(dim=0)
    return averaged[:, 0] >= float(threshold)


def component_majority_batch(
    preds: torch.Tensor,
    building_mask: torch.Tensor,
    connectivity: int,
) -> torch.Tensor:
    output = torch.zeros_like(preds)
    preds_cpu = preds.detach().cpu().numpy()
    mask_cpu = building_mask.detach().cpu().numpy().astype(bool)
    out_cpu = output.detach().cpu().numpy()
    for batch_index in range(preds_cpu.shape[0]):
        labels = label_connected_components(mask_cpu[batch_index], connectivity)
        for component_id in range(1, int(labels.max()) + 1):
            component = labels == component_id
            if not component.any():
                continue
            component_preds = preds_cpu[batch_index][component]
            no_damage = int(np.count_nonzero(component_preds == 1))
            damaged = int(np.count_nonzero(component_preds == 2))
            out_cpu[batch_index][component] = 2 if damaged > no_damage else 1
    return torch.from_numpy(out_cpu).to(device=preds.device, dtype=preds.dtype)


def label_connected_components(mask: np.ndarray, connectivity: int) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int32)
    current_label = 0
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            labels[y, x] = current_label
            queue: deque[tuple[int, int]] = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current_label
                        queue.append((ny, nx))
    return labels


def build_row(
    combo: dict[str, Any],
    metrics: dict[str, Any],
    loaded_candidates: list[LoadedCandidate],
) -> dict[str, Any]:
    return {
        "tta_mode": combo["tta_mode"],
        "ensemble_mode": combo["ensemble_mode"],
        "damage_bias": combo["damage_bias"],
        "building_constraint": combo["building_constraint"],
        "building_threshold": combo["building_threshold"],
        "num_loaded_candidates": len(loaded_candidates),
        "num_damage_models": sum(
            1 for candidate in loaded_candidates if candidate.config.role in {"damage", "both"}
        ),
        "num_building_models": sum(
            1 for candidate in loaded_candidates if candidate.config.role in {"building", "both"}
        ),
        "test_pixel_accuracy": metric_value(metrics, "pixel_accuracy"),
        "test_mean_iou": metric_value(metrics, "mean_iou"),
        "test_iou_background": metric_value(metrics, "iou_per_class", 0),
        "test_iou_no_damage": metric_value(metrics, "iou_per_class", 1),
        "test_iou_damaged": metric_value(metrics, "iou_per_class", 2),
        "test_precision_damaged": metric_value(metrics, "precision_per_class", 2),
        "test_recall_damaged": metric_value(metrics, "recall_per_class", 2),
        "test_f1_damaged": metric_value(metrics, "f1_per_class", 2),
    }


def metric_value(metrics: dict[str, Any], key: str, class_index: int | None = None) -> float | None:
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


def none_to_negative(value: Any) -> float:
    if value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def save_batch_examples(
    output_dir: Path,
    batch: dict[str, Any],
    preds: torch.Tensor,
    combo: dict[str, Any],
    start_index: int,
    max_examples: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    targets = batch["target"].detach().cpu()
    pair_ids = batch.get("pair_id", [f"sample_{idx}" for idx in range(images.shape[0])])
    saved = 0
    for idx in range(images.shape[0]):
        if start_index + saved >= max_examples:
            break
        pre = tensor_rgb_to_uint8(images[idx, :3])
        post = tensor_rgb_to_uint8(images[idx, 3:])
        target_rgb = mask_to_rgb(targets[idx].numpy())
        pred_rgb = mask_to_rgb(preds[idx].numpy())
        overlay = overlay_mask(post, preds[idx].numpy())
        safe_pair_id = str(pair_ids[idx]).replace("/", "_").replace("\\", "_")
        filename = output_dir / f"{start_index + saved:03d}_{safe_pair_id}_ensemble.png"
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        titles = ["Pre", "Post", "Target", "Prediction", "Overlay"]
        panels = [pre, post, target_rgb, pred_rgb, overlay]
        for axis, title, panel in zip(axes, titles, panels):
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"{combo['ensemble_mode']} | TTA {combo['tta_mode']} | "
            f"{combo['building_constraint']} | bias {combo['damage_bias']}"
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        saved += 1
    return saved


def tensor_rgb_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.moveaxis(array, 0, -1)
    return np.asarray(np.rint(array * 255.0), dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask.astype(np.int64), 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[clipped]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colors = mask_to_rgb(mask).astype(np.float32)
    output = image.astype(np.float32).copy()
    visible = mask > 0
    output[visible] = (1.0 - alpha) * output[visible] + alpha * colors[visible]
    return np.asarray(np.clip(output, 0, 255), dtype=np.uint8)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "tta_mode",
        "ensemble_mode",
        "damage_bias",
        "building_constraint",
        "building_threshold",
        "num_loaded_candidates",
        "num_damage_models",
        "num_building_models",
        "test_pixel_accuracy",
        "test_mean_iou",
        "test_iou_background",
        "test_iou_no_damage",
        "test_iou_damaged",
        "test_precision_damaged",
        "test_recall_damaged",
        "test_f1_damaged",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def print_summary(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    print("\nDamage Ensemble Evaluation")
    print("=" * 32)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if not rows:
        print("No rows were evaluated.")
        return
    print("\nTop rows by F1 damaged / IoU damaged:")
    for row in rows[:10]:
        print(
            f"{row['ensemble_mode']} | TTA={row['tta_mode']} | "
            f"constraint={row['building_constraint']} | threshold={row['building_threshold']} | "
            f"bias={row['damage_bias']} | IoU damaged={fmt(row['test_iou_damaged'])} | "
            f"F1 damaged={fmt(row['test_f1_damaged'])} | mean IoU={fmt(row['test_mean_iou'])}"
        )


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "NA"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        started_at = time.time()
        device = resolve_device(args.device)
        use_amp = bool(args.amp and device.type == "cuda")
        candidates = read_candidate_csv(args.candidates_csv)
        loaded, load_warnings = load_candidates(candidates, device)
        damage_loaded = [candidate for candidate in loaded if candidate.config.role in {"damage", "both"}]
        if len(damage_loaded) < args.min_damage_models:
            raise EnsembleEvaluationError(
                f"Only {len(damage_loaded)} damage candidates loaded; "
                f"--min-damage-models={args.min_damage_models}."
            )

        dataset = load_dataset(args)
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        rows, eval_warnings, examples_saved = evaluate_ensembles(
            loaded,
            loader,
            args,
            device,
            use_amp,
        )
        warnings = load_warnings + eval_warnings
        payload = {
            "config": {
                "candidates_csv": str(args.candidates_csv),
                "root": str(args.root),
                "split_csv": str(args.split_csv),
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "target_mode": args.target_mode,
                "device": str(device),
                "amp": use_amp,
                "tta_modes": args.tta_modes,
                "ensemble_modes": args.ensemble_modes,
                "damage_biases": args.damage_biases,
                "building_constraints": args.building_constraints,
                "building_thresholds": args.building_thresholds,
                "component_connectivity": args.component_connectivity,
            },
            "loaded_candidates": [
                {
                    "name": candidate.config.name,
                    "role": candidate.config.role,
                    "family": candidate.config.family,
                    "model": candidate.config.model,
                    "checkpoint": str(candidate.config.checkpoint),
                    "weight": candidate.config.weight,
                }
                for candidate in loaded
            ],
            "warnings": warnings,
            "dataset_size": len(dataset),
            "elapsed_seconds": time.time() - started_at,
            "rows": rows,
            "examples_saved": examples_saved,
        }
        write_json(args.output_json, payload)
        write_csv(args.output_csv, rows)
        if args.ranked_csv is not None:
            write_csv(args.ranked_csv, rows)
        print_summary(rows, warnings)
        return 0
    except (EnsembleEvaluationError, XBDDatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

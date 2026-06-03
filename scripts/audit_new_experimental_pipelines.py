#!/usr/bin/env python
"""Audit newly added experimental pipelines before Rorqual runs.

The default audit is static and does not touch data, outputs, checkpoints or
logs. Use --run-model-smoke when dependencies are available and you want to
instantiate models on CPU for dummy shape checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


PYTHON_FILES = [
    "scripts/export_xview2_format.py",
    "scripts/evaluate_xview2_style_metrics.py",
    "src/crisismap/models/multitemporal_fusion.py",
    "scripts/train_multitemporal_fusion.py",
    "scripts/evaluate_multitemporal_fusion.py",
    "src/crisismap/models/xview2_strong_baseline.py",
    "scripts/train_xview2_strong_baseline.py",
    "scripts/evaluate_xview2_strong_baseline.py",
]

CSV_FILES = [
    "configs/multitemporal_fusion_sweep_v1.csv",
    "configs/multitemporal_fusion_sweep_v2.csv",
    "configs/xview2_strong_baseline_sweep_v1.csv",
    "configs/xview2_strong_baseline_sweep_v2.csv",
]

SLURM_FILES = [
    "slurm/run_multitemporal_fusion_config.sh",
    "slurm/submit_multitemporal_fusion_sweep_v1.sh",
    "slurm/smoke_multitemporal_fusion.sbatch",
    "slurm/run_multitemporal_fusion_v2_config.sh",
    "slurm/submit_multitemporal_fusion_sweep_v2.sh",
    "slurm/smoke_multitemporal_fusion_v2.sbatch",
    "slurm/run_xview2_strong_baseline_config.sh",
    "slurm/submit_xview2_strong_baseline_sweep_v1.sh",
    "slurm/run_xview2_strong_baseline_v2_config.sh",
    "slurm/submit_xview2_strong_baseline_sweep_v2.sh",
    "slurm/smoke_xview2_strong_baseline.sbatch",
]

DOC_FILES = [
    "docs/multitemporal_fusion_plan.md",
    "docs/multitemporal_fusion_v2_plan.md",
    "docs/xview2_strong_baseline_plan.md",
    "docs/xview2_strong_baseline_v2_plan.md",
    "docs/xview2_metric_plan.md",
]

WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\|\\\\Users\\\\|\\Users\\")


class AuditFailure(Exception):
    """Raised when an audit check fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit new experimental pipelines.")
    parser.add_argument(
        "--run-model-smoke",
        action="store_true",
        help="Instantiate models and run dummy forward passes on CPU.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def read_text(path: str | Path) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def check_exists(path: str, results: list[dict[str, Any]]) -> None:
    exists = (PROJECT_ROOT / path).exists()
    results.append({"check": "file_exists", "path": path, "ok": exists})
    if not exists:
        raise AuditFailure(f"Missing required file: {path}")


def check_no_windows_paths(paths: list[str], results: list[dict[str, Any]]) -> None:
    offenders = []
    for path in paths:
        text = read_text(path)
        if WINDOWS_PATH_RE.search(text):
            offenders.append(path)
    ok = not offenders
    results.append({"check": "no_local_windows_paths", "ok": ok, "offenders": offenders})
    if not ok:
        raise AuditFailure(f"Found local Windows path assumptions: {offenders}")


def check_csv_plain(path: str, results: list[dict[str, Any]]) -> list[dict[str, str]]:
    full_path = PROJECT_ROOT / path
    text = full_path.read_text(encoding="utf-8")
    quoted = '"' in text
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise AuditFailure(f"CSV is empty: {path}")
    expected_len = len(rows[0])
    ragged = [
        {"line": idx + 1, "fields": len(row), "expected": expected_len}
        for idx, row in enumerate(rows)
        if len(row) != expected_len
    ]
    with full_path.open("r", encoding="utf-8-sig", newline="") as f:
        dict_rows = list(csv.DictReader(f))
    experiments = [str(row.get("experiment", "")).strip() for row in dict_rows]
    duplicates = sorted({name for name in experiments if experiments.count(name) > 1})
    ok = not quoted and not ragged and not duplicates
    results.append(
        {
            "check": "csv_plain_comma_separated",
            "path": path,
            "ok": ok,
            "has_quotes": quoted,
            "ragged_rows": ragged,
            "duplicate_experiments": duplicates,
            "rows": len(dict_rows),
        }
    )
    if not ok:
        raise AuditFailure(f"CSV audit failed for {path}")
    return dict_rows


def check_unique_checkpoint_folders(results: list[dict[str, Any]]) -> None:
    experiments = []
    for path in CSV_FILES:
        with (PROJECT_ROOT / path).open("r", encoding="utf-8-sig", newline="") as f:
            experiments.extend(str(row["experiment"]).strip() for row in csv.DictReader(f))
    duplicates = sorted({name for name in experiments if experiments.count(name) > 1})
    ok = not duplicates
    results.append(
        {
            "check": "unique_output_checkpoint_folders",
            "ok": ok,
            "checkpoint_folder_pattern": "outputs/checkpoints/<experiment>",
            "duplicate_experiments": duplicates,
            "num_experiments": len(experiments),
        }
    )
    if not ok:
        raise AuditFailure(f"Duplicate experiment/checkpoint folders: {duplicates}")


def check_slurm(path: str, results: list[dict[str, Any]]) -> None:
    text = read_text(path)
    no_partition = "--partition" not in text and "#SBATCH -p" not in text
    has_mail = "--mail-user=t.gourjault@gmail.com" in text and "BEGIN,END,FAIL,TIME_LIMIT" in text
    requires_runtime_paths = path.startswith("slurm/run_") or path.startswith("slurm/smoke_")
    has_scratch = "${SCRATCH}/CrisisMap-AI" in text if requires_runtime_paths else True
    mkdir_p = "mkdir -p" in text if requires_runtime_paths else True
    ok = has_scratch and no_partition and mkdir_p and (has_mail or path.startswith("slurm/submit_"))
    results.append(
        {
            "check": "slurm_runtime_paths_and_directives",
            "path": path,
            "ok": ok,
            "uses_scratch": has_scratch,
            "no_explicit_partition": no_partition,
            "mkdir_p": mkdir_p,
            "mail_notifications": has_mail,
        }
    )
    if not ok:
        raise AuditFailure(f"SLURM audit failed for {path}")


def check_runner_logic(path: str, results: list[dict[str, Any]]) -> None:
    text = read_text(path)
    skip_complete = "Run is complete" in text or "skip complete" in text
    evaluate_only = "evaluate_only" in text
    force_incomplete = "FORCE_INCOMPLETE" in text
    refuse_partial = "Incomplete" in text or "incomplete" in text
    ok = skip_complete and evaluate_only and force_incomplete and refuse_partial
    results.append(
        {
            "check": "safe_runner_logic",
            "path": path,
            "ok": ok,
            "skip_complete": skip_complete,
            "evaluate_only": evaluate_only,
            "force_incomplete": force_incomplete,
            "refuse_partial": refuse_partial,
        }
    )
    if not ok:
        raise AuditFailure(f"Runner safety logic audit failed for {path}")


def check_amp_and_mapping(results: list[dict[str, Any]]) -> None:
    training_text = "\n".join(read_text(path) for path in [
        "scripts/train_multitemporal_fusion.py",
        "scripts/train_xview2_strong_baseline.py",
    ])
    amp_ok = "--amp" in training_text and ("autocast" in training_text or "amp" in training_text)
    xview_text = read_text("scripts/export_xview2_format.py") + read_text(
        "docs/xview2_metric_plan.md"
    )
    mapping_ok = all(token in xview_text for token in ["background", "no_damage", "damaged"])
    five_class_doc_ok = "not comparable" in xview_text and "5-class" in xview_text
    results.append(
        {
            "check": "amp_target_mapping_and_5class_documentation",
            "ok": amp_ok and mapping_ok and five_class_doc_ok,
            "amp": amp_ok,
            "target_mapping_3class": mapping_ok,
            "future_5class_documented": five_class_doc_ok,
        }
    )
    if not (amp_ok and mapping_ok and five_class_doc_ok):
        raise AuditFailure("AMP/mapping/5-class documentation audit failed.")


def check_losses_and_metrics(results: list[dict[str, Any]]) -> None:
    strong_train = read_text("scripts/train_xview2_strong_baseline.py")
    mtf_eval = read_text("scripts/evaluate_multitemporal_fusion.py")
    xv2_eval = read_text("scripts/evaluate_xview2_strong_baseline.py")
    loss_ok = all(
        token in strong_train
        for token in [
            "building_targets = (targets > 0)",
            "damage_targets = (targets > 1)",
            "multiclass_focal_dice_loss",
        ]
    )
    metrics_ok = "metrics_from_confusion" in mtf_eval and "metrics_from_confusion" in xv2_eval
    results.append(
        {
            "check": "loss_background_handling_and_metric_consistency",
            "ok": loss_ok and metrics_ok,
            "loss_background_handling": (
                "Multiclass mode supervises background explicitly; multilabel mode "
                "uses building=(target>0) and damage=(target>1)."
            ),
            "uses_existing_metrics": metrics_ok,
        }
    )
    if not (loss_ok and metrics_ok):
        raise AuditFailure("Loss/metrics audit failed.")


def run_model_smoke(results: list[dict[str, Any]]) -> None:
    import torch

    from crisismap.models.multitemporal_fusion import create_multitemporal_fusion_model
    from crisismap.models.xview2_strong_baseline import create_xview2_strong_baseline_model

    checks = []
    x = torch.randn(1, 6, 128, 128)
    for model_name in ["resnet34_unet_shared", "resnet50_unet_shared", "resnet34_unet_attention"]:
        model = create_xview2_strong_baseline_model(model_name, damage_channels=3)
        model.eval()
        with torch.no_grad():
            out = model(x)
        checks.append(
            {
                "model": model_name,
                "building_shape": list(out["building_logits"].shape),
                "damage_shape": list(out["damage_logits"].shape),
                "ok": list(out["building_logits"].shape) == [1, 1, 128, 128]
                and list(out["damage_logits"].shape) == [1, 3, 128, 128],
            }
        )
    for model_name in [
        "mtf_resnet50_fpn_shared",
        "mtf_resnet34_fpn_shared",
        "mtf_resnet50_fpn_attention",
        "mtf_resnet50_fpn_gated",
        "mtf_resnet50_deeplab_shared",
        "mtf_effb3_deeplab_shared",
        "control_6ch_resnet50_fpn",
        "control_current_siamese_attention",
    ]:
        model = create_multitemporal_fusion_model(model_name, damage_channels=3)
        model.eval()
        with torch.no_grad():
            out = model(x)
        checks.append(
            {
                "model": model_name,
                "building_shape": list(out["building_logits"].shape),
                "damage_shape": list(out["damage_logits"].shape),
                "ok": list(out["building_logits"].shape) == [1, 1, 128, 128]
                and list(out["damage_logits"].shape) == [1, 3, 128, 128],
            }
        )
    ok = all(row["ok"] for row in checks)
    results.append({"check": "model_dummy_forward_shapes", "ok": ok, "models": checks})
    if not ok:
        raise AuditFailure("Model shape smoke failed.")


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []

    for path in PYTHON_FILES + CSV_FILES + SLURM_FILES + DOC_FILES:
        check_exists(path, results)

    check_no_windows_paths(PYTHON_FILES + CSV_FILES + SLURM_FILES + DOC_FILES, results)

    for path in CSV_FILES:
        check_csv_plain(path, results)
    check_unique_checkpoint_folders(results)

    for path in SLURM_FILES:
        check_slurm(path, results)
    for path in [
        "slurm/run_multitemporal_fusion_config.sh",
        "slurm/run_multitemporal_fusion_v2_config.sh",
        "slurm/run_xview2_strong_baseline_config.sh",
        "slurm/run_xview2_strong_baseline_v2_config.sh",
    ]:
        check_runner_logic(path, results)

    check_amp_and_mapping(results)
    check_losses_and_metrics(results)

    if args.run_model_smoke:
        run_model_smoke(results)

    payload = {"ok": True, "checks": results}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print("Audit OK")
    for row in results:
        print(f"- {row['check']}: {'OK' if row.get('ok') else 'FAIL'}")


if __name__ == "__main__":
    try:
        main()
    except AuditFailure as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

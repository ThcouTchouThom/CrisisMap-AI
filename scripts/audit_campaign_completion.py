"""Audit checkpoint/history/metrics completion for Rorqual campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


CAMPAIGNS = {"damage_extra", "building100"}


class AuditError(Exception):
    """Raised when campaign audit cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit campaign completion state.")
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument(
        "--damage-config",
        type=Path,
        default=Path("configs/damage_extra_sweep_v1.csv"),
    )
    parser.add_argument(
        "--building-config",
        type=Path,
        default=None,
        help=(
            "Building100 config CSV. Defaults to "
            "configs/building100_sweep_v1_relaunch.csv if present, else original."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AuditError(f"Config CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise AuditError(f"Config CSV is empty: {path}")
    return rows


def load_history(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return False, 0, 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, 0, 0
    history = (
        payload
        if isinstance(payload, list)
        else payload.get("history") if isinstance(payload, dict) else None
    )
    if not isinstance(history, list):
        return True, 0, 0

    epochs: list[int] = []
    for item in history:
        if isinstance(item, dict) and "epoch" in item:
            try:
                epochs.append(int(item["epoch"]))
            except (TypeError, ValueError):
                pass
    return True, len(history), max(epochs) if epochs else 0


def complete_from_history(epoch_count: int, last_epoch: int, expected_epochs: int) -> bool:
    return epoch_count >= expected_epochs and last_epoch >= expected_epochs


def suggested_action(
    history_is_complete: bool,
    best_exists: bool,
    last_exists: bool,
    metrics_exists: bool,
    checkpoint_dir_exists: bool,
) -> str:
    if history_is_complete and best_exists and metrics_exists:
        return "complete_skip"
    if history_is_complete and best_exists and not metrics_exists:
        return "evaluate_only"
    if last_exists:
        return "resume"
    if not checkpoint_dir_exists:
        return "train_missing"
    return "force_retrain_needed"


def damage_rows(config_rows: list[dict[str, str]], checkpoints_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config in config_rows:
        experiment = config["experiment"]
        checkpoint_dir = checkpoints_dir / experiment
        history_path = checkpoint_dir / "metrics_history.json"
        best_path = checkpoint_dir / "best_unet.pt"
        last_path = checkpoint_dir / "last_unet.pt"
        metrics_path = output_dir / f"{experiment}_test_metrics.json"
        expected_epochs = int(config["epochs"])
        history_exists, epoch_count, last_epoch = load_history(history_path)
        history_is_complete = complete_from_history(epoch_count, last_epoch, expected_epochs)
        best_exists = best_path.exists()
        last_exists = last_path.exists()
        metrics_exists = metrics_path.exists()
        complete = history_is_complete and best_exists and metrics_exists
        rows.append(
            {
                "campaign": "damage_extra",
                "experiment": experiment,
                "expected_epochs": expected_epochs,
                "checkpoint_dir": str(checkpoint_dir),
                "history_exists": history_exists,
                "epoch_count": epoch_count,
                "last_epoch": last_epoch,
                "best_checkpoint_exists": best_exists,
                "last_checkpoint_exists": last_exists,
                "test_metrics_exists": metrics_exists,
                "complete": complete,
                "suggested_action": suggested_action(
                    history_is_complete,
                    best_exists,
                    last_exists,
                    metrics_exists,
                    checkpoint_dir.exists(),
                ),
                "split": config["split"],
                "augment_mode": config["augment_mode"],
                "sampler": config["sampler"],
                "damage_sampling_alpha": config["damage_sampling_alpha"],
                "model": "",
                "input_mode": "",
                "loss": config["loss"],
            }
        )
    return rows


def building_rows(config_rows: list[dict[str, str]], checkpoints_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config in config_rows:
        experiment = config["experiment"]
        checkpoint_dir = checkpoints_dir / experiment
        history_path = checkpoint_dir / "metrics_history.json"
        best_path = checkpoint_dir / "best_building.pt"
        last_path = checkpoint_dir / "last_building.pt"
        metrics_json = output_dir / f"{experiment}_building_test_metrics.json"
        metrics_csv = output_dir / f"{experiment}_building_test_metrics.csv"
        expected_epochs = int(config["epochs"])
        history_exists, epoch_count, last_epoch = load_history(history_path)
        history_is_complete = complete_from_history(epoch_count, last_epoch, expected_epochs)
        best_exists = best_path.exists()
        last_exists = last_path.exists()
        metrics_exists = metrics_json.exists() and metrics_csv.exists()
        complete = history_is_complete and best_exists and metrics_exists
        rows.append(
            {
                "campaign": "building100",
                "experiment": experiment,
                "expected_epochs": expected_epochs,
                "checkpoint_dir": str(checkpoint_dir),
                "history_exists": history_exists,
                "epoch_count": epoch_count,
                "last_epoch": last_epoch,
                "best_checkpoint_exists": best_exists,
                "last_checkpoint_exists": last_exists,
                "test_metrics_exists": metrics_exists,
                "complete": complete,
                "suggested_action": suggested_action(
                    history_is_complete,
                    best_exists,
                    last_exists,
                    metrics_exists,
                    checkpoint_dir.exists(),
                ),
                "split": config["train_csv"],
                "augment_mode": config["augment_mode"],
                "sampler": config["sampler"],
                "damage_sampling_alpha": "",
                "model": config["model"],
                "input_mode": config["input_mode"],
                "loss": config["loss"],
            }
        )
    return rows


def fieldnames() -> list[str]:
    return [
        "campaign",
        "experiment",
        "expected_epochs",
        "checkpoint_dir",
        "history_exists",
        "epoch_count",
        "last_epoch",
        "best_checkpoint_exists",
        "last_checkpoint_exists",
        "test_metrics_exists",
        "complete",
        "suggested_action",
        "split",
        "augment_mode",
        "sampler",
        "damage_sampling_alpha",
        "model",
        "input_mode",
        "loss",
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Audit campagne {rows[0]['campaign'] if rows else ''}",
        "",
        "| experiment | expected | last epoch | best | last | metrics | complete | suggested action |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {expected_epochs} | {last_epoch} | "
            "{best_checkpoint_exists} | {last_checkpoint_exists} | "
            "{test_metrics_exists} | {complete} | {suggested_action} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_building_config(path: Path | None) -> Path:
    if path is not None:
        return path
    relaunch = Path("configs/building100_sweep_v1_relaunch.csv")
    return relaunch if relaunch.exists() else Path("configs/building100_sweep_v1.csv")


def run_audit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser()
    checkpoints_dir = args.checkpoints_dir.expanduser()
    if args.campaign == "damage_extra":
        rows = damage_rows(read_csv(args.damage_config), checkpoints_dir, output_dir)
    else:
        rows = building_rows(
            read_csv(resolve_building_config(args.building_config)),
            checkpoints_dir,
            output_dir,
        )

    csv_path = output_dir / f"{args.campaign}_audit.csv"
    md_path = output_dir / f"{args.campaign}_audit.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row["suggested_action"])
        counts[action] = counts.get(action, 0) + 1
    for action, count in sorted(counts.items()):
        print(f"{action}: {count}")


def main() -> int:
    args = parse_args()
    try:
        run_audit(args)
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

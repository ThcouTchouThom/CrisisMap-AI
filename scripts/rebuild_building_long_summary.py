"""Rebuild the long building segmentation campaign summary."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_building100_summary import (  # noqa: E402
    SummaryError,
    build_fieldnames,
    build_row,
    read_csv,
    sort_key,
)


DEFAULT_CONFIG = Path("configs/building_long_sweep_v1.csv")
DEFAULT_OUTPUT = Path("outputs/predictions/building_long_sweep_v1_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the long building segmentation sweep summary CSV.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        config_rows = read_csv(args.config)
        rows: list[dict[str, object]] = []
        for index, config_row in enumerate(config_rows, start=2):
            row = build_row(
                config_row=config_row,
                predictions_dir=args.predictions_dir,
                checkpoints_dir=args.checkpoints_dir,
                csv_path=args.config,
                row_index=index,
            )
            if row is not None:
                rows.append(row)
        rows.sort(key=sort_key, reverse=True)
        write_summary(args.output, rows)
    except SummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved long building summary: {args.output}")
    print(f"Rows with metrics: {len(rows)}")
    if rows:
        best = rows[0]
        print(
            "Best row: "
            f"{best.get('experiment')} | "
            f"best IoU={best.get('best_threshold_building_iou')} | "
            f"best F1={best.get('best_threshold_f1')} | "
            f"best recall={best.get('best_threshold_recall')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

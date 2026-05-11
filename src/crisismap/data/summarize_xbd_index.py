"""Summarize a CSV index generated from the xBD/xView2 training dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


TARGET_CLASSES = (0, 1, 2, 3, 4)
DAMAGE_CLASSES = (2, 3, 4)
REQUIRED_COLUMNS = {
    "pair_id",
    "disaster",
    "target_value_counts",
    "target_total_pixels",
    "target_nonzero_ratio",
}


class XbdSummaryError(Exception):
    """Raised when the xBD CSV index cannot be summarized."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a CSV index produced by build_xbd_index.py."
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="Path to xbd_train_index.csv.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top pairs to print for ranked sections.",
    )
    return parser.parse_args()


def load_index(index_path: Path) -> pd.DataFrame:
    index_path = index_path.expanduser().resolve()
    if not index_path.exists():
        raise XbdSummaryError(f"Index file does not exist: {index_path}")
    if not index_path.is_file():
        raise XbdSummaryError(f"Index path is not a file: {index_path}")

    try:
        df = pd.read_csv(index_path)
    except OSError as exc:
        raise XbdSummaryError(f"Could not read index CSV '{index_path}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise XbdSummaryError(f"Index CSV is empty: {index_path}") from exc
    except pd.errors.ParserError as exc:
        raise XbdSummaryError(f"Could not parse index CSV '{index_path}': {exc}") from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise XbdSummaryError(
            "Index CSV is missing required column(s): " + ", ".join(missing_columns)
        )
    if df.empty:
        raise XbdSummaryError("Index CSV has no rows.")

    return df


def parse_target_counts(value: object, pair_id: str) -> dict[int, int]:
    if pd.isna(value):
        raise XbdSummaryError(f"Missing target_value_counts for pair '{pair_id}'.")

    try:
        raw_counts = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise XbdSummaryError(
            f"Invalid target_value_counts JSON for pair '{pair_id}': {exc}"
        ) from exc

    if not isinstance(raw_counts, dict):
        raise XbdSummaryError(
            f"target_value_counts must be a JSON object for pair '{pair_id}'."
        )

    counts = {target_class: 0 for target_class in TARGET_CLASSES}
    for raw_key, raw_count in raw_counts.items():
        target_class = parse_target_class(raw_key)
        if target_class not in counts:
            continue
        try:
            counts[target_class] += int(raw_count)
        except (TypeError, ValueError) as exc:
            raise XbdSummaryError(
                f"Invalid pixel count for class '{raw_key}' in pair '{pair_id}'."
            ) from exc

    return counts


def parse_target_class(raw_key: object) -> int | None:
    key = str(raw_key).strip()
    try:
        return int(key)
    except ValueError:
        pass

    parsed_json_key = parse_json_class_key(key)
    if parsed_json_key is not None:
        return parsed_json_key

    try:
        float_value = float(key)
    except ValueError:
        return None

    if float_value.is_integer():
        return int(float_value)
    return None


def parse_json_class_key(key: str) -> int | None:
    try:
        parsed = json.loads(key)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list) or not parsed:
        return None

    try:
        values = [float(value) for value in parsed]
    except (TypeError, ValueError):
        return None

    first_value = values[0]
    if not first_value.is_integer():
        return None
    if all(value == first_value for value in values):
        return int(first_value)
    return None


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        counts = parse_target_counts(getattr(row, "target_value_counts"), pair_id)
        total_pixels = parse_total_pixels(getattr(row, "target_total_pixels"), pair_id)
        damage_pixels = sum(counts[target_class] for target_class in DAMAGE_CLASSES)
        damage_ratio = damage_pixels / total_pixels if total_pixels else 0.0

        parsed_row = row._asdict()
        parsed_row["target_total_pixels"] = total_pixels
        for target_class in TARGET_CLASSES:
            parsed_row[f"class_{target_class}_pixels"] = counts[target_class]
            parsed_row[f"has_class_{target_class}"] = counts[target_class] > 0
        parsed_row["damage_pixels"] = damage_pixels
        parsed_row["damage_ratio"] = damage_ratio
        rows.append(parsed_row)

    enriched = pd.DataFrame(rows)
    enriched["target_nonzero_ratio"] = pd.to_numeric(
        enriched["target_nonzero_ratio"],
        errors="coerce",
    ).fillna(0.0)
    return enriched


def parse_total_pixels(value: object, pair_id: str) -> int:
    if pd.isna(value):
        raise XbdSummaryError(f"Missing target_total_pixels for pair '{pair_id}'.")

    try:
        total_pixels = int(float(value))
    except (TypeError, ValueError) as exc:
        raise XbdSummaryError(
            f"Invalid target_total_pixels for pair '{pair_id}': {value}"
        ) from exc

    if total_pixels < 0:
        raise XbdSummaryError(
            f"target_total_pixels must be non-negative for pair '{pair_id}'."
        )
    return total_pixels


def print_count_table(series: pd.Series, value_name: str) -> None:
    for name, value in series.items():
        print(f"  {name}: {format_number(value)} {value_name}")


def format_number(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def print_target_pixel_summary(df: pd.DataFrame) -> None:
    class_totals = {
        target_class: int(df[f"class_{target_class}_pixels"].sum())
        for target_class in TARGET_CLASSES
    }
    total_pixels = int(df["target_total_pixels"].sum())

    print("Total Pixels Per Target Class")
    for target_class in TARGET_CLASSES:
        print(f"  class {target_class}: {class_totals[target_class]}")
    print()

    print("Percentage Of Pixels Per Target Class")
    for target_class in TARGET_CLASSES:
        pct = (class_totals[target_class] / total_pixels * 100) if total_pixels else 0.0
        print(f"  class {target_class}: {pct:.4f}%")
    print()

    print("Pairs Containing Each Target Class")
    for target_class in TARGET_CLASSES:
        pair_count = int(df[f"has_class_{target_class}"].sum())
        print(f"  class {target_class}: {pair_count}")
    print()


def print_ratio_by_disaster(df: pd.DataFrame) -> None:
    print("Average Nonzero Target Ratio Per Disaster")
    nonzero_by_disaster = df.groupby("disaster")["target_nonzero_ratio"].mean()
    nonzero_by_disaster = nonzero_by_disaster.sort_values(
        ascending=False,
    )
    for disaster, ratio in nonzero_by_disaster.items():
        print(f"  {disaster}: {ratio:.6f}")
    print()

    print("Average Damage Ratio Per Disaster")
    damage_by_disaster = (
        df.groupby("disaster")["damage_ratio"].mean().sort_values(ascending=False)
    )
    for disaster, ratio in damage_by_disaster.items():
        print(f"  {disaster}: {ratio:.6f}")
    print()


def print_ranked_pairs(df: pd.DataFrame, column: str, title: str, top_n: int) -> None:
    print(title)
    ranked = df.sort_values(column, ascending=False).head(top_n)
    for row in ranked.itertuples(index=False):
        print(
            f"  {row.pair_id}: {getattr(row, column):.6f} "
            f"({row.disaster})"
        )
    print()


def recommended_candidates(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = df.sort_values(
        ["damage_ratio", "target_nonzero_ratio"],
        ascending=[False, False],
    )
    per_disaster = ranked.groupby("disaster", as_index=False).head(1)
    remaining_slots = max(top_n - len(per_disaster), 0)

    if remaining_slots:
        remaining = ranked[~ranked["pair_id"].isin(per_disaster["pair_id"])].head(
            remaining_slots
        )
        candidates = pd.concat([per_disaster, remaining], ignore_index=True)
    else:
        candidates = per_disaster.head(top_n).copy()

    return candidates.sort_values(
        ["damage_ratio", "target_nonzero_ratio"],
        ascending=[False, False],
    ).head(top_n)


def print_recommendations(df: pd.DataFrame, top_n: int) -> None:
    print("Recommended Candidate Pairs For Visualization/Demo")
    candidates = recommended_candidates(df, top_n)
    for row in candidates.itertuples(index=False):
        print(
            f"  {row.pair_id}: damage={row.damage_ratio:.6f}, "
            f"nonzero={row.target_nonzero_ratio:.6f}, disaster={row.disaster}"
        )
    print()


def print_summary(df: pd.DataFrame, top_n: int) -> None:
    print("CrisisMap AI - xBD/xView2 Index Summary")
    print("=" * 45)
    print(f"Total pairs: {len(df)}")
    print()

    print("Pairs Per Disaster")
    disaster_counts = df["disaster"].value_counts().sort_index()
    print_count_table(disaster_counts, "pairs")
    print()

    print_target_pixel_summary(df)
    print_ratio_by_disaster(df)
    print_ranked_pairs(
        df,
        "target_nonzero_ratio",
        f"Top {top_n} Pairs By Nonzero Target Ratio",
        top_n,
    )
    print_ranked_pairs(
        df,
        "damage_ratio",
        f"Top {top_n} Pairs By Damage Ratio",
        top_n,
    )
    print_recommendations(df, top_n)


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        print("ERROR: --top-n must be a positive integer.", file=sys.stderr)
        return 1

    try:
        df = load_index(args.index)
        enriched = add_target_columns(df)
        print_summary(enriched, args.top_n)
    except XbdSummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create train/validation/test CSV splits from an xBD/xView2 index."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DAMAGE_CLASSES = {2, 3, 4}
REQUIRED_COLUMNS = {
    "pair_id",
    "disaster",
    "target_nonzero_ratio",
}


class XbdSplitError(Exception):
    """Raised when xBD split creation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test split CSVs from xBD index metadata."
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="Path to xbd_train_index.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where train_pairs.csv, val_pairs.csv, and test_pairs.csv are saved.",
    )
    parser.add_argument(
        "--disasters",
        nargs="+",
        default=None,
        help="Optional list of disaster names to keep.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation fraction of the selected dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test fraction of the selected dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling and splitting.",
    )
    parser.add_argument(
        "--min-nonzero-ratio",
        type=float,
        default=0.0,
        help="Minimum target_nonzero_ratio to keep.",
    )
    parser.add_argument(
        "--min-damage-ratio",
        type=float,
        default=0.0,
        help="Minimum damage ratio to keep, using target classes 2, 3, and 4.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional maximum number of selected pairs for quick experiments.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_size < 1.0:
        raise XbdSplitError("--val-size must be between 0 and 1.")
    if not 0.0 < args.test_size < 1.0:
        raise XbdSplitError("--test-size must be between 0 and 1.")
    if args.val_size + args.test_size >= 1.0:
        raise XbdSplitError("--val-size + --test-size must be less than 1.")
    if args.min_nonzero_ratio < 0.0:
        raise XbdSplitError("--min-nonzero-ratio must be non-negative.")
    if args.min_damage_ratio < 0.0:
        raise XbdSplitError("--min-damage-ratio must be non-negative.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise XbdSplitError("--max-pairs must be a positive integer.")


def load_index(index_path: Path) -> pd.DataFrame:
    index_path = index_path.expanduser().resolve()
    if not index_path.exists():
        raise XbdSplitError(f"Index file does not exist: {index_path}")
    if not index_path.is_file():
        raise XbdSplitError(f"Index path is not a file: {index_path}")

    try:
        df = pd.read_csv(index_path)
    except OSError as exc:
        raise XbdSplitError(f"Could not read index CSV '{index_path}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise XbdSplitError(f"Index CSV is empty: {index_path}") from exc
    except pd.errors.ParserError as exc:
        raise XbdSplitError(f"Could not parse index CSV '{index_path}': {exc}") from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise XbdSplitError(
            "Index CSV is missing required column(s): " + ", ".join(missing_columns)
        )
    if df.empty:
        raise XbdSplitError("Index CSV has no rows.")
    if df["pair_id"].duplicated().any():
        duplicates = sorted(df.loc[df["pair_id"].duplicated(), "pair_id"].unique())
        raise XbdSplitError(
            f"Index CSV contains duplicate pair_id values: {duplicates[:5]}"
        )

    return df


def add_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_nonzero_ratio"] = pd.to_numeric(
        df["target_nonzero_ratio"],
        errors="coerce",
    )
    if df["target_nonzero_ratio"].isna().any():
        raise XbdSplitError("target_nonzero_ratio contains missing or non-numeric values.")

    if "damage_ratio" in df.columns:
        df["damage_ratio"] = pd.to_numeric(df["damage_ratio"], errors="coerce")
        if df["damage_ratio"].isna().any():
            raise XbdSplitError("damage_ratio contains missing or non-numeric values.")
        return df

    required_for_damage = {"target_value_counts", "target_total_pixels"}
    missing = sorted(required_for_damage - set(df.columns))
    if missing:
        raise XbdSplitError(
            "Index CSV needs either damage_ratio or column(s): " + ", ".join(missing)
        )

    df["damage_ratio"] = [
        compute_damage_ratio(
            row.target_value_counts,
            row.target_total_pixels,
            row.pair_id,
        )
        for row in df.itertuples(index=False)
    ]
    return df


def compute_damage_ratio(
    counts_json: object,
    total_pixels_value: object,
    pair_id: str,
) -> float:
    if pd.isna(counts_json):
        raise XbdSplitError(f"Missing target_value_counts for pair '{pair_id}'.")

    try:
        raw_counts = json.loads(str(counts_json))
    except json.JSONDecodeError as exc:
        raise XbdSplitError(
            f"Invalid target_value_counts JSON for pair '{pair_id}': {exc}"
        ) from exc
    if not isinstance(raw_counts, dict):
        raise XbdSplitError(
            f"target_value_counts must be a JSON object for pair '{pair_id}'."
        )

    try:
        total_pixels = int(float(total_pixels_value))
    except (TypeError, ValueError) as exc:
        raise XbdSplitError(f"Invalid target_total_pixels for pair '{pair_id}'.") from exc
    if total_pixels <= 0:
        return 0.0

    damage_pixels = 0
    for raw_class, raw_count in raw_counts.items():
        target_class = parse_target_class(raw_class)
        if target_class not in DAMAGE_CLASSES:
            continue
        try:
            damage_pixels += int(raw_count)
        except (TypeError, ValueError) as exc:
            raise XbdSplitError(
                f"Invalid pixel count for class '{raw_class}' in pair '{pair_id}'."
            ) from exc

    return damage_pixels / total_pixels


def parse_target_class(raw_class: object) -> int | None:
    key = str(raw_class).strip()
    try:
        return int(key)
    except ValueError:
        pass

    try:
        parsed = json.loads(key)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and parsed:
        try:
            values = [float(value) for value in parsed]
        except (TypeError, ValueError):
            values = []
        if values and values[0].is_integer() and all(
            value == values[0] for value in values
        ):
            return int(values[0])

    try:
        value = float(key)
    except ValueError:
        return None
    return int(value) if value.is_integer() else None


def filter_index(
    df: pd.DataFrame,
    disasters: list[str] | None,
    min_nonzero_ratio: float,
    min_damage_ratio: float,
    max_pairs: int | None,
    seed: int,
) -> pd.DataFrame:
    selected = df.copy()

    if disasters:
        requested = {name.lower() for name in disasters}
        selected = selected[selected["disaster"].str.lower().isin(requested)]
        if selected.empty:
            available = ", ".join(sorted(df["disaster"].dropna().unique()))
            raise XbdSplitError(
                "No pairs remain after --disasters filter. "
                f"Available disasters: {available}"
            )

    selected = selected[selected["target_nonzero_ratio"] >= min_nonzero_ratio]
    selected = selected[selected["damage_ratio"] >= min_damage_ratio]
    if selected.empty:
        raise XbdSplitError("No pairs remain after ratio filters.")

    if max_pairs is not None and len(selected) > max_pairs:
        selected = selected.sample(n=max_pairs, random_state=seed)

    selected = selected.sort_values("pair_id").reset_index(drop=True)
    if len(selected) < 3:
        raise XbdSplitError("At least 3 selected pairs are required for train/val/test splits.")
    return selected


def stratify_labels_or_none(df: pd.DataFrame, test_size: float, split_name: str):
    disaster_counts = df["disaster"].value_counts()
    if len(disaster_counts) < 2:
        return None
    if disaster_counts.min() < 2:
        print(
            f"Warning: {split_name} split is not stratified because at least one "
            "disaster has fewer than 2 pairs."
        )
        return None

    n_samples = len(df)
    n_test = math.ceil(n_samples * test_size)
    n_train = n_samples - n_test
    n_classes = len(disaster_counts)
    if n_test < n_classes or n_train < n_classes:
        print(
            f"Warning: {split_name} split is not stratified because the requested "
            "split is too small for every disaster to appear in both sides."
        )
        return None

    return df["disaster"]


def split_train_val_test(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_expected_split_counts(len(df), val_size, test_size)

    test_stratify = stratify_labels_or_none(df, test_size, "test")
    try:
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=test_stratify,
        )
    except ValueError as exc:
        print(f"Warning: test split stratification failed ({exc}); falling back to shuffled split.")
        try:
            train_val, test = train_test_split(
                df,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=None,
            )
        except ValueError as fallback_exc:
            raise XbdSplitError(f"Could not create test split: {fallback_exc}") from fallback_exc

    val_fraction_of_remaining = val_size / (1.0 - test_size)
    val_stratify = stratify_labels_or_none(train_val, val_fraction_of_remaining, "validation")
    try:
        train, val = train_test_split(
            train_val,
            test_size=val_fraction_of_remaining,
            random_state=seed,
            shuffle=True,
            stratify=val_stratify,
        )
    except ValueError as exc:
        print(
            "Warning: validation split stratification failed "
            f"({exc}); falling back to shuffled split."
        )
        try:
            train, val = train_test_split(
                train_val,
                test_size=val_fraction_of_remaining,
                random_state=seed,
                shuffle=True,
                stratify=None,
            )
        except ValueError as fallback_exc:
            raise XbdSplitError(
                f"Could not create validation split: {fallback_exc}"
            ) from fallback_exc

    return (
        train.sort_values("pair_id").reset_index(drop=True),
        val.sort_values("pair_id").reset_index(drop=True),
        test.sort_values("pair_id").reset_index(drop=True),
    )


def validate_expected_split_counts(n_pairs: int, val_size: float, test_size: float) -> None:
    n_test = math.ceil(n_pairs * test_size)
    n_train_val = n_pairs - n_test
    val_fraction_of_remaining = val_size / (1.0 - test_size)
    n_val = math.ceil(n_train_val * val_fraction_of_remaining)
    n_train = n_train_val - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise XbdSplitError(
            "Selected pair count and split fractions would create an empty split: "
            f"train={n_train}, val={n_val}, test={n_test}."
        )


def with_split_column(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    output = df.copy()
    output.insert(0, "split", split_name)
    return output


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_df in splits.items():
        rows.append(summary_row(split_name, "ALL", split_df))
        for disaster, disaster_df in split_df.groupby("disaster"):
            rows.append(summary_row(split_name, disaster, disaster_df))
    return pd.DataFrame(rows)


def summary_row(split_name: str, disaster: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split_name,
        "disaster": disaster,
        "pairs": len(df),
        "avg_nonzero_ratio": df["target_nonzero_ratio"].mean() if len(df) else 0.0,
        "avg_damage_ratio": df["damage_ratio"].mean() if len(df) else 0.0,
    }


def validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if "raw" in {part.lower() for part in output_dir.parts}:
        raise XbdSplitError("Refusing to write split CSVs inside a raw data directory.")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise XbdSplitError(f"Could not create output directory '{output_dir}': {exc}") from exc
    if not output_dir.is_dir():
        raise XbdSplitError(f"Output path is not a directory: {output_dir}")
    return output_dir


def save_splits(
    output_dir: Path,
    splits: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
) -> None:
    paths = {
        "train": output_dir / "train_pairs.csv",
        "val": output_dir / "val_pairs.csv",
        "test": output_dir / "test_pairs.csv",
    }
    try:
        for split_name, path in paths.items():
            with_split_column(splits[split_name], split_name).to_csv(path, index=False)
        summary.to_csv(output_dir / "split_summary.csv", index=False)
    except OSError as exc:
        raise XbdSplitError(f"Could not write split CSVs: {exc}") from exc


def print_summary(selected: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    print("CrisisMap AI - xBD/xView2 Split Summary")
    print("=" * 44)
    print(f"Total selected pairs: {len(selected)}")
    print()

    print("Pairs per split")
    for split_name, split_df in splits.items():
        print(f"  {split_name}: {len(split_df)}")
    print()

    print("Disaster distribution per split")
    for split_name, split_df in splits.items():
        print(f"  {split_name}:")
        for disaster, count in split_df["disaster"].value_counts().sort_index().items():
            print(f"    {disaster}: {count}")
    print()

    print("Average nonzero ratio per split")
    for split_name, split_df in splits.items():
        print(f"  {split_name}: {split_df['target_nonzero_ratio'].mean():.6f}")
    print()

    print("Average damage ratio per split")
    for split_name, split_df in splits.items():
        print(f"  {split_name}: {split_df['damage_ratio'].mean():.6f}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        df = add_ratio_columns(load_index(args.index))
        selected = filter_index(
            df,
            args.disasters,
            args.min_nonzero_ratio,
            args.min_damage_ratio,
            args.max_pairs,
            args.seed,
        )
        train, val, test = split_train_val_test(
            selected,
            args.val_size,
            args.test_size,
            args.seed,
        )
        splits = {"train": train, "val": val, "test": test}
        output_dir = validate_output_dir(args.output_dir)
        summary = split_summary(splits)
        save_splits(output_dir, splits, summary)
        print_summary(selected, splits)
        print()
        print(f"Saved split CSVs to: {output_dir}")
    except XbdSplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

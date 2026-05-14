"""Create damage-balanced xBD split variants from the training index.

The old 4-disaster subset is used only to estimate target damage-ratio
proportions. New matched splits are sampled from all available disasters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DAMAGE_CLASSES = {2, 3, 4}
OLD_REFERENCE_DISASTERS = {
    "hurricane-harvey",
    "hurricane-michael",
    "palu-tsunami",
    "santa-rosa-wildfire",
}
DAMAGE_BINS = [
    ("damage_eq_0", 0.0, 0.0),
    ("damage_0_000_0_005", 0.0, 0.005),
    ("damage_0_005_0_020", 0.005, 0.02),
    ("damage_0_020_0_060", 0.02, 0.06),
    ("damage_0_060_0_120", 0.06, 0.12),
    ("damage_gt_0_120", 0.12, np.inf),
]
DAMAGE_BIN_LABELS = [label for label, _low, _high in DAMAGE_BINS]
REQUIRED_COLUMNS = {
    "pair_id",
    "disaster",
    "target_value_counts",
    "target_total_pixels",
    "target_nonzero_ratio",
}


class MatchedSplitError(Exception):
    """Raised when matched split creation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create damage-matched xBD train/val/test split variants."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/xbd_train_index.csv"),
        help="Path to xbd_train_index.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory under which split folders will be created.",
    )
    parser.add_argument(
        "--old-splits-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help=(
            "Optional old 4-disaster split folder. If train/val/test CSVs are "
            "present, their pair_ids define the old reference distribution."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_size < 1.0:
        raise MatchedSplitError("--val-size must be between 0 and 1.")
    if not 0.0 < args.test_size < 1.0:
        raise MatchedSplitError("--test-size must be between 0 and 1.")
    if args.val_size + args.test_size >= 1.0:
        raise MatchedSplitError("--val-size + --test-size must be less than 1.")


def load_index(index_path: Path) -> pd.DataFrame:
    index_path = index_path.expanduser().resolve()
    if not index_path.exists():
        raise MatchedSplitError(f"Index CSV does not exist: {index_path}")
    if not index_path.is_file():
        raise MatchedSplitError(f"Index path is not a file: {index_path}")

    try:
        df = pd.read_csv(index_path)
    except OSError as exc:
        raise MatchedSplitError(f"Could not read index CSV '{index_path}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise MatchedSplitError(f"Index CSV is empty: {index_path}") from exc
    except pd.errors.ParserError as exc:
        raise MatchedSplitError(f"Could not parse index CSV '{index_path}': {exc}") from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise MatchedSplitError(
            "Index CSV is missing required column(s): " + ", ".join(missing)
        )
    if df.empty:
        raise MatchedSplitError("Index CSV has no rows.")
    if df["pair_id"].duplicated().any():
        duplicates = sorted(df.loc[df["pair_id"].duplicated(), "pair_id"].unique())
        raise MatchedSplitError(
            f"Index CSV contains duplicate pair_id values: {duplicates[:5]}"
        )

    df = df.copy()
    df["nonzero_ratio"] = pd.to_numeric(df["target_nonzero_ratio"], errors="coerce")
    if df["nonzero_ratio"].isna().any():
        raise MatchedSplitError("target_nonzero_ratio contains non-numeric values.")

    df["damage_ratio"] = [
        compute_damage_ratio(
            counts_json=row.target_value_counts,
            total_pixels_value=row.target_total_pixels,
            pair_id=row.pair_id,
        )
        for row in df.itertuples(index=False)
    ]
    df["damage_bin"] = df["damage_ratio"].map(assign_damage_bin)
    return df


def compute_damage_ratio(
    counts_json: object,
    total_pixels_value: object,
    pair_id: str,
) -> float:
    if pd.isna(counts_json):
        raise MatchedSplitError(f"Missing target_value_counts for pair '{pair_id}'.")

    try:
        counts = json.loads(str(counts_json))
    except json.JSONDecodeError as exc:
        raise MatchedSplitError(
            f"Invalid target_value_counts JSON for pair '{pair_id}': {exc}"
        ) from exc
    if not isinstance(counts, dict):
        raise MatchedSplitError(
            f"target_value_counts must be a JSON object for pair '{pair_id}'."
        )

    try:
        total_pixels = int(float(total_pixels_value))
    except (TypeError, ValueError) as exc:
        raise MatchedSplitError(f"Invalid target_total_pixels for pair '{pair_id}'.") from exc
    if total_pixels <= 0:
        return 0.0

    damage_pixels = 0
    for raw_class, raw_count in counts.items():
        target_class = parse_target_class(raw_class)
        if target_class not in DAMAGE_CLASSES:
            continue
        try:
            damage_pixels += int(raw_count)
        except (TypeError, ValueError) as exc:
            raise MatchedSplitError(
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


def assign_damage_bin(damage_ratio: float) -> str:
    if damage_ratio == 0.0:
        return "damage_eq_0"
    for label, low, high in DAMAGE_BINS[1:]:
        if low < damage_ratio <= high:
            return label
    return "damage_gt_0_120"


def load_old_reference(df: pd.DataFrame, old_splits_dir: Path) -> pd.DataFrame:
    split_files = [
        old_splits_dir / "train_pairs.csv",
        old_splits_dir / "val_pairs.csv",
        old_splits_dir / "test_pairs.csv",
    ]
    if all(path.exists() and path.is_file() for path in split_files):
        frames = [pd.read_csv(path, usecols=["pair_id"]) for path in split_files]
        pair_ids = set(pd.concat(frames, ignore_index=True)["pair_id"].astype(str))
        reference = df[df["pair_id"].astype(str).isin(pair_ids)].copy()
        if reference.empty:
            raise MatchedSplitError(
                f"Old split folder has no pair_ids present in the index: {old_splits_dir}"
            )
        print(f"Old reference source: existing split CSVs in {old_splits_dir}")
        return reference

    reference = df[
        df["disaster"].str.lower().isin(OLD_REFERENCE_DISASTERS)
        & (df["nonzero_ratio"] >= 0.01)
    ].copy()
    if reference.empty:
        raise MatchedSplitError("Derived old 4-disaster reference set is empty.")
    print(
        "Old reference source: derived from the 4 old disasters with "
        "nonzero_ratio >= 0.01"
    )
    return reference


def damage_bin_proportions(df: pd.DataFrame) -> pd.Series:
    counts = df["damage_bin"].value_counts().reindex(DAMAGE_BIN_LABELS, fill_value=0)
    total = int(counts.sum())
    if total == 0:
        raise MatchedSplitError("Cannot compute damage-bin proportions for empty data.")
    return counts / total


def target_counts_from_proportions(
    proportions: pd.Series,
    target_size: int,
) -> dict[str, int]:
    raw_counts = proportions.reindex(DAMAGE_BIN_LABELS, fill_value=0.0) * target_size
    counts = np.floor(raw_counts).astype(int)
    remainder = target_size - int(counts.sum())
    fractions = (raw_counts - counts).sort_values(ascending=False)

    for label in fractions.index:
        if remainder <= 0:
            break
        counts[label] += 1
        remainder -= 1

    return {label: int(counts[label]) for label in DAMAGE_BIN_LABELS}


def sample_histogram_matched(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    candidates = df[df["nonzero_ratio"] >= 0.01].copy()
    if candidates.empty:
        raise MatchedSplitError("No candidates remain after nonzero_ratio >= 0.01.")

    target_size = min(target_size, len(candidates))
    proportions = damage_bin_proportions(reference)
    desired_counts = target_counts_from_proportions(proportions, target_size)

    selected_indices: list[int] = []
    for index, label in enumerate(DAMAGE_BIN_LABELS):
        bin_candidates = candidates[candidates["damage_bin"] == label]
        desired = min(desired_counts[label], len(bin_candidates))
        if desired <= 0:
            continue
        selected = bin_candidates.sample(n=desired, random_state=seed + index)
        selected_indices.extend(selected.index.tolist())

    selected_set = set(selected_indices)
    shortfall = target_size - len(selected_indices)
    if shortfall > 0:
        remaining = candidates[~candidates.index.isin(selected_set)]
        if not remaining.empty:
            fill_n = min(shortfall, len(remaining))
            fill = remaining.sample(n=fill_n, random_state=seed + 1000)
            selected_indices.extend(fill.index.tolist())

    selected = candidates.loc[selected_indices].sample(frac=1.0, random_state=seed)
    return selected.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise MatchedSplitError("At least 3 pairs are required to create splits.")

    temp_size = val_size + test_size
    stratify_first = stratify_labels(df, temp_size)
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_first,
    )

    relative_test_size = test_size / temp_size
    stratify_second = stratify_labels(temp_df, relative_test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_second,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def stratify_labels(df: pd.DataFrame, split_fraction: float) -> pd.Series | None:
    labels = df["damage_bin"]
    counts = labels.value_counts()
    split_count = int(round(len(df) * split_fraction))
    if labels.nunique() <= 1:
        return None
    if counts.min() < 2:
        return None
    if split_count < labels.nunique():
        return None
    return labels


def write_split_folder(
    selected: pd.DataFrame,
    output_dir: Path,
    val_size: float,
    test_size: float,
    seed: int,
) -> None:
    train_df, val_df, test_df = split_dataset(selected, val_size, test_size, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_split_csv(train_df, "train", output_dir / "train_pairs.csv")
    write_split_csv(val_df, "val", output_dir / "val_pairs.csv")
    write_split_csv(test_df, "test", output_dir / "test_pairs.csv")
    write_summary(
        {"all": selected, "train": train_df, "val": val_df, "test": test_df},
        output_dir / "split_summary.csv",
    )


def write_split_csv(df: pd.DataFrame, split_name: str, output_path: Path) -> None:
    output = df.copy()
    output.insert(0, "split", split_name)
    output.to_csv(output_path, index=False)


def write_summary(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    rows = [summary_row(name, frame) for name, frame in frames.items()]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def summary_row(split_name: str, df: pd.DataFrame) -> dict[str, object]:
    damage_bins = (
        df["damage_bin"].value_counts().reindex(DAMAGE_BIN_LABELS, fill_value=0).to_dict()
    )
    damage_bins = {str(key): int(value) for key, value in damage_bins.items()}
    disaster_counts = {
        str(key): int(value)
        for key, value in df["disaster"].value_counts().sort_index().to_dict().items()
    }
    zero_damage_count = int((df["damage_ratio"] == 0.0).sum())
    total = len(df)

    return {
        "split": split_name,
        "pair_count": total,
        "disaster_distribution": json.dumps(disaster_counts, sort_keys=True),
        "avg_nonzero_ratio": float(df["nonzero_ratio"].mean()) if total else 0.0,
        "min_damage_ratio": float(df["damage_ratio"].min()) if total else 0.0,
        "mean_damage_ratio": float(df["damage_ratio"].mean()) if total else 0.0,
        "median_damage_ratio": float(df["damage_ratio"].median()) if total else 0.0,
        "zero_damage_count": zero_damage_count,
        "zero_damage_pct": float(zero_damage_count / total * 100.0) if total else 0.0,
        "damage_bin_distribution": json.dumps(damage_bins, sort_keys=True),
    }


def print_split_summary(name: str, selected: pd.DataFrame, output_dir: Path) -> None:
    row = summary_row("all", selected)
    print()
    print(f"Created {name}: {output_dir}")
    print(f"  pairs: {row['pair_count']}")
    print(f"  avg nonzero ratio: {row['avg_nonzero_ratio']:.4f}")
    print(
        "  damage ratio min/mean/median: "
        f"{row['min_damage_ratio']:.4f} / "
        f"{row['mean_damage_ratio']:.4f} / "
        f"{row['median_damage_ratio']:.4f}"
    )
    print(f"  zero-damage images: {row['zero_damage_pct']:.1f}%")
    print(f"  damage bins: {row['damage_bin_distribution']}")


def create_all_splits(args: argparse.Namespace) -> None:
    df = load_index(args.index)
    old_reference = load_old_reference(df, args.old_splits_dir)
    output_root = args.output_root.expanduser().resolve()

    print()
    print("Reference damage-bin proportions:")
    print(damage_bin_proportions(old_reference).round(4).to_string())
    print()
    print("New matched splits sample from all available disasters.")

    split_specs = [
        (
            "splits_match_old_dmg001",
            df[(df["nonzero_ratio"] >= 0.01) & (df["damage_ratio"] >= 0.001)].copy(),
        ),
        (
            "splits_match_old_hist809",
            sample_histogram_matched(df, old_reference, target_size=809, seed=args.seed),
        ),
        (
            "splits_match_old_hist1000",
            sample_histogram_matched(df, old_reference, target_size=1000, seed=args.seed + 10),
        ),
    ]

    for name, selected in split_specs:
        if selected.empty:
            raise MatchedSplitError(f"No pairs selected for {name}.")
        output_dir = output_root / name
        write_split_folder(selected, output_dir, args.val_size, args.test_size, args.seed)
        print_split_summary(name, selected, output_dir)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        create_all_splits(args)
    except MatchedSplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

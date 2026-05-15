"""Create no-leak xBD splits with common validation and test sets.

The shared validation and test sets are exact copies of
data/processed/splits_full/val_pairs.csv and test_pairs.csv. Training pools are
rebuilt so no train pair_id appears in either common evaluation set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DAMAGE_CLASSES = {2, 3, 4}
OLD4_DISASTERS = {
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
REQUIRED_INDEX_COLUMNS = {
    "pair_id",
    "disaster",
    "target_value_counts",
    "target_total_pixels",
    "target_nonzero_ratio",
}


class NoLeakSplitError(Exception):
    """Raised when no-leak split creation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create no-leak splits with common full-data val/test sets."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/xbd_train_index.csv"),
        help="Path to xbd_train_index.csv.",
    )
    parser.add_argument(
        "--common-split-dir",
        type=Path,
        default=Path("data/processed/splits_full"),
        help="Directory containing common val_pairs.csv and test_pairs.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory under which no-leak split folders are created.",
    )
    parser.add_argument(
        "--historical-hist1000-dir",
        type=Path,
        default=Path("data/processed/splits_match_old_hist1000"),
        help=(
            "Existing matched hist1000 split folder used only to estimate "
            "historical damage-bin proportions when available."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_index(index_path: Path) -> pd.DataFrame:
    index_path = index_path.expanduser().resolve()
    if not index_path.exists():
        raise NoLeakSplitError(f"Index CSV does not exist: {index_path}")
    if not index_path.is_file():
        raise NoLeakSplitError(f"Index path is not a file: {index_path}")

    try:
        df = pd.read_csv(index_path)
    except OSError as exc:
        raise NoLeakSplitError(f"Could not read index CSV '{index_path}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise NoLeakSplitError(f"Index CSV is empty: {index_path}") from exc
    except pd.errors.ParserError as exc:
        raise NoLeakSplitError(f"Could not parse index CSV '{index_path}': {exc}") from exc

    missing = sorted(REQUIRED_INDEX_COLUMNS - set(df.columns))
    if missing:
        raise NoLeakSplitError(
            "Index CSV is missing required column(s): " + ", ".join(missing)
        )
    if df.empty:
        raise NoLeakSplitError("Index CSV has no rows.")
    if df["pair_id"].duplicated().any():
        duplicates = sorted(df.loc[df["pair_id"].duplicated(), "pair_id"].unique())
        raise NoLeakSplitError(
            f"Index CSV contains duplicate pair_id values: {duplicates[:5]}"
        )

    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df["nonzero_ratio"] = pd.to_numeric(df["target_nonzero_ratio"], errors="coerce")
    if df["nonzero_ratio"].isna().any():
        raise NoLeakSplitError("target_nonzero_ratio contains non-numeric values.")

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
        raise NoLeakSplitError(f"Missing target_value_counts for pair '{pair_id}'.")

    try:
        counts = json.loads(str(counts_json))
    except json.JSONDecodeError as exc:
        raise NoLeakSplitError(
            f"Invalid target_value_counts JSON for pair '{pair_id}': {exc}"
        ) from exc
    if not isinstance(counts, dict):
        raise NoLeakSplitError(
            f"target_value_counts must be a JSON object for pair '{pair_id}'."
        )

    try:
        total_pixels = int(float(total_pixels_value))
    except (TypeError, ValueError) as exc:
        raise NoLeakSplitError(f"Invalid target_total_pixels for pair '{pair_id}'.") from exc
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
            raise NoLeakSplitError(
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


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise NoLeakSplitError(f"Required split CSV does not exist: {path}")
    if not path.is_file():
        raise NoLeakSplitError(f"Split path is not a file: {path}")
    try:
        df = pd.read_csv(path)
    except OSError as exc:
        raise NoLeakSplitError(f"Could not read split CSV '{path}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise NoLeakSplitError(f"Split CSV is empty: {path}") from exc
    if "pair_id" not in df.columns:
        raise NoLeakSplitError(f"Split CSV is missing pair_id column: {path}")
    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    return df


def load_common_eval(common_split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_split_dir = common_split_dir.expanduser().resolve()
    val_df = load_split(common_split_dir / "val_pairs.csv")
    test_df = load_split(common_split_dir / "test_pairs.csv")
    return val_df, test_df


def historical_hist1000_reference(
    df: pd.DataFrame,
    historical_dir: Path,
) -> pd.DataFrame:
    split_files = [
        historical_dir / "train_pairs.csv",
        historical_dir / "val_pairs.csv",
        historical_dir / "test_pairs.csv",
    ]
    if all(path.exists() and path.is_file() for path in split_files):
        frames = [load_split(path)[["pair_id"]] for path in split_files]
        pair_ids = set(pd.concat(frames, ignore_index=True)["pair_id"])
        reference = df[df["pair_id"].isin(pair_ids)].copy()
        if not reference.empty:
            print(f"Histogram reference: {historical_dir}")
            return reference

    reference = df[
        df["disaster"].str.lower().isin(OLD4_DISASTERS)
        & (df["nonzero_ratio"] >= 0.01)
    ].copy()
    if reference.empty:
        raise NoLeakSplitError("Could not build fallback old4 histogram reference.")
    print("Histogram reference: fallback old4 nonzero_ratio >= 0.01 pool")
    return reference


def damage_bin_proportions(df: pd.DataFrame) -> pd.Series:
    counts = df["damage_bin"].value_counts().reindex(DAMAGE_BIN_LABELS, fill_value=0)
    total = int(counts.sum())
    if total <= 0:
        raise NoLeakSplitError("Cannot compute damage-bin proportions for empty data.")
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
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        raise NoLeakSplitError("No candidates available for histogram matching.")

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

    shortfall = target_size - len(selected_indices)
    if shortfall > 0:
        remaining = candidates[~candidates.index.isin(selected_indices)]
        if not remaining.empty:
            fill_n = min(shortfall, len(remaining))
            fill = remaining.sample(n=fill_n, random_state=seed + 1000)
            selected_indices.extend(fill.index.tolist())

    selected = candidates.loc[selected_indices].sample(frac=1.0, random_state=seed)
    return selected.reset_index(drop=True)


def write_split_folder(
    output_dir: Path,
    train_df: pd.DataFrame,
    common_val_path: Path,
    common_test_path: Path,
    common_val_df: pd.DataFrame,
    common_test_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = train_df.copy()
    if "split" in train_out.columns:
        train_out = train_out.drop(columns=["split"])
    train_out.insert(0, "split", "train")
    train_out.to_csv(output_dir / "train_pairs.csv", index=False)

    shutil.copyfile(common_val_path, output_dir / "val_pairs.csv")
    shutil.copyfile(common_test_path, output_dir / "test_pairs.csv")

    summary = build_summary(
        train_out,
        common_val_df,
        common_test_df,
        index_df,
    )
    summary.to_csv(output_dir / "split_summary.csv", index=False)

    print_verification(output_dir.name, summary)


def build_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    train_ids = set(train_df["pair_id"].astype(str))
    val_ids = set(val_df["pair_id"].astype(str))
    test_ids = set(test_df["pair_id"].astype(str))
    overlap_train_val = len(train_ids & val_ids)
    overlap_train_test = len(train_ids & test_ids)
    overlap_val_test = len(val_ids & test_ids)

    rows = []
    for split_name, frame in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        enriched = enrich_for_summary(frame, index_df)
        rows.append(
            summary_row(
                split_name,
                enriched,
                overlap_train_val,
                overlap_train_test,
                overlap_val_test,
            )
        )
    return pd.DataFrame(rows)


def enrich_for_summary(frame: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    pair_ids = frame["pair_id"].astype(str)
    enriched = index_df[index_df["pair_id"].isin(set(pair_ids))].copy()
    order = pd.DataFrame({"pair_id": pair_ids, "_order": range(len(pair_ids))})
    enriched = order.merge(enriched, on="pair_id", how="left").sort_values("_order")
    return enriched.drop(columns=["_order"])


def summary_row(
    split_name: str,
    df: pd.DataFrame,
    overlap_train_val: int,
    overlap_train_test: int,
    overlap_val_test: int,
) -> dict[str, object]:
    total = len(df)
    disaster_counts = {
        str(key): int(value)
        for key, value in df["disaster"].value_counts().sort_index().to_dict().items()
    }
    damage_bins = {
        str(key): int(value)
        for key, value in df["damage_bin"]
        .value_counts()
        .reindex(DAMAGE_BIN_LABELS, fill_value=0)
        .to_dict()
        .items()
    }
    return {
        "split": split_name,
        "pair_count": total,
        "train_overlap_common_val": overlap_train_val,
        "train_overlap_common_test": overlap_train_test,
        "val_overlap_common_test": overlap_val_test,
        "disaster_distribution": json.dumps(disaster_counts, sort_keys=True),
        "avg_nonzero_ratio": float(df["nonzero_ratio"].mean()) if total else 0.0,
        "avg_damage_ratio": float(df["damage_ratio"].mean()) if total else 0.0,
        "damage_bin_distribution": json.dumps(damage_bins, sort_keys=True),
    }


def print_verification(split_name: str, summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    print()
    print(f"Created {split_name}")
    print(f"  train overlap with common val: {int(row['train_overlap_common_val'])}")
    print(f"  train overlap with common test: {int(row['train_overlap_common_test'])}")
    print(f"  val overlap with common test: {int(row['val_overlap_common_test'])}")
    print("  pair counts:")
    for item in summary.itertuples(index=False):
        print(f"    {item.split}: {item.pair_count}")


def assert_no_leakage(summary: pd.DataFrame, output_dir: Path) -> None:
    row = summary.iloc[0]
    values = [
        int(row["train_overlap_common_val"]),
        int(row["train_overlap_common_test"]),
        int(row["val_overlap_common_test"]),
    ]
    if values != [0, 0, 0]:
        raise NoLeakSplitError(
            f"Leakage detected in {output_dir}: "
            f"train/val={values[0]}, train/test={values[1]}, val/test={values[2]}"
        )


def create_splits(args: argparse.Namespace) -> None:
    index_df = load_index(args.index)
    common_dir = args.common_split_dir.expanduser().resolve()
    common_val_path = common_dir / "val_pairs.csv"
    common_test_path = common_dir / "test_pairs.csv"
    common_val_df, common_test_df = load_common_eval(common_dir)

    common_eval_ids = set(common_val_df["pair_id"].astype(str)) | set(
        common_test_df["pair_id"].astype(str)
    )
    trainable = index_df[~index_df["pair_id"].isin(common_eval_ids)].copy()
    if trainable.empty:
        raise NoLeakSplitError("No trainable pairs remain after common eval exclusion.")

    old4_train = trainable[
        trainable["disaster"].str.lower().isin(OLD4_DISASTERS)
        & (trainable["nonzero_ratio"] >= 0.01)
    ].copy()

    reference = historical_hist1000_reference(index_df, args.historical_hist1000_dir)
    hist_candidates = trainable[trainable["nonzero_ratio"] >= 0.01].copy()
    hist1000_train = sample_histogram_matched(
        hist_candidates,
        reference,
        target_size=1000,
        seed=args.seed,
    )

    dmg001_train = trainable[
        (trainable["nonzero_ratio"] >= 0.01)
        & (trainable["damage_ratio"] >= 0.001)
    ].copy()

    specs = [
        ("splits_noleak_old4", old4_train),
        ("splits_noleak_match_hist1000", hist1000_train),
        ("splits_noleak_match_dmg001", dmg001_train),
    ]

    output_root = args.output_root.expanduser().resolve()
    for name, train_df in specs:
        if train_df.empty:
            raise NoLeakSplitError(f"No training pairs selected for {name}.")
        output_dir = output_root / name
        write_split_folder(
            output_dir,
            train_df,
            common_val_path,
            common_test_path,
            common_val_df,
            common_test_df,
            index_df,
        )
        summary = pd.read_csv(output_dir / "split_summary.csv")
        assert_no_leakage(summary, output_dir)


def main() -> int:
    args = parse_args()
    try:
        create_splits(args)
    except NoLeakSplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

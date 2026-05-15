"""Create advanced no-leak xBD training split variants.

All generated folders reuse the common validation and test CSVs from
data/processed/splits_full. Training rows always exclude every common
validation/test pair_id.
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


class AdvancedSplitError(Exception):
    """Raised when advanced no-leak split creation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create advanced no-leak xBD training split variants."
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
        help="Directory under which advanced split folders are created.",
    )
    parser.add_argument(
        "--historical-hist1000-dir",
        type=Path,
        default=Path("data/processed/splits_match_old_hist1000"),
        help="Existing hist1000 split used only to estimate damage-bin proportions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_index(index_path: Path) -> pd.DataFrame:
    index_path = index_path.expanduser().resolve()
    if not index_path.exists():
        raise AdvancedSplitError(f"Index CSV does not exist: {index_path}")
    if not index_path.is_file():
        raise AdvancedSplitError(f"Index path is not a file: {index_path}")

    try:
        df = pd.read_csv(index_path)
    except OSError as exc:
        raise AdvancedSplitError(f"Could not read index CSV '{index_path}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise AdvancedSplitError(f"Index CSV is empty: {index_path}") from exc
    except pd.errors.ParserError as exc:
        raise AdvancedSplitError(f"Could not parse index CSV '{index_path}': {exc}") from exc

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise AdvancedSplitError(
            "Index CSV is missing required column(s): " + ", ".join(missing)
        )
    if df.empty:
        raise AdvancedSplitError("Index CSV has no rows.")
    if df["pair_id"].duplicated().any():
        duplicates = sorted(df.loc[df["pair_id"].duplicated(), "pair_id"].unique())
        raise AdvancedSplitError(
            f"Index CSV contains duplicate pair_id values: {duplicates[:5]}"
        )

    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df["nonzero_ratio"] = pd.to_numeric(df["target_nonzero_ratio"], errors="coerce")
    if df["nonzero_ratio"].isna().any():
        raise AdvancedSplitError("target_nonzero_ratio contains non-numeric values.")

    df["damage_ratio"] = [
        compute_damage_ratio(
            row.target_value_counts,
            row.target_total_pixels,
            row.pair_id,
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
        raise AdvancedSplitError(f"Missing target_value_counts for pair '{pair_id}'.")
    try:
        counts = json.loads(str(counts_json))
    except json.JSONDecodeError as exc:
        raise AdvancedSplitError(
            f"Invalid target_value_counts JSON for pair '{pair_id}': {exc}"
        ) from exc
    if not isinstance(counts, dict):
        raise AdvancedSplitError(
            f"target_value_counts must be a JSON object for pair '{pair_id}'."
        )
    try:
        total_pixels = int(float(total_pixels_value))
    except (TypeError, ValueError) as exc:
        raise AdvancedSplitError(f"Invalid target_total_pixels for pair '{pair_id}'.") from exc
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
            raise AdvancedSplitError(
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
        raise AdvancedSplitError(f"Required split CSV does not exist: {path}")
    try:
        df = pd.read_csv(path)
    except OSError as exc:
        raise AdvancedSplitError(f"Could not read split CSV '{path}': {exc}") from exc
    if "pair_id" not in df.columns:
        raise AdvancedSplitError(f"Split CSV is missing pair_id column: {path}")
    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    return df


def historical_reference(df: pd.DataFrame, historical_dir: Path) -> pd.DataFrame:
    files = [
        historical_dir / "train_pairs.csv",
        historical_dir / "val_pairs.csv",
        historical_dir / "test_pairs.csv",
    ]
    if all(path.exists() and path.is_file() for path in files):
        ids = set(
            pd.concat([load_split(path)[["pair_id"]] for path in files], ignore_index=True)[
                "pair_id"
            ]
        )
        reference = df[df["pair_id"].isin(ids)].copy()
        if not reference.empty:
            print(f"Histogram reference: {historical_dir}")
            return reference

    reference = df[df["nonzero_ratio"] >= 0.01].copy()
    if reference.empty:
        raise AdvancedSplitError("Could not build histogram reference.")
    print("Histogram reference: fallback nonzero_ratio >= 0.01 pool")
    return reference


def damage_bin_proportions(df: pd.DataFrame) -> pd.Series:
    counts = df["damage_bin"].value_counts().reindex(DAMAGE_BIN_LABELS, fill_value=0)
    total = int(counts.sum())
    if total <= 0:
        raise AdvancedSplitError("Cannot compute damage-bin proportions for empty data.")
    return counts / total


def target_counts(proportions: pd.Series, target_size: int) -> dict[str, int]:
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


def sample_histogram(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    target_size = min(target_size, len(candidates))
    if target_size <= 0:
        return candidates.copy()
    proportions = damage_bin_proportions(reference)
    desired_counts = target_counts(proportions, target_size)

    selected_indices: list[int] = []
    for index, label in enumerate(DAMAGE_BIN_LABELS):
        bin_candidates = candidates[candidates["damage_bin"] == label]
        count = min(desired_counts[label], len(bin_candidates))
        if count > 0:
            selected = bin_candidates.sample(n=count, random_state=seed + index)
            selected_indices.extend(selected.index.tolist())

    shortfall = target_size - len(selected_indices)
    if shortfall > 0:
        remaining = candidates[~candidates.index.isin(selected_indices)]
        if not remaining.empty:
            fill = remaining.sample(
                n=min(shortfall, len(remaining)),
                random_state=seed + 1000,
            )
            selected_indices.extend(fill.index.tolist())
    return candidates.loc[selected_indices].sample(frac=1.0, random_state=seed)


def sample_balanced_damage(trainable: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = trainable["damage_bin"].value_counts().reindex(DAMAGE_BIN_LABELS, fill_value=0)
    anchor_counts = [
        int(counts[label])
        for label in DAMAGE_BIN_LABELS[2:]
        if int(counts[label]) > 0
    ]
    cap = max(100, int(np.median(anchor_counts))) if anchor_counts else 100
    low_damage_cap = cap * 2

    pieces = []
    for index, label in enumerate(DAMAGE_BIN_LABELS):
        bin_df = trainable[trainable["damage_bin"] == label]
        if bin_df.empty:
            continue
        limit = low_damage_cap if label in DAMAGE_BIN_LABELS[:2] else len(bin_df)
        count = min(limit, len(bin_df))
        pieces.append(bin_df.sample(n=count, random_state=seed + index))
    return pd.concat(pieces, ignore_index=False).sample(frac=1.0, random_state=seed)


def sample_mix(
    damaged: pd.DataFrame,
    low_damage: pd.DataFrame,
    damaged_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if damaged.empty:
        return low_damage.sample(frac=1.0, random_state=seed)
    if low_damage.empty:
        return damaged.sample(frac=1.0, random_state=seed)

    total_by_damaged = int(len(damaged) / damaged_fraction)
    total_by_low = int(len(low_damage) / (1.0 - damaged_fraction))
    total = max(1, min(total_by_damaged, total_by_low))
    damage_count = min(len(damaged), int(round(total * damaged_fraction)))
    low_count = min(len(low_damage), total - damage_count)

    selected = pd.concat(
        [
            damaged.sample(n=damage_count, random_state=seed),
            low_damage.sample(n=low_count, random_state=seed + 1),
        ],
        ignore_index=False,
    )
    return selected.sample(frac=1.0, random_state=seed)


def sample_per_disaster(trainable: pd.DataFrame, max_per_disaster: int, seed: int) -> pd.DataFrame:
    pieces = []
    for index, (_disaster, group) in enumerate(trainable.groupby("disaster", sort=True)):
        count = min(max_per_disaster, len(group))
        pieces.append(group.sample(n=count, random_state=seed + index))
    return pd.concat(pieces, ignore_index=False).sample(frac=1.0, random_state=seed)


def build_splits(
    index_df: pd.DataFrame,
    common_val: pd.DataFrame,
    common_test: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
) -> list[tuple[str, pd.DataFrame]]:
    common_ids = set(common_val["pair_id"]) | set(common_test["pair_id"])
    trainable = index_df[~index_df["pair_id"].isin(common_ids)].copy()
    nonzero_trainable = trainable[trainable["nonzero_ratio"] >= 0.01].copy()
    damaged = trainable[
        (trainable["nonzero_ratio"] >= 0.01) & (trainable["damage_ratio"] > 0.001)
    ].copy()
    low_damage = trainable[
        (trainable["nonzero_ratio"] >= 0.01) & (trainable["damage_ratio"] <= 0.001)
    ].copy()

    return [
        ("splits_noleak_full_train", trainable),
        ("splits_noleak_full_balanced_damage", sample_balanced_damage(trainable, seed)),
        (
            "splits_noleak_match_hist1500",
            sample_histogram(nonzero_trainable, reference, 1500, seed),
        ),
        (
            "splits_noleak_match_hist_all",
            sample_histogram(nonzero_trainable, reference, len(nonzero_trainable), seed + 10),
        ),
        ("splits_noleak_dmg001_v2", sample_mix(damaged, low_damage, 0.60, seed)),
        ("splits_noleak_damage_heavy", sample_mix(damaged, low_damage, 0.80, seed + 20)),
        ("splits_noleak_disaster_stratified_150", sample_per_disaster(trainable, 150, seed)),
        ("splits_noleak_disaster_stratified_200", sample_per_disaster(trainable, 200, seed)),
        (
            "splits_noleak_building_rich_002",
            trainable[trainable["nonzero_ratio"] >= 0.02].copy(),
        ),
        (
            "splits_noleak_building_rich_003",
            trainable[trainable["nonzero_ratio"] >= 0.03].copy(),
        ),
    ]


def write_split(
    name: str,
    train_df: pd.DataFrame,
    output_root: Path,
    common_val_path: Path,
    common_test_path: Path,
    common_val: pd.DataFrame,
    common_test: pd.DataFrame,
) -> dict[str, object]:
    if train_df.empty:
        raise AdvancedSplitError(f"No training pairs selected for {name}.")

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = train_df.copy()
    if "split" in train_out.columns:
        train_out = train_out.drop(columns=["split"])
    train_out.insert(0, "split", "train")
    train_out.to_csv(output_dir / "train_pairs.csv", index=False)
    shutil.copyfile(common_val_path, output_dir / "val_pairs.csv")
    shutil.copyfile(common_test_path, output_dir / "test_pairs.csv")

    summary = summary_row(name, train_out, common_val, common_test)
    pd.DataFrame([summary]).to_csv(output_dir / "split_summary.csv", index=False)
    assert_no_leakage(summary, output_dir)
    return summary


def summary_row(
    split_name: str,
    train_df: pd.DataFrame,
    common_val: pd.DataFrame,
    common_test: pd.DataFrame,
) -> dict[str, object]:
    train_ids = set(train_df["pair_id"].astype(str))
    val_ids = set(common_val["pair_id"].astype(str))
    test_ids = set(common_test["pair_id"].astype(str))
    disaster_counts = {
        str(key): int(value)
        for key, value in train_df["disaster"].value_counts().sort_index().to_dict().items()
    }
    damage_bins = {
        str(key): int(value)
        for key, value in train_df["damage_bin"]
        .value_counts()
        .reindex(DAMAGE_BIN_LABELS, fill_value=0)
        .to_dict()
        .items()
    }
    return {
        "split_name": split_name,
        "train_count": int(len(train_df)),
        "val_count": int(len(common_val)),
        "test_count": int(len(common_test)),
        "train_overlap_common_val": int(len(train_ids & val_ids)),
        "train_overlap_common_test": int(len(train_ids & test_ids)),
        "val_overlap_common_test": int(len(val_ids & test_ids)),
        "disaster_distribution": json.dumps(disaster_counts, sort_keys=True),
        "avg_nonzero_ratio": float(train_df["nonzero_ratio"].mean()),
        "avg_damage_ratio": float(train_df["damage_ratio"].mean()),
        "min_damage_ratio": float(train_df["damage_ratio"].min()),
        "median_damage_ratio": float(train_df["damage_ratio"].median()),
        "max_damage_ratio": float(train_df["damage_ratio"].max()),
        "damage_bin_distribution": json.dumps(damage_bins, sort_keys=True),
    }


def assert_no_leakage(summary: dict[str, object], output_dir: Path) -> None:
    values = [
        int(summary["train_overlap_common_val"]),
        int(summary["train_overlap_common_test"]),
        int(summary["val_overlap_common_test"]),
    ]
    if values != [0, 0, 0]:
        raise AdvancedSplitError(
            f"Leakage detected in {output_dir}: "
            f"train/val={values[0]}, train/test={values[1]}, val/test={values[2]}"
        )


def print_verification_table(rows: list[dict[str, object]]) -> None:
    print()
    print("Verification table")
    print(
        f"{'split':42s} {'train':>7s} {'val':>5s} {'test':>5s} "
        f"{'tr-val':>7s} {'tr-test':>8s} {'val-test':>9s} {'avg_dmg':>9s}"
    )
    for row in rows:
        print(
            f"{str(row['split_name']):42s} "
            f"{int(row['train_count']):7d} "
            f"{int(row['val_count']):5d} "
            f"{int(row['test_count']):5d} "
            f"{int(row['train_overlap_common_val']):7d} "
            f"{int(row['train_overlap_common_test']):8d} "
            f"{int(row['val_overlap_common_test']):9d} "
            f"{float(row['avg_damage_ratio']):9.4f}"
        )


def create_splits(args: argparse.Namespace) -> None:
    index_df = load_index(args.index)
    common_dir = args.common_split_dir.expanduser().resolve()
    common_val_path = common_dir / "val_pairs.csv"
    common_test_path = common_dir / "test_pairs.csv"
    common_val = load_split(common_val_path)
    common_test = load_split(common_test_path)
    reference = historical_reference(index_df, args.historical_hist1000_dir)

    output_root = args.output_root.expanduser().resolve()
    rows = []
    for name, train_df in build_splits(
        index_df,
        common_val,
        common_test,
        reference,
        args.seed,
    ):
        rows.append(
            write_split(
                name,
                train_df,
                output_root,
                common_val_path,
                common_test_path,
                common_val,
                common_test,
            )
        )
    print_verification_table(rows)


def main() -> int:
    args = parse_args()
    try:
        create_splits(args)
    except AdvancedSplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

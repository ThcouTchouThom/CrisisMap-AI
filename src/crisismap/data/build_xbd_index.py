"""Build a read-only tabular index for an extracted xBD/xView2 dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TARGET_SUFFIXES = IMAGE_SUFFIXES | {".npy", ".npz"}
XBD_STEM_PATTERN = re.compile(
    r"^(?P<pair_id>.+)_(?P<phase>pre|post)_disaster$",
    flags=re.IGNORECASE,
)


class XbdIndexError(Exception):
    """Raised when the xBD index cannot be built safely."""


@dataclass(frozen=True)
class XbdAsset:
    path: Path
    stem: str
    pair_id: str
    phase: str
    disaster: str


@dataclass(frozen=True)
class XbdPair:
    pair_id: str
    disaster: str
    pre_image: Path
    post_image: Path
    pre_label: Path
    post_label: Path
    target: Path


@dataclass(frozen=True)
class TargetStats:
    unique_values: list[str]
    value_counts: dict[str, int]
    total_pixels: int
    nonzero_pixels: int
    nonzero_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only CSV-style index for an extracted xBD/xView2 "
            "training folder."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional maximum number of pairs to index for quick debugging.",
    )
    return parser.parse_args()


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_named_dirs(root: Path, dirname: str) -> list[Path]:
    """Find folders named dirname under root without descending into matches."""

    matches: list[Path] = []
    stack = [root]
    target_name = dirname.lower()

    while stack:
        current = stack.pop()
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError as exc:
            raise XbdIndexError(f"Could not read directory '{current}': {exc}") from exc

        for child in children:
            if child.name.lower() == target_name:
                matches.append(child)
            else:
                stack.append(child)

    return sorted(matches)


def collect_files(folders: Iterable[Path], suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for folder in folders:
        try:
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in suffixes:
                    files.append(path)
        except OSError as exc:
            raise XbdIndexError(f"Could not scan files in '{folder}': {exc}") from exc
    return sorted(files)


def parse_xbd_asset(path: Path) -> XbdAsset | None:
    match = XBD_STEM_PATTERN.match(path.stem)
    if not match:
        return None

    pair_id = match.group("pair_id")
    return XbdAsset(
        path=path,
        stem=path.stem,
        pair_id=pair_id,
        phase=match.group("phase").lower(),
        disaster=extract_disaster_name(pair_id),
    )


def extract_disaster_name(pair_id: str) -> str:
    if "_" not in pair_id:
        return pair_id

    disaster, tile_id = pair_id.rsplit("_", 1)
    return disaster if tile_id else pair_id


def index_phase_assets(
    paths: Iterable[Path],
    kind: str,
) -> dict[tuple[str, str], XbdAsset]:
    grouped: dict[tuple[str, str], list[XbdAsset]] = defaultdict(list)
    ignored = 0

    for path in paths:
        asset = parse_xbd_asset(path)
        if asset is None:
            ignored += 1
            continue
        grouped[(asset.pair_id.lower(), asset.phase)].append(asset)

    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    if duplicates:
        pair_id, phase = sorted(duplicates)[0]
        examples = ", ".join(
            str(asset.path) for asset in duplicates[(pair_id, phase)][:3]
        )
        raise XbdIndexError(
            f"Duplicate {kind} files found for pair '{pair_id}' phase '{phase}': "
            f"{examples}"
        )

    if ignored:
        print(f"Warning: ignored {ignored} {kind} file(s) with non-standard xBD names.")

    return {key: values[0] for key, values in grouped.items()}


def target_stem_candidates(pair_id: str) -> list[str]:
    pair = pair_id.lower()
    post = f"{pair}_post_disaster"
    pre = f"{pair}_pre_disaster"
    return [
        f"{pair}_target",
        f"{pair}_targets",
        f"{pair}_mask",
        f"{pair}_damage",
        f"{post}_target",
        f"{post}_targets",
        f"{post}_mask",
        f"{post}_damage",
        post,
        f"{pre}_target",
        f"{pre}_targets",
        f"{pre}_mask",
        f"{pre}_damage",
        pre,
        pair,
    ]


def build_target_lookup(
    target_files: Iterable[Path],
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_stem: dict[str, Path] = {}
    by_pair_prefix: dict[str, list[Path]] = defaultdict(list)

    for target in sorted(target_files):
        stem = target.stem.lower()
        by_stem.setdefault(stem, target)
        pair_prefix = extract_pair_prefix_from_target_stem(stem)
        if pair_prefix:
            by_pair_prefix[pair_prefix].append(target)

    return by_stem, dict(by_pair_prefix)


def extract_pair_prefix_from_target_stem(stem: str) -> str | None:
    known_suffixes = (
        "_post_disaster_target",
        "_post_disaster_targets",
        "_post_disaster_mask",
        "_post_disaster_damage",
        "_post_disaster",
        "_pre_disaster_target",
        "_pre_disaster_targets",
        "_pre_disaster_mask",
        "_pre_disaster_damage",
        "_pre_disaster",
        "_target",
        "_targets",
        "_mask",
        "_damage",
    )
    for suffix in known_suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def find_target_mask(
    pair_id: str,
    targets_by_stem: dict[str, Path],
    targets_by_prefix: dict[str, list[Path]],
) -> Path | None:
    for candidate in target_stem_candidates(pair_id):
        target = targets_by_stem.get(candidate)
        if target is not None:
            return target

    prefix_matches = sorted(targets_by_prefix.get(pair_id.lower(), []))
    return prefix_matches[0] if prefix_matches else None


def discover_pairs(root: Path, max_pairs: int | None) -> list[XbdPair]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise XbdIndexError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise XbdIndexError(f"Root path is not a directory: {root}")
    if max_pairs is not None and max_pairs <= 0:
        raise XbdIndexError("--max-pairs must be a positive integer.")

    image_dirs = find_named_dirs(root, "images")
    label_dirs = find_named_dirs(root, "labels")
    target_dirs = find_named_dirs(root, "targets")

    if not image_dirs:
        raise XbdIndexError("No folder named 'images' was found under --root.")
    if not label_dirs:
        raise XbdIndexError("No folder named 'labels' was found under --root.")
    if not target_dirs:
        raise XbdIndexError("No folder named 'targets' was found under --root.")

    image_files = collect_files(image_dirs, IMAGE_SUFFIXES)
    label_files = collect_files(label_dirs, {".json"})
    target_files = collect_files(target_dirs, TARGET_SUFFIXES)

    if not image_files:
        raise XbdIndexError("Found 'images' folder(s), but no supported images.")
    if not label_files:
        raise XbdIndexError("Found 'labels' folder(s), but no JSON labels.")
    if not target_files:
        raise XbdIndexError("Found 'targets' folder(s), but no target masks.")

    images = index_phase_assets(image_files, "image")
    labels = index_phase_assets(label_files, "label")
    targets_by_stem, targets_by_prefix = build_target_lookup(target_files)

    pair_ids = sorted(
        pair_id
        for pair_id in {key[0] for key in images}
        if (pair_id, "pre") in images and (pair_id, "post") in images
    )
    if max_pairs is not None:
        pair_ids = pair_ids[:max_pairs]
    if not pair_ids:
        raise XbdIndexError("No valid pre/post image pairs were found.")

    pairs: list[XbdPair] = []
    missing: list[str] = []
    for pair_id in pair_ids:
        pre_label = labels.get((pair_id, "pre"))
        post_label = labels.get((pair_id, "post"))
        target = find_target_mask(pair_id, targets_by_stem, targets_by_prefix)

        if pre_label is None:
            missing.append(f"{pair_id}: pre label")
            continue
        if post_label is None:
            missing.append(f"{pair_id}: post label")
            continue
        if target is None:
            missing.append(f"{pair_id}: target mask")
            continue

        pre_image = images[(pair_id, "pre")]
        post_image = images[(pair_id, "post")]
        pairs.append(
            XbdPair(
                pair_id=pair_id,
                disaster=pre_image.disaster,
                pre_image=pre_image.path,
                post_image=post_image.path,
                pre_label=pre_label.path,
                post_label=post_label.path,
                target=target,
            )
        )

    if missing:
        examples = "; ".join(missing[:10])
        suffix = f"; ... {len(missing) - 10} more" if len(missing) > 10 else ""
        raise XbdIndexError(
            f"Missing expected files for {len(missing)} pair(s): {examples}{suffix}"
        )
    if not pairs:
        raise XbdIndexError("No complete pairs found after checking labels and targets.")

    return pairs


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise XbdIndexError(f"Could not parse JSON label '{path}': {exc}") from exc
    except OSError as exc:
        raise XbdIndexError(f"Could not read JSON label '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise XbdIndexError(f"JSON label is not an object: {path}")
    return data


def count_building_features(label: dict) -> int | None:
    features = label.get("features")
    if isinstance(features, dict):
        for key in ("xy", "lng_lat"):
            count = count_buildings_in_feature_list(features.get(key))
            if count is not None:
                return count
    if isinstance(features, list):
        return count_buildings_in_feature_list(features)
    return None


def count_buildings_in_feature_list(items: object) -> int | None:
    if not isinstance(items, list):
        return None

    building_count = 0
    saw_feature_type = False
    for item in items:
        if not isinstance(item, dict):
            continue
        properties = item.get("properties")
        if not isinstance(properties, dict):
            continue
        feature_type = properties.get("feature_type")
        if feature_type is not None:
            saw_feature_type = True
        if str(feature_type).lower() == "building":
            building_count += 1

    if saw_feature_type:
        return building_count
    return len(items)


def load_target_array(path: Path) -> np.ndarray:
    try:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path))
        if suffix == ".npz":
            with np.load(path) as data:
                if not data.files:
                    raise XbdIndexError(f"Target mask archive is empty: {path}")
                return np.asarray(data[data.files[0]])
        with Image.open(path) as image:
            return np.asarray(image)
    except OSError as exc:
        raise XbdIndexError(f"Could not load target mask '{path}': {exc}") from exc


def target_stats(mask: np.ndarray) -> TargetStats:
    mask = np.asarray(mask)
    if mask.size == 0:
        raise XbdIndexError("Target mask is empty.")

    normalized = normalize_mask_for_counts(mask)
    if normalized.ndim == 2:
        values, counts = np.unique(normalized, return_counts=True)
        value_counts = {
            format_value(value): int(count) for value, count in zip(values, counts)
        }
        total_pixels = int(normalized.size)
        nonzero_pixels = int(np.count_nonzero(normalized))
    elif normalized.ndim == 3:
        flat = normalized.reshape(-1, normalized.shape[-1])
        values, counts = np.unique(flat, axis=0, return_counts=True)
        value_counts = {
            format_value(tuple(value.tolist())): int(count)
            for value, count in zip(values, counts)
        }
        total_pixels = int(flat.shape[0])
        nonzero_pixels = int(np.count_nonzero(np.any(flat != 0, axis=1)))
    else:
        raise XbdIndexError(f"Unsupported target mask shape: {mask.shape}")

    unique_values = list(value_counts.keys())
    nonzero_ratio = nonzero_pixels / total_pixels if total_pixels else 0.0
    return TargetStats(
        unique_values=unique_values,
        value_counts=value_counts,
        total_pixels=total_pixels,
        nonzero_pixels=nonzero_pixels,
        nonzero_ratio=nonzero_ratio,
    )


def normalize_mask_for_counts(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[-1] in (1, 3, 4):
        return mask[:, :, 0] if mask.shape[-1] == 1 else mask
    if mask.ndim == 3 and mask.shape[0] in (1, 3, 4):
        moved = np.moveaxis(mask, 0, -1)
        return moved[:, :, 0] if moved.shape[-1] == 1 else moved
    return mask


def format_value(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        return json.dumps([format_scalar(part) for part in value])
    return str(value)


def format_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(f"{value:g}")
    return value


def build_rows(pairs: Iterable[XbdPair], root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        pre_label = read_json(pair.pre_label)
        post_label = read_json(pair.post_label)
        stats = target_stats(load_target_array(pair.target))

        rows.append(
            {
                "pair_id": pair.pair_id,
                "disaster": pair.disaster,
                "pre_image": relative_path(pair.pre_image, root),
                "post_image": relative_path(pair.post_image, root),
                "pre_label": relative_path(pair.pre_label, root),
                "post_label": relative_path(pair.post_label, root),
                "target": relative_path(pair.target, root),
                "pre_building_count": count_building_features(pre_label),
                "post_building_count": count_building_features(post_label),
                "target_unique_values": json.dumps(stats.unique_values),
                "target_value_counts": json.dumps(stats.value_counts, sort_keys=True),
                "target_total_pixels": stats.total_pixels,
                "target_nonzero_pixels": stats.nonzero_pixels,
                "target_nonzero_ratio": stats.nonzero_ratio,
            }
        )
    return rows


def print_summary(df: pd.DataFrame) -> None:
    print("CrisisMap AI - xBD/xView2 Dataset Index")
    print("=" * 45)
    print(f"Total pairs: {len(df)}")
    print()

    print("Disaster counts")
    disaster_counts = Counter(df["disaster"])
    for disaster, count in sorted(disaster_counts.items()):
        print(f"  {disaster}: {count}")
    print()

    print("Target values observed globally")
    global_counts: Counter[str] = Counter()
    for counts_json in df["target_value_counts"]:
        global_counts.update(json.loads(counts_json))
    for value, count in sorted(global_counts.items(), key=lambda item: item[0]):
        print(f"  {value}: {count}")
    print()

    average_ratio = float(df["target_nonzero_ratio"].mean()) if len(df) else 0.0
    print(f"Average nonzero target ratio: {average_ratio:.6f}")
    print()

    print("Top 10 pairs by nonzero target ratio")
    top_pairs = df.sort_values("target_nonzero_ratio", ascending=False).head(10)
    for row in top_pairs.itertuples(index=False):
        print(f"  {row.pair_id}: {row.target_nonzero_ratio:.6f}")


def write_csv(df: pd.DataFrame, output: Path) -> None:
    output = output.expanduser().resolve()
    if not output.parent.exists():
        raise XbdIndexError(f"Output parent folder does not exist: {output.parent}")
    df.to_csv(output, index=False)
    print()
    print(f"Saved CSV: {output}")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    try:
        pairs = discover_pairs(root, args.max_pairs)
        rows = build_rows(pairs, root)
        df = pd.DataFrame(rows)
        print_summary(df)
        if args.output is not None:
            write_csv(df, args.output)
    except XbdIndexError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Visualize one xBD/xView2 pre/post image pair with labels and target mask."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TARGET_SUFFIXES = IMAGE_SUFFIXES | {".npy", ".npz"}
XBD_STEM_PATTERN = re.compile(
    r"^(?P<pair_id>.+)_(?P<phase>pre|post)_disaster$",
    flags=re.IGNORECASE,
)


class VisualizationError(Exception):
    """Raised when a requested xBD sample cannot be loaded cleanly."""


@dataclass(frozen=True)
class XbdAsset:
    path: Path
    stem: str
    pair_id: str
    phase: str


@dataclass(frozen=True)
class XbdPair:
    pair_id: str
    pre_image: Path
    post_image: Path
    target_mask: Path
    pre_label: Path | None
    post_label: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one xBD/xView2 training sample. The dataset is read-only; "
            "the script only writes a file when --output is provided."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--pair-id",
        type=str,
        default=None,
        help="Optional xBD pair id, for example socal-fire_00001390.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the generated figure should be saved.",
    )
    return parser.parse_args()


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
            raise VisualizationError(f"Could not read directory '{current}': {exc}") from exc

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
            raise VisualizationError(f"Could not scan files in '{folder}': {exc}") from exc
    return sorted(files)


def parse_xbd_asset(path: Path) -> XbdAsset | None:
    match = XBD_STEM_PATTERN.match(path.stem)
    if not match:
        return None

    return XbdAsset(
        path=path,
        stem=path.stem,
        pair_id=match.group("pair_id"),
        phase=match.group("phase").lower(),
    )


def indexed_assets(paths: Iterable[Path]) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in paths:
        asset = parse_xbd_asset(path)
        if asset is None:
            continue
        index.setdefault((asset.pair_id.lower(), asset.phase), path)
    return index


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


def find_target_mask(pair_id: str, target_files: Iterable[Path]) -> Path | None:
    targets_by_stem: dict[str, list[Path]] = {}
    for target in target_files:
        targets_by_stem.setdefault(target.stem.lower(), []).append(target)

    for candidate in target_stem_candidates(pair_id):
        matches = sorted(targets_by_stem.get(candidate, []))
        if matches:
            return matches[0]

    pair_prefix = pair_id.lower()
    fallback_matches = sorted(
        target
        for target in target_files
        if target.stem.lower().startswith(pair_prefix)
    )
    return fallback_matches[0] if fallback_matches else None


def discover_pair(root: Path, pair_id: str | None) -> XbdPair:
    root = root.expanduser().resolve()
    if not root.exists():
        raise VisualizationError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise VisualizationError(f"Root path is not a directory: {root}")

    image_dirs = find_named_dirs(root, "images")
    label_dirs = find_named_dirs(root, "labels")
    target_dirs = find_named_dirs(root, "targets")

    if not image_dirs:
        raise VisualizationError("No folder named 'images' was found under --root.")
    if not label_dirs:
        raise VisualizationError("No folder named 'labels' was found under --root.")
    if not target_dirs:
        raise VisualizationError("No folder named 'targets' was found under --root.")

    image_files = collect_files(image_dirs, IMAGE_SUFFIXES)
    label_files = collect_files(label_dirs, {".json"})
    target_files = collect_files(target_dirs, TARGET_SUFFIXES)

    if not image_files:
        raise VisualizationError("Found 'images' folder(s), but no supported images.")
    if not label_files:
        raise VisualizationError("Found 'labels' folder(s), but no JSON labels.")
    if not target_files:
        raise VisualizationError("Found 'targets' folder(s), but no target masks.")

    images = indexed_assets(image_files)
    labels = indexed_assets(label_files)
    available_pair_ids = sorted(
        pair
        for pair in {key[0] for key in images}
        if (pair, "pre") in images and (pair, "post") in images
    )

    if not available_pair_ids:
        raise VisualizationError("No valid pre/post image pairs were found.")

    selected_pair = pair_id.strip().lower() if pair_id else None
    if selected_pair == "":
        raise VisualizationError("--pair-id was provided but is empty.")

    if selected_pair is None:
        for candidate_pair in available_pair_ids:
            if not has_complete_sample(candidate_pair, labels, target_files):
                continue
            selected_pair = candidate_pair
            break

    if selected_pair is None:
        raise VisualizationError(
            "No complete sample found with pre image, post image, both JSON labels, "
            "and a target mask."
        )

    if selected_pair not in available_pair_ids:
        raise VisualizationError(
            f"Pair id '{pair_id}' was not found with both pre and post images."
        )

    target_mask = find_target_mask(selected_pair, target_files)
    if target_mask is None:
        raise VisualizationError(f"No target mask found for pair id '{selected_pair}'.")

    pre_label = labels.get((selected_pair, "pre"))
    post_label = labels.get((selected_pair, "post"))
    if pre_label is None:
        raise VisualizationError(
            f"No pre-disaster JSON label found for pair id '{selected_pair}'."
        )
    if post_label is None:
        raise VisualizationError(
            f"No post-disaster JSON label found for pair id '{selected_pair}'."
        )

    return XbdPair(
        pair_id=selected_pair,
        pre_image=images[(selected_pair, "pre")],
        post_image=images[(selected_pair, "post")],
        target_mask=target_mask,
        pre_label=pre_label,
        post_label=post_label,
    )


def has_complete_sample(
    pair_id: str,
    labels: dict[tuple[str, str], Path],
    target_files: Iterable[Path],
) -> bool:
    return (
        (pair_id, "pre") in labels
        and (pair_id, "post") in labels
        and find_target_mask(pair_id, target_files) is not None
    )


def load_rgb_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except OSError as exc:
        raise VisualizationError(f"Could not load image '{path}': {exc}") from exc


def load_target_mask(path: Path) -> np.ndarray:
    try:
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path))
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                if not data.files:
                    raise VisualizationError(f"Target mask archive is empty: {path}")
                return np.asarray(data[data.files[0]])
        with Image.open(path) as image:
            return np.asarray(image)
    except OSError as exc:
        raise VisualizationError(f"Could not load target mask '{path}': {exc}") from exc


def mask_for_overlay(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    overlay_mask = np.asarray(mask)
    if overlay_mask.ndim == 3:
        if overlay_mask.shape[:2] == expected_shape:
            overlay_mask = np.max(overlay_mask, axis=2)
        elif overlay_mask.shape[1:] == expected_shape:
            overlay_mask = np.max(overlay_mask, axis=0)
        else:
            raise VisualizationError(
                "Target mask shape does not match post-disaster image: "
                f"mask={overlay_mask.shape}, post_image={expected_shape}."
            )
    if overlay_mask.ndim != 2:
        raise VisualizationError(
            f"Target mask must be 2D or RGB-like; got shape {overlay_mask.shape}."
        )
    if overlay_mask.shape != expected_shape:
        raise VisualizationError(
            "Target mask shape does not match post-disaster image: "
            f"mask={overlay_mask.shape}, post_image={expected_shape}."
        )
    return overlay_mask


def mask_for_display(mask: np.ndarray) -> tuple[np.ndarray, str | None]:
    display_mask = np.asarray(mask)
    if display_mask.ndim == 2:
        return display_mask, "gray"
    if display_mask.ndim == 3 and display_mask.shape[2] in (3, 4):
        return display_mask, None
    if display_mask.ndim == 3 and display_mask.shape[0] in (1, 3, 4):
        return np.max(display_mask, axis=0), "gray"
    if display_mask.ndim == 3:
        return np.max(display_mask, axis=2), "gray"
    raise VisualizationError(
        f"Target mask cannot be displayed with shape {display_mask.shape}."
    )


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise VisualizationError(f"Could not parse JSON label '{path}': {exc}") from exc
    except OSError as exc:
        raise VisualizationError(f"Could not read JSON label '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise VisualizationError(f"JSON label is not an object: {path}")
    return data


def count_building_features(label: dict) -> int | None:
    features = label.get("features")
    if isinstance(features, dict):
        for key in ("xy", "lng_lat"):
            items = features.get(key)
            count = count_buildings_in_feature_list(items)
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


def print_label_info(label_name: str, label_path: Path | None, root: Path) -> None:
    print(f"{label_name} label:")
    if label_path is None:
        print("  missing")
        return

    label = read_json(label_path)
    metadata = label.get("metadata")
    metadata_keys = sorted(metadata.keys()) if isinstance(metadata, dict) else []
    building_count = count_building_features(label)
    image_filename = None
    if isinstance(metadata, dict):
        image_filename = metadata.get("img_name") or metadata.get("image_filename")
    image_filename = image_filename or label_path.name

    print(f"  image filename: {image_filename}")
    if building_count is None:
        print("  building features: unavailable")
    else:
        print(f"  building features: {building_count}")
    print(f"  metadata keys: {', '.join(metadata_keys) if metadata_keys else 'none'}")
    print(f"  label file: {relative_path(label_path, root)}")


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_figure(pair: XbdPair, root: Path) -> plt.Figure:
    pre_image = load_rgb_image(pair.pre_image)
    post_image = load_rgb_image(pair.post_image)
    target_mask = load_target_mask(pair.target_mask)
    overlay_mask = mask_for_overlay(target_mask, post_image.shape[:2])
    display_mask, display_cmap = mask_for_display(target_mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"xBD/xView2 sample: {pair.pair_id}", fontsize=14)

    panels = [
        (axes[0, 0], pre_image, "Pre-disaster image", None),
        (axes[0, 1], post_image, "Post-disaster image", None),
        (axes[1, 0], display_mask, "Raw target mask", display_cmap),
    ]

    for axis, data, title, cmap in panels:
        axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    axes[1, 1].imshow(post_image)
    axes[1, 1].imshow(
        np.ma.masked_where(overlay_mask == 0, overlay_mask),
        cmap="Reds",
        alpha=0.45,
    )
    axes[1, 1].set_title("Post-disaster image with target overlay")
    axes[1, 1].axis("off")

    print("Selected sample")
    print(f"  pair id: {pair.pair_id}")
    print(f"  pre image: {relative_path(pair.pre_image, root)}")
    print(f"  post image: {relative_path(pair.post_image, root)}")
    print(f"  target mask: {relative_path(pair.target_mask, root)}")
    print()
    print_label_info("Pre-disaster", pair.pre_label, root)
    print_label_info("Post-disaster", pair.post_label, root)

    return fig


def save_or_show(fig: plt.Figure, output: Path | None) -> None:
    if output is None:
        plt.show()
        return

    output = output.expanduser().resolve()
    if not output.parent.exists():
        raise VisualizationError(f"Output parent folder does not exist: {output.parent}")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {output}")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    try:
        pair = discover_pair(root, args.pair_id)
        fig = build_figure(pair, root)
        save_or_show(fig, args.output)
    except VisualizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

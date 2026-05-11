"""Visualize one xBD/xView2 sample with class-aware target colors."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TARGET_SUFFIXES = IMAGE_SUFFIXES | {".npy", ".npz"}
PRE_SUFFIX = "_pre_disaster"
POST_SUFFIX = "_post_disaster"
UNKNOWN_CLASS = -1

CLASS_DEFINITIONS = {
    "5-class": {
        0: ("background", (0.05, 0.05, 0.05)),
        1: ("no damage", (0.10, 0.58, 0.24)),
        2: ("minor damage", (1.00, 0.82, 0.14)),
        3: ("major damage", (1.00, 0.46, 0.08)),
        4: ("destroyed", (0.82, 0.05, 0.08)),
    },
    "3-class": {
        0: ("background", (0.05, 0.05, 0.05)),
        1: ("no damage", (0.10, 0.58, 0.24)),
        2: ("damaged", (0.88, 0.09, 0.11)),
    },
}
UNKNOWN_DEFINITION = ("unknown", (0.70, 0.12, 0.80))


class VisualizationError(Exception):
    """Raised when a requested xBD sample cannot be loaded cleanly."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one xBD/xView2 training sample. The dataset is read-only; "
            "the script only writes a figure when --output is provided."
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
    parser.add_argument(
        "--mode",
        choices=("5-class", "3-class"),
        default="5-class",
        help="Target mask display mode.",
    )
    return parser.parse_args()


def find_named_dirs(root, dirname):
    matches = []
    stack = [root]
    target_name = dirname.lower()

    while stack:
        current = stack.pop()
        try:
            children = sorted(path for path in current.iterdir() if path.is_dir())
        except OSError as exc:
            raise VisualizationError(f"Could not read directory '{current}': {exc}") from exc

        for child in children:
            if child.name.lower() == target_name:
                matches.append(child)
            else:
                stack.append(child)

    return sorted(matches)


def collect_files(folders, suffixes):
    files = []
    for folder in folders:
        try:
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in suffixes:
                    files.append(path)
        except OSError as exc:
            raise VisualizationError(f"Could not scan files in '{folder}': {exc}") from exc
    return sorted(files)


def parse_xbd_asset(path):
    stem = path.stem
    stem_lower = stem.lower()

    if stem_lower.endswith(PRE_SUFFIX):
        return {
            "path": path,
            "stem": stem,
            "pair_id": stem[: -len(PRE_SUFFIX)],
            "phase": "pre",
        }
    if stem_lower.endswith(POST_SUFFIX):
        return {
            "path": path,
            "stem": stem,
            "pair_id": stem[: -len(POST_SUFFIX)],
            "phase": "post",
        }
    return None


def indexed_assets(paths):
    index = {}
    for path in paths:
        asset = parse_xbd_asset(path)
        if asset is None:
            continue
        key = (asset["pair_id"].lower(), asset["phase"])
        index.setdefault(key, path)
    return index


def target_stem_candidates(pair_id):
    pair = pair_id.lower()
    post = f"{pair}{POST_SUFFIX}"
    pre = f"{pair}{PRE_SUFFIX}"
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


def find_target_mask(pair_id, target_files):
    targets_by_stem = {}
    for target in target_files:
        targets_by_stem.setdefault(target.stem.lower(), []).append(target)

    for candidate in target_stem_candidates(pair_id):
        matches = sorted(targets_by_stem.get(candidate, []))
        if matches:
            return matches[0]

    fallback_matches = sorted(
        target
        for target in target_files
        if target.stem.lower().startswith(pair_id.lower())
    )
    return fallback_matches[0] if fallback_matches else None


def has_complete_sample(pair_id, labels, target_files):
    return (
        (pair_id, "pre") in labels
        and (pair_id, "post") in labels
        and find_target_mask(pair_id, target_files) is not None
    )


def discover_pair(root, pair_id):
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
            if has_complete_sample(candidate_pair, labels, target_files):
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

    return {
        "pair_id": selected_pair,
        "pre_image": images[(selected_pair, "pre")],
        "post_image": images[(selected_pair, "post")],
        "target_mask": target_mask,
        "pre_label": pre_label,
        "post_label": post_label,
    }


def load_rgb_image(path):
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except OSError as exc:
        raise VisualizationError(f"Could not load image '{path}': {exc}") from exc


def load_target_mask(path):
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


def target_class_map(mask, expected_shape):
    class_map = np.asarray(mask)

    if class_map.ndim == 3:
        if class_map.shape[:2] == expected_shape:
            class_map = reduce_mask_channels(class_map)
        elif class_map.shape[1:] == expected_shape:
            class_map = reduce_mask_channels(np.moveaxis(class_map, 0, -1))
        else:
            raise VisualizationError(
                "Target mask shape does not match post-disaster image: "
                f"mask={class_map.shape}, post_image={expected_shape}."
            )

    if class_map.ndim != 2:
        raise VisualizationError(
            f"Target mask must be 2D or RGB-like; got shape {class_map.shape}."
        )
    if class_map.shape != expected_shape:
        raise VisualizationError(
            "Target mask shape does not match post-disaster image: "
            f"mask={class_map.shape}, post_image={expected_shape}."
        )
    if not np.issubdtype(class_map.dtype, np.number):
        raise VisualizationError(
            f"Target mask must contain numeric class ids: {class_map.dtype}."
        )

    rounded = np.rint(class_map)
    if not np.allclose(class_map, rounded):
        raise VisualizationError("Target mask contains non-integer class values.")
    return rounded.astype(np.int16, copy=False)


def reduce_mask_channels(mask):
    if mask.shape[2] == 1:
        return mask[:, :, 0]
    if mask.shape[2] in (3, 4):
        rgb = mask[:, :, :3]
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
            rgb[:, :, 0],
            rgb[:, :, 2],
        ):
            return rgb[:, :, 0]
        return np.max(rgb, axis=2)
    raise VisualizationError(f"Unsupported target mask channel count: {mask.shape}.")


def display_class_map(class_map, mode):
    if mode == "5-class":
        display_map = class_map.copy()
        known = np.isin(display_map, list(CLASS_DEFINITIONS[mode]))
        display_map[~known] = UNKNOWN_CLASS
        return display_map

    display_map = np.full(class_map.shape, UNKNOWN_CLASS, dtype=np.int16)
    display_map[class_map == 0] = 0
    display_map[class_map == 1] = 1
    display_map[np.isin(class_map, [2, 3, 4])] = 2
    return display_map


def colorize_class_map(display_map, mode):
    colors = CLASS_DEFINITIONS[mode]
    color_image = np.zeros((*display_map.shape, 3), dtype=np.float32)

    for class_id, (_, color) in colors.items():
        color_image[display_map == class_id] = color
    color_image[display_map == UNKNOWN_CLASS] = UNKNOWN_DEFINITION[1]
    return color_image


def blend_overlay(image, color_image, display_map):
    image_float = image.astype(np.float32) / 255.0
    alpha = np.zeros(display_map.shape, dtype=np.float32)
    alpha[display_map == 1] = 0.38
    alpha[(display_map != 0) & (display_map != 1)] = 0.52
    alpha = alpha[:, :, None]
    return image_float * (1.0 - alpha) + color_image * alpha


def legend_handles(display_map, mode):
    handles = []
    colors = CLASS_DEFINITIONS[mode]
    observed = set(int(value) for value in np.unique(display_map))

    for class_id, (label, color) in colors.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="white",
                label=f"{class_id} = {label}",
                markersize=10,
            )
        )

    if UNKNOWN_CLASS in observed:
        label, color = UNKNOWN_DEFINITION
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="white",
                label=label,
                markersize=10,
            )
        )
    return handles


def read_json(path):
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


def count_building_features(label):
    features = label.get("features")
    if isinstance(features, dict):
        for key in ("xy", "lng_lat"):
            count = count_buildings_in_feature_list(features.get(key))
            if count is not None:
                return count
    if isinstance(features, list):
        return count_buildings_in_feature_list(features)
    return None


def count_buildings_in_feature_list(items):
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


def print_label_info(label_name, label_path, root):
    print(f"{label_name} label:")
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


def print_target_counts(class_map):
    values, counts = np.unique(class_map, return_counts=True)
    print("Target mask values:")
    for value, count in zip(values, counts):
        print(f"  {int(value)}: {int(count)} pixels")
    print(f"  total: {int(class_map.size)} pixels")


def relative_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_figure(pair, root, mode):
    pre_image = load_rgb_image(pair["pre_image"])
    post_image = load_rgb_image(pair["post_image"])
    target_mask = load_target_mask(pair["target_mask"])
    class_map = target_class_map(target_mask, post_image.shape[:2])
    display_map = display_class_map(class_map, mode)
    color_mask = colorize_class_map(display_map, mode)
    overlay = blend_overlay(post_image, color_mask, display_map)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"xBD/xView2 sample: {pair['pair_id']} ({mode})", fontsize=14)

    panels = [
        (axes[0, 0], pre_image, "Pre-disaster image"),
        (axes[0, 1], post_image, "Post-disaster image"),
        (axes[1, 0], color_mask, "Colorized target mask"),
        (axes[1, 1], overlay, "Post-disaster image with class overlay"),
    ]

    for axis, image, title in panels:
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    fig.legend(
        handles=legend_handles(display_map, mode),
        loc="lower center",
        ncol=3 if mode == "5-class" else 2,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.14, left=0.02, right=0.98, top=0.92, wspace=0.04)

    print("Selected sample")
    print(f"  pair id: {pair['pair_id']}")
    print(f"  mode: {mode}")
    print(f"  pre image: {relative_path(pair['pre_image'], root)}")
    print(f"  post image: {relative_path(pair['post_image'], root)}")
    print(f"  target mask: {relative_path(pair['target_mask'], root)}")
    print()
    print_target_counts(class_map)
    print()
    print_label_info("Pre-disaster", pair["pre_label"], root)
    print_label_info("Post-disaster", pair["post_label"], root)

    return fig


def save_or_show(fig, output):
    if output is None:
        plt.show()
        return

    output = output.expanduser().resolve()
    if not output.parent.exists():
        raise VisualizationError(f"Output parent folder does not exist: {output.parent}")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {output}")


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    try:
        pair = discover_pair(root, args.pair_id)
        fig = build_figure(pair, root, args.mode)
        save_or_show(fig, args.output)
    except VisualizationError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

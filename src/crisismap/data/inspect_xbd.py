"""Inspect an extracted xBD/xView2 training dataset without modifying data."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TARGET_SUFFIXES = IMAGE_SUFFIXES | {".npy", ".npz"}
XBD_STEM_PATTERN = re.compile(
    r"^(?P<pair_id>.+)_(?P<phase>pre|post)_disaster$",
    flags=re.IGNORECASE,
)


class InspectionError(Exception):
    """Raised when the dataset root is missing required xBD folders."""


@dataclass(frozen=True)
class ParsedAsset:
    """Parsed metadata from a standard xBD filename."""

    path: Path
    stem: str
    pair_id: str
    phase: str
    disaster: str


@dataclass(frozen=True)
class FolderDiscovery:
    images: list[Path]
    labels: list[Path]
    targets: list[Path]


@dataclass(frozen=True)
class InspectionReport:
    root: Path
    folders: FolderDiscovery
    image_files: list[Path]
    label_files: list[Path]
    target_files: list[Path]
    parsed_images: list[ParsedAsset]
    unparsed_images: list[Path]
    unparsed_labels: list[Path]
    pre_images: list[ParsedAsset]
    post_images: list[ParsedAsset]
    missing_post_pairs: list[ParsedAsset]
    missing_pre_pairs: list[ParsedAsset]
    duplicate_pre_pairs: dict[str, list[ParsedAsset]]
    duplicate_post_pairs: dict[str, list[ParsedAsset]]
    missing_label_images: list[ParsedAsset]
    extra_label_files: list[Path]
    missing_target_pairs: list[ParsedAsset]
    disaster_counts: Counter[str]

    @property
    def has_integrity_issues(self) -> bool:
        return any(
            (
                self.unparsed_images,
                self.unparsed_labels,
                self.missing_post_pairs,
                self.missing_pre_pairs,
                self.duplicate_pre_pairs,
                self.duplicate_post_pairs,
                self.missing_label_images,
                self.extra_label_files,
                self.missing_target_pairs,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an extracted xBD/xView2 training folder. The script is "
            "read-only: it does not modify, move, delete, or download data."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Maximum number of example problem paths to print per check.",
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
            raise InspectionError(f"Could not read directory '{current}': {exc}") from exc

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
            raise InspectionError(f"Could not scan files in '{folder}': {exc}") from exc
    return sorted(files)


def parse_xbd_asset(path: Path) -> ParsedAsset | None:
    match = XBD_STEM_PATTERN.match(path.stem)
    if not match:
        return None

    pair_id = match.group("pair_id")
    phase = match.group("phase").lower()
    disaster = extract_disaster_name(pair_id)
    return ParsedAsset(
        path=path,
        stem=path.stem,
        pair_id=pair_id,
        phase=phase,
        disaster=disaster,
    )


def extract_disaster_name(pair_id: str) -> str:
    """Extract the disaster name from the pair id before the final tile id."""

    if "_" not in pair_id:
        return pair_id

    disaster, tile_id = pair_id.rsplit("_", 1)
    if tile_id:
        return disaster
    return pair_id


def group_by_pair(assets: Iterable[ParsedAsset], phase: str) -> dict[str, list[ParsedAsset]]:
    grouped: dict[str, list[ParsedAsset]] = defaultdict(list)
    for asset in assets:
        if asset.phase == phase:
            grouped[asset.pair_id.lower()].append(asset)
    return dict(grouped)


def find_extra_labels(label_files: list[Path], image_stems: set[str]) -> list[Path]:
    extras: list[Path] = []
    for label in label_files:
        parsed = parse_xbd_asset(label)
        if parsed and parsed.stem.lower() not in image_stems:
            extras.append(label)
    return sorted(extras)


def target_candidates(asset: ParsedAsset) -> set[str]:
    pair = asset.pair_id.lower()
    stem = asset.stem.lower()
    return {
        pair,
        f"{pair}_target",
        f"{pair}_targets",
        f"{pair}_mask",
        f"{pair}_damage",
        stem,
        f"{stem}_target",
        f"{stem}_targets",
        f"{stem}_mask",
        f"{stem}_damage",
    }


def discover_folders(root: Path) -> FolderDiscovery:
    return FolderDiscovery(
        images=find_named_dirs(root, "images"),
        labels=find_named_dirs(root, "labels"),
        targets=find_named_dirs(root, "targets"),
    )


def inspect_dataset(root: Path) -> InspectionReport:
    root = root.expanduser().resolve()
    if not root.exists():
        raise InspectionError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise InspectionError(f"Root path is not a directory: {root}")

    folders = discover_folders(root)
    if not folders.images:
        raise InspectionError("No folder named 'images' was found under --root.")
    if not folders.labels:
        raise InspectionError("No folder named 'labels' was found under --root.")

    image_files = collect_files(folders.images, IMAGE_SUFFIXES)
    label_files = collect_files(folders.labels, {".json"})
    target_files = collect_files(folders.targets, TARGET_SUFFIXES)

    if not image_files:
        raise InspectionError("Found 'images' folder(s), but no supported image files.")
    if not label_files:
        raise InspectionError("Found 'labels' folder(s), but no JSON label files.")

    parsed_images: list[ParsedAsset] = []
    unparsed_images: list[Path] = []
    for image in image_files:
        parsed = parse_xbd_asset(image)
        if parsed is None:
            unparsed_images.append(image)
        else:
            parsed_images.append(parsed)

    unparsed_labels = [
        label for label in label_files if parse_xbd_asset(label) is None
    ]

    pre_images = [asset for asset in parsed_images if asset.phase == "pre"]
    post_images = [asset for asset in parsed_images if asset.phase == "post"]
    pre_by_pair = group_by_pair(parsed_images, "pre")
    post_by_pair = group_by_pair(parsed_images, "post")

    missing_post_pairs = [
        assets[0]
        for pair_id, assets in pre_by_pair.items()
        if pair_id not in post_by_pair
    ]
    missing_pre_pairs = [
        assets[0]
        for pair_id, assets in post_by_pair.items()
        if pair_id not in pre_by_pair
    ]
    duplicate_pre_pairs = {
        pair_id: assets for pair_id, assets in pre_by_pair.items() if len(assets) > 1
    }
    duplicate_post_pairs = {
        pair_id: assets for pair_id, assets in post_by_pair.items() if len(assets) > 1
    }

    label_stems = {label.stem.lower() for label in label_files}
    image_stems = {image.stem.lower() for image in image_files}
    missing_label_images = [
        asset for asset in parsed_images if asset.stem.lower() not in label_stems
    ]
    extra_label_files = find_extra_labels(label_files, image_stems)

    target_stems = {target.stem.lower() for target in target_files}
    missing_target_pairs: list[ParsedAsset] = []
    if folders.targets:
        for asset in post_images:
            if target_candidates(asset).isdisjoint(target_stems):
                missing_target_pairs.append(asset)

    disaster_by_pair: dict[str, str] = {}
    for asset in parsed_images:
        disaster_by_pair.setdefault(asset.pair_id.lower(), asset.disaster)
    disaster_counts = Counter(disaster_by_pair.values())

    return InspectionReport(
        root=root,
        folders=folders,
        image_files=image_files,
        label_files=label_files,
        target_files=target_files,
        parsed_images=parsed_images,
        unparsed_images=unparsed_images,
        unparsed_labels=unparsed_labels,
        pre_images=pre_images,
        post_images=post_images,
        missing_post_pairs=missing_post_pairs,
        missing_pre_pairs=missing_pre_pairs,
        duplicate_pre_pairs=duplicate_pre_pairs,
        duplicate_post_pairs=duplicate_post_pairs,
        missing_label_images=missing_label_images,
        extra_label_files=extra_label_files,
        missing_target_pairs=missing_target_pairs,
        disaster_counts=disaster_counts,
    )


def print_paths(title: str, paths: Iterable[Path], root: Path, examples: int) -> None:
    paths = list(paths)
    if not paths:
        return

    print(f"  {title}: {len(paths)}")
    for path in paths[:examples]:
        print(f"    - {relative_path(path, root)}")
    if len(paths) > examples:
        print(f"    ... {len(paths) - examples} more")


def print_pair_assets(
    title: str,
    assets: Iterable[ParsedAsset],
    root: Path,
    examples: int,
) -> None:
    print_paths(title, [asset.path for asset in assets], root, examples)


def print_duplicate_pairs(
    title: str,
    duplicates: dict[str, list[ParsedAsset]],
    root: Path,
    examples: int,
) -> None:
    if not duplicates:
        return

    print(f"  {title}: {len(duplicates)}")
    for pair_id, assets in list(sorted(duplicates.items()))[:examples]:
        paths = ", ".join(relative_path(asset.path, root) for asset in assets)
        print(f"    - {pair_id}: {paths}")
    if len(duplicates) > examples:
        print(f"    ... {len(duplicates) - examples} more")


def print_report(report: InspectionReport, examples: int) -> None:
    root = report.root
    folders = report.folders

    print("CrisisMap AI - xBD/xView2 Dataset Inspection")
    print("=" * 48)
    print(f"Root: {root}")
    print()

    print("Discovered folders")
    for name, folder_list in (
        ("images", folders.images),
        ("labels", folders.labels),
        ("targets", folders.targets),
    ):
        print(f"  {name}: {len(folder_list)}")
        for folder in folder_list:
            print(f"    - {relative_path(folder, root)}")
        if name == "targets" and not folder_list:
            print("    - not found; target mask checks skipped")
    print()

    print("File counts")
    print(f"  total image files: {len(report.image_files)}")
    print(f"  pre-disaster images: {len(report.pre_images)}")
    print(f"  post-disaster images: {len(report.post_images)}")
    print(f"  JSON labels: {len(report.label_files)}")
    print(f"  target masks: {len(report.target_files)}")
    print()

    print("Disasters by image pair")
    if report.disaster_counts:
        for disaster, count in sorted(report.disaster_counts.items()):
            print(f"  {disaster}: {count}")
    else:
        print("  none parsed")
    print()

    print("Integrity checks")
    if not report.has_integrity_issues:
        print("  OK: image pairs, labels, and discovered targets look consistent.")
    else:
        print("  Issues found:")
        print_paths(
            "images with non-standard xBD names",
            report.unparsed_images,
            root,
            examples,
        )
        print_paths(
            "labels with non-standard xBD names",
            report.unparsed_labels,
            root,
            examples,
        )
        print_pair_assets(
            "pre-disaster images missing matching post-disaster image",
            report.missing_post_pairs,
            root,
            examples,
        )
        print_pair_assets(
            "post-disaster images missing matching pre-disaster image",
            report.missing_pre_pairs,
            root,
            examples,
        )
        print_duplicate_pairs(
            "duplicate pre-disaster pair ids",
            report.duplicate_pre_pairs,
            root,
            examples,
        )
        print_duplicate_pairs(
            "duplicate post-disaster pair ids",
            report.duplicate_post_pairs,
            root,
            examples,
        )
        print_pair_assets(
            "images missing matching JSON labels",
            report.missing_label_images,
            root,
            examples,
        )
        print_pair_assets(
            "post-disaster pairs missing target masks",
            report.missing_target_pairs,
            root,
            examples,
        )
        print_paths(
            "extra parsed JSON labels without matching images",
            report.extra_label_files,
            root,
            examples,
        )

    print()
    if report.has_integrity_issues:
        print("Status: completed with integrity issues.")
    else:
        print("Status: completed successfully.")


def main() -> int:
    args = parse_args()
    try:
        report = inspect_dataset(args.root)
    except InspectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(report, max(args.examples, 0))
    return 2 if report.has_integrity_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

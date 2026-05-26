"""PyTorch Dataset for xBD/xView2 pre/post image pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


REQUIRED_COLUMNS = {"pair_id", "pre_image", "post_image", "target"}
TARGET_MODES = {"3-class", "5-class", "building-binary"}
AUGMENT_MODES = {"none", "safe", "damage-aware"}
DAMAGE_CLASSES = {2, 3, 4}


class XBDDatasetError(Exception):
    """Raised when xBD pair samples cannot be loaded safely."""


class XBDPairDataset(torch.utils.data.Dataset):
    """Load xBD pre/post image pairs as 6-channel tensors."""

    def __init__(
        self,
        root: str | Path,
        split_csv: str | Path,
        image_size: int = 512,
        target_mode: str = "3-class",
        augment_mode: str = "none",
        augment_prob: float = 0.0,
        damage_augment_threshold: float = 0.001,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split_csv = Path(split_csv).expanduser().resolve()
        self.image_size = validate_image_size(image_size)
        self.target_mode = validate_target_mode(target_mode)
        self.augment_mode = validate_augment_mode(augment_mode)
        self.augment_prob = validate_probability(augment_prob, "augment_prob")
        self.damage_augment_threshold = validate_nonnegative_float(
            damage_augment_threshold,
            "damage_augment_threshold",
        )
        self.samples = load_split_csv(self.split_csv)

        if not self.root.exists():
            raise XBDDatasetError(f"Root path does not exist: {self.root}")
        if not self.root.is_dir():
            raise XBDDatasetError(f"Root path is not a directory: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index = len(self.samples) + index
        if index < 0 or index >= len(self.samples):
            raise IndexError(index)

        row = self.samples.iloc[index]
        pair_id = str(row["pair_id"])
        pre_path = resolve_data_path(self.root, row["pre_image"])
        post_path = resolve_data_path(self.root, row["post_image"])
        target_path = resolve_data_path(self.root, row["target"])

        pre_image = load_rgb_image(pre_path, self.image_size)
        post_image = load_rgb_image(post_path, self.image_size)
        target_mask = load_target_mask(target_path, self.image_size)
        target_mask = convert_target_mode(target_mask, self.target_mode)
        damage_ratio = sample_damage_ratio(row)
        pre_image, post_image, target_mask = maybe_augment_sample(
            pre_image,
            post_image,
            target_mask,
            damage_ratio=damage_ratio,
            augment_mode=self.augment_mode,
            augment_prob=self.augment_prob,
            damage_augment_threshold=self.damage_augment_threshold,
        )

        image = np.concatenate([pre_image, post_image], axis=2)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
        image_tensor = image_tensor.to(dtype=torch.float32).div(255.0)
        target_tensor = torch.from_numpy(target_mask.copy()).to(dtype=torch.long)

        return {
            "image": image_tensor,
            "target": target_tensor,
            "pair_id": pair_id,
            "pre_image": str(pre_path),
            "post_image": str(post_path),
            "target_path": str(target_path),
        }


def validate_image_size(image_size: int) -> int:
    try:
        image_size = int(image_size)
    except (TypeError, ValueError) as exc:
        raise XBDDatasetError("image_size must be a positive integer.") from exc
    if image_size <= 0:
        raise XBDDatasetError("image_size must be a positive integer.")
    return image_size


def validate_target_mode(target_mode: str) -> str:
    if target_mode not in TARGET_MODES:
        raise XBDDatasetError(
            "target_mode must be one of: " + ", ".join(sorted(TARGET_MODES))
        )
    return target_mode


def validate_augment_mode(augment_mode: str) -> str:
    if augment_mode not in AUGMENT_MODES:
        raise XBDDatasetError(
            "augment_mode must be one of: " + ", ".join(sorted(AUGMENT_MODES))
        )
    return augment_mode


def validate_probability(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise XBDDatasetError(f"{name} must be a float between 0 and 1.") from exc
    if not 0.0 <= value <= 1.0:
        raise XBDDatasetError(f"{name} must be between 0 and 1.")
    return value


def validate_nonnegative_float(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise XBDDatasetError(f"{name} must be a non-negative float.") from exc
    if value < 0.0:
        raise XBDDatasetError(f"{name} must be non-negative.")
    return value


def load_split_csv(split_csv: Path) -> pd.DataFrame:
    if not split_csv.exists():
        raise XBDDatasetError(f"Split CSV does not exist: {split_csv}")
    if not split_csv.is_file():
        raise XBDDatasetError(f"Split CSV path is not a file: {split_csv}")

    try:
        df = pd.read_csv(split_csv)
    except OSError as exc:
        raise XBDDatasetError(f"Could not read split CSV '{split_csv}': {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise XBDDatasetError(f"Split CSV is empty: {split_csv}") from exc
    except pd.errors.ParserError as exc:
        raise XBDDatasetError(f"Could not parse split CSV '{split_csv}': {exc}") from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise XBDDatasetError(
            "Split CSV is missing required column(s): " + ", ".join(missing_columns)
        )
    if df.empty:
        raise XBDDatasetError("Split CSV has no rows.")

    return df.reset_index(drop=True)


def resolve_data_path(root: Path, value: object) -> Path:
    if pd.isna(value):
        raise XBDDatasetError("Encountered missing path value in split CSV.")

    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def load_rgb_image(path: Path, image_size: int) -> np.ndarray:
    if not path.exists():
        raise XBDDatasetError(f"Image file does not exist: {path}")
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    except OSError as exc:
        raise XBDDatasetError(f"Could not load image '{path}': {exc}") from exc


def load_target_mask(path: Path, image_size: int) -> np.ndarray:
    if not path.exists():
        raise XBDDatasetError(f"Target mask file does not exist: {path}")

    try:
        if path.suffix.lower() == ".npy":
            mask = np.asarray(np.load(path))
        elif path.suffix.lower() == ".npz":
            with np.load(path) as data:
                if not data.files:
                    raise XBDDatasetError(f"Target mask archive is empty: {path}")
                mask = np.asarray(data[data.files[0]])
        else:
            with Image.open(path) as image:
                mask = np.asarray(image)
    except OSError as exc:
        raise XBDDatasetError(f"Could not load target mask '{path}': {exc}") from exc

    mask = target_class_map(mask)
    mask_image = Image.fromarray(mask.astype(np.int32), mode="I")
    mask_image = mask_image.resize((image_size, image_size), Image.NEAREST)
    return np.asarray(mask_image, dtype=np.int64)


def target_class_map(mask: np.ndarray) -> np.ndarray:
    class_map = np.asarray(mask)

    if class_map.ndim == 3:
        class_map = reduce_mask_channels(class_map)
    if class_map.ndim != 2:
        raise XBDDatasetError(
            f"Target mask must be 2D or RGB-like; got shape {class_map.shape}."
        )
    if not np.issubdtype(class_map.dtype, np.number):
        raise XBDDatasetError(
            f"Target mask must contain numeric class ids: {class_map.dtype}."
        )

    rounded = np.rint(class_map)
    if not np.allclose(class_map, rounded):
        raise XBDDatasetError("Target mask contains non-integer class values.")
    return rounded.astype(np.int16, copy=False)


def reduce_mask_channels(mask: np.ndarray) -> np.ndarray:
    if mask.shape[-1] in (1, 3, 4):
        if mask.shape[-1] == 1:
            return mask[:, :, 0]
        rgb = mask[:, :, :3]
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
            rgb[:, :, 0],
            rgb[:, :, 2],
        ):
            return rgb[:, :, 0]
        return np.max(rgb, axis=2)

    if mask.shape[0] in (1, 3, 4):
        return reduce_mask_channels(np.moveaxis(mask, 0, -1))

    raise XBDDatasetError(f"Unsupported target mask shape: {mask.shape}.")


def convert_target_mode(mask: np.ndarray, target_mode: str) -> np.ndarray:
    mask = np.asarray(mask)
    validate_known_target_values(mask)

    if target_mode == "building-binary":
        return (mask > 0).astype(np.int64, copy=False)

    if target_mode == "5-class":
        return mask.astype(np.int64, copy=False)

    converted = np.zeros(mask.shape, dtype=np.int64)
    converted[mask == 1] = 1
    converted[np.isin(mask, [2, 3, 4])] = 2
    return converted


def validate_known_target_values(mask: np.ndarray) -> None:
    allowed_values = {0, 1, 2, 3, 4}
    observed_values = {int(value) for value in np.unique(mask)}
    unknown_values = sorted(observed_values - allowed_values)
    if unknown_values:
        raise XBDDatasetError(
            "Target mask contains values outside expected xBD classes 0-4: "
            + ", ".join(str(value) for value in unknown_values[:10])
        )


def sample_damage_ratio(row: pd.Series) -> float:
    if "damage_ratio" in row.index and not pd.isna(row["damage_ratio"]):
        try:
            return float(row["damage_ratio"])
        except (TypeError, ValueError):
            return 0.0

    if {"target_value_counts", "target_total_pixels"}.issubset(row.index):
        return compute_damage_ratio(
            row["target_value_counts"],
            row["target_total_pixels"],
        )
    return 0.0


def compute_damage_ratio(counts_json: object, total_pixels_value: object) -> float:
    if pd.isna(counts_json):
        return 0.0
    try:
        counts = json_loads_dict(counts_json)
        total_pixels = int(float(total_pixels_value))
    except (TypeError, ValueError):
        return 0.0
    if total_pixels <= 0:
        return 0.0

    damage_pixels = 0
    for raw_class, raw_count in counts.items():
        try:
            target_class = int(float(str(raw_class).strip()))
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if target_class in DAMAGE_CLASSES:
            damage_pixels += count
    return damage_pixels / total_pixels


def json_loads_dict(value: object) -> dict[object, object]:
    import json

    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def maybe_augment_sample(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    target_mask: np.ndarray,
    damage_ratio: float,
    augment_mode: str,
    augment_prob: float,
    damage_augment_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if augment_mode == "none" or augment_prob <= 0.0:
        return pre_image, post_image, target_mask

    probability = augment_prob
    if augment_mode == "damage-aware" and damage_ratio <= damage_augment_threshold:
        probability = augment_prob * 0.5

    if np.random.random() >= probability:
        return pre_image, post_image, target_mask

    pre_image, post_image, target_mask = apply_geometric_augmentation(
        pre_image,
        post_image,
        target_mask,
    )
    pre_image = apply_photometric_augmentation(pre_image)
    post_image = apply_photometric_augmentation(post_image)
    return pre_image, post_image, target_mask


def apply_geometric_augmentation(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.random.random() < 0.5:
        pre_image = np.flip(pre_image, axis=1)
        post_image = np.flip(post_image, axis=1)
        target_mask = np.flip(target_mask, axis=1)
    if np.random.random() < 0.5:
        pre_image = np.flip(pre_image, axis=0)
        post_image = np.flip(post_image, axis=0)
        target_mask = np.flip(target_mask, axis=0)

    rotations = int(np.random.randint(0, 4))
    if rotations:
        pre_image = np.rot90(pre_image, k=rotations, axes=(0, 1))
        post_image = np.rot90(post_image, k=rotations, axes=(0, 1))
        target_mask = np.rot90(target_mask, k=rotations, axes=(0, 1))

    return pre_image, post_image, target_mask


def apply_photometric_augmentation(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    brightness = np.random.uniform(0.95, 1.05)
    contrast = np.random.uniform(0.95, 1.05)
    channel_mean = image_float.mean(axis=(0, 1), keepdims=True)
    image_float = (image_float - channel_mean) * contrast + channel_mean
    image_float = image_float * brightness

    if np.random.random() < 0.3:
        noise = np.random.normal(loc=0.0, scale=2.0, size=image_float.shape)
        image_float = image_float + noise

    return np.clip(image_float, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the xBD PyTorch dataset.")
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the extracted xBD/xView2 training folder.",
    )
    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
        help="Path to train_pairs.csv, val_pairs.csv, or test_pairs.csv.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square image and mask size used by the dataset.",
    )
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_MODES),
        default="3-class",
        help="Target mask class mapping.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to inspect.",
    )
    return parser.parse_args()


def smoke_test(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise XBDDatasetError("--num-samples must be a positive integer.")

    dataset = XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
    )

    print(f"Dataset length: {len(dataset)}")
    samples_to_show = min(args.num_samples, len(dataset))
    for index in range(samples_to_show):
        sample = dataset[index]
        image = sample["image"]
        target = sample["target"]
        unique_values = torch.unique(target).cpu().tolist()

        print(f"Sample {index}: {sample['pair_id']}")
        print(f"  input tensor shape: {tuple(image.shape)}")
        print(f"  target tensor shape: {tuple(target.shape)}")
        print(f"  input dtype: {image.dtype}")
        print(f"  target dtype: {target.dtype}")
        print(f"  unique target values: {unique_values}")


def main() -> int:
    args = parse_args()
    try:
        smoke_test(args)
    except (XBDDatasetError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

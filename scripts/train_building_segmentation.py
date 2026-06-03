"""Train a binary building segmentation model for CrisisMap AI / Aftermath."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.models.unet import UNet  # noqa: E402


MODEL_CHOICES = {
    "unet",
    "unetplusplus_effb3",
    "unetplusplus_effb4",
    "deeplabv3plus_resnet50",
    "deeplabv3plus_effb3",
    "fpn_effb3",
}
INPUT_MODES = {"pre", "post", "pre-post"}
LOSS_CHOICES = {
    "bce-dice",
    "dice-bce",
    "focal-dice",
    "focal-tversky",
    "bce-dice-boundary",
    "focal-tversky-boundary",
}
AUGMENT_MODES = {"none", "safe", "building-safe", "building-strong"}
SAMPLER_MODES = {"none", "building-sqrt"}
TRAIN_MODES = {"full1024", "crop512", "crop608"}


class BuildingTrainingError(Exception):
    """Raised when building segmentation training cannot continue safely."""


class BuildingInputDataset(torch.utils.data.Dataset):
    """Select pre, post, or pre-post channels and apply train-only augmentation."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        input_mode: str,
        augment_mode: str = "none",
        crop_size: int | None = None,
        rare_building_crop_prob: float = 0.75,
        rare_building_crop_alpha: float | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.input_mode = validate_input_mode(input_mode)
        self.augment_mode = validate_augment_mode(augment_mode)
        self.crop_size = crop_size
        self.rare_building_crop_prob = rare_building_crop_prob
        self.rare_building_crop_alpha = rare_building_crop_alpha

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.base_dataset[index]
        image = sample["image"]
        if self.input_mode == "pre":
            image = image[:3]
        elif self.input_mode == "post":
            image = image[3:6]
        elif self.input_mode == "pre-post":
            image = image
        else:
            raise BuildingTrainingError(f"Unsupported input mode: {self.input_mode}")

        target = sample["target"]
        if self.crop_size is not None:
            image, target = random_building_crop(
                image,
                target,
                self.crop_size,
                self.rare_building_crop_prob,
                self.rare_building_crop_alpha,
            )
        if self.augment_mode != "none":
            image, target = apply_building_augmentation(image, target, self.augment_mode)

        return {
            **sample,
            "image": image.contiguous(),
            "target": target.contiguous(),
        }


class BinaryDiceLoss(torch.nn.Module):
    """Binary Dice loss for logits and float masks."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        intersection = torch.sum(probabilities * targets)
        denominator = torch.sum(probabilities) + torch.sum(targets)
        dice = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        return 1.0 - dice


class BinaryFocalLoss(torch.nn.Module):
    """Binary focal loss for logits and float masks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        return loss.mean()


class BCEDiceLoss(torch.nn.Module):
    """BCEWithLogits plus binary Dice loss."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


class FocalDiceLoss(torch.nn.Module):
    """Binary focal loss plus Dice loss."""

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice(logits, targets)


class FocalTverskyLoss(torch.nn.Module):
    """Binary focal Tversky loss for recall-oriented building segmentation."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        true_positive = torch.sum(probabilities * targets)
        false_positive = torch.sum(probabilities * (1.0 - targets))
        false_negative = torch.sum((1.0 - probabilities) * targets)
        tversky = (true_positive + self.epsilon) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.epsilon
        )
        return torch.pow(1.0 - tversky, self.gamma)


class BoundaryLoss(torch.nn.Module):
    """Differentiable boundary alignment loss for binary building masks."""

    def __init__(self, kernel_size: int = 3, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        pred_boundary = soft_boundary(probabilities, self.kernel_size, self.padding)
        target_boundary = soft_boundary(targets, self.kernel_size, self.padding)
        bce = F.binary_cross_entropy(pred_boundary.clamp(self.epsilon, 1.0 - self.epsilon), target_boundary)
        intersection = torch.sum(pred_boundary * target_boundary)
        denominator = torch.sum(pred_boundary) + torch.sum(target_boundary)
        dice = 1.0 - (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        return bce + dice


class BoundaryAwareLoss(torch.nn.Module):
    """Base binary loss plus a lightweight boundary term."""

    def __init__(self, base_loss: torch.nn.Module, boundary_weight: float) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.boundary_loss = BoundaryLoss()
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.base_loss(logits, targets) + self.boundary_weight * self.boundary_loss(logits, targets)


def soft_boundary(
    mask: torch.Tensor,
    kernel_size: int = 3,
    padding: int = 1,
) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary building segmentation model on xBD/xView2.",
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="unetplusplus_effb3")
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), default="pre")
    parser.add_argument("--train-mode", choices=sorted(TRAIN_MODES), default="full1024")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=sorted(LOSS_CHOICES), default="focal-tversky")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument(
        "--drop-last-train",
        action="store_true",
        help="Drop the final incomplete train batch; val/test loaders are unchanged.",
    )
    parser.add_argument(
        "--augment-mode",
        choices=sorted(AUGMENT_MODES),
        default="building-safe",
        help="Train-only augmentation mode.",
    )
    parser.add_argument(
        "--sampler",
        choices=sorted(SAMPLER_MODES),
        default="none",
        help="Train-only WeightedRandomSampler mode.",
    )
    parser.add_argument(
        "--sampler-alpha",
        type=float,
        default=4.0,
        help="Alpha used by --sampler building-sqrt.",
    )
    parser.add_argument(
        "--rare-building-crop-prob",
        type=float,
        default=0.75,
        help="Probability of centering train crops on building pixels.",
    )
    parser.add_argument(
        "--rare-building-crop-alpha",
        type=float,
        default=None,
        help="Optional alpha converted to alpha/(1+alpha) for building-centered crops.",
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=0.2,
        help="Boundary loss weight for *-boundary loss modes.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["building-binary"],
        default="building-binary",
        help="Only building-binary is supported by this script.",
    )
    parser.add_argument("--resume", dest="resume_checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", dest="resume_checkpoint", type=Path, default=None)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-tversky-alpha", type=float, default=0.3)
    parser.add_argument("--focal-tversky-beta", type=float, default=0.7)
    parser.add_argument("--focal-tversky-gamma", type=float, default=0.75)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print concise train progress every N batches. Use 0 to disable.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ["image_size", "batch_size", "epochs"]:
        if getattr(args, name) <= 0:
            raise BuildingTrainingError(f"--{name.replace('_', '-')} must be positive.")
    if args.crop_size is not None and args.crop_size <= 0:
        raise BuildingTrainingError("--crop-size must be positive when provided.")
    if args.train_mode != "full1024":
        crop_size = resolve_crop_size(args.train_mode, args.crop_size)
        if crop_size >= args.image_size:
            raise BuildingTrainingError("--crop-size must be smaller than --image-size.")
    if args.lr <= 0:
        raise BuildingTrainingError("--lr must be positive.")
    if args.num_workers < 0:
        raise BuildingTrainingError("--num-workers must be non-negative.")
    if args.log_every < 0:
        raise BuildingTrainingError("--log-every must be non-negative.")
    if args.sampler_alpha < 0:
        raise BuildingTrainingError("--sampler-alpha must be non-negative.")
    if not 0.0 <= args.rare_building_crop_prob <= 1.0:
        raise BuildingTrainingError("--rare-building-crop-prob must be between 0 and 1.")
    if args.rare_building_crop_alpha is not None and args.rare_building_crop_alpha < 0:
        raise BuildingTrainingError("--rare-building-crop-alpha must be non-negative.")
    if args.boundary_loss_weight < 0:
        raise BuildingTrainingError("--boundary-loss-weight must be non-negative.")
    for name in ["max_train_samples", "max_val_samples", "max_test_samples"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise BuildingTrainingError(f"--{name.replace('_', '-')} must be positive.")
    for name in [
        "focal_alpha",
        "focal_gamma",
        "focal_tversky_alpha",
        "focal_tversky_beta",
        "focal_tversky_gamma",
    ]:
        value = getattr(args, name)
        if value <= 0:
            raise BuildingTrainingError(f"--{name.replace('_', '-')} must be positive.")


def validate_input_mode(input_mode: str) -> str:
    if input_mode not in INPUT_MODES:
        raise BuildingTrainingError(
            "input_mode must be one of: " + ", ".join(sorted(INPUT_MODES))
        )
    return input_mode


def validate_augment_mode(augment_mode: str) -> str:
    if augment_mode not in AUGMENT_MODES:
        raise BuildingTrainingError(
            "augment_mode must be one of: " + ", ".join(sorted(AUGMENT_MODES))
        )
    return augment_mode


def canonical_augment_mode(augment_mode: str) -> str:
    return "building-safe" if augment_mode == "safe" else augment_mode


def resolve_crop_size(train_mode: str, crop_size: int | None) -> int | None:
    if train_mode == "full1024":
        return None
    if crop_size is not None:
        return crop_size
    if train_mode == "crop512":
        return 512
    if train_mode == "crop608":
        return 608
    raise BuildingTrainingError(f"Unsupported train mode: {train_mode}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BuildingTrainingError("CUDA was requested, but CUDA is not available.")
    return device


def prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if "raw" in {part.lower() for part in output_dir.parts}:
        raise BuildingTrainingError("Refusing to write outputs inside a raw data folder.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise BuildingTrainingError(f"Output path is not a directory: {output_dir}")
    return output_dir


def make_dataset(
    root: Path,
    split_csv: Path,
    image_size: int,
    input_mode: str,
    target_mode: str,
    max_samples: int | None,
    augment_mode: str,
    crop_size: int | None = None,
    rare_building_crop_prob: float = 0.75,
    rare_building_crop_alpha: float | None = None,
) -> torch.utils.data.Dataset:
    base_dataset = XBDPairDataset(
        root=root,
        split_csv=split_csv,
        image_size=image_size,
        target_mode=target_mode,
        augment_mode="none",
        augment_prob=0.0,
    )
    if max_samples is not None:
        base_dataset = Subset(base_dataset, range(min(max_samples, len(base_dataset))))
    return BuildingInputDataset(
        base_dataset,
        input_mode,
        augment_mode=augment_mode,
        crop_size=crop_size,
        rare_building_crop_prob=rare_building_crop_prob,
        rare_building_crop_alpha=rare_building_crop_alpha,
    )


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )


def input_channels(input_mode: str) -> int:
    return 6 if input_mode == "pre-post" else 3


def build_model(
    model_name: str,
    in_channels: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if model_name == "unet":
        return UNet(in_channels=in_channels, num_classes=1, base_channels=32).to(device), "unet"

    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except ImportError as exc:
        raise BuildingTrainingError(
            "segmentation_models_pytorch is required for model "
            f"'{model_name}'. Install requirements.txt or load the correct Rorqual env."
        ) from exc

    smp_specs = {
        "unetplusplus_effb3": (smp.UnetPlusPlus, "efficientnet-b3"),
        "unetplusplus_effb4": (smp.UnetPlusPlus, "efficientnet-b4"),
        "deeplabv3plus_resnet50": (smp.DeepLabV3Plus, "resnet50"),
        "deeplabv3plus_effb3": (smp.DeepLabV3Plus, "efficientnet-b3"),
        "fpn_effb3": (smp.FPN, "efficientnet-b3"),
    }
    if model_name not in smp_specs:
        raise BuildingTrainingError(f"Unsupported model: {model_name}")

    model_class, encoder_name = smp_specs[model_name]
    try:
        model = model_class(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
    except Exception as exc:
        raise BuildingTrainingError(
            f"Could not create SMP model '{model_name}' with encoder "
            f"'{encoder_name}': {exc}"
        ) from exc
    return model.to(device), model_name


def build_loss(args: argparse.Namespace) -> torch.nn.Module:
    if args.loss in {"bce-dice", "dice-bce"}:
        return BCEDiceLoss()
    if args.loss == "focal-dice":
        return FocalDiceLoss(focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma)
    if args.loss == "focal-tversky":
        return FocalTverskyLoss(
            alpha=args.focal_tversky_alpha,
            beta=args.focal_tversky_beta,
            gamma=args.focal_tversky_gamma,
        )
    if args.loss == "bce-dice-boundary":
        return BoundaryAwareLoss(BCEDiceLoss(), args.boundary_loss_weight)
    if args.loss == "focal-tversky-boundary":
        return BoundaryAwareLoss(
            FocalTverskyLoss(
                alpha=args.focal_tversky_alpha,
                beta=args.focal_tversky_beta,
                gamma=args.focal_tversky_gamma,
            ),
            args.boundary_loss_weight,
        )
    raise BuildingTrainingError(f"Unsupported loss: {args.loss}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise BuildingTrainingError(
            f"Expected binary logits with shape Bx1xHxW, got {tuple(logits.shape)}."
        )
    return logits


def targets_to_float(targets: torch.Tensor) -> torch.Tensor:
    return targets.to(dtype=torch.float32).unsqueeze(1)


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def apply_building_augmentation(
    image: torch.Tensor,
    target: torch.Tensor,
    augment_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    augment_mode = canonical_augment_mode(augment_mode)
    image, target = apply_building_geometric_augmentation(image, target)
    image = apply_building_photometric_augmentation(image, augment_mode)
    return image.contiguous(), target.contiguous()


def random_building_crop(
    image: torch.Tensor,
    target: torch.Tensor,
    crop_size: int,
    rare_building_crop_prob: float,
    rare_building_crop_alpha: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    crop = int(crop_size)
    if crop >= height or crop >= width:
        return image, target
    crop_probability = rare_building_crop_prob
    if rare_building_crop_alpha is not None:
        crop_probability = rare_building_crop_alpha / (1.0 + rare_building_crop_alpha)
    building_pixels = (target > 0).nonzero(as_tuple=False)
    if building_pixels.numel() and torch.rand(()) < crop_probability:
        choice = building_pixels[torch.randint(0, building_pixels.shape[0], (1,)).item()]
        center_y, center_x = int(choice[0]), int(choice[1])
        top = min(max(center_y - crop // 2, 0), height - crop)
        left = min(max(center_x - crop // 2, 0), width - crop)
    else:
        top = int(torch.randint(0, height - crop + 1, (1,)).item())
        left = int(torch.randint(0, width - crop + 1, (1,)).item())
    return image[:, top : top + crop, left : left + crop], target[top : top + crop, left : left + crop]


def apply_building_geometric_augmentation(
    image: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(2,))
        target = torch.flip(target, dims=(1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(1,))
        target = torch.flip(target, dims=(0,))

    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        image = torch.rot90(image, k=rotations, dims=(1, 2))
        target = torch.rot90(target, k=rotations, dims=(0, 1))
    return image, target


def apply_building_photometric_augmentation(
    image: torch.Tensor,
    augment_mode: str,
) -> torch.Tensor:
    if augment_mode == "building-strong":
        brightness_range = (0.88, 1.12)
        contrast_range = (0.88, 1.12)
        gamma_range = (0.90, 1.10)
        noise_std = 0.018
        noise_probability = 0.45
        blur_probability = 0.25
    else:
        brightness_range = (0.94, 1.06)
        contrast_range = (0.94, 1.06)
        gamma_range = (0.96, 1.04)
        noise_std = 0.008
        noise_probability = 0.25
        blur_probability = 0.12

    image = apply_per_image_photometric(
        image,
        start_channel=0,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
        gamma_range=gamma_range,
        noise_std=noise_std,
        noise_probability=noise_probability,
        blur_probability=blur_probability,
    )
    if image.shape[0] == 6:
        image = apply_per_image_photometric(
            image,
            start_channel=3,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            gamma_range=gamma_range,
            noise_std=noise_std,
            noise_probability=noise_probability,
            blur_probability=blur_probability,
        )
    return image.clamp(0.0, 1.0)


def uniform_float(bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(torch.empty(()).uniform_(low, high).item())


def apply_per_image_photometric(
    image: torch.Tensor,
    start_channel: int,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
    gamma_range: tuple[float, float],
    noise_std: float,
    noise_probability: float,
    blur_probability: float,
) -> torch.Tensor:
    end_channel = start_channel + 3
    channels = image[start_channel:end_channel]
    brightness = uniform_float(brightness_range)
    contrast = uniform_float(contrast_range)
    gamma = uniform_float(gamma_range)

    mean = channels.mean(dim=(1, 2), keepdim=True)
    channels = (channels - mean) * contrast + mean
    channels = channels * brightness
    channels = channels.clamp(0.0, 1.0).pow(gamma)

    if torch.rand(()) < noise_probability:
        channels = channels + torch.randn_like(channels) * noise_std
    if torch.rand(()) < blur_probability:
        channels = F.avg_pool2d(
            channels.unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(0)

    image = image.clone()
    image[start_channel:end_channel] = channels.clamp(0.0, 1.0)
    return image


def make_train_sampler(
    dataset: torch.utils.data.Dataset,
    sampler_mode: str,
    alpha: float,
) -> WeightedRandomSampler | None:
    if sampler_mode == "none":
        return None
    if sampler_mode != "building-sqrt":
        raise BuildingTrainingError(f"Unsupported sampler: {sampler_mode}")

    rows = extract_sample_rows(dataset)
    building_ratios = rows.apply(sample_building_ratio, axis=1).to_numpy(dtype="float64")
    ratio_tensor = torch.as_tensor(building_ratios, dtype=torch.double).clamp_min(0.0)
    weights = (1.0 + float(alpha) * torch.sqrt(ratio_tensor)).numpy()
    if not torch.isfinite(torch.as_tensor(weights)).all() or (weights <= 0).any():
        raise BuildingTrainingError("Sampler weights must be finite and positive.")

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def extract_sample_rows(dataset: torch.utils.data.Dataset):
    if isinstance(dataset, BuildingInputDataset):
        return extract_sample_rows(dataset.base_dataset)
    if isinstance(dataset, Subset):
        base_rows = extract_sample_rows(dataset.dataset)
        return base_rows.iloc[list(dataset.indices)].reset_index(drop=True)
    if isinstance(dataset, XBDPairDataset):
        return dataset.samples.reset_index(drop=True)
    raise BuildingTrainingError("Unsupported dataset type for sampler.")


def sample_building_ratio(row) -> float:
    for column in [
        "building_ratio",
        "target_nonzero_ratio",
        "nonzero_target_ratio",
        "nonzero_ratio",
    ]:
        if column in row.index and not is_missing(row[column]):
            try:
                return float(row[column])
            except (TypeError, ValueError):
                pass

    if {"target_value_counts", "target_total_pixels"}.issubset(row.index):
        try:
            counts = json.loads(str(row["target_value_counts"]))
            total = int(float(row["target_total_pixels"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        if total <= 0 or not isinstance(counts, dict):
            return 0.0
        building_pixels = 0
        for key, value in counts.items():
            try:
                target_class = int(float(str(key)))
                count = int(value)
            except (TypeError, ValueError):
                continue
            if target_class > 0:
                building_pixels += count
        return building_pixels / total
    return 0.0


def is_missing(value: object) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return value is None


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader)
    epoch_started_at = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_float = targets_to_float(targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = normalize_logits(model(images))
            loss = criterion(logits, target_float)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

        should_log = (
            log_every > 0
            and (
                batch_index == 1
                or batch_index % log_every == 0
                or batch_index == total_batches
            )
        )
        if should_log:
            elapsed_minutes = (time.time() - epoch_started_at) / 60.0
            running_loss = total_loss / max(total_samples, 1)
            print(
                f"Epoch {epoch:03d} batch {batch_index}/{total_batches} | "
                f"loss {float(loss.detach().item()):.4f} | "
                f"running {running_loss:.4f} | "
                f"elapsed {elapsed_minutes:.1f} min",
                flush=True,
            )
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    boundary_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_float = targets_to_float(targets)

        with autocast_context(use_amp):
            logits = normalize_logits(model(images))
            loss = criterion(logits, target_float)

        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = probabilities >= 0.5
        target_bool = targets > 0
        pred_boundary = soft_boundary(predictions.float().unsqueeze(1)) > 0
        target_boundary = soft_boundary(target_float) > 0

        counts["tp"] += int(torch.count_nonzero(predictions & target_bool).item())
        counts["tn"] += int(torch.count_nonzero((~predictions) & (~target_bool)).item())
        counts["fp"] += int(torch.count_nonzero(predictions & (~target_bool)).item())
        counts["fn"] += int(torch.count_nonzero((~predictions) & target_bool).item())
        boundary_counts["tp"] += int(torch.count_nonzero(pred_boundary & target_boundary).item())
        boundary_counts["tn"] += int(torch.count_nonzero((~pred_boundary) & (~target_boundary)).item())
        boundary_counts["fp"] += int(torch.count_nonzero(pred_boundary & (~target_boundary)).item())
        boundary_counts["fn"] += int(torch.count_nonzero((~pred_boundary) & target_boundary).item())

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    metrics = metrics_from_counts(counts)
    boundary_metrics = metrics_from_counts(boundary_counts)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["confusion_counts"] = counts
    metrics["boundary_f1"] = boundary_metrics["building_f1"]
    metrics["boundary_precision"] = boundary_metrics["building_precision"]
    metrics["boundary_recall"] = boundary_metrics["building_recall"]
    return metrics


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = float(counts["tp"])
    tn = float(counts["tn"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    total = tp + tn + fp + fn

    background_iou = safe_divide(tn, tn + fp + fn)
    building_iou = safe_divide(tp, tp + fp + fn)
    building_precision = safe_divide(tp, tp + fp)
    building_recall = safe_divide(tp, tp + fn)
    building_f1 = safe_divide(
        2.0 * building_precision * building_recall,
        building_precision + building_recall,
    )

    return {
        "pixel_accuracy": safe_divide(tp + tn, total),
        "mean_iou": (background_iou + building_iou) / 2.0,
        "background_iou": background_iou,
        "building_iou": building_iou,
        "building_precision": building_precision,
        "building_recall": building_recall,
        "building_f1": building_f1,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, object],
    args: argparse.Namespace,
    actual_model: str,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(args),
        "actual_model": actual_model,
    }
    torch.save(checkpoint, path)


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    if not path.exists():
        raise BuildingTrainingError(f"Resume checkpoint does not exist: {path}")
    checkpoint = load_checkpoint_file(path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = int(checkpoint.get("epoch", 0))
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (RuntimeError, ValueError, KeyError) as exc:
                print(
                    "WARNING: Could not load optimizer_state_dict; "
                    f"resuming model weights only. Details: {exc}",
                    file=sys.stderr,
                )
    else:
        state_dict = checkpoint
        epoch = 0
    model.load_state_dict(clean_state_dict(state_dict))
    return epoch + 1


def clean_state_dict(state_dict: object) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        raise BuildingTrainingError("Checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned[str(key).removeprefix("module.")] = value
    if not cleaned:
        raise BuildingTrainingError("Checkpoint contains no tensor weights.")
    return cleaned


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def write_summary_csv(
    path: Path,
    args: argparse.Namespace,
    actual_model: str,
    best_val_metrics: dict[str, object],
    test_metrics: dict[str, object] | None,
) -> None:
    fields = [
        "experiment",
        "model_requested",
        "model_actual",
        "input_mode",
        "train_mode",
        "crop_size",
        "image_size",
        "batch_size",
        "epochs",
        "lr",
        "loss",
        "augment_mode",
        "sampler",
        "sampler_alpha",
        "rare_building_crop_prob",
        "rare_building_crop_alpha",
        "boundary_loss_weight",
        "val_building_iou",
        "val_building_f1",
        "val_boundary_f1",
        "test_building_iou",
        "test_building_f1",
        "test_boundary_f1",
        "test_mean_iou",
    ]
    row = {
        "experiment": args.output_dir.name,
        "model_requested": args.model,
        "model_actual": actual_model,
        "input_mode": args.input_mode,
        "train_mode": args.train_mode,
        "crop_size": resolve_crop_size(args.train_mode, args.crop_size),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "augment_mode": args.augment_mode,
        "sampler": args.sampler,
        "sampler_alpha": args.sampler_alpha,
        "rare_building_crop_prob": args.rare_building_crop_prob,
        "rare_building_crop_alpha": args.rare_building_crop_alpha,
        "boundary_loss_weight": args.boundary_loss_weight,
        "val_building_iou": best_val_metrics.get("building_iou"),
        "val_building_f1": best_val_metrics.get("building_f1"),
        "val_boundary_f1": best_val_metrics.get("boundary_f1"),
        "test_building_iou": None if test_metrics is None else test_metrics.get("building_iou"),
        "test_building_f1": None if test_metrics is None else test_metrics.get("building_f1"),
        "test_boundary_f1": None if test_metrics is None else test_metrics.get("boundary_f1"),
        "test_mean_iou": None if test_metrics is None else test_metrics.get("mean_iou"),
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def metric_or_default(path: Path, metric: str, default: float) -> float:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return float(payload.get(metric, default))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return default


def print_epoch(epoch: int, train_loss: float, val_metrics: dict[str, object]) -> None:
    print(
        f"Epoch {epoch:03d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_metrics['loss']:.4f} | "
        f"pixel acc {val_metrics['pixel_accuracy']:.4f} | "
        f"mean IoU {val_metrics['mean_iou']:.4f} | "
        f"building IoU {val_metrics['building_iou']:.4f} | "
        f"building F1 {val_metrics['building_f1']:.4f}"
    )


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    args.augment_mode = canonical_augment_mode(args.augment_mode)
    output_dir = prepare_output_dir(args.output_dir)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    crop_size = resolve_crop_size(args.train_mode, args.crop_size)

    train_dataset = make_dataset(
        args.root,
        args.train_csv,
        args.image_size,
        args.input_mode,
        args.target_mode,
        args.max_train_samples,
        augment_mode=args.augment_mode,
        crop_size=crop_size,
        rare_building_crop_prob=args.rare_building_crop_prob,
        rare_building_crop_alpha=args.rare_building_crop_alpha,
    )
    val_dataset = make_dataset(
        args.root,
        args.val_csv,
        args.image_size,
        args.input_mode,
        args.target_mode,
        args.max_val_samples,
        augment_mode="none",
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise BuildingTrainingError("Train and validation datasets must be non-empty.")

    train_sampler = make_train_sampler(train_dataset, args.sampler, args.sampler_alpha)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=train_sampler is None,
        device=device,
        sampler=train_sampler,
        drop_last=args.drop_last_train,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
        drop_last=False,
    )

    model, actual_model = build_model(args.model, input_channels(args.input_mode), device)
    criterion = build_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    if args.resume_checkpoint is not None:
        start_epoch = load_resume_checkpoint(args.resume_checkpoint, model, optimizer, device)

    best_metric = metric_or_default(output_dir / "best_val_metrics.json", "building_iou", -1.0)
    history: list[dict[str, object]] = []
    history_path = output_dir / "metrics_history.json"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as file:
                existing_history = json.load(file)
            if isinstance(existing_history, list):
                history = existing_history
            elif isinstance(existing_history, dict) and isinstance(
                existing_history.get("history"), list
            ):
                history = existing_history["history"]
        except (OSError, json.JSONDecodeError):
            history = []
    if args.resume_checkpoint is not None:
        history = [
            item
            for item in history
            if isinstance(item, dict) and int(item.get("epoch", 0) or 0) < start_epoch
        ]

    print("CrisisMap AI - Building Segmentation Training")
    print("=" * 49)
    print(f"Device: {device}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"Model requested: {args.model}")
    print(f"Model actual: {actual_model}")
    print(f"Input mode: {args.input_mode}")
    print(f"Target mode: {args.target_mode}")
    print(f"Train mode: {args.train_mode}")
    print(f"Crop size: {crop_size}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Parameters: {parameter_count(model):,}")
    print(f"Loss: {args.loss}")
    print(f"Augment mode: {args.augment_mode}")
    print(f"Sampler: {args.sampler}")
    print(f"Sampler alpha: {args.sampler_alpha}")
    print(f"Rare building crop prob: {args.rare_building_crop_prob}")
    print(f"Rare building crop alpha: {args.rare_building_crop_alpha}")
    print(f"Boundary loss weight: {args.boundary_loss_weight}")
    print(f"Drop last train batch: {args.drop_last_train}")
    print(f"Log every: {args.log_every} batches")
    if args.resume_checkpoint is not None:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
        print(f"Starting epoch: {start_epoch}")

    started_at = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
            args.log_every,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_background_iou": val_metrics["background_iou"],
            "val_building_iou": val_metrics["building_iou"],
            "val_building_precision": val_metrics["building_precision"],
            "val_building_recall": val_metrics["building_recall"],
            "val_building_f1": val_metrics["building_f1"],
            "val_boundary_f1": val_metrics.get("boundary_f1"),
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_metrics)
        print_epoch(epoch, train_loss, val_metrics)

        if float(val_metrics["building_iou"]) > best_metric:
            best_metric = float(val_metrics["building_iou"])
            save_checkpoint(
                output_dir / "best_building.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                args,
                actual_model,
            )
            write_json(output_dir / "best_val_metrics.json", val_metrics)

        save_checkpoint(
            output_dir / "last_building.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            args,
            actual_model,
        )
        write_json(history_path, history)

    best_val_metrics = read_metrics(output_dir / "best_val_metrics.json")
    test_metrics = None
    if args.test_csv is not None:
        test_dataset = make_dataset(
            args.root,
            args.test_csv,
            args.image_size,
            args.input_mode,
            args.target_mode,
            args.max_test_samples,
            augment_mode="none",
        )
        test_loader = make_loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            device=device,
            drop_last=False,
        )
        load_resume_checkpoint(output_dir / "best_building.pt", model, None, device)
        test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
        write_json(output_dir / "test_metrics.json", test_metrics)

    write_summary_csv(
        output_dir / "summary.csv",
        args,
        actual_model,
        best_val_metrics,
        test_metrics,
    )
    print(f"Training complete in {(time.time() - started_at) / 60.0:.2f} minutes.")
    print(f"Best validation building IoU: {best_metric:.4f}")
    if test_metrics is not None:
        print(f"Test building IoU: {test_metrics['building_iou']:.4f}")
        print(f"Test building F1: {test_metrics['building_f1']:.4f}")
    print(f"Outputs saved to: {output_dir}")


def read_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (
        BuildingTrainingError,
        XBDDatasetError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

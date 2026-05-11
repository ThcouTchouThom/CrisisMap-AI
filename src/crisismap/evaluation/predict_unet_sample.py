"""Run U-Net inference on one xBD sample and visualize the result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crisismap.data.xbd_dataset import XBDDatasetError, XBDPairDataset  # noqa: E402
from crisismap.models.unet import UNet  # noqa: E402


CLASS_DEFINITIONS = {
    "3-class": {
        0: ("background", (0.05, 0.05, 0.05)),
        1: ("no damage", (0.10, 0.58, 0.24)),
        2: ("damaged", (0.88, 0.09, 0.11)),
    },
    "5-class": {
        0: ("background", (0.05, 0.05, 0.05)),
        1: ("no damage", (0.10, 0.58, 0.24)),
        2: ("minor damage", (1.00, 0.82, 0.14)),
        3: ("major damage", (1.00, 0.46, 0.08)),
        4: ("destroyed", (0.82, 0.05, 0.08)),
    },
}


class PredictionError(Exception):
    """Raised when inference or visualization cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict and visualize one xBD sample.")
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
        help="Path to val_pairs.csv or test_pairs.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to best_unet.pt or last_unet.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the PNG figure.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--pair-id", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--target-mode",
        choices=sorted(CLASS_DEFINITIONS),
        default="3-class",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device string, for example cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise PredictionError("--image-size must be a positive integer.")
    if args.base_channels <= 0:
        raise PredictionError("--base-channels must be a positive integer.")
    if args.output is not None and args.output.suffix.lower() != ".png":
        raise PredictionError("--output should be a .png path.")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise PredictionError("CUDA device was requested, but CUDA is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(args: argparse.Namespace) -> XBDPairDataset:
    return XBDPairDataset(
        root=args.root,
        split_csv=args.split_csv,
        image_size=args.image_size,
        target_mode=args.target_mode,
    )


def select_sample_index(
    dataset: XBDPairDataset,
    pair_id: str | None,
    sample_index: int,
) -> int:
    if pair_id:
        pair_id = pair_id.strip()
        matches = dataset.samples.index[
            dataset.samples["pair_id"].astype(str).str.lower() == pair_id.lower()
        ].tolist()
        if not matches:
            raise PredictionError(f"Pair id '{pair_id}' was not found in the split CSV.")
        return int(matches[0])

    if sample_index < 0:
        sample_index = len(dataset) + sample_index
    if sample_index < 0 or sample_index >= len(dataset):
        raise PredictionError(
            f"--sample-index {sample_index} is outside dataset length {len(dataset)}."
        )
    return sample_index


def load_model(args: argparse.Namespace, device: torch.device) -> UNet:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise PredictionError(f"Checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise PredictionError(f"Checkpoint path is not a file: {checkpoint_path}")

    num_classes = 3 if args.target_mode == "3-class" else 5
    model = UNet(
        in_channels=6,
        num_classes=num_classes,
        base_channels=args.base_channels,
    ).to(device)

    try:
        checkpoint = load_checkpoint_file(checkpoint_path, device)
    except OSError as exc:
        raise PredictionError(f"Could not read checkpoint '{checkpoint_path}': {exc}") from exc
    except RuntimeError as exc:
        raise PredictionError(f"Could not load checkpoint '{checkpoint_path}': {exc}") from exc

    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise PredictionError(
            "Checkpoint weights do not match this UNet configuration. "
            "Check --target-mode and --base-channels."
        ) from exc

    model.eval()
    return model


def load_checkpoint_file(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise PredictionError("Checkpoint is not a state_dict or a model checkpoint dict.")

    cleaned = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        cleaned[key.removeprefix("module.")] = value
    if not cleaned:
        raise PredictionError("Checkpoint does not contain any tensor weights.")
    return cleaned


@torch.no_grad()
def run_prediction(
    model: UNet,
    sample: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    image = sample["image"].unsqueeze(0).to(device)
    logits = model(image)
    return torch.argmax(logits, dim=1).squeeze(0).cpu()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return np.clip(array, 0.0, 1.0)


def colorize_mask(mask: np.ndarray, target_mode: str) -> np.ndarray:
    definitions = CLASS_DEFINITIONS[target_mode]
    color_image = np.zeros((*mask.shape, 3), dtype=np.float32)
    for class_id, (_, color) in definitions.items():
        color_image[mask == class_id] = color
    return color_image


def overlay_mask(image: np.ndarray, mask: np.ndarray, target_mode: str) -> np.ndarray:
    color_mask = colorize_mask(mask, target_mode)
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[mask == 1] = 0.35
    alpha[mask >= 2] = 0.55
    alpha = alpha[:, :, None]
    return image * (1.0 - alpha) + color_mask * alpha


def legend_handles(target_mode: str):
    handles = []
    for class_id, (label, color) in CLASS_DEFINITIONS[target_mode].items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=10,
                label=f"{class_id} = {label}",
            )
        )
    return handles


def build_figure(
    pair_id: str,
    image_tensor: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    target_mode: str,
) -> plt.Figure:
    pre_image = tensor_to_image(image_tensor[:3])
    post_image = tensor_to_image(image_tensor[3:])
    target_np = target.detach().cpu().numpy()
    pred_np = prediction.detach().cpu().numpy()

    target_color = colorize_mask(target_np, target_mode)
    pred_color = colorize_mask(pred_np, target_mode)
    target_overlay = overlay_mask(post_image, target_np, target_mode)
    pred_overlay = overlay_mask(post_image, pred_np, target_mode)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"U-Net prediction: {pair_id} ({target_mode})", fontsize=14)

    panels = [
        (axes[0, 0], pre_image, "Pre-disaster image"),
        (axes[0, 1], post_image, "Post-disaster image"),
        (axes[0, 2], target_color, "Ground truth mask"),
        (axes[1, 0], pred_color, "Predicted mask"),
        (axes[1, 1], target_overlay, "Ground truth overlay"),
        (axes[1, 2], pred_overlay, "Prediction overlay"),
    ]
    for axis, image, title in panels:
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    fig.legend(
        handles=legend_handles(target_mode),
        loc="lower center",
        ncol=3 if target_mode == "3-class" else 5,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.12, left=0.02, right=0.98, top=0.91, wspace=0.04)
    return fig


def save_or_show(fig: plt.Figure, output: Path | None) -> None:
    if output is None:
        plt.show()
        return

    output = output.expanduser().resolve()
    if not output.parent.exists():
        raise PredictionError(f"Output parent folder does not exist: {output.parent}")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {output}")


def print_sample_info(sample: dict[str, object], prediction: torch.Tensor) -> None:
    target = sample["target"]
    print(f"Pair id: {sample['pair_id']}")
    print(f"Input tensor shape: {tuple(sample['image'].shape)}")
    print(f"Ground truth unique values: {torch.unique(target).cpu().tolist()}")
    print(f"Prediction unique values: {torch.unique(prediction).cpu().tolist()}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        device = resolve_device(args.device)
        dataset = load_dataset(args)
        sample_index = select_sample_index(dataset, args.pair_id, args.sample_index)
        sample = dataset[sample_index]
        model = load_model(args, device)
        prediction = run_prediction(model, sample, device)

        print(f"Device: {device}")
        print_sample_info(sample, prediction)
        fig = build_figure(
            pair_id=str(sample["pair_id"]),
            image_tensor=sample["image"],
            target=sample["target"],
            prediction=prediction,
            target_mode=args.target_mode,
        )
        save_or_show(fig, args.output)
    except (PredictionError, XBDDatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

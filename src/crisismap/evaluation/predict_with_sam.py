"""Inférence U-Net + raffinement SAM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crisismap.models.unet import UNet
from crisismap.models.sam_refiner import SAMRefiner


PALETTE = {
    0: (0, 0, 0),        # background
    1: (0, 255, 0),      # no-damage
    2: (255, 0, 0),      # damaged
}


def load_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in PALETTE.items():
        rgb[mask == cls] = color
    return rgb


def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Charge U-Net ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = 3 if ckpt["config"].get("target_mode") == "3-class" else 5
    base_channels = ckpt["config"].get("base_channels", 32)
    model = UNet(in_channels=6, num_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # --- Charge images ---
    pre = load_image(args.pre_image, args.image_size).to(device)
    post = load_image(args.post_image, args.image_size).to(device)
    stacked = torch.cat([pre, post], dim=0).unsqueeze(0)  # (1, 6, H, W)

    # --- Prédiction U-Net ---
    with torch.no_grad():
        logits = model(stacked)
    unet_pred = logits.argmax(dim=1)  # (1, H, W)

    # --- Raffinement SAM ---
    if args.sam_checkpoint:
        refiner = SAMRefiner(
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=device,
            building_classes=set(range(1, num_classes)),
        )
        final_pred = refiner.refine_batch(
            post_images=post.unsqueeze(0),
            unet_preds=unet_pred,
        ).squeeze(0)
    else:
        final_pred = unet_pred.squeeze(0)

    # --- Sauvegarde ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgb = mask_to_rgb(final_pred.cpu().numpy())
    Image.fromarray(rgb).save(args.output)
    print(f"Masque sauvegardé : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pre-image", type=Path, required=True)
    parser.add_argument("--post-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.png"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model-type", default="vit_b")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()

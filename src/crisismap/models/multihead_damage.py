"""Axis 3 multi-head building localization and damage prediction models.

The models are separate from the existing U-Net/Siamese damage pipelines. They
use a shared Siamese encoder over pre/post RGB images, fuse temporal features,
and expose two heads:

* building_logits: binary building localization
* damage_logits: multiclass or binary damage logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisismap.models.damage_model_factory import create_damage_model
from crisismap.models.xview2_strong_baseline import (
    DecoderBlock,
    FusionProjection,
    ResNetEncoder,
    TimmEncoder,
    XView2StrongBaselineError,
)


class MultiHeadDamageError(RuntimeError):
    """Raised when a multi-head damage model cannot be created."""


class MultiHeadSiameseUNet(nn.Module):
    """Shared encoder U-Net with building and damage heads."""

    def __init__(
        self,
        encoder: str = "resnet34",
        fusion_mode: str = "shared",
        attention: bool = True,
        damage_channels: int = 3,
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 64),
    ) -> None:
        super().__init__()
        if damage_channels <= 0:
            raise MultiHeadDamageError("damage_channels must be positive.")
        self.encoder_name = encoder
        self.fusion_mode = fusion_mode
        self.attention_enabled = attention
        self.damage_channels = damage_channels

        try:
            if encoder in {"resnet34", "resnet50"}:
                self.encoder = ResNetEncoder(encoder)
            elif encoder == "efficientnet_b3":
                self.encoder = TimmEncoder("efficientnet_b3")
            else:
                raise MultiHeadDamageError(
                    "Unsupported encoder. Expected resnet34, resnet50, or efficientnet_b3."
                )
        except XView2StrongBaselineError as exc:
            raise MultiHeadDamageError(str(exc)) from exc

        channels = self.encoder.channels
        self.fusion = nn.ModuleList(
            FusionProjection(channel, mode=fusion_mode, attention=attention)
            for channel in channels
        )
        self.dec4 = DecoderBlock(channels[4], channels[3], decoder_channels[0])
        self.dec3 = DecoderBlock(decoder_channels[0], channels[2], decoder_channels[1])
        self.dec2 = DecoderBlock(decoder_channels[1], channels[1], decoder_channels[2])
        self.dec1 = DecoderBlock(decoder_channels[2], channels[0], decoder_channels[3])
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )
        self.building_head = nn.Conv2d(decoder_channels[3], 1, kernel_size=1)
        self.damage_head = nn.Conv2d(decoder_channels[3], damage_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected input shape [N, 6, H, W], got {tuple(x.shape)}.")
        pre_features = self.encoder(x[:, :3])
        post_features = self.encoder(x[:, 3:])
        fused = [
            fusion(pre, post)
            for fusion, pre, post in zip(self.fusion, pre_features, post_features)
        ]
        output = fused[-1]
        output = self.dec4(output, fused[3])
        output = self.dec3(output, fused[2])
        output = self.dec2(output, fused[1])
        output = self.dec1(output, fused[0])
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)
        output = self.refine(output)
        return {
            "building_logits": self.building_head(output),
            "damage_logits": self.damage_head(output),
        }


class ExistingSiameseAttentionWithBuildingProxy(nn.Module):
    """Reuse the current Siamese attention damage model with a proxy building head."""

    def __init__(self, damage_channels: int = 3) -> None:
        super().__init__()
        if damage_channels != 3:
            raise MultiHeadDamageError(
                "current_siamese_attention proxy currently supports only 3 damage classes."
            )
        self.damage_model = create_damage_model(
            "siamese_unet_attention",
            num_classes=3,
            in_channels=6,
        )
        self.building_proxy = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        damage_logits = self.damage_model(x)
        return {
            "building_logits": self.building_proxy(damage_logits),
            "damage_logits": damage_logits,
        }


MODEL_SPECS = {
    "mh_resnet34_attention": {
        "encoder": "resnet34",
        "fusion_mode": "shared",
        "attention": True,
    },
    "mh_resnet50_attention": {
        "encoder": "resnet50",
        "fusion_mode": "shared",
        "attention": True,
    },
    "mh_effb3_attention": {
        "encoder": "efficientnet_b3",
        "fusion_mode": "shared",
        "attention": True,
    },
    "mh_resnet34_abs_signed": {
        "encoder": "resnet34",
        "fusion_mode": "abs_signed",
        "attention": False,
    },
    "mh_resnet50_abs_signed": {
        "encoder": "resnet50",
        "fusion_mode": "abs_signed",
        "attention": False,
    },
}


def supported_multihead_damage_models() -> list[str]:
    return sorted(list(MODEL_SPECS) + ["mh_current_siamese_attention_proxy"])


def create_multihead_damage_model(
    model_name: str,
    damage_channels: int = 3,
) -> nn.Module:
    """Create an Axis 3 multi-head model."""

    if model_name == "mh_current_siamese_attention_proxy":
        return ExistingSiameseAttentionWithBuildingProxy(damage_channels=damage_channels)
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise MultiHeadDamageError(
            f"Unsupported multi-head model '{model_name}'. Supported: "
            + ", ".join(supported_multihead_damage_models())
        )
    return MultiHeadSiameseUNet(damage_channels=damage_channels, **spec)

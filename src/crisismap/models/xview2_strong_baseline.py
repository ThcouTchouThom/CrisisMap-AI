"""xView2 strong-baseline-inspired shared-encoder ResNet U-Net.

The model is intentionally self-contained and does not alter the existing
damage U-Net/Siamese code paths. It applies the same ResNet encoder to the
pre-disaster and post-disaster images, fuses every feature level with:

concat(pre_feat, post_feat, abs(post_feat - pre_feat))

and decodes the fused pyramid into two heads:

* building_logits: one binary localization channel
* damage_logits: either multiclass damage channels or one binary damage channel
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class XView2StrongBaselineError(Exception):
    """Raised when the strong-baseline model cannot be created."""


def _load_resnet(encoder: str) -> nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise XView2StrongBaselineError(
            "torchvision is required for xView2 strong-baseline models."
        ) from exc

    if encoder == "resnet34":
        try:
            return models.resnet34(weights=None)
        except TypeError:
            return models.resnet34(pretrained=False)
    if encoder == "resnet50":
        try:
            return models.resnet50(weights=None)
        except TypeError:
            return models.resnet50(pretrained=False)
    raise XView2StrongBaselineError(
        "Unsupported encoder. Expected one of: resnet34, resnet50."
    )


class ResNetEncoder(nn.Module):
    """Expose a ResNet feature pyramid without classification layers."""

    CHANNELS = {
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
    }

    def __init__(self, encoder: str) -> None:
        super().__init__()
        if encoder not in self.CHANNELS:
            raise XView2StrongBaselineError(
                f"Unsupported encoder '{encoder}'. Supported: {', '.join(self.CHANNELS)}"
            )
        backbone = _load_resnet(encoder)
        self.channels = self.CHANNELS[encoder]
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stem = self.stem(x)
        pooled = self.maxpool(stem)
        layer1 = self.layer1(pooled)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [stem, layer1, layer2, layer3, layer4]


class TimmEncoder(nn.Module):
    """timm feature pyramid encoder used for EfficientNet-B3 variants."""

    def __init__(self, name: str = "efficientnet_b3") -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise XView2StrongBaselineError(
                "timm is required for EfficientNet strong-baseline variants. "
                "Install timm==1.0.27."
            ) from exc
        self.encoder = timm.create_model(
            name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=3,
        )
        self.channels = list(self.encoder.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.encoder(x))


class ConvNormAct(nn.Module):
    """Small convolutional block used by the fusion projections and decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention for fused pre/post features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class FusionProjection(nn.Module):
    """Project fused pre/post/change features to a compact feature tensor."""

    MODE_FACTORS = {
        "shared": 3,
        "absdiff": 2,
        "abs_signed": 4,
        "abs_signed_product": 5,
    }

    def __init__(self, channels: int, mode: str = "shared", attention: bool = False) -> None:
        super().__init__()
        if mode not in self.MODE_FACTORS:
            raise XView2StrongBaselineError(
                f"Unsupported fusion mode '{mode}'. Supported: {', '.join(self.MODE_FACTORS)}"
            )
        self.mode = mode
        self.proj = nn.Sequential(
            nn.Conv2d(channels * self.MODE_FACTORS[mode], channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SqueezeExcitation(channels) if attention else nn.Identity()

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        signed_change = post_feat - pre_feat
        abs_change = torch.abs(signed_change)
        if self.mode == "shared":
            features = [pre_feat, post_feat, abs_change]
        elif self.mode == "absdiff":
            features = [post_feat, abs_change]
        elif self.mode == "abs_signed":
            features = [pre_feat, post_feat, abs_change, signed_change]
        elif self.mode == "abs_signed_product":
            features = [pre_feat, post_feat, abs_change, signed_change, pre_feat * post_feat]
        else:
            raise XView2StrongBaselineError(f"Unsupported fusion mode: {self.mode}")
        fused = torch.cat(features, dim=1)
        return self.attention(self.proj(fused))


class DecoderBlock(nn.Module):
    """Upsample decoder features and merge with the corresponding skip level."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvNormAct(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class XView2StrongBaselineUNet(nn.Module):
    """Shared ResNet encoder U-Net with building and damage heads."""

    def __init__(
        self,
        encoder: str = "resnet34",
        damage_channels: int = 3,
        fusion_mode: str = "shared",
        attention: bool = False,
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 64),
    ) -> None:
        super().__init__()
        if damage_channels <= 0:
            raise XView2StrongBaselineError("damage_channels must be positive.")
        self.encoder_name = encoder
        self.damage_channels = damage_channels
        self.fusion_mode = fusion_mode
        self.attention_enabled = attention
        if encoder in {"resnet34", "resnet50"}:
            self.encoder = ResNetEncoder(encoder)
        elif encoder == "efficientnet_b3":
            self.encoder = TimmEncoder("efficientnet_b3")
        else:
            raise XView2StrongBaselineError(
                "Unsupported encoder. Expected resnet34, resnet50, or efficientnet_b3."
            )
        channels = self.encoder.channels
        self.fusion = nn.ModuleList(
            FusionProjection(channel, mode=fusion_mode, attention=attention) for channel in channels
        )

        self.dec4 = DecoderBlock(channels[4], channels[3], decoder_channels[0])
        self.dec3 = DecoderBlock(decoder_channels[0], channels[2], decoder_channels[1])
        self.dec2 = DecoderBlock(decoder_channels[1], channels[1], decoder_channels[2])
        self.dec1 = DecoderBlock(decoder_channels[2], channels[0], decoder_channels[3])
        self.refine = ConvNormAct(decoder_channels[3], decoder_channels[3])
        self.building_head = nn.Conv2d(decoder_channels[3], 1, kernel_size=1)
        self.damage_head = nn.Conv2d(decoder_channels[3], damage_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected input shape [N, 6, H, W], got {tuple(x.shape)}.")
        pre = x[:, :3]
        post = x[:, 3:]
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        fused = [
            fusion_block(pre_feat, post_feat)
            for fusion_block, pre_feat, post_feat in zip(
                self.fusion,
                pre_features,
                post_features,
            )
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


def create_xview2_strong_baseline_model(
    model_name: str,
    damage_channels: int,
) -> XView2StrongBaselineUNet:
    """Create a configured strong-baseline-inspired model."""

    mapping = {
        "resnet34_unet_shared": {
            "encoder": "resnet34",
            "fusion_mode": "shared",
            "attention": False,
        },
        "resnet50_unet_shared": {
            "encoder": "resnet50",
            "fusion_mode": "shared",
            "attention": False,
        },
        "efficientnet_b3_unet_shared": {
            "encoder": "efficientnet_b3",
            "fusion_mode": "shared",
            "attention": False,
        },
        "resnet34_unet_absdiff": {
            "encoder": "resnet34",
            "fusion_mode": "absdiff",
            "attention": False,
        },
        "resnet34_unet_abs_signed": {
            "encoder": "resnet34",
            "fusion_mode": "abs_signed",
            "attention": False,
        },
        "resnet34_unet_abs_signed_product": {
            "encoder": "resnet34",
            "fusion_mode": "abs_signed_product",
            "attention": False,
        },
        "resnet34_unet_attention": {
            "encoder": "resnet34",
            "fusion_mode": "shared",
            "attention": True,
        },
        "resnet50_unet_attention": {
            "encoder": "resnet50",
            "fusion_mode": "shared",
            "attention": True,
        },
        "resnet34_unet_abs_signed_attention": {
            "encoder": "resnet34",
            "fusion_mode": "abs_signed",
            "attention": True,
        },
    }
    if model_name not in mapping:
        raise XView2StrongBaselineError(
            f"Unsupported model '{model_name}'. Supported: {', '.join(sorted(mapping))}"
        )
    return XView2StrongBaselineUNet(
        damage_channels=damage_channels,
        **mapping[model_name],
    )


def supported_xview2_strong_baseline_models() -> list[str]:
    return [
        "efficientnet_b3_unet_shared",
        "resnet34_unet_abs_signed",
        "resnet34_unet_abs_signed_attention",
        "resnet34_unet_abs_signed_product",
        "resnet34_unet_absdiff",
        "resnet34_unet_shared",
        "resnet50_unet_shared",
        "resnet34_unet_attention",
        "resnet50_unet_attention",
    ]

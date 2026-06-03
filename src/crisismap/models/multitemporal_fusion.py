"""Multi-temporal fusion damage segmentation models.

These models are inspired by multi-temporal fusion approaches for building
damage assessment: pre and post images are processed independently, features
are fused before the segmentation decoder, and the network predicts both
building localization and damage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisismap.models.damage_model_factory import create_damage_model


class MultiTemporalFusionError(RuntimeError):
    """Raised when a multi-temporal fusion model cannot be created."""


class ConvBNReLU(nn.Module):
    """Convolution, BatchNorm, ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelAttention(nn.Module):
    """Squeeze-excitation style attention for fused feature maps."""

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


def _load_resnet(name: str) -> nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise MultiTemporalFusionError("torchvision is required for ResNet MTF models.") from exc
    if name == "resnet34":
        factory = models.resnet34
    elif name == "resnet50":
        factory = models.resnet50
    else:
        raise MultiTemporalFusionError(f"Unsupported ResNet encoder: {name}")
    try:
        return factory(weights=None)
    except TypeError:
        return factory(pretrained=False)


class ResNetFeatureEncoder(nn.Module):
    """ResNet feature pyramid encoder."""

    CHANNELS_BY_NAME = {
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
    }

    def __init__(self, name: str = "resnet50", in_channels: int = 3) -> None:
        super().__init__()
        if name not in self.CHANNELS_BY_NAME:
            raise MultiTemporalFusionError(f"Unsupported ResNet encoder: {name}")
        backbone = _load_resnet(name)
        self.channels = self.CHANNELS_BY_NAME[name]
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = new_conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
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


class TimmFeatureEncoder(nn.Module):
    """timm features_only encoder for EfficientNet-B3."""

    def __init__(self, name: str = "efficientnet_b3", in_channels: int = 3) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise MultiTemporalFusionError(
                "timm is required for EfficientNet MTF models. Install timm==1.0.27."
            ) from exc
        self.encoder = timm.create_model(
            name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=in_channels,
        )
        self.channels = list(self.encoder.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.encoder(x))


class FeatureFusion(nn.Module):
    """Fuse pre/post feature pyramids with explicit change features."""

    FACTOR_BY_MODE = {
        "shared": 3,
        "absdiff": 2,
        "abs_signed": 4,
        "abs_signed_product": 5,
    }

    def __init__(
        self,
        channels: list[int],
        fusion_mode: str = "shared",
        attention: bool = False,
    ) -> None:
        super().__init__()
        if fusion_mode not in self.FACTOR_BY_MODE:
            raise MultiTemporalFusionError(
                f"Unsupported fusion mode '{fusion_mode}'. "
                f"Supported: {', '.join(sorted(self.FACTOR_BY_MODE))}"
            )
        self.fusion_mode = fusion_mode
        self.blocks = nn.ModuleList()
        for channel in channels:
            block: nn.Module = nn.Sequential(
                nn.Conv2d(channel * self.FACTOR_BY_MODE[fusion_mode], channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
            )
            if attention:
                block = nn.Sequential(block, ChannelAttention(channel))
            self.blocks.append(block)

    def _stack(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        signed = post - pre
        absolute = torch.abs(signed)
        if self.fusion_mode == "shared":
            return torch.cat([pre, post, absolute], dim=1)
        if self.fusion_mode == "absdiff":
            return torch.cat([post, absolute], dim=1)
        if self.fusion_mode == "abs_signed":
            return torch.cat([pre, post, absolute, signed], dim=1)
        if self.fusion_mode == "abs_signed_product":
            return torch.cat([pre, post, absolute, signed, pre * post], dim=1)
        raise MultiTemporalFusionError(f"Unsupported fusion mode: {self.fusion_mode}")

    def forward(
        self,
        pre_features: list[torch.Tensor],
        post_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            block(self._stack(pre, post))
            for block, pre, post in zip(self.blocks, pre_features, post_features)
        ]


class GatedFeatureFusion(nn.Module):
    """Fuse features with a learned gate driven by absolute change."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.gates = nn.ModuleList()
        self.projections = nn.ModuleList()
        for channel in channels:
            self.gates.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=1),
                    nn.Sigmoid(),
                )
            )
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(channel * 2, channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(
        self,
        pre_features: list[torch.Tensor],
        post_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        fused: list[torch.Tensor] = []
        for gate_block, projection, pre, post in zip(
            self.gates,
            self.projections,
            pre_features,
            post_features,
        ):
            change = torch.abs(post - pre)
            gate = gate_block(change)
            mixed = gate * post + (1.0 - gate) * pre
            fused.append(projection(torch.cat([mixed, change], dim=1)))
        return fused


class FPNDecoder(nn.Module):
    """FPN-style top-down decoder."""

    def __init__(self, in_channels: list[int], out_channels: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, out_channels, kernel_size=1) for channel in in_channels
        )
        self.smooth = nn.ModuleList(
            ConvBNReLU(out_channels, out_channels) for _ in in_channels
        )
        self.refine = nn.Sequential(
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )
        self.out_channels = out_channels

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        pyramid: list[torch.Tensor] = [self.lateral[-1](features[-1])]
        for feature, lateral in zip(reversed(features[:-1]), reversed(self.lateral[:-1])):
            top = F.interpolate(
                pyramid[-1],
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid.append(top + lateral(feature))
        pyramid = list(reversed(pyramid))
        smoothed = [
            smooth(feature) for smooth, feature in zip(self.smooth, pyramid)
        ]
        fused = torch.zeros_like(smoothed[0])
        for feature in smoothed:
            fused = fused + F.interpolate(
                feature,
                size=smoothed[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused = fused / float(len(smoothed))
        fused = F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)
        return self.refine(fused)


class ASPP(nn.Module):
    """Small DeepLab-style atrous spatial pyramid pooling block."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0),
                ConvBNReLU(in_channels, out_channels, dilation=6, padding=6),
                ConvBNReLU(in_channels, out_channels, dilation=12, padding=12),
                ConvBNReLU(in_channels, out_channels, dilation=18, padding=18),
            ]
        )
        self.project = ConvBNReLU(out_channels * len(self.branches), out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabDecoder(nn.Module):
    """DeepLab-style decoder using the deepest and lowest fused features."""

    def __init__(self, in_channels: list[int], out_channels: int = 128) -> None:
        super().__init__()
        self.aspp = ASPP(in_channels[-1], out_channels=256)
        self.low_proj = ConvBNReLU(in_channels[0], 64, kernel_size=1, padding=0)
        self.refine = nn.Sequential(
            ConvBNReLU(256 + 64, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )
        self.out_channels = out_channels

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        high = self.aspp(features[-1])
        low = self.low_proj(features[0])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.refine(torch.cat([high, low], dim=1))
        return F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)


class MultiTemporalFusionNet(nn.Module):
    """Shared-backbone MTF model with building and damage heads."""

    def __init__(
        self,
        encoder: str = "resnet50",
        decoder: str = "fpn",
        damage_channels: int = 3,
        fusion_mode: str = "shared",
        attention: bool = False,
        six_channel_control: bool = False,
    ) -> None:
        super().__init__()
        if damage_channels <= 0:
            raise MultiTemporalFusionError("damage_channels must be positive.")
        self.encoder_name = encoder
        self.decoder_name = decoder
        self.damage_channels = damage_channels
        self.six_channel_control = six_channel_control
        self.fusion_mode = fusion_mode
        self.attention = attention

        in_channels = 6 if six_channel_control else 3
        if encoder in {"resnet34", "resnet50"}:
            self.encoder = ResNetFeatureEncoder(encoder, in_channels=in_channels)
            channels = self.encoder.channels
        elif encoder == "efficientnet_b3":
            if six_channel_control:
                raise MultiTemporalFusionError("EfficientNet control_6ch is not implemented.")
            self.encoder = TimmFeatureEncoder("efficientnet_b3", in_channels=3)
            channels = self.encoder.channels
        else:
            raise MultiTemporalFusionError(
                "Unsupported encoder. Expected resnet34, resnet50 or efficientnet_b3."
            )

        if six_channel_control:
            self.fusion = None
        elif fusion_mode == "gated":
            self.fusion = GatedFeatureFusion(channels)
        else:
            self.fusion = FeatureFusion(
                channels,
                fusion_mode=fusion_mode,
                attention=attention,
            )
        if decoder == "fpn":
            self.decoder = FPNDecoder(channels, out_channels=128)
        elif decoder == "deeplab":
            self.decoder = DeepLabDecoder(channels, out_channels=128)
        else:
            raise MultiTemporalFusionError("Unsupported decoder. Expected fpn or deeplab.")
        self.building_head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)
        self.damage_head = nn.Conv2d(self.decoder.out_channels, damage_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected input shape [N, 6, H, W], got {tuple(x.shape)}.")
        output_size = x.shape[-2:]
        if self.six_channel_control:
            fused = self.encoder(x)
        else:
            pre_features = self.encoder(x[:, :3])
            post_features = self.encoder(x[:, 3:])
            assert self.fusion is not None
            fused = self.fusion(pre_features, post_features)
        decoded = self.decoder(fused, output_size=output_size)
        return {
            "building_logits": self.building_head(decoded),
            "damage_logits": self.damage_head(decoded),
        }


class DamageModelWithBuildingProxy(nn.Module):
    """Wrap an existing 3-class damage model with a learned building proxy head."""

    def __init__(self, model_name: str, num_classes: int = 3) -> None:
        super().__init__()
        self.damage_model = create_damage_model(model_name, num_classes=num_classes, in_channels=6)
        self.building_proxy = nn.Conv2d(num_classes, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        damage_logits = self.damage_model(x)
        return {
            "building_logits": self.building_proxy(damage_logits),
            "damage_logits": damage_logits,
        }


MODEL_SPECS = {
    "mtf_resnet34_fpn_shared": {
        "encoder": "resnet34",
        "decoder": "fpn",
        "fusion_mode": "shared",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_shared": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "shared",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_effb3_fpn_shared": {
        "encoder": "efficientnet_b3",
        "decoder": "fpn",
        "fusion_mode": "shared",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_deeplab_shared": {
        "encoder": "resnet50",
        "decoder": "deeplab",
        "fusion_mode": "shared",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_effb3_deeplab_shared": {
        "encoder": "efficientnet_b3",
        "decoder": "deeplab",
        "fusion_mode": "shared",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_absdiff": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "absdiff",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_abs_signed": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "abs_signed",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_abs_signed_product": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "abs_signed_product",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_attention": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "shared",
        "attention": True,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_gated": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "gated",
        "attention": False,
        "six_channel_control": False,
    },
    "mtf_resnet50_fpn_shared_attention": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "shared",
        "attention": True,
        "six_channel_control": False,
    },
    "control_6ch_resnet50_fpn": {
        "encoder": "resnet50",
        "decoder": "fpn",
        "fusion_mode": "shared",
        "attention": False,
        "six_channel_control": True,
    },
}


def supported_multitemporal_fusion_models() -> list[str]:
    return sorted(list(MODEL_SPECS) + ["control_current_siamese_attention"])


def create_multitemporal_fusion_model(
    model_name: str,
    damage_channels: int = 3,
) -> nn.Module:
    """Create an MTF model by name."""

    if model_name == "control_current_siamese_attention":
        if damage_channels != 3:
            raise MultiTemporalFusionError(
                "control_current_siamese_attention currently expects 3 damage classes."
            )
        return DamageModelWithBuildingProxy("siamese_unet_attention", num_classes=3)

    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise MultiTemporalFusionError(
            f"Unsupported MTF model '{model_name}'. Supported: "
            + ", ".join(supported_multitemporal_fusion_models())
        )
    return MultiTemporalFusionNet(damage_channels=damage_channels, **spec)

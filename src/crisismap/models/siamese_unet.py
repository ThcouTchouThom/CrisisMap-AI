"""Siamese U-Net variants for pre/post disaster damage segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crisismap.models.unet import DoubleConv, DownBlock, UpBlock


class SharedUNetEncoder(nn.Module):
    """U-Net encoder reused with shared weights for pre and post images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.enc1 = DoubleConv(in_channels, self.channels[0])
        self.enc2 = DownBlock(self.channels[0], self.channels[1])
        self.enc3 = DownBlock(self.channels[1], self.channels[2])
        self.enc4 = DownBlock(self.channels[2], self.channels[3])
        self.bottleneck = DownBlock(self.channels[3], self.channels[4])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        bottleneck = self.bottleneck(skip4)
        return [skip1, skip2, skip3, skip4, bottleneck]


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention for fused Siamese features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class FeatureFusionBlock(nn.Module):
    """Fuse pre/post feature tensors with a configurable change representation."""

    MODE_FACTORS = {
        "shared": 3,
        "absdiff": 2,
        "abs_signed": 4,
        "abs_signed_product": 5,
    }

    def __init__(
        self,
        channels: int,
        mode: str = "shared",
        attention: bool = False,
    ) -> None:
        super().__init__()
        if mode not in self.MODE_FACTORS:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        self.mode = mode
        self.block = nn.Sequential(
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
            raise ValueError(f"Unsupported fusion mode: {self.mode}")

        return self.attention(self.block(torch.cat(features, dim=1)))


class GatedFusionBlock(nn.Module):
    """Fuse pre/post features with an abs-change gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.block = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        abs_change = torch.abs(post_feat - pre_feat)
        gate = self.gate(abs_change)
        mixed = gate * post_feat + (1.0 - gate) * pre_feat
        return self.block(torch.cat([mixed, abs_change], dim=1))


class SiameseUNet(nn.Module):
    """Shared-encoder Siamese U-Net with configurable feature fusion."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
        fusion_mode: str = "shared",
        attention: bool = False,
        gated: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 6:
            raise ValueError("SiameseUNet expects in_channels=6.")
        if gated and fusion_mode != "gated":
            raise ValueError("Set fusion_mode='gated' when gated=True.")

        self.fusion_mode = fusion_mode
        self.encoder = SharedUNetEncoder(in_channels=3, base_channels=base_channels)
        channels = self.encoder.channels
        if gated:
            self.fusion = nn.ModuleList(GatedFusionBlock(channels_i) for channels_i in channels)
        else:
            self.fusion = nn.ModuleList(
                FeatureFusionBlock(channels_i, mode=fusion_mode, attention=attention)
                for channels_i in channels
            )

        self.dec4 = UpBlock(channels[4], channels[3], channels[3])
        self.dec3 = UpBlock(channels[3], channels[2], channels[2])
        self.dec2 = UpBlock(channels[2], channels[1], channels[1])
        self.dec1 = UpBlock(channels[1], channels[0], channels[0])
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 6:
            raise ValueError(f"Expected 6 input channels, got {x.shape[1]}.")

        pre_image = x[:, :3]
        post_image = x[:, 3:]
        pre_features = self.encoder(pre_image)
        post_features = self.encoder(post_image)
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
        logits = self.classifier(output)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


class SiameseUNetSharedEncoder(SiameseUNet):
    """Original Axis 2 Siamese model: concat(pre, post, abs(post - pre))."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            fusion_mode="shared",
        )


class SiameseUNetAbsDiff(SiameseUNet):
    """Siamese U-Net using post features plus absolute feature difference."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            fusion_mode="absdiff",
        )


class SiameseUNetAbsSigned(SiameseUNet):
    """Siamese U-Net with absolute and signed feature differences."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            fusion_mode="abs_signed",
        )


class SiameseUNetAbsSignedProduct(SiameseUNet):
    """Siamese U-Net with abs, signed, and feature-product fusion."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            fusion_mode="abs_signed_product",
        )


class SiameseUNetGatedFusion(SiameseUNet):
    """Siamese U-Net with a learned abs-change gate for pre/post fusion."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            fusion_mode="gated",
            gated=True,
        )


class SiameseUNetAttention(SiameseUNet):
    """Siamese U-Net with channel attention after each fusion projection."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            fusion_mode="shared",
            attention=True,
        )


def smoke_test() -> None:
    model_names = [
        ("shared", SiameseUNetSharedEncoder),
        ("absdiff", SiameseUNetAbsDiff),
        ("abs_signed", SiameseUNetAbsSigned),
        ("abs_signed_product", SiameseUNetAbsSignedProduct),
        ("gated", SiameseUNetGatedFusion),
        ("attention", SiameseUNetAttention),
    ]
    x = torch.randn(1, 6, 256, 256)
    for name, model_class in model_names:
        model = model_class(in_channels=6, num_classes=3, base_channels=16)
        with torch.no_grad():
            y = model(x)
        print(f"{name}: {tuple(y.shape)}")
        assert tuple(y.shape) == (1, 3, 256, 256)


if __name__ == "__main__":
    smoke_test()

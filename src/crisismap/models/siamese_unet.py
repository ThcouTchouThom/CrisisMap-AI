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


class FeatureFusionBlock(nn.Module):
    """Fuse pre, post, and absolute-change features at one encoder level."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        change_feat = torch.abs(post_feat - pre_feat)
        return self.block(torch.cat([pre_feat, post_feat, change_feat], dim=1))


class SiameseUNetSharedEncoder(nn.Module):
    """Shared-encoder Siamese U-Net for 6-channel pre/post xBD inputs.

    The model splits an input tensor into pre-disaster RGB and post-disaster RGB,
    encodes both streams with the same weights, fuses every encoder level with
    concat(pre, post, abs(post - pre)), and decodes the fused features into a
    3-class damage map.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels != 6:
            raise ValueError("SiameseUNetSharedEncoder expects in_channels=6.")

        self.encoder = SharedUNetEncoder(in_channels=3, base_channels=base_channels)
        channels = self.encoder.channels
        self.fusion = nn.ModuleList(FeatureFusionBlock(channels_i) for channels_i in channels)

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


def smoke_test() -> None:
    model = SiameseUNetSharedEncoder(in_channels=6, num_classes=3, base_channels=16)
    x = torch.randn(1, 6, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (1, 3, 256, 256)


if __name__ == "__main__":
    smoke_test()

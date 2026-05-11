"""Lightweight U-Net baseline for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolution blocks used throughout the U-Net."""

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


class DownBlock(nn.Module):
    """Encoder block: downsample, then apply two convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: upsample, concatenate skip features, then convolve."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """A compact 4-level U-Net for xBD damage segmentation."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.enc1 = DoubleConv(in_channels, channels[0])
        self.enc2 = DownBlock(channels[0], channels[1])
        self.enc3 = DownBlock(channels[1], channels[2])
        self.enc4 = DownBlock(channels[2], channels[3])
        self.bottleneck = DownBlock(channels[3], channels[4])

        self.dec4 = UpBlock(channels[4], channels[3], channels[3])
        self.dec3 = UpBlock(channels[3], channels[2], channels[2])
        self.dec2 = UpBlock(channels[2], channels[1], channels[1])
        self.dec1 = UpBlock(channels[1], channels[0], channels[0])

        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)
        x = self.bottleneck(skip4)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.classifier(x)


def smoke_test() -> None:
    model = UNet(in_channels=6, num_classes=3, base_channels=32)
    x = torch.randn(2, 6, 512, 512)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (2, 3, 512, 512)


if __name__ == "__main__":
    smoke_test()

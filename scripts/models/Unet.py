"""Residual U-Net implemented in PyTorch.

This mirrors ``scripts_old/models/Res_UNet.py``. The network returns logits;
apply ``torch.sigmoid`` only for probabilities, metrics, and inference.
"""

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    """The three-convolution residual block used by the TensorFlow model."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.skip_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        skip = self.skip_bn(F.relu(self.skip_conv(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.relu(x + skip)


class UNetRes(nn.Module):
    """Residual U-Net with nearest-neighbour decoder upsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: Sequence[int] = (64, 128, 256, 512),
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        filters = tuple(int(value) for value in num_filters)
        if not filters or any(value <= 0 for value in filters):
            raise ValueError("num_filters must contain positive integers")

        encoder_channels = (in_channels, *filters[:-1])
        self.enc_blocks = nn.ModuleList(
            ResConvBlock(input_channels, output_channels)
            for input_channels, output_channels in zip(encoder_channels, filters)
        )
        self.pool = nn.MaxPool2d(2)
        self.bridge = ResConvBlock(filters[-1], filters[-1])

        self.dec_blocks = nn.ModuleList()
        current_channels = filters[-1]
        for skip_channels in reversed(filters):
            self.dec_blocks.append(
                ResConvBlock(current_channels + skip_channels, skip_channels)
            )
            current_channels = skip_channels

        self.out_conv = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        skips: list[torch.Tensor] = []
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bridge(x)
        for block, skip in zip(self.dec_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat((x, skip), dim=1))
        return self.out_conv(x)


def build_model(
    input_size: int | None = None,
    num_filters: Sequence[int] = (64, 128, 256, 512),
) -> UNetRes:
    """Compatibility wrapper for the TensorFlow ``build_model`` function."""
    del input_size
    return UNetRes(num_filters=num_filters)

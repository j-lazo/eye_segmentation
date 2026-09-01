"""Paper-specific ResUNet for corneal nerve fibre segmentation."""

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions plus the paper's 1x1 projected shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(inputs) + self.shortcut(inputs), inplace=True)


class CNFLResUNet(nn.Module):
    """Figure S1 architecture: 1-channel input and 2-class softmax logits."""

    def __init__(
        self,
        filters: Sequence[int] = (64, 128, 256, 512),
        bridge_channels: int = 1024,
    ) -> None:
        super().__init__()
        filters = tuple(int(value) for value in filters)
        encoder_inputs = (1, *filters[:-1])
        self.encoder = nn.ModuleList(
            ResidualBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(encoder_inputs, filters)
        )
        self.pool = nn.MaxPool2d(2)
        self.bridge = ResidualBlock(filters[-1], bridge_channels)
        self.decoder = nn.ModuleList()
        channels = bridge_channels
        for skip_channels in reversed(filters):
            self.decoder.append(ResidualBlock(channels + skip_channels, skip_channels))
            channels = skip_channels
        self.output = nn.Conv2d(filters[0], 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = inputs
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat((x, skip), dim=1))
        return self.output(x)


def two_class_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    """Mean soft Dice over background and nerve channels, as in equation S1."""
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target[:, 0].long(), num_classes=2).permute(0, 3, 1, 2).float()
    intersection = (probabilities * target_one_hot).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()

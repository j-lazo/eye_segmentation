"""Loss functions for binary segmentation."""

import torch

from .metric_functions import dice_coef


def dice_coef_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coef(torch.sigmoid(logits), target)


def dice_coef_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compatibility function for callers that already supply probabilities."""
    return 1.0 - dice_coef(pred, target)

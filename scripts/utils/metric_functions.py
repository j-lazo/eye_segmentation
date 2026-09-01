"""Segmentation metrics matching the TensorFlow implementation."""

import torch


def dice_coef(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    reduce_dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=reduce_dims)
    union = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean()


@torch.no_grad()
def bin_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[float, float, float, float]:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    acc = (pred == target).float().mean()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = dice_coef(pred, target)
    return acc.item(), precision.item(), recall.item(), dice.item()

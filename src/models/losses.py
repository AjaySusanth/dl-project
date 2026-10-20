"""Loss functions for quantile forecasting."""

from __future__ import annotations

import torch
from torch import Tensor


def pinball_loss(pred: Tensor, target: Tensor, quantiles: Tensor) -> Tensor:
    """Return mean pinball loss for ``pred`` (B,H,Q) and ``target`` (B,H)."""
    if pred.ndim != 3 or target.ndim != 2 or quantiles.ndim != 1:
        raise ValueError("pred must be 3-D, target 2-D, and quantiles 1-D")
    if pred.shape[:2] != target.shape or pred.shape[2] != quantiles.shape[0]:
        raise ValueError("pred, target, and quantiles have incompatible shapes")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("pred and target must contain finite values")
    if not torch.isfinite(quantiles).all() or ((quantiles <= 0) | (quantiles >= 1)).any():
        raise ValueError("quantiles must be finite values strictly between 0 and 1")

    errors = target.unsqueeze(-1) - pred
    quantile_values = quantiles.to(device=pred.device, dtype=pred.dtype)
    return torch.maximum(
        quantile_values * errors,
        (quantile_values - 1) * errors,
    ).mean()

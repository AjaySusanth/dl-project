"""Tests for quantile pinball loss."""

from __future__ import annotations

import pytest
import torch

from src.models.losses import pinball_loss
from src.evaluate.metrics import mae


def test_pinball_loss_is_zero_for_perfect_prediction() -> None:
    """Matching every quantile target has zero loss."""
    target = torch.tensor([[1.0, 2.0]])
    pred = target.unsqueeze(-1).expand(1, 2, 3)
    quantiles = torch.tensor([0.1, 0.5, 0.9])

    assert pinball_loss(pred, target, quantiles).item() == pytest.approx(0.0)


def test_pinball_loss_has_correct_asymmetry() -> None:
    """Underprediction is weighted by q and overprediction by 1-q."""
    target = torch.tensor([[10.0]])
    quantiles = torch.tensor([0.1, 0.9])
    pred = torch.tensor([[[8.0, 12.0]]])

    # (0.1 * 2 + 0.1 * 2) / 2
    assert pinball_loss(pred, target, quantiles).item() == pytest.approx(0.2)


def test_pinball_loss_matches_hand_computed_example() -> None:
    """The mean is taken over batch, horizon, and quantile dimensions."""
    target = torch.tensor([[10.0, 20.0]])
    pred = torch.tensor([[[8.0, 10.0], [23.0, 18.0]]])
    quantiles = torch.tensor([0.25, 0.75])

    # Losses: [0.5, 0.0, 2.25, 1.5], mean = 1.0625.
    assert pinball_loss(pred, target, quantiles).item() == pytest.approx(1.0625)


def test_median_pinball_loss_is_half_mae() -> None:
    """At q=0.5, pinball loss equals half the absolute error."""
    target = torch.tensor([[10.0, 20.0]])
    pred = torch.tensor([[[8.0], [23.0]]])
    quantiles = torch.tensor([0.5])

    assert pinball_loss(pred, target, quantiles).item() == pytest.approx(
        mae(target.numpy(), pred.squeeze(-1).numpy()) / 2
    )


def test_pinball_loss_rejects_incompatible_inputs() -> None:
    """Invalid shapes and quantiles fail explicitly."""
    with pytest.raises(ValueError, match="incompatible"):
        pinball_loss(torch.zeros(1, 2, 3), torch.zeros(1, 3), torch.tensor([0.1, 0.5, 0.9]))
    with pytest.raises(ValueError, match="strictly"):
        pinball_loss(torch.zeros(1, 1, 1), torch.zeros(1, 1), torch.tensor([1.0]))

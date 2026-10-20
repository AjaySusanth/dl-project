"""Hand-computed tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate.metrics import coverage, interval_width, mae, peak_mae, rmse, wape


def test_point_metrics_match_hand_computed_values() -> None:
    """MAE, RMSE, and WAPE match a small raw-count example."""
    actual = np.array([2.0, 4.0, 8.0])
    predicted = np.array([1.0, 6.0, 5.0])

    assert mae(actual, predicted) == pytest.approx(2)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(14 / 3))
    assert wape(actual, predicted) == pytest.approx(6 / 14)


def test_peak_mae_filters_on_actual_values() -> None:
    """Peak MAE includes actual peaks, not predictions that happen to be high."""
    actual = np.array([2.0, 10.0, 12.0, 4.0])
    predicted = np.array([100.0, 8.0, 15.0, 100.0])

    assert peak_mae(actual, predicted, peak_threshold=10) == pytest.approx(2.5)


def test_interval_metrics_match_hand_computed_values() -> None:
    """Coverage is inclusive and width is the mean P90-P10 distance."""
    actual = np.array([1.0, 5.0, 10.0, 20.0])
    lower = np.array([0.0, 5.0, 11.0, 21.0])
    upper = np.array([2.0, 6.0, 12.0, 25.0])

    assert coverage(actual, lower, upper) == pytest.approx(0.5)
    assert interval_width(lower, upper) == pytest.approx(2.0)


def test_metrics_reject_invalid_shapes_and_denominators() -> None:
    """Invalid input cannot produce a success-shaped metric."""
    with pytest.raises(ValueError, match="shapes differ"):
        mae([1, 2], [1])
    with pytest.raises(ValueError, match="sum of actual"):
        wape([0, 0], [1, 2])
    with pytest.raises(ValueError, match="no actual values"):
        peak_mae([1, 2], [1, 2], peak_threshold=10)
    with pytest.raises(ValueError, match="must not exceed"):
        coverage([1], [2], [1])

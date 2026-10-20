"""Metrics for raw-count forecasts and quantile prediction intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert and validate two equally shaped metric inputs."""
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            f"actual and predicted shapes differ: {actual_array.shape} != {predicted_array.shape}"
        )
    if actual_array.size == 0:
        raise ValueError("Metric inputs must not be empty")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("Metric inputs must contain only finite values")
    return actual_array, predicted_array


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return mean absolute error on raw pickup counts."""
    actual_array, predicted_array = _pair(actual, predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return root mean squared error on raw pickup counts."""
    actual_array, predicted_array = _pair(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_array - predicted_array))))


def wape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return weighted absolute percentage error."""
    actual_array, predicted_array = _pair(actual, predicted)
    denominator = float(np.sum(np.abs(actual_array)))
    if denominator == 0:
        raise ValueError("WAPE is undefined when the sum of actual values is zero")
    return float(np.sum(np.abs(actual_array - predicted_array)) / denominator)


def peak_mae(
    actual: ArrayLike, predicted: ArrayLike, peak_threshold: float
) -> float:
    """Return MAE for actual values at or above a zone's peak threshold."""
    actual_array, predicted_array = _pair(actual, predicted)
    if not np.isfinite(peak_threshold):
        raise ValueError("peak_threshold must be finite")
    peak_mask = actual_array >= float(peak_threshold)
    if not peak_mask.any():
        raise ValueError("peak_mae is undefined when no actual values reach the threshold")
    return mae(actual_array[peak_mask], predicted_array[peak_mask])


def coverage(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    """Return the fraction of actual values inside the inclusive interval."""
    actual_array, lower_array = _pair(actual, lower)
    upper_array = np.asarray(upper, dtype=np.float64)
    if upper_array.shape != actual_array.shape:
        raise ValueError(
            f"actual and upper shapes differ: {actual_array.shape} != {upper_array.shape}"
        )
    if not np.isfinite(upper_array).all():
        raise ValueError("Interval bounds must contain only finite values")
    if (lower_array > upper_array).any():
        raise ValueError("Lower interval bounds must not exceed upper bounds")
    return float(np.mean((actual_array >= lower_array) & (actual_array <= upper_array)))


def interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Return the mean width of an inclusive prediction interval."""
    lower_array, upper_array = _pair(lower, upper)
    if (lower_array > upper_array).any():
        raise ValueError("Lower interval bounds must not exceed upper bounds")
    return float(np.mean(upper_array - lower_array))

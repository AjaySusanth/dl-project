"""Compute chronological train, validation, and test boundaries."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def compute_splits(
    hourly: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Return inclusive ISO timestamp boundaries for chronological splits."""
    if "hour_ts" not in hourly.columns:
        raise ValueError("Hourly table must contain an hour_ts column")

    try:
        fractions = config["split"]["fractions"]
    except KeyError as error:
        raise KeyError(f"Missing required configuration key: {error}") from error

    if len(fractions) != 3 or any(float(fraction) <= 0 for fraction in fractions):
        raise ValueError("split.fractions must contain three positive values")
    if abs(sum(float(fraction) for fraction in fractions) - 1.0) > 1e-9:
        raise ValueError("split.fractions must sum to 1.0")

    timestamps = pd.to_datetime(hourly["hour_ts"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Hourly table contains invalid hour_ts values")
    timeline = pd.DatetimeIndex(timestamps.drop_duplicates().sort_values())
    if len(timeline) < 3:
        raise ValueError("At least three unique timestamps are required")

    train_end_index = max(0, int(len(timeline) * float(fractions[0])) - 1)
    val_end_index = max(
        train_end_index + 1,
        int(len(timeline) * float(fractions[0] + fractions[1])) - 1,
    )
    if val_end_index >= len(timeline) - 1:
        val_end_index = len(timeline) - 2
    if train_end_index >= val_end_index:
        train_end_index = val_end_index - 1
    if train_end_index < 0:
        raise ValueError("Split fractions do not produce a non-empty train split")

    boundaries = {
        "train": [timeline[0], timeline[train_end_index]],
        "val": [timeline[train_end_index + 1], timeline[val_end_index]],
        "test": [timeline[val_end_index + 1], timeline[-1]],
    }
    return {
        name: [start.isoformat(), end.isoformat()]
        for name, (start, end) in boundaries.items()
    }

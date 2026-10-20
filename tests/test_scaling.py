"""Tests for train-only scaling and busy-time statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.scaling import compute_busy_time, fit_zone_stats
from src.features.splits import compute_splits


def _hourly() -> pd.DataFrame:
    """Build a deterministic two-zone fixture with future rows."""
    hours = pd.date_range("2026-01-01", periods=12, freq="h")
    return pd.DataFrame(
        [
            {
                "zone_id": zone_id,
                "hour_ts": hour,
                "pickups": (index % 4) + zone_id,
                "mean_duration_min": 10.0 + zone_id + index,
            }
            for zone_id in (1, 2)
            for index, hour in enumerate(hours)
        ]
    )


def _config() -> dict:
    """Return evaluation configuration for the fixture."""
    return {
        "split": {"fractions": [0.5, 0.25, 0.25]},
        "evaluation": {"peak_percentile": 90},
    }


def test_zone_stats_ignore_test_period_changes() -> None:
    """Changing future pickups cannot change train-fitted statistics."""
    hourly = _hourly()
    config = _config()
    splits = compute_splits(hourly, config)
    original = fit_zone_stats(hourly, splits, config)

    changed = hourly.copy()
    test_start = pd.to_datetime(splits["test"][0])
    changed.loc[changed["hour_ts"] >= test_start, "pickups"] = 9999

    assert original == fit_zone_stats(changed, splits, config)


def test_zero_log_demand_standard_deviation_raises() -> None:
    """A constant zone cannot be standardized."""
    hourly = _hourly()
    hourly.loc[hourly["zone_id"] == 1, "pickups"] = 4
    config = _config()
    splits = compute_splits(hourly, config)

    with pytest.raises(ValueError, match="zero train-period"):
        fit_zone_stats(hourly, splits, config)


def test_busy_time_fills_missing_zone_hours_from_zone_mean() -> None:
    """A missing zone-hour duration receives the zone fallback."""
    hourly = _hourly()
    hourly.loc[
        (hourly["zone_id"] == 1) & (hourly["hour_ts"].dt.hour == 5),
        ["pickups", "mean_duration_min"],
    ] = [0, float("nan")]
    config = _config()
    splits = compute_splits(hourly, config)
    busy = compute_busy_time(hourly, splits)

    assert len(busy) == 48
    assert busy["mean_duration_min"].notna().all()
    zone_values = busy[busy["zone_id"] == 1]["mean_duration_min"]
    assert zone_values.nunique() > 1

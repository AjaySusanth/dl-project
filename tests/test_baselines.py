"""Tests for same-day and same-week seasonal baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.baselines import lag_baseline


def _hourly(periods: int = 200) -> pd.DataFrame:
    """Build a complete, identifiable hourly history for two zones."""
    hours = pd.date_range("2026-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        [
            {
                "zone_id": zone_id,
                "hour_ts": hour,
                "pickups": index + zone_id * 1000,
            }
            for zone_id in (1, 2)
            for index, hour in enumerate(hours)
        ]
    )


def test_lag_baselines_use_the_expected_historical_hours() -> None:
    """Lag 24 and lag 168 return exact values for each target horizon."""
    hourly = _hourly()
    origins = np.array([pd.Timestamp("2026-01-08 12:00")], dtype="datetime64[ns]")

    b1, b2 = lag_baseline(hourly, origins, np.array([1]), horizon=6)

    expected_b1 = np.array([1000 + 156 + offset for offset in range(1, 7)])
    expected_b2 = np.array([1000 + 12 + offset for offset in range(1, 7)])
    np.testing.assert_array_equal(b1[0], expected_b1)
    np.testing.assert_array_equal(b2[0], expected_b2)


def test_lag_baselines_align_zone_rows_and_return_float32() -> None:
    """Multiple origins and zones remain aligned in the returned arrays."""
    hourly = _hourly()
    origins = np.array(
        [pd.Timestamp("2026-01-08 12:00"), pd.Timestamp("2026-01-08 13:00")],
        dtype="datetime64[ns]",
    )

    b1, b2 = lag_baseline(hourly, origins, np.array([1, 2]), horizon=2)

    assert b1.shape == (2, 2)
    assert b2.shape == (2, 2)
    assert b1.dtype == np.float32
    assert b2.dtype == np.float32
    assert b1[1, 0] - b1[0, 0] == 1001
    assert b2[1, 1] - b2[0, 1] == 1001


def test_missing_historical_hours_fail_explicitly() -> None:
    """A missing source hour must not become a fabricated baseline value."""
    hourly = _hourly()
    hourly = hourly[
        ~(
            (hourly["zone_id"] == 1)
            & (hourly["hour_ts"] == pd.Timestamp("2026-01-01 13:00"))
        )
    ].copy()
    with pytest.raises(ValueError, match="missing historical hours"):
        lag_baseline(
            hourly,
            np.array([pd.Timestamp("2026-01-08 12:00")], dtype="datetime64[ns]"),
            np.array([1]),
            horizon=6,
        )

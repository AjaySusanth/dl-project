"""Tests for fleet requirement calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fleet.fleet import fleet_table, required_taxis


def _busy_time() -> pd.DataFrame:
    """Return a complete synthetic duration table for one zone."""
    return pd.DataFrame(
        {
            "zone_id": [7] * 24,
            "hour_of_day": list(range(24)),
            "mean_duration_min": [22.0] * 24,
        }
    )


def test_required_taxis_matches_prd_examples() -> None:
    """The documented fleet examples use the specified ceiling formula."""
    assert required_taxis(48, 22, 0.7) == 26
    assert required_taxis(61, 22, 0.7) == 32


@pytest.mark.parametrize(
    ("pickups", "busy_min", "utilization"),
    [(-1, 22, 0.7), (48, 0, 0.7), (48, 22, 0), (48, 22, 1.1)],
)
def test_required_taxis_rejects_invalid_inputs(
    pickups: float, busy_min: float, utilization: float
) -> None:
    """Invalid demand, duration, and utilization fail explicitly."""
    with pytest.raises(ValueError):
        required_taxis(pickups, busy_min, utilization)


def test_fleet_table_contains_contract_and_status_rules() -> None:
    """Fleet rows calculate expected/safe gaps and documented statuses."""
    predictions = np.array([[48.0, 48.0, 61.0], [1.0, 1.0, 1.0]])
    table = fleet_table(
        "2026-01-01 10:00",
        7,
        predictions,
        _busy_time(),
        available=26,
        deadhead_min=0,
        utilization=0.7,
    )

    assert list(table.columns) == [
        "horizon",
        "hour_ts",
        "p10",
        "p50",
        "p90",
        "busy_min",
        "n_expected",
        "n_safe",
        "available",
        "gap_expected",
        "gap_safe",
        "status",
    ]
    assert table.loc[0, "n_expected"] == 26
    assert table.loc[0, "n_safe"] == 32
    assert table.loc[0, "status"] == "risk"
    assert table.loc[1, "status"] == "ok"


def test_fleet_table_rejects_missing_zone_duration() -> None:
    """A missing train-derived duration cannot silently become a default."""
    with pytest.raises(ValueError, match="no rows"):
        fleet_table(
            "2026-01-01 10:00",
            99,
            np.ones((1, 3)),
            _busy_time(),
            available=1,
            deadhead_min=0,
            utilization=0.7,
        )

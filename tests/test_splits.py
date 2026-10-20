"""Tests for chronological split boundaries and target containment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.scaling import fit_zone_stats
from src.features.splits import compute_splits
from src.features.windows import build_windows


def _hourly(periods: int = 100) -> pd.DataFrame:
    """Build two complete synthetic zone timelines."""
    hours = pd.date_range("2026-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        [
            {
                "zone_id": zone_id,
                "hour_ts": hour,
                "pickups": (index % 9) + zone_id,
                "mean_duration_min": 10.0 + zone_id,
            }
            for zone_id in (1, 2)
            for index, hour in enumerate(hours)
        ]
    )


def _config() -> dict:
    """Return the split and window settings used by the fixture."""
    return {
        "split": {"fractions": [0.7, 0.15, 0.15]},
        "windows": {"lookback": 10, "horizon": 6},
        "evaluation": {"peak_percentile": 90},
    }


def test_compute_splits_is_chronological_and_iso() -> None:
    """Split boundaries are sorted, disjoint, and ISO-compatible."""
    hourly = _hourly()
    splits = compute_splits(hourly, _config())

    assert set(splits) == {"train", "val", "test"}
    parsed = {name: pd.to_datetime(bounds) for name, bounds in splits.items()}
    assert parsed["train"][1] < parsed["val"][0]
    assert parsed["val"][1] < parsed["test"][0]
    assert all("T" in value for bounds in splits.values() for value in bounds)


def test_windows_match_documented_per_zone_counts() -> None:
    """Window counts follow n_train-L-H+1 and n_eval-H+1."""
    hourly = _hourly()
    config = _config()
    splits = compute_splits(hourly, config)
    stats = fit_zone_stats(hourly, splits, config)
    windows = build_windows(hourly, stats, splits, config)

    assert len(windows["train"]["zone_id"]) == 2 * (70 - 10 - 6 + 1)
    assert len(windows["val"]["zone_id"]) == 2 * (15 - 6 + 1)
    assert len(windows["test"]["zone_id"]) == 2 * (15 - 6 + 1)


def test_invalid_split_fractions_raise() -> None:
    """Fractions must have three positive values summing to one."""
    with pytest.raises(ValueError, match="three positive values"):
        compute_splits(_hourly(), {"split": {"fractions": [0.5, 0.5]}})

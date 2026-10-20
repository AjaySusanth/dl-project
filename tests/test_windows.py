"""Tests for model-ready forecasting windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.scaling import fit_zone_stats
from src.features.splits import compute_splits
from src.features.windows import build_windows


def _fixture() -> tuple[pd.DataFrame, dict, dict]:
    """Return hourly data, configuration, and computed split boundaries."""
    hours = pd.date_range("2026-01-01", periods=100, freq="h")
    hourly = pd.DataFrame(
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
    config = {
        "split": {"fractions": [0.7, 0.15, 0.15]},
        "windows": {"lookback": 10, "horizon": 6},
        "evaluation": {"peak_percentile": 90},
    }
    return hourly, config, compute_splits(hourly, config)


def test_window_shapes_and_keys_match_architecture() -> None:
    """Each split contains the documented arrays and dimensions."""
    hourly, config, splits = _fixture()
    stats = fit_zone_stats(hourly, splits, config)
    windows = build_windows(hourly, stats, splits, config)

    expected_keys = {"X", "zone_idx", "y_scaled", "y_raw", "origin_ts", "zone_id"}
    for arrays in windows.values():
        assert set(arrays) == expected_keys
        assert arrays["X"].shape[1:] == (10, 6)
        assert arrays["y_scaled"].shape[1] == 6
        assert arrays["y_raw"].shape == arrays["y_scaled"].shape
        assert arrays["zone_idx"].shape == arrays["zone_id"].shape
        assert not np.isnan(arrays["X"]).any()
        assert not np.isnan(arrays["y_scaled"]).any()


def test_targets_align_after_origin_and_stay_inside_split() -> None:
    """Every target is origin plus one through six hours in its split."""
    hourly, config, splits = _fixture()
    stats = fit_zone_stats(hourly, splits, config)
    windows = build_windows(hourly, stats, splits, config)

    for split_name, arrays in windows.items():
        boundaries = pd.to_datetime(splits[split_name]).to_numpy()
        target_hours = arrays["origin_ts"][:, None] + np.arange(
            1, 7, dtype="timedelta64[h]"
        )
        assert (target_hours >= boundaries[0]).all()
        assert (target_hours <= boundaries[1]).all()

        for origin, zone_id in zip(arrays["origin_ts"], arrays["zone_id"]):
            matching = hourly[
                (hourly["zone_id"] == zone_id)
                & (hourly["hour_ts"] == pd.Timestamp(origin))
            ]
            assert len(matching) == 1

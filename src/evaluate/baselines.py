"""Seasonal pickup baselines for evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

HOURLY_COLUMNS = ("zone_id", "hour_ts", "pickups")
LAGS = (24, 168)


def lag_baseline(
    hourly: pd.DataFrame,
    origin_ts: ArrayLike,
    zone_id: ArrayLike,
    horizon: int = 6,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return lag-24 and lag-168 forecasts with shape ``(N, horizon)``.

    For target hour ``origin_ts + h``, each baseline uses the pickup count at
    ``origin_ts + h - lag``. With the six-hour horizon, all source hours are
    already observed before the forecast origin.
    """
    missing = sorted(set(HOURLY_COLUMNS) - set(hourly.columns))
    if missing:
        raise ValueError("Hourly table is missing required columns: " + ", ".join(missing))
    if horizon < 1:
        raise ValueError("horizon must be positive")

    origins = pd.to_datetime(np.asarray(origin_ts), errors="coerce")
    zones = np.asarray(zone_id)
    if origins.ndim != 1 or zones.ndim != 1 or len(origins) != len(zones):
        raise ValueError("origin_ts and zone_id must be one-dimensional arrays of equal length")
    if origins.isna().any():
        raise ValueError("origin_ts contains invalid timestamps")
    if len(origins) == 0:
        raise ValueError("origin_ts and zone_id must not be empty")

    table = hourly.loc[:, list(HOURLY_COLUMNS)].copy()
    table["hour_ts"] = pd.to_datetime(table["hour_ts"], errors="coerce")
    if table["hour_ts"].isna().any():
        raise ValueError("Hourly table contains invalid hour_ts values")
    if table[["zone_id", "pickups"]].isna().any().any():
        raise ValueError("Hourly table contains missing zone_id or pickups values")
    if (table["pickups"] < 0).any():
        raise ValueError("Hourly table contains negative pickup values")
    if table.duplicated(["zone_id", "hour_ts"]).any():
        raise ValueError("Hourly table contains duplicate zone-hour rows")

    lookup = table.set_index(["zone_id", "hour_ts"])["pickups"]
    predictions: list[NDArray[np.float32]] = []
    target_offsets = np.arange(1, horizon + 1, dtype="timedelta64[h]")
    for lag in LAGS:
        values = np.empty((len(origins), horizon), dtype=np.float32)
        for row, (origin, current_zone) in enumerate(zip(origins, zones)):
            targets = origin.to_datetime64() + target_offsets
            sources = pd.to_datetime(targets) - pd.Timedelta(hours=lag)
            if (sources >= origin).any():
                raise ValueError(f"Lag {lag} uses a source hour at or after the forecast origin")
            keys = [(current_zone, timestamp) for timestamp in sources]
            missing_keys = [key for key in keys if key not in lookup.index]
            if missing_keys:
                raise ValueError(
                    f"Lag {lag} has missing historical hours for zone {current_zone}"
                )
            values[row] = lookup.loc[keys].to_numpy(dtype=np.float32)
        predictions.append(values)

    return predictions[0], predictions[1]

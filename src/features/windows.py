"""Build chronological forecasting windows from hourly zone demand."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features.scaling import apply_scaling

FEATURE_COUNT = 6
REQUIRED_OUTPUT_KEYS = ("X", "zone_idx", "y_scaled", "y_raw", "origin_ts", "zone_id")


def _parse_boundaries(splits: Mapping[str, Any], name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse and validate one inclusive split range."""
    try:
        start_value, end_value = splits[name]
    except (KeyError, ValueError) as error:
        raise ValueError(f"splits must contain {name} [start, end] boundaries") from error
    start = pd.to_datetime(start_value, errors="coerce")
    end = pd.to_datetime(end_value, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError(f"{name} split boundaries must be valid and ordered")
    return start, end


def _feature_row(row: pd.Series) -> list[float]:
    """Return the six documented features for one hourly row."""
    timestamp = row["hour_ts"]
    hour = timestamp.hour
    weekday = timestamp.dayofweek
    return [
        float(row["z_pickups"]),
        float(np.sin(2 * np.pi * hour / 24)),
        float(np.cos(2 * np.pi * hour / 24)),
        float(np.sin(2 * np.pi * weekday / 7)),
        float(np.cos(2 * np.pi * weekday / 7)),
        float(weekday >= 5),
    ]


def _empty_windows(lookback: int, horizon: int) -> dict[str, np.ndarray]:
    """Create correctly typed empty arrays for a split with no samples."""
    return {
        "X": np.empty((0, lookback, FEATURE_COUNT), dtype=np.float32),
        "zone_idx": np.empty(0, dtype=np.int64),
        "y_scaled": np.empty((0, horizon), dtype=np.float32),
        "y_raw": np.empty((0, horizon), dtype=np.float32),
        "origin_ts": np.empty(0, dtype="datetime64[ns]"),
        "zone_id": np.empty(0, dtype=np.int16),
    }


def build_windows(
    hourly: pd.DataFrame,
    zone_stats: Mapping[str, Any],
    splits: Mapping[str, Any],
    config: Mapping[str, Any],
    lookback: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build train, validation, and test windows.

    Each ``X`` has shape ``(N, lookback, 6)`` and each target array has shape
    ``(N, horizon)``. Targets remain wholly inside their named split.
    """
    try:
        configured_lookback = int(config["windows"]["lookback"])
        horizon = int(config["windows"]["horizon"])
    except KeyError as error:
        raise KeyError(f"Missing required configuration key: {error}") from error
    if lookback is not None:
        configured_lookback = int(lookback)
    if configured_lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")

    required = {"zone_id", "hour_ts", "pickups", "mean_duration_min"}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError("Hourly table is missing required columns: " + ", ".join(missing))
    normalized = hourly.loc[:, list(required)].copy()
    normalized["hour_ts"] = pd.to_datetime(normalized["hour_ts"], errors="coerce")
    if normalized["hour_ts"].isna().any():
        raise ValueError("Hourly table contains invalid hour_ts values")
    if normalized.duplicated(["zone_id", "hour_ts"]).any():
        raise ValueError("Hourly table contains duplicate zone-hour rows")

    scaled = apply_scaling(normalized, zone_stats)
    try:
        zone_index = {int(zone_id): int(index) for zone_id, index in zone_stats["zone_index"].items()}
    except KeyError as error:
        raise KeyError("zone_stats must contain a zone_index mapping") from error
    if not zone_index:
        raise ValueError("zone_stats contains no zones")
    unknown_zones = set(scaled["zone_id"].astype(int)) - set(zone_index)
    if unknown_zones:
        raise ValueError(f"Missing zone indices for zones: {sorted(unknown_zones)}")

    boundaries = {name: _parse_boundaries(splits, name) for name in ("train", "val", "test")}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for split_name, (split_start, split_end) in boundaries.items():
        x_rows: list[np.ndarray] = []
        scaled_targets: list[np.ndarray] = []
        raw_targets: list[np.ndarray] = []
        origins: list[np.datetime64] = []
        sample_zone_ids: list[int] = []

        for zone_id in sorted(zone_index):
            zone = scaled[scaled["zone_id"].astype(int) == zone_id].sort_values("hour_ts")
            zone = zone.set_index("hour_ts")
            zone["hour_ts"] = zone.index
            timeline = pd.DatetimeIndex(zone.index)
            if timeline.empty:
                continue
            first_target = split_start
            if split_name == "train":
                first_target += pd.Timedelta(hours=configured_lookback)
            candidate_starts = timeline[
                (timeline >= first_target) & (timeline <= split_end)
            ]
            for target_start in candidate_starts:
                target_hours = pd.date_range(target_start, periods=horizon, freq="h")
                target_end = target_hours[-1]
                if target_end > split_end:
                    continue
                origin = target_start - pd.Timedelta(hours=1)
                input_hours = pd.date_range(
                    origin - pd.Timedelta(hours=configured_lookback - 1),
                    origin,
                    freq="h",
                )
                if not input_hours.isin(timeline).all() or not target_hours.isin(timeline).all():
                    raise ValueError(f"Zone {zone_id} does not have a complete hourly timeline")

                input_frame = zone.loc[input_hours]
                target_frame = zone.loc[target_hours]
                x_rows.append(np.asarray([_feature_row(row) for _, row in input_frame.iterrows()]))
                scaled_targets.append(
                    np.log1p(target_frame["pickups"].to_numpy(dtype=np.float64))
                )
                stats = zone_stats["stats"][str(zone_id)]
                scaled_targets[-1] = (
                    scaled_targets[-1] - float(stats["mean_log"])
                ) / float(stats["std_log"])
                raw_targets.append(target_frame["pickups"].to_numpy(dtype=np.float64))
                origins.append(np.datetime64(origin.to_datetime64()))
                sample_zone_ids.append(zone_id)

        if x_rows:
            outputs[split_name] = {
                "X": np.asarray(x_rows, dtype=np.float32),
                "zone_idx": np.asarray([zone_index[z] for z in sample_zone_ids], dtype=np.int64),
                "y_scaled": np.asarray(scaled_targets, dtype=np.float32),
                "y_raw": np.asarray(raw_targets, dtype=np.float32),
                "origin_ts": np.asarray(origins, dtype="datetime64[ns]"),
                "zone_id": np.asarray(sample_zone_ids, dtype=np.int16),
            }
        else:
            outputs[split_name] = _empty_windows(configured_lookback, horizon)

        if any(np.isnan(outputs[split_name][key]).any() for key in ("X", "y_scaled", "y_raw")):
            raise ValueError(f"{split_name} windows contain NaN values")

    return outputs

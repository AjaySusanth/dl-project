"""Fit and apply train-only demand and duration statistics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("zone_id", "hour_ts", "pickups", "mean_duration_min")


def _validate_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of the hourly table after schema checks."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(hourly.columns))
    if missing:
        raise ValueError("Hourly table is missing required columns: " + ", ".join(missing))

    result = hourly.loc[:, list(REQUIRED_COLUMNS)].copy()
    result["hour_ts"] = pd.to_datetime(result["hour_ts"], errors="coerce")
    if result["hour_ts"].isna().any():
        raise ValueError("Hourly table contains invalid hour_ts values")
    if result[["zone_id", "pickups"]].isna().any().any():
        raise ValueError("Hourly table contains missing zone_id or pickups values")
    if (result["pickups"] < 0).any():
        raise ValueError("Hourly table contains negative pickup values")
    return result


def _train_mask(hourly: pd.DataFrame, splits: Mapping[str, Any]) -> pd.Series:
    """Build an inclusive mask for the configured training range."""
    try:
        train_start, train_end = splits["train"]
    except (KeyError, ValueError) as error:
        raise ValueError("splits must contain train [start, end] boundaries") from error

    start = pd.to_datetime(train_start, errors="coerce")
    end = pd.to_datetime(train_end, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Train split boundaries must be valid and ordered")
    return hourly["hour_ts"].between(start, end, inclusive="both")


def fit_zone_stats(
    hourly: pd.DataFrame,
    splits: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit per-zone log-demand statistics using training rows only."""
    normalized = _validate_hourly(hourly)
    train = normalized.loc[_train_mask(normalized, splits)]
    if train.empty:
        raise ValueError("Training split contains no hourly rows")

    try:
        percentile = float(config["evaluation"]["peak_percentile"])
    except KeyError as error:
        raise KeyError(f"Missing required configuration key: {error}") from error
    if not 0 <= percentile <= 100:
        raise ValueError("evaluation.peak_percentile must be between 0 and 100")

    zone_ids = sorted(int(zone_id) for zone_id in train["zone_id"].unique())
    stats: dict[str, dict[str, float]] = {}
    for zone_id in zone_ids:
        values = np.log1p(
            train.loc[train["zone_id"] == zone_id, "pickups"].to_numpy(dtype=np.float64)
        )
        mean_log = float(np.mean(values))
        std_log = float(np.std(values))
        if std_log == 0:
            raise ValueError(f"Zone {zone_id} has zero train-period log-demand standard deviation")
        raw_pickups = train.loc[train["zone_id"] == zone_id, "pickups"].to_numpy(
            dtype=np.float64
        )
        stats[str(zone_id)] = {
            "mean_log": mean_log,
            "std_log": std_log,
            "peak_threshold": float(np.percentile(raw_pickups, percentile)),
        }

    return {
        "zone_ids": zone_ids,
        "zone_index": {str(zone_id): index for index, zone_id in enumerate(zone_ids)},
        "stats": stats,
    }


def apply_scaling(hourly: pd.DataFrame, zone_stats: Mapping[str, Any]) -> pd.DataFrame:
    """Add standardized log-demand values using fitted zone statistics."""
    normalized = _validate_hourly(hourly)
    try:
        stats = zone_stats["stats"]
    except KeyError as error:
        raise KeyError("zone_stats must contain a stats mapping") from error

    def scale_row(row: pd.Series) -> float:
        key = str(int(row["zone_id"]))
        if key not in stats:
            raise ValueError(f"Missing fitted statistics for zone {key}")
        values = stats[key]
        return float(
            (np.log1p(float(row["pickups"])) - float(values["mean_log"]))
            / float(values["std_log"])
        )

    normalized["z_pickups"] = normalized.apply(scale_row, axis=1).astype("float32")
    return normalized


def _weighted_mean(frame: pd.DataFrame) -> float:
    """Return a pickup-weighted duration mean, or NaN without valid weight."""
    valid = frame.dropna(subset=["mean_duration_min"])
    valid = valid[valid["pickups"] > 0]
    if valid.empty or valid["pickups"].sum() <= 0:
        return float("nan")
    return float(
        np.average(
            valid["mean_duration_min"].to_numpy(dtype=np.float64),
            weights=valid["pickups"].to_numpy(dtype=np.float64),
        )
    )


def compute_busy_time(
    hourly: pd.DataFrame, splits: Mapping[str, Any]
) -> pd.DataFrame:
    """Compute train-only pickup-weighted duration by zone and hour of day."""
    normalized = _validate_hourly(hourly)
    train = normalized.loc[_train_mask(normalized, splits)].copy()
    if train.empty:
        raise ValueError("Training split contains no hourly rows")
    train["hour_of_day"] = train["hour_ts"].dt.hour.astype("int8")

    zone_ids = sorted(int(zone_id) for zone_id in train["zone_id"].unique())
    zone_means = {
        int(zone_id): _weighted_mean(group)
        for zone_id, group in train.groupby("zone_id", sort=False)
    }
    global_mean = _weighted_mean(train)
    if np.isnan(global_mean):
        raise ValueError("Training data contains no valid pickup-weighted durations")

    rows: list[dict[str, float | int]] = []
    for zone_id in zone_ids:
        zone_frame = train[train["zone_id"] == zone_id]
        zone_mean = zone_means[zone_id]
        for hour_of_day in range(24):
            hour_frame = zone_frame[zone_frame["hour_of_day"] == hour_of_day]
            value = _weighted_mean(hour_frame)
            if np.isnan(value):
                value = zone_mean if not np.isnan(zone_mean) else global_mean
            rows.append(
                {
                    "zone_id": zone_id,
                    "hour_of_day": hour_of_day,
                    "mean_duration_min": value,
                }
            )

    return pd.DataFrame(rows).astype(
        {"zone_id": "int16", "hour_of_day": "int8", "mean_duration_min": "float32"}
    )

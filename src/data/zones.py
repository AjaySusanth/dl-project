"""Select modeling zones from the complete hourly demand table."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

HOURLY_COLUMNS = ("zone_id", "hour_ts", "pickups", "mean_duration_min")


def _select_evenly(values: list[int], count: int) -> list[int]:
    """Select deterministic, evenly spaced values from an ordered list."""
    if count == 0:
        return []
    if count > len(values):
        raise ValueError(f"Cannot select {count} values from a tier of size {len(values)}")
    positions = np.linspace(0, len(values) - 1, num=count, dtype=int)
    return [values[position] for position in positions]


def select_zones(
    hourly: pd.DataFrame,
    config: Mapping[str, Any],
    zone_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Select configured high, medium, and low volume zones.

    Selection uses only the chronological training portion of ``hourly``.
    The returned table retains all timestamps for the selected zones.
    """
    missing_columns = sorted(set(HOURLY_COLUMNS) - set(hourly.columns))
    if missing_columns:
        raise ValueError(
            "Hourly table is missing required columns: " + ", ".join(missing_columns)
        )
    if "LocationID" not in zone_lookup.columns or "Zone" not in zone_lookup.columns:
        raise ValueError("Zone lookup must contain LocationID and Zone columns")

    try:
        fractions = config["split"]["fractions"]
        n_high = int(config["zones"]["n_high"])
        n_medium = int(config["zones"]["n_medium"])
        n_low = int(config["zones"]["n_low"])
        min_nonzero = float(config["zones"]["min_nonzero_hour_fraction"])
    except KeyError as error:
        raise KeyError(f"Missing required configuration key: {error}") from error

    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split.fractions must contain three values summing to 1.0")
    if min_nonzero < 0 or min_nonzero > 1:
        raise ValueError("zones.min_nonzero_hour_fraction must be between 0 and 1")
    requested = (n_high, n_medium, n_low)
    if any(count < 0 for count in requested):
        raise ValueError("Zone tier counts cannot be negative")

    working = hourly.loc[:, list(HOURLY_COLUMNS)].copy()
    working["hour_ts"] = pd.to_datetime(working["hour_ts"], errors="coerce")
    if working[["zone_id", "hour_ts", "pickups"]].isna().any().any():
        raise ValueError("Hourly table contains missing zone, timestamp, or pickup values")
    timeline = pd.Index(working["hour_ts"].drop_duplicates().sort_values())
    if timeline.empty:
        raise ValueError("Hourly table contains no timestamps")

    train_size = int(len(timeline) * float(fractions[0]))
    if train_size < 1:
        raise ValueError("Training fraction produces no training hours")
    train_hours = set(timeline[:train_size])
    train = working[working["hour_ts"].isin(train_hours)]

    stats = (
        train.groupby("zone_id")["pickups"]
        .agg(total_pickups="sum", nonzero_hours=lambda values: (values > 0).sum())
        .assign(nonzero_fraction=lambda frame: frame["nonzero_hours"] / len(train_hours))
    )
    eligible = stats[stats["nonzero_fraction"] >= min_nonzero].sort_values(
        ["total_pickups", "zone_id"], ascending=[False, True]
    )
    tier_values = np.array_split(eligible.index.to_list(), 3)
    selected: list[int] = []
    for tier, count in zip(tier_values, requested):
        tier_ids = [int(zone_id) for zone_id in tier]
        if count > len(tier_ids):
            raise ValueError(
                f"Requested {count} zones from a tier with only {len(tier_ids)} eligible zones"
            )
        selected.extend(_select_evenly(tier_ids, count))

    if not selected:
        raise ValueError("Zone selection produced no zones")

    selected_set = set(selected)
    result = working[working["zone_id"].isin(selected_set)].copy()
    result["zone_id"] = result["zone_id"].astype("int16")
    result["pickups"] = result["pickups"].astype("int32")
    result["mean_duration_min"] = result["mean_duration_min"].astype("float32")
    result = result.sort_values(["zone_id", "hour_ts"], ignore_index=True)

    names = zone_lookup.set_index("LocationID")["Zone"].to_dict()
    LOGGER.info(
        "Selected zones: %s",
        [
            {"zone_id": zone_id, "name": names.get(zone_id, "<unknown>")}
            for zone_id in sorted(selected_set)
        ],
    )
    return result[list(HOURLY_COLUMNS)]

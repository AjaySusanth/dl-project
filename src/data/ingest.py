"""Load raw TLC trip records for the configured months."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

RAW_COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
)
LOOKUP_FILENAME = "taxi_zone_lookup.csv"


def _raw_file_path(raw_dir: Path, month: str) -> Path:
    """Return the expected Yellow Taxi Parquet path for one month."""
    return raw_dir / f"yellow_tripdata_{month}.parquet"


def load_trips(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load configured Yellow Taxi Parquet files into one raw trip table.

    The returned table contains only the architecture-defined raw columns:
    pickup timestamp, dropoff timestamp, and pickup location ID.
    """
    try:
        raw_dir = Path(config["paths"]["raw_dir"])
        months = config["data"]["months"]
    except KeyError as error:
        raise KeyError(f"Missing required configuration key: {error}") from error

    if not isinstance(months, list) or not all(
        isinstance(month, str) and month for month in months
    ):
        raise ValueError("config['data']['months'] must be a non-empty list of strings")
    if not months:
        raise ValueError("config['data']['months'] must contain at least one month")

    frames: list[pd.DataFrame] = []
    missing_files: list[Path] = []
    for month in months:
        path = _raw_file_path(raw_dir, month)
        if not path.is_file():
            missing_files.append(path)
            continue

        available_columns = set(pq.read_schema(path).names)
        missing_columns = [
            column for column in RAW_COLUMNS if column not in available_columns
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = pd.read_parquet(path, columns=list(RAW_COLUMNS))
        frames.append(frame)

    for path in missing_files:
        LOGGER.warning("Configured raw data file is missing: %s", path)

    if not frames:
        expected = ", ".join(str(_raw_file_path(raw_dir, month)) for month in months)
        raise FileNotFoundError(f"No configured raw trip files were found. Expected: {expected}")

    return pd.concat(frames, ignore_index=True)


def clean_trips(
    trips: pd.DataFrame,
    config: Mapping[str, Any],
    zone_lookup: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean raw trips and return the cleaned table with rule counts.

    The returned table retains the three raw columns and normalized naive
    datetime values. ``zone_lookup`` may be supplied by tests or callers;
    otherwise it is loaded from ``paths.raw_dir/taxi_zone_lookup.csv``.
    """
    required = set(RAW_COLUMNS)
    missing_columns = sorted(required - set(trips.columns))
    if missing_columns:
        raise ValueError(
            "Raw trip table is missing required columns: "
            + ", ".join(missing_columns)
        )

    try:
        months = config["data"]["months"]
        min_duration = float(config["data"]["min_duration_min"])
        max_duration = float(config["data"]["max_duration_min"])
        raw_dir = Path(config["paths"]["raw_dir"])
    except KeyError as error:
        raise KeyError(f"Missing required configuration key: {error}") from error

    if not months or not all(isinstance(month, str) and month for month in months):
        raise ValueError("config['data']['months'] must be a non-empty list of strings")
    if min_duration < 0 or max_duration < min_duration:
        raise ValueError("Duration limits must satisfy 0 <= min_duration <= max_duration")

    if zone_lookup is None:
        lookup_path = raw_dir / LOOKUP_FILENAME
        if not lookup_path.is_file():
            raise FileNotFoundError(f"Taxi zone lookup file does not exist: {lookup_path}")
        zone_lookup = pd.read_csv(lookup_path)
    if "LocationID" not in zone_lookup.columns:
        raise ValueError("Taxi zone lookup must contain a LocationID column")

    cleaned = trips.loc[:, list(RAW_COLUMNS)].copy()
    report = {"input_rows": len(cleaned)}

    pickup = pd.to_datetime(cleaned["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(cleaned["tpep_dropoff_datetime"], errors="coerce")
    cleaned["tpep_pickup_datetime"] = pickup
    cleaned["tpep_dropoff_datetime"] = dropoff

    missing_mask = (
        pickup.isna()
        | dropoff.isna()
        | cleaned["PULocationID"].isna()
    )
    report["dropped_missing_fields"] = int(missing_mask.sum())
    cleaned = cleaned.loc[~missing_mask].copy()

    pickup_months = cleaned["tpep_pickup_datetime"].dt.strftime("%Y-%m")
    outside_month_mask = ~pickup_months.isin(months)
    report["dropped_outside_configured_months"] = int(outside_month_mask.sum())
    cleaned = cleaned.loc[~outside_month_mask].copy()

    duration_min = (
        cleaned["tpep_dropoff_datetime"] - cleaned["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    invalid_duration_mask = (
        (duration_min < min_duration)
        | (duration_min > max_duration)
        | duration_min.isna()
    )
    report["dropped_invalid_duration"] = int(invalid_duration_mask.sum())
    cleaned = cleaned.loc[~invalid_duration_mask].copy()

    valid_zone_ids = set(zone_lookup["LocationID"].dropna().astype("int64"))
    unknown_zone_mask = ~cleaned["PULocationID"].isin(valid_zone_ids)
    report["dropped_unknown_zone_id"] = int(unknown_zone_mask.sum())
    cleaned = cleaned.loc[~unknown_zone_mask].copy()

    report["output_rows"] = len(cleaned)
    for key, count in report.items():
        if key.startswith("dropped_") and count:
            LOGGER.info("%s: %d rows", key, count)
    return cleaned.reset_index(drop=True), report


def aggregate_hourly(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned trips into a complete zone-by-hour demand table."""
    required = set(RAW_COLUMNS)
    missing_columns = sorted(required - set(trips.columns))
    if missing_columns:
        raise ValueError(
            "Cleaned trip table is missing required columns: "
            + ", ".join(missing_columns)
        )
    if trips.empty:
        raise ValueError("Cannot aggregate an empty cleaned trip table")

    cleaned = trips.loc[:, list(RAW_COLUMNS)].copy()
    cleaned["tpep_pickup_datetime"] = pd.to_datetime(
        cleaned["tpep_pickup_datetime"], errors="coerce"
    )
    cleaned["tpep_dropoff_datetime"] = pd.to_datetime(
        cleaned["tpep_dropoff_datetime"], errors="coerce"
    )
    if cleaned[list(RAW_COLUMNS)].isna().any().any():
        raise ValueError("Cleaned trips contain missing required values")

    cleaned["hour_ts"] = cleaned["tpep_pickup_datetime"].dt.floor("h")
    cleaned["duration_min"] = (
        cleaned["tpep_dropoff_datetime"] - cleaned["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    if cleaned["duration_min"].isna().any():
        raise ValueError("Cleaned trips contain invalid duration values")

    zone_ids = sorted(cleaned["PULocationID"].astype("int64").unique())
    hours = pd.date_range(
        start=cleaned["hour_ts"].min(),
        end=cleaned["hour_ts"].max(),
        freq="h",
    )
    grid = pd.MultiIndex.from_product(
        [zone_ids, hours], names=["zone_id", "hour_ts"]
    ).to_frame(index=False)

    grouped = (
        cleaned.groupby(["PULocationID", "hour_ts"], as_index=False)
        .agg(
            pickups=("PULocationID", "size"),
            mean_duration_min=("duration_min", "mean"),
        )
        .rename(columns={"PULocationID": "zone_id"})
    )
    hourly = grid.merge(grouped, on=["zone_id", "hour_ts"], how="left")
    hourly["pickups"] = hourly["pickups"].fillna(0).astype("int32")
    hourly["mean_duration_min"] = hourly["mean_duration_min"].astype("float32")
    hourly["zone_id"] = hourly["zone_id"].astype("int16")
    hourly["hour_ts"] = pd.to_datetime(hourly["hour_ts"]).astype("datetime64[ns]")
    return hourly[
        ["zone_id", "hour_ts", "pickups", "mean_duration_min"]
    ].sort_values(["zone_id", "hour_ts"], ignore_index=True)

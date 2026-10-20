"""Tests for raw-trip cleaning and hourly aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.data.ingest import RAW_COLUMNS, aggregate_hourly, clean_trips


@pytest.fixture
def cleaning_config() -> dict:
    """Return a compact configuration for deterministic cleaning tests."""
    return {
        "paths": {"raw_dir": "data/raw"},
        "data": {
            "months": ["2026-01"],
            "min_duration_min": 1,
            "max_duration_min": 180,
        },
    }


@pytest.fixture
def zone_lookup() -> pd.DataFrame:
    """Return the valid synthetic pickup zones."""
    return pd.DataFrame({"LocationID": [10, 20]})


def test_clean_trips_applies_each_rule(
    cleaning_config: dict, zone_lookup: pd.DataFrame
) -> None:
    """Each cleaning rule removes exactly the expected synthetic rows."""
    rows = [
        (datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 5), 10),
        (None, datetime(2026, 1, 1, 0, 5), 10),
        (datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 5), None),
        (datetime(2026, 2, 1), datetime(2026, 2, 1, 0, 5), 10),
        (datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 30), 10),
        (datetime(2026, 1, 1), datetime(2026, 1, 1, 3, 1), 10),
        (datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 5), 99),
    ]
    trips = pd.DataFrame(rows, columns=RAW_COLUMNS)

    cleaned, report = clean_trips(trips, cleaning_config, zone_lookup)

    assert len(cleaned) == 1
    assert report == {
        "input_rows": 7,
        "dropped_missing_fields": 2,
        "dropped_outside_configured_months": 1,
        "dropped_invalid_duration": 2,
        "dropped_unknown_zone_id": 1,
        "output_rows": 1,
    }


def test_aggregate_hourly_is_complete_and_zero_fills() -> None:
    """A two-zone 48-hour input produces a complete 96-row hourly table."""
    start = datetime(2026, 1, 1)
    rows = []
    for zone in (10, 20):
        for offset in range(48):
            hour = start + timedelta(hours=offset)
            if not (zone == 20 and offset == 12):
                rows.append(
                    (hour + timedelta(minutes=5), hour + timedelta(minutes=25), zone)
                )
            if zone == 10 and offset == 0:
                rows.append(
                    (hour + timedelta(minutes=10), hour + timedelta(minutes=40), zone)
                )

    result = aggregate_hourly(pd.DataFrame(rows, columns=RAW_COLUMNS))

    assert len(result) == 96
    assert result.groupby("zone_id").size().tolist() == [48, 48]
    zero_row = result[
        (result["zone_id"] == 20)
        & (result["hour_ts"] == pd.Timestamp(start + timedelta(hours=12)))
    ].iloc[0]
    assert zero_row["pickups"] == 0
    assert pd.isna(zero_row["mean_duration_min"])


def test_aggregate_hourly_duration_is_nan_only_for_zero_pickups() -> None:
    """Observed pickups have a duration, while zero-filled hours have NaN."""
    trips = pd.DataFrame(
        [
            (
                datetime(2026, 1, 1, 0, 5),
                datetime(2026, 1, 1, 0, 25),
                10,
            ),
            (
                datetime(2026, 1, 1, 2, 5),
                datetime(2026, 1, 1, 2, 25),
                10,
            ),
        ],
        columns=RAW_COLUMNS,
    )

    result = aggregate_hourly(trips)

    assert ((result["pickups"] == 0) == result["mean_duration_min"].isna()).all()


def test_aggregate_hourly_schema_and_dtypes() -> None:
    """The hourly output matches the architecture schema and dtypes."""
    trips = pd.DataFrame(
        [
            (
                datetime(2026, 1, 1),
                datetime(2026, 1, 1, 0, 5),
                10,
            )
        ],
        columns=RAW_COLUMNS,
    )

    result = aggregate_hourly(trips)

    assert list(result.columns) == [
        "zone_id",
        "hour_ts",
        "pickups",
        "mean_duration_min",
    ]
    assert result.dtypes["zone_id"] == "int16"
    assert result.dtypes["hour_ts"] == "datetime64[ns]"
    assert result.dtypes["pickups"] == "int32"
    assert result.dtypes["mean_duration_min"] == "float32"


def test_ingestion_cli_returns_nonzero_for_missing_files(tmp_path: Path) -> None:
    """The CLI warns and fails when no configured raw file exists."""
    (tmp_path / "raw").mkdir()
    default_config = tmp_path / "default.yaml"
    smoke_config = tmp_path / "missing.yaml"
    config_text = (
        "paths:\n"
        "  raw_dir: raw\n"
        "  processed_dir: processed\n"
        "data:\n"
        "  months: ['2099-01']\n"
    )
    default_config.write_text(config_text, encoding="utf-8")
    smoke_config.write_text(config_text, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest.py",
            "--config",
            str(smoke_config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Configured raw data file is missing" in result.stderr
    assert "No configured raw trip files were found" in result.stderr

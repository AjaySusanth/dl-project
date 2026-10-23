"""Convert pickup forecasts into fleet requirements."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

FLEET_COLUMNS = (
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
)


def required_taxis(pickups: float, busy_min: float, utilization: float) -> int:
    """Return taxis needed for demand, trip duration, and utilization."""
    values = (pickups, busy_min, utilization)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("pickups, busy_min, and utilization must be finite")
    if pickups < 0:
        raise ValueError("pickups must be non-negative")
    if busy_min <= 0:
        raise ValueError("busy_min must be positive")
    if utilization <= 0 or utilization > 1:
        raise ValueError("utilization must be in (0, 1]")
    return int(ceil(pickups * busy_min / (60 * utilization)))


def fleet_table(
    origin_ts: Any,
    zone_id: int,
    q_pred: np.ndarray,
    busy_time: pd.DataFrame,
    available: int,
    deadhead_min: float,
    utilization: float,
) -> pd.DataFrame:
    """Build one fleet-requirement row for each forecast horizon."""
    predictions = np.asarray(q_pred, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] < 3 or predictions.shape[0] < 1:
        raise ValueError("q_pred must have shape (H, Q) with Q >= 3")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("q_pred must contain finite, non-negative values")
    if not isinstance(zone_id, (int, np.integer)):
        raise ValueError("zone_id must be an integer")
    if available < 0:
        raise ValueError("available must be non-negative")
    if not np.isfinite(deadhead_min) or deadhead_min < 0:
        raise ValueError("deadhead_min must be finite and non-negative")
    pd_origin = pd.to_datetime(origin_ts, errors="coerce")
    if pd.isna(pd_origin):
        raise ValueError("origin_ts must be a valid timestamp")

    required_busy_columns = {"zone_id", "hour_of_day", "mean_duration_min"}
    missing = sorted(required_busy_columns - set(busy_time.columns))
    if missing:
        raise ValueError("busy_time is missing required columns: " + ", ".join(missing))
    matching = busy_time[
        (busy_time["zone_id"].astype(int) == int(zone_id))
        & busy_time["hour_of_day"].between(0, 23)
    ]
    if matching.empty:
        raise ValueError(f"busy_time has no rows for zone {zone_id}")
    duration_by_hour = matching.drop_duplicates("hour_of_day").set_index("hour_of_day")[
        "mean_duration_min"
    ]

    rows: list[dict[str, Any]] = []
    for index, quantiles in enumerate(predictions, start=1):
        hour_ts = pd_origin + pd.Timedelta(hours=index)
        hour_of_day = int(hour_ts.hour)
        if hour_of_day not in duration_by_hour.index:
            raise ValueError(
                f"busy_time has no duration for zone {zone_id}, hour {hour_of_day}"
            )
        duration = float(duration_by_hour.loc[hour_of_day])
        busy_min = duration + float(deadhead_min)
        n_expected = required_taxis(float(quantiles[1]), busy_min, utilization)
        n_safe = required_taxis(float(quantiles[2]), busy_min, utilization)
        gap_expected = n_expected - int(available)
        gap_safe = n_safe - int(available)
        if gap_expected > 0:
            status = "shortage"
        elif gap_safe > 0:
            status = "risk"
        else:
            status = "ok"
        rows.append(
            {
                "horizon": index,
                "hour_ts": hour_ts,
                "p10": float(quantiles[0]),
                "p50": float(quantiles[1]),
                "p90": float(quantiles[2]),
                "busy_min": busy_min,
                "n_expected": n_expected,
                "n_safe": n_safe,
                "available": int(available),
                "gap_expected": gap_expected,
                "gap_safe": gap_safe,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=FLEET_COLUMNS)

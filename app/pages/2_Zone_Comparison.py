"""Multi-zone forecast comparison dashboard page."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir()),
    _Path(__file__).resolve().parent.parent,
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path

import pandas as pd
import streamlit as st

from app.loaders import load_app_config, load_app_model, load_busy_time, load_predictions, load_zone_lookup
from src.fleet.fleet import fleet_table

def _show_error(value: object) -> bool:
    """Display a loader error and return whether execution should stop."""
    if isinstance(value, str):
        st.error(value)
        return True
    return False

st.title("Zone comparison")
st.info("All forecasts are saved test-period predictions; no live data is used.")

config = load_app_config("config/default.yaml")
if _show_error(config):
    st.stop()
paths = config["paths"]
app_model = load_app_model(Path(paths["artifacts_dir"]) / "app_model.json")
if _show_error(app_model):
    st.stop()
predictions = load_predictions(
    Path(paths["artifacts_dir"]) / app_model["tag"] / "pred_test.npz"
)
busy_time = load_busy_time(Path(paths["processed_dir"]) / "busy_time.parquet")
lookup = load_zone_lookup(Path(paths["raw_dir"]) / "taxi_zone_lookup.csv")
for loaded in (predictions, busy_time, lookup):
    if _show_error(loaded):
        st.stop()

names = {
    int(row.LocationID): str(row.Zone)
    for row in lookup.itertuples()
    if pd.notna(row.LocationID)
}
zone_ids = sorted({int(zone) for zone in predictions["zone_id"]})
selected_zones = st.multiselect(
    "Zones",
    zone_ids,
    default=zone_ids[: min(3, len(zone_ids))],
    format_func=lambda zone: f"{zone} — {names.get(zone, 'Unknown zone')}",
)
origins = sorted({pd.Timestamp(origin) for origin in predictions["origin_ts"]})
selected_origin = st.selectbox("Forecast origin", origins)
available = st.number_input("Available taxis", min_value=0, value=25, step=1)
deadhead = st.number_input(
    "Deadhead minutes",
    min_value=0.0,
    value=float(config["fleet"]["deadhead_min"]),
)
utilization = st.number_input(
    "Utilization",
    min_value=0.01,
    max_value=1.0,
    value=float(config["fleet"]["utilization"]),
    step=0.05,
)

rows = []
for zone_id in selected_zones:
    matches = [
        index
        for index, (origin, zone) in enumerate(
            zip(predictions["origin_ts"], predictions["zone_id"])
        )
        if int(zone) == zone_id and pd.Timestamp(origin) == selected_origin
    ]
    if len(matches) != 1:
        st.warning(f"No unique saved prediction for zone {zone_id} at {selected_origin}.")
        continue
    rows.append(
        fleet_table(
            selected_origin,
            zone_id,
            predictions["q_pred"][matches[0]],
            busy_time,
            int(available),
            float(deadhead),
            float(utilization),
        ).assign(zone_id=zone_id)
    )

if rows:
    comparison = pd.concat(rows, ignore_index=True)
    st.dataframe(
        comparison[
            ["zone_id", "horizon", "hour_ts", "p50", "p90", "n_expected", "n_safe", "status"]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Select at least one zone with a saved forecast.")

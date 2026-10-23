"""Forecast and fleet-planning dashboard page."""

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
import plotly.graph_objects as go
import streamlit as st

from app.loaders import (
    load_app_config,
    load_app_model,
    load_busy_time,
    load_hourly,
    load_predictions,
    load_zone_lookup,
)
from src.fleet.fleet import fleet_table

def _show_error(value: object) -> bool:
    """Display a loader error and return whether execution should stop."""
    if isinstance(value, str):
        st.error(value)
        return True
    return False

st.title("Forecast")
st.info("No live data is used. Forecasts and actuals shown here are from the saved test period.")

config = load_app_config("config/default.yaml")
if _show_error(config):
    st.stop()

paths = config["paths"]
app_model = load_app_model(Path(paths["artifacts_dir"]) / "app_model.json")
if _show_error(app_model):
    st.stop()
tag = app_model["tag"]
predictions = load_predictions(Path(paths["artifacts_dir"]) / tag / "pred_test.npz")
hourly = load_hourly(Path(paths["processed_dir"]) / "hourly_selected.parquet")
busy_time = load_busy_time(Path(paths["processed_dir"]) / "busy_time.parquet")
lookup = load_zone_lookup(Path(paths["raw_dir"]) / "taxi_zone_lookup.csv")
for loaded in (predictions, hourly, busy_time, lookup):
    if _show_error(loaded):
        st.stop()

prediction_zone_ids = sorted({int(zone) for zone in predictions["zone_id"]})
lookup_names = {
    int(row.LocationID): str(row.Zone)
    for row in lookup.itertuples()
    if pd.notna(row.LocationID)
}
zone_options = {
    zone_id: f"{zone_id} — {lookup_names.get(zone_id, 'Unknown zone')}"
    for zone_id in prediction_zone_ids
}
selected_zone = st.selectbox(
    "Zone",
    prediction_zone_ids,
    format_func=lambda zone: zone_options[zone],
)
origins = sorted(
    {
        pd.Timestamp(origin)
        for origin, zone in zip(predictions["origin_ts"], predictions["zone_id"])
        if int(zone) == selected_zone
    }
)
selected_origin = st.selectbox("Forecast origin", origins)

matches = [
    index
    for index, (origin, zone) in enumerate(
        zip(predictions["origin_ts"], predictions["zone_id"])
    )
    if int(zone) == selected_zone and pd.Timestamp(origin) == selected_origin
]
if len(matches) != 1:
    st.error("The selected zone and origin do not identify exactly one saved forecast.")
    st.stop()
prediction_index = matches[0]
quantiles = predictions["q_pred"][prediction_index]

history = hourly[
    (hourly["zone_id"].astype(int) == selected_zone)
    & (pd.to_datetime(hourly["hour_ts"]) <= selected_origin)
].sort_values("hour_ts").tail(48)
forecast_hours = pd.date_range(selected_origin, periods=len(quantiles) + 1, freq="h")[1:]

chart = go.Figure()
chart.add_trace(
    go.Scatter(
        x=history["hour_ts"],
        y=history["pickups"],
        mode="lines",
        name="Actual pickups",
    )
)
chart.add_trace(
    go.Scatter(
        x=forecast_hours,
        y=quantiles[:, 2],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        name="P90",
    )
)
chart.add_trace(
    go.Scatter(
        x=forecast_hours,
        y=quantiles[:, 0],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line={"width": 0},
        name="P10–P90 interval",
    )
)
chart.add_trace(
    go.Scatter(
        x=forecast_hours,
        y=quantiles[:, 1],
        mode="lines+markers",
        name="P50 forecast",
    )
)
chart.update_layout(xaxis_title="Hour", yaxis_title="Pickups", hovermode="x unified")
st.plotly_chart(chart, use_container_width=True)

st.subheader("Fleet requirement")
available = st.number_input("Available taxis", min_value=0, value=25, step=1)
deadhead = st.number_input("Deadhead minutes", min_value=0.0, value=float(config["fleet"]["deadhead_min"]))
utilization = st.number_input(
    "Utilization",
    min_value=0.01,
    max_value=1.0,
    value=float(config["fleet"]["utilization"]),
    step=0.05,
)
table = fleet_table(
    selected_origin,
    selected_zone,
    quantiles,
    busy_time,
    int(available),
    float(deadhead),
    float(utilization),
)
st.dataframe(table, use_container_width=True, hide_index=True)

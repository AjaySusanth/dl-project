"""Model evaluation dashboard page."""

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
import plotly.express as px
import streamlit as st

from app.loaders import load_app_config, load_results

def _show_error(value: object) -> bool:
    """Display a loader error and return whether execution should stop."""
    if isinstance(value, str):
        st.error(value)
        return True
    return False

st.title("Model evaluation")
st.info("Metrics are calculated from saved validation and test-period artifacts.")

config = load_app_config("config/default.yaml")
if _show_error(config):
    st.stop()
results = load_results(Path(config["paths"]["artifacts_dir"]) / "results.json")
if _show_error(results):
    st.stop()

experiment = st.selectbox("Experiment", sorted(results))
experiment_results = results[experiment]
rows = []
for model, splits in experiment_results.items():
    for split, metrics in splits.items():
        rows.append(
            {
                "model": model,
                "split": split,
                "MAE": metrics["mae"]["mean"],
                "RMSE": metrics["rmse"]["mean"],
                "WAPE": metrics["wape"]["mean"],
                "Coverage": metrics["coverage"]["mean"],
            }
        )
summary = pd.DataFrame(rows)
st.dataframe(summary, use_container_width=True, hide_index=True)

plot_rows = summary[summary["MAE"].notna()]
if not plot_rows.empty:
    st.plotly_chart(
        px.bar(plot_rows, x="model", y="MAE", color="split", barmode="group", title="MAE by model"),
        use_container_width=True,
    )
coverage_rows = summary[summary["Coverage"].notna()]
if not coverage_rows.empty:
    st.metric("Mean test coverage", f"{coverage_rows[coverage_rows['split'] == 'test']['Coverage'].mean():.3f}")

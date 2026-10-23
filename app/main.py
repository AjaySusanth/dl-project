"""CrowdCast Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir()),
    _Path(__file__).resolve().parent.parent,
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="CrowdCast", page_icon="🚕", layout="wide")

st.title("CrowdCast")
st.subheader("NYC taxi pickup forecasting and fleet planning")
st.info(
    "This dashboard uses saved test-period forecasts and historical artifacts. "
    "It does not use live data or train models."
)
st.markdown(
    "Use the pages in the sidebar to inspect forecasts, compare zones, and review model evaluation."
)

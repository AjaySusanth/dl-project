"""Cached, user-facing loaders for dashboard artifacts."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = next(
    (p for p in _Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir()),
    _Path(__file__).resolve().parent.parent,
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config

def _missing_message(path: Path, command: str) -> str:
    """Explain a missing artifact and identify the command that creates it."""
    return f"Required file is missing: {path}. Run `{command}` first."

@st.cache_data
def load_app_config(path: str | Path) -> dict[str, Any] | str:
    """Load dashboard configuration or return an actionable error message."""
    config_path = Path(path)
    command = "streamlit run app/main.py"
    if not config_path.is_file():
        return _missing_message(config_path, command)
    try:
        return load_config(config_path)
    except (FileNotFoundError, OSError, ValueError) as error:
        return f"Could not load configuration `{config_path}`: {error}. Run `{command}` after fixing the file."

@st.cache_data
def load_app_model(path: str | Path) -> dict[str, Any] | str:
    """Load the validation-selected application model metadata."""
    return _load_json(path, "python scripts/evaluate.py --config config/default.yaml --experiment all")

@st.cache_data
def load_predictions(path: str | Path) -> dict[str, np.ndarray] | str:
    """Load one saved prediction archive into ordinary NumPy arrays."""
    prediction_path = Path(path)
    if not prediction_path.is_file():
        return _missing_message(
            prediction_path,
            "python scripts/predict.py --config config/default.yaml --model gru --seed 0",
        )
    try:
        with np.load(prediction_path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as error:
        return f"Could not load predictions `{prediction_path}`: {error}"

@st.cache_data
def load_hourly(path: str | Path) -> pd.DataFrame | str:
    """Load hourly pickup data for the forecast history chart."""
    return _load_parquet(path, "python scripts/ingest.py --config config/default.yaml")

@st.cache_data
def load_busy_time(path: str | Path) -> pd.DataFrame | str:
    """Load train-derived busy-time statistics for fleet calculations."""
    return _load_parquet(
        path,
        "python scripts/build_features.py --config config/default.yaml",
    )

@st.cache_data
def load_results(path: str | Path) -> dict[str, Any] | str:
    """Load aggregated experiment results."""
    return _load_json(
        path,
        "python scripts/evaluate.py --config config/default.yaml --experiment all",
    )

@st.cache_data
def load_metrics(path: str | Path) -> dict[str, Any] | str:
    """Load one per-run metric JSON artifact."""
    return _load_json(
        path,
        "python scripts/evaluate.py --config config/default.yaml --experiment all",
    )

@st.cache_data
def load_zone_lookup(path: str | Path) -> pd.DataFrame | str:
    """Load taxi-zone names for dashboard labels."""
    return _load_csv(path, "python scripts/ingest.py --config config/default.yaml")

def _load_json(path: str | Path, command: str) -> dict[str, Any] | str:
    """Load a JSON object or return a user-facing error."""
    json_path = Path(path)
    if not json_path.is_file():
        return _missing_message(json_path, command)
    try:
        with json_path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        return f"Could not load JSON `{json_path}`: {error}"
    if not isinstance(value, dict):
        return f"Could not load JSON `{json_path}`: expected an object."
    return value

def _load_parquet(path: str | Path, command: str) -> pd.DataFrame | str:
    """Load a Parquet table or return a user-facing error."""
    parquet_path = Path(path)
    if not parquet_path.is_file():
        return _missing_message(parquet_path, command)
    try:
        return pd.read_parquet(parquet_path)
    except (OSError, ValueError) as error:
        return f"Could not load Parquet `{parquet_path}`: {error}"

def _load_csv(path: str | Path, command: str) -> pd.DataFrame | str:
    """Load a CSV table or return a user-facing error."""
    csv_path = Path(path)
    if not csv_path.is_file():
        return _missing_message(csv_path, command)
    try:
        return pd.read_csv(csv_path)
    except (OSError, ValueError) as error:
        return f"Could not load CSV `{csv_path}`: {error}"

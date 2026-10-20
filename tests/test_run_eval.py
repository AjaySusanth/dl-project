"""Tests for evaluation aggregation and validation-only model selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate.run_eval import aggregate_results, evaluate_run, pick_app_model


def _stats() -> dict:
    return {
        "stats": {
            "1": {"peak_threshold": 8.0},
            "2": {"peak_threshold": 8.0},
        }
    }


def _prediction() -> dict[str, np.ndarray]:
    return {
        "origin_ts": np.array(["2026-01-01T00:00"], dtype="datetime64[m]"),
        "zone_id": np.array([1]),
        "y_true": np.array([[10.0, 12.0]]),
        "q_pred": np.array([[[8.0, 10.0, 12.0], [10.0, 12.0, 14.0]]]),
    }


def test_evaluate_run_matches_prediction_schema() -> None:
    """Model evaluation produces the documented metric structure."""
    result = evaluate_run(_prediction(), _stats(), "gru_L168_s0", "test")

    assert result["tag"] == "gru_L168_s0"
    assert set(result["per_horizon"]) == {"1", "2"}
    assert result["overall"]["mae"] == pytest.approx(0)
    assert result["overall"]["coverage"] == pytest.approx(1)
    assert result["overall"]["width"] == pytest.approx(4)
    assert result["per_zone"]["1"]["mae"] == pytest.approx(0)


def test_baseline_evaluation_selects_requested_point_forecast() -> None:
    """Baseline artifacts are evaluated by the selected b1/b2 array."""
    prediction = _prediction()
    del prediction["q_pred"]
    prediction["b1"] = np.array([[9.0, 11.0]])
    prediction["b2"] = np.array([[1.0, 1.0]])

    result = evaluate_run(
        prediction, _stats(), "baseline_L168_s0", "test", forecast_key="b1"
    )
    assert result["overall"]["mae"] == pytest.approx(1)


def test_aggregation_groups_seeds_and_selection_uses_validation_only() -> None:
    """Repeated seeds aggregate, and test MAE cannot choose the app model."""
    results = [
        {"tag": "lstm_L168_s0", "split": "val", "overall": {"mae": 2}},
        {"tag": "lstm_L168_s1", "split": "val", "overall": {"mae": 4}},
        {"tag": "gru_L168_s0", "split": "val", "overall": {"mae": 3}},
        {"tag": "lstm_L168_s0", "split": "test", "overall": {"mae": 100}},
        {"tag": "gru_L168_s0", "split": "test", "overall": {"mae": 1}},
    ]
    aggregated = aggregate_results(results, "E1")

    assert aggregated["E1"]["lstm_L168"]["val"]["mae"]["mean"] == pytest.approx(3)
    assert aggregated["E1"]["lstm_L168"]["val"]["mae"]["std"] == pytest.approx(1)
    assert pick_app_model(results)["tag"] == "lstm_L168_s0"

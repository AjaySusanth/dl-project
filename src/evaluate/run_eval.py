"""Evaluate prediction artifacts and aggregate experiment results."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluate.metrics import coverage, interval_width, mae, peak_mae, rmse, wape

METRIC_NAMES = ("mae", "rmse", "wape", "peak_mae", "coverage", "width")


def _load_prediction(prediction: Mapping[str, Any] | str | Path) -> dict[str, np.ndarray]:
    """Load a prediction mapping or NumPy archive into ordinary arrays."""
    if isinstance(prediction, (str, Path)):
        with np.load(prediction, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    return {key: np.asarray(value) for key, value in prediction.items()}


def _validate_predictions(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return origins, zones, and true targets."""
    required = {"origin_ts", "zone_id", "y_true"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("Prediction artifact is missing: " + ", ".join(missing))
    origin_ts = np.asarray(data["origin_ts"])
    zone_id = np.asarray(data["zone_id"])
    y_true = np.asarray(data["y_true"], dtype=np.float64)
    if origin_ts.ndim != 1 or zone_id.ndim != 1 or y_true.ndim != 2:
        raise ValueError("origin_ts and zone_id must be 1-D; y_true must be 2-D")
    if y_true.shape[0] != len(origin_ts) or len(zone_id) != len(origin_ts):
        raise ValueError("Prediction arrays have inconsistent sample counts")
    if y_true.size == 0 or not np.isfinite(y_true).all():
        raise ValueError("y_true must contain finite, non-empty values")
    return origin_ts, zone_id, y_true


def _peak_thresholds(zone_stats: Mapping[str, Any], zone_ids: np.ndarray) -> np.ndarray:
    """Resolve each sample's train-derived zone peak threshold."""
    try:
        stats = zone_stats["stats"]
    except KeyError as error:
        raise KeyError("zone_stats must contain stats") from error
    thresholds = []
    for zone in zone_ids:
        key = str(int(zone))
        if key not in stats or "peak_threshold" not in stats[key]:
            raise ValueError(f"Missing peak threshold for zone {key}")
        thresholds.append(float(stats[key]["peak_threshold"]))
    return np.asarray(thresholds, dtype=np.float64)


def _metric_values(
    actual: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    thresholds: np.ndarray,
) -> dict[str, float]:
    """Calculate all metrics for one group of samples."""
    threshold_values = thresholds
    if actual.ndim == thresholds.ndim + 1:
        threshold_values = thresholds[:, None]
    if actual.shape != threshold_values.shape and not (
        actual.ndim == 2 and threshold_values.shape == (actual.shape[0], 1)
    ):
        raise ValueError("Peak thresholds are not aligned with actual values")
    peak_mask = actual >= threshold_values
    values: dict[str, float | None] = {
        "mae": mae(actual, point),
        "rmse": rmse(actual, point),
        "wape": wape(actual, point),
        "coverage": None,
        "width": None,
    }
    if peak_mask.any():
        values["peak_mae"] = mae(actual[peak_mask], point[peak_mask])
    else:
        values["peak_mae"] = None
    if lower is not None and upper is not None:
        values["coverage"] = coverage(actual, lower, upper)
        values["width"] = interval_width(lower, upper)
    return values


def evaluate_run(
    prediction: Mapping[str, Any] | str | Path,
    zone_stats: Mapping[str, Any],
    tag: str,
    split: str,
    *,
    forecast_key: str | None = None,
    train_time_sec: float = 0.0,
    n_params: int = 0,
) -> dict[str, Any]:
    """Evaluate one model or baseline prediction artifact."""
    data = _load_prediction(prediction)
    _, zone_ids, y_true = _validate_predictions(data)
    thresholds = _peak_thresholds(zone_stats, zone_ids)

    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    if "q_pred" in data:
        quantiles = np.asarray(data["q_pred"], dtype=np.float64)
        if quantiles.ndim != 3 or quantiles.shape[:2] != y_true.shape or quantiles.shape[2] < 3:
            raise ValueError("q_pred must have shape (N, H, Q) with Q >= 3")
        lower, point, upper = quantiles[:, :, 0], quantiles[:, :, 1], quantiles[:, :, 2]
    else:
        candidates = [key for key in ("b1", "b2") if key in data]
        selected = forecast_key or (candidates[0] if len(candidates) == 1 else None)
        if selected not in candidates:
            raise ValueError("forecast_key must select one of b1 or b2 for baseline predictions")
        point = np.asarray(data[selected], dtype=np.float64)
        if point.shape != y_true.shape:
            raise ValueError(f"{selected} must have shape {y_true.shape}")

    per_horizon: dict[str, dict[str, float]] = {}
    for index in range(y_true.shape[1]):
        per_horizon[str(index + 1)] = _metric_values(
            y_true[:, index],
            point[:, index],
            None if lower is None else lower[:, index],
            None if upper is None else upper[:, index],
            thresholds,
        )

    per_zone: dict[str, dict[str, float]] = {}
    for zone in sorted(set(int(value) for value in zone_ids)):
        mask = zone_ids.astype(int) == zone
        zone_values = _metric_values(
            y_true[mask],
            point[mask],
            None if lower is None else lower[mask],
            None if upper is None else upper[mask],
            thresholds[mask],
        )
        per_zone[str(zone)] = {"mae": zone_values["mae"], "wape": zone_values["wape"]}

    overall = _metric_values(y_true, point, lower, upper, thresholds)
    return {
        "tag": tag,
        "split": split,
        "per_horizon": per_horizon,
        "overall": overall,
        "per_zone": per_zone,
        "train_time_sec": float(train_time_sec),
        "n_params": int(n_params),
    }


def _base_tag(tag: str) -> str:
    """Remove a trailing seed suffix when grouping repeated runs."""
    return re.sub(r"_s\d+$", "", tag)


def aggregate_results(
    metrics: Sequence[Mapping[str, Any]], experiment: str
) -> dict[str, Any]:
    """Aggregate overall metric values by model, split, and experiment."""
    if not metrics:
        raise ValueError("At least one metric result is required")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in metrics:
        grouped.setdefault((_base_tag(str(result["tag"])), str(result["split"])), []).append(result)

    experiment_results: dict[str, Any] = {}
    for (model, split), results in grouped.items():
        metric_result: dict[str, dict[str, float]] = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [
                    float(result["overall"][metric])
                    for result in results
                    if result["overall"].get(metric) is not None
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            metric_result[metric] = {
                "mean": float(np.mean(finite)) if len(finite) else None,
                "std": float(np.std(finite)) if len(finite) else None,
            }
        experiment_results.setdefault(model, {})[split] = metric_result
    return {experiment: experiment_results}


def pick_app_model(metrics: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Select the lowest-validation-MAE run without inspecting test metrics."""
    validation = [
        result
        for result in metrics
        if str(result.get("split")) == "val"
        and np.isfinite(float(result["overall"]["mae"]))
    ]
    if not validation:
        raise ValueError("No validation metric results are available")
    selected = min(validation, key=lambda result: float(result["overall"]["mae"]))
    return {"tag": str(selected["tag"]), "selected_by": "val_mae"}

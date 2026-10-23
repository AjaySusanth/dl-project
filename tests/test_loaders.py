"""Tests for dashboard artifact loaders."""

from __future__ import annotations

from pathlib import Path

from app.loaders import (
    load_app_config,
    load_app_model,
    load_busy_time,
    load_hourly,
    load_metrics,
    load_predictions,
    load_results,
    load_zone_lookup,
)


def test_missing_loaders_return_actionable_messages(tmp_path: Path) -> None:
    """Missing dashboard artifacts produce text instead of exceptions."""
    missing = tmp_path / "missing.file"
    results = [
        load_app_config(missing),
        load_app_model(missing),
        load_predictions(missing),
        load_hourly(missing),
        load_busy_time(missing),
        load_results(missing),
        load_metrics(missing),
        load_zone_lookup(missing),
    ]

    assert all(isinstance(result, str) for result in results)
    assert all("Run `" in result for result in results)


def test_json_loader_returns_object(tmp_path: Path) -> None:
    """JSON artifacts are returned as dictionaries."""
    path = tmp_path / "results.json"
    path.write_text('{"E1": {}}', encoding="utf-8")

    result = load_results(path)

    assert result == {"E1": {}}

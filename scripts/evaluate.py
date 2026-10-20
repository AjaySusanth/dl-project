"""Command-line entry point for prediction evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.evaluate.baselines import lag_baseline
from src.evaluate.run_eval import aggregate_results, evaluate_run, pick_app_model

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate models and seasonal baselines.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--experiment", choices=("E1", "E2", "E3", "all"), default="all")
    return parser.parse_args()


def _include_tag(tag: str, experiment: str) -> bool:
    """Return whether a model tag belongs to the requested experiment."""
    if experiment == "all":
        return True
    if experiment == "E1":
        return "_L168_" in tag
    if experiment == "E2":
        return "_L24_" in tag or "_L168_" in tag
    return "notime" in tag


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open(encoding="utf-8") as input_file:
        loaded = json.load(input_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def _write_json(path: Path, value: Any) -> None:
    """Write indented JSON metadata."""
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, allow_nan=False)


def _evaluate_models(
    artifacts_dir: Path, zone_stats: dict[str, Any], experiment: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    """Evaluate discovered model prediction files and return one reference artifact."""
    model_results: list[dict[str, Any]] = []
    reference: dict[str, np.ndarray] | None = None
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        tag = run_dir.name
        if tag in {"baselines", "metrics"} or not _include_tag(tag, experiment):
            continue
        for split in ("val", "test"):
            prediction_path = run_dir / f"pred_{split}.npz"
            if not prediction_path.is_file():
                continue
            result = evaluate_run(prediction_path, zone_stats, tag, split)
            model_results.append(result)
            _write_json(metrics_dir / f"{tag}_{split}.json", result)
            if reference is None and split == "val":
                with np.load(prediction_path, allow_pickle=False) as loaded:
                    reference = {key: loaded[key] for key in loaded.files}

    if not model_results:
        raise FileNotFoundError("No model prediction files found for the requested experiment")
    return model_results, reference


def _evaluate_baselines(
    artifacts_dir: Path,
    processed_dir: Path,
    zone_stats: dict[str, Any],
    reference: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Create and evaluate lag baselines using model-origin metadata."""
    hourly = pd.read_parquet(processed_dir / "hourly_selected.parquet")
    baseline_dir = artifacts_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = artifacts_dir / "metrics"
    results: list[dict[str, Any]] = []

    for split in ("val", "test"):
        if split == "val":
            source = reference
        else:
            model_paths = sorted(
                path
                for path in artifacts_dir.iterdir()
                if path.is_dir() and path.name not in {"baselines", "metrics"}
            )
            test_path = next(
                (path / "pred_test.npz" for path in model_paths if (path / "pred_test.npz").is_file()),
                None,
            )
            if test_path is None:
                raise FileNotFoundError("No model test prediction exists for baseline construction")
            with np.load(test_path, allow_pickle=False) as loaded:
                source = {key: loaded[key] for key in loaded.files}

        b1, b2 = lag_baseline(hourly, source["origin_ts"], source["zone_id"], source["y_true"].shape[1])
        baseline_path = baseline_dir / f"pred_{split}.npz"
        np.savez(
            baseline_path,
            origin_ts=source["origin_ts"],
            zone_id=source["zone_id"],
            y_true=source["y_true"],
            b1=b1,
            b2=b2,
        )
        for key, label in (("b1", "baseline_lag24"), ("b2", "baseline_lag168")):
            result = evaluate_run(
                baseline_path,
                zone_stats,
                label,
                split,
                forecast_key=key,
            )
            results.append(result)
            _write_json(metrics_dir / f"{label}_{split}.json", result)
    return results


def main() -> int:
    """Run model evaluation, baselines, aggregation, and app-model selection."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config)
        processed_dir = Path(config["paths"]["processed_dir"])
        artifacts_dir = Path(config["paths"]["artifacts_dir"])
        zone_stats = _load_json(processed_dir / "zone_stats.json")
        model_results, reference = _evaluate_models(artifacts_dir, zone_stats, args.experiment)
        if reference is None:
            raise ValueError("A validation prediction is required to construct baselines")
        baseline_results = _evaluate_baselines(
            artifacts_dir, processed_dir, zone_stats, reference
        )
        all_results = model_results + baseline_results
        _write_json(artifacts_dir / "results.json", aggregate_results(all_results, args.experiment))
        model_only = [result for result in model_results if not result["tag"].startswith("baseline_")]
        _write_json(artifacts_dir / "app_model.json", pick_app_model(model_only))
        LOGGER.info("Evaluated %d prediction groups", len(all_results))
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        LOGGER.error("Evaluation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

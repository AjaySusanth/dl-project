"""Command-line entry point for feature and forecasting-window construction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.zones import select_zones
from src.features.scaling import compute_busy_time, fit_zone_stats
from src.features.splits import compute_splits
from src.features.windows import build_windows

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse feature-building command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build selected-zone features and forecasting windows."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Build only this lookback instead of experiments.lookbacks.",
    )
    return parser.parse_args()


def _lookbacks(config: dict[str, object], override: int | None) -> list[int]:
    """Resolve and validate configured or command-line lookbacks."""
    if override is not None:
        values = [override]
    else:
        experiments = config.get("experiments")
        if not isinstance(experiments, dict) or "lookbacks" not in experiments:
            raise KeyError("Missing required configuration key: experiments.lookbacks")
        values = [int(value) for value in experiments["lookbacks"]]
    if not values or any(value < 1 for value in values):
        raise ValueError("lookback values must be positive")
    return values


def main() -> int:
    """Run zone selection, statistics, and window artifact construction."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_config(args.config)
        processed_dir = Path(config["paths"]["processed_dir"])
        raw_dir = Path(config["paths"]["raw_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)

        hourly_path = processed_dir / "hourly_all.parquet"
        lookup_path = raw_dir / "taxi_zone_lookup.csv"
        hourly = pd.read_parquet(hourly_path)
        zone_lookup = pd.read_csv(lookup_path)
        selected = select_zones(hourly, config, zone_lookup)
        selected.to_parquet(processed_dir / "hourly_selected.parquet", index=False)

        splits = compute_splits(selected, config)
        with (processed_dir / "splits.json").open("w", encoding="utf-8") as output:
            json.dump(splits, output, indent=2)

        zone_stats = fit_zone_stats(selected, splits, config)
        with (processed_dir / "zone_stats.json").open("w", encoding="utf-8") as output:
            json.dump(zone_stats, output, indent=2)

        busy_time = compute_busy_time(selected, splits)
        busy_time.to_parquet(processed_dir / "busy_time.parquet", index=False)

        for lookback in _lookbacks(config, args.lookback):
            windows = build_windows(selected, zone_stats, splits, config, lookback=lookback)
            for split_name, arrays in windows.items():
                output_path = processed_dir / f"windows_L{lookback}_{split_name}.npz"
                npz_arrays = {key: value for key, value in arrays.items()}

                np.savez(output_path, **npz_arrays)
                LOGGER.info(
                    "Wrote %s windows: %s",
                    split_name,
                    output_path,
                )

        LOGGER.info("Selected %d zones and wrote feature artifacts to %s", selected["zone_id"].nunique(), processed_dir)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        LOGGER.error("Feature construction failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

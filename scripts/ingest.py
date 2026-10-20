"""Command-line entry point for raw-trip ingestion."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.data.ingest import aggregate_hourly, clean_trips, load_trips

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse ingestion command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest and aggregate TLC trips.")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run loading, cleaning, aggregation, and hourly output writing."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_config(args.config)
        trips = load_trips(config)
        cleaned, report = clean_trips(trips, config)
        hourly = aggregate_hourly(cleaned)

        output_path = Path(config["paths"]["processed_dir"]) / "hourly_all.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(output_path, index=False)

        LOGGER.info("Cleaning report: %s", json.dumps(report, sort_keys=True))
        LOGGER.info("Wrote %d hourly rows to %s", len(hourly), output_path)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        LOGGER.error("Ingestion failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

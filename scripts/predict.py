"""Command-line entry point for saved-model prediction generation."""

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
from src.evaluate.predict import generate_predictions

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse prediction-generation CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate CrowdCast prediction artifacts.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", choices=("lstm", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lookback", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Load configuration and generate validation and test predictions."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config)
        lookback = int(args.lookback or config["windows"]["lookback"])
        tag = f"{args.model}_L{lookback}_s{args.seed}"
        artifacts_dir = Path(config["paths"]["artifacts_dir"])
        processed_dir = Path(config["paths"]["processed_dir"])
        checkpoint = artifacts_dir / tag / "model.pt"
        zone_stats = processed_dir / "zone_stats.json"
        for split in ("val", "test"):
            summary = generate_predictions(
                checkpoint,
                processed_dir / f"windows_L{lookback}_{split}.npz",
                zone_stats,
                artifacts_dir / tag / f"pred_{split}.npz",
            )
            LOGGER.info("Generated predictions: %s", json.dumps(summary, sort_keys=True))
    except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Prediction generation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

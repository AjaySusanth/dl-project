"""Command-line entry point for one LSTM or GRU training run."""

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
from src.train.trainer import train_model

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train one CrowdCast recurrent model.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", choices=("lstm", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lookback", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Load configuration and run one training job."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = train_model(
            load_config(args.config),
            model=args.model,
            seed=args.seed,
            lookback=args.lookback,
        )
        LOGGER.info("Training complete: %s", json.dumps(summary, sort_keys=True))
    except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Training failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

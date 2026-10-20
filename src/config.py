"""Configuration loading and reproducible random seeding utilities."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Config = dict[str, Any]


def _read_yaml(path: Path) -> Config:
    """Read one YAML mapping from ``path``."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return loaded


def _merge_config(base: Config, overrides: Config) -> Config:
    """Recursively merge mapping overrides into a copied base mapping."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> Config:
    """Load a default config and merge an optional sibling override file.

    ``path`` may point to ``default.yaml`` or ``smoke.yaml``. When it points
    to ``smoke.yaml``, the sibling ``default.yaml`` is loaded first and the
    smoke values override it recursively.
    """
    config_path = Path(path)
    overrides = _read_yaml(config_path)

    if config_path.name == "default.yaml":
        return overrides

    default_path = config_path.with_name("default.yaml")
    defaults = _read_yaml(default_path)
    return _merge_config(defaults, overrides)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch is required by set_seed; install project dependencies first."
        ) from error

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""Generate raw-count quantile prediction artifacts from saved checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.rnn import QuantileRNN


def _device(config: dict[str, Any]) -> torch.device:
    """Resolve the checkpoint device without requiring CUDA."""
    requested = str(config.get("train", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("train.device must be auto, cpu, or cuda")
    return torch.device(requested)


def _load_checkpoint(path: Path) -> tuple[QuantileRNN, dict[str, Any], torch.device]:
    """Load a saved model and its metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "config", "zone_index", "n_features", "lookback"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing: {', '.join(missing)}")

    config = checkpoint["config"]
    zone_index = checkpoint["zone_index"]
    if not isinstance(config, dict) or not isinstance(zone_index, dict) or not zone_index:
        raise ValueError("Checkpoint metadata is invalid")
    quantiles = config.get("windows", {}).get("quantiles")
    if not isinstance(quantiles, list) or len(quantiles) < 3:
        raise ValueError("Checkpoint config must contain at least three quantiles")

    model = QuantileRNN(
        str(checkpoint.get("model", Path(path).parent.name).split("_")[0]),
        n_features=int(checkpoint["n_features"]),
        n_zones=len(zone_index),
        emb_dim=int(config["model"]["zone_emb_dim"]),
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        horizon=int(config["windows"]["horizon"]),
        n_quantiles=len(quantiles),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = _device(config)
    model.to(device)
    model.eval()
    return model, checkpoint, device


def _inverse_transform(
    predictions: np.ndarray,
    zone_ids: np.ndarray,
    zone_stats: dict[str, Any],
) -> np.ndarray:
    """Convert scaled log predictions to clipped raw pickup counts."""
    output = np.empty_like(predictions, dtype=np.float64)
    stats = zone_stats.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("zone_stats must contain a stats mapping")
    for row, zone_id in enumerate(zone_ids):
        zone = stats.get(str(int(zone_id)))
        if not isinstance(zone, dict):
            raise ValueError(f"Missing fitted statistics for zone {zone_id}")
        mean_log = float(zone["mean_log"])
        std_log = float(zone["std_log"])
        if not np.isfinite(mean_log) or not np.isfinite(std_log) or std_log <= 0:
            raise ValueError(f"Invalid fitted statistics for zone {zone_id}")
        output[row] = np.expm1(predictions[row] * std_log + mean_log)
    return np.maximum(output, 0.0).astype(np.float32)


def generate_predictions(
    checkpoint_path: str | Path,
    window_path: str | Path,
    zone_stats_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Generate one raw-count prediction archive from a saved checkpoint."""
    model, checkpoint, device = _load_checkpoint(Path(checkpoint_path))
    with np.load(window_path, allow_pickle=False) as data:
        required = {"X", "zone_id", "y_raw", "origin_ts"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Window file is missing: {', '.join(missing)}")
        x = torch.from_numpy(data["X"]).float()
        zone_idx = torch.from_numpy(data["zone_idx"]).long()
        zone_ids = np.asarray(data["zone_id"])
        y_true = np.asarray(data["y_raw"], dtype=np.float32)
        origin_ts = np.asarray(data["origin_ts"])

    with Path(zone_stats_path).open(encoding="utf-8") as input_file:
        zone_stats = json.load(input_file)
    if len(x) == 0:
        raise ValueError(f"Window file contains no samples: {window_path}")
    with torch.no_grad():
        scaled = model.predict_quantiles(x.to(device), zone_idx.to(device)).cpu().numpy()
    q_pred = _inverse_transform(scaled, zone_ids, zone_stats)
    if q_pred.shape[:2] != y_true.shape or not np.isfinite(q_pred).all():
        raise ValueError("Generated predictions have an invalid shape or non-finite values")
    q_pred.sort(axis=-1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, origin_ts=origin_ts, zone_id=zone_ids, y_true=y_true, q_pred=q_pred)
    return {
        "tag": str(output.parent.name),
        "split": output.stem.removeprefix("pred_"),
        "n_samples": int(len(x)),
        "shape": list(q_pred.shape),
        "lookback": int(checkpoint["lookback"]),
    }

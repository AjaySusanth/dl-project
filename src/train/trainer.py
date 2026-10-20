"""Reusable CPU-safe training loop for CrowdCast recurrent models."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.config import set_seed
from src.models.losses import pinball_loss
from src.models.rnn import QuantileRNN


def _load_split(path: Path) -> TensorDataset:
    """Load model inputs and scaled targets from one window archive."""
    if not path.is_file():
        raise FileNotFoundError(f"Window file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "zone_idx", "y_scaled"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Window file is missing: {', '.join(missing)}")
        x = torch.from_numpy(data["X"]).float()
        zone_idx = torch.from_numpy(data["zone_idx"]).long()
        target = torch.from_numpy(data["y_scaled"]).float()
    if x.ndim != 3 or zone_idx.ndim != 1 or target.ndim != 2:
        raise ValueError("Window arrays have invalid dimensions")
    if x.shape[0] != zone_idx.shape[0] or x.shape[0] != target.shape[0]:
        raise ValueError("Window arrays have inconsistent sample counts")
    if x.shape[0] == 0:
        raise ValueError(f"Window file contains no samples: {path}")
    return TensorDataset(x, zone_idx, target)


def _device(config: dict[str, Any]) -> torch.device:
    """Resolve configured device while preserving CPU portability."""
    requested = str(config["train"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("train.device must be auto, cpu, or cuda")
    return torch.device(requested)


def train_model(
    config: dict[str, Any],
    model: str,
    seed: int,
    lookback: int | None = None,
) -> dict[str, Any]:
    """Train one recurrent model and save its best validation checkpoint."""
    if model not in {"lstm", "gru"}:
        raise ValueError("model must be 'lstm' or 'gru'")
    set_seed(seed)

    configured_lookback = int(lookback or config["windows"]["lookback"])
    processed_dir = Path(config["paths"]["processed_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    zone_stats_path = processed_dir / "zone_stats.json"
    with zone_stats_path.open(encoding="utf-8") as input_file:
        zone_stats = json.load(input_file)
    zone_index = zone_stats.get("zone_index")
    if not isinstance(zone_index, dict) or not zone_index:
        raise ValueError("zone_stats.json must contain a non-empty zone_index")

    train_data = _load_split(processed_dir / f"windows_L{configured_lookback}_train.npz")
    val_data = _load_split(processed_dir / f"windows_L{configured_lookback}_val.npz")
    n_features = train_data.tensors[0].shape[2]
    horizon = train_data.tensors[2].shape[1]
    configured_quantiles = config["windows"].get("quantiles")
    if not isinstance(configured_quantiles, list) or not configured_quantiles:
        raise ValueError("windows.quantiles must be a non-empty list")
    quantiles = torch.tensor(configured_quantiles, dtype=torch.float32)

    model_instance = QuantileRNN(
        model,
        n_features=n_features,
        n_zones=len(zone_index),
        emb_dim=int(config["model"]["zone_emb_dim"]),
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        horizon=horizon,
        n_quantiles=len(quantiles),
    )
    device = _device(config)
    model_instance.to(device)
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=float(config["train"]["lr"]))
    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    max_epochs = int(config["train"]["max_epochs"])
    patience = int(config["train"]["patience"])
    if max_epochs < 1 or patience < 0:
        raise ValueError("max_epochs must be positive and patience cannot be negative")
    tag = f"{model}_L{configured_lookback}_s{seed}"
    output_dir = artifacts_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model_instance.train()
        train_losses: list[float] = []
        for x, zone_idx, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model_instance(x.to(device), zone_idx.to(device))
            loss = pinball_loss(prediction, target.to(device), quantiles.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model_instance.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x, zone_idx, target in val_loader:
                prediction = model_instance(x.to(device), zone_idx.to(device))
                loss = pinball_loss(prediction, target.to(device), quantiles.to(device))
                val_losses.append(float(loss.cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model_instance.state_dict(),
                    "config": config,
                    "zone_index": zone_index,
                    "n_features": n_features,
                    "lookback": configured_lookback,
                },
                output_dir / "model.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_epoch == 0:
        raise RuntimeError("Training produced no finite validation checkpoint")
    with (output_dir / "history.json").open("w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
    summary = {
        "tag": tag,
        "model": model,
        "seed": seed,
        "lookback": configured_lookback,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_time_sec": time.perf_counter() - started,
        "n_params": sum(parameter.numel() for parameter in model_instance.parameters()),
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    return summary

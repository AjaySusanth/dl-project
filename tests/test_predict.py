"""Tests for checkpoint prediction artifact generation."""

from __future__ import annotations

import json

import numpy as np
import torch

from src.evaluate.predict import generate_predictions
from src.models.rnn import QuantileRNN


def test_generate_predictions_inverse_transforms_clips_and_orders(tmp_path) -> None:
    """Inference writes finite raw counts with sorted quantiles."""
    config = {
        "model": {"zone_emb_dim": 2, "hidden_size": 3, "num_layers": 1, "dropout": 0.0},
        "windows": {"horizon": 2, "quantiles": [0.1, 0.5, 0.9]},
        "train": {"device": "cpu"},
    }
    model = QuantileRNN("gru", 2, 1, 2, 3, 1, 0.0, 2, 3)
    checkpoint_path = tmp_path / "gru_L2_s0" / "model.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "zone_index": {"7": 0},
            "n_features": 2,
            "lookback": 2,
        },
        checkpoint_path,
    )
    window_path = tmp_path / "windows.npz"
    np.savez(
        window_path,
        X=np.zeros((1, 2, 2), dtype=np.float32),
        zone_idx=np.array([0], dtype=np.int64),
        y_raw=np.array([[2.0, 3.0]], dtype=np.float32),
        origin_ts=np.array(["2026-01-01T00:00"], dtype="datetime64[ns]"),
        zone_id=np.array([7], dtype=np.int16),
    )
    stats_path = tmp_path / "zone_stats.json"
    stats_path.write_text(
        json.dumps({"stats": {"7": {"mean_log": 0.0, "std_log": 1.0}}}),
        encoding="utf-8",
    )

    output_path = tmp_path / "pred_test.npz"
    result = generate_predictions(checkpoint_path, window_path, stats_path, output_path)

    with np.load(output_path, allow_pickle=False) as output:
        q_pred = output["q_pred"]
    assert result["shape"] == [1, 2, 3]
    assert np.isfinite(q_pred).all()
    assert (q_pred[..., 1:] >= q_pred[..., :-1]).all()
    assert (q_pred >= 0).all()

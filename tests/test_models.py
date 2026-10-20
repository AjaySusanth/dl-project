"""Tests for the shared LSTM/GRU quantile model."""

from __future__ import annotations

import pytest
import torch

from src.models.rnn import QuantileRNN


def _model(cell: str) -> QuantileRNN:
    return QuantileRNN(
        cell,
        n_features=6,
        n_zones=4,
        emb_dim=3,
        hidden_size=8,
        num_layers=2,
        dropout=0.0,
        horizon=6,
        n_quantiles=3,
    )


def test_lstm_and_gru_forward_shapes_and_sorted_predictions() -> None:
    """Both recurrent cells implement the same public tensor contract."""
    x = torch.randn(4, 12, 6)
    zone_idx = torch.tensor([0, 1, 2, 3])

    for cell in ("lstm", "gru"):
        model = _model(cell)
        raw = model(x, zone_idx)
        ordered = model.predict_quantiles(x, zone_idx)
        assert raw.shape == (4, 6, 3)
        assert ordered.shape == (4, 6, 3)
        assert (ordered[..., 1:] >= ordered[..., :-1]).all()


def test_models_use_the_configured_recurrent_cell() -> None:
    """The only recurrent-cell difference is LSTM versus GRU."""
    lstm = _model("lstm")
    gru = _model("gru")
    assert isinstance(lstm.encoder, torch.nn.LSTM)
    assert isinstance(gru.encoder, torch.nn.GRU)
    assert sum(parameter.numel() for parameter in lstm.parameters()) != sum(
        parameter.numel() for parameter in gru.parameters()
    )


def test_model_rejects_invalid_cell_and_zone_indices() -> None:
    """Invalid architecture choices and embedding indices fail clearly."""
    with pytest.raises(ValueError, match="cell"):
        _model("rnn")
    model = _model("lstm")
    with pytest.raises(ValueError, match="out-of-range"):
        model(torch.randn(1, 4, 6), torch.tensor([4]))

"""Shared LSTM and GRU quantile forecasting model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QuantileRNN(nn.Module):
    """Encode hourly features and predict all horizons and quantiles."""

    def __init__(
        self,
        cell: str,
        n_features: int,
        n_zones: int,
        emb_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        n_quantiles: int,
    ) -> None:
        super().__init__()
        if cell not in {"lstm", "gru"}:
            raise ValueError("cell must be 'lstm' or 'gru'")
        if min(n_features, n_zones, emb_dim, hidden_size, num_layers, horizon, n_quantiles) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.cell = cell
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.zone_embedding = nn.Embedding(n_zones, emb_dim)
        self.encoder = recurrent(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size + emb_dim, horizon * n_quantiles)

    def forward(self, x: Tensor, zone_idx: Tensor) -> Tensor:
        """Return unsorted scaled-space quantiles with shape ``(B, H, Q)``."""
        if x.ndim != 3 or zone_idx.ndim != 1 or x.shape[0] != zone_idx.shape[0]:
            raise ValueError("x must be (B,L,F) and zone_idx must be (B,)")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain finite values")
        if zone_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("zone_idx must contain integer indices")
        if (zone_idx < 0).any() or (zone_idx >= self.zone_embedding.num_embeddings).any():
            raise ValueError("zone_idx contains an out-of-range embedding index")

        encoded, _ = self.encoder(x)
        last_hidden = encoded[:, -1, :]
        zone_vector = self.zone_embedding(zone_idx.long())
        output = self.head(torch.cat((last_hidden, zone_vector), dim=-1))
        return output.reshape(x.shape[0], self.horizon, self.n_quantiles)

    def predict_quantiles(self, x: Tensor, zone_idx: Tensor) -> Tensor:
        """Return predictions sorted along the quantile dimension."""
        return torch.sort(self.forward(x, zone_idx), dim=-1).values

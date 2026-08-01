"""LSTM/GRU encoder with a per-satellite embedding and a direct multi-step head.

The encoder reads the lookback window, and a feed-forward head maps the final
hidden state directly to the full (horizon_steps x n_channels) forecast. Direct
multi-output avoids autoregressive error accumulation over a 24h horizon.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RNNForecaster(nn.Module):
    def __init__(self, n_channels: int, horizon_steps: int, n_satellites: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1,
                 sat_embed_dim: int = 8, rnn_type: str = "lstm"):
        super().__init__()
        self.n_channels = n_channels
        self.horizon_steps = horizon_steps

        self.sat_embedding = nn.Embedding(n_satellites, sat_embed_dim)
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_channels + sat_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon_steps * n_channels),
        )

    def forward(self, x: torch.Tensor, sat_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C), sat_idx: (B,)
        sat_emb = self.sat_embedding(sat_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        rnn_in = torch.cat([x, sat_emb], dim=-1)
        out, _ = self.rnn(rnn_in)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.view(-1, self.horizon_steps, self.n_channels)

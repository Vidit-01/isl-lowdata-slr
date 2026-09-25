"""Shared sequence heads used by several arch.md baselines."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class StackedLSTM(nn.Module):
    """`nn.LSTM(num_layers=L, dropout=p)` rebuilt as L single-layer LSTMs with explicit
    dropout in between. Same function, but the multi-layer cuDNN RNN with inter-layer
    dropout crashed at interpreter exit on Windows. Parameter names differ from a
    multi-layer nn.LSTM; `load_legacy_lstm` maps old checkpoints."""

    def __init__(self, input_size: int, hidden: int, layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = True):
        super().__init__()
        out = hidden * (2 if bidirectional else 1)
        self.layers = nn.ModuleList(
            nn.LSTM(input_size if i == 0 else out, hidden, num_layers=1, batch_first=True,
                    bidirectional=bidirectional)
            for i in range(max(1, int(layers)))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        state = None
        for i, lstm in enumerate(self.layers):
            if i:
                x = self.drop(x)
            x, state = lstm(x)
        return x, state


def load_legacy_lstm(state_dict: dict) -> dict:
    """Map multi-layer nn.LSTM keys (`lstm.weight_ih_l1_reverse`) to StackedLSTM keys
    (`lstm.layers.1.weight_ih_l0_reverse`)."""
    import re

    out = {}
    for k, v in state_dict.items():
        m = re.match(r"^(.*lstm)\.(weight|bias)_(ih|hh)_l(\d+)(_reverse)?$", k)
        if m and ".layers." not in k:
            k = f"{m.group(1)}.layers.{m.group(4)}.{m.group(2)}_{m.group(3)}_l0{m.group(5) or ''}"
        out[k] = v
    return out


class BiLSTMClassifier(nn.Module):
    """BiLSTM over a (B, T, F) sequence with mean-pool + linear head."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = StackedLSTM(feat_dim, hidden, layers=layers, dropout=dropout,
                                bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h.mean(dim=1))


class TransformerClassifier(nn.Module):
    """Standard encoder stack over projected (B, T, F) tokens."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.cls = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos(self.input_proj(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.encode(x).mean(dim=1))

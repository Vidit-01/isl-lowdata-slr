"""Hierarchical Windowed Graph Attention (HWGAT-style, arch.md §6).

Spatio-temporal graph, body-part windows, Fourier feature mapping, stacked
part-attention. Isolated-sign head (mean-pool + linear).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import IN_CHANNELS, N_JOINTS, PARTS


def fourier_encode(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Tancik et al. Fourier features on the last dim. (..., C) -> (..., C*(1+2B))."""
    bands = (2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype)) * math.pi
    ang = x.unsqueeze(-1) * bands  # (..., C, B)
    return torch.cat([x, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], dim=-1)


class WindowedPartAttention(nn.Module):
    """MHSA inside a (temporal window × part joints) token group."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        return x + y


class HWGATLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window: int,
        dropout: float,
        parts: dict[str, tuple[int, ...]],
    ):
        super().__init__()
        self.window = window
        self.parts = parts
        self.win_attn = nn.ModuleDict(
            {name: WindowedPartAttention(d_model, nhead, dropout) for name in parts}
        )
        self.part_attn = WindowedPartAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def _windowed(self, tokens: torch.Tensor, module: WindowedPartAttention) -> torch.Tensor:
        # tokens: (B, T, V_part, D)
        b, t, v, d = tokens.shape
        w = min(self.window, t)
        pad = (w - t % w) % w
        if pad:
            tokens = F.pad(tokens, (0, 0, 0, 0, 0, pad))
        t2 = tokens.size(1)
        n_win = t2 // w
        grouped = tokens.view(b, n_win, w * v, d)
        grouped = grouped.reshape(b * n_win, w * v, d)
        grouped = module(grouped)
        grouped = grouped.view(b, n_win, w, v, d).reshape(b, t2, v, d)
        return grouped[:, :t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, D)
        out = x.clone()
        part_tokens = []
        for name, idx in self.parts.items():
            sl = x[:, :, list(idx)]
            sl = self._windowed(sl, self.win_attn[name])
            out[:, :, list(idx)] = sl
            part_tokens.append(sl.mean(dim=2))  # (B, T, D)
        parts = torch.stack(part_tokens, dim=2)  # (B, T, P, D)
        b, t, p, d = parts.shape
        parts = self.part_attn(parts.reshape(b * t, p, d)).view(b, t, p, d)
        # broadcast part residuals back onto joints of that part
        for i, idx in enumerate(self.parts.values()):
            out[:, :, list(idx)] = out[:, :, list(idx)] + parts[:, :, i].unsqueeze(2)
        return out + self.ff(out)


class HWGAT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = IN_CHANNELS,
        num_joints: int = N_JOINTS,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        fourier_bands: int = 8,
        window: int = 6,
        dropout: float = 0.2,
        max_len: int = 64,
        **_ignored,
    ):
        super().__init__()
        self.in_channels = in_channels
        in_dim = in_channels * (1 + 2 * fourier_bands)
        self.fourier_bands = fourier_bands
        self.in_proj = nn.Linear(in_dim, d_model)
        self.joint_pe = nn.Parameter(torch.zeros(1, 1, num_joints, d_model))
        self.time_pe = nn.Parameter(torch.zeros(1, max_len, 1, d_model))
        self.layers = nn.ModuleList(
            [
                HWGATLayer(d_model, nhead, window, dropout, PARTS)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.joint_pe, std=0.02)
        nn.init.trunc_normal_(self.time_pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x = fourier_encode(x, self.fourier_bands)
        h = self.in_proj(x)
        t = h.size(1)
        h = h + self.joint_pe + self.time_pe[:, :t]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h).mean(dim=(1, 2))
        return self.head(h)

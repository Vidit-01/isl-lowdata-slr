"""Part-wise Graph Fourier attention (PGF-SLR-style, arch.md §9).

Isolated-sign adaptation: part nodes (upper body, left hand, right hand) at each
frame, frequency-domain attention across parts, adaptive frequency enhancement,
and an auxiliary next-step motion branch used only during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import IN_CHANNELS, PARTS


class PartPool(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.parts = list(PARTS.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V) -> (B, T, P, D)
        x = x.permute(0, 2, 3, 1)
        pooled = []
        for idx in self.parts:
            pooled.append(x[:, :, list(idx)].mean(dim=2))
        return self.proj(torch.stack(pooled, dim=2))


class FrequencyPartAttention(nn.Module):
    """Attention over parts inside each FFT bin (complex as real|imag channels)."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.to_real = nn.Linear(d_model * 2, d_model)
        self.to_complex = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x complex: (B, F, P, D)
        ri = torch.cat([x.real, x.imag], dim=-1)
        h = self.to_real(ri)
        b, f, p, d = h.shape
        h = h.reshape(b * f, p, d)
        y, _ = self.attn(self.norm(h), self.norm(h), h, need_weights=False)
        real, imag = self.to_complex(y).chunk(2, dim=-1)
        return torch.complex(real, imag).view(b, f, p, d)


class AdaptiveFrequencyEnhance(nn.Module):
    def __init__(self, max_freq: int = 64):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(max_freq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, P, D) complex
        f = x.size(1)
        scale = torch.sigmoid(self.gate[:f]).view(1, f, 1, 1)
        return x * scale


class PGFSLR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = IN_CHANNELS,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        aux_weight: float = 0.2,
        **_ignored,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.pool = PartPool(in_channels, d_model)
        self.freq_layers = nn.ModuleList(
            [FrequencyPartAttention(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.enhance = AdaptiveFrequencyEnhance()
        self.time_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.aux = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        parts = self.pool(x)  # (B, T, P, D)
        spec = torch.fft.rfft(parts, dim=1)
        for layer in self.freq_layers:
            spec = layer(spec)
        spec = self.enhance(spec)
        rec = torch.fft.irfft(spec, n=parts.size(1), dim=1)
        fused = rec.mean(dim=2)  # (B, T, D)
        return self.time_enc(fused), rec

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        seq, rec = self._encode(x)
        logits = self.head(self.norm(seq).mean(dim=1))
        if not return_aux:
            return logits
        pred = self.aux(rec[:, :-1])
        aux = F.mse_loss(pred, rec[:, 1:].detach())
        return logits, aux

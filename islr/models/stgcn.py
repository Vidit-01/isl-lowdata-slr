"""ST-GCN on the 27-joint ISL skeleton (arch.md §4 / Yan et al. 2018)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import IN_CHANNELS, N_JOINTS, spatial_partition_adjacency


class SpatialGraphConv(nn.Module):
    """Partitioned spatial GCN: x' = sum_k W_k x A_k, A_k learned importance-masked."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.k = int(A.size(0))
        self.register_buffer("A_base", A, persistent=False)
        self.mask = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(in_channels, out_channels * self.k, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        a = self.A_base * torch.tanh(self.mask)
        y = self.conv(x).view(n, self.k, -1, t, v)  # (N, K, C_out, T, V)
        y = torch.einsum("nkctv,kvw->nctw", y, a)
        return y


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class STGCN(nn.Module):
    """Nine-ish ST-GCN blocks, global average pool, softmax classifier."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = IN_CHANNELS,
        num_joints: int = N_JOINTS,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        a = torch.from_numpy(spatial_partition_adjacency(num_joints))
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.blocks = nn.Sequential(
            STGCNBlock(in_channels, 64, a, residual=False, dropout=dropout),
            STGCNBlock(64, 64, a, dropout=dropout),
            STGCNBlock(64, 64, a, dropout=dropout),
            STGCNBlock(64, 128, a, stride=2, dropout=dropout),
            STGCNBlock(128, 128, a, dropout=dropout),
            STGCNBlock(128, 256, a, stride=2, dropout=dropout),
            STGCNBlock(256, 256, a, dropout=dropout),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        x = self.data_bn(x.reshape(n, c * v, t)).reshape(n, c, t, v)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(n, -1)
        return self.head(x)

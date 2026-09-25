"""CTR-GCN and TD-GCN (arch.md §5).

CTR-GC: shared topology + channel-wise refinement (Chen et al., ICCV 2021).
TD-GC: same idea, but the adjacency is also time-dependent (Liu et al., 2024).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import IN_CHANNELS, N_JOINTS, spatial_partition_adjacency


class CTRGC(nn.Module):
    """Channel-wise topology refinement graph convolution."""

    def __init__(self, in_channels: int, out_channels: int, rel_reduction: int = 8):
        super().__init__()
        rel = 8 if in_channels <= 9 else max(8, in_channels // rel_reduction)
        self.conv1 = nn.Conv2d(in_channels, rel, 1)
        self.conv2 = nn.Conv2d(in_channels, rel, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(rel, out_channels, 1)
        self.tanh = nn.Tanh()
        self.temporal_dependent = False

    def forward(self, x: torch.Tensor, a: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # x: (N, C, T, V)   a: (V, V)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        n, _, t, v = x3.shape
        if self.temporal_dependent:
            rel = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))  # (N, rel, T, V, V)
            rel = self.conv4(rel.permute(0, 2, 1, 3, 4).reshape(n * t, -1, v, v))
            rel = rel.view(n, t, -1, v, v).permute(0, 2, 1, 3, 4)
            a_ref = rel * alpha + a.view(1, 1, 1, v, v)
            return torch.einsum("nctv,nctvw->nctw", x3, a_ref)
        x1 = x1.mean(dim=2)
        x2 = x2.mean(dim=2)
        rel = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        rel = self.conv4(rel)
        a_ref = rel * alpha + a.view(1, 1, v, v)
        return torch.einsum("nctv,ncvw->nctw", x3, a_ref)


class TDGC(CTRGC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_dependent = True


class MultiScaleTCN(nn.Module):
    def __init__(self, channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        mid = channels // 2
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, mid, kernel_size=1),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid, mid, kernel_size=(k, 1), padding=(k // 2, 0), stride=(stride, 1)),
                    nn.BatchNorm2d(mid),
                )
                for k in (3, 5)
            ]
        )
        self.pool = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid),
        )
        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([b(x) for b in self.branches] + [self.pool(x)], dim=1)
        return self.fuse(y)


class GraphTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        gcn_cls: type[CTRGC],
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.k = int(A.size(0))
        self.gcn = nn.ModuleList([gcn_cls(in_channels, out_channels) for _ in range(self.k)])
        self.a = nn.Parameter(A.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.tcn = MultiScaleTCN(out_channels, stride=stride, dropout=dropout)
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
        y = 0
        for i, g in enumerate(self.gcn):
            y = y + g(x, self.a[i], alpha=self.alpha)
        y = self.bn(y)
        return self.relu(self.tcn(y) + self.residual(x))


class _GraphNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        gcn_cls: type[CTRGC],
        in_channels: int = IN_CHANNELS,
        num_joints: int = N_JOINTS,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        a = torch.from_numpy(spatial_partition_adjacency(num_joints))
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.blocks = nn.Sequential(
            GraphTemporalBlock(in_channels, 64, a, gcn_cls, residual=False, dropout=dropout),
            GraphTemporalBlock(64, 64, a, gcn_cls, dropout=dropout),
            GraphTemporalBlock(64, 128, a, gcn_cls, stride=2, dropout=dropout),
            GraphTemporalBlock(128, 128, a, gcn_cls, dropout=dropout),
            GraphTemporalBlock(128, 256, a, gcn_cls, stride=2, dropout=dropout),
            GraphTemporalBlock(256, 256, a, gcn_cls, dropout=dropout),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.data_bn(x.reshape(n, c * v, t)).reshape(n, c, t, v)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(n, -1)
        return self.head(x)


class CTRGCN(_GraphNet):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, CTRGC, **kwargs)


class TDGCN(_GraphNet):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, TDGC, **kwargs)

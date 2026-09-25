"""CNN + BiLSTM over raw RGB frames (arch.md §1).

Frame CNN → 1D temporal conv → BiLSTM → softmax. Isolated-sign head (no CTC).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameCNN(nn.Module):
    """ResNet-18 trunk (optional ImageNet init) producing a per-frame embedding."""

    def __init__(self, embed_dim: int = 512, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = resnet18(weights=weights)
        except Exception:
            net = resnet18(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Identity() if feat_dim == embed_dim else nn.Linear(feat_dim, embed_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 3, H, W)
        h = self.backbone(x)
        return self.proj(h)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        hidden: int = 512,
        lstm_layers: int = 2,
        conv_kernel: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cnn = FrameCNN(embed_dim=embed_dim, pretrained=pretrained, freeze_backbone=freeze_backbone)
        pad = conv_kernel // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_kernel, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )
        mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) in [0, 1]
        b, t, c, h, w = x.shape
        x = (x - self.mean) / self.std
        frames = x.reshape(b * t, c, h, w)
        emb = self.cnn(frames).reshape(b, t, self.embed_dim)
        sm = self.temporal(emb.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.lstm(sm)
        return self.head(seq.mean(dim=1))

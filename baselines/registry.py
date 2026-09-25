"""Registry of arch.md baseline models: modality, builder, and default hparams."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch.nn as nn

from .skeleton import IN_CHANNELS, N_JOINTS

POSE_HANDS_DIM = (33 + 21 + 21) * 3  # drop face mesh
JOINT_FLAT_DIM = N_JOINTS * IN_CHANNELS
FFT_DIM = JOINT_FLAT_DIM * 5  # pos, vel, acc, peak mag, peak phase
CWT_DIM = JOINT_FLAT_DIM * 7  # kin (3) + 4 wavelet bands

TRAIN_KEYS = {
    "epochs",
    "batch_size",
    "lr",
    "num_frames",
    "num_workers",
    "weight_decay",
    "size",
    "patience",
    "feat_dim",
    "in_channels",
    "use_bone",
}


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}


def _pgf_forward(model, batch, criterion, device, train: bool = True):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if train:
        logits, aux = model(x, return_aux=True)
        loss = criterion(logits, y) + float(getattr(model, "aux_weight", 0.2)) * aux
    else:
        logits = model(x, return_aux=False)
        loss = criterion(logits, y)
    return logits, y, loss


@dataclass
class BaselineSpec:
    name: str
    title: str
    modality: str  # rgb | landmarks | skeleton | spectral_fft | spectral_cwt
    build: Callable[[int, dict[str, Any]], nn.Module]
    defaults: dict[str, Any] = field(default_factory=dict)
    forward_fn: Optional[Callable] = None
    family: str = ""  # arch.md row grouping (e.g. ctr_gcn family)
    use_amp: bool = True


def _cnn_bilstm(num_classes: int, cfg: dict) -> nn.Module:
    from .cnn_bilstm import CNNBiLSTM

    return CNNBiLSTM(num_classes=num_classes, **_model_kwargs(cfg))


def _mp_bilstm(num_classes: int, cfg: dict) -> nn.Module:
    from .mp_bilstm import MPBiLSTM

    return MPBiLSTM(
        feat_dim=int(cfg["feat_dim"]),
        num_classes=num_classes,
        hidden=int(cfg.get("hidden", 256)),
        layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


def _mp_transformer(num_classes: int, cfg: dict) -> nn.Module:
    from .mp_transformer import MPTransformer

    return MPTransformer(
        feat_dim=int(cfg["feat_dim"]),
        num_classes=num_classes,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("layers", 3)),
        dim_feedforward=int(cfg.get("dim_feedforward", 256)),
        dropout=float(cfg.get("dropout", 0.2)),
        max_len=int(cfg.get("num_frames", 64)),
    )


def _stgcn(num_classes: int, cfg: dict) -> nn.Module:
    from .stgcn import STGCN

    return STGCN(num_classes=num_classes, in_channels=int(cfg.get("in_channels", IN_CHANNELS)))


def _ctr_gcn(num_classes: int, cfg: dict) -> nn.Module:
    from .ctr_gcn import CTRGCN

    return CTRGCN(num_classes=num_classes, in_channels=int(cfg.get("in_channels", IN_CHANNELS)))


def _td_gcn(num_classes: int, cfg: dict) -> nn.Module:
    from .ctr_gcn import TDGCN

    return TDGCN(num_classes=num_classes, in_channels=int(cfg.get("in_channels", IN_CHANNELS)))


def _hwgat(num_classes: int, cfg: dict) -> nn.Module:
    from .hwgat import HWGAT

    return HWGAT(
        num_classes=num_classes,
        in_channels=int(cfg.get("in_channels", IN_CHANNELS)),
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("layers", 3)),
        fourier_bands=int(cfg.get("fourier_bands", 8)),
        window=int(cfg.get("window", 6)),
        dropout=float(cfg.get("dropout", 0.2)),
        max_len=int(cfg.get("num_frames", 64)),
    )


def _fft_bilstm(num_classes: int, cfg: dict) -> nn.Module:
    from .fft_bilstm import FFTBiLSTM

    return FFTBiLSTM(
        feat_dim=int(cfg["feat_dim"]),
        num_classes=num_classes,
        hidden=int(cfg.get("hidden", 256)),
        layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


def _cwt_bilstm(num_classes: int, cfg: dict) -> nn.Module:
    from .cwt_seq import CWTBiLSTM

    return CWTBiLSTM(
        feat_dim=int(cfg["feat_dim"]),
        num_classes=num_classes,
        hidden=int(cfg.get("hidden", 256)),
        layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


def _cwt_transformer(num_classes: int, cfg: dict) -> nn.Module:
    from .cwt_seq import CWTTransformer

    return CWTTransformer(
        feat_dim=int(cfg["feat_dim"]),
        num_classes=num_classes,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("layers", 3)),
        dropout=float(cfg.get("dropout", 0.2)),
        max_len=int(cfg.get("num_frames", 64)),
    )


def _pgf_slr(num_classes: int, cfg: dict) -> nn.Module:
    from .pgf_slr import PGFSLR

    return PGFSLR(
        num_classes=num_classes,
        in_channels=int(cfg.get("in_channels", IN_CHANNELS)),
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("layers", 3)),
        dropout=float(cfg.get("dropout", 0.2)),
        aux_weight=float(cfg.get("aux_weight", 0.2)),
    )


_SEQ = dict(epochs=80, batch_size=16, lr=1e-3, num_frames=30, num_workers=0, weight_decay=1e-2, patience=20)
_GRAPH = dict(epochs=80, batch_size=16, lr=1e-3, num_frames=30, num_workers=0, weight_decay=1e-2, patience=20)
_RGB = dict(
    epochs=40,
    batch_size=4,
    lr=1e-4,
    num_frames=16,
    num_workers=0,
    weight_decay=1e-2,
    patience=12,
    size=112,
    pretrained=True,
    freeze_backbone=True,
    hidden=512,
    lstm_layers=2,
    conv_kernel=3,
)

SPECS: dict[str, BaselineSpec] = {
    "cnn_bilstm": BaselineSpec(
        "cnn_bilstm",
        "CNN + BiLSTM (raw RGB)",
        "rgb",
        _cnn_bilstm,
        defaults=dict(_RGB),
        family="tier1",
    ),
    "mp_bilstm": BaselineSpec(
        "mp_bilstm",
        "MediaPipe + BiLSTM",
        "landmarks",
        _mp_bilstm,
        defaults=dict(_SEQ, feat_dim=POSE_HANDS_DIM, hidden=256, layers=2, dropout=0.3),
        family="tier1",
    ),
    "mp_transformer": BaselineSpec(
        "mp_transformer",
        "MediaPipe + Transformer",
        "landmarks",
        _mp_transformer,
        defaults=dict(_SEQ, feat_dim=POSE_HANDS_DIM, d_model=128, nhead=4, layers=3, dropout=0.2),
        family="tier1",
    ),
    "stgcn": BaselineSpec(
        "stgcn",
        "ST-GCN",
        "skeleton",
        _stgcn,
        defaults=dict(_GRAPH, in_channels=IN_CHANNELS),
        family="tier2",
    ),
    "ctr_gcn": BaselineSpec(
        "ctr_gcn",
        "CTR-GCN",
        "skeleton",
        _ctr_gcn,
        defaults=dict(_GRAPH, in_channels=IN_CHANNELS),
        family="tier2-ctr",
    ),
    "td_gcn": BaselineSpec(
        "td_gcn",
        "TD-GCN",
        "skeleton",
        _td_gcn,
        defaults=dict(_GRAPH, in_channels=IN_CHANNELS, batch_size=8),
        family="tier2-ctr",
    ),
    "hwgat": BaselineSpec(
        "hwgat",
        "HWGAT",
        "skeleton",
        _hwgat,
        defaults=dict(_GRAPH, in_channels=IN_CHANNELS, d_model=128, nhead=4, layers=3, window=6, fourier_bands=8),
        family="tier2",
    ),
    "fft_bilstm": BaselineSpec(
        "fft_bilstm",
        "FFT + kinematic + BiLSTM",
        "spectral_fft",
        _fft_bilstm,
        defaults=dict(_SEQ, feat_dim=FFT_DIM, hidden=256, layers=2),
        family="tier3",
    ),
    "cwt_bilstm": BaselineSpec(
        "cwt_bilstm",
        "CWT + BiLSTM",
        "spectral_cwt",
        _cwt_bilstm,
        defaults=dict(_SEQ, feat_dim=CWT_DIM, hidden=256, layers=2, epochs=60),
        family="tier3",
    ),
    "cwt_transformer": BaselineSpec(
        "cwt_transformer",
        "CWT + Transformer",
        "spectral_cwt",
        _cwt_transformer,
        defaults=dict(_SEQ, feat_dim=CWT_DIM, d_model=128, nhead=4, layers=3, epochs=60),
        family="tier3",
    ),
    "pgf_slr": BaselineSpec(
        "pgf_slr",
        "Graph-Fourier attention (PGF-SLR-style)",
        "skeleton",
        _pgf_slr,
        defaults=dict(_GRAPH, in_channels=IN_CHANNELS, d_model=128, nhead=4, layers=3, aux_weight=0.2),
        family="tier3",
        forward_fn=_pgf_forward,
        use_amp=False,
    ),
}

ALL_NAMES: tuple[str, ...] = tuple(SPECS.keys())


def get_spec(name: str) -> BaselineSpec:
    if name not in SPECS:
        known = ", ".join(ALL_NAMES)
        raise KeyError(f"Unknown baseline '{name}'. Choose from: {known}")
    return SPECS[name]

"""Low-data landmark models: part-factorised encoders over pose | hands | lips.

Input: (B, T, V, 3) normalised Holistic points in the reduced layout of
`lowdata.store` (V = 119: pose 33 | left hand 21 | right hand 21 | lips 40 + nose 4),
or the full 543-point layout of the legacy cache (reduced here). Missing points are 0.

Design choices for few-shot data:
  * Per-part encoders; each part is projected first and normalised AFTER the
    projection. A LayerNorm over raw hand coordinates would remove where the hand is
    relative to the body, which is part of the sign.
  * Hands carry two views: position in the body frame (location) and wrist-relative,
    size-normalised shape (handshape).
  * One shared (symmetric) hand encoder for both hands; the left hand is mirrored
    (x -> -x) and a side embedding is added. Each hand sees twice the training data,
    and left-/right-dominant signers share weights.
  * Missing parts (no hand detected) are replaced by a learned token, and during
    training whole parts are dropped at random (part-token dropout), so the model
    cannot lean on one part being present.
  * Optional cosine classifier (normalised features and class weights), which is
    known to transfer better from few examples.

Models:
  PartFormer      part encoders -> frame fusion -> Transformer -> attention pooling
  Conv1DFormer    same front end, then depthwise temporal conv blocks interleaved with
                  Transformer blocks (the ASL-Signs 1st-place pattern), BatchNorm
                  (SyncBatchNorm under DDP)
  KDFPartFormer   PartFormer + the Hankel-DMD spectrum/mode branch and class-wise
                  Koopman head of kdf_stgcn (novel model on the new backbone)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from islr.lowdata.store import FACE_KEEP, N_REDUCED, REDUCED_ROWS

from .heads import PositionalEncoding
from .kdf import EIG_DIM, MODE_MAP_DIM, N_PARTS, PART_ORDER, ClassKoopmanHead, kdf_forward
from .skeleton import PARTS as SKEL_PARTS

N_POSE, N_HAND = 33, 21
LH0, RH0, FACE0 = 33, 54, 75
BODY_KEEP = (0, 11, 12, 13, 14, 15, 16, 23, 24)  # nose, shoulders, elbows, wrists, hips
N_LIPS = 40
NOSE_LOCAL = N_LIPS  # first nose point inside the face subset
PARTS_DIM = N_REDUCED * 3


def _present(pts: torch.Tensor) -> torch.Tensor:
    """(..., V, 3) -> (..., V) bool: a point is missing when all coordinates are 0."""
    return pts.abs().sum(-1) > 0


def _masked_vel(pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Temporal difference, zero where either frame is missing. pts (B,T,V,3)."""
    vel = torch.zeros_like(pts)
    both = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
    vel[:, 1:] = (pts[:, 1:] - pts[:, :-1]) * both
    return vel


class PartFrontend(nn.Module):
    """(B, T, V, 3) points -> (B, T, d) frame tokens + (B, T) frame-valid mask."""

    def __init__(self, d_model: int = 128, part_dim: int = 96, dropout: float = 0.2,
                 part_drop: float = 0.1, use_face: bool = True):
        super().__init__()
        self.part_drop = float(part_drop)
        self.use_face = bool(use_face)

        def enc(n_in):
            # projection first, LayerNorm after it (never on raw coordinates)
            return nn.Sequential(nn.Linear(n_in, part_dim), nn.LayerNorm(part_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(part_dim, part_dim))

        nb = len(BODY_KEEP)
        self.body = enc(nb * 3 * 2)
        self.hand = enc(N_HAND * 3 * 3 + 1)  # location, shape, velocity, presence
        self.face = enc(len(FACE_KEEP) * 3 * 2) if self.use_face else None
        n_parts = 4 if self.use_face else 3
        self.side = nn.Parameter(torch.zeros(2, part_dim))
        self.missing = nn.Parameter(torch.zeros(n_parts, part_dim))
        self.fuse = nn.Sequential(nn.LayerNorm(n_parts * part_dim), nn.Linear(n_parts * part_dim, d_model),
                                  nn.GELU(), nn.Dropout(dropout))
        self.register_buffer("_rows", torch.as_tensor(REDUCED_ROWS), persistent=False)
        self.register_buffer("_body", torch.as_tensor(BODY_KEEP), persistent=False)
        mirror = torch.ones(3)
        mirror[0] = -1.0
        self.register_buffer("_mirror", mirror, persistent=False)

    def _hand_feats(self, h: torch.Tensor, mirror: bool) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = h.shape[:2]
        if mirror:
            h = h * self._mirror
        m = _present(h)  # (B,T,21)
        pres = m.any(-1)  # (B,T)
        wrist = h[:, :, :1]
        local = (h - wrist) * m.unsqueeze(-1)
        size = local[..., :2].norm(dim=-1).amax(-1, keepdim=True).clamp_min(1e-3)  # (B,T,1)
        shape = local / size.unsqueeze(-1)
        vel = _masked_vel(h, m)
        f = torch.cat([h.reshape(b, t, -1), shape.reshape(b, t, -1), vel.reshape(b, t, -1),
                       pres.unsqueeze(-1).to(h.dtype)], dim=-1)
        return f, pres

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:  # (B, T, V*3)
            x = x.reshape(x.size(0), x.size(1), -1, 3)
        if x.size(2) == 543:
            x = x.index_select(2, self._rows)
        b, t = x.shape[:2]
        body = x[:, :, :N_POSE].index_select(2, self._body)
        bm = _present(body)
        fb = torch.cat([body.reshape(b, t, -1), _masked_vel(body, bm).reshape(b, t, -1)], -1)
        tokens = [self.body(fb)]
        present = [bm.any(-1)]
        for side, (lo, mirror) in enumerate(((LH0, True), (RH0, False))):
            f, pres = self._hand_feats(x[:, :, lo:lo + N_HAND], mirror)
            tokens.append(self.hand(f) + self.side[side])
            present.append(pres)
        if self.use_face:
            face = x[:, :, FACE0:]
            fm = _present(face)
            nose = face[:, :, NOSE_LOCAL:NOSE_LOCAL + 1]
            local = (face - nose) * fm.unsqueeze(-1)
            ff = torch.cat([local.reshape(b, t, -1), _masked_vel(face, fm).reshape(b, t, -1)], -1)
            tokens.append(self.face(ff))
            present.append(fm.any(-1))
        tok = torch.stack(tokens, 2)  # (B,T,P,D)
        pres = torch.stack(present, 2)  # (B,T,P)
        if self.training and self.part_drop > 0:
            keep = torch.rand(b, 1, pres.size(2), device=x.device) >= self.part_drop
            pres = pres & keep
        tok = torch.where(pres.unsqueeze(-1), tok, self.missing.to(tok.dtype).expand_as(tok))
        frames = self.fuse(tok.reshape(b, t, -1))
        valid = pres[:, :, 1:3].any(-1) | pres[:, :, 0]  # a hand or the body is visible
        return frames, valid


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        s = self.score(h).squeeze(-1).float()
        if valid is not None:
            s = s.masked_fill(~valid, -1e4)
        w = torch.softmax(s, dim=1).to(h.dtype)
        return (w.unsqueeze(-1) * h).sum(1)


class CosineHead(nn.Module):
    """Cosine-similarity classifier with a learned temperature (initial scale 16)."""

    def __init__(self, d: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    @property
    def out_features(self) -> int:
        return self.weight.size(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp() * F.linear(F.normalize(z.float(), dim=-1), F.normalize(self.weight.float(), dim=-1))


def _head(d: int, num_classes: int, cosine: bool, dropout: float) -> nn.Module:
    if cosine:
        return nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), CosineHead(d, num_classes))
    return nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes))


def _encoder(d_model, nhead, layers, dropout):
    layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=True,
                                       activation="gelu", norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=int(layers), enable_nested_tensor=False)


class PartFormer(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 128, part_dim: int = 96, nhead: int = 4,
                 layers: int = 2, dropout: float = 0.2, part_drop: float = 0.1, cosine: bool = False,
                 use_face: bool = True, max_len: int = 256, mixup: float = 0.2,
                 label_smoothing: float = 0.1, **_ignored):
        super().__init__()
        self.mixup = float(mixup)
        self.label_smoothing = float(label_smoothing)
        self.front = PartFrontend(d_model, part_dim, dropout, part_drop, use_face)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.encoder = _encoder(d_model, nhead, layers, dropout)
        self.pool = AttentionPool(d_model)
        self.head = _head(d_model, num_classes, cosine, dropout)

    def encode(self, x: torch.Tensor):
        frames, valid = self.front(x)
        tokens = self.encoder(self.pos(frames))
        return tokens, valid

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens, valid = self.encode(x)
        return self.pool(tokens, valid)

    def forward(self, x: torch.Tensor, *_unused, return_aux: bool = False):
        logits = self.head(self.embed(x))
        return (logits, None) if return_aux else logits


class ConvBlock(nn.Module):
    """Depthwise temporal conv block: expand -> depthwise conv -> BN -> project, residual."""

    def __init__(self, d: int, kernel: int = 7, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        h = d * expand
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, 2 * h)
        self.dw = nn.Conv1d(h, h, kernel, padding=kernel // 2, groups=h)
        self.bn = nn.BatchNorm1d(h)
        self.out = nn.Linear(h, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        a, g = self.inp(self.norm(x)).chunk(2, dim=-1)
        h = (a * torch.sigmoid(g)) * valid.unsqueeze(-1).to(x.dtype)  # GLU, padded frames zeroed
        h = self.bn(self.dw(h.transpose(1, 2))).transpose(1, 2)
        return x + self.drop(self.out(F.silu(h)))


class Conv1DFormer(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 128, part_dim: int = 96, nhead: int = 4,
                 blocks: int = 2, kernel: int = 7, dropout: float = 0.2, part_drop: float = 0.1,
                 cosine: bool = False, use_face: bool = True, max_len: int = 256, mixup: float = 0.2,
                 label_smoothing: float = 0.1, **_ignored):
        super().__init__()
        self.mixup = float(mixup)
        self.label_smoothing = float(label_smoothing)
        self.front = PartFrontend(d_model, part_dim, dropout, part_drop, use_face)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.convs = nn.ModuleList(ConvBlock(d_model, kernel, 2, dropout) for _ in range(int(blocks)))
        self.attns = nn.ModuleList(_encoder(d_model, nhead, 1, dropout) for _ in range(int(blocks)))
        self.pool = AttentionPool(d_model)
        self.head = _head(d_model, num_classes, cosine, dropout)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h, valid = self.front(x)
        h = self.pos(h)
        for conv, attn in zip(self.convs, self.attns):
            h = attn(conv(h, valid))
        return self.pool(h, valid)

    def forward(self, x: torch.Tensor, *_unused, return_aux: bool = False):
        logits = self.head(self.embed(x))
        return (logits, None) if return_aux else logits


class KDFPartFormer(nn.Module):
    """PartFormer tokens + part-wise Hankel-DMD spectrum/modes + class-wise Koopman head."""

    def __init__(self, num_classes: int, d_model: int = 128, part_dim: int = 96, nhead: int = 4,
                 layers: int = 2, dropout: float = 0.2, part_drop: float = 0.1, eig_hidden: int = 64,
                 cosine: bool = False, use_face: bool = True, max_len: int = 256, mixup: float = 0.2,
                 label_smoothing: float = 0.1, koopman_weight: float = 0.3, koopman_rank: int = 8,
                 use_dmd: bool = True, use_kc: bool = True, **_ignored):
        super().__init__()
        self.mixup = float(mixup)
        self.label_smoothing = float(label_smoothing)
        self.koopman_weight = float(koopman_weight)
        self.use_dmd, self.use_kc = bool(use_dmd), bool(use_kc)
        self.backbone = PartFormer(num_classes, d_model, part_dim, nhead, layers, dropout, part_drop,
                                   cosine=False, use_face=use_face, max_len=max_len)
        self.backbone.head = nn.Identity()  # the backbone's own head is not used
        dyn = 0
        if self.use_dmd:
            mode_in = N_PARTS * MODE_MAP_DIM
            self.eig_mlp = nn.Sequential(nn.LayerNorm(EIG_DIM), nn.Linear(EIG_DIM, eig_hidden), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(eig_hidden, 64))
            self.mode_mlp = nn.Sequential(nn.LayerNorm(mode_in), nn.Linear(mode_in, eig_hidden), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(eig_hidden, 64))
            for name in PART_ORDER:
                self.register_buffer(f"_idx_{name}", torch.tensor(SKEL_PARTS[name]), persistent=False)
            dyn = 128
        self.head = _head(d_model + dyn, num_classes, cosine, dropout)
        if self.use_kc:
            self.koopman = ClassKoopmanHead(d_model, num_classes, rank=int(koopman_rank))
            self.dyn_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, eig=None, modes=None, return_aux: bool = False):
        tokens, valid = self.backbone.encode(x)
        feats = [self.backbone.pool(tokens, valid)]
        if self.use_dmd:
            b = tokens.size(0)
            eig = eig if eig is not None else tokens.new_zeros(b, EIG_DIM)
            if modes is None:
                modes = tokens.new_zeros(b, 27, MODE_MAP_DIM)
            part_modes = torch.cat([modes.index_select(1, getattr(self, f"_idx_{n}")).mean(1)
                                    for n in PART_ORDER], -1)
            feats += [self.eig_mlp(eig), self.mode_mlp(part_modes)]
        logits = self.head(torch.cat(feats, -1))
        k = None
        if self.use_kc:
            k = self.koopman.scores(tokens.float())
            logits = logits + self.dyn_scale * k
        return (logits, k) if return_aux else logits


def reg_forward(model, batch, criterion, device, train: bool = True):
    """Mixup + label smoothing for single-input models (the regularisation KDF uses),
    so part models and the KDF ablations are compared under the same recipe."""
    x, y = batch
    if isinstance(x, (tuple, list)):
        return kdf_forward(model, batch, criterion, device, train)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    mix = float(getattr(model, "mixup", 0.0) or 0.0) if train and model.training else 0.0
    smooth = float(getattr(model, "label_smoothing", 0.0) or 0.0) if train else 0.0
    weight = getattr(criterion, "weight", None)
    if mix > 0 and x.size(0) > 1:
        lam = float(np.random.beta(mix, mix))
        idx = torch.randperm(x.size(0), device=x.device)
        logits = model(lam * x + (1 - lam) * x[idx])
        loss = lam * F.cross_entropy(logits, y, weight=weight, label_smoothing=smooth) + \
            (1 - lam) * F.cross_entropy(logits, y[idx], weight=weight, label_smoothing=smooth)
        return logits, y, loss
    logits = model(x)
    return logits, y, F.cross_entropy(logits, y, weight=weight, label_smoothing=smooth)

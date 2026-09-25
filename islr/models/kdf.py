"""Hankel-DMD / Koopman features on the MediaPipe Transformer backbone.

Koopman extraction (Kalman, part-wise Hankel-DMD, eigenvalues, spatial modes)
is unchanged. The classifier is no longer a GCN: few-shot SLR here is won by
mp_transformer, and the Koopman literature treats the operator as a plug-in
on a strong sequence backbone.

  Wang et al., CVPR 2023, Neural Koopman Pooling — class-wise Koopman matrices
  as dynamical templates; DMD matching for one-shot skeleton recognition.
  Zhang et al., 2021, Action recognition based on DMD — concat DMD spectrum
  with a learned encoder; helps quasi-few-shot when CNNs cannot be trained well.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import TransformerClassifier
from .skeleton import N_JOINTS, PARTS

CACHE_TAG = "kdfv2"
N_MODES = 3  # per body part
N_DELAYS = 8
PART_ORDER = ("upper", "left_hand", "right_hand")
N_PARTS = len(PART_ORDER)
KIN_CHANNELS = 3
KDF_IN_CHANNELS = KIN_CHANNELS
FEAT_PER_MODE = 4  # log1p|λ|, sin θ, cos θ, RMS amplitude
EIG_DIM = N_PARTS * N_MODES * FEAT_PER_MODE + N_PARTS
MODE_MAP_DIM = N_PARTS * N_MODES  # 9 spatial-mode channels over joints


def _kalman_nd(z: np.ndarray, q: float = 8e-4, r: float = 1.5e-2) -> np.ndarray:
    """Independent constant-velocity Kalman on each column. z: (T, N) → (T, N)."""
    t_len, n = z.shape
    z64 = np.asarray(z, dtype=np.float64).reshape(t_len, n)
    pos = z64[0].copy()
    vel = np.zeros(n, dtype=np.float64)
    p11 = np.ones(n, dtype=np.float64)
    p12 = np.zeros(n, dtype=np.float64)
    p22 = np.ones(n, dtype=np.float64)
    q11, q12, q22 = 0.25 * q, 0.5 * q, q
    out = np.empty_like(z64)
    r = float(r)
    for t in range(t_len):
        pos = pos + vel
        p11 = p11 + 2.0 * p12 + p22 + q11
        p12 = p12 + p22 + q12
        p22 = p22 + q22
        innov = z64[t] - pos
        s = np.maximum(p11 + r, 1e-12)
        k0 = p11 / s
        k1 = p12 / s
        pos = pos + k0 * innov
        vel = vel + k1 * innov
        n11 = (1.0 - k0) * p11
        n12 = (1.0 - k0) * p12
        n21 = -k1 * p11 + p12
        n22 = -k1 * p12 + p22
        p11 = n11
        p12 = 0.5 * (n12 + n21)
        p22 = n22
        out[t] = pos
    return out.astype(np.float32)


def _interp_missing_matrix(joints: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill occluded joints along time. joints: (T, V, C), missing: (T, V)."""
    out = joints.astype(np.float32, copy=True)
    t_idx = np.arange(joints.shape[0], dtype=np.float64)
    for v in range(joints.shape[1]):
        miss = missing[:, v]
        if not bool(miss.any()) or bool(miss.all()):
            continue
        good = ~miss
        for c in range(joints.shape[2]):
            out[:, v, c] = np.interp(t_idx, t_idx[good], joints[good, v, c].astype(np.float64))
    return out


def kalman_smooth_joints(joints: np.ndarray, q: float = 8e-4, r: float = 1.5e-2) -> np.ndarray:
    """Occlusion fill + forward–backward Kalman on (T, V, C) trajectories."""
    t_len, n_j, n_c = joints.shape
    missing = np.linalg.norm(joints, axis=-1) < 1e-6
    filled = _interp_missing_matrix(joints, missing).reshape(t_len, n_j * n_c)
    fwd = _kalman_nd(filled, q=q, r=r)
    bwd = _kalman_nd(fwd[::-1], q=q, r=r)[::-1]
    return (0.5 * (fwd + bwd)).reshape(t_len, n_j, n_c).astype(np.float32)


def normalize_joints(joints: np.ndarray) -> np.ndarray:
    """Root-center at the nose and scale by mean shoulder width."""
    x = joints.astype(np.float32, copy=True)
    x = x - x[:, :1, :]
    shoulder = np.linalg.norm(x[:, 1] - x[:, 2], axis=-1)
    scale = float(np.nanmean(shoulder))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = float(np.std(x) * 2.0 + 1e-3)
    return (x / scale).astype(np.float32)


def _hankel(x: np.ndarray, delays: int) -> np.ndarray:
    """x: (D, T) → Hankel (D*delays, T-delays+1)."""
    dim, t_len = x.shape
    cols = t_len - delays + 1
    h = np.empty((dim * delays, cols), dtype=np.float64)
    for i in range(delays):
        h[i * dim : (i + 1) * dim] = x[:, i : i + cols]
    return h


def hankel_dmd(x: np.ndarray, delays: int = N_DELAYS, rank: int = N_MODES):
    """Exact DMD on a delay-embedded Hankel matrix.

    x: (D, T)  →  |spatial| (D, r), |amp| (r, T), eigs (r,)
    Modes are ordered oscillatory-first so clips align better than raw |λ|.
    """
    x64 = np.asarray(x, dtype=np.float64)
    dim, t_len = x64.shape
    delays = int(max(2, min(int(delays), max(2, t_len // 3))))
    empty = (
        np.zeros((dim, rank), dtype=np.float32),
        np.zeros((rank, t_len), dtype=np.float32),
        np.zeros(rank, dtype=np.complex64),
    )
    if t_len < delays + 2 or dim < 1:
        return empty
    h0 = _hankel(x64[:, :-1], delays)
    h1 = _hankel(x64[:, 1:], delays)
    cols = min(h0.shape[1], h1.shape[1])
    if cols < 2:
        return empty
    h0, h1 = h0[:, :cols], h1[:, :cols]
    u, s, vh = np.linalg.svd(h0, full_matrices=False)
    r = int(min(rank, u.shape[1], max(1, int((s > 1e-6).sum()))))
    u, s, vh = u[:, :r], s[:r], vh[:r]
    k = u.T @ h1 @ vh.T @ np.diag(1.0 / np.clip(s, 1e-8, None))
    eigvals, eigvecs = np.linalg.eig(k)
    order = np.lexsort((-np.abs(eigvals), -np.abs(np.angle(eigvals))))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    phi = u @ eigvecs
    spatial_c = phi[:dim, :r]
    mag = np.clip(np.abs(eigvals), 0.5, 1.5)
    lam = mag * np.exp(1j * np.angle(eigvals))
    try:
        b0 = np.linalg.pinv(spatial_c, rcond=1e-6) @ x64[:, 0]
    except np.linalg.LinAlgError:
        b0 = np.zeros(r, dtype=np.complex128)
    t_idx = np.arange(t_len, dtype=np.float64)
    amp = np.zeros((r, t_len), dtype=np.float64)
    for i in range(r):
        amp[i] = np.abs(b0[i] * np.power(lam[i], t_idx))
    spatial = np.abs(spatial_c).astype(np.float32)
    if spatial.shape[1] < rank:
        pad = rank - spatial.shape[1]
        spatial = np.pad(spatial, ((0, 0), (0, pad)))
        amp = np.pad(amp, ((0, pad), (0, 0)))
        eigvals = np.concatenate([eigvals, np.zeros(pad, dtype=np.complex128)])
    return (
        spatial[:, :rank],
        amp[:rank].astype(np.float32),
        np.asarray(eigvals[:rank], dtype=np.complex64),
    )


def _part_spectrum(eigs: np.ndarray, amp: np.ndarray, n_modes: int) -> np.ndarray:
    """log|λ|, sin θ, cos θ, RMS amp for one body part. Shape (n_modes * 4,)."""
    out = np.zeros(n_modes * FEAT_PER_MODE, dtype=np.float32)
    z = np.asarray(eigs).ravel()
    for i in range(n_modes):
        base = i * FEAT_PER_MODE
        out[base + 2] = 1.0
        if i >= z.size:
            continue
        val = complex(z[i])
        ang = np.angle(val)
        out[base] = np.log1p(abs(val))
        out[base + 1] = np.sin(ang)
        out[base + 2] = np.cos(ang)
        if amp is not None and i < amp.shape[0]:
            out[base + 3] = float(np.sqrt(np.mean(np.square(amp[i]))))
    return out


def kdf_joint_features(
    joints: np.ndarray,
    n_modes: int = N_MODES,
    delays: int = N_DELAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman + signer-normalized pose, part-wise Hankel-DMD.

    joints: (T, V, 3)
    returns:
      x:     (3, T, V)  Kalman-smoothed, root-centered pose
      eig:   (EIG_DIM,) part-wise Koopman spectrum + part kinetic energy
      modes: (V, MODE_MAP_DIM) spatial graph-mode energy per joint
    """
    joints = np.asarray(joints, dtype=np.float32)
    t_len, n_j, n_c = joints.shape
    smooth = normalize_joints(kalman_smooth_joints(joints))
    vel = np.diff(smooth, axis=0, prepend=smooth[:1])
    mode_map = np.zeros((n_j, N_PARTS * n_modes), dtype=np.float32)
    spec_parts: list[np.ndarray] = []
    kinetic: list[float] = []
    for p, name in enumerate(PART_ORDER):
        idx = list(PARTS[name])
        sub = smooth[:, idx, :]
        spatial, amp, eigs = hankel_dmd(sub.reshape(t_len, -1).T, delays=delays, rank=n_modes)
        vp = len(idx)
        spat = spatial.reshape(vp, n_c, -1).mean(axis=1)  # (Vp, r)
        for i in range(n_modes):
            col = spat[:, i]
            peak = float(col.max()) + 1e-6
            mode_map[idx, p * n_modes + i] = col / peak
        spec_parts.append(_part_spectrum(eigs, amp, n_modes))
        kinetic.append(float(np.sqrt(np.mean(np.square(vel[:, idx])))))
    eig = np.concatenate([*spec_parts, np.asarray(kinetic, dtype=np.float32)], axis=0)
    x = np.transpose(smooth, (2, 0, 1)).astype(np.float32)
    return x, eig.astype(np.float32), mode_map.astype(np.float32)


class ClassKoopmanHead(nn.Module):
    """Low-rank class-wise Koopman templates (Wang et al., CVPR 2023).

    Each class is a linear map K_c ≈ U_c V_c^T on transformer tokens.
    Score is negative reconstruction error of h_{t+1} ≈ K_c h_t.
    """

    def __init__(self, d_model: int, num_classes: int, rank: int = 8):
        super().__init__()
        self.U = nn.Parameter(torch.randn(num_classes, d_model, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_classes, d_model, rank) * 0.02)

    def scores(self, tokens: torch.Tensor) -> torch.Tensor:
        h0, h1 = tokens[:, :-1], tokens[:, 1:]
        u = F.normalize(self.U, dim=1)
        v = F.normalize(self.V, dim=1)
        z = torch.einsum("ntd,cdr->nctr", h0, v)
        pred = torch.einsum("nctr,cdr->nctd", z, u)
        err = (pred - h1.unsqueeze(1)).pow(2).mean(dim=(2, 3))
        return -err


class KDFSTGCN(nn.Module):
    """mp_transformer backbone + Hankel-DMD concat + class-wise Koopman matching."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 225,
        in_channels: int = KDF_IN_CHANNELS,
        n_modes: int = N_MODES,
        eig_hidden: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        dropout: float = 0.2,
        mixup: float = 0.2,
        label_smoothing: float = 0.1,
        koopman_weight: float = 0.3,
        koopman_rank: int = 8,
        use_dmd: bool = True,
        use_kc: bool = True,
        **_ignored,
    ):
        """use_dmd / use_kc switch off the Hankel-DMD branch / the class-wise Koopman
        head. Both off = mp_transformer + the same regularisation (mixup, label
        smoothing): the ablation that isolates what the Koopman parts add."""
        super().__init__()
        self.use_dmd = bool(use_dmd)
        self.use_kc = bool(use_kc)
        self.n_modes = int(n_modes)
        self.mixup = float(mixup)
        self.label_smoothing = float(label_smoothing)
        self.koopman_weight = float(koopman_weight)
        self.seq = TransformerClassifier(
            feat_dim=int(feat_dim),
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(layers),
            dim_feedforward=d_model * 2,
            dropout=dropout,
            max_len=128,
        )
        mode_in = N_PARTS * MODE_MAP_DIM
        dyn_dim = 128 if self.use_dmd else 0
        self.mode_mlp = None if not self.use_dmd else nn.Sequential(
            nn.LayerNorm(mode_in),
            nn.Linear(mode_in, eig_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(eig_hidden, 64),
        )
        self.eig_mlp = None if not self.use_dmd else nn.Sequential(
            nn.LayerNorm(EIG_DIM),
            nn.Linear(EIG_DIM, eig_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(eig_hidden, 64),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + dyn_dim),
            nn.Dropout(dropout),
            nn.Linear(d_model + dyn_dim, num_classes),
        )
        self.koopman = ClassKoopmanHead(d_model, num_classes, rank=int(koopman_rank)) if self.use_kc else None
        self.dyn_scale = nn.Parameter(torch.tensor(0.5)) if self.use_kc else None
        for name in PART_ORDER:
            self.register_buffer(f"_idx_{name}", torch.tensor(PARTS[name], dtype=torch.long), persistent=False)

    def _part_mode_vec(self, modes: torch.Tensor) -> torch.Tensor:
        parts = [modes.index_select(1, getattr(self, f"_idx_{name}")).mean(1) for name in PART_ORDER]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        eig: torch.Tensor | None = None,
        modes: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if x.dim() == 4:
            n, c, t, v = x.shape
            x = x.permute(0, 2, 1, 3).reshape(n, t, c * v)
        tokens = self.seq.encode(x)
        feats = [tokens.mean(dim=1)]
        if self.use_dmd:
            if eig is None:
                eig = x.new_zeros(x.size(0), EIG_DIM)
            if modes is None:
                modes = x.new_zeros(x.size(0), N_JOINTS, MODE_MAP_DIM)
            feats += [self.eig_mlp(eig), self.mode_mlp(self._part_mode_vec(modes))]
        logits = self.head(torch.cat(feats, dim=-1))
        k_scores = None
        if self.use_kc:
            k_scores = self.koopman.scores(tokens)
            logits = logits + self.dyn_scale * k_scores
        if return_aux:
            return logits, k_scores
        return logits


def _ce(logits, y, criterion, smoothing: float):
    return F.cross_entropy(
        logits,
        y,
        weight=getattr(criterion, "weight", None),
        label_smoothing=float(smoothing),
    )


def kdf_forward(model, batch, criterion, device, train: bool = True):
    """Unpack (landmark sequence, koopman eigs, spatial modes); mixup while training."""
    x, y = batch
    if isinstance(x, (tuple, list)):
        pose, eig = x[0], x[1]
        modes = x[2] if len(x) > 2 else None
        pose = pose.to(device, non_blocking=True)
        eig = eig.to(device, non_blocking=True)
        if modes is not None:
            modes = modes.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        pose = x.to(device, non_blocking=True)
        eig, modes = None, None
        y = y.to(device, non_blocking=True)

    mix = float(getattr(model, "mixup", 0.0) or 0.0) if train and model.training else 0.0
    smooth = float(getattr(model, "label_smoothing", 0.0) or 0.0) if train else 0.0
    kw = float(getattr(model, "koopman_weight", 0.0) or 0.0) if train else 0.0

    def _loss(logits, aux, target):
        loss = _ce(logits, target, criterion, smooth)
        if aux is not None and kw > 0:
            loss = loss + kw * _ce(aux, target, criterion, 0.0)
        return loss

    if mix > 0.0 and pose.size(0) > 1:
        lam = float(np.random.beta(mix, mix))
        idx = torch.randperm(pose.size(0), device=device)
        pose = lam * pose + (1.0 - lam) * pose[idx]
        if eig is not None:
            eig = lam * eig + (1.0 - lam) * eig[idx]
        if modes is not None:
            modes = lam * modes + (1.0 - lam) * modes[idx]
        out = model(pose, eig, modes, return_aux=True)
        logits, aux = out if isinstance(out, tuple) else (out, None)
        loss = lam * _loss(logits, aux, y) + (1.0 - lam) * _loss(logits, aux, y[idx])
        return logits, y, loss

    out = model(pose, eig, modes, return_aux=train)
    logits, aux = out if isinstance(out, tuple) else (out, None)
    return logits, y, _loss(logits, aux, y)

"""Batch augmentation on the GPU, applied in training only (never at evaluation).

Point modalities (landmarks, skeleton, parts, and the pose input of the KDF models):
  random in-plane rotation (+-15 deg), scale (+-15 %), small anisotropy, shift, and
  coordinate noise, on detected points only (missing points stay exactly 0), then a
  non-circular temporal warp (speed 0.8-1.2, shift +-15 %, clamped at the edges).
Spectral modalities (FFT, CWT features): temporal warp and multiplicative feature noise.
KDF spectra/modes are not augmented (they describe the unaugmented clip; mixup still
mixes them).
No horizontal flip: it would swap the dominant hand.
"""
from __future__ import annotations

import math

import torch


def time_warp(x: torch.Tensor, dim: int = 1, max_speed: float = 0.2, max_shift: float = 0.15) -> torch.Tensor:
    b, t = x.size(0), x.size(dim)
    dev = x.device
    speed = 1 + (torch.rand(b, 1, device=dev) * 2 - 1) * max_speed
    shift = (torch.rand(b, 1, device=dev) * 2 - 1) * max_shift * t
    base = torch.arange(t, device=dev, dtype=torch.float32).unsqueeze(0) - (t - 1) / 2
    src = (base * speed + (t - 1) / 2 + shift).round().clamp(0, t - 1).long()  # (B, T)
    shape = [b] + [1] * (x.dim() - 1)
    shape[dim] = t
    return x.gather(dim, src.view(shape).expand_as(x))


def point_aug(p: torch.Tensor, rot_deg: float = 15, scale: float = 0.15, shift: float = 0.1,
              noise: float = 0.01) -> torch.Tensor:
    """p: (B, T, V, 3) normalised points, missing = 0."""
    b, dev, dt = p.size(0), p.device, p.dtype
    mask = (p.abs().sum(-1, keepdim=True) > 0).to(dt)
    th = (torch.rand(b, device=dev) * 2 - 1) * math.radians(rot_deg)
    c, s = torch.cos(th), torch.sin(th)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(dt)  # (B,2,2)
    sc = (1 + (torch.rand(b, 1, device=dev) * 2 - 1) * scale).to(dt)
    an = (1 + (torch.rand(b, 2, device=dev) * 2 - 1) * 0.08).to(dt)
    sh = ((torch.rand(b, 2, device=dev) * 2 - 1) * shift).to(dt)
    xy = torch.einsum("btvi,bji->btvj", p[..., :2], rot) * (sc * an)[:, None, None] + sh[:, None, None]
    z = p[..., 2:] * sc[:, None, None]
    out = torch.cat([xy, z], -1)
    out = out + torch.randn_like(out) * noise
    out = out * mask
    return time_warp(out, dim=1)


def augment(modality: str, xs: tuple) -> tuple:
    x0 = xs[0]
    if modality in ("landmarks", "skeleton_kdf"):
        b, t, f = x0.shape
        x0 = point_aug(x0.view(b, t, f // 3, 3)).reshape(b, t, f)
    elif modality == "skeleton":  # (B, 3, T, V)
        x0 = point_aug(x0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
    elif modality in ("parts", "parts_kdf"):
        x0 = point_aug(x0)
    elif modality in ("spectral_fft", "spectral_cwt"):
        x0 = time_warp(x0, dim=1)
        x0 = x0 * (1 + torch.randn_like(x0) * 0.05)
    return (x0,) + tuple(xs[1:])

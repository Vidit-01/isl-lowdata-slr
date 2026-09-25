"""Kinematic, FFT, and CWT feature builders over joint trajectories."""
from __future__ import annotations

import numpy as np


def finite_diff(x: np.ndarray) -> np.ndarray:
    """First difference along time, padded so the length is unchanged."""
    d = np.diff(x, axis=0)
    return np.concatenate([d[:1], d], axis=0).astype(np.float32)


def kinematic_stack(x: np.ndarray) -> np.ndarray:
    """(T, D) -> (T, 3D) position | velocity | acceleration."""
    vel = finite_diff(x)
    acc = finite_diff(vel)
    return np.concatenate([x, vel, acc], axis=-1).astype(np.float32)


def _window_slice(t: int, length: int, win: int) -> tuple[int, int]:
    half = win // 2
    start = max(0, t - half)
    end = min(length, start + win)
    start = max(0, end - win)
    return start, end


def sliding_fft_features(x: np.ndarray, win: int = 8) -> np.ndarray:
    """Per-frame adaptive-window FFT: dominant magnitude + phase, plus kinematics.

    x: (T, D)  ->  (T, 5D)  = pos | vel | acc | peak_mag | peak_phase
    """
    x = x.astype(np.float32)
    t_len, dim = x.shape
    kin = kinematic_stack(x)
    mag_peak = np.zeros((t_len, dim), dtype=np.float32)
    phase_peak = np.zeros((t_len, dim), dtype=np.float32)
    win = max(4, min(win, t_len))
    for t in range(t_len):
        s, e = _window_slice(t, t_len, win)
        spec = np.fft.rfft(x[s:e], axis=0)
        mag = np.abs(spec)
        phase = np.angle(spec)
        idx = mag.argmax(axis=0)
        rows = np.arange(dim)
        mag_peak[t] = mag[idx, rows]
        phase_peak[t] = phase[idx, rows]
    return np.concatenate([kin, mag_peak, phase_peak], axis=-1)


def _morlet(length: int, scale: float, omega0: float = 5.0) -> np.ndarray:
    t = np.arange(length) - (length // 2)
    u = t / max(scale, 1e-6)
    psi = (np.pi ** -0.25) * np.exp(1j * omega0 * u) * np.exp(-0.5 * u * u)
    return psi.astype(np.complex64)


def cwt_scalogram(x: np.ndarray, n_scales: int = 16, omega0: float = 5.0) -> np.ndarray:
    """Morlet CWT magnitude. x: (T,) -> (n_scales, T)."""
    t_len = int(x.shape[0])
    scales = np.geomspace(1.0, max(t_len / 2.0, 2.0), n_scales)
    out = np.zeros((n_scales, t_len), dtype=np.float32)
    sig = x.astype(np.float32)
    for i, scale in enumerate(scales):
        psi = _morlet(t_len, float(scale), omega0=omega0)
        conv = np.convolve(sig, psi, mode="same")
        out[i] = np.abs(conv).astype(np.float32)
    return out


def sliding_cwt_features(x: np.ndarray, n_scales: int = 16, n_bands: int = 4) -> np.ndarray:
    """Band-pooled CWT + kinematics per frame.

    x: (T, D) -> (T, D * (3 + n_bands))
    """
    x = x.astype(np.float32)
    t_len, dim = x.shape
    kin = kinematic_stack(x)
    n_bands = max(1, n_bands)
    bands = np.zeros((t_len, dim * n_bands), dtype=np.float32)
    for d in range(dim):
        scalo = cwt_scalogram(x[:, d], n_scales=n_scales)  # (S, T)
        edges = np.linspace(0, scalo.shape[0], n_bands + 1).astype(int)
        for b in range(n_bands):
            chunk = scalo[edges[b] : max(edges[b + 1], edges[b] + 1)]
            bands[:, d * n_bands + b] = chunk.mean(axis=0)
    return np.concatenate([kin, bands], axis=-1).astype(np.float32)

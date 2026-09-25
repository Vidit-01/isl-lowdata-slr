"""Datasets for RGB, landmark, skeleton, and spectral modalities."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from islr.common.landmarks import cache_key, load_or_extract

from islr.models.features import sliding_cwt_features, sliding_fft_features
from islr.models.kdf import CACHE_TAG, KDF_IN_CHANNELS, N_MODES, kdf_joint_features
from islr.models.skeleton import IN_CHANNELS, N_JOINTS, joints_to_bones, landmarks_to_joints, pose_hands_vec


def sample_rgb_frames(path: str, num_frames: int, size: int) -> np.ndarray:
    """Uniformly sample RGB frames as (T, C, H, W) float32 in [0, 1]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.zeros((num_frames, 3, size, size), dtype=np.float32)
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return np.zeros((num_frames, 3, size, size), dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(np.int64)
    out = np.zeros((num_frames, 3, size, size), dtype=np.float32)
    for t, i in enumerate(idx.tolist()):
        rgb = cv2.cvtColor(frames[int(i)], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size))
        out[t] = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return out


def temporal_jitter(x: np.ndarray, max_shift: float = 0.15, max_stretch: float = 0.2) -> np.ndarray:
    """Random time shift + stretch with edge clamping along axis 0.

    Replaces `np.roll`, which wrapped the end of the sign around to its start and
    produced sequences that never occur in real signing."""
    t = x.shape[0]
    if t < 2:
        return x
    speed = np.random.uniform(1.0 - max_stretch, 1.0 + max_stretch)
    shift = np.random.uniform(-max_shift, max_shift) * t
    src = (np.arange(t) - (t - 1) / 2.0) * speed + (t - 1) / 2.0 + shift
    idx = np.clip(np.rint(src), 0, t - 1).astype(np.int64)
    return x[idx]


def coord_noise(x: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Gaussian noise on detected landmarks only: missing landmarks are exactly 0 and
    must stay 0, or models (and KDF's occlusion fill) can no longer tell they are missing."""
    return (x + np.random.normal(0, std, size=x.shape).astype(np.float32) * (x != 0)).astype(np.float32)


def _feature_name(path: str, num_frames: int) -> str:
    # Same identity as the landmark cache (path hash): two clips with the same file stem
    # in different word folders used to share one feature-cache entry.
    return cache_key(path, num_frames)[: -len(".npy")]


class LandmarkSeqDataset(Dataset):
    """(T, F) landmark sequences with optional per-item transform."""

    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        cache_dir: Path,
        num_frames: int,
        augment: bool = False,
        require_cache: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.augment = augment
        self.require_cache = require_cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int) -> np.ndarray:
        arr = load_or_extract(
            self.paths[idx],
            self.cache_dir,
            self.num_frames,
            require_cache=self.require_cache,
        ).astype(np.float32)
        if self.transform is None:
            return arr
        tag = getattr(self.transform, "__name__", "transform")
        extra = self.cache_dir.parent / f"{tag}_T{self.num_frames}"
        extra.mkdir(parents=True, exist_ok=True)
        out = extra / f"{_feature_name(self.paths[idx], self.num_frames)}.npy"
        if out.exists():
            return np.load(out).astype(np.float32)
        feat = self.transform(arr).astype(np.float32)
        np.save(out, feat)
        return feat

    def __getitem__(self, idx: int):
        x = self._load(idx)
        if self.augment:
            if np.random.rand() < 0.5:
                if self.transform is None or self.transform is pose_hands_transform:
                    x = coord_noise(x)
                else:  # derived spectral features: plain feature noise
                    x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            if x.ndim >= 1 and np.random.rand() < 0.5:
                x = temporal_jitter(x)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.labels[idx], dtype=torch.long)


class SkeletonDataset(LandmarkSeqDataset):
    """ST-GCN layout: (C, T, V) from cached Holistic landmarks."""

    def __init__(self, *args, use_bone: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bone = use_bone

    def __getitem__(self, idx: int):
        seq = load_or_extract(
            self.paths[idx],
            self.cache_dir,
            self.num_frames,
            require_cache=self.require_cache,
        ).astype(np.float32)
        joints = landmarks_to_joints(seq)  # (T, V, C)
        if self.augment:
            if np.random.rand() < 0.5:
                joints = coord_noise(joints)
            if np.random.rand() < 0.5:
                joints = temporal_jitter(joints)
        x = np.transpose(joints, (2, 0, 1))  # (C, T, V)
        if self.use_bone:
            bone = np.transpose(joints_to_bones(joints), (2, 0, 1))
            x = np.concatenate([x, bone], axis=0)  # (2C, T, V)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.labels[idx], dtype=torch.long)


class KDFSkeletonDataset(LandmarkSeqDataset):
    """MediaPipe pose+hands sequence + cached Hankel-DMD features for kdf_transformer.
    full_seq=True returns the whole landmark sequence (for kdf_partformer)."""

    def __init__(self, *args, full_seq: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_seq = full_seq

    def __getitem__(self, idx):
        seq = load_or_extract(
            self.paths[idx],
            self.cache_dir,
            self.num_frames,
            require_cache=self.require_cache,
        ).astype(np.float32)
        extra = self.cache_dir.parent / f"{CACHE_TAG}_T{self.num_frames}_m{N_MODES}"
        extra.mkdir(parents=True, exist_ok=True)
        feat_path = extra / f"{_feature_name(self.paths[idx], self.num_frames)}.npz"
        pose = seq if self.full_seq else pose_hands_vec(seq)
        eig = modes = None
        if feat_path.exists():
            blob = np.load(feat_path)
            if "x" in blob and "eig" in blob and "modes" in blob and blob["x"].shape[0] == KDF_IN_CHANNELS:
                eig = blob["eig"].astype(np.float32)
                modes = blob["modes"].astype(np.float32)
        if eig is None:
            joints = landmarks_to_joints(seq)
            x_out, eig, modes = kdf_joint_features(joints)
            np.savez_compressed(feat_path, x=x_out, eig=eig, modes=modes)
        if self.augment:
            if np.random.rand() < 0.5:
                pose = coord_noise(pose)
            if np.random.rand() < 0.5:
                pose = temporal_jitter(pose)
        return (
            torch.from_numpy(np.ascontiguousarray(pose)),
            torch.from_numpy(np.ascontiguousarray(eig)),
            torch.from_numpy(np.ascontiguousarray(modes)),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def collate_kdf(batch):
    xs, eigs, modes, ys = zip(*batch)
    return (torch.stack(xs, 0), torch.stack(eigs, 0), torch.stack(modes, 0)), torch.stack(ys, 0)


class RGBClipDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        num_frames: int = 16,
        size: int = 112,
        augment: bool = False,
    ):
        self.paths = paths
        self.labels = labels
        self.num_frames = num_frames
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = sample_rgb_frames(self.paths[idx], self.num_frames, self.size)
        if self.augment and np.random.rand() < 0.5:
            # Color jitter only — no horizontal flip (would swap signing hands).
            jitter = np.random.uniform(0.9, 1.1, size=(3, 1, 1)).astype(np.float32)
            x = np.clip(x * jitter, 0.0, 1.0)
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


def pose_hands_transform(seq: np.ndarray) -> np.ndarray:
    return pose_hands_vec(seq)


def joints_flat_transform(seq: np.ndarray) -> np.ndarray:
    return landmarks_to_joints(seq).reshape(seq.shape[0], N_JOINTS * IN_CHANNELS)


def fft_transform(seq: np.ndarray) -> np.ndarray:
    return sliding_fft_features(joints_flat_transform(seq))


def cwt_transform(seq: np.ndarray) -> np.ndarray:
    return sliding_cwt_features(joints_flat_transform(seq), n_scales=12, n_bands=4)


def collate_xy(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

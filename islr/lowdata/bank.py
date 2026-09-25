"""In-memory feature banks: every clip of the loaded stores, featurised once per modality.

A bank is built from the landmark stores (not from videos), cached as one .npz per
(clip set, modality, frames, trim) and loaded into memory. Low-data runs are a few
batches per epoch, so a DataLoader would cost more than the model; batches are
gathered straight from the bank tensors on the GPU.

Under DDP only rank 0 builds a missing cache while the other ranks wait at a barrier,
then every rank loads the same file.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BANK_VERSION = 2
MODALITIES = ("landmarks", "skeleton", "skeleton_kdf", "spectral_fft", "spectral_cwt", "parts", "parts_kdf")


def featurize(seq: np.ndarray, modality: str) -> tuple[np.ndarray, ...]:
    """seq: (T, 119*3) normalised reduced-layout sequence (`store.to_codebase_seq`)."""
    from islr.fewshot.data import cwt_transform, fft_transform
    from islr.models.kdf import kdf_joint_features
    from islr.models.skeleton import landmarks_to_joints, pose_hands_vec

    t = seq.shape[0]
    if modality == "landmarks":
        return (pose_hands_vec(seq),)
    if modality == "skeleton":
        return (np.transpose(landmarks_to_joints(seq), (2, 0, 1)),)
    if modality == "spectral_fft":
        return (fft_transform(seq),)
    if modality == "spectral_cwt":
        return (cwt_transform(seq),)
    if modality in ("skeleton_kdf", "parts_kdf"):
        _, eig, modes = kdf_joint_features(landmarks_to_joints(seq))
        x = pose_hands_vec(seq) if modality == "skeleton_kdf" else seq.reshape(t, -1, 3)
        return (x, eig, modes)
    if modality == "parts":
        return (seq.reshape(t, -1, 3),)
    raise ValueError(f"no bank for modality {modality!r}")


def _work(args):
    path, modality, num_frames, trim = args
    from islr.lowdata.store import to_codebase_seq

    raw = np.load(path).astype(np.float32)
    seq = to_codebase_seq(raw, num_frames=num_frames, trim=trim, reduced=True)
    return [a.astype(np.float32) for a in featurize(seq, modality)]


def _init(paths):
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)


def bank_file(df: pd.DataFrame, modality: str, num_frames: int, trim: bool, cache_dir: str | Path) -> Path:
    blob = "|".join(sorted(df["source"] + ":" + df["key"])) + f"|{modality}|{num_frames}|{trim}|v{BANK_VERSION}"
    h = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"bank_{modality}_T{num_frames}_{'trim' if trim else 'full'}_{h}.npz"


def build_bank(df: pd.DataFrame, modality: str, num_frames: int = 30, trim: bool = True,
               cache_dir: str | Path = "cache", workers: int | None = None) -> Path:
    out = bank_file(df, modality, num_frames, trim, cache_dir)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(p, modality, num_frames, trim) for p in df["abs_path"]]
    workers = workers if workers is not None else min(8, max(1, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    heavy = modality in ("spectral_cwt", "spectral_fft", "skeleton_kdf", "parts_kdf")
    if workers > 1 and len(jobs) > 64 and heavy:
        from multiprocessing import get_context

        with get_context("spawn").Pool(workers, _init, (list(sys.path),)) as pool:
            feats = pool.map(_work, jobs, chunksize=16)
    else:
        feats = [_work(j) for j in jobs]
    arrays = {f"a{i}": np.stack([f[i] for f in feats]) for i in range(len(feats[0]))} if feats else {}
    keys = (df["source"] + ":" + df["key"]).to_numpy()
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, keys=keys, **arrays)
    os.replace(tmp, out)
    print(f"[bank] {modality}: {len(df)} clips -> {out.name} ({time.time() - t0:.0f}s)", flush=True)
    return out


class Bank:
    """Arrays aligned with the rows of the clip table `df` (same order)."""

    def __init__(self, path: Path, df: pd.DataFrame):
        blob = np.load(path, allow_pickle=True)
        keys = list(blob["keys"])
        pos = {k: i for i, k in enumerate(keys)}
        want = (df["source"] + ":" + df["key"]).tolist()
        order = np.asarray([pos[k] for k in want], dtype=np.int64)
        n = sum(1 for k in blob.files if k.startswith("a"))
        self.arrays = [blob[f"a{i}"][order] for i in range(n)]
        self.tensors = None

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays))

    def to(self, device):
        import torch

        self.tensors = [torch.from_numpy(a).to(device) for a in self.arrays]
        return self

    def gather(self, idx, device=None):
        """Rows `idx` (LongTensor, any device) -> tuple of batch tensors on `device`."""
        bdev = self.tensors[0].device
        out = tuple(t.index_select(0, idx.to(bdev)) for t in self.tensors)
        return out if device is None else tuple(t.to(device, non_blocking=True) for t in out)


def load_bank(df: pd.DataFrame, modality: str, num_frames: int = 30, trim: bool = True,
              cache_dir: str | Path = "cache", rank: int = 0, world: int = 1, workers: int | None = None) -> Bank:
    path = bank_file(df, modality, num_frames, trim, cache_dir)
    if rank == 0 and not path.exists():
        build_bank(df, modality, num_frames, trim, cache_dir, workers)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
    return Bank(path, df)

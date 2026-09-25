"""Shared helpers for ISL recognition models."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "ISL_DATASET"
METADATA = DATASET_DIR / "metadata.csv"
CACHE_DIR = ROOT / "outputs" / "cache"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
WEIGHTS_DIR = ROOT / "outputs" / "weights"


def default_dataset_dir() -> Path:
    """Resolve the isolated-sign corpus: env, cwd, then known folders under ROOT."""
    env = os.environ.get("ISL_DATASET_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "metadata.csv").exists():
        return cwd.resolve()
    for name in ("ISL_DATASET", "ISL_DATASET_8WORDS"):
        candidate = ROOT / name
        if (candidate / "metadata.csv").exists():
            return candidate.resolve()
    return DATASET_DIR.resolve()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(min_clips: int = 1, dataset_dir: str | Path | None = None) -> pd.DataFrame:
    ddir = Path(dataset_dir).expanduser().resolve() if dataset_dir is not None else default_dataset_dir()
    meta = ddir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"metadata.csv not found in {ddir}")
    df = pd.read_csv(meta)
    df["video_path"] = (
        df["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace(r"^.*?ISL_DATASET_8WORDS/", "", regex=True)
        .str.replace(r"^.*?ISL_DATASET/", "", regex=True)
    )
    counts = df.groupby("word").size()
    keep = counts[counts >= min_clips].index
    df = df[df["word"].isin(keep)].copy()
    df["abs_path"] = df["video_path"].map(lambda p: str(ddir / p))
    df.attrs["dataset_dir"] = str(ddir)
    return df.reset_index(drop=True)


def build_label_maps(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    words = sorted(set(words))
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test. Tiny classes (<3) go train-only (or train+val if n==2)."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for _, g in df.groupby("word"):
        # copy: pandas Index views can be read-only (numpy shuffle fails otherwise)
        idx = np.array(g.index, dtype=np.int64, copy=True)
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx.extend(idx.tolist())
            continue
        if n == 2:
            train_idx.append(int(idx[0]))
            val_idx.append(int(idx[1]))
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test : n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val :].tolist())
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def configure_cuda_gpu() -> torch.device:
    """Enable cuDNN benchmark / matmul settings for NVIDIA GPUs (T4, etc.)."""
    torch.backends.cudnn.benchmark = True
    # TF32 is Ampere+; harmless no-op on T4 (Turing)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU")
        return torch.device("cpu")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  VRAM={props.total_memory / 1e9:.1f} GB")
    return torch.device("cuda")


# Backwards-compatible aliases
configure_t4 = configure_cuda_gpu
configure_l40s = configure_cuda_gpu

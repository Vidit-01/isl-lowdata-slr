"""Shared fit/eval loop for every arch.md baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from common import CACHE_DIR, CHECKPOINT_DIR, WEIGHTS_DIR, save_json
from common.engine import append_train_log, evaluate, save_weights, train_one_epoch

from .data import (
    RGBClipDataset,
    SkeletonDataset,
    LandmarkSeqDataset,
    collate_xy,
    cwt_transform,
    fft_transform,
    pose_hands_transform,
)
from .registry import BaselineSpec, get_spec


def class_weights(train_y, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = pd.Series(list(train_y)).value_counts().reindex(range(num_classes), fill_value=1)
    w = 1.0 / torch.tensor(counts.values, dtype=torch.float32)
    return (w / w.sum() * num_classes).to(device)


def merge_cfg(spec: BaselineSpec, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = dict(spec.defaults)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _cache_dir(num_frames: int) -> Path:
    return CACHE_DIR / f"landmarks_T{num_frames}"


def ensure_landmark_cache(num_frames: int, min_files: int = 1) -> Path:
    cache = _cache_dir(num_frames)
    n = len(list(cache.glob("*.npy"))) if cache.exists() else 0
    if n < min_files:
        raise SystemExit(
            f"Landmark cache almost empty ({n} files in {cache}).\n"
            f"Extract first:\n"
            f"  python models/mediapipe_transformer/extract_landmarks.py --num-frames {num_frames}\n"
            f"Then re-run training."
        )
    print(f"Using landmark cache: {cache} ({n} .npy files)")
    return cache


def _loader_kwargs(num_workers: int, pin_memory: bool) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_xy,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
    return kw


def make_loaders(
    spec: BaselineSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    bs = int(cfg["batch_size"])
    workers = int(cfg.get("num_workers", 0))
    kw = _loader_kwargs(workers, pin_memory)
    t = int(cfg["num_frames"])

    if spec.modality == "rgb":
        size = int(cfg.get("size", 112))
        train_ds = RGBClipDataset(train_df["abs_path"].tolist(), train_df["y"].tolist(), t, size, True)
        val_ds = RGBClipDataset(val_df["abs_path"].tolist(), val_df["y"].tolist(), t, size, False)
        test_ds = RGBClipDataset(test_df["abs_path"].tolist(), test_df["y"].tolist(), t, size, False)
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=bs, shuffle=False, **kw),
            DataLoader(test_ds, batch_size=bs, shuffle=False, **kw),
        )

    cache = ensure_landmark_cache(t)
    transform = {
        "landmarks": pose_hands_transform,
        "spectral_fft": fft_transform,
        "spectral_cwt": cwt_transform,
    }.get(spec.modality)

    def _mk(df, augment: bool):
        paths, ys = df["abs_path"].tolist(), df["y"].tolist()
        if spec.modality == "skeleton":
            return SkeletonDataset(paths, ys, cache, t, augment=augment, require_cache=True)
        return LandmarkSeqDataset(
            paths, ys, cache, t, augment=augment, require_cache=True, transform=transform
        )

    return (
        DataLoader(_mk(train_df, True), batch_size=bs, shuffle=True, **kw),
        DataLoader(_mk(val_df, False), batch_size=bs, shuffle=False, **kw),
        DataLoader(_mk(test_df, False), batch_size=bs, shuffle=False, **kw),
    )


class Trainer:
    """AMP + cosine schedule + best-val checkpoint + one-shot test eval."""

    def __init__(
        self,
        spec: BaselineSpec,
        model: nn.Module,
        cfg: dict[str, Any],
        labels: dict[str, Any],
        device: torch.device,
        train_y,
    ):
        self.spec = spec
        self.model = model.to(device)
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights(train_y, labels["num_classes"], device)
        )
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 1e-2)),
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=int(cfg["epochs"]))
        self.use_amp = bool(spec.use_amp)
        self.scaler = GradScaler(enabled=device.type == "cuda" and self.use_amp)
        self.forward_fn = spec.forward_fn

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> dict[str, Any]:
        name = self.spec.name
        epochs = int(self.cfg["epochs"])
        patience = int(self.cfg.get("patience") or 0)
        weights_dir = WEIGHTS_DIR / name
        weights_dir.mkdir(parents=True, exist_ok=True)
        log_path = weights_dir / "train.log"
        append_train_log(
            log_path,
            f"# {name}  {self.spec.title}  epochs={epochs} batch={self.cfg['batch_size']} lr={self.cfg['lr']}",
        )

        history: list[dict[str, Any]] = []
        best_acc = -1.0
        best_state = None
        stale = 0
        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(
                self.model,
                train_loader,
                self.opt,
                self.criterion,
                self.device,
                self.scaler,
                self.forward_fn,
                use_amp=self.use_amp,
                desc=f"{name} ep{epoch}/{epochs}",
            )
            va = evaluate(
                self.model,
                val_loader,
                self.criterion,
                self.device,
                self.forward_fn,
                use_amp=self.use_amp,
                desc=f"{name} val",
            )
            self.sched.step()
            row = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
            }
            history.append(row)
            append_train_log(
                log_path,
                f"[{name}] epoch {epoch:03d}/{epochs}  "
                f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.3f}  "
                f"val_loss={va['loss']:.4f} val_acc={va['acc']:.3f}",
            )
            if va["acc"] >= best_acc:
                best_acc = va["acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if patience and stale >= patience:
                    append_train_log(log_path, f"[{name}] early stop at epoch {epoch} (patience={patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test = evaluate(
            self.model,
            test_loader,
            self.criterion,
            self.device,
            self.forward_fn,
            use_amp=self.use_amp,
            desc=f"{name} test",
        )
        append_train_log(
            log_path,
            f"[{name}] TEST acc={test['acc']:.3f} loss={test['loss']:.4f} n={test['n']} best_val={best_acc:.3f}",
        )

        ckpt_dir = CHECKPOINT_DIR / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "best_val_acc": best_acc}, ckpt_dir / "best.pt")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        meta = {
            "model": name,
            "title": self.spec.title,
            "modality": spec_modality(self.spec),
            "family": self.spec.family,
            "num_classes": self.labels["num_classes"],
            "best_val_acc": best_acc,
            "n_params": n_params,
            "cfg": {k: _jsonable(v) for k, v in self.cfg.items()},
        }
        save_weights(
            self.model,
            weights_dir,
            meta=meta,
            labels=self.labels,
            history=history,
            test_metrics=test,
        )
        save_json(meta, weights_dir / "meta.json")
        return test


def spec_modality(spec: BaselineSpec) -> str:
    return spec.modality


def _jsonable(v: Any):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def train_one_baseline(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: dict[str, Any],
    device: torch.device,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    spec = get_spec(name)
    cfg = merge_cfg(spec, overrides)
    print(f"\n=== {spec.title} ({name}) ===")
    print(cfg)
    loaders = make_loaders(spec, train_df, val_df, test_df, cfg, pin_memory=device.type == "cuda")
    model = spec.build(labels["num_classes"], cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params:,}  device={device}")
    trainer = Trainer(spec, model, cfg, labels, device, train_df["y"].tolist())
    return trainer.fit(*loaders)

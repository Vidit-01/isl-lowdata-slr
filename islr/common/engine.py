"""Training / evaluation engine with AMP (T4-friendly FP16)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import accuracy, save_json


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    forward_fn: Optional[Callable] = None,
    use_amp: bool = True,
    desc: str = "val",
) -> dict[str, Any]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_pred, all_y = [], []
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        if forward_fn is not None:
            logits, y, loss = forward_fn(model, batch, criterion, device, train=False)
        else:
            x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
        all_pred.append(logits.argmax(-1).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}", acc=f"{total_acc / max(n, 1):.3f}")
    preds = np.concatenate(all_pred) if all_pred else np.array([])
    ys = np.concatenate(all_y) if all_y else np.array([])
    return {
        "loss": total_loss / max(n, 1),
        "acc": total_acc / max(n, 1),
        "n": n,
        "preds": preds.tolist(),
        "labels": ys.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    forward_fn: Optional[Callable] = None,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    desc: str = "train",
) -> dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    amp = use_amp and device.type == "cuda"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        opt.zero_grad(set_to_none=True)
        if forward_fn is not None:
            with autocast("cuda", enabled=amp):
                logits, y, loss = forward_fn(model, batch, criterion, device, train=True)
        else:
            x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast("cuda", enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits.detach(), y) * bs
        n += bs
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}", acc=f"{total_acc / max(n, 1):.3f}")
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


def append_train_log(log_path: Path, line: str, also_print: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
    if also_print:
        print(line)


def save_weights(
    model: nn.Module,
    out_dir: Path,
    meta: dict,
    labels: dict,
    history: list,
    test_metrics: dict,
) -> Path:
    """Persist deployable weights + metrics under outputs/weights/<name>/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "test_acc": test_metrics.get("acc"),
            "test_loss": test_metrics.get("loss"),
        },
        ckpt_path,
    )
    save_json(labels, out_dir / "labels.json")
    save_json(history, out_dir / "history.json")
    # strip large arrays from summary
    summary = {k: v for k, v in test_metrics.items() if k not in ("preds", "labels")}
    summary["preds"] = test_metrics.get("preds", [])
    summary["labels"] = test_metrics.get("labels", [])
    save_json(summary, out_dir / "test_metrics.json")
    return ckpt_path

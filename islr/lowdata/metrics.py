"""Metrics for the low-data study (numpy only)."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean F1 over the classes present in y (classes never tested are not scored)."""
    classes = np.unique(y)
    if not len(classes):
        return float("nan")
    f1 = []
    for c in classes:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def _acc(mask: np.ndarray, ok: np.ndarray) -> float:
    return float(ok[mask].mean()) if mask.any() else float("nan")


def classification_metrics(y: np.ndarray, pred: np.ndarray, words: list[str], targets: list[str],
                           logits: np.ndarray | None = None, proto_pred: np.ndarray | None = None) -> dict:
    """Top-1/top-5/macro-F1 overall, and separately for target (scarce) and rich words.

    target_to_rich_rate: share of target-word test clips predicted as a non-target
    word, i.e. how often a scarce word is absorbed by the well-resourced vocabulary.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    tset = np.array([w in set(targets) for w in words], dtype=bool)
    is_t = tset[y] if len(y) else np.zeros(0, bool)
    ok = pred == y
    out = {
        "n_test": int(len(y)),
        "n_test_target": int(is_t.sum()),
        "top1": _acc(np.ones_like(ok), ok),
        "macro_f1": macro_f1(y, pred),
        "target_top1": _acc(is_t, ok),
        "rich_top1": _acc(~is_t, ok),
        "target_macro_f1": macro_f1(y[is_t], pred[is_t]) if is_t.any() else float("nan"),
        "target_to_rich_rate": float((~tset[pred[is_t]]).mean()) if is_t.any() else float("nan"),
        "rich_to_target_rate": float(tset[pred[~is_t]].mean()) if (~is_t).any() else float("nan"),
    }
    lo, hi = wilson(int(ok.sum()), len(ok))
    out["top1_ci95"] = [lo, hi]
    if is_t.any():
        out["target_top1_ci95"] = list(wilson(int(ok[is_t].sum()), int(is_t.sum())))
    if logits is not None and logits.shape[1] >= 5:
        top5 = np.argsort(-logits, axis=1)[:, :5]
        hit = (top5 == y[:, None]).any(1)
        out["top5"] = _acc(np.ones_like(hit), hit)
        out["target_top5"] = _acc(is_t, hit)
    if proto_pred is not None:
        pok = np.asarray(proto_pred) == y
        out["proto_top1"] = _acc(np.ones_like(pok), pok)
        out["proto_target_top1"] = _acc(is_t, pok)
    out["per_word_top1"] = {w: _acc(y == i, ok) for i, w in enumerate(words) if (y == i).any()}
    return out


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired per-clip correctness."""
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p": float(min(1.0, 2 * p))}


def paired_bootstrap(a_ok: np.ndarray, b_ok: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Mean accuracy difference (b - a) and its 95 % bootstrap CI over clips."""
    rng = np.random.default_rng(seed)
    d = b_ok.astype(float) - a_ok.astype(float)
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

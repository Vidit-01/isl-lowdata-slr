"""Few-shot SLR metrics: Wilson CIs, bootstrap, McNemar, per-class errors."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def wilson_interval(k: int, n: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval for a binomial proportion (better than normal-approx at small n)."""
    if n <= 0:
        return {"p": 0.0, "low": 0.0, "high": 0.0, "n": 0, "k": 0}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    rad = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return {
        "p": float(p),
        "low": float(max(0.0, center - rad)),
        "high": float(min(1.0, center + rad)),
        "n": int(n),
        "k": int(k),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for an arbitrary prediction metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(y_true.size)
    point = float(fn(y_true, y_pred))
    if n == 0:
        return {"point": point, "low": point, "high": point, "n_boot": 0}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": point,
        "low": float(lo),
        "high": float(hi),
        "mean_boot": float(stats.mean()),
        "n_boot": n_boot,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _macro_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    if labels.size == 0:
        return 0.0
    accs = [(y_pred[y_true == c] == c).mean() if (y_true == c).any() else 0.0 for c in labels]
    return float(np.mean(accs))


def per_class_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    for i, name in enumerate(class_names):
        mask = y_true == i
        n = int(mask.sum())
        k = int((y_pred[mask] == i).sum()) if n else 0
        wilson = wilson_interval(k, n)
        confused = ""
        if n and cm[i].sum():
            off = cm[i].copy()
            off[i] = 0
            j = int(off.argmax())
            if off[j] > 0:
                confused = class_names[j]
        rows.append(
            {
                "class": name,
                "n": n,
                "correct": k,
                "acc": wilson["p"],
                "acc_wilson_low": wilson["low"],
                "acc_wilson_high": wilson["high"],
                "most_confused_with": confused,
                "errors": n - k,
            }
        )
    return rows


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar test (binomial on discordant pairs). A vs B on the same items."""
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n01 = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    n_disc = n01 + n10
    if n_disc == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest

        p = float(binomtest(min(n01, n10), n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "a_better": n01,
        "b_better": n10,
        "discordant": n_disc,
        "p_value": p,
        "winner": "tie" if n01 == n10 else ("A" if n01 > n10 else "B"),
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI on (macro-acc A − macro-acc B) with shared resamples."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = int(y_true.size)
    point = _macro_acc(y_true, pred_a) - _macro_acc(y_true, pred_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, max(n, 1), n)
        deltas[i] = _macro_acc(y_true[idx], pred_a[idx]) - _macro_acc(y_true[idx], pred_b[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_macro_acc": float(point),
        "low": float(lo),
        "high": float(hi),
        "a_beats_b": bool(lo > 0),
        "b_beats_a": bool(hi < 0),
    }


def score_split(
    y_true,
    y_pred,
    class_names: list[str],
    best_val_acc: float | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    overall = wilson_interval(int((y_true == y_pred).sum()), int(y_true.size))
    macro_acc = bootstrap_ci(y_true, y_pred, _macro_acc, n_boot=n_boot, seed=seed)
    macro_f1 = bootstrap_ci(y_true, y_pred, _macro_f1, n_boot=n_boot, seed=seed + 1)
    per_class = per_class_scores(y_true, y_pred, class_names)
    gap = None
    if best_val_acc is not None:
        gap = float(best_val_acc) - float(macro_acc["point"])
    return {
        "n": int(y_true.size),
        "overall_acc": overall,
        "macro_acc": macro_acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "best_val_acc": None if best_val_acc is None else float(best_val_acc),
        "val_test_gap": gap,
        "preds": y_pred.tolist(),
        "labels": y_true.tolist(),
    }


def mean_std(values: list[float] | tuple[float, ...]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0, "n": 1}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "n": int(arr.size)}


def pairwise_tests(
    labels: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """McNemar + paired bootstrap on a locked test set. Do not rank on point accuracy alone."""
    names = list(preds_by_model.keys())
    rows: list[dict[str, Any]] = []
    y = np.asarray(labels)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = np.asarray(preds_by_model[a]), np.asarray(preds_by_model[b])
            if pa.shape != y.shape or pb.shape != y.shape:
                continue
            mc = mcnemar_exact(y, pa, pb)
            boot = paired_bootstrap_delta(y, pa, pb, n_boot=n_boot, seed=seed)
            rows.append({"model_a": a, "model_b": b, "mcnemar": mc, "bootstrap": boot})
    return rows

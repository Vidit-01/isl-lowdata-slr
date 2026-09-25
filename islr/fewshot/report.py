"""Build a few-shot comparison table from saved baseline weights + test metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from islr.common import WEIGHTS_DIR, save_json
from .metrics import pairwise_tests
from .protocol import load_protocol
from islr.models.registry import ALL_NAMES, SPECS


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int) and nd == 0:
        return str(v)
    if isinstance(v, int) and v >= 1000:
        return f"{v:,}"
    return str(v)


def _ci(blob: dict | None, nd: int = 3) -> str:
    if not blob:
        return "—"
    point = blob.get("point", blob.get("p"))
    lo = blob.get("low")
    hi = blob.get("high")
    if point is None:
        return "—"
    if lo is None or hi is None:
        return _fmt(point, nd)
    return f"{point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _ms(blob: dict | None, nd: int = 3) -> str:
    if not blob or blob.get("mean") is None:
        return "—"
    std = blob.get("std") or 0.0
    n = blob.get("n") or 0
    if n <= 1:
        return f"{blob['mean']:.{nd}f}"
    return f"{blob['mean']:.{nd}f} ± {std:.{nd}f}"


def collect_rows(weights_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    summary = _load_json(root / "baselines_summary.json") or {}
    rows: list[dict[str, Any]] = []
    for name in ALL_NAMES:
        wdir = root / name
        metrics = _load_json(wdir / "test_metrics.json") or {}
        meta = _load_json(wdir / "meta.json") or {}
        history = _load_json(wdir / "history.json") or []
        spec = SPECS.get(name)
        best_val = meta.get("best_val_acc")
        if best_val is None and history:
            best_val = max((h.get("val_acc") or 0.0) for h in history)
        err = None
        if not (wdir / "model.pt").exists() and not metrics:
            err = (summary.get(name) or {}).get("error")
            if isinstance(summary.get(name), dict) and not err:
                draws = (summary.get(name) or {}).get("draws") or []
                errs = [d.get("error") for d in draws if d.get("error")]
                err = errs[0] if errs else None
        agg = summary.get(name) or {}
        gap = metrics.get("val_test_gap")
        if gap is None and best_val is not None and metrics.get("acc") is not None:
            gap = float(best_val) - float(metrics["acc"])
        rows.append(
            {
                "model": name,
                "title": (spec.title if spec else name),
                "modality": (spec.modality if spec else meta.get("modality")),
                "n_params": meta.get("n_params"),
                "best_val_acc": best_val,
                "test_acc": metrics.get("acc"),
                "test_loss": metrics.get("loss"),
                "n_test": metrics.get("n"),
                "macro_acc": (metrics.get("macro_acc") or {}).get("point"),
                "macro_f1": (metrics.get("macro_f1") or {}).get("point"),
                "macro_f1_ci": metrics.get("macro_f1"),
                "macro_acc_ci": metrics.get("macro_acc"),
                "overall_acc_ci": metrics.get("overall_acc"),
                "per_class": metrics.get("per_class") or [],
                "val_test_gap": gap,
                "macro_f1_draws": agg.get("macro_f1"),
                "test_acc_draws": agg.get("test_acc"),
                "val_test_gap_draws": agg.get("val_test_gap"),
                "preds": metrics.get("preds"),
                "labels": metrics.get("labels"),
                "epochs_ran": len(history) if history else None,
                "error": err,
            }
        )
    return rows


def write_report(weights_dir: Path | None = None) -> Path:
    root = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    proto = load_protocol(root / "fewshot_protocol.json") or {}
    save_json(
        [{k: v for k, v in r.items() if k not in ("preds", "labels", "per_class")} for r in rows],
        root / "baselines_comparison.json",
    )

    csv_path = root / "baselines_comparison.csv"
    fields = [
        "model",
        "title",
        "modality",
        "n_params",
        "best_val_acc",
        "test_acc",
        "macro_acc",
        "macro_f1",
        "val_test_gap",
        "test_loss",
        "n_test",
        "epochs_ran",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_draws = int(proto.get("n_draws") or 1)
    lines = [
        "# Few-shot SLR baseline results",
        "",
        "Headline metric is **macro-F1** (equal weight per sign). Do not rank models on point accuracy alone.",
        "",
    ]
    if proto:
        lines += [
            "## Protocol",
            "",
            f"- Train shots / word: `{proto.get('train_shots')}`",
            f"- Val shots / word: `{proto.get('val_shots')}`",
            f"- Test / word (target): `{proto.get('test_per_class')}`",
            f"- Draws of the train set: `{n_draws}` (locked test, resample train)",
            f"- Protocol seed: `{proto.get('protocol_seed')}`",
            f"- Words: {', '.join(proto.get('words') or [])}",
            "",
        ]
        per = proto.get("per_class") or {}
        if per:
            lines += [
                "| word | pool | train | val | test |",
                "|---|---:|---:|---:|---:|",
            ]
            for word, row in per.items():
                lines.append(
                    f"| {word} | {row.get('n_pool')} | {row.get('n_train')} | "
                    f"{row.get('n_val')} | {row.get('n_test')} |"
                )
            lines.append("")
        warns = proto.get("warnings") or []
        if warns:
            lines.append("**Split warnings** (15 test clips/word needs leftover after train+val; the 7-clip 8-word set cannot reach that):")
            lines.append("")
            for wnote in warns:
                lines.append(f"- {wnote}")
            lines.append("")
        leak = proto.get("leakage") or {}
        if leak.get("train_test") or leak.get("val_test"):
            lines.append(
                "Same-signer/session leakage was detected. Accuracy is inflated and the scarcity claim is weaker."
            )
            lines.append("")

    lines += [
        "## Headline (macro-F1)",
        "",
        "| model | modality | params | macro-F1 (draws) | macro-F1 (last draw, bootstrap 95% CI) | test acc (Wilson 95% CI) | val−test gap | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        params = f"{r['n_params']:,}" if isinstance(r.get("n_params"), int) else "—"
        if r.get("error") and r.get("test_acc") is None:
            f1_s = f"ERROR: {str(r['error'])[:40]}"
            acc_s = "—"
            gap_s = "—"
            f1_draw = "—"
        else:
            f1_s = _ci(r.get("macro_f1_ci"))
            acc_s = _ci(r.get("overall_acc_ci"))
            gap_s = _fmt(r.get("val_test_gap"))
            f1_draw = _ms(r.get("macro_f1_draws"))
        lines.append(
            f"| `{r['model']}` | {r.get('modality') or '—'} | {params} | "
            f"{f1_draw} | {f1_s} | {acc_s} | {gap_s} | {_fmt(r.get('n_test'), 0)} |"
        )

    lines += ["", "## Per-class accuracy (last draw / canonical weights)", ""]
    class_names: list[str] = []
    for r in rows:
        if r.get("per_class"):
            class_names = [c["class"] for c in r["per_class"]]
            break
    if class_names:
        header = "| model | " + " | ".join(class_names) + " | hard sign (lowest acc) |"
        sep = "|---|" + "---:|" * len(class_names) + "---|"
        lines += [header, sep]
        for r in rows:
            pcs = {c["class"]: c for c in (r.get("per_class") or [])}
            if not pcs:
                continue
            cells = []
            worst, worst_acc = None, 2.0
            for name in class_names:
                c = pcs.get(name) or {}
                acc = c.get("acc")
                n = c.get("n") or 0
                k = c.get("correct")
                if acc is None:
                    cells.append("—")
                    continue
                cells.append(f"{acc:.2f} ({k}/{n})" if n else "—")
                if acc < worst_acc:
                    worst_acc, worst = acc, name
            confused = ""
            if worst and pcs.get(worst, {}).get("most_confused_with"):
                confused = f" → {pcs[worst]['most_confused_with']}"
            hard = f"{worst}{confused}" if worst else "—"
            lines.append(f"| `{r['model']}` | " + " | ".join(cells) + f" | {hard} |")
        lines.append("")
        lines.append(
            "Low accuracy on the same gloss across models is a **hard-sign** effect. "
            "A collapse isolated to one architecture (especially `cnn_bilstm`) is a **model** effect."
        )
        lines.append("")

    preds_by_model = {}
    labels_ref = None
    for r in rows:
        if r.get("preds") and r.get("labels"):
            preds_by_model[r["model"]] = np.asarray(r["preds"])
            labels_ref = np.asarray(r["labels"])
    if labels_ref is not None and len(preds_by_model) >= 2:
        pairs = pairwise_tests(labels_ref, preds_by_model)
        save_json(pairs, root / "baselines_pairwise.json")
        lines += [
            "## Pairwise tests on the locked test set",
            "",
            "McNemar (exact) and paired bootstrap on Δ macro-acc. A claim that A beats B needs a CI that excludes 0 or a McNemar p below 0.05 — not a higher point accuracy.",
            "",
            "| A | B | A right / B wrong | B right / A wrong | McNemar p | Δ macro-acc (A−B) 95% CI |",
            "|---|---|---:|---:|---:|---|",
        ]
        for p in pairs:
            mc = p["mcnemar"]
            bt = p["bootstrap"]
            sig = ""
            if bt.get("a_beats_b"):
                sig = " **A > B**"
            elif bt.get("b_beats_a"):
                sig = " **B > A**"
            lines.append(
                f"| `{p['model_a']}` | `{p['model_b']}` | {mc['a_better']} | {mc['b_better']} | "
                f"{mc['p_value']:.3f} | {bt['delta_macro_acc']:.3f} [{bt['low']:.3f}, {bt['high']:.3f}]{sig} |"
            )
        lines.append("")

    lines += [
        "## How to read this under scarcity",
        "",
        "- Primary axis: which model **degrades most gracefully** as shots/class shrink — not peak accuracy on a lucky draw.",
        "- Raw-pixel / high-param models (`cnn_bilstm`) are expected to degrade hardest: no pretraining, no pose inductive bias, everything is learned from ~7 clips.",
        "- Landmark and spectral inputs (`mp_*`, graphs, `fft_bilstm`, `cwt_*`) should hold up better — preprocessing already did feature extraction the net would otherwise need data to learn.",
        "- A large **val−test gap** is the overfitting signal that matters in the few-shot regime; a high val number with a collapsed test is not a win.",
        "- Mean ± std across train-set draws is only meaningful when the leftover pool is larger than k shots. On the current 7-clip/word set the draws are nearly identical.",
        "",
        "Per-model artifacts: `outputs/weights/<name>/{model.pt,labels.json,history.json,test_metrics.json,train.log}`",
        "Locked split: `outputs/weights/fewshot_protocol.json`. Multi-draw weights: `outputs/weights/draws/<dd>/<name>/`.",
    ]

    md_path = root / "baselines_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    return md_path


if __name__ == "__main__":
    write_report()

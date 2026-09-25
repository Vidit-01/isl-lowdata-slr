"""Build a comparison table from saved baseline weights + test metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from common import WEIGHTS_DIR, save_json
from .registry import ALL_NAMES, SPECS


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(weights_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    rows: list[dict[str, Any]] = []
    for name in ALL_NAMES:
        wdir = root / name
        metrics = _load_json(wdir / "test_metrics.json") or {}
        meta = _load_json(wdir / "meta.json") or {}
        history = _load_json(wdir / "history.json") or []
        spec = SPECS.get(name)
        best_val = None
        if history:
            best_val = max((h.get("val_acc") or 0.0) for h in history)
        err = None
        if not (wdir / "model.pt").exists() and not metrics:
            summary = _load_json(root / "baselines_summary.json") or {}
            err = (summary.get(name) or {}).get("error")
        rows.append(
            {
                "model": name,
                "title": (spec.title if spec else name),
                "modality": (spec.modality if spec else meta.get("modality")),
                "n_params": meta.get("n_params"),
                "best_val_acc": meta.get("best_val_acc", best_val),
                "test_acc": metrics.get("acc"),
                "test_loss": metrics.get("loss"),
                "n_test": metrics.get("n"),
                "epochs_ran": len(history) if history else None,
                "error": err,
            }
        )
    return rows


def write_report(weights_dir: Path | None = None) -> Path:
    root = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    save_json(rows, root / "baselines_comparison.json")

    csv_path = root / "baselines_comparison.csv"
    fields = [
        "model",
        "title",
        "modality",
        "n_params",
        "best_val_acc",
        "test_acc",
        "test_loss",
        "n_test",
        "epochs_ran",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# arch.md baseline results",
        "",
        "| model | modality | params | best val acc | test acc | test loss | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(v, nd=3):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{nd}f}"
            if isinstance(v, int) and nd == 0:
                return str(v)
            if isinstance(v, int) and v >= 1000:
                return f"{v:,}"
            return str(v)

        params = f"{r['n_params']:,}" if isinstance(r.get("n_params"), int) else "—"
        test = r.get("test_acc")
        if r.get("error") and test is None:
            test_s = f"ERROR: {r['error'][:40]}"
        else:
            test_s = fmt(test)
        lines.append(
            f"| `{r['model']}` | {r.get('modality') or '—'} | {params} | "
            f"{fmt(r.get('best_val_acc'))} | {test_s} | {fmt(r.get('test_loss'), 4)} | {fmt(r.get('n_test'), 0)} |"
        )
    lines.append("")
    lines.append("Per-model artifacts: `models/_weights/<name>/{model.pt,labels.json,history.json,test_metrics.json,train.log}`")
    md_path = root / "baselines_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    return md_path


if __name__ == "__main__":
    write_report()

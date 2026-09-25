"""Merge results from any number of sweep outputs (one per team member / session).

    python lowdata.py report --out sweeps/member0 sweeps/member1 ... --dest report/ --reference mp_transformer

Writes to --dest:
  results.csv          one row per run (deduplicated by run name; newest wins)
  summary.csv / .md    mean +- std over seeds per (protocol tag, model, K):
                       top1, target_top1 (scarce words), rich_top1 (well-resourced words),
                       target_to_rich_rate, proto_top1, macro_f1
  paired.md            each model vs --reference on identical test clips: exact McNemar
                       (pooled over seeds) and a paired bootstrap CI of the accuracy gap
  curve_<tag>.png      accuracy vs K per model (target and rich separately for scarce)
  checks.md            runs whose test fingerprint disagrees across models for the same
                       (tag, seed) - should never happen - and missing runs of the grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("top1", "target_top1", "rich_top1", "target_to_rich_rate", "proto_top1", "proto_target_top1",
           "macro_f1", "top5")


def collect(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in Path(d).rglob("result.json"):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            m = r.get("metrics", {})
            row = {k: r.get(k) for k in ("run", "model", "family", "modality", "protocol", "tag", "shots", "seed",
                                         "n_words", "n_targets", "n_train", "n_params", "steps", "train_seconds",
                                         "gpu", "test_fingerprint")}
            row.update({k: m.get(k) for k in METRICS})
            row["chance"] = m.get("chance")
            row["n_test"] = m.get("n_test")
            row["dir"] = str(f.parent)
            row["mtime"] = f.stat().st_mtime
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("mtime").drop_duplicates("run", keep="last").reset_index(drop=True)
        df["K"] = df["shots"].fillna(-1).astype(int)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["tag", "model", "K"])
    agg = {f"{m}_mean": (m, "mean") for m in METRICS if m in df}
    agg.update({f"{m}_std": (m, "std") for m in METRICS if m in df})
    agg["n_seeds"] = ("seed", "nunique")
    agg["n_words"] = ("n_words", "first")
    agg["chance"] = ("chance", "first")
    return g.agg(**agg).reset_index()


def _fmt(mean, std):
    if pd.isna(mean):
        return "-"
    return f"{100 * mean:.1f}" + ("" if pd.isna(std) else f" ± {100 * std:.1f}")


def markdown_tables(summ: pd.DataFrame) -> str:
    out = []
    for tag, t in summ.groupby("tag"):
        out.append(f"## {tag}\n")
        cols = ["top1", "target_top1", "rich_top1", "target_to_rich_rate", "proto_top1"] \
            if tag.startswith("scarce") else ["top1", "macro_f1", "proto_top1", "top5"]
        for metric in cols:
            if f"{metric}_mean" not in t or t[f"{metric}_mean"].isna().all():
                continue
            piv_m = t.pivot(index="model", columns="K", values=f"{metric}_mean")
            piv_s = t.pivot(index="model", columns="K", values=f"{metric}_std")
            ks = sorted(piv_m.columns)
            out.append(f"**{metric}** (% , mean ± std over seeds; K = training clips per "
                       f"{'target ' if tag.startswith('scarce') else ''}word, -1 = all)\n")
            out.append("| model | " + " | ".join(f"K={k}" for k in ks) + " |")
            out.append("|---|" + "---|" * len(ks))
            order = piv_m[ks[-1]].sort_values(ascending=False).index if ks else piv_m.index
            for mdl in order:
                out.append(f"| {mdl} | " + " | ".join(_fmt(piv_m.loc[mdl, k], piv_s.loc[mdl, k]) for k in ks) + " |")
            out.append("")
        seeds = t["n_seeds"].min(), t["n_seeds"].max()
        out.append(f"seeds per cell: {seeds[0]}-{seeds[1]}; chance: {100 * t['chance'].iloc[0]:.1f} %\n")
    return "\n".join(out)


def _load_preds(run_dir: str):
    f = Path(run_dir) / "preds.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return dict(zip(z["keys"].tolist(), (z["pred"] == z["y"]).tolist()))


def paired(df: pd.DataFrame, reference: str) -> str:
    from islr.lowdata.metrics import mcnemar, paired_bootstrap

    lines = [f"# Paired comparisons against `{reference}` (same test clips)\n",
             "b = reference right / model wrong, c = reference wrong / model right; "
             "Δ = model − reference top-1 over the pooled seeds, 95 % bootstrap CI.\n"]
    for (tag, k), t in df.groupby(["tag", "K"]):
        ref = t[t["model"] == reference]
        if ref.empty:
            continue
        lines.append(f"## {tag}, K={k}\n")
        lines.append("| model | Δ top1 | 95 % CI | b | c | McNemar p |")
        lines.append("|---|---|---|---|---|---|")
        for mdl, tm in t.groupby("model"):
            if mdl == reference:
                continue
            a_ok, b_ok = [], []
            for _, r in tm.iterrows():
                rr = ref[ref["seed"] == r["seed"]]
                if rr.empty:
                    continue
                pa, pb = _load_preds(rr.iloc[0]["dir"]), _load_preds(r["dir"])
                if pa is None or pb is None:
                    continue
                common = sorted(set(pa) & set(pb))
                a_ok += [pa[c] for c in common]
                b_ok += [pb[c] for c in common]
            if not a_ok:
                continue
            a_ok, b_ok = np.array(a_ok), np.array(b_ok)
            mc = mcnemar(a_ok, b_ok)
            d, lo, hi = paired_bootstrap(a_ok, b_ok)
            lines.append(f"| {mdl} | {100 * d:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {mc['b']} | {mc['c']} | "
                         f"{mc['p']:.3g} |")
        lines.append("")
    return "\n".join(lines)


def checks(df: pd.DataFrame) -> str:
    lines = ["# Consistency checks\n"]
    bad = df.groupby(["tag", "seed"])["test_fingerprint"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        lines.append("Test set differs between runs of the same (tag, seed) - investigate:\n")
        lines += [f"- {t} seed {s}" for (t, s) in bad.index]
    else:
        lines.append("Test sets identical across models and K for every (tag, seed): OK\n")
    cells = df.groupby(["tag", "model", "K"])["seed"].nunique()
    want = int(cells.max()) if len(cells) else 0
    short = cells[cells < want]
    if len(short):
        lines.append(f"\nCells with fewer than {want} seeds:\n")
        lines += [f"- {t} / {m} / K={k}: {n}" for (t, m, k), n in short.items()]
    return "\n".join(lines) + "\n"


def data_checks(outs: list[str]) -> str:
    """Compare the stores each member trained on (written by the notebook from
    `lowdata.py data`). Different fingerprints mean different clip tables, hence
    different splits, so results from those members must not be pooled."""
    import json

    seen = {}
    for o in outs:
        for f in sorted(Path(o).glob("_sweep/data_member*.json")):
            seen[f"{Path(o).parent.name}/{f.stem}"] = json.loads(f.read_text())
    if not seen:
        return ""
    lines = ["\n## Data used by each member\n", "| member output | store | clips | words | keys |", "|---|---|---|---|---|"]
    per_store = {}
    for who, st in seen.items():
        for name, info in st.items():
            lines.append(f"| {who} | {name} | {info.get('n_clips')} | {info.get('n_words')} | {info.get('keys_sha1')} |")
            per_store.setdefault(name, set()).add(info.get("keys_sha1"))
    bad = [n for n, v in per_store.items() if len(v) > 1]
    lines.append("")
    lines.append(f"**Members used different data for {', '.join(bad)}: their splits differ - do not pool them.**"
                 if bad else "Every member used identical stores: OK")
    return "\n".join(lines) + "\n"


def plots(summ: pd.DataFrame, dest: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    for tag, t in summ.groupby("tag"):
        panels = ["target_top1", "rich_top1"] if tag.startswith("scarce") else ["top1"]
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), squeeze=False)
        for ax, metric in zip(axes[0], panels):
            for mdl, tm in t[t["K"] > 0].groupby("model"):
                tm = tm.sort_values("K")
                ax.errorbar(tm["K"], 100 * tm[f"{metric}_mean"], yerr=100 * tm[f"{metric}_std"].fillna(0),
                            marker="o", capsize=2, label=mdl)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("training clips per " + ("target " if tag.startswith("scarce") else "") + "word (K)")
            ax.set_ylabel(f"{metric} (%)")
            ax.axhline(100 * t["chance"].iloc[0], color="grey", ls=":", lw=1)
            ax.set_title(f"{tag}: {metric}")
            ax.grid(alpha=0.3)
        axes[0][-1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        f = dest / f"curve_{tag}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(f.name)
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lowdata.py report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", nargs="+", required=True, help="sweep output dirs (any number)")
    p.add_argument("--dest", default=None, help="default: <first --out>/_report")
    p.add_argument("--reference", default="mp_transformer")
    a = p.parse_args(argv)
    dest = Path(a.dest or Path(a.out[0]) / "_report")
    dest.mkdir(parents=True, exist_ok=True)
    df = collect(a.out)
    if df.empty:
        print("no result.json found")
        return 1
    df.drop(columns=["mtime"]).to_csv(dest / "results.csv", index=False)
    summ = summarise(df)
    summ.to_csv(dest / "summary.csv", index=False)
    md = markdown_tables(summ)
    (dest / "summary.md").write_text(md, encoding="utf-8")
    (dest / "paired.md").write_text(paired(df, a.reference), encoding="utf-8")
    (dest / "checks.md").write_text(checks(df) + data_checks(a.out), encoding="utf-8")
    figs = plots(summ, dest)
    print(md)
    print(f"[report] {len(df)} runs -> {dest} ({', '.join(['summary.md', 'paired.md', 'checks.md'] + figs)})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())

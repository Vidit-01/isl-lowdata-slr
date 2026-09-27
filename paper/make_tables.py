"""Build every table and figure of the paper from the sweep outputs.

    python paper/make_tables.py --sweeps sweeps            # graph models excluded (default)
    python paper/make_tables.py --sweeps sweeps --include-graph

Reads <sweeps>/<grid>/<model>/K<k>/s<seed>/{result.json,preds.npz,split.json}; writes
paper/generated/*.tex (tables and \\newcommand numbers) and paper/figures/*.pdf.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRAPH = ("stgcn", "ctr_gcn", "td_gcn", "pgf_slr")
REF = "mp_transformer"
TARGETS = ("eat", "go", "hello", "help", "no", "please", "water", "yes")

# display order: baselines, spectral, KDF family, part models (ours)
ORDER = ["mp_bilstm", "mp_transformer", "stgcn", "ctr_gcn", "td_gcn", "hwgat", "pgf_slr",
         "fft_bilstm", "cwt_bilstm", "cwt_transformer",
         "kdf_transformer", "kdf_transformer_nodmd", "kdf_transformer_nokc", "mp_transformer_reg",
         "partformer", "partformer_cos", "conv1d_former", "kdf_partformer"]
GROUP = {"mp_bilstm": "Landmark", "mp_transformer": "Landmark",
         "stgcn": "Graph", "ctr_gcn": "Graph", "td_gcn": "Graph", "hwgat": "Graph", "pgf_slr": "Graph",
         "fft_bilstm": "Spectral", "cwt_bilstm": "Spectral", "cwt_transformer": "Spectral",
         "kdf_transformer": "Koopman", "kdf_transformer_nodmd": "Koopman", "kdf_transformer_nokc": "Koopman",
         "mp_transformer_reg": "Koopman",
         "partformer": "Part", "partformer_cos": "Part", "conv1d_former": "Part", "kdf_partformer": "Part"}
NAME = {m: m.replace("_", r"\_") for m in ORDER}
NAME.update({"partformer": "PartFormer", "partformer_cos": "PartFormer-cos", "conv1d_former": "Conv1DFormer",
             "kdf_partformer": "KDF-PartFormer", "mp_transformer": "MP-Transformer", "mp_bilstm": "MP-BiLSTM",
             "hwgat": "HWGAT", "fft_bilstm": "FFT-BiLSTM", "cwt_bilstm": "CWT-BiLSTM",
             "cwt_transformer": "CWT-Transformer", "kdf_transformer": "KDF-Transformer",
             "kdf_transformer_nodmd": r"\quad $-$DMD", "kdf_transformer_nokc": r"\quad $-$Koopman head",
             "mp_transformer_reg": r"\quad $-$both (mixup+LS only)", "stgcn": "ST-GCN", "ctr_gcn": "CTR-GCN",
             "td_gcn": "TD-GCN", "pgf_slr": "PGF-SLR"})
PLAIN = dict(NAME, kdf_transformer_nodmd="KDF $-$DMD", kdf_transformer_nokc="KDF $-$Koopman head",
             mp_transformer_reg="MP-Transformer + mixup/LS")
FIG_NAME = {"partformer": "PartFormer", "partformer_cos": "PartFormer-cos", "conv1d_former": "Conv1DFormer",
            "kdf_partformer": "KDF-PartFormer", "cwt_transformer": "CWT-Transformer",
            "kdf_transformer": "KDF-Transformer", "mp_transformer": "MP-Transformer (ref.)", "hwgat": "HWGAT"}
# fixed categorical order (validated palette, slots 1-7); the reference baseline is neutral grey, dashed
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
FIG_MODELS = ["partformer", "conv1d_former", "partformer_cos", "kdf_partformer", "cwt_transformer",
              "kdf_transformer", "hwgat", "mp_transformer"]
COLOR = dict(zip(FIG_MODELS[:-1], PALETTE))
COLOR["mp_transformer"] = "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- loading
def load(sweeps: str, exclude: set) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(sweeps, "*", "*", "K*", "s*", "result.json")):
        d = json.load(open(f))
        if d["model"] in exclude:
            continue
        m = d["metrics"]
        r = dict(grid=d["tag"], model=d["model"], K=int(d["shots"]), seed=int(d["seed"]), dir=os.path.dirname(f),
                 n_train=d["n_train"], n_params=d["n_params"], steps=d["steps"], secs=d["train_seconds"],
                 modality=d["modality"], fp=d["test_fingerprint"])
        r.update({k: m.get(k) for k in ("top1", "macro_f1", "target_top1", "rich_top1", "target_to_rich_rate",
                                         "proto_top1", "proto_target_top1", "top5", "n_test", "n_test_target",
                                         "chance")})
        r["per_word"] = m.get("per_word_top1") or {}
        rows.append(r)
    return rows


def cell(rows, grid, model, K, key):
    v = [r[key] for r in rows if r["grid"] == grid and r["model"] == model and r["K"] == K
         and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    if not v:
        return None
    v = 100 * np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def models_in(rows, grid):
    have = {r["model"] for r in rows if r["grid"] == grid}
    return [m for m in ORDER if m in have]


def shots_in(rows, grid):
    return sorted({r["K"] for r in rows if r["grid"] == grid})


def preds(run_dir):
    z = np.load(os.path.join(run_dir, "preds.npz"), allow_pickle=True)
    words = [str(w) for w in z["words"]]
    tgt = np.isin(np.array(words)[z["y"]], TARGETS)
    return dict(zip(z["keys"].tolist(), zip((z["pred"] == z["y"]).tolist(), tgt.tolist())))


def mcnemar(a, b):
    nb, nc = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = nb + nc
    if n == 0:
        return nb, nc, 1.0
    p = sum(math.comb(n, i) for i in range(min(nb, nc) + 1)) / 2 ** n
    return nb, nc, min(1.0, 2 * p)


def boot(a, b, n_boot=4000, seed=0):
    d = b.astype(float) - a.astype(float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def paired(rows, grid, model, K, subset):
    a_all, b_all = [], []
    for s in (0, 1, 2):
        ra = [r for r in rows if (r["grid"], r["model"], r["K"], r["seed"]) == (grid, REF, K, s)]
        rb = [r for r in rows if (r["grid"], r["model"], r["K"], r["seed"]) == (grid, model, K, s)]
        if not ra or not rb:
            continue
        pa, pb = preds(ra[0]["dir"]), preds(rb[0]["dir"])
        for k in sorted(set(pa) & set(pb)):
            if subset == "target" and not pa[k][1]:
                continue
            a_all.append(pa[k][0])
            b_all.append(pb[k][0])
    if not a_all:
        return None
    a, b = np.array(a_all), np.array(b_all)
    return (*boot(a, b), *mcnemar(a, b), len(a))


def paired_grids(rows, model, K, ga="uniform", gb="scarce_legacy8"):
    """Same model, same target training clips, same test clips; only the rich-word pools differ."""
    a_all, b_all = [], []
    for s in (0, 1, 2):
        ra = [r for r in rows if (r["grid"], r["model"], r["K"], r["seed"]) == (ga, model, K, s)]
        rb = [r for r in rows if (r["grid"], r["model"], r["K"], r["seed"]) == (gb, model, K, s)]
        if not ra or not rb:
            continue
        pa, pb = preds(ra[0]["dir"]), preds(rb[0]["dir"])
        for k in sorted(set(pa) & set(pb)):
            if pa[k][1]:
                a_all.append(pa[k][0])
                b_all.append(pb[k][0])
    if not a_all:
        return None
    a, b = np.array(a_all), np.array(b_all)
    return (*boot(a, b), *mcnemar(a, b), len(a))


# ---------------------------------------------------------------- latex helpers
def fmt(c, bold=False, std=True):
    if c is None:
        return "--"
    s = f"{c[0]:.1f}" + (rf"\,{{\scriptsize$\pm${c[1]:.1f}}}" if std else "")
    return rf"\textbf{{{s}}}" if bold else s


def pfmt(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    return rf"${p / 10 ** e:.1f}\!\times\!10^{{{e}}}$"


def table(rows, grid, key, caption, label, models=None, shots=None, std=True, lower=False):
    models = models or models_in(rows, grid)
    shots = shots or shots_in(rows, grid)
    best = {}
    for K in shots:
        vals = [c[0] for c in (cell(rows, grid, m, K, key) for m in models) if c]
        best[K] = min(vals) if lower else max(vals)
    cols = "l" + "c" * len(shots)
    out = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           "Model & " + " & ".join(f"$K={K}$" for K in shots) + r" \\", r"\midrule"]
    prev = None
    for m in models:
        g = GROUP[m]
        if prev is not None and g != prev:
            out.append(r"\midrule")
        prev = g
        cs = [cell(rows, grid, m, K, key) for K in shots]
        out.append(NAME[m] + " & " + " & ".join(
            fmt(c, bold=c is not None and abs(c[0] - best[K]) < 1e-9, std=std) for c, K in zip(cs, shots)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


# ---------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7.5, width=0.6, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def curve(ax, rows, grid, key, models, shots):
    for m in models:
        pts = [(K, cell(rows, grid, m, K, key)) for K in shots]
        pts = [(K, c) for K, c in pts if c is not None]
        if not pts:
            continue
        x = np.array([K for K, _ in pts])
        y = np.array([c[0] for _, c in pts])
        ref = m == REF
        ax.plot(x, y, color=COLOR[m], lw=1.5, ls="--" if ref else "-", marker="o", ms=4.2,
                markeredgecolor="white", markeredgewidth=0.8, label=FIG_NAME[m], zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(shots)
    ax.set_xticklabels([str(k) for k in shots])
    ax.minorticks_off()
    ax.set_xlabel("training clips per target word ($K$)", fontsize=8, color=INK)


def figures(rows, dest: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                         "axes.labelcolor": INK, "text.color": INK})
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    present = {r["model"] for r in rows}
    fm = [m for m in FIG_MODELS if m in present]

    # Fig. 2: scarce protocol, target accuracy and absorption
    g = "scarce_legacy8"
    shots = shots_in(rows, g)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55))
    curve(axs[0], rows, g, "target_top1", fm, shots)
    axs[0].set_ylabel("target-word top-1 (%)", fontsize=8)
    axs[0].set_ylim(0, 80)
    curve(axs[1], rows, g, "target_to_rich_rate", fm, shots)
    axs[1].set_ylabel("absorbed into a rich word (%)", fontsize=8)
    axs[1].set_ylim(0, 100)
    for ax, t in zip(axs, ("(a) accuracy on the 8 scarce words", "(b) absorption into rich words")):
        style(ax)
        ax.set_title(t, fontsize=8.5, color=INK, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(dest / "scarce.pdf", bbox_inches="tight")
    fig.savefig(dest / "scarce.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    made.append("scarce")

    # Fig. 3: uniform protocol, top-1 and prototype top-1
    g = "uniform"
    if any(r["grid"] == g for r in rows):
        shots = shots_in(rows, g)
        um = [m for m in fm if cell(rows, g, m, shots[-1], "top1")]
        fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), sharey=True)
        curve(axs[0], rows, g, "top1", um, shots)
        curve(axs[1], rows, g, "proto_top1", um, shots)
        axs[0].set_ylabel("top-1 over 30 words (%)", fontsize=8)
        axs[0].set_ylim(0, 80)
        for ax, t in zip(axs, ("(a) trained classifier", "(b) nearest class mean (prototype)")):
            style(ax)
            ax.set_title(t, fontsize=8.5, color=INK, loc="left")
            ax.axhline(100 / 30, color=INK2, lw=0.6, ls=":")
        axs[0].text(shots[-1], 100 / 30 + 1.5, "chance", fontsize=7, color=INK2, ha="right")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(dest / "uniform.pdf", bbox_inches="tight")
        fig.savefig(dest / "uniform.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append("uniform")

    # Fig. 4: vocabulary size (target words fixed; K on the 8 targets)
    grids = [("scarce_legacy8_V8", 8), ("scarce_legacy8_V16", 16), ("scarce_legacy8", 30)]
    if all(any(r["grid"] == gg for r in rows) for gg, _ in grids):
        fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
        vm = [m for m in fm if all(cell(rows, gg, m, 8, "target_top1") for gg, _ in grids)]
        for ax, K in zip(axs, (2, 8)):
            for m in vm:
                y = [cell(rows, gg, m, K, "target_top1")[0] for gg, _ in grids]
                ax.plot([n for _, n in grids], y, color=COLOR[m], lw=1.5, ls="--" if m == REF else "-",
                        marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=0.8, label=FIG_NAME[m])
            style(ax)
            ax.set_xticks([8, 16, 30])
            ax.set_xlabel("vocabulary size (words)", fontsize=8)
            ax.set_title(f"({'ab'[K == 8]}) $K={K}$ clips per target word", fontsize=8.5, loc="left")
        axs[0].set_ylabel("target-word top-1 (%)", fontsize=8)
        axs[0].set_ylim(0, 100)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.savefig(dest / "vocab.pdf", bbox_inches="tight")
        fig.savefig(dest / "vocab.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append("vocab")
    return made


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="sweeps")
    ap.add_argument("--include-graph", action="store_true",
                    help="include stgcn/ctr_gcn/td_gcn/pgf_slr (only after the corrected rerun)")
    a = ap.parse_args()
    rows = load(a.sweeps, set() if a.include_graph else set(GRAPH))
    gen = HERE / "generated"
    gen.mkdir(exist_ok=True)
    T = {}
    g = "scarce_legacy8"

    T["tab_scarce_target"] = table(
        rows, g, "target_top1",
        r"Scarce protocol: top-1 accuracy (\%) on the 8 target words, which have $K$ training clips each "
        r"while the other 22 words keep all their clips. Mean $\pm$ s.d.\ over 3 seeds; "
        r"62 target test clips per seed from unseen signers. Best per column in bold.", "tab:scarce")
    T["tab_scarce_rich"] = table(
        rows, g, "rich_top1",
        r"Scarce protocol: top-1 accuracy (\%) on the 22 rich words (178 test clips per seed). "
        r"Their training data are the same for every $K$; only the target words change.", "tab:rich")
    T["tab_scarce_absorb"] = table(
        rows, g, "target_to_rich_rate",
        r"Absorption: percentage of target-word test clips predicted as one of the 22 rich words "
        r"(lower is better; best in bold).", "tab:absorb", lower=True)
    T["tab_scarce_top1"] = table(
        rows, g, "top1", r"Scarce protocol: overall top-1 accuracy (\%) over all 30 words (240 test clips per seed).",
        "tab:scarce_all")
    T["tab_uniform"] = table(
        rows, "uniform", "top1",
        r"Uniform protocol: top-1 accuracy (\%) over 30 words when every word has $K$ training clips "
        r"(same 240 test clips as the scarce protocol). Chance is 3.3\,\%.", "tab:uniform")
    T["tab_uniform_top5"] = table(rows, "uniform", "top5", r"Uniform protocol: top-5 accuracy (\%).",
                                  "tab:uniform5")

    # vocabulary table: target top-1 at K=2 and 8 for V=8/16/30
    grids = [("scarce_legacy8_V8", 8), ("scarce_legacy8_V16", 16), ("scarce_legacy8", 30)]
    vm = models_in(rows, "scarce_legacy8_V8")
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Vocabulary size: top-1 accuracy (\%) on the 8 target words at $K\in\{2,8\}$ when they are "
           r"embedded in 8 (targets only), 16 or 30 words. The target test clips are the same in every column.}",
           r"\label{tab:vocab}", r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{$K=2$} & \multicolumn{3}{c}{$K=8$} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Model & $V{=}8$ & $V{=}16$ & $V{=}30$ & $V{=}8$ & $V{=}16$ & $V{=}30$ \\", r"\midrule"]
    best = {(gg, K): max((cell(rows, gg, m, K, "target_top1") or (-1,))[0] for m in vm) for gg, _ in grids for K in (2, 8)}
    for m in vm:
        cs = []
        for K in (2, 8):
            for gg, _ in grids:
                c = cell(rows, gg, m, K, "target_top1")
                cs.append(fmt(c, bold=c is not None and abs(c[0] - best[(gg, K)]) < 1e-9, std=False))
        out.append(PLAIN[m] + " & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_vocab"] = "\n".join(out)

    # paired tests vs the reference, scarce protocol, target clips and all clips
    shots = shots_in(rows, g)
    pm = [m for m in ("partformer", "partformer_cos", "conv1d_former", "kdf_partformer", "cwt_transformer",
                      "kdf_transformer", "mp_transformer_reg", "hwgat") if m in models_in(rows, g)]
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Paired comparison with MP-Transformer on the scarce protocol, pooled over 3 seeds "
           r"(same test clips). $\Delta$: accuracy difference in points with a 95\,\% paired-bootstrap CI; "
           r"$p$: exact two-sided McNemar test. Target clips: $n=186$; all clips: $n=720$.}",
           r"\label{tab:paired}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{target words} & \multicolumn{2}{c}{all words} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"$K$ & Model & $\Delta$ [95\,\% CI] & $p$ & $\Delta$ [95\,\% CI] & $p$ \\", r"\midrule"]
    csv = ["K,model,subset,delta,lo,hi,b,c,p,n"]
    for i, K in enumerate(shots):
        if i:
            out.append(r"\midrule")
        for m in pm:
            t, a = paired(rows, g, m, K, "target"), paired(rows, g, m, K, "all")
            if t is None:
                continue
            for sub, v in (("target", t), ("all", a)):
                csv.append(f"{K},{m},{sub}," + ",".join(f"{x:.6g}" for x in v))
            f = lambda v: rf"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"
            out.append(f"{K} & {PLAIN[m]} & {f(t)} & {pfmt(t[5])} & {f(a)} & {pfmt(a[5])}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_paired"] = "\n".join(out)
    (gen / "paired.csv").write_text("\n".join(csv) + "\n")

    # KDF ablation (scarce target top-1, all K) + PartFormer pair
    km = [m for m in ("mp_transformer", "mp_transformer_reg", "kdf_transformer_nokc", "kdf_transformer_nodmd",
                      "kdf_transformer", "partformer", "kdf_partformer") if m in models_in(rows, g)]
    comp = {"mp_transformer": "--", "mp_transformer_reg": "mixup+LS", "kdf_transformer_nokc": "mixup+LS, DMD",
            "kdf_transformer_nodmd": "mixup+LS, KH", "kdf_transformer": "mixup+LS, DMD, KH",
            "partformer": "part front end, mixup+LS", "kdf_partformer": "part front end, mixup+LS, DMD, KH"}
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Koopman/DMD ablation on the scarce protocol: target-word top-1 (\%). "
           r"DMD: Hankel-DMD spectrum and mode branch; KH: class-wise Koopman head. The first five rows share "
           r"the landmark Transformer backbone.}", r"\label{tab:kdf}",
           r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l>{\raggedright\arraybackslash}p{2.9cm}" + "c" * len(shots) + "}", r"\toprule",
           "Model & Components & " + " & ".join(f"$K={K}$" for K in shots) + r" \\", r"\midrule"]
    for m in km:
        if m == "partformer":
            out.append(r"\midrule")
        out.append(f"{PLAIN[m]} & {comp[m]} & " + " & ".join(
            fmt(cell(rows, g, m, K, "target_top1")) for K in shots) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_kdf"] = "\n".join(out)

    # prototype vs trained head (uniform, all 30 words)
    gu = "uniform"
    us = shots_in(rows, gu)
    um = models_in(rows, gu)
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Trained linear head versus a nearest-class-mean (prototype) classifier built from the same "
           r"embeddings of the training clips, uniform protocol, top-1 (\%) over 30 words.}", r"\label{tab:proto}",
           r"\begin{tabular}{l" + "cc" * len(us) + "}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$K={K}$}}" for K in us) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(us))),
           "Model & " + " & ".join(["head & proto"] * len(us)) + r" \\", r"\midrule"]
    for m in um:
        cs = []
        for K in us:
            h, p = cell(rows, gu, m, K, "top1"), cell(rows, gu, m, K, "proto_top1")
            win = h is not None and p is not None and p[0] > h[0]
            cs += [fmt(h, std=False), fmt(p, bold=win, std=False)]
        out.append(PLAIN[m] + " & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_proto"] = "\n".join(out)

    # uniform vs scarce on the target words
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Effect of the rich words' data on the target words. Target-word top-1 (\%) when every word has "
           r"$K$ clips (uniform) $\to$ when the 22 other words keep their full pools (scarce). The target words' "
           r"training clips and all test clips are identical in the two settings. In parentheses: scarce $-$ uniform "
           r"in points, pooled over 3 seeds ($n=186$ target clips); $^{*}$: exact McNemar $p<0.05$.}",
           r"\label{tab:univsscarce}", r"\begin{tabular}{lcccc}", r"\toprule",
           r"Model & " + " & ".join(f"$K={K}$" for K in us) + r" \\", r"\midrule"]
    for m in um:
        cs = []
        for K in us:
            u, c, pg = cell(rows, gu, m, K, "target_top1"), cell(rows, g, m, K, "target_top1"), paired_grids(rows, m, K)
            if u is None or c is None or pg is None:
                cs.append("--")
                continue
            star = "$^{*}$" if pg[5] < 0.05 else ""
            cs.append(rf"{u[0]:.0f}$\to${c[0]:.0f} ({100 * pg[0]:+.0f}{star})")
        out.append(PLAIN[m] + " & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_uni_vs_scarce"] = "\n".join(out)

    # model inventory
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Models. Parameters for 30 classes; training time is the mean wall-clock time of a "
           r"scarce $K{=}8$ run (1\,815 steps, batch 16) on one laptop RTX~2050 GPU.}", r"\label{tab:models}",
           r"\begin{tabular}{lllrr}", r"\toprule",
           r"Model & Family & Input & Params & Train (s) \\", r"\midrule"]
    inp = {"landmarks": "pose+hands (75 pts)", "skeleton": "27-joint skeleton", "spectral_fft": "FFT + kinematics",
           "spectral_cwt": "CWT bands", "skeleton_kdf": "pose+hands + DMD", "parts": "parts (119 pts)",
           "parts_kdf": "parts + DMD"}
    inp_m = {"kdf_transformer_nodmd": "pose+hands (75 pts)", "mp_transformer_reg": "pose+hands (75 pts)"}
    prev = None
    for m in models_in(rows, g):
        rr = [r for r in rows if r["grid"] == g and r["model"] == m and r["K"] == 8]
        if not rr:
            continue
        if prev and GROUP[m] != prev:
            out.append(r"\midrule")
        prev = GROUP[m]
        out.append(f"{PLAIN[m]} & {GROUP[m]} & {inp_m.get(m, inp.get(rr[0]['modality'], rr[0]['modality']))} & "
                   f"{rr[0]['n_params'] / 1e6:.2f}\\,M & {np.mean([r['secs'] for r in rr]):.0f}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_models"] = "\n".join(out)

    # per target word, K=1 and K=8, two models
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Per-word top-1 (\%) on the 8 target words, scarce protocol, mean over 3 seeds.}",
           r"\label{tab:perword}", r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{MP-Transformer} & \multicolumn{2}{c}{PartFormer} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Word & $K=1$ & $K=8$ & $K=1$ & $K=8$ \\", r"\midrule"]
    for w in TARGETS:
        cs = []
        for m in (REF, "partformer"):
            for K in (1, 8):
                v = [r["per_word"].get(w) for r in rows if (r["grid"], r["model"], r["K"]) == (g, m, K)]
                v = [x for x in v if x is not None]
                cs.append(f"{100 * np.mean(v):.0f}" if v else "--")
        out.append(f"{w} & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    T["tab_perword"] = "\n".join(out)

    for k, v in T.items():
        (gen / f"{k}.tex").write_text(v + "\n")

    # numbers quoted in the text
    def n(model, K, key, grid=g):
        c = cell(rows, grid, model, K, key)
        return f"{c[0]:.1f}" if c else "--"
    num = {}
    for m, tag in (("partformer", "PF"), ("conv1d_former", "CF"), ("mp_transformer", "MP"), ("hwgat", "HW"),
                   ("kdf_transformer", "KDF"), ("kdf_partformer", "KPF"), ("partformer_cos", "PFC")):
        for K in shots:
            num[f"{tag}tgt{'ABCDE'[shots.index(K)]}"] = n(m, K, "target_top1")
            num[f"{tag}rich{'ABCDE'[shots.index(K)]}"] = n(m, K, "rich_top1")
            num[f"{tag}abs{'ABCDE'[shots.index(K)]}"] = n(m, K, "target_to_rich_rate")
        for K in us:
            num[f"{tag}uni{'ABCDE'[us.index(K)]}"] = n(m, K, "top1", "uniform")
    pt = paired(rows, g, "partformer", 1, "target")
    num["PFdKone"] = f"{100 * pt[0]:+.1f}"
    num["PFdKoneLo"] = f"{100 * pt[1]:+.1f}"
    num["PFdKoneHi"] = f"{100 * pt[2]:+.1f}"
    fps = defaultdict(set)
    for r in rows:
        if r["grid"] in ("scarce_legacy8", "uniform"):
            fps[r["seed"]].add(r["fp"])
    num["NRuns"] = str(len(rows))
    (gen / "numbers.tex").write_text("".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in num.items()))
    print("runs:", len(rows), "| test fingerprints per seed (scarce+uniform):", {s: len(v) for s, v in fps.items()})
    print("tables:", ", ".join(T), "| figures:", ", ".join(figures(rows, HERE / "figures")))


if __name__ == "__main__":
    main()

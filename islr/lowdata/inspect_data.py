"""Describe landmark stores before spending GPU time on them.

    python lowdata.py inspect --stores $STORES/isl40 $STORES/include [--grid configs/scarce_legacy8.json]

Prints (and writes to --dest as markdown if given):
  * clips / words / identities per source, clips-per-word distribution, licences
  * detection rates from a sample of clips: frames with pose, left hand, right hand, face
  * protocol feasibility per seed: test size, words without test clips, target words
    with fewer pool clips than K (for the protocol of --grid, or scarce/legacy8)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    d = df.reset_index() if index else df
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def source_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("source")
    t = pd.DataFrame({
        "clips": g.size(),
        "words": g["word"].nunique(),
        "identities": g["identity"].nunique(),
        "signer ids": g["signer"].apply(lambda s: int((s.astype(str).str.len() > 0).sum())),
        "median frames": g["n_frames"].median(),
        "licence": g["license"].agg(lambda s: "; ".join(sorted(set(map(str, s)))[:2])),
    })
    t.loc["ALL"] = [len(df), df["word"].nunique(), df["identity"].nunique(), "", df["n_frames"].median(), ""]
    return t


def detection_rates(df: pd.DataFrame, per_source: int = 40, seed: int = 0) -> pd.DataFrame:
    from islr.lowdata.store import FACE0, LH0, RH0

    rows = []
    for src, g in df.groupby("source"):
        g = g.sample(n=min(per_source, len(g)), random_state=seed)
        acc = np.zeros(4)
        n = 0
        for p in g["abs_path"]:
            a = np.load(p, mmap_mode="r")
            ok = np.isfinite(np.asarray(a[..., 0], dtype=np.float32))
            acc += [ok[:, :LH0].any(1).mean(), ok[:, LH0:RH0].any(1).mean(), ok[:, RH0:FACE0].any(1).mean(),
                    ok[:, FACE0:].any(1).mean()]
            n += 1
        rows.append([src, n, *(100 * acc / max(n, 1))])
    return pd.DataFrame(rows, columns=["source", "clips sampled", "pose %", "left hand %", "right hand %", "face %"])


def feasibility(df: pd.DataFrame, base: dict, shots: list, seeds: list) -> pd.DataFrame:
    from islr.lowdata.protocol import make_split, split_config_from

    rows = []
    for seed in seeds:
        for k in shots:
            s = make_split(df, split_config_from(dict(base, shots=k, seed=seed)))
            i = s.info
            rows.append({"seed": seed, "K": k, "words": i["n_words"], "targets": len(i["targets"]),
                         "train": i["n_train"], "test": i["n_test"], "test (targets)": i["n_test_target"],
                         "test ids": i["n_test_identities"], "words w/o test": len(i["words_without_test"]),
                         "targets short of K": len(i["shots_short"]),
                         "missing targets": ",".join(i["missing_targets"])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import json

    p = argparse.ArgumentParser(prog="lowdata.py inspect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stores", nargs="+", required=True)
    p.add_argument("--grid", default=None, help="take protocol settings, K and seeds from a sweep grid")
    p.add_argument("--sample", type=int, default=40, help="clips per source for detection rates")
    p.add_argument("--dest", default=None, help="write inspect.md here")
    a = p.parse_args(argv)
    from islr.lowdata.store import load_stores

    df = load_stores(a.stores)
    base, shots, seeds = {"protocol": "scarce", "targets": "legacy8"}, [1, 2, 4, 8, 16], [0, 1, 2]
    if a.grid:
        from islr.lowdata.sweep import load_grid

        g = load_grid(a.grid)
        base = {k: v for k, v in g["base"].items() if k != "ddp"}
        base.update(g["variants"][0])
        shots = g["grid"].get("shots", shots)
        seeds = g["grid"].get("seed", seeds)
    per_word = df.groupby("word").size()
    parts = [
        "# Store inspection\n",
        "## Sources\n", md_table(source_table(df)), "",
        f"clips per word: min {per_word.min()}, median {per_word.median():.0f}, max {per_word.max()}; "
        f"words with >= 3 clips: {(per_word >= 3).sum()} of {len(per_word)}\n",
        "## Detection rates (share of frames with the part detected)\n",
        md_table(detection_rates(df, a.sample), index=False), "",
        f"## Protocol feasibility ({json.dumps(base)})\n",
        md_table(feasibility(df, base, shots, seeds), index=False), "",
    ]
    md = "\n".join(parts)
    print(md)
    if a.dest:
        Path(a.dest).mkdir(parents=True, exist_ok=True)
        (Path(a.dest) / "inspect.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())

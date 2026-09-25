"""Get the landmark stores a grid needs, for one team member, without waiting for anyone.

    python lowdata.py data --grid islr/lowdata/configs/isl40_scarce.json --dest stores \\
        --member 0 --members 4 --inputs /kaggle/input --time-budget-h 11

For every store the grid lists (`$STORES/<name>`), in this order:
  1. `<dest>/<name>` is already complete (COMPLETE.json)            -> nothing to do
  2. a complete copy sits under --inputs (a teammate's notebook output, a shared
     "stores" dataset, or this notebook's previous version)          -> copy it
  3. otherwise build it: partial copies under --inputs are merged first (INCLUDE zips
     they finished are skipped), then the source is ingested.
Nothing here depends on other members: attaching their outputs only saves time.

A grid can pin the data version, e.g. "data": {"isl40": {"revision": "<HF commit>"}}, and
say what to build, e.g. "data": {"include_words": {"words": {"House": "home"}}}; a stored
copy is reused only when its revision/words match.
Complete stores record their revision and a fingerprint of their clip keys; two members
with the same fingerprint get identical splits. The fingerprints are written to
<dest>/_status.json, and the report compares them across members.

Exit code 0 when every store is complete, 3 when time ran out (re-run to continue).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .store import completion, init_store, merge_stores, store_fingerprint

INCOMPLETE = 3
BUILDERS = ("isl40", "isl30", "include", "include_words")
MATCH_KEYS = ("repo", "revision", "words")  # a stored copy is reused only if these equal the grid's


def grid_stores(grid: dict) -> list[str]:
    return [Path(str(s).replace("\\", "/")).name for s in grid.get("stores", [])]


def find_candidates(name: str, roots: list[str], exclude: Path | None = None, max_depth: int = 5) -> list[Path]:
    """Store dirs called <name> (or <name>_shard<i>) anywhere under `roots`."""
    out = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            if len(d.parts) - base >= max_depth:
                dirnames[:] = []
            dirnames[:] = [x for x in dirnames if x not in ("npy", "_download")]
            if "store.json" in filenames and (d.name == name or d.name.startswith(name + "_shard")):
                if exclude is None or d.resolve() != exclude.resolve():
                    out.append(d)
    return sorted(set(out))


def _matches(info: dict | None, opts: dict) -> bool:
    return bool(info) and all(json.dumps(info.get(k), sort_keys=True) == json.dumps(opts[k], sort_keys=True)
                              for k in MATCH_KEYS if opts.get(k) is not None)


def build(name: str, store: Path, opts: dict, left_h: float | None, member: int, members: int,
          workers: int | None, work: str | None) -> bool:
    from . import sources

    if name in ("isl40", "isl30"):
        return sources.ingest_isl40(str(store), opts.get("root"), workers, None, left_h, opts.get("revision"),
                                    work, opts.get("repo", sources.HF_ISL30 if name == "isl30" else sources.HF_ISL40))
    if name == "include":
        return sources.ingest_include(str(store), work=work, zip_dir=opts.get("zip_dir"), workers=workers,
                                      time_budget_h=left_h, member=member, members=members)
    if name == "include_words":
        return sources.ingest_include_words(str(store), opts["words"], opts.get("categories"), work, workers,
                                            left_h)
    raise SystemExit(f"don't know how to build store {name!r} (known: {', '.join(BUILDERS)}); "
                     f"attach a complete copy under --inputs instead")


def prepare(grid: dict, dest: str | Path, member: int = 0, members: int = 1, inputs: list[str] = (),
            time_budget_h: float | None = None, workers: int | None = None, work: str | None = None) -> dict:
    t0 = time.time()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data_cfg = grid.get("data", {})
    status = {}
    for name in grid_stores(grid):
        opts = dict(data_cfg.get(name, {}))
        store = dest / name
        info = completion(store)
        how = "already here"
        if not _matches(info, opts):
            cands = find_candidates(name, list(inputs), exclude=store)
            full = [c for c in cands if _matches(completion(c), opts)]
            if full:
                if store.exists():
                    shutil.rmtree(store)
                shutil.copytree(full[0], store, ignore=shutil.ignore_patterns("_download"))
                how = f"copied from {full[0]}"
            else:
                # partial copies only help INCLUDE (zips they finished are skipped); the 40-word
                # corpus takes minutes, and a copy from another revision must not leak in
                partial = [c for c in cands if name == "include"]
                if partial:
                    merge_stores([str(c) for c in partial], str(store))
                    print(f"[data] {name}: merged {len(partial)} partial copies: {[str(c) for c in partial]}",
                          flush=True)
                left = time_budget_h - (time.time() - t0) / 3600 if time_budget_h else None
                if left is not None and left <= 0.05:
                    how = "not started (no time left)"
                else:
                    init_store(store)
                    build(name, store, opts, left, member, members, workers, work)
                    how = "built here"
            info = completion(store)
        fp = store_fingerprint(store)
        ok = _matches(info, opts) and fp["n_clips"] > 0  # an empty store is never a finished one
        if _matches(info, opts) and not ok:
            print(f"[data] {name}: store has 0 clips (extraction failed? see {store}/failed.txt)", flush=True)
        status[name] = {"complete": ok, "how": how, "revision": (info or {}).get("revision"), **fp}
        print(f"[data] {name}: {'COMPLETE' if ok else 'INCOMPLETE'} ({how}) - {fp['n_clips']} clips, "
              f"{fp['n_words']} words, keys {fp['keys_sha1']}", flush=True)
    (dest / "_status.json").write_text(json.dumps(status, indent=1))
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs="+", required=True, help="grid file(s); their stores are unioned")
    ap.add_argument("--dest", required=True, help="where the stores go (= $STORES for the sweep)")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--inputs", nargs="*", default=[], help="folders to search for existing stores")
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--work", default=None, help="scratch dir for downloads")
    a = ap.parse_args(argv)

    from .sweep import load_grid

    merged: dict = {"stores": [], "data": {}}
    for g in a.grid:
        grid = load_grid(g)
        for s in grid.get("stores", []):
            if Path(str(s)).name not in grid_stores(merged):
                merged["stores"].append(s)
        for k, v in grid.get("data", {}).items():
            if k in merged["data"] and merged["data"][k] != v:
                raise SystemExit(f"grids disagree on the data version of {k}: {merged['data'][k]} vs {v}")
            merged["data"][k] = v
    status = prepare(merged, a.dest, a.member, a.members, a.inputs, a.time_budget_h, a.workers, a.work)
    return 0 if all(s["complete"] for s in status.values()) else INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())

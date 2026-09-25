"""Few-shot SLR split: locked test set, k-shot train draws, cross-signer/session."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EIGHT_WORDS = ("eat", "go", "hello", "help", "no", "please", "water", "yes")  # legacy smoke set
_USER = re.compile(r"^User\d+", re.I)
_SESSION = re.compile(r"session(\d+)", re.I)
_TOP = re.compile(r"^top(\d+)?$", re.I)
_MVI = re.compile(r"MVI_(\d+)", re.I)
INCLUDE_SESSION_GAP = 200


def _token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def clip_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("word").size().sort_values(ascending=False)


def top_words(df: pd.DataFrame, k: int = 8) -> list[str]:
    """Highest-count glosses in `df`, ties broken alphabetically."""
    counts = df.groupby("word").size()
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [str(w) for w, _ in ranked[: max(0, int(k))]]


def align_words(requested: list[str], df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Map CLI tokens (hello, thank_you) onto metadata `word` labels. Returns (found, missing)."""
    alias: dict[str, str] = {}
    for w in df["word"].astype(str).unique():
        alias[_token(w)] = w
        alias[w.lower()] = w
        alias[w.lower().replace(" ", "_")] = w
    if "normalized_word" in df.columns:
        for orig, norm in zip(df["word"].astype(str), df["normalized_word"].astype(str)):
            alias[_token(norm)] = orig
    found: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in requested:
        key = _token(raw)
        hit = alias.get(key) or alias.get(raw.strip().lower())
        if hit is None:
            missing.append(raw)
            continue
        if hit not in seen:
            found.append(hit)
            seen.add(hit)
    return found, missing


def resolve_words(
    raw: list[str] | None,
    df: pd.DataFrame | None = None,
    n_words: int = 8,
) -> list[str] | None:
    """Choose glosses. Default / `top` / `top8`: the n_words classes with the most clips.

    `all` → every class (return None). `legacy8` → original eat/go/hello/… set.
    None return means 'do not filter'.
    """
    if df is None:
        if not raw:
            return list(EIGHT_WORDS)
        if len(raw) == 1 and raw[0].lower() in {"all", "*"}:
            return None
        if len(raw) == 1 and raw[0].lower() in {"legacy8", "eight", "old8"}:
            return list(EIGHT_WORDS)
        return [w.strip().lower() for w in raw]

    if not raw:
        return top_words(df, n_words)

    token0 = str(raw[0]).strip().lower()
    if len(raw) == 1 and token0 in {"all", "*"}:
        return None
    if len(raw) == 1 and token0 in {"legacy8", "eight", "old8"}:
        found, missing = align_words(list(EIGHT_WORDS), df)
        if missing:
            print(f"WARNING: legacy8 words missing from metadata: {missing}")
        return found
    if len(raw) == 1 and _TOP.match(token0):
        m = _TOP.match(token0)
        k = int(m.group(1) or n_words)
        return top_words(df, k)
    if len(raw) == 1 and token0 in {"top"}:
        return top_words(df, n_words)

    found, missing = align_words(raw, df)
    if missing:
        print(f"WARNING: words not in metadata: {missing}")
    return found


def print_word_choice(df: pd.DataFrame, words: list[str] | None) -> None:
    counts = df.groupby("word").size()
    chosen = list(words) if words is not None else sorted(counts.index.astype(str).tolist())
    print(f"selected {len(chosen)} word(s) (by clip count):")
    print(f"{'word':16s} {'n':>5s}")
    for w in chosen:
        print(f"{w:16s} {int(counts.get(w, 0)):5d}")


def identity_key(row: pd.Series) -> str:
    """Group clips so the same signer (or session) cannot leak into train and test.

    ISL500 `User00x` is a real signer id. Session tags in the filename are the
    fallback. CISLR/INCLUDE `signer` fields are not person ids — those clips
    stay as singleton units.
    """
    signer = str(row.get("signer") or "").strip()
    if _USER.match(signer):
        return f"signer:{signer}"
    blob = f"{row.get('video_path', '')} {row.get('original_filename', '')}"
    m = _SESSION.search(blob)
    if m:
        return f"session:{m.group(1)}"
    return f"clip:{row.get('video_path')}"


def assign_identities(df: pd.DataFrame, gap: int = INCLUDE_SESSION_GAP) -> pd.Series:
    """`identity_key` per row, except INCLUDE clips: they carry no signer id, but their
    MVI_xxxx camera counters run consecutively within a recording session. A new
    session starts at every jump > `gap` (merging two sessions is only coarser, never
    leaky). Previously every INCLUDE clip was its own identity, so one signer's
    session could land in both train and test."""
    if df.empty:
        return pd.Series([], index=df.index, dtype=str)
    ids = df.apply(identity_key, axis=1).astype(str)
    empty = pd.Series("", index=df.index)
    blob = df.get("video_path", empty).astype(str) + " " + df.get("original_filename", empty).astype(str)
    is_inc = df.get("dataset", empty).astype(str).str.upper().eq("INCLUDE") & ids.str.startswith("clip:")
    nums = pd.to_numeric(blob.str.extract(_MVI, expand=False), errors="coerce")
    sel = is_inc & nums.notna()
    if sel.any():
        order = np.sort(nums[sel].unique())
        starts = order[np.r_[True, np.diff(order) > gap]]
        ids.loc[sel] = [f"session:include_s{int(starts[np.searchsorted(starts, n, side='right') - 1])}"
                        for n in nums[sel]]
    return ids


def _coverage(work: pd.DataFrame, ids: set[str]) -> dict[str, int]:
    if not ids:
        return {}
    sub = work[work["_id"].isin(ids)]
    return sub.groupby("word").size().astype(int).to_dict()


def _lock_identities(
    work: pd.DataFrame,
    words: list[str],
    train_shots: int,
    val_shots: int,
    test_per_class: int,
    proto_rng: np.random.Generator,
) -> tuple[set[str], set[str], list[str]]:
    """Partition identities once so a signer/session cannot be train on one word and test on another."""
    warnings: list[str] = []
    identities = work["_id"].drop_duplicates().tolist()
    order = np.arange(len(identities))
    proto_rng.shuffle(order)
    shuffled = [identities[int(i)] for i in order]

    need_rest = {}
    for w in words:
        n = int((work["word"] == w).sum())
        need_rest[w] = min(int(train_shots) + int(val_shots), max(0, n - 1))

    test_ids: list[str] = []
    rest_ids = list(shuffled)
    for uid in shuffled:
        test_cov = _coverage(work, set(test_ids))
        if all(int(test_cov.get(w, 0)) >= test_per_class for w in words):
            break
        new_rest = [x for x in rest_ids if x != uid]
        rest_cov = _coverage(work, set(new_rest))
        if all(int(rest_cov.get(w, 0)) >= need_rest[w] for w in words):
            test_ids.append(uid)
            rest_ids = new_rest

    if not test_ids:
        warnings.append(
            "could not hold out any identity-disjoint test signer/session; "
            "falling back to per-class clip holdout"
        )
    return set(test_ids), set(rest_ids), warnings


def _draw_from_indices(
    idxs: list[int],
    train_shots: int,
    val_shots: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    idxs = list(idxs)
    rng.shuffle(idxs)
    n_val = min(val_shots, max(0, len(idxs) - 1)) if idxs else 0
    n_tr = min(train_shots, max(0, len(idxs) - n_val))
    if n_tr == 0 and idxs:
        n_tr = 1
        n_val = max(0, len(idxs) - 1)
    tr = idxs[:n_tr]
    va = idxs[n_tr : n_tr + n_val]
    if not va and len(tr) >= 2:
        va = [tr[-1]]
        tr = tr[:-1]
    return tr, va


def fewshot_protocol_split(
    df: pd.DataFrame,
    train_shots: int = 7,
    test_per_class: int = 15,
    val_shots: int = 1,
    protocol_seed: int = 42,
    draw_seed: int = 42,
    words: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Lock a balanced test set, then draw k-shot train (+ val) from the rest.

    Identities (signer, else session) are partitioned **globally** so the same
    person cannot be train on one gloss and test on another. Test assignment
    uses `protocol_seed`; train resampling uses `draw_seed`. If the pool is too
    small for 15 test clips / class, take as many as remain after reserving
    train+val — do not silently reuse signers.
    """
    work = df.copy()
    if words:
        work = work[work["word"].isin(list(words))].copy()
    if work.empty:
        raise ValueError("no rows left after word filter")
    work["_id"] = assign_identities(work)
    word_list = sorted(work["word"].astype(str).unique().tolist())
    proto_rng = np.random.default_rng(protocol_seed)
    draw_rng = np.random.default_rng(draw_seed)

    test_ids, rest_ids, warnings = _lock_identities(
        work, word_list, train_shots, val_shots, test_per_class, proto_rng
    )

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    per_class: dict[str, Any] = {}

    for word, g in work.groupby("word", sort=True):
        g_test = g[g["_id"].isin(test_ids)]
        g_rest = g[g["_id"].isin(rest_ids)]
        te = [int(i) for i in g_test.index.tolist()]
        rest = [int(i) for i in g_rest.index.tolist()]
        if len(te) > test_per_class:
            proto_rng.shuffle(te)
            te = te[:test_per_class]
        if not te and rest:
            n_te = max(1, int(len(g)) - min(train_shots + val_shots, max(0, int(len(g)) - 1)))
            proto_rng.shuffle(rest)
            te = rest[:n_te]
            rest = rest[n_te:]
            warnings.append(
                f"{word}: no identity-disjoint test clips; clip-level holdout of {len(te)} "
                f"(same-signer leakage)"
            )
        if len(te) < test_per_class:
            warnings.append(
                f"{word}: {len(g)} clips, test={len(te)} (wanted {test_per_class}); "
                f"pool too small for {test_per_class} held-out videos/class"
            )
        tr, va = _draw_from_indices(rest, train_shots, val_shots, draw_rng)
        if len(tr) < train_shots or len(va) < val_shots:
            warnings.append(
                f"{word}: train={len(tr)} val={len(va)} (wanted train={train_shots} val={val_shots}) "
                f"after locking test={len(te)}"
            )
        train_idx.extend(tr)
        val_idx.extend(va)
        test_idx.extend(te)
        per_class[str(word)] = {
            "n_pool": int(len(g)),
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "train_ids": sorted({str(work.loc[i, "_id"]) for i in tr}),
            "val_ids": sorted({str(work.loc[i, "_id"]) for i in va}),
            "test_ids": sorted({str(work.loc[i, "_id"]) for i in te}),
        }

    if not val_idx and train_idx:
        val_idx = [train_idx[-1]]
        train_idx = train_idx[:-1]
        warnings.append("global val split was empty — stole 1 clip from train")
    if not train_idx:
        raise ValueError("few-shot split produced an empty train set")
    if not test_idx:
        raise ValueError("few-shot split produced an empty test set")

    drop = ["_id"]
    train_df = work.loc[train_idx].drop(columns=drop).reset_index(drop=True)
    val_df = work.loc[val_idx].drop(columns=drop).reset_index(drop=True)
    test_df = work.loc[test_idx].drop(columns=drop).reset_index(drop=True)

    def _ids(idx: list[int]) -> set[str]:
        return set(work.loc[idx, "_id"].astype(str).tolist())

    leak_tv = sorted(_ids(train_idx) & _ids(test_idx))
    leak_vv = sorted(_ids(val_idx) & _ids(test_idx))
    if leak_tv:
        warnings.append(f"SIGNER/SESSION LEAKAGE train∩test: {leak_tv}")
    if leak_vv:
        warnings.append(f"SIGNER/SESSION LEAKAGE val∩test: {leak_vv}")

    audit = {
        "train_shots": int(train_shots),
        "val_shots": int(val_shots),
        "test_per_class": int(test_per_class),
        "protocol_seed": int(protocol_seed),
        "draw_seed": int(draw_seed),
        "words": sorted(work["word"].unique().tolist()),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "per_class": per_class,
        "warnings": warnings,
        "leakage": {"train_test": leak_tv, "val_test": leak_vv},
        "train_paths": train_df["video_path"].astype(str).tolist() if "video_path" in train_df.columns else [],
        "val_paths": val_df["video_path"].astype(str).tolist() if "video_path" in val_df.columns else [],
        "test_paths": test_df["video_path"].astype(str).tolist() if "video_path" in test_df.columns else [],
    }
    return train_df, val_df, test_df, audit


def frame_from_paths(df: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    """Rebuild a split in locked-path order. Matches relative video_path or abs_path."""
    by: dict[str, int] = {}
    for i, row in df.iterrows():
        by[str(row.get("video_path", ""))] = int(i)
        by[str(row.get("abs_path", ""))] = int(i)
    idx = [by[p] for p in paths if p in by]
    missing = [p for p in paths if p not in by]
    if missing:
        print(f"WARNING: {len(missing)} locked paths not in metadata (data-dir mismatch?)")
    return df.loc[idx].reset_index(drop=True)


def protocol_meets_spec(audit: dict[str, Any], test_per_class: int = 15) -> bool:
    per = audit.get("per_class") or {}
    if not per:
        return False
    return all(int(row.get("n_test") or 0) >= test_per_class for row in per.values())


def print_split_audit(audit: dict[str, Any]) -> None:
    print(
        f"few-shot split  train_shots={audit['train_shots']} val_shots={audit['val_shots']} "
        f"test/class={audit['test_per_class']}  "
        f"sizes train={audit['split_sizes']['train']} val={audit['split_sizes']['val']} "
        f"test={audit['split_sizes']['test']}"
    )
    print(f"{'word':16s} {'pool':>4s} {'train':>5s} {'val':>4s} {'test':>4s}")
    for word, row in audit["per_class"].items():
        print(
            f"{word:16s} {row['n_pool']:4d} {row['n_train']:5d} {row['n_val']:4d} {row['n_test']:4d}"
        )
    for w in audit.get("warnings") or []:
        print(f"WARNING: {w}")
    if audit.get("leakage", {}).get("train_test") or audit.get("leakage", {}).get("val_test"):
        print("WARNING: identity leakage between train/val and test — accuracy will be inflated")


def save_protocol(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_protocol(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

"""Landmark store: one directory per data source, raw MediaPipe Holistic output.

    <store>/index.csv      one row per clip (INDEX_COLUMNS)
    <store>/npy/<key>.npy  (T, 543, 3) float16, NaN = landmark not detected
    <store>/store.json     layout tag

Row order is the order this codebase already uses (`common.landmarks`):

    pose 0..32 | left_hand 33..53 | right_hand 54..74 | face 75..542

This is NOT the Kaggle ASL-Signs (GISLR) order, which is face | left_hand | pose |
right_hand. `from_gislr` / `to_gislr` convert; never mix arrays without them.

Frames are stored at the extraction frame rate (not resampled). `to_codebase_seq`
turns a stored clip into exactly the (T, 1629) normalised vector the existing models
were trained on (uniform frame sampling + `common.landmarks.normalize_landmarks`).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

N_POSE, N_HAND, N_FACE = 33, 21, 468
N_LM = N_POSE + 2 * N_HAND + N_FACE  # 543
POSE0, LH0, RH0, FACE0 = 0, N_POSE, N_POSE + N_HAND, N_POSE + 2 * N_HAND
LAYOUT = "holistic_pose_lh_rh_face_v1"

# GISLR (Kaggle asl-signs) row offsets
G_FACE0, G_LH0, G_POSE0, G_RH0 = 0, 468, 489, 522

INDEX_COLUMNS = [
    "key", "word", "source", "signer", "session", "split", "n_frames", "fps",
    "path", "video_rel", "license",
]

# Face-mesh subset (lips + nose) kept for the part-based models. Same indices as the
# ASL-Signs winning solutions; lets non-manual mouth shapes reach the model cheaply.
LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324,
]
NOSE = [1, 2, 98, 327]
FACE_KEEP = LIPS + NOSE  # 44 points
# Reduced layout used in memory: pose | lh | rh | face subset (119 points). The first 75
# points are identical to the full layout, so `islr.models.skeleton` helpers work on it.
REDUCED_ROWS = np.array(
    list(range(POSE0, RH0 + N_HAND)) + [FACE0 + i for i in FACE_KEEP], dtype=np.int64
)
N_REDUCED = len(REDUCED_ROWS)


# ---------------------------------------------------------------------------------
# layout conversions
# ---------------------------------------------------------------------------------
def from_gislr(arr: np.ndarray) -> np.ndarray:
    """(T, 543, 3) GISLR order (face, lh, pose, rh) -> this store's order."""
    a = np.asarray(arr)
    if a.shape[1] != N_LM:
        raise ValueError(f"expected 543 landmarks, got {a.shape}")
    face = a[:, G_FACE0:G_FACE0 + N_FACE]
    lh = a[:, G_LH0:G_LH0 + N_HAND]
    pose = a[:, G_POSE0:G_POSE0 + N_POSE]
    rh = a[:, G_RH0:G_RH0 + N_HAND]
    return np.concatenate([pose, lh, rh, face], axis=1)


def to_gislr(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    pose, lh, rh, face = a[:, :LH0], a[:, LH0:RH0], a[:, RH0:FACE0], a[:, FACE0:]
    return np.concatenate([face, lh, pose, rh], axis=1)


def load_gislr_parquet(path) -> np.ndarray:
    """Kaggle asl-signs parquet (rows = frame x 543 landmarks) -> (T, 543, 3) GISLR order."""
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    n = len(df)
    if n % N_LM:
        raise ValueError(f"{path}: {n} rows is not a multiple of {N_LM}")
    # Verify the row order instead of assuming it (it is face, left_hand, pose, right_hand).
    types = df["type"].to_numpy()[:N_LM]
    want = ["face"] * N_FACE + ["left_hand"] * N_HAND + ["pose"] * N_POSE + ["right_hand"] * N_HAND
    if list(types) != want:
        raise ValueError(f"{path}: unexpected GISLR row order {pd.unique(types)}")
    return df[["x", "y", "z"]].to_numpy(np.float32).reshape(-1, N_LM, 3)


# ---------------------------------------------------------------------------------
# model inputs
# ---------------------------------------------------------------------------------
def sample_indices(n: int, target: int) -> np.ndarray:
    """Same rule as `common.landmarks.sample_frame_indices` (uniform, pad with last)."""
    if n <= 0:
        return np.zeros(target, dtype=np.int64)
    if n >= target:
        return np.linspace(0, n - 1, target).astype(np.int64)
    return np.concatenate([np.arange(n), np.full(target - n, n - 1)]).astype(np.int64)


def trim_handless(seq: np.ndarray, min_keep: int = 4) -> np.ndarray:
    """Drop leading/trailing frames with no hand detected (rest pose before/after the sign)."""
    hand = np.isfinite(seq[:, LH0:FACE0, 0]).any(1)
    if hand.sum() < min_keep:
        return seq
    idx = np.where(hand)[0]
    return seq[idx[0]: idx[-1] + 1]


def to_codebase_seq(raw: np.ndarray, num_frames: int = 30, trim: bool = True,
                    reduced: bool = True) -> np.ndarray:
    """Stored clip (T, 543, 3) -> (num_frames, V*3) normalised like the legacy cache.

    reduced=True keeps pose|lh|rh|lips+nose (119 points); the legacy cache kept all 543.
    Every existing model only reads the first 75 points, so both work.
    """
    from islr.common.landmarks import normalize_landmarks  # noqa: PLC0415

    seq = np.asarray(raw, dtype=np.float32)
    if trim:
        seq = trim_handless(seq)
    seq = seq[sample_indices(len(seq), num_frames)]
    seq = np.nan_to_num(seq, nan=0.0)
    out = np.stack([normalize_landmarks(f) for f in seq])  # (T, 543, 3), missing stay 0
    if reduced:
        out = out[:, REDUCED_ROWS]
    return out.reshape(num_frames, -1).astype(np.float32)


# ---------------------------------------------------------------------------------
# identities (signer-independent splits)
# ---------------------------------------------------------------------------------
_USER = re.compile(r"^User\d+$", re.I)
_MVI = re.compile(r"(?:MVI|IMG|VID)_(\d+)", re.I)


def include_sessions(numbers: pd.Series, gap: int = 200) -> pd.Series:
    """INCLUDE has no signer ids, but its clip numbers (MVI_xxxx) are camera counters.
    A recording session covers every word of a category consecutively, and sessions are
    separated by jumps of hundreds. A new session starts at every jump > `gap`.
    Over-merging two sessions only makes the split coarser (safe); splitting one
    session would leak a signer, so `gap` is deliberately large."""
    nums = pd.to_numeric(numbers, errors="coerce")
    order = np.sort(nums.dropna().unique())
    if not len(order):
        return pd.Series([""] * len(numbers), index=numbers.index)
    starts = order[np.r_[True, np.diff(order) > gap]]
    def sess(n):
        if not np.isfinite(n):
            return ""
        return f"include_s{int(starts[np.searchsorted(starts, n, side='right') - 1])}"
    return nums.map(sess)


def identity(row) -> str:
    """Group key that must never be split across train and test."""
    signer = str(row.get("signer") or "").strip()
    session = str(row.get("session") or "").strip()
    if signer and signer.lower() not in {"nan", "none", "unknown"}:
        return f"signer:{signer}"
    if session and session.lower() != "nan":
        return f"session:{session}"
    return f"clip:{row.get('source')}:{row.get('key')}"


# ---------------------------------------------------------------------------------
# index io
# ---------------------------------------------------------------------------------
def clip_key(source: str, rel: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(rel).stem)[:40].strip("_")
    digest = hashlib.sha1(f"{source}/{rel}".encode("utf-8")).hexdigest()[:10]
    return f"{source}_{stem}_{digest}"


def init_store(store: str | Path) -> Path:
    store = Path(store)
    (store / "npy").mkdir(parents=True, exist_ok=True)
    meta = store / "store.json"
    if not meta.exists():
        meta.write_text(json.dumps({"layout": LAYOUT, "dtype": "float16", "missing": "nan"}, indent=1))
    else:
        tag = json.loads(meta.read_text()).get("layout")
        if tag != LAYOUT:
            raise ValueError(f"{store} has layout {tag!r}, expected {LAYOUT!r}")
    return store


def append_index(store: str | Path, rows: pd.DataFrame) -> pd.DataFrame:
    path = Path(store) / "index.csv"
    rows = rows.copy()
    for c in INDEX_COLUMNS:
        if c not in rows:
            rows[c] = ""
    rows = rows[INDEX_COLUMNS].astype(str)
    if path.exists():
        rows = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False), rows])
    rows = rows.drop_duplicates("key", keep="last").sort_values(["word", "key"])
    tmp = path.with_suffix(".tmp")
    rows.to_csv(tmp, index=False)
    tmp.replace(path)
    return rows


def read_index(store: str | Path) -> pd.DataFrame:
    store = Path(store)
    meta = store / "store.json"
    if meta.exists() and json.loads(meta.read_text()).get("layout") != LAYOUT:
        raise ValueError(f"{store}: wrong layout")
    df = pd.read_csv(store / "index.csv", dtype=str, keep_default_na=False)
    df["abs_path"] = [str(store / p) for p in df["path"]]
    df["store"] = str(store)
    df["n_frames"] = pd.to_numeric(df["n_frames"], errors="coerce").fillna(0).astype(int)
    return df[df["n_frames"] > 0].reset_index(drop=True)


def normalize_word(w: str) -> str:
    """Canonical gloss: lower-case, INCLUDE 'NN. Word' prefix and punctuation removed."""
    w = str(w).strip()
    w = re.sub(r"^\d+\s*[.)]\s*", "", w)
    w = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", w)
    w = w.replace("_", " ").lower()
    w = re.sub(r"[^a-z0-9' ]+", " ", w)
    return re.sub(r"\s+", " ", w).strip()


def mvi_number(rel: str) -> float:
    m = _MVI.search(str(rel))
    return float(m.group(1)) if m else np.nan


def load_stores(stores: list[str], sources: list[str] | None = None, words: list[str] | None = None,
                dedup: bool = True) -> pd.DataFrame:
    """Concatenate store indexes into one clip table with an `identity` column.

    * Words are normalised (`normalize_word`) so glosses from different sources match.
    * The same INCLUDE clip can sit in two stores (the 40-word corpus contains 143
      INCLUDE clips). `dedup` keeps the first copy in `stores` order, matching on
      (word, MVI number).
    * INCLUDE sessions are computed here over all INCLUDE clips jointly (see
      `include_sessions`), so identities are consistent whichever stores are loaded.
    """
    frames = [read_index(s) for s in stores]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLUMNS)
    df["word"] = df["word"].map(normalize_word)
    if sources:
        df = df[df["source"].isin(sources)]
    if words:
        df = df[df["word"].isin({normalize_word(w) for w in words})]
    df = df.reset_index(drop=True)
    inc = df["source"].eq("include")
    df["mvi"] = [mvi_number(r) if i else np.nan for r, i in zip(df["video_rel"], inc)]
    if dedup and len(df):
        uid = np.where(inc & df["mvi"].notna(),
                       "include:" + df["word"] + ":" + df["mvi"].fillna(-1).astype(int).astype(str),
                       df["source"] + ":" + df["key"])
        df = df[~pd.Series(uid, index=df.index).duplicated(keep="first")].reset_index(drop=True)
        inc = df["source"].eq("include")
    need = inc & df["mvi"].notna() & df["session"].astype(str).str.strip().eq("")
    if need.any():
        df.loc[need, "session"] = include_sessions(df.loc[inc, "mvi"]).loc[need[need].index]
    df["identity"] = [identity(r) for r in df.to_dict("records")]
    return df.sort_values(["word", "source", "key"]).reset_index(drop=True)


def merge_stores(srcs: list[str], dst: str) -> pd.DataFrame:
    """Copy several stores (a teammate's output, a previous session, extraction shards)
    into one. Progress logs (`*_done.txt`, e.g. the INCLUDE zips already processed) are
    unioned too, so the merged store skips work any of them finished."""
    import shutil

    out = init_store(dst)
    rows = []
    for s in srcs:
        s = Path(s)
        for log in s.glob("*_done.txt"):
            mine = out / log.name
            have = set(mine.read_text().split()) if mine.exists() else set()
            new = [k for k in log.read_text().split() if k not in have]
            if new:
                with open(mine, "a") as f:
                    f.writelines(k + "\n" for k in new)
        if not (s / "index.csv").exists():
            continue
        idx = read_index(s)
        for p in idx["path"]:
            target = out / p
            if not target.exists():
                shutil.copy2(s / p, target)
        rows.append(idx.drop(columns=["abs_path", "store"]))
    return append_index(out, pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


COMPLETE = "COMPLETE.json"


def store_fingerprint(store: str | Path) -> dict:
    """Clip count + hash of the clip keys. Two members whose stores have the same
    fingerprint get identical splits (test sets, training subsets) for every seed."""
    idx = read_index(store) if (Path(store) / "index.csv").exists() else pd.DataFrame({"key": []})
    keys = sorted(idx["key"])
    return {"n_clips": len(keys), "n_words": int(idx["word"].nunique()) if len(idx) else 0,
            "keys_sha1": hashlib.sha1("\n".join(keys).encode()).hexdigest()[:12]}


def mark_complete(store: str | Path, **extra) -> dict:
    """Written when a source has been ingested completely (not stopped by a time budget)."""
    import time

    info = dict(store_fingerprint(store), finished=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    (Path(store) / COMPLETE).write_text(json.dumps(info, indent=1))
    return info


def completion(store: str | Path) -> dict | None:
    f = Path(store) / COMPLETE
    return json.loads(f.read_text()) if f.exists() else None

"""Data sources -> landmark stores. Designed for Kaggle's ~20 GB working disk: big
archives are streamed part by part (download -> extract landmarks -> delete).

  isl40          the project's own corpus (HF vidit031/isl-isolated-40words; 642 clips,
                 40 words; ISL500 + INCLUDE + CISLR + ISLRTC clips). Local folder or HF.
  include        INCLUDE (AI4Bharat/IITM, Zenodo record 4010759): 263 words, 4,292 videos,
                 ~57 GB in 44 category zips, CC BY 4.0. The project used only 143 of them.
                 Zenodo throttles each connection (~50 KB/s observed), so files are fetched
                 as parallel HTTP byte ranges with resume.
  isl_dictionary ISLRTC dictionary (HF silentone0725/Indian_Sign_Language_Data.gov_Rencoded,
                 MIT): ~13.6k clips, ONE per word, <Letter>/<Word>.mp4. Only words that
                 overlap the target vocabulary are downloaded; "(Explaination)" clips skipped.
  cislr          CISLR (HF Exploration-Lab/CISLR, gated: accept the terms, set HF_TOKEN).
  gislr          Kaggle 'asl-signs' competition (ASL, 94k clips, already landmarks).
                 Converted to this store's row order. Competition data: do not redistribute.
  folder         any <root>/<word>/<video> tree.

    python lowdata.py sources list
    python lowdata.py sources isl40 --root ../IPD/ISL_DATASET_40WORDS --store stores/isl40
    python lowdata.py sources include --store stores/include --files Electronics_2of2.zip
    python lowdata.py sources isl_dictionary --store stores/isl_dict --vocab-from stores/isl40
    python lowdata.py sources gislr --root /kaggle/input/asl-signs --store stores/gislr
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .extract import VIDEO_EXTS, extract_to_store, in_shard, parse_shard, videos_from_folder
from .store import (append_index, clip_key, from_gislr, init_store, load_gislr_parquet, merge_stores,
                    normalize_word, read_index)

ZENODO_INCLUDE = "https://zenodo.org/api/records/4010759"
HF_DICT = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
HF_CISLR = "Exploration-Lab/CISLR"
HF_ISL40 = "vidit031/isl-isolated-40words"

LICENSES = {
    "include": "CC-BY-4.0 (INCLUDE, Zenodo 4010759)",
    "isl_dictionary": "MIT (HF re-encode of ISLRTC dictionary)",
    "cislr": "research-only, gated (Exploration-Lab/CISLR card)",
    "gislr": "Kaggle competition rules - do not redistribute",
}


# ----------------------------------------------------------------------------------
# HTTP with resume + parallel ranges
# ----------------------------------------------------------------------------------
def _get_json(url: str, token: str | None = None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    for attempt in range(6):  # listing APIs rate-limit bursts
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode()), r.headers
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"[http] {url}: {exc!r}, retrying", flush=True)
            time.sleep(5 * 2 ** attempt)


def _fetch_range(url, part: Path, start: int, end: int, retries: int, token) -> None:
    want = end - start + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return
        headers = {"Range": f"bytes={start + have}-{end}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError("server ignored Range")
                with open(part, "ab") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
        except Exception as exc:
            print(f"[download] {part.name}: {exc!r}, retry {attempt + 1}/{retries}", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))
    if not part.exists() or part.stat().st_size < want:
        raise RuntimeError(f"failed to download bytes {start}-{end} of {url}")


def download(url: str, dst: Path, size: int | None = None, token: str | None = None,
             connections: int = 8, retries: int = 10, min_parallel: int = 32 << 20) -> Path:
    """Resumable download. With a known size, fetch `connections` byte ranges in
    parallel; every range resumes independently after a dropped connection."""
    dst = Path(dst)
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if size and size >= min_parallel and connections > 1:
        from concurrent.futures import ThreadPoolExecutor

        chunk = -(-size // connections)
        spans = [(i, i * chunk, min(size, (i + 1) * chunk) - 1) for i in range(connections)]
        parts = [dst.with_name(dst.name + f".part{i}") for i, _, _ in spans]
        with ThreadPoolExecutor(connections) as ex:
            list(ex.map(lambda s: _fetch_range(url, parts[s[0]], s[1], s[2], retries, token), spans))
        tmp = dst.with_name(dst.name + ".tmp")
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 20)
        if tmp.stat().st_size != size:
            raise RuntimeError(f"{dst.name}: {tmp.stat().st_size} bytes, expected {size}")
        tmp.replace(dst)
        for p in parts:
            p.unlink(missing_ok=True)
        return dst
    part = dst.with_name(dst.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, \
                    open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored Range -> restart
                    f.seek(0)
                    f.truncate()
                shutil.copyfileobj(r, f, 1 << 20)
            part.replace(dst)
            return dst
        except Exception as exc:
            print(f"[download] {dst.name}: {exc!r}, retry {attempt + 1}/{retries}", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"failed to download {url}")


# ----------------------------------------------------------------------------------
# the project's own corpus
# ----------------------------------------------------------------------------------
def ingest_isl40(store: str, root: str | None = None, workers: int | None = None, shard=None,
                 time_budget_h: float | None = None) -> None:
    """Existing corpus. metadata.csv carries dataset/signer; INCLUDE clips get a
    session from their MVI number at load time (store.identity via load_stores)."""
    if root is None or not (Path(root) / "metadata.csv").exists():
        from huggingface_hub import snapshot_download

        root = root or str(Path(store) / "_download")
        snapshot_download(repo_id=HF_ISL40, repo_type="dataset", local_dir=root,
                          token=os.environ.get("HF_TOKEN"))
    root = Path(root)
    meta = pd.read_csv(root / "metadata.csv", dtype=str, keep_default_na=False)
    rows = []
    for r in meta.itertuples(index=False):
        rel = str(r.video_path).replace("\\", "/")
        rel = re.sub(r"^.*?ISL_DATASET[^/]*/", "", rel)
        p = root / rel
        if not p.exists():
            continue
        ds = str(r.dataset)
        signer = str(r.signer) if re.match(r"^User\d+$", str(r.signer)) else ""
        if ds.startswith("ISLRTC"):
            signer = "islrtc"
        m = re.search(r"session(\d+)", rel)
        session = f"isl500_s{m.group(1)}" if (ds == "ISL500" and m) else ""
        rows.append({"video_path": str(p), "video_rel": rel, "word": r.word,
                     "source": ds.lower(), "signer": signer, "session": session,
                     "license": r.license})
    print(f"[isl40] {len(rows)} videos found under {root}")
    extract_to_store(pd.DataFrame(rows), store, workers=workers, shard=shard,
                     time_budget_s=time_budget_h * 3600 if time_budget_h else None)


# ----------------------------------------------------------------------------------
# INCLUDE
# ----------------------------------------------------------------------------------
def include_files(categories=None, keys=None) -> list[dict]:
    meta, _ = _get_json(ZENODO_INCLUDE)
    files = [{"key": f["key"], "url": f["links"]["self"], "size": int(f["size"])}
             for f in meta["files"] if f["key"].lower().endswith(".zip")]
    if categories:
        want = {c.casefold().replace(" ", "_") for c in categories}
        files = [f for f in files if re.sub(r"_\d+of\d+\.zip$", "", f["key"]).casefold() in want]
    if keys:
        files = [f for f in files if f["key"] in set(keys)]
    return sorted(files, key=lambda f: f["key"])


def _include_rows(vdir: Path) -> list[dict]:
    rows = []
    for p in sorted(vdir.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS or "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(vdir)
        parts = [q for q in rel.parts[:-1] if q]
        if not parts:
            continue
        rows.append({"video_path": str(p), "video_rel": rel.as_posix(), "word": parts[-1],
                     "source": "include", "license": LICENSES["include"]})
    return rows


def ingest_include(store: str, categories=None, keys=None, work: str | None = None,
                   zip_dir: str | None = None, workers: int | None = None, keep_zips: bool = False,
                   time_budget_h: float | None = None, shard=None) -> None:
    """Zip by zip: download (or take from --zip-dir) -> unzip -> landmarks -> delete.

    shard=(i, n) takes every n-th zip (largest first, dealt round-robin), so the ~50 GB
    of downloads and the extraction are split between n people."""
    t0 = time.time()
    init_store(store)
    work_dir = Path(work or Path(store) / "_download")
    done_log = Path(store) / "include_done.txt"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    if zip_dir and not categories and not keys:
        files = [{"key": p.name, "url": None, "size": p.stat().st_size}
                 for p in sorted(Path(zip_dir).glob("*.zip"))]
    else:
        files = include_files(categories, keys)
    if shard is not None:
        files = sorted(files, key=lambda f: (-f["size"], f["key"]))[shard[0]::shard[1]]
    print(f"[include] {len(files)} zips, {sum(f['size'] for f in files) / 1e9:.1f} GB, {len(done)} done")
    for f in files:
        if f["key"] in done:
            continue
        if time_budget_h and time.time() - t0 > time_budget_h * 3600:
            print("[include] time budget reached; re-run to continue", flush=True)
            break
        local = Path(zip_dir) / f["key"] if zip_dir else None
        if local is not None and local.exists():
            zpath = local
        else:
            zpath = work_dir / f["key"]
            print(f"[include] downloading {f['key']} ({f['size'] / 1e9:.2f} GB)", flush=True)
            download(f["url"], zpath, size=f["size"])
        vdir = work_dir / Path(f["key"]).stem
        with zipfile.ZipFile(zpath) as z:
            z.extractall(vdir)
        rows = _include_rows(vdir)
        print(f"[include] {f['key']}: {len(rows)} videos, {len({r['word'] for r in rows})} words", flush=True)
        if rows:
            extract_to_store(pd.DataFrame(rows), store, workers=workers)
        shutil.rmtree(vdir, ignore_errors=True)
        if not keep_zips and zpath.parent == work_dir:
            zpath.unlink(missing_ok=True)
        with open(done_log, "a") as fh:
            fh.write(f["key"] + "\n")


# ----------------------------------------------------------------------------------
# Hugging Face: ISLRTC dictionary, CISLR
# ----------------------------------------------------------------------------------
def hf_list(repo: str, path: str = "", token: str | None = None) -> list[dict]:
    out = []
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"
    while url:
        page, headers = _get_json(url, token)
        out += page
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or "")
        url = m.group(1) if m else None
    return out


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def dictionary_word(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"\s*-\s*(English|Hindi|ISL)\s*$", "", stem, flags=re.I)
    return normalize_word(stem)


def ingest_isl_dictionary(store: str, vocab: set[str] | None, work: str | None = None,
                          workers: int | None = None, limit: int | None = None, shard=None) -> None:
    token = os.environ.get("HF_TOKEN")
    work_dir = Path(work or Path(store) / "_download")
    letters = [d["path"] for d in hf_list(HF_DICT, token=token) if d["type"] == "directory"]
    files = []
    for letter in letters:
        files += [f["path"] for f in hf_list(HF_DICT, letter, token)
                  if f["type"] == "file" and Path(f["path"]).suffix.lower() in VIDEO_EXTS]
    print(f"[dict] {len(files)} dictionary clips listed")
    # "(Explaination)" clips are long explanations, not citation-form signs
    files = [f for f in files if not re.search(r"expla", f, re.I)]
    if vocab:
        files = [f for f in files if dictionary_word(f) in vocab]
        hit = {dictionary_word(f) for f in files}
        print(f"[dict] {len(files)} clips for {len(hit)}/{len(vocab)} vocabulary words; "
              f"missing: {sorted(vocab - hit)[:30]}")
    if limit:
        files = files[:limit]
    if shard is not None:
        files = [f for f in files if in_shard(f, shard)]
    rows = []
    for f in files:
        dst = work_dir / f
        download(hf_url(HF_DICT, f), dst, token=token)
        rows.append({"video_path": str(dst), "video_rel": f, "word": dictionary_word(f),
                     "source": "isl_dictionary", "signer": "islrtc",
                     "license": LICENSES["isl_dictionary"]})
    if rows:
        extract_to_store(pd.DataFrame(rows), store, workers=workers)
    shutil.rmtree(work_dir, ignore_errors=True)


def ingest_cislr(store: str, work: str | None = None, workers: int | None = None,
                 vocab: set[str] | None = None, shard=None) -> None:
    """Gated: accept the terms at huggingface.co/datasets/Exploration-Lab/CISLR, then set
    HF_TOKEN. The CSV schema is detected at run time (untested without access)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("CISLR is gated: accept its terms on Hugging Face and set HF_TOKEN.")
    work_dir = Path(work or Path(store) / "_download")
    listing = [f["path"] for f in hf_list(HF_CISLR, token=token) if f["type"] == "file"]
    print(f"[cislr] repo files: {listing}")
    csvs = [p for p in listing if p.lower().endswith(".csv")]
    zips = [p for p in listing if p.lower().endswith(".zip")]
    for p in csvs + zips:
        download(hf_url(HF_CISLR, p), work_dir / Path(p).name, token=token)
    vdir = work_dir / "videos"
    for z in zips:
        with zipfile.ZipFile(work_dir / Path(z).name) as zf:
            zf.extractall(vdir)
    videos = {p.stem: p for p in vdir.rglob("*") if p.suffix.lower() in VIDEO_EXTS}
    rows = []
    for c in csvs:
        df = pd.read_csv(work_dir / Path(c).name)
        vcol = next((k for k in df.columns
                     if df[k].astype(str).map(lambda s: Path(s).stem in videos).mean() > 0.5), None)
        wcol = next((k for k in df.columns if k != vcol and re.search(r"word|gloss|label|sign", k, re.I)), None)
        print(f"[cislr] {c}: columns={list(df.columns)} video={vcol} word={wcol}")
        if vcol is None or wcol is None:
            continue
        for v, w in zip(df[vcol].astype(str), df[wcol].astype(str)):
            p = videos.get(Path(v).stem)
            if p is not None and (not vocab or normalize_word(w) in vocab):
                rows.append({"video_path": str(p), "video_rel": p.name, "word": w, "source": "cislr",
                             "license": LICENSES["cislr"]})
    df = pd.DataFrame(rows).drop_duplicates("video_rel")
    extract_to_store(df, store, workers=workers, shard=shard)
    shutil.rmtree(work_dir, ignore_errors=True)


# ----------------------------------------------------------------------------------
# already-landmarked: ASL Signs (GISLR)
# ----------------------------------------------------------------------------------
def ingest_gislr(store: str, root: str, max_samples: int | None = None, signs=None) -> None:
    root, store = Path(root), init_store(store)
    meta = pd.read_csv(root / "train.csv")
    if signs:
        meta = meta[meta["sign"].isin(signs)]
    if max_samples:
        meta = meta.sample(n=min(max_samples, len(meta)), random_state=0)
    rows = []
    for i, r in enumerate(meta.itertuples(index=False), 1):
        key = clip_key("gislr", f"{r.participant_id}/{r.sequence_id}")
        dst = store / "npy" / f"{key}.npy"
        if not dst.exists():
            arr = from_gislr(load_gislr_parquet(root / r.path))
            np.save(dst, arr.astype(np.float16))
            n = len(arr)
        else:
            n = np.load(dst, mmap_mode="r").shape[0]
        rows.append({"key": key, "word": r.sign, "source": "gislr", "signer": f"asl{r.participant_id}",
                     "n_frames": n, "fps": "30", "path": f"npy/{key}.npy", "video_rel": r.path,
                     "license": LICENSES["gislr"]})
        if i % 5000 == 0:
            print(f"[gislr] {i}/{len(meta)}", flush=True)
            append_index(store, pd.DataFrame(rows))
    append_index(store, pd.DataFrame(rows))
    print(f"[gislr] {len(rows)} sequences -> {store}")


def vocab_from_stores(stores: list[str]) -> set[str]:
    out: set[str] = set()
    for s in stores:
        out |= set(read_index(s)["word"].map(normalize_word))
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    p = sub.add_parser("isl40")
    p.add_argument("--root", default=None, help="local folder with metadata.csv (else HF download)")
    p.add_argument("--time-budget-h", type=float, default=None)
    p = sub.add_parser("include")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--files", nargs="*", default=None, help="exact zip names, e.g. Greetings_1of2.zip")
    p.add_argument("--zip-dir", default=None, help="use already-downloaded zips from here")
    p.add_argument("--keep-zips", action="store_true")
    p.add_argument("--time-budget-h", type=float, default=None)
    p = sub.add_parser("isl_dictionary")
    p.add_argument("--vocab-from", nargs="*", default=None, help="stores whose words define the vocabulary")
    p.add_argument("--vocab", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("cislr")
    p.add_argument("--vocab-from", nargs="*", default=None)
    p = sub.add_parser("gislr")
    p.add_argument("--root", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p = sub.add_parser("merge", help="merge shard stores into one")
    p.add_argument("--from", dest="srcs", nargs="+", required=True)
    p = sub.add_parser("folder")
    p.add_argument("--root", required=True)
    p.add_argument("--name", default="folder")
    sub.add_parser("list")
    for s in sub.choices.values():
        s.add_argument("--store", default=None)
        s.add_argument("--workers", type=int, default=None)
        s.add_argument("--work", default=None, help="scratch dir for downloads")
        s.add_argument("--shard", default=None, help="i/n: do only shard i of n (0-based), one per team member")
    a = ap.parse_args(argv)
    if a.source != "list" and not a.store:
        ap.error("--store is required")

    shard = parse_shard(a.shard)
    if a.source == "isl40":
        ingest_isl40(a.store, a.root, a.workers, shard, getattr(a, "time_budget_h", None))
    elif a.source == "include":
        ingest_include(a.store, a.categories, a.files, a.work, a.zip_dir, a.workers, a.keep_zips,
                       a.time_budget_h, shard)
    elif a.source == "isl_dictionary":
        vocab = vocab_from_stores(a.vocab_from) if a.vocab_from else set()
        vocab |= {normalize_word(v) for v in (a.vocab or [])}
        ingest_isl_dictionary(a.store, vocab or None, a.work, a.workers, a.limit, shard)
    elif a.source == "cislr":
        ingest_cislr(a.store, a.work, a.workers, vocab_from_stores(a.vocab_from) if a.vocab_from else None, shard)
    elif a.source == "gislr":
        ingest_gislr(a.store, a.root, a.max_samples)
    elif a.source == "folder":
        extract_to_store(videos_from_folder(a.root, a.name), a.store, workers=a.workers, shard=shard)
    elif a.source == "merge":
        rows = merge_stores(a.srcs, a.store)
        print(f"[merge] {len(a.srcs)} stores -> {a.store}: {len(rows)} clips")
    elif a.source == "list":
        files = include_files()
        print(f"INCLUDE (Zenodo 4010759): {len(files)} zips, {sum(f['size'] for f in files) / 1e9:.1f} GB")
        for f in files:
            print(f"  {f['key']:34s} {f['size'] / 1e9:6.2f} GB")


if __name__ == "__main__":
    sys.exit(main())

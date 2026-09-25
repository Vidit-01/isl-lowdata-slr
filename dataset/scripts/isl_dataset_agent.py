"""
ISL Dataset Collection Agent — reproducible pipeline for Indian Sign Language
isolated-word video aggregation, normalization, and reporting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
RAW = ROOT / "raw_datasets"
OUT = ROOT / "ISL_DATASET"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"

for d in (CONFIG, RAW, OUT, REPORTS, LOGS, CACHE):
    d.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v", ".wmv"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logger(name: str = "isl_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOGS / "agent_actions.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_logger()


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, str] = {
    # Only high-confidence orthographic variants of the SAME gloss string.
    # Semantic synonyms (thanks≠thank you, eat≠food) are NEVER auto-merged.
    "thankyou": "thank you",
    "goodmorning": "good morning",
    "goodnight": "good night",
    "howareyou": "how are you",
}


def normalize_label(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).strip().lower()
    t = t.replace("_", " ").replace("-", " ").replace("/", " ")
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    compact = t.replace(" ", "")
    if compact in ALIAS_MAP:
        return ALIAS_MAP[compact]
    return t


def folder_name(word: str) -> str:
    return normalize_label(word).replace(" ", "_")


def load_target_words(path: Path | None = None) -> list[str]:
    path = path or (CONFIG / "target_words.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    words = data.get("words") or []
    if not words:
        raise SystemExit(
            f"No target words in {path}. Supply your ~40-word list and re-run."
        )
    if data.get("_status") == "AWAITING_USER_LIST":
        log.warning(
            "target_words.json still marked AWAITING_USER_LIST — using listed "
            "words as provisional. Replace with your full list for production."
        )
    # Preserve order, unique by normalized form
    seen = set()
    out = []
    for w in words:
        n = normalize_label(w)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Checksums / quality helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already present: %s", dest)
        return dest
    # Prefer curl for large resumable downloads (Zenodo etc.)
    curl = which("curl") or which("curl.exe")
    if curl:
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                log.info("curl download (%s/%s): %s -> %s", attempt, retries, url, dest)
                subprocess.check_call(
                    [
                        curl,
                        "-L",
                        "--retry",
                        "5",
                        "--continue-at",
                        "-",
                        "-o",
                        str(dest),
                        url,
                    ]
                )
                if dest.exists() and dest.stat().st_size > 0:
                    return dest
                raise IOError("zero-byte download")
            except Exception as e:
                last_err = e
                log.error("curl failed attempt %s: %s", attempt, e)
                time.sleep(2 * attempt)
        raise RuntimeError(f"Failed to download {url}: {last_err}")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading (%s/%s): %s -> %s", attempt, retries, url, dest)
            req = Request(url, headers={"User-Agent": "ISL-Dataset-Agent/1.0"})
            with urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out)
            if dest.stat().st_size == 0:
                raise IOError("zero-byte download")
            return dest
        except Exception as e:
            last_err = e
            log.error("Download failed attempt %s: %s", attempt, e)
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ffprobe_meta(path: Path) -> dict[str, Any]:
    ffprobe = which("ffprobe")
    if not ffprobe:
        # Fallback via OpenCV
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return {"ok": False, "error": "cannot open"}
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (n / fps) if fps > 0 else 0.0
            ok_read, frame = cap.read()
            cap.release()
            if not ok_read:
                return {"ok": False, "error": "no frames"}
            return {
                "ok": True,
                "fps": fps,
                "width": w,
                "height": h,
                "duration": duration,
                "frames": n,
                "brightness": float(frame.mean()) if frame is not None else None,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        data = json.loads(raw)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        rate = stream.get("r_frame_rate") or "0/1"
        if "/" in rate:
            a, b = rate.split("/")
            fps = float(a) / float(b) if float(b) else 0.0
        else:
            fps = float(rate or 0)
        duration = float(stream.get("duration") or fmt.get("duration") or 0)
        return {
            "ok": True,
            "fps": fps,
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "duration": duration,
            "codec": stream.get("codec_name"),
            "frames": stream.get("nb_frames"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def normalize_video(src: Path, dest: Path, target_fps: int = 30, height: int = 480) -> dict[str, Any]:
    """Convert to MP4 H.264, constant FPS, letterboxed to height, no crop of signing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = which("ffmpeg")
    transform = {
        "src": str(src),
        "dest": str(dest),
        "target_fps": target_fps,
        "height": height,
        "method": None,
    }
    if ffmpeg:
        # Scale by height, pad to even dims, keep aspect ratio (no crop)
        vf = (
            f"fps={target_fps},"
            f"scale=-2:{height}:force_original_aspect_ratio=decrease,"
            f"pad=ceil(iw/2)*2:ceil(ih/2)*2"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(dest),
        ]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            transform["method"] = "ffmpeg_h264"
            return transform
        except Exception as e:
            log.error("ffmpeg failed for %s: %s — copying original", src, e)

    # Fallback: copy if already mp4 else fail soft-copy
    shutil.copy2(src, dest.with_suffix(src.suffix) if dest.suffix != src.suffix else dest)
    if dest.suffix.lower() != ".mp4":
        alt = dest
        dest = dest.with_suffix(src.suffix)
    transform["method"] = "copy_fallback"
    transform["dest"] = str(dest)
    return transform


def simple_phash(path: Path) -> str:
    """Lightweight perceptual hash from a mid-frame (avg hash)."""
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return ""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean = small.mean()
        bits = (small > mean).flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return f"{val:016x}"
    except Exception:
        return ""


def hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 64
    return sum(c1 != c2 for c1, c2 in zip(bin(int(a, 16))[2:].zfill(64), bin(int(b, 16))[2:].zfill(64)))


# ---------------------------------------------------------------------------
# Dataset inventory (discovered sources)
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    name: str
    authors: str
    year: int | str
    paper: str
    repository: str
    download_link: str
    license: str
    citation: str
    n_videos: str
    vocabulary_size: str
    continuous_or_isolated: str
    modalities: str
    resolution: str
    fps: str
    n_signers: str
    language: str = "Indian Sign Language"
    downloadable: str = "yes"
    preprocessing_required: str = "varies"
    redistribution: str = "check_license"
    notes: str = ""
    status: str = "discovered"


DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="INCLUDE",
        authors="Sridhar, Ganesan, Kumar, Khapra (AI4Bharat / IIT Madras)",
        year=2020,
        paper="https://doi.org/10.1145/3394171.3413528",
        repository="https://github.com/AI4Bharat/INCLUDE",
        download_link="https://zenodo.org/records/4010759",
        license="CC-BY-4.0",
        citation="@inproceedings{sridhar_include_2020,...}",
        n_videos="4292",
        vocabulary_size="263 (INCLUDE-50: 50)",
        continuous_or_isolated="isolated",
        modalities="RGB",
        resolution="varies",
        fps="varies",
        n_signers="multiple (St. Louis School for the Deaf, Chennai)",
        redistribution="allowed_with_attribution",
        notes="Category zip files on Zenodo; train/test CSVs included",
    ),
    DatasetInfo(
        name="CISLR",
        authors="Joshi, Bhat, S, Gole, Gupta, Agarwal, Modi",
        year=2022,
        paper="https://aclanthology.org/2022.emnlp-main.707",
        repository="https://huggingface.co/datasets/Exploration-Lab/CISLR",
        download_link="https://huggingface.co/datasets/Exploration-Lab/CISLR",
        license="research (see HF card)",
        citation="@inproceedings{cislr-2022,...}",
        n_videos="~7050",
        vocabulary_size="~4765",
        continuous_or_isolated="isolated",
        modalities="RGB (+ I3D features)",
        resolution="varies (YouTube / dictionary sources)",
        fps="varies",
        n_signers="~71",
        notes="Scraped from ISLRTC + indiansignlanguage.org YouTube dictionaries",
    ),
    DatasetInfo(
        name="iSign / ISLTranslate",
        authors="Joshi et al. (Exploration-Lab)",
        year=2024,
        paper="https://aclanthology.org/2024.findings-acl.643",
        repository="https://huggingface.co/datasets/Exploration-Lab/iSign",
        download_link="https://huggingface.co/datasets/Exploration-Lab/iSign",
        license="research non-commercial",
        citation="@inproceedings{iSign-2024,...}",
        n_videos="~118k video-text pairs (includes continuous)",
        vocabulary_size="large (sentence/phrase)",
        continuous_or_isolated="continuous (+ includes CISLR isolated)",
        modalities="RGB + pose",
        resolution="varies",
        fps="varies",
        n_signers="many (YouTube)",
        redistribution="research_only_no_commercial",
        notes="Continuous translation; isolated via CISLR subset",
    ),
    DatasetInfo(
        name="ISL-DATA (ISL500)",
        authors="ISL500",
        year=2025,
        paper="",
        repository="https://huggingface.co/datasets/ISL500/ISL-DATA",
        download_link="https://huggingface.co/datasets/ISL500/ISL-DATA",
        license="see HF card",
        citation="ISL500/ISL-DATA Hugging Face",
        n_videos="~7500",
        vocabulary_size="500",
        continuous_or_isolated="isolated",
        modalities="RGB + MediaPipe + MMPose",
        resolution="controlled",
        fps="controlled",
        n_signers="15",
    ),
    DatasetInfo(
        name="ISLRTC Indian Sign Language Dictionary (data.gov.in)",
        authors="ISLRTC, Government of India",
        year="2024",
        paper="",
        repository="https://www.data.gov.in/resource/indian-sign-language-dictionary-till-january-2024",
        download_link="https://huggingface.co/datasets/Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded",
        license="Government open data / MIT re-encode (check both)",
        citation="ISLRTC Indian Sign Language Dictionary",
        n_videos="thousands (dictionary clips)",
        vocabulary_size="large (words, alphabets, numbers)",
        continuous_or_isolated="isolated",
        modalities="RGB (H.265 re-encode ~75GB)",
        resolution="varies",
        fps="varies",
        n_signers="dictionary signers",
        redistribution="check_data_gov_terms",
        notes="Official dictionary; CISLR also derives from this source — expect overlap",
    ),
    DatasetInfo(
        name="ISL-CSLTR",
        authors="Elakkiya R, Natarajan B",
        year=2021,
        paper="Mendeley Data 10.17632/kcmpdxky7p",
        repository="https://data.mendeley.com/datasets/kcmpdxky7p/1",
        download_link="https://data.mendeley.com/datasets/kcmpdxky7p/1",
        license="CC-BY-4.0",
        citation="Elakkiya & Natarajan 2021",
        n_videos="700 (+ word-level images)",
        vocabulary_size="100 sentences",
        continuous_or_isolated="continuous",
        modalities="RGB",
        resolution="DSLR",
        fps="varies",
        n_signers="7",
        notes="Sentence-level; less useful for isolated word extraction without segmentation",
    ),
    DatasetInfo(
        name="ISL-50",
        authors="Surbhi Maheshwari (01surbhi)",
        year=2025,
        paper="https://doi.org/10.5281/zenodo.18679858",
        repository="https://zenodo.org/records/18679858",
        download_link="https://drive.google.com/file/d/14HhiA4ki2x0_FO_YcRy-XiAR-g-O-bmk/view",
        license="academic/research only",
        citation="ISL-50 Zenodo",
        n_videos="800",
        vocabulary_size="50",
        continuous_or_isolated="isolated",
        modalities="RGB (.mov)",
        resolution="720p",
        fps="varies",
        n_signers="4",
        redistribution="research_only",
    ),
    DatasetInfo(
        name="ISL-52",
        authors="01surbhi",
        year=2025,
        paper="https://doi.org/10.5281/zenodo.18679890",
        repository="https://zenodo.org/records/18679890",
        download_link="Google Drive (see Zenodo)",
        license="academic/research only",
        citation="ISL-52 Zenodo",
        n_videos="832",
        vocabulary_size="52",
        continuous_or_isolated="isolated",
        modalities="RGB (.mov)",
        resolution="720p",
        fps="varies",
        n_signers="4",
        redistribution="research_only",
        notes="May overlap heavily with ISL-50",
    ),
    DatasetInfo(
        name="IIITA-ROBITA ISL Gesture Database",
        authors="Nandy, Mondal, Prasad, Chakraborty, Nandi (IIIT Allahabad)",
        year=2010,
        paper="ICCCT-10 / Springer LNCS-CCIS 2010",
        repository="https://robita.iiita.ac.in/dataset.php",
        download_link="https://robita.iiita.ac.in/dataset.php",
        license="copyright IIITA-ROBITA (request access)",
        citation="Nandy et al. 2010",
        n_videos="23 gesture categories (frame sequences)",
        vocabulary_size="23",
        continuous_or_isolated="isolated",
        modalities="RGB frames",
        resolution="320x240",
        fps="30",
        n_signers="lab collection",
        downloadable="request",
        redistribution="restricted",
        notes="Access via lab; not bulk-downloadable without permission",
        status="restricted",
    ),
]


INCLUDE_CATEGORY_TO_ZIP_PREFIX = {
    "adjectives": "Adjectives",
    "animals": "Animals",
    "clothes": "Clothes",
    "colours": "Colours",
    "colors": "Colours",
    "days and time": "Days_and_Time",
    "electronics": "Electronics",
    "greetings": "Greetings",
    "home": "Home",
    "jobs": "Jobs",
    "means of transportation": "Means_of_Transportation",
    "people": "People",
    "places": "Places",
    "pronouns": "Pronouns",
    "seasons": "Seasons",
    "society": "Society",
}


def write_inventory() -> Path:
    path = REPORTS / "dataset_inventory.csv"
    fields = list(asdict(DATASETS[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in DATASETS:
            w.writerow(asdict(d))
    license_path = REPORTS / "license_report.csv"
    with license_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "license",
                "redistribution",
                "citation",
                "notes",
                "downloadable",
            ],
        )
        w.writeheader()
        for d in DATASETS:
            w.writerow(
                {
                    "dataset": d.name,
                    "license": d.license,
                    "redistribution": d.redistribution,
                    "citation": d.citation,
                    "notes": d.notes,
                    "downloadable": d.downloadable,
                }
            )
    log.info("Wrote inventory (%s datasets) -> %s", len(DATASETS), path)
    return path


# ---------------------------------------------------------------------------
# INCLUDE helpers
# ---------------------------------------------------------------------------

def zenodo_include_files() -> list[dict]:
    cache = CACHE / "zenodo_4010759.json"
    if not cache.exists():
        download_file("https://zenodo.org/api/records/4010759", cache)
    return json.loads(cache.read_text(encoding="utf-8")).get("files", [])


def download_include_category(category_prefix: str) -> list[Path]:
    """Download and extract all Zenodo zips for a category prefix (e.g. Greetings)."""
    files = zenodo_include_files()
    out_dir = RAW / "INCLUDE"
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    matched = [f for f in files if f["key"].startswith(category_prefix) and f["key"].endswith(".zip")]
    log.info("INCLUDE category %s: %s zip parts", category_prefix, len(matched))
    for fmeta in matched:
        key = fmeta["key"]
        url = fmeta["links"]["self"]
        zip_path = out_dir / key
        try:
            download_file(url, zip_path, timeout=600)
        except Exception as e:
            log.error("INCLUDE zip failed %s: %s", key, e)
            continue
        extract_to = out_dir / "extracted"
        extract_to.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            log.info("Extracted %s", key)
            extracted.append(extract_to)
        except Exception as e:
            log.error("Extract failed %s: %s", key, e)
    return extracted


def iter_videos(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def strip_include_index(label: str) -> str:
    """INCLUDE folders look like '48. Hello' or '55. Thank you'."""
    return re.sub(r"^\d+\.\s*", "", str(label)).strip()


def infer_label_from_path(path: Path, dataset_root: Path) -> str:
    """Heuristic: parent folder name is often the gloss/class."""
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        rel = path
    parts = list(rel.parts)
    generic = {
        "videos",
        "video",
        "train",
        "test",
        "val",
        "validation",
        "data",
        "raw",
        "extracted",
        "include",
        "mp4",
        "mov",
        "greetings",
        "adjectives",
        "animals",
        "clothes",
        "colours",
        "colors",
        "home",
        "jobs",
        "people",
        "places",
        "pronouns",
        "seasons",
        "society",
        "electronics",
        "days_and_time",
        "means_of_transportation",
    }
    for part in reversed(parts[:-1]):
        pl = part.lower()
        if pl in generic or re.fullmatch(r"user\d+|isl_data_user\d+", pl):
            continue
        return normalize_label(strip_include_index(part))
    # ISL500 style: Word__sessionN__clipM
    stem = path.stem
    if "__" in stem:
        return normalize_label(stem.split("__")[0])
    stem = re.sub(r"_(session|clip|id).*$", "", stem, flags=re.I)
    stem = re.sub(r"(_?\d+)+$", "", stem)
    return normalize_label(stem.split("_")[0] if "_" in stem else stem)


# ---------------------------------------------------------------------------
# Matching & unified build
# ---------------------------------------------------------------------------

@dataclass
class VideoRecord:
    word: str
    normalized_word: str
    dataset: str
    original_label: str
    video_path: str
    original_filename: str
    signer: str
    split: str
    fps: str
    resolution: str
    duration: str
    license: str
    paper: str
    repository: str
    download_url: str
    sha256: str
    phash: str
    quality_score: str
    duplicate_status: str
    review_status: str


def quality_score(meta: dict) -> float:
    if not meta.get("ok"):
        return 0.0
    score = 1.0
    w, h = meta.get("width") or 0, meta.get("height") or 0
    if min(w, h) < 224:
        score -= 0.4
    elif min(w, h) < 360:
        score -= 0.15
    fps = meta.get("fps") or 0
    if fps < 12:
        score -= 0.3
    elif fps < 20:
        score -= 0.1
    dur = meta.get("duration") or 0
    if dur < 0.3 or dur > 30:
        score -= 0.4
    bright = meta.get("brightness")
    if bright is not None and (bright < 20 or bright > 235):
        score -= 0.2
    return max(0.0, round(score, 3))


def match_targets(label: str, targets: list[str]) -> tuple[Optional[str], str]:
    n = normalize_label(label)
    for t in targets:
        if n == t:
            return t, "exact"
    # compact equality
    nc = n.replace(" ", "")
    for t in targets:
        if nc == t.replace(" ", ""):
            return t, "normalized"
    return None, "no_match"


def include_categories_for_targets(targets: list[str]) -> set[str]:
    """Use HF parquet metadata to decide which Zenodo category zips to fetch."""
    meta_dir = RAW / "INCLUDE_meta" / "data"
    cats: set[str] = set()
    if not meta_dir.exists():
        # fallback heuristics
        greet = {"hello", "thank you", "good morning", "good night", "how are you", "good afternoon"}
        if any(t in greet for t in targets):
            cats.add("Greetings")
        return cats
    try:
        import pandas as pd

        frames = []
        for split in ("train", "test", "val"):
            p = meta_dir / f"{split}-00000-of-00001.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            return cats
        df = pd.concat(frames, ignore_index=True)
        df["clean"] = df["label"].astype(str).map(lambda x: normalize_label(strip_include_index(x)))
        hit = df[df["clean"].isin(targets)]
        for pl in hit["parent_label"].unique():
            cats.add(str(pl))
        log.info("INCLUDE categories needed for targets: %s", sorted(cats))
    except Exception as e:
        log.error("INCLUDE meta category resolve failed: %s", e)
    return cats


def build_from_include(targets: list[str]) -> list[VideoRecord]:
    """Scan extracted INCLUDE tree for matching words; download needed categories."""
    extract_root = RAW / "INCLUDE" / "extracted"
    for cat in sorted(include_categories_for_targets(targets)):
        # Skip download if category already extracted
        already = any(cat.lower() in str(p).lower() for p in extract_root.rglob("*")) if extract_root.exists() else False
        if not already:
            download_include_category(cat)
        else:
            log.info("INCLUDE category already present: %s", cat)

    if not extract_root.exists():
        log.warning("INCLUDE extracted root missing")
        return []

    ds = next(d for d in DATASETS if d.name == "INCLUDE")
    records: list[VideoRecord] = []
    for vid in iter_videos(extract_root):
        label = infer_label_from_path(vid, extract_root)
        matched, how = match_targets(label, targets)
        if not matched:
            continue
        meta = ffprobe_meta(vid)
        if not meta.get("ok"):
            log.info("Reject INCLUDE video %s: %s", vid, meta.get("error"))
            continue
        qs = quality_score(meta)
        if qs < 0.35:
            log.info("Reject low quality %s score=%s", vid, qs)
            continue
        records.append(
            VideoRecord(
                word=matched,
                normalized_word=matched,
                dataset="INCLUDE",
                original_label=label,
                video_path=str(vid),
                original_filename=vid.name,
                signer=f"include_{vid.stem}",
                split="",
                fps=str(meta.get("fps") or ""),
                resolution=f"{meta.get('width')}x{meta.get('height')}",
                duration=str(meta.get("duration") or ""),
                license=ds.license,
                paper=ds.paper,
                repository=ds.repository,
                download_url=ds.download_link,
                sha256=sha256_file(vid),
                phash=simple_phash(vid),
                quality_score=str(qs),
                duplicate_status="unique",
                review_status="accepted" if how == "exact" else "normalized_match",
            )
        )
        log.info("INCLUDE match %s <- %s (%s)", matched, vid.name, how)
    return records


def try_download_cislr_metadata() -> Optional[Path]:
    dest = RAW / "CISLR" / "dataset.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Exploration-Lab/CISLR",
            filename="dataset.csv",
            repo_type="dataset",
            local_dir=str(RAW / "CISLR"),
        )
        log.info("CISLR metadata: %s", path)
        return Path(path)
    except Exception as e:
        log.error("CISLR metadata download failed: %s", e)
        return None


def build_from_cislr(targets: list[str]) -> list[VideoRecord]:
    """Download CISLR video zip (~1.2GB) and extract clips matching target glosses."""
    import pandas as pd

    csv_path = try_download_cislr_metadata()
    if not csv_path:
        return []
    df = pd.read_csv(csv_path)
    df["n"] = df["gloss"].astype(str).map(normalize_label)
    matched = df[df["n"].isin(targets)].copy()
    # documented near-miss reviews (do NOT auto-merge)
    review_hits = []
    if "hello" in targets and matched[matched["n"] == "hello"].empty:
        namaste = df[df["n"] == "namaste"]
        for _, row in namaste.iterrows():
            review_hits.append(
                {
                    "target": "hello",
                    "candidate_gloss": row["gloss"],
                    "uid": row["uid"],
                    "reason": "Possible greeting relative (namaste); Needs Manual Review — not auto-merged",
                }
            )
    if review_hits:
        rev = REPORTS / "manual_review_candidates.csv"
        with rev.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(review_hits[0].keys()))
            w.writeheader()
            w.writerows(review_hits)
        log.info("Wrote manual review candidates: %s", rev)

    log.info("CISLR exact gloss matches: %s rows across %s words", len(matched), matched["n"].nunique() if len(matched) else 0)
    if matched.empty:
        return []

    try:
        from huggingface_hub import hf_hub_download

        zip_path = Path(
            hf_hub_download(
                repo_id="Exploration-Lab/CISLR",
                filename="CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip",
                repo_type="dataset",
                local_dir=str(RAW / "CISLR"),
            )
        )
    except Exception as e:
        log.error("CISLR zip download failed: %s", e)
        return []

    extract_dir = RAW / "CISLR" / "videos"
    extract_dir.mkdir(parents=True, exist_ok=True)
    needed = set()
    for uid in matched["uid"].astype(str):
        # uid may be youtubeid or youtubeid_1
        needed.add(f"{uid}.mp4")
        if "_" in uid:
            needed.add(f"{uid.split('_')[0]}.mp4")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            for name in names:
                base = Path(name).name
                if base in needed or any(base.startswith(u.split("_")[0]) for u in matched["uid"].astype(str)):
                    # tighter: extract if basename stem matches uid stem
                    stem = Path(base).stem
                    if any(stem == str(u) or stem == str(u).split("_")[0] for u in matched["uid"].astype(str)):
                        target = extract_dir / base
                        if not target.exists():
                            with zf.open(name) as src, target.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
    except Exception as e:
        log.error("CISLR extract failed: %s", e)
        return []

    ds = next(d for d in DATASETS if d.name == "CISLR")
    records: list[VideoRecord] = []
    for _, row in matched.iterrows():
        uid = str(row["uid"])
        candidates = [
            extract_dir / f"{uid}.mp4",
            extract_dir / f"{uid.split('_')[0]}.mp4",
        ]
        vid = next((p for p in candidates if p.exists()), None)
        if not vid:
            # search extracted
            hits = list(extract_dir.glob(f"{uid}*.mp4")) + list(extract_dir.glob(f"{uid.split('_')[0]}*.mp4"))
            vid = hits[0] if hits else None
        if not vid:
            log.error("CISLR video missing for uid=%s gloss=%s", uid, row["gloss"])
            continue
        meta = ffprobe_meta(vid)
        if not meta.get("ok"):
            log.info("Reject CISLR %s: %s", vid, meta.get("error"))
            continue
        word = normalize_label(row["gloss"])
        records.append(
            VideoRecord(
                word=word,
                normalized_word=word,
                dataset="CISLR",
                original_label=str(row["gloss"]),
                video_path=str(vid),
                original_filename=vid.name,
                signer=str(row.get("category") or "cislr"),
                split="",
                fps=str(meta.get("fps") or ""),
                resolution=f"{meta.get('width')}x{meta.get('height')}",
                duration=str(meta.get("duration") or row.get("duration") or ""),
                license=ds.license,
                paper=ds.paper,
                repository=ds.repository,
                download_url=ds.download_link,
                sha256=sha256_file(vid),
                phash=simple_phash(vid),
                quality_score=str(quality_score(meta)),
                duplicate_status="unique",
                review_status="accepted",
            )
        )
    return records


def try_download_isl500_for_words(targets: list[str], max_per_word: int = 40) -> list[VideoRecord]:
    """Selective HF download of ISL500 videos matching targets (Word__session__clip.mp4)."""
    records: list[VideoRecord] = []
    cache_list = CACHE / "isl500_video_files.txt"
    try:
        from huggingface_hub import HfApi, hf_hub_download

        if cache_list.exists():
            video_files = [ln for ln in cache_list.read_text(encoding="utf-8").splitlines() if ln]
            log.info("ISL500 file list from cache: %s", len(video_files))
        else:
            api = HfApi()
            log.info("Listing ISL500 files (may take time)...")
            files = api.list_repo_files("ISL500/ISL-DATA", repo_type="dataset")
            video_files = [f for f in files if "/Videos/" in f.replace("\\", "/") and f.endswith(".mp4")]
            if not video_files:
                video_files = [f for f in files if f.endswith(".mp4") and "Videos" in f]
            cache_list.write_text("\n".join(video_files), encoding="utf-8")
            log.info("ISL500 video files listed: %s", len(video_files))

        ds = next(d for d in DATASETS if d.name.startswith("ISL-DATA"))
        for t in targets:
            # Match Word__ at start of filename (case-insensitive), spaces as nothing or underscore
            variants = {
                t,
                t.replace(" ", "_"),
                t.replace(" ", ""),
                t.title().replace(" ", ""),
                "".join(w.capitalize() for w in t.split()),
            }
            candidates = []
            for f in video_files:
                name = Path(f).name
                stem0 = name.split("__")[0] if "__" in name else name.split("_")[0]
                if normalize_label(stem0) == t or stem0.lower() in {v.lower() for v in variants}:
                    candidates.append(f)
            log.info("ISL500 candidates for '%s': %s", t, len(candidates))
            for f in candidates[:max_per_word]:
                try:
                    local = hf_hub_download(
                        repo_id="ISL500/ISL-DATA",
                        filename=f,
                        repo_type="dataset",
                        local_dir=str(RAW / "ISL500"),
                    )
                    vid = Path(local)
                    meta = ffprobe_meta(vid)
                    if not meta.get("ok"):
                        continue
                    qs = quality_score(meta)
                    if qs < 0.35:
                        continue
                    user = "unknown"
                    for part in Path(f).parts:
                        if "user" in part.lower():
                            user = part
                    records.append(
                        VideoRecord(
                            word=t,
                            normalized_word=t,
                            dataset="ISL500",
                            original_label=Path(f).name.split("__")[0],
                            video_path=str(vid),
                            original_filename=Path(f).name,
                            signer=user,
                            split="",
                            fps=str(meta.get("fps") or ""),
                            resolution=f"{meta.get('width')}x{meta.get('height')}",
                            duration=str(meta.get("duration") or ""),
                            license=ds.license,
                            paper=ds.paper,
                            repository=ds.repository,
                            download_url=ds.download_link,
                            sha256=sha256_file(vid),
                            phash=simple_phash(vid),
                            quality_score=str(qs),
                            duplicate_status="unique",
                            review_status="accepted",
                        )
                    )
                except Exception as e:
                    log.error("ISL500 file %s failed: %s", f, e)
    except Exception as e:
        log.error("ISL500 download path failed: %s", e)
    return records


def deduplicate(records: list[VideoRecord]) -> tuple[list[VideoRecord], list[dict]]:
    """Remove exact SHA duplicates and near-identical phash (same signer context)."""
    by_sha: dict[str, VideoRecord] = {}
    dup_rows: list[dict] = []
    kept: list[VideoRecord] = []
    for r in records:
        if r.sha256 in by_sha:
            r.duplicate_status = "exact_sha_duplicate"
            prev = by_sha[r.sha256]
            dup_rows.append(
                {
                    "kept": prev.video_path,
                    "removed": r.video_path,
                    "reason": "sha256",
                    "word": r.word,
                }
            )
            continue
        by_sha[r.sha256] = r
        # phash near-dup against kept same word
        is_near = False
        for k in kept:
            if k.word != r.word:
                continue
            if r.phash and k.phash and hamming(r.phash, k.phash) <= 3:
                # Only drop if same dataset AND same signer — otherwise keep diversity
                if r.dataset == k.dataset and r.signer == k.signer:
                    r.duplicate_status = "near_phash_duplicate"
                    dup_rows.append(
                        {
                            "kept": k.video_path,
                            "removed": r.video_path,
                            "reason": "phash<=3 same signer",
                            "word": r.word,
                        }
                    )
                    is_near = True
                    break
        if not is_near:
            kept.append(r)
    return kept, dup_rows


def materialize_dataset(records: list[VideoRecord]) -> list[VideoRecord]:
    """Copy/normalize into ISL_DATASET/<word>/ and rewrite paths."""
    out_records: list[VideoRecord] = []
    transform_log = LOGS / "normalization_transforms.jsonl"
    for i, r in enumerate(records):
        src = Path(r.video_path)
        if not src.exists():
            log.error("Missing source video: %s", src)
            continue
        word_dir = OUT / folder_name(r.word)
        word_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{folder_name(r.word)}__{r.dataset}__{i:05d}__{src.stem}.mp4"
        dest = word_dir / dest_name
        transform = normalize_video(src, dest)
        with transform_log.open("a", encoding="utf-8") as tf:
            tf.write(json.dumps(transform) + "\n")
        # provenance sidecar
        side = dest.with_suffix(".json")
        side.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")
        meta = ffprobe_meta(dest if dest.exists() else src)
        final_path = dest if dest.exists() else src
        r2 = VideoRecord(**{**asdict(r), "video_path": str(final_path.relative_to(ROOT))})
        if meta.get("ok"):
            r2.fps = str(meta.get("fps") or r2.fps)
            r2.resolution = f"{meta.get('width')}x{meta.get('height')}"
            r2.duration = str(meta.get("duration") or r2.duration)
            r2.sha256 = sha256_file(final_path)
            r2.phash = simple_phash(final_path) or r2.phash
        out_records.append(r2)
        # per-word source info
        (word_dir / "sources.txt").write_text(
            "\n".join(sorted({x.dataset for x in out_records if x.word == r.word})),
            encoding="utf-8",
        )
    return out_records


def write_metadata(records: list[VideoRecord]) -> Path:
    path = OUT / "metadata.csv"
    fields = list(asdict(records[0]).keys()) if records else list(VideoRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    # also copy to reports
    shutil.copy2(path, REPORTS / "metadata.csv")
    return path


def write_reports(
    targets: list[str],
    records: list[VideoRecord],
    dup_rows: list[dict],
    searched: int,
    downloaded: int,
    skipped: list[str],
) -> None:
    # dataset_statistics
    stats_path = REPORTS / "dataset_statistics.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["target_words", len(targets)])
        w.writerow(["accepted_videos", len(records)])
        w.writerow(["unique_signers", len({r.signer for r in records})])
        w.writerow(["source_datasets", len({r.dataset for r in records})])
        w.writerow(["datasets_searched", searched])
        w.writerow(["datasets_downloaded", downloaded])

    # duplicates
    with (REPORTS / "duplicate_report.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["kept", "removed", "reason", "word"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in dup_rows:
            w.writerow(row)

    # missing words
    with (REPORTS / "missing_words.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "videos_found", "status"])
        for t in targets:
            n = sum(1 for r in records if r.word == t)
            status = "ok" if n >= 40 else ("partial" if n > 0 else "missing")
            w.writerow([t, n, status])

    # normalization map
    with (REPORTS / "normalization_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_label", "normalized_word", "dataset", "review_status"])
        seen = set()
        for r in records:
            key = (r.original_label, r.normalized_word, r.dataset)
            if key in seen:
                continue
            seen.add(key)
            w.writerow([r.original_label, r.normalized_word, r.dataset, r.review_status])

    # per-word coverage + summary.md
    lines = [
        f"# ISL Dataset Collection Summary",
        f"",
        f"Generated: {utc_now()}",
        f"",
        f"## Global",
        f"",
        f"- Datasets searched (inventory): **{searched}**",
        f"- Datasets with successful local downloads this run: **{downloaded}**",
        f"- Datasets skipped / restricted: {', '.join(skipped) or 'none'}",
        f"- Videos accepted: **{len(records)}**",
        f"- Duplicates removed: **{len(dup_rows)}**",
        f"- Target words: **{len(targets)}**",
        f"",
        f"## Word coverage",
        f"",
    ]
    for t in targets:
        subset = [r for r in records if r.word == t]
        signers = sorted({r.signer for r in subset})
        sources = sorted({r.dataset for r in subset})
        lines += [
            f"### {t}",
            f"",
            f"- Datasets searched: {searched}",
            f"- Videos found/accepted: {len(subset)}",
            f"- Unique signers (as labeled): {len(signers)}",
            f"- Sources: {', '.join(sources) or 'none'}",
            f"- Target (>=40): {'YES' if len(subset) >= 40 else 'NO'}",
            f"",
        ]
    lines += [
        f"## License notes",
        f"",
        f"See `reports/license_report.csv`. Research-only datasets must not be redistributed commercially.",
        f"INCLUDE is CC-BY-4.0. iSign/CISLR: research use. ISL-50/52: academic only.",
        f"",
        f"## Reproducibility",
        f"",
        f"Action log: `logs/agent_actions.log`",
        f"Transforms: `logs/normalization_transforms.jsonl`",
        f"Config: `config/target_words.json`",
        f"",
    ]
    (REPORTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Reports written under %s", REPORTS)


def main() -> None:
    log.info("=== ISL Dataset Agent start %s ===", utc_now())
    write_inventory()
    targets = load_target_words()
    log.info("Target words (%s): %s", len(targets), targets)

    skipped = [d.name for d in DATASETS if d.downloadable == "request" or d.status == "restricted"]
    all_records: list[VideoRecord] = []

    # 1) INCLUDE
    try:
        recs = build_from_include(targets)
        all_records.extend(recs)
        log.info("INCLUDE contributed %s videos", len(recs))
    except Exception as e:
        log.error("INCLUDE pipeline error: %s", e)

    # 2) CISLR
    try:
        recs = build_from_cislr(targets)
        all_records.extend(recs)
        log.info("CISLR contributed %s videos", len(recs))
    except Exception as e:
        log.error("CISLR pipeline error: %s", e)

    # 3) ISL500 selective
    try:
        recs = try_download_isl500_for_words(targets)
        all_records.extend(recs)
        log.info("ISL500 contributed %s videos", len(recs))
    except Exception as e:
        log.error("ISL500 pipeline error: %s", e)

    kept, dups = deduplicate(all_records)
    log.info("After dedup: %s kept, %s removed", len(kept), len(dups))
    final = materialize_dataset(kept)
    write_metadata(final)
    downloaded = len({r.dataset for r in final})
    write_reports(
        targets,
        final,
        dups,
        searched=len(DATASETS),
        downloaded=downloaded,
        skipped=skipped,
    )
    log.info("=== Done. Accepted videos: %s ===", len(final))
    print(f"\nOutput: {OUT}")
    print(f"Summary: {REPORTS / 'summary.md'}")


if __name__ == "__main__":
    main()

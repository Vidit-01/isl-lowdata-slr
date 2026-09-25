"""Fast local materialization: scan already-downloaded raw videos, skip network."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from isl_dataset_agent import (  # noqa: E402
    DATASETS,
    OUT,
    RAW,
    REPORTS,
    VideoRecord,
    deduplicate,
    ffprobe_meta,
    folder_name,
    infer_label_from_path,
    iter_videos,
    load_target_words,
    log,
    match_targets,
    materialize_dataset,
    normalize_label,
    quality_score,
    sha256_file,
    simple_phash,
    write_inventory,
    write_metadata,
    write_reports,
)


def scan_cislr(targets: list[str]) -> list[VideoRecord]:
    import pandas as pd

    csv_path = RAW / "CISLR" / "dataset.csv"
    vid_dir = RAW / "CISLR" / "videos"
    if not csv_path.exists() or not vid_dir.exists():
        return []
    df = pd.read_csv(csv_path)
    df["n"] = df["gloss"].astype(str).map(normalize_label)
    matched = df[df["n"].isin(targets)]
    ds = next(d for d in DATASETS if d.name == "CISLR")
    records: list[VideoRecord] = []
    for _, row in matched.iterrows():
        uid = str(row["uid"])
        candidates = [
            vid_dir / f"{uid}.mp4",
            vid_dir / f"{uid.split('_')[0]}.mp4",
        ]
        hits = list(vid_dir.glob(f"{uid}*.mp4")) + list(
            vid_dir.glob(f"{uid.split('_')[0]}*.mp4")
        )
        vid = next((p for p in candidates if p.exists()), None) or (hits[0] if hits else None)
        if not vid:
            continue
        meta = ffprobe_meta(vid)
        if not meta.get("ok"):
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
                duration=str(meta.get("duration") or ""),
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


def scan_isl500(targets: list[str]) -> list[VideoRecord]:
    root = RAW / "ISL500"
    if not root.exists():
        return []
    ds = next(d for d in DATASETS if d.name.startswith("ISL-DATA"))
    records: list[VideoRecord] = []
    for vid in iter_videos(root):
        label = infer_label_from_path(vid, root)
        matched, how = match_targets(label, targets)
        if not matched:
            continue
        meta = ffprobe_meta(vid)
        if not meta.get("ok"):
            continue
        qs = quality_score(meta)
        if qs < 0.35:
            continue
        user = "unknown"
        for part in vid.parts:
            if "user" in part.lower():
                user = part
        records.append(
            VideoRecord(
                word=matched,
                normalized_word=matched,
                dataset="ISL500",
                original_label=label,
                video_path=str(vid),
                original_filename=vid.name,
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
                review_status="accepted" if how == "exact" else "normalized_match",
            )
        )
    return records


# High-likelihood INCLUDE label bridges used only to fill sparse classes.
# Kept as Needs Manual Review (<90% documented same-sign certainty).
INCLUDE_REVIEW_ALIASES = {
    "i": "me",
    "house": "home",
    "alright": "okay",
}

# ISLRTC dictionary filename stem -> target word
ISLRTC_STEM_MAP = {
    "bye_goodbye": "goodbye",
    "bye": "goodbye",
    "goodbye": "goodbye",
    "sorry": "sorry",
    "sorry_(sign_2)": "sorry",
    "sorry_(sign_3)": "sorry",
    "stop": "stop",
    "cease_discontinue_halt_stop": "stop",
    "okay": "okay",
    "okay_(sign_2)": "okay",
    "come": "come",
    "come_(sign_2)": "come",
    "stand": "stand",
    "read": "read",
    "write": "write",
    "when_(for_days)": "when",
    "myself": "me",  # review-level; dictionary self-reference
}


def scan_include(targets: list[str]) -> list[VideoRecord]:
    root = RAW / "INCLUDE" / "extracted"
    if not root.exists():
        return []
    ds = next(d for d in DATASETS if d.name == "INCLUDE")
    records: list[VideoRecord] = []
    for vid in iter_videos(root):
        label = infer_label_from_path(vid, root)
        matched, how = match_targets(label, targets)
        review = "accepted" if how == "exact" else "normalized_match"
        if not matched:
            alias = INCLUDE_REVIEW_ALIASES.get(label)
            if alias and alias in targets:
                matched = alias
                how = "review_alias"
                review = "Needs Manual Review"
            else:
                continue
        meta = ffprobe_meta(vid)
        if not meta.get("ok"):
            continue
        qs = quality_score(meta)
        if qs < 0.35:
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
                review_status=review,
            )
        )
    return records


def scan_islrtc_dict(targets: list[str]) -> list[VideoRecord]:
    root = RAW / "ISLRTC_dict"
    if not root.exists():
        return []
    records: list[VideoRecord] = []
    for vid in iter_videos(root):
        stem = vid.stem.lower()
        word = ISLRTC_STEM_MAP.get(stem)
        if not word or word not in targets:
            continue
        meta = ffprobe_meta(vid)
        if not meta.get("ok"):
            continue
        qs = quality_score(meta)
        if qs < 0.35:
            continue
        review = (
            "Needs Manual Review"
            if stem in {"myself", "bye", "cease_discontinue_halt_stop", "when_(for_days)"}
            else "accepted"
        )
        records.append(
            VideoRecord(
                word=word,
                normalized_word=word,
                dataset="ISLRTC_dictionary",
                original_label=vid.stem,
                video_path=str(vid),
                original_filename=vid.name,
                signer="islrtc_dict",
                split="",
                fps=str(meta.get("fps") or ""),
                resolution=f"{meta.get('width')}x{meta.get('height')}",
                duration=str(meta.get("duration") or ""),
                license="Government open data / check data.gov.in",
                paper="",
                repository="https://huggingface.co/datasets/Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded",
                download_url="https://www.data.gov.in/resource/indian-sign-language-dictionary-till-january-2024",
                sha256=sha256_file(vid),
                phash=simple_phash(vid),
                quality_score=str(qs),
                duplicate_status="unique",
                review_status=review,
            )
        )
    return records


def main() -> None:
    write_inventory()
    targets = load_target_words()
    log.info("Local finalize targets=%s", len(targets))

    all_records: list[VideoRecord] = []
    for name, fn in (
        ("CISLR", scan_cislr),
        ("ISL500", scan_isl500),
        ("INCLUDE", scan_include),
        ("ISLRTC_dictionary", scan_islrtc_dict),
    ):
        recs = fn(targets)
        log.info("%s local scan: %s videos", name, len(recs))
        all_records.extend(recs)

    kept, dups = deduplicate(all_records)
    log.info("After dedup: %s kept, %s dups", len(kept), len(dups))

    # Rebuild output folders cleanly so rematerialize does not accumulate orphans
    import shutil

    if OUT.exists():
        for p in list(OUT.iterdir()):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.name == "metadata.csv":
                p.unlink(missing_ok=True)

    final = materialize_dataset(kept)
    write_metadata(final)
    write_reports(
        targets,
        final,
        dups,
        searched=len(DATASETS),
        downloaded=len({r.dataset for r in final}),
        skipped=["IIITA-ROBITA ISL Gesture Database"],
    )
    print(f"FINAL={len(final)} OUT={OUT} SUMMARY={REPORTS / 'summary.md'}")


if __name__ == "__main__":
    main()

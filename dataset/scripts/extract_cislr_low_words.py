"""Extract all CISLR videos for low-coverage target words."""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOW = [
    "goodbye",
    "sorry",
    "stop",
    "okay",
    "me",
    "brother",
    "home",
    "come",
    "stand",
    "read",
    "write",
    "when",
]


def norm(t: str) -> str:
    t = str(t).strip().lower().replace("_", " ").replace("-", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    df = pd.read_csv(ROOT / "raw_datasets" / "CISLR" / "dataset.csv")
    df["n"] = df["gloss"].astype(str).map(norm)
    matched = df[df["n"].isin(LOW)]
    zip_path = (
        ROOT
        / "raw_datasets"
        / "CISLR"
        / "CISLR_v1.5-a_videos"
        / "CISLR_v1.5-a_videos.zip"
    )
    extract = ROOT / "raw_datasets" / "CISLR" / "videos"
    extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        basenames = {Path(n).name: n for n in zf.namelist()}
        for _, row in matched.iterrows():
            uid = str(row["uid"])
            stems = {uid, uid.split("_")[0]}
            found = None
            for bn, full in basenames.items():
                stem = Path(bn).stem
                if stem in stems or any(stem.startswith(s) for s in stems):
                    found = full
                    break
            if not found:
                print("MISSING_IN_ZIP", uid, row["gloss"])
                continue
            dest = extract / Path(found).name
            if not dest.exists() or dest.stat().st_size == 0:
                with zf.open(found) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print("EXTRACTED", dest.name, row["gloss"])
            else:
                print("HAVE", dest.name, row["gloss"])


if __name__ == "__main__":
    main()

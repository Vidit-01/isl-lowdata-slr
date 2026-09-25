"""Download INCLUDE category zips via curl (resumable) and extract matching videos."""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_datasets" / "INCLUDE"
CACHE = ROOT / "cache" / "zenodo_4010759.json"
RAW.mkdir(parents=True, exist_ok=True)

# Categories that contain exact matches for current target_words.json
PRIORITY = [
    "Greetings",
    "Pronouns",
    "People",
    "Jobs",
    "Places",
    "Days_and_Time",
]


def zenodo_files() -> list[dict]:
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    return data["files"]


def curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe",
        "-L",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--continue-at",
        "-",
        "-o",
        str(dest),
        url,
    ]
    print("RUN", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    files = zenodo_files()
    extract_to = RAW / "extracted"
    extract_to.mkdir(exist_ok=True)
    for cat in PRIORITY:
        matched = [
            f for f in files if f["key"].startswith(cat) and f["key"].endswith(".zip")
        ]
        print(f"=== {cat}: {len(matched)} parts ===")
        for fmeta in matched:
            key = fmeta["key"]
            dest = RAW / key
            expected = int(fmeta["size"])
            if dest.exists() and dest.stat().st_size == expected:
                print(f"complete {key}")
            else:
                curl_download(fmeta["links"]["self"], dest)
            print(f"extract {key}")
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(extract_to)
    print("DONE INCLUDE priority categories")


if __name__ == "__main__":
    main()

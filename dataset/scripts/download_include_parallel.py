"""Download INCLUDE priority zips in parallel via curl (skip complete files)."""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_datasets" / "INCLUDE"
CACHE = ROOT / "cache" / "zenodo_4010759.json"
RAW.mkdir(parents=True, exist_ok=True)

PRIORITY_PREFIXES = (
    "Greetings",
    "Pronouns",
    "People",
    "Jobs",
    "Places",
    "Days_and_Time",
)
WORKERS = 3


def main() -> None:
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    jobs = []
    for f in data["files"]:
        key = f["key"]
        if not key.endswith(".zip"):
            continue
        if not key.startswith(PRIORITY_PREFIXES):
            continue
        dest = RAW / key
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"SKIP complete {key}")
            continue
        jobs.append((key, f["links"]["self"], dest, f["size"]))

    print(f"QUEUE {len(jobs)} downloads with {WORKERS} workers")

    def download(item):
        key, url, dest, expected = item
        print(f"START {key}")
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
        t0 = time.time()
        subprocess.check_call(cmd)
        sz = dest.stat().st_size if dest.exists() else -1
        ok = sz == expected
        print(f"{'OK' if ok else 'FAIL'} {key} size={sz} expected={expected} secs={time.time()-t0:.0f}")
        return key, ok

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download, j) for j in jobs]
        for fut in as_completed(futs):
            key, ok = fut.result()
            if not ok:
                failed.append(key)
    print("DONE parallel downloads; failed=", failed)


if __name__ == "__main__":
    main()

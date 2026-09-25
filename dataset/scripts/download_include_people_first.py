"""Download only missing high-value INCLUDE zips (People, Jobs, Days, remaining Places/Pronouns)."""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_datasets" / "INCLUDE"
CACHE = ROOT / "cache" / "zenodo_4010759.json"
WORKERS = 4

# Prefer People first (brother/father hit 40+), then Jobs, Days, remaining
ORDER_PREFIX = ("People", "Jobs", "Days_and_Time", "Pronouns", "Places")


def main() -> None:
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    by_prefix = {p: [] for p in ORDER_PREFIX}
    for f in data["files"]:
        key = f["key"]
        if not key.endswith(".zip"):
            continue
        dest = RAW / key
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"SKIP {key}")
            continue
        for p in ORDER_PREFIX:
            if key.startswith(p):
                by_prefix[p].append((key, f["links"]["self"], dest, f["size"]))
                break
    jobs = []
    for p in ORDER_PREFIX:
        jobs.extend(by_prefix[p])
    print(f"QUEUE {len(jobs)}")

    def download(item):
        key, url, dest, expected = item
        print(f"START {key}")
        cmd = [
            "curl.exe",
            "-L",
            "--retry",
            "8",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "-o",
            str(dest),
            url,
        ]
        t0 = time.time()
        subprocess.check_call(cmd)
        sz = dest.stat().st_size if dest.exists() else -1
        ok = sz == expected
        print(f"{'OK' if ok else 'FAIL'} {key} {sz}/{expected} {time.time()-t0:.0f}s")
        return key, ok

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download, j) for j in jobs]
        for fut in as_completed(futs):
            key, ok = fut.result()
            if not ok:
                failed.append(key)
    print("DONE failed=", failed)


if __name__ == "__main__":
    main()

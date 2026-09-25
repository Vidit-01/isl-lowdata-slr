"""Watch INCLUDE People zips; extract each as it completes (validate before extract)."""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_datasets" / "INCLUDE"
CACHE = ROOT / "cache" / "zenodo_4010759.json"
EXTRACT = RAW / "extracted"


def main() -> None:
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    wanted = {
        f["key"]: f
        for f in data["files"]
        if f["key"].startswith("People") and f["key"].endswith(".zip")
    }
    print("watching", sorted(wanted), flush=True)
    EXTRACT.mkdir(exist_ok=True)
    while True:
        done = []
        for key, meta in wanted.items():
            dest = RAW / key
            expected = meta["size"]
            marker = RAW / f".extracted_{key}"
            bad_marker = RAW / f".corrupt_{key}"
            if bad_marker.exists():
                print(f"  {key}: marked corrupt — needs clean re-download", flush=True)
                continue
            if dest.exists() and dest.stat().st_size == expected:
                if marker.exists():
                    done.append(key)
                    continue
                print("validate", key, flush=True)
                try:
                    with zipfile.ZipFile(dest) as zf:
                        bad = zf.testzip()
                        if bad is not None:
                            raise zipfile.BadZipFile(f"CRC fail on {bad}")
                        print("extract", key, flush=True)
                        out = EXTRACT / Path(key).stem
                        out.mkdir(parents=True, exist_ok=True)
                        zf.extractall(out)
                    (out / ".done").write_text("ok", encoding="utf-8")
                    marker.write_text("ok", encoding="utf-8")
                    done.append(key)
                except Exception as e:
                    print(f"CORRUPT {key}: {e}", flush=True)
                    bad_marker.write_text(str(e), encoding="utf-8")
                    try:
                        dest.unlink()
                        print(f"deleted corrupt {key}", flush=True)
                    except OSError as oe:
                        print(f"could not delete {key}: {oe}", flush=True)
            else:
                sz = dest.stat().st_size if dest.exists() else 0
                pct = 100.0 * sz / expected if expected else 0
                print(f"  {key}: {sz}/{expected} ({pct:.1f}%)", flush=True)
        print(f"complete_parts {len(done)}/{len(wanted)}", flush=True)
        if len(done) >= 3:
            print("ENOUGH_PEOPLE", flush=True)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()

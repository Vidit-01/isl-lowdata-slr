"""Download ISL_DATASET from Hugging Face Hub into the project root."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo",
        default="vidit031/isl-isolated-40words",
        help="HF dataset repo (8-word test set: vidit031/isl-isolated-8words)",
    )
    ap.add_argument("--out", default=str(ROOT / "ISL_DATASET"))
    ap.add_argument(
        "--8words",
        action="store_true",
        help="Shortcut: download vidit031/isl-isolated-8words into ISL_DATASET/",
    )
    ap.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN environment variable)",
    )
    args = ap.parse_args()
    if args.__dict__["8words"]:
        args.repo = "vidit031/isl-isolated-8words"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(out),
        token=args.token,
    )
    try:
        path = snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        path = snapshot_download(**kwargs)

    print("Downloaded to", path)
    n = sum(1 for _ in out.rglob("*.mp4"))
    print(f"mp4 count={n} metadata={(out / 'metadata.csv').exists()}")


if __name__ == "__main__":
    main()
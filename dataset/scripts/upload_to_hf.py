"""Upload ISL_DATASET/ to Hugging Face Datasets Hub.

Requires a WRITE-scoped HF token:
  hf auth login
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "ISL_DATASET"
DEFAULT_REPO = "vidit031/isl-isolated-40words"


def prepare_metadata(dataset_dir: Path) -> Path:
    meta = dataset_dir / "metadata.csv"
    df = pd.read_csv(meta)
    df["video_path"] = (
        df["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace(r"^.*?ISL_DATASET/", "", regex=True)
        .str.replace(r"^ISL_DATASET/", "", regex=True)
    )
    # Ensure paths exist relative to dataset root
    missing = [p for p in df["video_path"] if not (dataset_dir / p).is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} video files; e.g. {missing[:3]}")
    out = dataset_dir / "metadata.csv"
    df.to_csv(out, index=False)
    print(f"Prepared metadata: {len(df)} rows")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Local dataset folder to upload (default: ISL_DATASET/)",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="Write-scoped HF token (optional if logged in)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Missing {dataset_dir}")
    if not (dataset_dir / "README.md").is_file():
        raise SystemExit(f"Missing {dataset_dir / 'README.md'} dataset card")

    prepare_metadata(dataset_dir)

    api = HfApi(token=args.token)
    info = whoami(token=args.token)
    print(f"Logged in as: {info.get('name')}")
    auth = info.get("auth") or {}
    access = (auth.get("accessToken") or {})
    role = access.get("role")
    if role == "read":
        print(
            "ERROR: current Hugging Face token is read-only.\n"
            "Create a Write token at https://huggingface.co/settings/tokens\n"
            "then run:  hf auth login\n"
            "or:        python src/upload_to_hf.py --token hf_...",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
    except HfHubHTTPError as e:
        raise SystemExit(f"create_repo failed: {e}") from e

    print(f"Uploading {dataset_dir} -> {args.repo_id} ...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        print_report_every=30,
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

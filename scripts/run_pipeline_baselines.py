"""Cloud one-shot: download the 40-word ISL set + train/eval arch.md baselines.

On Lightning / RunPod / Colab (from the git checkout):

  git pull origin main
  python scripts/run_pipeline_baselines.py --skip-clone

That downloads vidit031/isl-isolated-40words (~642 clips), extracts landmarks,
keeps the 8 highest-count glosses, and trains the few-shot protocol
(7-shot train, 15-clip locked test, 3 train-set draws).

  python scripts/run_pipeline_baselines.py --skip-clone --models kdf_transformer
  python scripts/run_pipeline_baselines.py --skip-clone --smoke   # 3-epoch GPU check

Do not use WORKDIR — Lightning sets that to a folder that is not this checkout.
Do not --skip-download unless the 40-word folder already has ~642 mp4s.
The old 56-clip ISL_DATASET is ignored on purpose.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_CANDIDATE = Path(__file__).resolve().parents[1]
TRAIN_PY = Path("islr") / "fewshot" / "train.py"
EXTRACT_PY = Path("islr") / "fewshot" / "extract_landmarks.py"
EVAL_PY = Path("islr") / "fewshot" / "eval.py"
HF_40 = "vidit031/isl-isolated-40words"
LIGHTNING_CONTENT = Path("/home/zeus/content")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def is_checkout(path: Path) -> bool:
    return (path / TRAIN_PY).is_file() and (path / EXTRACT_PY).is_file()


def _mp4_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.mp4"))


def expected_min_mp4(hf_dataset: str) -> int:
    if "8words" in (hf_dataset or ""):
        return 40
    return 400


def _snapshot_download(repo: str, out: Path, token: str | None) -> None:
    import time

    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(6):
        kwargs = dict(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(out),
            token=token,
            max_workers=2,
            resume_download=True,
        )
        try:
            try:
                snapshot_download(**kwargs, local_dir_use_symlinks=False)
            except TypeError:
                kwargs.pop("resume_download", None)
                try:
                    snapshot_download(**kwargs)
                except TypeError:
                    snapshot_download(
                        repo_id=repo,
                        repo_type="dataset",
                        local_dir=str(out),
                        token=token,
                    )
            return
        except Exception as exc:  # noqa: BLE001 — Hub client versions differ
            last = exc
            text = str(exc)
            rate = "429" in text or "rate limit" in text.lower() or "too many requests" in text.lower()
            if not rate:
                raise
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            wait = min(90, 8 * (2 ** attempt))
            print(
                f"Hugging Face 429 (attempt {attempt + 1}/6). "
                f"Waiting {wait}s then resuming with Xet disabled...",
                flush=True,
            )
            time.sleep(wait)
    raise last  # type: ignore[misc]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download 40-word ISL set and train arch.md few-shot baselines"
    )
    ap.add_argument("--repo-url", default=os.getenv("REPO_URL", "https://github.com/Vidit-01/isl-arch-islr.models.git"))
    ap.add_argument("--branch", default=os.getenv("GIT_BRANCH", "main"), help="Git branch to clone")
    ap.add_argument(
        "--workdir",
        default=os.getenv("ISL_WORKDIR", ""),
        help="Clone destination if this script is not already inside the repo. "
        "Ignored when islr/fewshot/train.py is next to this scripts/ folder. "
        "Do not pass Lightning's WORKDIR.",
    )
    ap.add_argument("--hf-dataset", default=os.getenv("HF_DATASET", HF_40))
    ap.add_argument(
        "--data-dir",
        default=os.getenv("ISL_DATA_DIR", "ISL_DATASET_40WORDS"),
        help="Where to put the 40-word clips. Defaults to ISL_DATASET_40WORDS so the "
        "old 56-clip ISL_DATASET is not reused.",
    )
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-clone", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-landmarks", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, skip RGB CNN, single train draw")
    ap.add_argument(
        "--draws",
        type=int,
        default=3,
        help="k-shot train draws (default 3 on the 40-word set; smoke forces 1)",
    )
    ap.add_argument("--train-shots", type=int, default=7)
    ap.add_argument("--val-shots", type=int, default=1)
    ap.add_argument("--test-per-class", type=int, default=15, help="Locked test clips per word")
    ap.add_argument("--n-words", type=int, default=8, help="Keep this many highest-count classes")
    ap.add_argument(
        "--words",
        nargs="+",
        default=None,
        help="Override class list. Default: top --n-words by clip count. 'all' or 'legacy8' also valid.",
    )
    ap.add_argument("--strict-protocol", action="store_true")
    ap.add_argument("--git-pull", action="store_true", default=True, help="git pull when already cloned")
    ap.add_argument("--no-git-pull", action="store_false", dest="git_pull")
    ap.add_argument("--install-deps", action="store_true", default=True)
    ap.add_argument("--no-install-deps", action="store_false", dest="install_deps")
    return ap.parse_args()


def resolve_repo_root(args: argparse.Namespace) -> Path:
    """Prefer the checkout that contains this file, never Lightning's WORKDIR."""
    if is_checkout(ROOT_CANDIDATE):
        return ROOT_CANDIDATE
    cwd = Path.cwd()
    if is_checkout(cwd):
        return cwd
    explicit = (args.workdir or "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if is_checkout(p):
            return p
        nested = p / "isl-lowdata-slr"
        if is_checkout(nested):
            return nested
        return nested if p.name != "isl-lowdata-slr" else p
    return Path(os.getenv("HOME", str(Path.cwd()))) / "isl-lowdata-slr"


def resolve_data_dir(args: argparse.Namespace, repo_root: Path, min_mp4: int) -> Path:
    """Pick a 40-word folder. Ignore leftover 8-word / 7-cap ISL_DATASET copies."""
    raw = Path(args.data_dir).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        (repo_root / raw).resolve(),
        (Path.cwd() / raw).resolve(),
        LIGHTNING_CONTENT / raw.name,
        Path.home() / "content" / raw.name,
        Path("/teamspace/studios/this_studio") / raw.name,
    ]
    if LIGHTNING_CONTENT.is_dir():
        candidates.insert(0, (LIGHTNING_CONTENT / raw.name).resolve())

    for path in candidates:
        n = _mp4_count(path)
        if (path / "metadata.csv").exists() and n >= min_mp4:
            print(f"reusing dataset at {path} ({n} mp4)")
            return path
        if (path / "metadata.csv").exists():
            print(f"ignoring undersized dataset at {path} ({n} mp4, need >={min_mp4})")

    if LIGHTNING_CONTENT.is_dir():
        dest = (LIGHTNING_CONTENT / raw.name).resolve()
        print(f"Lightning: downloading into {dest}")
        return dest
    return candidates[0]


def git_pull_checkout(repo_root: Path, branch: str) -> None:
    try:
        _run(["git", "-C", str(repo_root), "pull", "--ff-only"])
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: git pull failed ({exc}); continuing with local checkout")


def clone_if_needed(repo_url: str, repo_root: Path, skip: bool, branch: str = "") -> None:
    if is_checkout(repo_root):
        print(f"using existing checkout {repo_root}")
        return
    if skip:
        raise SystemExit(
            f"SKIP clone but {repo_root} is not an isl-lowdata-slr checkout "
            f"(missing {repo_root / TRAIN_PY})"
        )
    print(f"Cloning {repo_url} -> {repo_root}")
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    branch = (branch or os.getenv("GIT_BRANCH", "")).strip()
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(repo_root)]
    _run(cmd)
    if not is_checkout(repo_root):
        raise SystemExit(f"clone finished but {repo_root / EXTRACT_PY} is missing")


def install_deps(repo_root: Path) -> None:
    py = sys.executable
    _run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    req_root = repo_root / "requirements.txt"
    if req_root.exists():
        _run([py, "-m", "pip", "install", "-r", str(req_root)])
    _run([py, "-m", "pip", "install", "-U", "huggingface_hub"])


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo_root = resolve_repo_root(args)
    already_cloned = is_checkout(repo_root)
    clone_if_needed(args.repo_url, repo_root, args.skip_clone, branch=args.branch)
    if args.git_pull and already_cloned:
        git_pull_checkout(repo_root, args.branch)
    os.chdir(repo_root)
    print(f"REPO_ROOT={repo_root}")

    extract = repo_root / EXTRACT_PY
    train = repo_root / TRAIN_PY
    if not extract.is_file() or not train.is_file():
        raise SystemExit(
            f"repo root is wrong: {repo_root}\n"
            f"  extract exists={extract.is_file()} ({extract})\n"
            f"  train exists={train.is_file()} ({train})\n"
            f"Lightning WORKDIR is ignored; run from the isl-lowdata-slr clone."
        )

    if args.install_deps:
        install_deps(repo_root)

    py = sys.executable
    min_mp4 = expected_min_mp4(args.hf_dataset)
    data_dir = resolve_data_dir(args, repo_root, min_mp4)
    token = os.getenv("HF_TOKEN")

    if not args.skip_download:
        print(f"=== Hugging Face dataset {args.hf_dataset} -> {data_dir} ===")
        _snapshot_download(args.hf_dataset, data_dir, token)
    meta = data_dir / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"missing {meta} — download failed or --data-dir is wrong")
    n_mp4 = _mp4_count(data_dir)
    print(f"mp4 count={n_mp4}  metadata={meta.exists()}  data_dir={data_dir}")
    if n_mp4 < min_mp4:
        raise SystemExit(
            f"{data_dir} has only {n_mp4} mp4s (need >={min_mp4} for {args.hf_dataset}).\n"
            f"Do not --skip-download over the old 8-word folder. Re-run without "
            f"--skip-download, or pass --data-dir ISL_DATASET_40WORDS."
        )

    models = list(args.models)
    epochs = args.epochs
    draws = 1 if args.smoke else int(args.draws)
    if args.smoke:
        epochs = epochs or 3
        if models == ["all"]:
            models = [
                "mp_bilstm",
                "mp_transformer",
                "stgcn",
                "ctr_gcn",
                "td_gcn",
                "hwgat",
                "fft_bilstm",
                "cwt_bilstm",
                "cwt_transformer",
                "pgf_slr",
                "kdf_transformer",
            ]

    workers = args.num_workers
    if workers is None:
        workers = 0 if sys.platform == "win32" else 4

    if not args.skip_landmarks:
        print(f"=== MediaPipe landmarks T={args.num_frames} ===")
        _run(
            [
                py,
                str(extract),
                "--num-frames",
                str(args.num_frames),
                "--data-dir",
                str(data_dir),
            ],
            cwd=repo_root,
        )

    if not args.skip_train:
        print("=== Train arch.md baselines (few-shot, top-count words) ===")
        cmd = [
            py,
            str(train),
            "--data-dir",
            str(data_dir),
            "--models",
            *models,
            "--num-frames",
            str(args.num_frames),
            "--num-workers",
            str(workers),
            "--seed",
            str(args.seed),
            "--protocol",
            "fewshot",
            "--draws",
            str(draws),
            "--train-shots",
            str(args.train_shots),
            "--val-shots",
            str(args.val_shots),
            "--test-per-class",
            str(args.test_per_class),
            "--n-words",
            str(args.n_words),
        ]
        if args.words:
            cmd.extend(["--words", *args.words])
        if epochs is not None:
            cmd.extend(["--epochs", str(epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.cpu:
            cmd.append("--cpu")
        if args.strict_protocol:
            cmd.append("--strict-protocol")
        _run(cmd, cwd=repo_root)

        print("=== Re-eval locked few-shot test split ===")
        eval_models = models if models != ["all"] else ["all"]
        _run(
            [
                py,
                str(repo_root / EVAL_PY),
                "--data-dir",
                str(data_dir),
                "--models",
                *eval_models,
                "--seed",
                str(args.seed),
                "--protocol",
                "fewshot",
                "--n-words",
                str(args.n_words),
                "--test-per-class",
                str(args.test_per_class),
                *(["--cpu"] if args.cpu else []),
            ],
            cwd=repo_root,
        )

    sys.path.insert(0, str(repo_root))
    from islr.fewshot.report import write_report  # noqa: WPS433

    write_report()
    print("=== PIPELINE COMPLETE ===")
    print(f"Weights + metrics: {repo_root / "outputs" / "weights"}")
    print(f"Comparison table:  {repo_root / "outputs" / "weights" / 'baselines_report.md'}")
    print(f"Dataset:           {data_dir}  ({n_mp4} mp4)")


if __name__ == "__main__":
    main()

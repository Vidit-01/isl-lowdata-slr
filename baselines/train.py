"""Train any / all arch.md baseline models with one shared loop.

Examples (from the IPD repo root):

  python models/baselines/train.py --list
  python models/baselines/train.py --data-dir ISL_DATASET_8WORDS --models mp_bilstm stgcn
  python models/baselines/train.py --data-dir ISL_DATASET_8WORDS --models all --epochs 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))

from common import (  # noqa: E402
    WEIGHTS_DIR,
    build_label_maps,
    configure_cuda_gpu,
    default_dataset_dir,
    load_metadata,
    save_json,
    set_seed,
    stratified_split,
)
from baselines.loop import train_one_baseline  # noqa: E402
from baselines.registry import ALL_NAMES, SPECS  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Modular trainer for arch.md ISL baselines")
    ap.add_argument("--models", nargs="+", default=["all"], help="Baseline names, or 'all'")
    ap.add_argument("--data-dir", type=str, default=None, help="Folder with metadata.csv + clips")
    ap.add_argument("--min-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--size", type=int, default=None, help="RGB frame size (cnn_bilstm)")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--list", action="store_true", help="Print registered models and exit")
    ap.add_argument("--unfreeze", action="store_true", help="Unfreeze CNN backbone")
    return ap.parse_args()


def resolve_names(raw: list[str]) -> list[str]:
    if len(raw) == 1 and raw[0].lower() in {"all", "*"}:
        return list(ALL_NAMES)
    names = []
    for n in raw:
        n = n.strip().lower().replace("-", "_")
        if n not in SPECS:
            known = ", ".join(ALL_NAMES)
            raise SystemExit(f"Unknown model '{n}'. Choose from: {known}")
        names.append(n)
    return names


def main() -> None:
    args = parse_args()
    if args.list:
        print("arch.md baselines:\n")
        for name, spec in SPECS.items():
            print(f"  {name:18s}  [{spec.modality:13s}]  {spec.title}")
        return

    names = resolve_names(args.models)
    set_seed(args.seed)
    device = torch_device(args.cpu)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_dataset_dir()
    print(f"dataset={data_dir}")

    df = load_metadata(min_clips=args.min_clips, dataset_dir=data_dir)
    w2i, i2w = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    train_df, val_df, test_df = stratified_split(df, seed=args.seed)
    labels = {
        "word_to_id": w2i,
        "id_to_word": {str(k): v for k, v in i2w.items()},
        "num_classes": len(w2i),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "dataset_dir": str(data_dir),
    }
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(labels, WEIGHTS_DIR / "baselines_labels.json")
    print(f"split train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={len(w2i)}")
    print("models:", ", ".join(names))

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_frames": args.num_frames,
        "num_workers": args.num_workers,
        "size": args.size,
        "patience": args.patience,
    }
    if args.unfreeze:
        overrides["freeze_backbone"] = False

    results = {}
    for name in names:
        try:
            test = train_one_baseline(name, train_df, val_df, test_df, labels, device, overrides)
            results[name] = {"test_acc": test["acc"], "test_loss": test["loss"], "n": test["n"]}
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results[name] = {"error": str(exc)}
            print(f"FAILED {name}: {exc}")

    summary_path = WEIGHTS_DIR / "baselines_summary.json"
    save_json(results, summary_path)
    print("\n=== baselines done ===")
    for k, v in results.items():
        if "test_acc" in v:
            print(f"  {k:18s}  test_acc={v['test_acc']:.3f}  loss={v['test_loss']:.4f}")
        else:
            print(f"  {k:18s}  {v}")
    print(f"weights: {WEIGHTS_DIR}")
    print(f"summary: {summary_path}")
    from baselines.report import write_report

    write_report()


def torch_device(force_cpu: bool):
    import torch

    if force_cpu:
        return torch.device("cpu")
    return configure_cuda_gpu()


if __name__ == "__main__":
    main()

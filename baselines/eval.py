"""Evaluate saved arch.md baseline weights on the held-out test split."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import torch
import torch.nn as nn

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
from common.engine import evaluate  # noqa: E402
from baselines.loop import make_loaders, merge_cfg  # noqa: E402
from baselines.registry import ALL_NAMES, get_spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval arch.md baselines")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--min-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    names = list(ALL_NAMES) if args.models == ["all"] else args.models
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else configure_cuda_gpu()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_dataset_dir()

    df = load_metadata(min_clips=args.min_clips, dataset_dir=data_dir)
    w2i, _ = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    _, _, test_df = stratified_split(df, seed=args.seed)
    dummy_train = test_df
    dummy_val = test_df

    results = {}
    for name in names:
        wdir = WEIGHTS_DIR / name
        ckpt = wdir / "model.pt"
        if not ckpt.exists():
            print(f"SKIP {name}: missing {ckpt}")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = blob.get("meta", {})
        cfg = dict(meta.get("cfg", {}))
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        spec = get_spec(name)
        cfg = merge_cfg(spec, cfg)
        num_classes = int(meta.get("num_classes", len(w2i)))
        model = spec.build(num_classes, cfg)
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()
        _, _, test_loader = make_loaders(spec, dummy_train, dummy_val, test_df, cfg, pin_memory=device.type == "cuda")
        metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, spec.forward_fn, desc=f"{name} test")
        summary = {k: v for k, v in metrics.items()}
        save_json(summary, wdir / "test_metrics.json")
        results[name] = {"test_acc": metrics["acc"], "test_loss": metrics["loss"], "n": metrics["n"]}
        print(f"{name}: test_acc={metrics['acc']:.3f} n={metrics['n']}")

    out = WEIGHTS_DIR / "baselines_eval_summary.json"
    save_json(results, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()

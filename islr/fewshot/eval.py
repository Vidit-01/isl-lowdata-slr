"""Evaluate saved arch.md baseline weights on the locked few-shot test split."""
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
sys.path.insert(0, str(ROOT))

from islr.common import (  # noqa: E402
    WEIGHTS_DIR,
    build_label_maps,
    configure_cuda_gpu,
    default_dataset_dir,
    load_metadata,
    save_json,
    set_seed,
    stratified_split,
)
from islr.common.engine import evaluate  # noqa: E402
from islr.fewshot.loop import make_loaders, merge_cfg  # noqa: E402
from islr.fewshot.metrics import score_split  # noqa: E402
from islr.fewshot.protocol import (  # noqa: E402
    fewshot_protocol_split,
    frame_from_paths,
    load_protocol,
    print_split_audit,
    resolve_words,
)
from islr.models.registry import ALL_NAMES, canonical_name, get_spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval arch.md baselines")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--min-clips", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--protocol", choices=["fewshot", "stratified"], default="fewshot")
    ap.add_argument("--train-shots", type=int, default=7)
    ap.add_argument("--val-shots", type=int, default=1)
    ap.add_argument("--test-per-class", type=int, default=15)
    ap.add_argument("--words", nargs="+", default=None)
    ap.add_argument("--n-words", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    names = list(ALL_NAMES) if args.models == ["all"] else [canonical_name(n) for n in args.models]
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else configure_cuda_gpu()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_dataset_dir()

    df = load_metadata(min_clips=args.min_clips, dataset_dir=data_dir)
    proto = load_protocol(WEIGHTS_DIR / "fewshot_protocol.json")
    if args.words is None and proto and proto.get("words"):
        words = list(proto["words"])
        print("words from fewshot_protocol.json:", ", ".join(words))
    else:
        words = resolve_words(args.words, df, n_words=args.n_words)
    if words:
        df = df[df["word"].isin(words)].copy()
    w2i, i2w = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    class_names = [i2w[i] for i in range(len(i2w))]

    if args.protocol == "fewshot" and proto and proto.get("locked_test_paths"):
        test_df = frame_from_paths(df, proto["locked_test_paths"])
        print(f"locked test from fewshot_protocol.json  n={len(test_df)}")
        if proto.get("per_class"):
            print_split_audit(
                {
                    "train_shots": proto.get("train_shots"),
                    "val_shots": proto.get("val_shots"),
                    "test_per_class": proto.get("test_per_class"),
                    "split_sizes": {"train": "—", "val": "—", "test": len(test_df)},
                    "per_class": proto["per_class"],
                    "warnings": proto.get("warnings") or [],
                    "leakage": proto.get("leakage") or {},
                }
            )
    elif args.protocol == "fewshot":
        _, _, test_df, audit = fewshot_protocol_split(
            df,
            train_shots=args.train_shots,
            test_per_class=args.test_per_class,
            val_shots=args.val_shots,
            protocol_seed=args.seed,
            draw_seed=args.seed,
            words=words,
        )
        print_split_audit(audit)
    else:
        _, _, test_df = stratified_split(df, seed=args.seed)

    if test_df.empty:
        raise SystemExit("empty test split — check --data-dir / fewshot_protocol.json paths")

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
        _, _, test_loader = make_loaders(
            spec, dummy_train, dummy_val, test_df, cfg, pin_memory=device.type == "cuda"
        )
        metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, spec.forward_fn, desc=f"{name} test")
        best_val = meta.get("best_val_acc")
        scored = score_split(
            metrics["labels"],
            metrics["preds"],
            class_names,
            best_val_acc=best_val,
            n_boot=args.n_boot,
        )
        metrics.update({k: v for k, v in scored.items() if k not in ("preds", "labels")})
        metrics["best_val_acc"] = best_val
        if best_val is not None:
            metrics["val_test_gap"] = float(best_val) - float(metrics["acc"])
        save_json(metrics, wdir / "test_metrics.json")
        results[name] = {
            "test_acc": metrics["acc"],
            "test_loss": metrics["loss"],
            "n": metrics["n"],
            "macro_f1": scored["macro_f1"]["point"],
            "macro_acc": scored["macro_acc"]["point"],
            "val_test_gap": metrics.get("val_test_gap"),
        }
        print(
            f"{name}: acc={metrics['acc']:.3f} macro-F1={scored['macro_f1']['point']:.3f} "
            f"n={metrics['n']} gap={metrics.get('val_test_gap')}"
        )

    out = WEIGHTS_DIR / "baselines_eval_summary.json"
    save_json(results, out)
    print("Wrote", out)
    from islr.fewshot.report import write_report

    write_report()


if __name__ == "__main__":
    main()

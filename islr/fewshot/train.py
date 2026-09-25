"""Train any / all arch.md baseline models with one shared loop.

Few-shot protocol (default): locked cross-signer test, k-shot train draws.

  python -m islr.fewshot.train --list
  python -m islr.fewshot.train --data-dir ISL_DATASET --models mp_bilstm stgcn
  python -m islr.fewshot.train --data-dir ISL_DATASET --models all --epochs 20
  python -m islr.fewshot.train --data-dir ISL_DATASET --draws 3 --train-shots 7
  python -m islr.fewshot.train --protocol stratified   # old random split
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

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
from islr.fewshot.loop import train_one_baseline  # noqa: E402
from islr.fewshot.metrics import mean_std  # noqa: E402
from islr.fewshot.protocol import (  # noqa: E402
    fewshot_protocol_split,
    print_split_audit,
    print_word_choice,
    protocol_meets_spec,
    resolve_words,
    save_protocol,
)
from islr.models.registry import ALL_NAMES, SPECS, canonical_name  # noqa: E402


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
    ap.add_argument(
        "--protocol",
        choices=["fewshot", "stratified"],
        default="fewshot",
        help="fewshot = locked test + k-shot train draws (default)",
    )
    ap.add_argument("--train-shots", type=int, default=7, help="Training clips per word")
    ap.add_argument("--val-shots", type=int, default=1, help="Val clips per word (from leftover)")
    ap.add_argument("--test-per-class", type=int, default=15, help="Locked test clips per word")
    ap.add_argument(
        "--draws",
        type=int,
        default=3,
        help="Random draws of the k-shot train set (same locked test). Use 1 for smoke.",
    )
    ap.add_argument(
        "--words",
        nargs="+",
        default=None,
        help="Glosses to keep. Default: the --n-words classes with the most clips. "
        "'top8', 'all', or 'legacy8' (eat/go/hello/…). Use thank_you for 'thank you'.",
    )
    ap.add_argument(
        "--n-words",
        type=int,
        default=8,
        help="How many highest-count classes to keep when --words is omitted or 'top'",
    )
    ap.add_argument(
        "--strict-protocol",
        action="store_true",
        help="Exit if any class has fewer than --test-per-class held-out clips",
    )
    ap.add_argument("--protocol-seed", type=int, default=None, help="Seed for the locked test (default: --seed)")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for CIs")
    return ap.parse_args()


def resolve_names(raw: list[str]) -> list[str]:
    if len(raw) == 1 and raw[0].lower() in {"all", "*"}:
        return list(ALL_NAMES)
    names = []
    for n in raw:
        n = canonical_name(n.strip().lower().replace("-", "_"))
        if n not in SPECS:
            known = ", ".join(ALL_NAMES)
            raise SystemExit(f"Unknown model '{n}'. Choose from: {known}")
        names.append(n)
    return names


def _compact(test: dict) -> dict:
    return {
        "test_acc": test.get("acc"),
        "test_loss": test.get("loss"),
        "n": test.get("n"),
        "macro_acc": (test.get("macro_acc") or {}).get("point"),
        "macro_f1": (test.get("macro_f1") or {}).get("point"),
        "macro_f1_low": (test.get("macro_f1") or {}).get("low"),
        "macro_f1_high": (test.get("macro_f1") or {}).get("high"),
        "best_val_acc": test.get("best_val_acc"),
        "val_test_gap": test.get("val_test_gap"),
    }


def _copy_canonical(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    if args.list:
        print("arch.md baselines:\n")
        for name, spec in SPECS.items():
            print(f"  {name:18s}  [{spec.modality:13s}]  {spec.title}")
        return

    names = resolve_names(args.models)
    protocol_seed = args.seed if args.protocol_seed is None else args.protocol_seed
    device = torch_device(args.cpu)
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_dataset_dir()
    print(f"dataset={data_dir}")

    df = load_metadata(min_clips=args.min_clips, dataset_dir=data_dir)
    words = resolve_words(args.words, df, n_words=args.n_words)
    print_word_choice(df, words)
    if words:
        df = df[df["word"].isin(words)].copy()
        if df.empty:
            raise SystemExit(f"no clips left after --words {words}")

    w2i, i2w = build_label_maps(df["word"].tolist())
    df["y"] = df["word"].map(w2i)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

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

    n_draws = max(1, int(args.draws)) if args.protocol == "fewshot" else 1
    draw_rows: list[dict] = []
    locked_test: list[str] | None = None
    protocol_blob: dict = {
        "protocol": args.protocol,
        "train_shots": args.train_shots,
        "val_shots": args.val_shots,
        "test_per_class": args.test_per_class,
        "protocol_seed": protocol_seed,
        "n_draws": n_draws,
        "words": words if words is not None else sorted(df["word"].unique().tolist()),
        "dataset_dir": str(data_dir),
        "draws": [],
    }

    results: dict = {}
    for draw in range(n_draws):
        draw_seed = args.seed + draw * 1009
        set_seed(draw_seed)
        if args.protocol == "fewshot":
            train_df, val_df, test_df, audit = fewshot_protocol_split(
                df,
                train_shots=args.train_shots,
                test_per_class=args.test_per_class,
                val_shots=args.val_shots,
                protocol_seed=protocol_seed,
                draw_seed=draw_seed,
                words=words,
            )
            print(f"\n=== draw {draw + 1}/{n_draws}  seed={draw_seed} ===")
            print_split_audit(audit)
            if locked_test is None:
                locked_test = list(audit["test_paths"])
                protocol_blob["locked_test_paths"] = locked_test
                protocol_blob["per_class"] = audit["per_class"]
                protocol_blob["warnings"] = list(audit["warnings"])
                protocol_blob["leakage"] = audit["leakage"]
                if args.strict_protocol and not protocol_meets_spec(audit, args.test_per_class):
                    save_protocol(protocol_blob, WEIGHTS_DIR / "fewshot_protocol.json")
                    raise SystemExit(
                        "strict few-shot protocol failed: some classes have "
                        f"< {args.test_per_class} test clips. Add more videos or drop --strict-protocol."
                    )
            elif audit["test_paths"] != locked_test:
                raise SystemExit(
                    f"draw {draw}: locked test set changed (protocol_seed should keep it fixed)"
                )
            protocol_blob["draws"].append(
                {
                    "draw": draw,
                    "draw_seed": draw_seed,
                    "train_paths": audit["train_paths"],
                    "val_paths": audit["val_paths"],
                    "split_sizes": audit["split_sizes"],
                    "warnings": audit["warnings"],
                }
            )
        else:
            train_df, val_df, test_df = stratified_split(df, seed=draw_seed)
            print(f"stratified split train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        labels = {
            "word_to_id": w2i,
            "id_to_word": {str(k): v for k, v in i2w.items()},
            "num_classes": len(w2i),
            "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "dataset_dir": str(data_dir),
            "protocol": args.protocol,
            "draw": draw,
            "draw_seed": draw_seed,
        }
        save_json(labels, WEIGHTS_DIR / "baselines_labels.json")
        print("models:", ", ".join(names))

        for name in names:
            out_dir = WEIGHTS_DIR / name if n_draws == 1 else WEIGHTS_DIR / "draws" / f"{draw:02d}" / name
            try:
                test = train_one_baseline(
                    name,
                    train_df,
                    val_df,
                    test_df,
                    labels,
                    device,
                    overrides,
                    out_dir=out_dir,
                    n_boot=args.n_boot,
                )
                row = {"model": name, "draw": draw, **_compact(test)}
                draw_rows.append(row)
                results.setdefault(name, {"draws": []})["draws"].append(row)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                err = {"model": name, "draw": draw, "error": str(exc)}
                draw_rows.append(err)
                results.setdefault(name, {"draws": []})["draws"].append(err)
                print(f"FAILED {name} draw {draw}: {exc}")

        if n_draws > 1:
            for name in names:
                src = WEIGHTS_DIR / "draws" / f"{draw:02d}" / name
                if src.exists():
                    _copy_canonical(src, WEIGHTS_DIR / name)

    for name, blob in results.items():
        f1s = [d["macro_f1"] for d in blob["draws"] if d.get("macro_f1") is not None]
        accs = [d["test_acc"] for d in blob["draws"] if d.get("test_acc") is not None]
        gaps = [d["val_test_gap"] for d in blob["draws"] if d.get("val_test_gap") is not None]
        blob["macro_f1"] = mean_std(f1s)
        blob["test_acc"] = mean_std(accs)
        blob["val_test_gap"] = mean_std(gaps)

    if args.protocol == "fewshot":
        save_protocol(protocol_blob, WEIGHTS_DIR / "fewshot_protocol.json")
    save_json(draw_rows, WEIGHTS_DIR / "baselines_draws.json")
    summary_path = WEIGHTS_DIR / "baselines_summary.json"
    save_json(results, summary_path)

    print("\n=== baselines done ===")
    for k, v in results.items():
        f1 = v.get("macro_f1") or {}
        acc = v.get("test_acc") or {}
        gap = v.get("val_test_gap") or {}
        if f1.get("mean") is not None:
            print(
                f"  {k:18s}  macro-F1={f1['mean']:.3f}±{f1['std']:.3f}  "
                f"acc={acc['mean']:.3f}±{acc['std']:.3f}  "
                f"val-test gap={gap['mean']:.3f}±{gap['std']:.3f}  draws={f1['n']}"
            )
        else:
            print(f"  {k:18s}  {v}")
    print(f"weights: {WEIGHTS_DIR}")
    print(f"summary: {summary_path}")
    from islr.fewshot.report import write_report

    write_report()


def torch_device(force_cpu: bool):
    import torch

    if force_cpu:
        return torch.device("cpu")
    return configure_cuda_gpu()


if __name__ == "__main__":
    main()

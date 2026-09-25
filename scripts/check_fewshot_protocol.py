"""Sanity checks for few-shot protocol (not part of the training pipeline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from islr.fewshot.metrics import mcnemar_exact, score_split, wilson_interval
from islr.fewshot.protocol import fewshot_protocol_split, protocol_meets_spec, resolve_words, top_words


def _pool(n_users: int, words: list[str]) -> pd.DataFrame:
    rows = []
    for w in words:
        for u in range(1, n_users + 1):
            rows.append(
                dict(
                    word=w,
                    signer=f"User{u:03d}",
                    video_path=f"{w}/User{u:03d}.mp4",
                    abs_path=f"/d/{w}/User{u:03d}.mp4",
                    original_filename=f"User{u:03d}.mp4",
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    words8 = ["eat", "go", "hello", "help", "no", "please", "water", "yes"]
    df = _pool(7, words8)
    _, _, _, a0 = fewshot_protocol_split(df, protocol_seed=42, draw_seed=42)
    _, _, _, a1 = fewshot_protocol_split(df, protocol_seed=42, draw_seed=1051)
    assert a0["test_paths"] == a1["test_paths"], "locked test must be stable"
    assert not protocol_meets_spec(a0, 15)
    assert not a0["leakage"]["train_test"], a0["leakage"]
    assert not a0["leakage"]["val_test"], a0["leakage"]
    pc = a0["per_class"]["eat"]
    assert pc["n_train"] + pc["n_val"] + pc["n_test"] == 7
    assert pc["n_test"] == 1 and pc["n_train"] == 5 and pc["n_val"] == 1
    assert any("too small" in w for w in a0["warnings"])
    print("tiny-pool OK", a0["split_sizes"])

    big = _pool(30, ["eat", "go"])
    t0, _, _, p0 = fewshot_protocol_split(
        big, protocol_seed=0, draw_seed=1, words=["eat", "go"]
    )
    t1, _, _, p1 = fewshot_protocol_split(
        big, protocol_seed=0, draw_seed=2, words=["eat", "go"]
    )
    assert p0["test_paths"] == p1["test_paths"]
    assert protocol_meets_spec(p0, 15)
    assert p0["per_class"]["eat"]["n_test"] == 15
    assert set(t0["video_path"]) != set(t1["video_path"]), "draws should resample train"
    assert not p0["leakage"]["train_test"]
    print("large-pool OK", p0["per_class"]["eat"])

    ranked = pd.DataFrame(
        {
            "word": ["hello"] * 36 + ["friend"] * 38 + ["eat"] * 7 + ["sit"] * 19,
            "signer": [f"User{i:03d}" for i in range(36 + 38 + 7 + 19)],
            "video_path": [f"c{i}.mp4" for i in range(36 + 38 + 7 + 19)],
            "abs_path": [f"/d/c{i}.mp4" for i in range(36 + 38 + 7 + 19)],
            "original_filename": [f"c{i}.mp4" for i in range(36 + 38 + 7 + 19)],
        }
    )
    assert top_words(ranked, 3) == ["friend", "hello", "sit"]
    assert resolve_words(None, ranked, n_words=3) == ["friend", "hello", "sit"]
    assert resolve_words(["top2"], ranked) == ["friend", "hello"]

    w = wilson_interval(18, 20)
    assert 0.0 <= w["low"] <= w["p"] <= w["high"] <= 1.0
    yt = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    pa = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    pb = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    mc = mcnemar_exact(yt, pa, pb)
    sc = score_split(yt, pa, ["a", "b"], best_val_acc=0.9, n_boot=200, seed=0)
    print("mcnemar", mc)
    print("macro_f1", sc["macro_f1"]["point"], "gap", sc["val_test_gap"])
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

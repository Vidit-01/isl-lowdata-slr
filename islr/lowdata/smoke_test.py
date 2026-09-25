"""One-command check of the whole low-data pipeline on synthetic data (CPU by default).

    python lowdata.py smoke                 # everything, ~10-15 min on a laptop CPU
    python lowdata.py smoke --fast          # fewer steps, subset of models (~3 min)
    python lowdata.py smoke --device cuda   # on Kaggle, to check the GPU path (fp16)

What it checks
  1. synthetic stores (two sources, signers, INCLUDE-style MVI sessions, a duplicate
     clip in a second store, missing hands) -> load_stores dedup and identities
  2. protocols: fixed test set across K and protocols, nested training subsets, no
     identity leakage, vocabulary restriction keeps the target words, cross_source
  3. feature banks for every modality (shapes, missing points stay 0)
  4. every study model trains for a short fixed budget and beats chance
     (the synthetic words differ in hand trajectory, so this is learnable)
  5. the RGB CNN+BiLSTM (not part of the study; stores hold no video) forward/backward,
     and the legacy engine's train/evaluate path
  6. checkpoint/resume: an interrupted + resumed run gives the same predictions as an
     uninterrupted one
  7. DDP: 2 ranks (gloo on CPU, launched by hand), shared bank build with a barrier
  8. sweep: member split covers every job exactly once, runs, skips finished jobs;
     report aggregates the result
  9. extraction sharding partitions clips; shard stores merge
 10. extraction worker pool survives a worker that aborts (as MediaPipe's C++ CHECKs do):
     the clip is recorded as failed, the worker replaced, every other clip delivered
It never uses two GPU jobs at once (DDP here is CPU-only unless --device cuda).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "lowdata.py"
EXTRA_WORDS = ("book", "cat", "dog", "friend")
FAST_MODELS = ("mp_bilstm", "stgcn", "cwt_transformer", "kdf_transformer", "partformer", "conv1d_former",
               "kdf_partformer")


# ---------------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------------
def _frame_seq(word_id: int, rng: np.random.Generator, ident_off: np.ndarray, T: int, two_hands: bool):
    """(T, 543, 3) in store order; each word = a distinct right-hand trajectory/shape."""
    from islr.lowdata.store import FACE0, LH0, N_LM, RH0

    t = np.linspace(0, 1, T)
    f = 0.6 + 0.35 * (word_id % 4)
    ph = 2 * np.pi * (word_id // 4) / 3
    cx = 0.56 + 0.10 * np.cos(2 * np.pi * f * t + ph)
    cy = 0.55 + 0.10 * np.sin(2 * np.pi * f * t + ph) * (1 if word_id % 2 else -1)
    arr = np.zeros((T, N_LM, 3), np.float32)
    pose = np.zeros((33, 3), np.float32)
    pose[:, 0], pose[:, 1] = 0.5, 0.45
    pose[0, :2] = (0.5, 0.25)
    pose[11, :2], pose[12, :2] = (0.60, 0.40), (0.40, 0.40)
    pose[23, :2], pose[24, :2] = (0.57, 0.80), (0.43, 0.80)
    pose[13, :2], pose[14, :2] = (0.63, 0.55), (0.37, 0.55)
    spread = 0.3 + 0.12 * (word_id % 3)
    ang = np.linspace(-spread, spread, 5)
    hand = np.zeros((21, 3), np.float32)
    for k, a in enumerate(ang):  # 5 fingers x 4 joints + wrist
        for j in range(4):
            hand[1 + 4 * k + j, :2] = (0.012 * (j + 1) * np.sin(a), -0.012 * (j + 1) * np.cos(a))
    face = rng.normal(0, 0.02, (468, 3)).astype(np.float32)
    face[:, :2] += (0.5, 0.25)
    for i in range(T):
        p = pose.copy()
        p[16, :2] = (cx[i], cy[i])
        rh = hand.copy()
        rh[:, :2] += (cx[i], cy[i])
        arr[i, :33] = p
        arr[i, RH0:RH0 + 21] = rh
        if two_hands:
            lh = hand.copy()
            lh[:, 0] *= -1
            lh[:, :2] += (1 - cx[i], cy[i])
            arr[i, LH0:LH0 + 21] = lh
            arr[i, 15, :2] = (1 - cx[i], cy[i])
        else:
            arr[i, LH0:LH0 + 21] = np.nan
        arr[i, FACE0:] = face
    arr[..., :2] = (arr[..., :2] - 0.5) * ident_off[2] + 0.5 + ident_off[:2]
    arr += rng.normal(0, 0.004, arr.shape).astype(np.float32)
    drop = rng.random(T) < 0.12  # detector misses
    arr[drop, RH0:RH0 + 21] = np.nan
    arr[:2, LH0:FACE0] = np.nan  # rest pose before the sign (trimmed)
    return arr


def make_synthetic(root: Path, seed: int = 0) -> tuple[list[str], dict]:
    from islr.lowdata.protocol import LEGACY8
    from islr.lowdata.store import append_index, clip_key, init_store

    rng = np.random.default_rng(seed)
    words = list(LEGACY8) + list(EXTRA_WORDS)
    a, b = init_store(root / "storeA"), init_store(root / "storeB")
    rows_a, rows_b = [], []
    idents = {f"User{i}": rng.normal(0, 0.03, 3) * (1, 1, 0) + (0, 0, 1 + 0.1 * i) for i in range(1, 5)}
    sessions = {s: rng.normal(0, 0.03, 3) * (1, 1, 0) + (0, 0, 0.95) for s in (1000, 3000, 6000)}

    def put(store, rows, source, rel, word, wid, off, signer="", session=""):
        key = clip_key(source, rel)
        arr = _frame_seq(wid, rng, off, int(rng.integers(22, 48)), two_hands=wid % 3 == 0)
        np.save(store / "npy" / f"{key}.npy", arr.astype(np.float16))
        rows.append({"key": key, "word": word, "source": source, "signer": signer, "session": session,
                     "n_frames": len(arr), "fps": "15", "path": f"npy/{key}.npy", "video_rel": rel,
                     "license": "synthetic"})

    for wid, w in enumerate(words):
        for u, off in idents.items():
            for c in range(2):
                put(a, rows_a, "isl500", f"{w}/{u}_{c}.mp4", w, wid, off, signer=u)
        for s0, off in sessions.items():
            for c in range(2):
                put(a, rows_a, "include", f"{w.title()}/MVI_{s0 + 13 * wid + c:04d}.MOV", w, wid, off)
    # store B: the same INCLUDE clip of 'hello' again (must be deduplicated) + 1 new clip
    hello = words.index("hello")
    put(b, rows_b, "include", f"Greetings/1. Hello/MVI_{1000 + 13 * hello:04d}.MOV", "1. Hello", hello,
        sessions[1000])
    put(b, rows_b, "include", "Greetings/1. Hello/MVI_6005.MOV", "1. Hello", hello, sessions[6000])
    append_index(a, pd.DataFrame(rows_a))
    append_index(b, pd.DataFrame(rows_b))
    return [str(a), str(b)], {"n_a": len(rows_a), "n_b": len(rows_b), "words": words}


# ---------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name, fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            self.rows.append((name, True, detail, time.time() - t0))
            print(f"  PASS {name} {detail} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # report and carry on with the rest
            self.rows.append((name, False, f"{type(e).__name__}: {e}", time.time() - t0))
            print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    @property
    def ok(self):
        return all(r[1] for r in self.rows)


def check_stores(stores, meta):
    from islr.lowdata.store import load_stores

    df = load_stores(stores)
    assert len(df) == meta["n_a"] + 1, f"dedup failed: {len(df)} rows"
    inc = df[df["source"] == "include"]
    sess = sorted(inc["identity"].unique())
    assert len(sess) == 3, f"expected 3 INCLUDE sessions, got {sess}"
    assert set(df["word"]) == set(meta["words"])
    return f"{len(df)} clips, {df['identity'].nunique()} identities"


def check_protocols(stores):
    from islr.lowdata.protocol import LEGACY8, SplitConfig, make_split
    from islr.lowdata.store import load_stores

    df = load_stores(stores)
    for seed in (0, 1):
        prev, fps = None, set()
        for k in (1, 2, 4, 8):
            s = make_split(df, SplitConfig("scarce", k, seed))
            fps.add(s.info["test_fingerprint"])
            tr = set(df.loc[s.train_idx, "key"])
            if prev is not None:
                assert prev <= tr, f"K={k} training set does not contain the smaller K's"
            prev = tr
            per = s.info["train_per_word"]
            assert all(per[w] == min(k, per[w]) for w in LEGACY8), per
            assert all(per[w] > k for w in EXTRA_WORDS if k < 8), "rich words should keep their pool"
        u = make_split(df, SplitConfig("uniform", 2, seed))
        fps.add(u.info["test_fingerprint"])
        assert len(fps) == 1, "test set changed with K or protocol"
    v = make_split(df, SplitConfig("uniform", 2, 0, n_words=9))
    assert len(v.words) == 9 and set(LEGACY8) <= set(v.words), v.words
    c = make_split(df, SplitConfig("cross_source", 2, 0, train_sources=["isl500"], test_sources=["include"]))
    assert set(df.loc[c.test_idx, "source"]) == {"include"} and len(c.train_idx) == 2 * len(c.words)
    return "fixed test across K/protocols, nested, leak-free"


def check_banks(stores, cache):
    from islr.lowdata.bank import MODALITIES, load_bank
    from islr.lowdata.store import load_stores

    df = load_stores(stores)
    shapes = {}
    for m in MODALITIES:
        b = load_bank(df, m, 30, True, cache, workers=1)
        assert all(len(a) == len(df) for a in b.arrays)
        assert all(np.isfinite(a).all() for a in b.arrays), f"{m}: non-finite features"
        shapes[m] = tuple(b.arrays[0].shape[1:])
    parts = load_bank(df, "parts", 30, True, cache).arrays[0]
    from islr.lowdata.protocol import LEGACY8

    words = list(LEGACY8) + list(EXTRA_WORDS)
    one_handed = next(w for i, w in enumerate(words) if i % 3)  # make_synthetic: two hands iff id % 3 == 0
    one = df.index[df["word"] == one_handed][0]  # its left hand must be exactly 0
    assert np.all(parts[one, :, 33:54] == 0), "missing hand is not 0"
    return ", ".join(f"{k}{v}" for k, v in shapes.items())


def train_models(stores, out, models, steps, device, results):
    from islr.lowdata.run import make_config, run

    def one(m):
        def fn():
            cfg = make_config(stores=stores, model=m, protocol="full", shots=None, seed=0, out=str(out),
                              cache_dir=str(out / "_bank"), min_steps=steps, max_steps=steps, device=device,
                              quiet=True, workers=1)
            rc = run(cfg)
            assert rc == 0, f"rc={rc}"
            r = json.loads((Path(out) / f"full/{m}/Kall/s0/result.json").read_text())
            mt = r["metrics"]
            results[m] = (mt["top1"], mt.get("proto_top1"), mt["chance"], r["train_seconds"], r["n_params"])
            assert mt["top1"] > 1.5 * mt["chance"], f"top1 {mt['top1']:.3f} not above chance {mt['chance']:.3f}"
            return f"top1 {mt['top1']:.2f} proto {mt.get('proto_top1', float('nan')):.2f} " \
                   f"(chance {mt['chance']:.2f})"
        return fn

    return one


def check_rgb():
    import torch

    from islr.models.registry import get_spec

    spec = get_spec("cnn_bilstm")
    hp = dict(spec.defaults, pretrained=False, hidden=64, lstm_layers=2)
    model = spec.build(5, hp)
    x = torch.rand(2, 4, 3, 64, 64)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1, 3]))
    loss.backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]).step()
    assert torch.isfinite(loss)
    return f"loss {loss.item():.3f}"


def check_legacy_engine(stores, cache):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from islr.models.registry import get_spec
    from islr.common.engine import evaluate, train_one_epoch
    from islr.lowdata.bank import load_bank
    from islr.lowdata.store import load_stores

    df = load_stores(stores)
    words = sorted(df["word"].unique())
    y = torch.tensor([words.index(w) for w in df["word"]])
    x = torch.from_numpy(load_bank(df, "landmarks", 30, True, cache).arrays[0])
    spec = get_spec("mp_bilstm")
    model = spec.build(len(words), dict(spec.defaults))
    dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    tr = train_one_epoch(model, dl, opt, torch.nn.CrossEntropyLoss(), torch.device("cpu"), use_amp=False)
    ev = evaluate(model, dl, torch.nn.CrossEntropyLoss(), torch.device("cpu"), use_amp=False)
    assert ev["n"] == len(df)
    return f"train loss {tr['loss']:.3f}, eval acc {ev['acc']:.3f}"


def check_resume(stores, out, device):
    from islr.lowdata.run import INCOMPLETE, make_config, run

    base = dict(stores=stores, model="partformer", protocol="scarce", shots=2, seed=1, cache_dir=str(out / "_bank"),
                min_steps=60, max_steps=60, device=device, quiet=True, workers=1)
    assert run(make_config(**base, run_dir=str(out / "straight"))) == 0
    rc = run(make_config(**base, run_dir=str(out / "resumed"), stop_after_steps=25))
    assert rc == INCOMPLETE and (out / "resumed" / "ckpt.pt").exists(), f"rc={rc}"
    assert run(make_config(**base, run_dir=str(out / "resumed"))) == 0
    a = np.load(out / "straight" / "preds.npz")["logits"].astype(np.float32)
    b = np.load(out / "resumed" / "preds.npz")["logits"].astype(np.float32)
    diff = float(np.abs(a - b).max())
    assert diff < 1e-2, f"resumed run differs from the uninterrupted one (max |dlogit| {diff:.4f})"
    return f"max |logit diff| {diff:.2e}"


def check_ddp(stores, out, device):
    import socket

    out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = []
    for model in ("kdf_transformer", "partformer"):
        run_dir = out / f"ddp_{model}"
        procs = []
        for rank in range(2):
            env = dict(os.environ, RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                       MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0", PYTHONUNBUFFERED="1")
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = "-1"  # "" deletes the variable on Windows
            cmd = [sys.executable, str(LAUNCHER), "run", "--stores", *stores, "--model", model, "--protocol",
                   "uniform", "--shots", "4", "--seed", "0", "--run-dir", str(run_dir), "--cache-dir",
                   str(out / "_bank_ddp"), "--min-steps", "30", "--max-steps", "30", "--quiet"]
            cmd += ["--device", device]
            logf = open(out / f"rank{rank}_{model}.log", "w")
            procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
        # if one rank dies the other waits forever in a collective: kill it
        t_end = time.time() + 900
        while any(p.poll() is None for p, _ in procs):
            if time.time() > t_end or any(p.poll() not in (None, 0) for p, _ in procs):
                for p, _ in procs:
                    if p.poll() is None:
                        p.kill()
                break
            time.sleep(1)
        rcs = [p.wait() for p, _ in procs]
        for _, f in procs:
            f.close()
        if any(rcs):
            logs = "\n".join((out / f"rank{r}_{model}.log").read_text()[-3000:] for r in range(2))
            raise RuntimeError(f"{model}: DDP ranks exited {rcs}\n{logs}")
        r = json.loads((run_dir / "result.json").read_text())
        assert r["world"] == 2, r["world"]
        results.append(f"{model} top1 {r['metrics']['top1']:.2f}")
        port += 1
    return "; ".join(results)


def check_sweep(stores, out, device):
    from islr.lowdata import report, sweep

    grid = {
        "name": "smoke", "stores": stores,
        "base": {"protocol": "scarce", "targets": "legacy8", "min_steps": 30, "max_steps": 30, "device": device,
                 "workers": 1},
        "grid": {"model": ["mp_transformer", "partformer"], "shots": [1, 2, 4], "seed": [0]},
        "split_by": "shots",
    }
    out.mkdir(parents=True, exist_ok=True)
    gf = out / "grid.json"
    gf.write_text(json.dumps(grid))
    jobs = sweep.expand_jobs(sweep.load_grid(str(gf)), str(out / "sw"))
    sweep.estimate_steps(jobs)
    for j in jobs:
        j["cost"] = sweep.job_cost(j, {})
    plan = sweep.assign_members(jobs, 2)
    names = [j["name"] for m in plan.values() for j in m]
    assert sorted(names) == sorted(j["name"] for j in jobs) and len(set(names)) == len(names)
    assert all(len({j["group"] for j in m}) >= 1 for m in plan.values())
    gpu_flag = ["--gpus", "0"] if device == "cpu" else []
    rc = sweep.main(["--grid", str(gf), "--out", str(out / "sw"), "--member", "0", "--members", "2", *gpu_flag])
    assert rc == 0, f"member 0 rc={rc}"
    rc = sweep.main(["--grid", str(gf), "--out", str(out / "sw2"), "--member", "1", "--members", "2", *gpu_flag])
    assert rc == 0, f"member 1 rc={rc}"
    done = len(list((out / "sw").rglob("result.json"))) + len(list((out / "sw2").rglob("result.json")))
    assert done == len(jobs), f"{done} results for {len(jobs)} jobs"
    # second session with --resume-from: everything is adopted and nothing is rerun
    rc = sweep.main(["--grid", str(gf), "--out", str(out / "sw3"), "--member", "0", "--members", "2",
                     "--resume-from", str(out / "sw"), *gpu_flag])
    assert rc == 0
    assert report.main(["--out", str(out / "sw"), str(out / "sw2"), "--dest", str(out / "rep"),
                        "--reference", "mp_transformer"]) == 0
    chk = (out / "rep" / "checks.md").read_text()
    assert "OK" in chk, chk
    return f"{len(jobs)} jobs over 2 members, resume + report OK"


def _crashing_job(args):
    """Stand-in for extract._job in check_extract_crash (module level so spawn can pickle it)."""
    video, npy, _ = args
    if video.endswith("crash"):
        os.abort()  # what "Check failed: holder_ != nullptr" does inside MediaPipe
    return npy, 10, 15.0, None


def check_extract_crash():
    from islr.lowdata.extract import _run_pool

    names = [f"v{i}" + ("crash" if i in (2, 7, 8) else "") for i in range(20)]
    todo = [(n, n + ".npy", None) for n in names]
    got = []
    stopped = _run_pool(todo, 2, "solutions", None, 640, 300, got.append, job=_crashing_job)
    errs = sorted(r[0] for r in got if r[3])
    assert not stopped, "hit the time budget: the pool hung"
    assert len(got) == 20 and len({r[0] for r in got}) == 20, f"{len(got)} results for 20 clips"
    assert errs == ["v2crash.npy", "v7crash.npy", "v8crash.npy"], errs
    return "3 aborts recorded, 17 clips delivered"


def check_sharding(stores, out):
    from islr.lowdata.extract import in_shard
    from islr.lowdata.store import merge_stores, read_index

    keys = read_index(stores[0])["key"].tolist()
    counts = [sum(in_shard(k, (i, 4)) for k in keys) for i in range(4)]
    assert sum(counts) == len(keys) and all(c > 0 for c in counts), counts
    merged = merge_stores(stores, str(out / "merged"))
    assert len(merged) == len(keys) + len(read_index(stores[1]))
    return f"4 shards {counts}, merge {len(merged)} clips"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lowdata.py smoke", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="work dir (default: a temp dir, deleted unless --keep)")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--steps", type=int, default=None, help="training steps per model (default 150, fast 80)")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip", nargs="*", default=[], help="stores protocols banks models rgb engine resume ddp sweep shard extract")
    a = p.parse_args(argv)
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # also for sweep workers ("" is dropped on Windows)
    import torch

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    from islr.models.registry import ALL_NAMES

    root = Path(a.dir) if a.dir else Path(tempfile.mkdtemp(prefix="lowdata_smoke_"))
    if root.exists() and a.dir:
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    steps = a.steps or (80 if a.fast else 150)
    models = a.models or [m for m in (FAST_MODELS if a.fast else ALL_NAMES) if m != "cnn_bilstm"]
    print(f"[smoke] dir={root} device={a.device} models={len(models)} steps={steps}", flush=True)
    t0 = time.time()
    stores, meta = make_synthetic(root / "data")
    C = Checks()
    sk = set(a.skip)
    if "stores" not in sk:
        C("stores: dedup + INCLUDE sessions", lambda: check_stores(stores, meta))
    if "protocols" not in sk:
        C("protocols", lambda: check_protocols(stores))
    if "banks" not in sk:
        C("feature banks", lambda: check_banks(stores, root / "runs" / "_bank"))
    results = {}
    if "models" not in sk:
        one = train_models(stores, root / "runs", models, steps, a.device, results)
        for m in models:
            C(f"train {m}", one(m))
    if "rgb" not in sk:
        C("cnn_bilstm (RGB) forward/backward", check_rgb)
    if "engine" not in sk:
        C("legacy engine train/eval", lambda: check_legacy_engine(stores, root / "runs" / "_bank"))
    if "resume" not in sk:
        C("checkpoint resume", lambda: check_resume(stores, root / "resume", a.device))
    if "ddp" not in sk:
        C("DDP 2 ranks", lambda: check_ddp(stores, root / "ddp", a.device))
    if "sweep" not in sk:
        C("sweep member split + resume + report", lambda: check_sweep(stores, root / "sweep", a.device))
    if "shard" not in sk:
        C("extraction shards + merge", lambda: check_sharding(stores, root))
    if "extract" not in sk:
        C("extraction survives worker abort", check_extract_crash)

    if results:
        print("\nmodel                   top1   proto  chance  params   train_s")
        for m, (t1, pr, ch, s, n) in results.items():
            print(f"{m:22s} {t1:6.2f} {pr if pr is not None else float('nan'):6.2f} {ch:7.2f} "
                  f"{n / 1e6:6.2f}M {s:8.1f}")
    n_ok = sum(r[1] for r in C.rows)
    print(f"\n[smoke] {n_ok}/{len(C.rows)} checks passed in {time.time() - t0:.0f}s")
    for name, ok, detail, _ in C.rows:
        if not ok:
            print(f"  FAILED: {name}: {detail}")
    if not a.keep and not a.dir:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if C.ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())

"""Grid sweeps split across team members (one Kaggle account each) and GPUs.

    python lowdata.py sweep --grid islr/lowdata/configs/scarce_legacy8.json --plan --members 4
    python lowdata.py sweep --grid ... --member 0 --members 4 --out /kaggle/working/sweeps \
        --time-budget-h 11 --resume-from /kaggle/input/<previous-output>/sweeps

Grid file (JSON; "$VAR" / "${VAR}" are expanded from the environment):
    {
      "name": "scarce_legacy8",
      "stores": ["$STORES/isl40", "$STORES/include"],
      "base":   {...any `run` option...},
      "grid":   {"model": [...], "shots": [1, 2, 4, 8, 16], "seed": [0, 1, 2]},
      "variants": [{}, {"n_words": 50}],      # optional; each is merged into base
      "split_by": "shots",                     # how work is divided between members
      "assign": {"0": [1, 16], "1": [2], ...}, # optional explicit member -> values
      "ddp": false                             # true: run jobs with torchrun on all GPUs
    }

Dividing the work
  Jobs are grouped by `split_by` (default: shots, i.e. "videos per word") and the
  groups are dealt to members greedily by estimated GPU time (largest first, to the
  least-loaded member). The rule is deterministic, so each member computes the same
  plan from the same grid file and needs to know only its own number. If there are
  fewer groups than members, groups are split further by seed. `--plan` prints the
  assignment and GPU-hour estimate without running anything.

Using both T4s
  Low-data jobs are tiny (a few hundred clips, ~1 M parameters): one step does not
  fill a T4 and DDP would spend its time synchronising. So by default each GPU runs
  its own queue of whole jobs (twice the throughput, no communication). A job whose
  grid sets "ddp": true (large-data runs such as protocol=full pre-training) is instead
  launched with torchrun over all GPUs, after the independent jobs.

Resuming
  Every finished run leaves result.json; `--resume-from` (previous session outputs,
  mounted read-only as Kaggle inputs) is searched first, finished runs are copied into
  --out and skipped, and half-finished runs (ckpt.pt) are copied and continued. Workers
  stop taking new jobs shortly before the time budget, and the running job checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "lowdata.py"

# Rough T4 seconds per optimiser step at batch 16, T=30 (fp16 where the model allows).
# Used only for planning; --plan calibrates from finished runs when it can find any.
STEP_SECONDS = {
    "mp_bilstm": 0.010, "mp_transformer": 0.010, "fft_bilstm": 0.012, "cwt_bilstm": 0.014,
    "cwt_transformer": 0.012, "stgcn": 0.030, "ctr_gcn": 0.060, "td_gcn": 0.080, "hwgat": 0.030,
    "pgf_slr": 0.045, "kdf_transformer": 0.015, "mp_transformer_reg": 0.012, "kdf_transformer_nodmd": 0.014,
    "kdf_transformer_nokc": 0.014, "partformer": 0.020, "partformer_cos": 0.020, "conv1d_former": 0.020,
    "kdf_partformer": 0.024, "cnn_bilstm": 0.25,
}
JOB_OVERHEAD_S = 15.0  # model build, evaluation, writing results
DEADLINE_MARGIN_S = 600.0  # stop well before Kaggle kills the session


# ---------------------------------------------------------------------------------
# grid -> jobs
# ---------------------------------------------------------------------------------
def _expand(obj):
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def load_grid(path: str) -> dict:
    g = _expand(json.loads(Path(path).read_text()))
    g.setdefault("name", Path(path).stem)
    g.setdefault("base", {})
    g.setdefault("grid", {})
    g.setdefault("variants", [{}])
    g.setdefault("split_by", "shots")
    return g


def expand_jobs(grid: dict, out: str, cache_dir: str | None = None) -> list[dict]:
    from islr.lowdata.run import make_config, run_name

    keys = list(grid["grid"])
    jobs, seen = [], set()
    for variant in grid["variants"]:
        for combo in itertools.product(*(grid["grid"][k] for k in keys)):
            cfg = copy.deepcopy(grid["base"])
            cfg.update(copy.deepcopy(variant))
            cfg.update(dict(zip(keys, combo)))
            cfg["stores"] = grid.get("stores", cfg.get("stores", []))
            ddp = bool(cfg.pop("ddp", grid.get("ddp", False)))
            cfg["out"] = out
            if cache_dir:
                cfg["cache_dir"] = cache_dir
            cfg = make_config(**cfg)
            name = run_name(cfg)
            if name in seen:
                continue
            seen.add(name)
            jobs.append({"name": name, "cfg": cfg, "ddp": ddp,
                         "group": cfg.get(grid["split_by"]), "seed": cfg["seed"]})
    return jobs


# ---------------------------------------------------------------------------------
# cost estimate and member split
# ---------------------------------------------------------------------------------
def estimate_steps(jobs: list[dict]) -> None:
    """Fill job['steps'] from the real split sizes when the stores are readable."""
    from islr.models.registry import get_spec
    from islr.lowdata.protocol import make_split, split_config_from
    from islr.lowdata.run import SPLIT_KEYS, budget, effective_batch
    from islr.lowdata.store import load_stores

    tables, sizes = {}, {}
    for j in jobs:
        c = j["cfg"]
        spec = get_spec(c["model"])
        hp = dict(spec.defaults, **(c.get("hparams") or {}))
        bs0 = int(c["batch_size"] or hp.get("batch_size", 16))
        ep = float(c["epochs"] or hp.get("epochs", 80))
        n_train = None
        try:
            tk = (tuple(c["stores"]), tuple(c["sources"] or ()))
            if tk not in tables:
                tables[tk] = load_stores(c["stores"], sources=c["sources"])
            sk = (tk, json.dumps({k: c[k] for k in SPLIT_KEYS}, sort_keys=True))
            if sk not in sizes:
                sizes[sk] = len(make_split(tables[tk], split_config_from({k: c[k] for k in SPLIT_KEYS})).train_idx)
            n_train = sizes[sk]
        except Exception as e:  # stores not mounted here: fall back to the step floor
            j["estimate_note"] = f"no split ({type(e).__name__})"
        world = 2 if j["ddp"] else 1
        if n_train:
            bs = effective_batch(bs0, n_train, world)
            j["steps"] = budget(n_train, bs, ep, c["min_steps"], c["max_steps"])
            j["n_train"] = n_train
        else:
            j["steps"] = c["min_steps"]


def calibrate(result_dirs: list[Path]) -> dict[str, float]:
    """Median measured seconds/step per model from finished runs (any T4 session)."""
    import statistics

    per = {}
    for d in result_dirs:
        for f in Path(d).rglob("result.json"):
            try:
                r = json.loads(f.read_text())
                if r.get("steps") and r.get("train_seconds") and "T4" in str(r.get("gpu", "")):
                    per.setdefault(r["model"], []).append(r["train_seconds"] / r["steps"])
            except (OSError, ValueError):
                continue
    return {m: statistics.median(v) for m, v in per.items()}


def job_cost(j: dict, sec: dict) -> float:
    from islr.models.registry import canonical_name

    m = canonical_name(j["cfg"]["model"])
    s = sec.get(m, STEP_SECONDS.get(m, 0.03))
    gpus = 2 if j["ddp"] else 1
    return (j.get("steps", 300) * s * (0.6 if j["ddp"] else 1.0) + JOB_OVERHEAD_S) * gpus  # GPU-seconds


def assign_members(jobs: list[dict], members: int, explicit: dict | None = None) -> dict[int, list[dict]]:
    """Deterministic split of jobs over members by `group` (shots by default)."""
    out = {m: [] for m in range(members)}
    if explicit:
        where = {str(v): int(m) for m, vals in explicit.items() for v in vals}
        for j in jobs:
            m = where.get(str(j["group"]))
            if m is None:
                raise KeyError(f"'assign' does not place group {j['group']!r}")
            out[m].append(j)
        return out
    def deal(key):
        groups: dict = {}
        for j in jobs:
            groups.setdefault(key(j), []).append(j)
        res, load = {m: [] for m in range(members)}, [0.0] * members
        for g in sorted(groups, key=lambda g: (-sum(j["cost"] for j in groups[g]), g)):
            m = min(range(members), key=lambda i: (load[i], i))
            res[m] += groups[g]
            load[m] += sum(j["cost"] for j in groups[g])
        return res, load

    # whole K values per member when that is balanced; otherwise (K, seed) units, e.g.
    # 5 K values over 4 members would give one member twice the work
    out, load = deal(lambda j: str(j["group"]))
    if members > 1 and max(load) > 1.25 * (sum(load) / members):
        out, _ = deal(lambda j: f"{j['group']}|s{j['seed']}")
    return out


def print_plan(plan: dict[int, list[dict]], gpus: int) -> None:
    tot = 0.0
    for m, js in plan.items():
        gsec = sum(j["cost"] for j in js)
        tot += gsec
        grp = sorted({str(j["group"]) for j in js}, key=lambda v: (len(v), v))
        print(f"member {m}: {len(js):4d} jobs | groups {', '.join(grp)} | {gsec / 3600:6.2f} GPU-h "
              f"| ~{gsec / 3600 / gpus:5.2f} h wall on {gpus} GPU(s)")
    print(f"total: {sum(len(v) for v in plan.values())} jobs, {tot / 3600:.2f} GPU-h "
          f"(rough: per-step times are T4 guesses until calibrated from finished runs)")


# ---------------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------------
def adopt_previous(jobs: list[dict], out: Path, resume_from: list[str]) -> tuple[int, int]:
    """Copy finished / checkpointed run dirs from earlier session outputs into `out`."""
    done = partial = 0
    for j in jobs:
        dst = out / j["name"]
        if (dst / "result.json").exists():
            done += 1
            continue
        for r in resume_from:
            src = Path(r) / j["name"]
            if (src / "result.json").exists() or ((src / "ckpt.pt").exists() and not (dst / "ckpt.pt").exists()):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                if (src / "result.json").exists():
                    done += 1
                else:
                    partial += 1
                break
    return done, partial


# ---------------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------------
def prebuild_banks(jobs: list[dict], workers: int | None) -> None:
    """Build each feature bank once in the parent, so GPU workers never race on it."""
    from islr.models.registry import get_spec
    from islr.lowdata.bank import bank_file, build_bank
    from islr.lowdata.store import load_stores

    tables = {}
    for j in jobs:
        c = j["cfg"]
        tk = (tuple(c["stores"]), tuple(c["sources"] or ()))
        if tk not in tables:
            tables[tk] = load_stores(c["stores"], sources=c["sources"])
        mod = get_spec(c["model"]).modality
        cache = c["cache_dir"] or str(Path(c["out"]) / "_bank")
        if not bank_file(tables[tk], mod, c["num_frames"], c["trim"], cache).exists():
            build_bank(tables[tk], mod, c["num_frames"], c["trim"], cache, workers)


RESTART = 5  # worker exit code: rerun me in a fresh process


def worker(jobs_file: str, slot: int, deadline: float | None) -> int:
    """Runs jobs in-process (bank loaded once) on the GPU given by CUDA_VISIBLE_DEVICES.
    Jobs are claimed through exclusive files, so two workers never run the same job."""
    import traceback

    from islr.lowdata import run as R

    spec = json.loads(Path(jobs_file).read_text())
    claims = Path(spec["claims"])
    claims.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for j in spec["jobs"]:
        out = R.run_dir_of(j["cfg"])
        if (out / "result.json").exists():
            continue
        if deadline and time.time() + j["cost"] * 1.2 > deadline and not (out / "ckpt.pt").exists():
            print(f"[slot {slot}] not enough time left for {j['name']}", flush=True)
            continue
        tag = hashlib.sha1(j["name"].encode()).hexdigest()[:16]
        try:
            fd = os.open(claims / tag, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{slot} {j['name']}".encode())
            os.close(fd)
        except FileExistsError:
            continue
        cfg = dict(j["cfg"], deadline=deadline, quiet=True)
        t0 = time.time()
        try:
            rc = R.run(cfg)
        except Exception as e:  # keep the queue going; the error is kept next to the run
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(traceback.format_exc())
            print(f"[slot {slot}] FAILED {j['name']} (see error.txt)", flush=True)
            n_fail += 1
            rc = 1
            if "CUDA error" in str(e):
                # The CUDA context may be unusable now; let run_member start a fresh worker.
                print(f"[slot {slot}] CUDA error, restarting worker ({n_ok} ok, {n_fail} failed so far)",
                      flush=True)
                return RESTART
        finally:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        if rc == R.INCOMPLETE:
            print(f"[slot {slot}] out of time during {j['name']} (checkpointed)", flush=True)
            return R.INCOMPLETE
        if rc == 0 and (out / "result.json").exists():
            n_ok += 1
            r = json.loads((out / "result.json").read_text())["metrics"]
            print(f"[slot {slot}] {j['name']}: top1 {r['top1']:.3f} target {r['target_top1']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[slot {slot}] finished: {n_ok} ok, {n_fail} failed", flush=True)
    return 0


def run_member(jobs: list[dict], out: Path, gpus: int, deadline: float | None, workers: int | None) -> int:
    session = time.strftime("%Y%m%d-%H%M%S")
    meta = out / "_sweep"
    meta.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / j["name"] / "result.json").exists()]
    print(f"[sweep] {len(todo)} of {len(jobs)} jobs to run", flush=True)
    if not todo:
        return 0
    prebuild_banks(todo, workers)
    solo = sorted([j for j in todo if not j["ddp"]], key=lambda j: -j["cost"])  # longest first
    ddp = [j for j in todo if j["ddp"]]
    rc = 0
    if solo:
        jf = meta / f"jobs_{session}.json"
        jf.write_text(json.dumps({"claims": str(meta / f"claims_{session}"), "jobs": solo}, default=str))
        # one worker process per GPU, restarted after a CUDA error; each worker's output
        # goes to its log file and to this process's stdout
        import threading

        lock = threading.Lock()
        codes = {}

        def slot_loop(s):
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            env["CUDA_VISIBLE_DEVICES"] = str(s) if gpus > 0 else "-1"  # "" is dropped on Windows
            cmd = [sys.executable, str(LAUNCHER), "sweep", "--worker", str(jf), "--slot", str(s)]
            if deadline:
                cmd += ["--deadline", str(deadline)]
            with open(meta / f"slot{s}_{session}.log", "w") as f:
                for _ in range(len(solo) + 1):
                    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1)
                    for line in proc.stdout:
                        f.write(line)
                        f.flush()
                        with lock:
                            print(line, end="", flush=True)
                    codes[s] = proc.wait()
                    if codes[s] != RESTART:
                        break

        threads = [threading.Thread(target=slot_loop, args=(s,), daemon=True) for s in range(max(1, gpus))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rc = max([rc] + [0 if c == RESTART else c for c in codes.values()])
    for j in ddp:
        if deadline and time.time() + j["cost"] / 2 * 1.2 > deadline:
            print(f"[sweep] skipping DDP job {j['name']}: not enough time left", flush=True)
            continue
        jf = meta / (hashlib.sha1(j["name"].encode()).hexdigest()[:12] + ".json")
        jf.write_text(json.dumps(j["cfg"], default=str))
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={max(1, gpus)}",
               str(LAUNCHER), "run", "--config", str(jf)]
        if deadline:
            cmd += ["--deadline", str(deadline)]
        print(f"[sweep] DDP {j['name']}", flush=True)
        r = subprocess.call(cmd)
        rc = max(rc, r)
        if r == 3:
            break
    return rc


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lowdata.py sweep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid")
    p.add_argument("--out", default="sweeps")
    p.add_argument("--member", type=int, default=0)
    p.add_argument("--members", type=int, default=1)
    p.add_argument("--plan", action="store_true", help="print the member split and GPU-hour estimate")
    p.add_argument("--gpus", type=int, default=None, help="default: all visible GPUs")
    p.add_argument("--time-budget-h", type=float, default=None)
    p.add_argument("--resume-from", nargs="*", default=[])
    p.add_argument("--cache-dir", default=None, help="feature-bank cache (default <out>/_bank)")
    p.add_argument("--workers", type=int, default=None, help="CPU processes for bank building")
    p.add_argument("--only", nargs="*", help="run only jobs whose name contains one of these strings")
    # internal
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--slot", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.worker:
        return worker(a.worker, a.slot, a.deadline)
    if not a.grid:
        p.error("--grid is required")
    t0 = time.time()
    grid = load_grid(a.grid)
    out = Path(a.out)
    jobs = expand_jobs(grid, str(out), a.cache_dir)
    if a.only:
        jobs = [j for j in jobs if any(s in j["name"] for s in a.only)]
    estimate_steps(jobs)
    sec = calibrate([out] + [Path(r) for r in a.resume_from])
    for j in jobs:
        j["cost"] = job_cost(j, sec)
    plan = assign_members(jobs, a.members, grid.get("assign"))
    gpus = a.gpus if a.gpus is not None else _gpu_count()
    if a.plan:
        print(f"grid {grid['name']}: {len(jobs)} jobs, split by {grid['split_by']} over {a.members} member(s)"
              + (f"; calibrated models: {', '.join(sorted(sec))}" if sec else ""))
        print_plan(plan, max(1, gpus or 2))
        return 0
    mine = plan[a.member]
    out.mkdir(parents=True, exist_ok=True)
    (out / "_sweep").mkdir(exist_ok=True)
    (out / "_sweep" / f"plan_member{a.member}.json").write_text(json.dumps(
        {"grid": grid, "member": a.member, "members": a.members,
         "jobs": [{k: j[k] for k in ("name", "group", "cost", "steps") if k in j} for j in mine]},
        indent=1, default=str))
    if a.resume_from:
        d, pt = adopt_previous(mine, out, a.resume_from)
        print(f"[sweep] from previous sessions: {d} finished, {pt} partial", flush=True)
    deadline = t0 + 3600 * a.time_budget_h - DEADLINE_MARGIN_S if a.time_budget_h else None
    rc = run_member(mine, out, gpus, deadline, a.workers)
    left = [j["name"] for j in mine if not (out / j["name"] / "result.json").exists()]
    print(f"[sweep] member {a.member}: {len(mine) - len(left)}/{len(mine)} runs finished"
          + (f"; {len(left)} left - rerun with --resume-from pointing at this output" if left else ""), flush=True)
    return 3 if left and rc in (0, 3) else rc


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())

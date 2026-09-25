"""One low-data training run: split -> feature bank -> fixed-budget training -> test.

    python lowdata.py run --stores S1 S2 --model partformer --protocol scarce --shots 4 --seed 0 --out runs
    python lowdata.py run --config job.json                  # same keys as the CLI (see DEFAULTS)
    torchrun --standalone --nproc_per_node=2 lowdata.py run ...   # DDP over both T4s

Training recipe (identical for every model, so K is the only thing that changes)
  * no validation set and no early stopping: with K <= 8 clips per word a validation
    split would eat the training data and select on noise. Every run trains for a
    fixed number of optimiser steps: clamp(epochs * n_train / batch, min_steps, max_steps)
  * AdamW, linear warmup (5 %) + cosine decay, grad clip 1.0, fp16 autocast + GradScaler
    on CUDA (off for models with use_amp=False)
  * batches are drawn from a stateless, seeded stream (step -> indices), so a resumed
    run sees exactly the batches it would have seen, and all DDP ranks agree on the
    global batch; each rank takes its slice
  * GPU augmentation (`augment.py`) on the training batch only
Evaluation runs on rank 0 on the fixed signer-independent test set (`protocol.py`).

Resuming (Kaggle's 12 h limit): a checkpoint is written every `ckpt_minutes` and when
`deadline` / `time_budget_h` is reached. The run then exits with code 3 (INCOMPLETE);
running the same command again continues from the checkpoint. A finished run writes
result.json and is skipped next time unless --force.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

INCOMPLETE = 3

DEFAULTS = dict(
    stores=[],
    sources=None,
    model="partformer",
    # split (protocol.SplitConfig)
    protocol="scarce", shots=4, seed=0, targets="legacy8", rich_shots=None, n_words=None, words=None,
    min_clips=3, test_frac=0.25, min_test=2, max_test=20, max_test_frac=0.5,
    train_sources=None, test_sources=None,
    # training
    num_frames=30, trim=True, batch_size=None, lr=None, weight_decay=None, epochs=None,
    min_steps=300, max_steps=6000, warmup_frac=0.05, balance=False, augment=True, amp=True,
    hparams={},  # model hyper-parameter overrides (registry defaults otherwise)
    init_from=None,  # checkpoint/state_dict to initialise from (matching shapes only)
    # io / runtime
    out="runs", cache_dir=None, run_dir=None, device=None, workers=None, eval_bs=256,
    ckpt_minutes=20.0, time_budget_h=None, deadline=None, stop_after_steps=None,
    save_preds=True, force=False, quiet=False,
)
SPLIT_KEYS = ("protocol", "shots", "seed", "targets", "rich_shots", "n_words", "words", "min_clips",
              "test_frac", "min_test", "max_test", "max_test_frac", "train_sources", "test_sources")


# ---------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------
def _none_int(v):
    return None if v in (None, "", "none", "None", "all") else int(v)


def parse_args(argv):
    p = argparse.ArgumentParser(prog="lowdata.py run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file with any of the keys below")
    p.add_argument("--stores", nargs="+")
    p.add_argument("--sources", nargs="+")
    p.add_argument("--model")
    p.add_argument("--protocol")
    p.add_argument("--shots", type=_none_int)
    p.add_argument("--seed", type=int)
    p.add_argument("--targets")
    p.add_argument("--rich-shots", dest="rich_shots", type=_none_int)
    p.add_argument("--n-words", dest="n_words", type=_none_int)
    p.add_argument("--train-sources", dest="train_sources", nargs="+")
    p.add_argument("--test-sources", dest="test_sources", nargs="+")
    p.add_argument("--num-frames", dest="num_frames", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--min-steps", dest="min_steps", type=int)
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--balance", action="store_true", default=None)
    p.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    p.add_argument("--hparams", type=json.loads, help='JSON dict, e.g. \'{"d_model": 96}\'')
    p.add_argument("--init-from", dest="init_from")
    p.add_argument("--out")
    p.add_argument("--run-dir", dest="run_dir")
    p.add_argument("--cache-dir", dest="cache_dir")
    p.add_argument("--device")
    p.add_argument("--workers", type=int)
    p.add_argument("--ckpt-minutes", dest="ckpt_minutes", type=float)
    p.add_argument("--time-budget-h", dest="time_budget_h", type=float)
    p.add_argument("--deadline", type=float, help="unix time at which to checkpoint and stop")
    p.add_argument("--stop-after-steps", dest="stop_after_steps", type=int, help="debug: simulate a timeout")
    p.add_argument("--force", action="store_true", default=None)
    p.add_argument("--quiet", action="store_true", default=None)
    return p.parse_args(argv)


def make_config(args=None, **over) -> dict:
    cfg = dict(DEFAULTS)
    if args is not None and getattr(args, "config", None):
        cfg.update(json.loads(Path(args.config).read_text()))
    if args is not None:
        cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    cfg.update(over)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown run options: {sorted(unknown)}")
    if isinstance(cfg["stores"], str):
        cfg["stores"] = [cfg["stores"]]
    return cfg


def run_name(cfg: dict) -> str:
    """<protocol tag>/<model>/K<shots>/s<seed> - also used by sweep.py to find results."""
    from islr.models.registry import canonical_name
    from islr.lowdata.protocol import split_config_from

    sc = split_config_from({k: cfg[k] for k in SPLIT_KEYS})
    k = "all" if cfg["shots"] is None else cfg["shots"]
    return f"{sc.tag()}/{canonical_name(cfg['model'])}/K{k}/s{cfg['seed']}"


def run_dir_of(cfg: dict) -> Path:
    return Path(cfg["run_dir"]) if cfg.get("run_dir") else Path(cfg["out"]) / run_name(cfg)


# ---------------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------------
class Dist:
    def __init__(self, device_arg=None):
        import torch

        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        cuda = torch.cuda.is_available() and (device_arg is None or str(device_arg).startswith("cuda"))
        if cuda and self.world > 1 and self.local >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK {self.local} but only {torch.cuda.device_count()} GPU(s) visible")
        if cuda:
            idx = self.local if self.world > 1 else (torch.device(device_arg).index or 0 if device_arg else 0)
            torch.cuda.set_device(idx)
            self.device = torch.device("cuda", idx)
        else:
            self.device = torch.device("cpu")
        if self.world > 1:
            import torch.distributed as dist

            if sys.platform == "win32":
                os.environ.setdefault("USE_LIBUV", "0")
            if not dist.is_initialized():
                dist.init_process_group("nccl" if cuda else "gloo")

    @property
    def main(self) -> bool:
        return self.rank == 0

    def barrier(self):
        if self.world > 1:
            import torch.distributed as dist

            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.device.index])
            else:
                dist.barrier()

    def any(self, flag: bool) -> bool:
        if self.world == 1:
            return flag
        import torch
        import torch.distributed as dist

        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def close(self):
        if self.world > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


# ---------------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------------
class BatchStream:
    """Stateless batch indices: batch(step) depends only on (seed, step).

    shuffle: consecutive per-epoch permutations of the training set.
    balance: every batch draws classes uniformly, then a clip of that class.
    """

    def __init__(self, labels: np.ndarray, bs: int, seed: int, balance: bool):
        self.labels, self.bs, self.seed, self.balance = np.asarray(labels), bs, seed, balance
        self.n = len(labels)
        self.by_class = [np.where(self.labels == c)[0] for c in np.unique(self.labels)]
        self._perm_cache: dict[int, np.ndarray] = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm_cache:
            if len(self._perm_cache) > 4:
                self._perm_cache.clear()
            self._perm_cache[epoch] = np.random.default_rng([self.seed, 17, epoch]).permutation(self.n)
        return self._perm_cache[epoch]

    def batch(self, step: int) -> np.ndarray:
        if self.balance:
            rng = np.random.default_rng([self.seed, 29, step])
            cls = rng.integers(0, len(self.by_class), self.bs)
            return np.array([self.by_class[c][rng.integers(0, len(self.by_class[c]))] for c in cls])
        start = step * self.bs
        out = []
        while len(out) < self.bs:
            e, o = divmod(start + len(out), self.n)
            take = self._perm(e)[o: o + self.bs - len(out)]
            out.extend(take.tolist())
        return np.asarray(out)


def budget(n_train: int, bs: int, epochs: float, min_steps: int, max_steps: int) -> int:
    return int(min(max_steps, max(min_steps, math.ceil(epochs * n_train / bs))))


def effective_batch(bs: int, n_train: int, world: int) -> int:
    """Global batch: <= n_train (no point repeating clips), >= 2 per rank (BatchNorm),
    divisible by the number of ranks."""
    b = max(min(bs, n_train), 2 * world)
    return int(math.ceil(b / world) * world)


# ---------------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------------
def build_model(spec, num_classes: int, hp: dict):
    return spec.build(num_classes, hp)


def load_matching(model, path: str) -> dict:
    """Initialise from a checkpoint/state_dict, skipping tensors whose shape differs
    (e.g. the classifier when the vocabulary changes)."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "skipped": len(sd) - len(ok), "own": len(own)}


def forward_logits(spec, model, xs, y, criterion, device, train: bool):
    """(logits, loss) for a batch of bank tensors, through the model's forward_fn."""
    x = xs if len(xs) > 1 else xs[0]
    if spec.forward_fn is not None:
        logits, _, loss = spec.forward_fn(model, (x, y), criterion, device, train=train)
        return logits, loss
    logits = model(x)
    return logits, criterion(logits, y)


def find_head(model, num_classes: int, spec, xs, device):
    """The classifier layer used for prototype evaluation: the last module called in
    a forward pass that maps to `num_classes` outputs (auxiliary/Koopman heads excluded).
    Its input is the clip embedding."""
    import torch

    cands = [(n, m) for n, m in model.named_modules()
             if getattr(m, "out_features", None) == num_classes
             and not any(s in n.lower() for s in ("aux", "koopman"))]
    order, hooks = [], []
    for n, m in cands:
        hooks.append(m.register_forward_hook(lambda mod, inp, out, n=n: order.append(n)))
    with torch.no_grad():
        forward_logits(spec, model, tuple(t[:2] for t in xs), torch.zeros(min(2, len(xs[0])), dtype=torch.long,
                                                                          device=device),
                       torch.nn.CrossEntropyLoss(), device, train=False)
    for h in hooks:
        h.remove()
    return dict(cands)[order[-1]] if order else None


def predict(spec, model, bank, idx: np.ndarray, device, bs: int, use_amp: bool, head=None):
    """Logits (and head-input embeddings if `head` is given) for the clips `idx`."""
    import torch
    from torch.amp import autocast

    model.eval()
    feats, logits = [], []
    hook = head.register_forward_hook(lambda m, inp, out: feats.append(inp[0].detach().float().cpu())) \
        if head is not None else None
    crit = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            xs = bank.gather(torch.as_tensor(idx[s: s + bs]), device)
            ii = xs[0]
            y = torch.zeros(len(ii), dtype=torch.long, device=device)
            with autocast("cuda", enabled=use_amp):
                lg, _ = forward_logits(spec, model, xs, y, crit, device, train=False)
            logits.append(lg.float().cpu())
    if hook is not None:
        hook.remove()
    lg = torch.cat(logits).numpy() if logits else np.zeros((0, 0), np.float32)
    emb = torch.cat(feats).numpy() if feats else None
    return lg, emb


def prototype_predict(train_emb, train_y, test_emb) -> np.ndarray:
    """Nearest class mean (cosine) in the embedding space of the trained model."""
    def _n(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    classes = np.unique(train_y)
    protos = _n(np.stack([_n(train_emb[train_y == c]).mean(0) for c in classes]))
    return classes[np.argmax(_n(test_emb) @ protos.T, axis=1)]


def _make_scaler(enabled: bool):
    try:
        from torch.amp import GradScaler

        return GradScaler("cuda", enabled=enabled)
    except (ImportError, TypeError):  # older torch
        from torch.cuda.amp import GradScaler

        return GradScaler(enabled=enabled)


def _rng_state():
    import torch

    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    import torch

    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except RuntimeError:
            pass


def _seed_all(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_MEMO: dict = {}


def _memo(key, fn, evict: str | None = None):
    """In-process cache used when a sweep worker runs many jobs in one process:
    the clip table and the current feature bank are loaded once. `evict` drops other
    entries of the same kind first (one bank on the GPU at a time)."""
    if key not in _MEMO:
        if evict:
            for k in [k for k in _MEMO if k[0] == evict]:
                del _MEMO[k]
        _MEMO[key] = fn()
    return _MEMO[key]


def _write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------
def run(cfg: dict) -> int:
    import torch
    import torch.nn as nn
    from torch.amp import autocast

    from islr.models.registry import get_spec
    from islr.lowdata.augment import augment
    from islr.lowdata.bank import load_bank
    from islr.lowdata.metrics import classification_metrics
    from islr.lowdata.protocol import make_split, split_config_from
    from islr.lowdata.store import load_stores

    t_start = time.time()
    deadline = cfg["deadline"]
    if cfg["time_budget_h"]:
        deadline = min(deadline or math.inf, t_start + 3600 * float(cfg["time_budget_h"]))
    dist_ = Dist(cfg["device"])
    dev = dist_.device
    log = (lambda *a: print(*a, flush=True)) if dist_.main and not cfg["quiet"] else (lambda *a: None)
    try:
        spec = get_spec(cfg["model"])
        rdir = run_dir_of(cfg)
        if (rdir / "result.json").exists() and not cfg["force"]:
            log(f"[run] done already: {rdir}")
            return 0
        if dist_.main:
            rdir.mkdir(parents=True, exist_ok=True)

        # ---- data
        df = _memo(("df", tuple(cfg["stores"]), tuple(cfg["sources"] or ())),
                   lambda: load_stores(cfg["stores"], sources=cfg["sources"]))
        split = make_split(df, split_config_from({k: cfg[k] for k in SPLIT_KEYS}))
        if not len(split.train_idx) or not len(split.test_idx):
            raise RuntimeError(f"empty split: {split.info['n_train']} train / {split.info['n_test']} test")
        lab = split.label_of
        y_all = np.array([lab.get(w, -1) for w in df["word"]], dtype=np.int64)
        tr_idx, te_idx = split.train_idx, split.test_idx
        # One bank per (clip table, modality): every K, seed and protocol of a sweep
        # reuses it. It lives on the GPU unless it is large (e.g. all of GISLR).
        pos_tr, pos_te, y_sub = tr_idx, te_idx, y_all
        C = len(split.words)
        cache = cfg["cache_dir"] or str(Path(cfg["out"]) / "_bank")
        def _load():
            b = load_bank(df, spec.modality, cfg["num_frames"], cfg["trim"], cache, dist_.rank, dist_.world,
                          cfg["workers"])
            return b.to(dev if dev.type == "cuda" and b.nbytes < 3e9 else "cpu")

        bank = _memo(("bank", id(df), spec.modality, cfg["num_frames"], cfg["trim"], cache, str(dev)), _load,
                     evict="bank")
        y_dev = torch.as_tensor(y_sub, device=dev)
        if dist_.main:
            _write_json(rdir / "split.json", dict(split.info, words=split.words, run=run_name(cfg),
                                                  test_keys=df.loc[te_idx, "key"].tolist(),
                                                  train_keys=df.loc[tr_idx, "key"].tolist()))

        # ---- model
        hp = dict(spec.defaults)
        hp.update(cfg["hparams"] or {})
        hp["num_frames"] = cfg["num_frames"]
        for k in ("batch_size", "lr", "weight_decay", "epochs"):
            if cfg[k] is not None:
                hp[k] = cfg[k]
        _seed_all(1000 * int(cfg["seed"]) + 7)
        model = build_model(spec, C, hp)
        init_info = load_matching(model, cfg["init_from"]) if cfg["init_from"] else None
        model.to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        net = model
        if dist_.world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if dev.type == "cuda":
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            net = DDP(model, device_ids=[dev.index] if dev.type == "cuda" else None,
                      find_unused_parameters=True)
            for a in ("mixup", "label_smoothing", "koopman_weight", "aux_weight"):
                if hasattr(model, a):
                    setattr(net, a, getattr(model, a))

        n_train = len(tr_idx)
        bs = effective_batch(int(hp.get("batch_size", 16)), n_train, dist_.world)
        steps = budget(n_train, bs, float(hp.get("epochs", 80)), cfg["min_steps"], cfg["max_steps"])
        per_rank = bs // dist_.world
        lr = float(hp.get("lr", 1e-3))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(hp.get("weight_decay", 1e-2)))
        warm = max(1, int(cfg["warmup_frac"] * steps))

        def lr_at(s):
            if s < warm:
                return (s + 1) / warm
            p = (s - warm) / max(1, steps - warm)
            return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        use_amp = bool(cfg["amp"] and spec.use_amp and dev.type == "cuda")
        scaler = _make_scaler(use_amp)
        criterion = nn.CrossEntropyLoss()
        stream = BatchStream(y_sub[pos_tr], bs, int(cfg["seed"]), bool(cfg["balance"]))

        # ---- resume
        ckpt = rdir / "ckpt.pt"
        step = 0
        if ckpt.exists() and not cfg["force"]:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            if blob.get("steps") == steps and blob.get("run") == run_name(cfg):
                model.load_state_dict(blob["model"])
                opt.load_state_dict(blob["opt"])
                sched.load_state_dict(blob["sched"])
                scaler.load_state_dict(blob["scaler"])
                step = int(blob["step"])
                if dist_.main:
                    _set_rng_state(blob["rng"])
                log(f"[run] resumed at step {step}/{steps}")
            else:
                log("[run] checkpoint does not match this run; starting over")

        def save_ckpt():
            if dist_.main:
                tmp = ckpt.with_name("ckpt.pt.tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step, "steps": steps, "run": run_name(cfg),
                            "rng": _rng_state()}, tmp)
                os.replace(tmp, ckpt)
            dist_.barrier()

        log(f"[run] {run_name(cfg)} | {spec.modality} | C={C} train={n_train} test={len(te_idx)} | "
            f"bs={bs} steps={steps} amp={use_amp} world={dist_.world} params={n_params / 1e6:.2f}M")

        # ---- train
        net.train()
        t_ck = time.time()
        loss_hist, t_train = [], time.time()
        stopped = False
        while step < steps:
            gidx = stream.batch(step)
            mine = gidx[dist_.rank * per_rank:(dist_.rank + 1) * per_rank]
            ii = torch.as_tensor(pos_tr[mine], device=dev)
            xs = bank.gather(ii, dev)
            y = y_dev.index_select(0, ii)
            if cfg["augment"]:
                xs = augment(spec.modality, xs)
            with autocast("cuda", enabled=use_amp):
                _, loss = forward_logits(spec, net, xs, y, criterion, dev, train=True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            loss_hist.append(float(loss.detach()))
            if dist_.main and not cfg["quiet"] and (step % max(1, steps // 10) == 0 or step == steps):
                log(f"  step {step:5d}/{steps} loss {np.mean(loss_hist[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e}")
            if step < steps and step % 10 == 0 or cfg["stop_after_steps"] == step:
                now = time.time()
                late = (deadline is not None and now > deadline) or cfg["stop_after_steps"] == step
                if dist_.any(late):
                    stopped = True
                    break
                if dist_.any(now - t_ck > 60 * cfg["ckpt_minutes"]):
                    save_ckpt()
                    t_ck = time.time()
        train_s = time.time() - t_train
        if stopped:
            save_ckpt()
            if dist_.main:
                _write_json(rdir / "status.json", {"state": "incomplete", "step": step, "steps": steps})
            log(f"[run] time budget reached at step {step}/{steps}; checkpoint saved, rerun to resume")
            return INCOMPLETE

        # ---- evaluate (rank 0)
        if dist_.main:
            head = find_head(model, C, spec, bank.gather(torch.as_tensor(pos_te[:2]), dev), dev)
            lg_te, emb_te = predict(spec, model, bank, pos_te, dev, cfg["eval_bs"], use_amp, head)
            y_te = y_sub[pos_te]
            proto = None
            if emb_te is not None:
                _, emb_tr = predict(spec, model, bank, pos_tr, dev, cfg["eval_bs"], use_amp, head)
                proto = prototype_predict(emb_tr, y_sub[pos_tr], emb_te)
            pred = lg_te.argmax(1)
            m = classification_metrics(y_te, pred, split.words, split.targets, lg_te, proto)
            m["chance"] = 1.0 / C
            res = {
                "run": run_name(cfg), "model": spec.name, "family": spec.family, "modality": spec.modality,
                "protocol": cfg["protocol"], "tag": split.info["tag"], "shots": cfg["shots"], "seed": cfg["seed"],
                "n_words": C, "n_targets": len(split.targets), "n_train": n_train, "n_params": n_params,
                "steps": steps, "batch_size": bs, "world": dist_.world, "amp": use_amp,
                "train_seconds": train_s, "final_loss": float(np.mean(loss_hist[-50:])) if loss_hist else None,
                "test_fingerprint": split.info["test_fingerprint"], "init": init_info,
                "hparams": {k: v for k, v in hp.items() if isinstance(v, (int, float, str, bool, type(None)))},
                "config": cfg, "metrics": m,
                "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            }
            if cfg["save_preds"]:
                np.savez_compressed(rdir / "preds.npz", keys=df.loc[te_idx, "key"].to_numpy(), y=y_te, pred=pred,
                                    proto=proto if proto is not None else np.zeros(0), logits=lg_te.astype(np.float16),
                                    words=np.array(split.words))
            torch.save({"model": model.state_dict(), "words": split.words, "spec": spec.name, "hparams": res["hparams"]},
                       rdir / "model.pt")
            _write_json(rdir / "result.json", res)
            if ckpt.exists():
                ckpt.unlink()
            st = rdir / "status.json"
            if st.exists():
                st.unlink()
            log(f"[run] top1 {m['top1']:.3f} target {m['target_top1']:.3f} rich {m['rich_top1']:.3f} "
                f"proto {m.get('proto_top1', float('nan')):.3f} (chance {1 / C:.3f}) {train_s:.0f}s")
        dist_.barrier()
        return 0
    finally:
        dist_.close()


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg = make_config(args)
    if not cfg["stores"]:
        print("need --stores (or 'stores' in --config)")
        return 2
    return run(cfg)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())

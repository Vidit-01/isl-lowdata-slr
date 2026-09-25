"""Video -> MediaPipe Holistic -> landmark store (see store.py). CPU bound, resumable.

Works with both MediaPipe APIs:
  * legacy `mp.solutions.holistic` (present up to mediapipe 0.10.21, which the Kaggle
    notebook pins). Preferred: the old pipeline used it, and it does not have the Tasks
    graph's "Check failed: holder_ != nullptr The packet is empty" abort.
  * Tasks API `HolisticLandmarker`, used only when `solutions` is missing (newer
    mediapipe) or MP_BACKEND=tasks. Needs `holistic_landmarker.task`, downloaded on first use.

One MediaPipe graph per worker process and a fresh graph per video, so tracking state
never leaks between clips. A MediaPipe C++ abort kills only its worker: that clip is
recorded in failed.txt and the worker is replaced. Clips whose .npy already exists are
skipped, so a killed run (Kaggle 12 h limit) resumes where it stopped.

    python lowdata.py extract --videos-csv videos.csv --store stores/mine
    python lowdata.py extract --folder my_videos/ --source mine --store stores/mine
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .store import FACE0, LH0, N_FACE, N_HAND, N_LM, N_POSE, POSE0, RH0, append_index, clip_key, init_store

TASK_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
            "holistic_landmarker/float16/latest/holistic_landmarker.task")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mts"}


def ensure_task_model(path: str | None = None) -> str:
    cands = [path, os.environ.get("HOLISTIC_TASK_PATH"),
             Path.home() / ".cache" / "isl_lowdata" / "holistic_landmarker.task"]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    dst = Path(path) if path else Path(cands[-1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[extract] downloading {TASK_URL} -> {dst}", flush=True)
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(TASK_URL, tmp)
    tmp.replace(dst)
    return str(dst)


def _backend() -> str:
    forced = os.environ.get("MP_BACKEND")
    if forced:
        return forced
    try:
        import mediapipe as mp

        if hasattr(getattr(mp, "solutions", None), "holistic"):
            return "solutions"
    except Exception:
        pass
    return "tasks"


def _fill(out: np.ndarray, t: int, start: int, count: int, lms) -> None:
    if lms is None:
        return
    seq = lms.landmark if hasattr(lms, "landmark") else lms
    if not seq:
        return
    pts = np.array([[p.x, p.y, p.z] for p in list(seq)[:count]], dtype=np.float32)
    out[t, start:start + len(pts)] = pts


def _store_result(res, out: np.ndarray, t: int) -> None:
    _fill(out, t, POSE0, N_POSE, res.pose_landmarks)
    _fill(out, t, LH0, N_HAND, res.left_hand_landmarks)
    _fill(out, t, RH0, N_HAND, res.right_hand_landmarks)
    _fill(out, t, FACE0, N_FACE, res.face_landmarks)


_W: dict = {}


def _init_worker(backend: str, task_path: str | None, max_side: int) -> None:
    import cv2

    cv2.setNumThreads(1)
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _W.update(backend=backend, task_path=task_path, max_side=max_side)


def auto_stride(fps: float, target_fps: float = 15.0) -> int:
    return max(1, int(round((fps or 30.0) / target_fps)))


def extract_video(path: str, stride: int | None = None, max_frames: int = 256) -> tuple[np.ndarray, float]:
    """Returns ((T, 543, 3) float32 with NaN for missing, fps)."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():  # freshly written files are occasionally still locked (Windows)
        time.sleep(2)
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = stride or auto_stride(fps)
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = fr.shape[:2]
            s = _W["max_side"] / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    out = np.full((len(frames), N_LM, 3), np.nan, dtype=np.float32)
    if _W["backend"] == "tasks":
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_W["task_path"]),
            running_mode=vision.RunningMode.VIDEO,
        )
        with vision.HolisticLandmarker.create_from_options(opts) as det:
            for t, rgb in enumerate(frames):
                ts = int(t * stride * 1000.0 / fps) + t  # strictly increasing
                res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                _store_result(res, out, t)
    else:
        with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                            refine_face_landmarks=False) as det:
            for t, rgb in enumerate(frames):
                _store_result(det.process(rgb), out, t)
    return out, float(fps) / stride


def _worker_loop(slot, tasks, results, current, backend, task_path, max_side, job):
    """Runs in a worker process. `current[slot]` is shared memory, written before each
    clip, so the parent still knows which clip was in flight if MediaPipe aborts."""
    _init_worker(backend, task_path, max_side)
    while True:
        item = tasks.get()
        if item is None:
            return
        idx, args = item
        current[slot] = idx
        results.put((idx, job(args)))
        current[slot] = -1


def _run_pool(todo, workers, backend, task, max_side, time_budget_s, on_result, job=None):
    """Like Pool.imap_unordered(_job, todo), but survives workers that die (MediaPipe
    CHECK failures call abort(), which no try/except can catch and which makes a
    multiprocessing.Pool wait forever for the lost task). Returns True if stopped by
    the time budget."""
    from multiprocessing import get_context

    ctx = get_context("spawn")
    # SimpleQueue writes straight to the pipe: a result put just before the next clip's
    # abort is not lost (Queue's feeder thread would die with the unsent message)
    tasks, results = ctx.Queue(), ctx.SimpleQueue()
    current = ctx.Array("l", [-1] * workers, lock=False)
    for item in enumerate(todo):
        tasks.put(item)
    for _ in range(workers):
        tasks.put(None)

    def start(slot):
        pr = ctx.Process(target=_worker_loop, daemon=True,
                         args=(slot, tasks, results, current, backend, task, max_side, job or _job))
        pr.start()
        return pr

    procs = [start(k) for k in range(workers)]
    seen: set[int] = set()
    t0 = time.time()

    def deliver(idx, res):
        if idx not in seen:
            seen.add(idx)
            on_result(res)

    try:
        while True:
            if time_budget_s and time.time() - t0 > time_budget_s:
                return True
            if not results.empty():
                deliver(*results.get())
                continue
            time.sleep(0.5)
            alive = False
            for k, pr in enumerate(procs):
                if pr.is_alive():
                    alive = True
                elif pr.exitcode != 0:  # died mid-clip: record it, replace the worker
                    idx = current[k]
                    current[k] = -1
                    if idx >= 0:
                        deliver(idx, (todo[idx][1], 0, 0.0,
                                      f"worker crashed (exit code {pr.exitcode}): MediaPipe abort on {todo[idx][0]}"))
                    procs[k] = start(k)
                    alive = True
            if not alive:  # every worker got its sentinel and exited cleanly
                while not results.empty():
                    deliver(*results.get())
                return False
    finally:
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        tasks.cancel_join_thread()  # unsent clips after a budget stop must not block exit


def _job(args):
    video, npy, stride = args
    try:
        arr, fps = extract_video(video, stride)
        tmp = npy + ".tmp.npy"
        np.save(tmp, arr.astype(np.float16))
        os.replace(tmp, npy)
        return npy, arr.shape[0], fps, None
    except Exception as exc:  # recorded, never fatal for the whole run
        return npy, 0, 0.0, f"{type(exc).__name__}: {exc}"


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """'1/4' -> (1, 4): this machine does shard 1 of 4 (0-based)."""
    if not text:
        return None
    i, n = (int(x) for x in str(text).split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {text!r}: need 0 <= i < n")
    return i, n


def in_shard(key: str, shard: tuple[int, int] | None) -> bool:
    """Stable assignment of a clip to one of n shards (same on every machine)."""
    import zlib

    return shard is None or zlib.crc32(key.encode()) % shard[1] == shard[0]


def extract_to_store(videos: pd.DataFrame, store: str | Path, workers: int | None = None,
                     stride: int | None = None, max_side: int = 640,
                     task_path: str | None = None, time_budget_s: float | None = None,
                     shard: tuple[int, int] | None = None) -> pd.DataFrame:
    """videos: columns video_path, word, source[, signer, session, split, video_rel, license].

    shard=(i, n) keeps only the clips of shard i, so n people can extract one corpus
    in parallel into n stores and merge them (`sources merge`)."""
    store = init_store(store)
    v = videos.copy()
    for col in ("signer", "session", "split", "license"):
        if col not in v:
            v[col] = ""
    if "video_rel" not in v:
        v["video_rel"] = v["video_path"]
    v["key"] = [clip_key(s, r) for s, r in zip(v["source"], v["video_rel"])]
    v["path"] = "npy/" + v["key"] + ".npy"
    if shard is not None:
        v = v[[in_shard(k, shard) for k in v["key"]]].reset_index(drop=True)
        print(f"[extract] shard {shard[0]}/{shard[1]}: {len(v)} videos", flush=True)
    todo = [(p, str(store / q), stride) for p, q in zip(v["video_path"], v["path"])
            if not (store / q).exists()]
    backend = _backend()
    task = ensure_task_model(task_path) if backend == "tasks" else None
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[extract] {len(v)} videos, {len(todo)} to do, backend={backend}, workers={workers}", flush=True)
    fps_by_path: dict[str, float] = {}
    failed = []
    t0 = time.time()
    if todo:
        count = [0]

        def on_result(r):
            npy, n, fps, err = r
            count[0] += 1
            if err:
                failed.append((npy, err))
                if err.startswith("worker crashed"):
                    print(f"[extract] {err}", flush=True)
            else:
                fps_by_path[npy] = fps
            if count[0] % 20 == 0 or count[0] == len(todo):
                rate = count[0] / max(1e-6, time.time() - t0)
                print(f"[extract] {count[0]}/{len(todo)} {rate:.2f} vid/s failed={len(failed)}", flush=True)

        if _run_pool(todo, workers, backend, task, max_side, time_budget_s, on_result):
            print("[extract] time budget reached; stopping (re-run to resume)", flush=True)
    if failed:
        with open(store / "failed.txt", "a", encoding="utf-8") as f:
            f.writelines(f"{p}\t{e}\n" for p, e in failed)
    done = v[[(store / p).exists() for p in v["path"]]].copy()
    done["n_frames"] = [np.load(store / p, mmap_mode="r").shape[0] for p in done["path"]]
    done["fps"] = [f"{fps_by_path.get(str(store / p), 0.0):.2f}" for p in done["path"]]
    append_index(store, done)
    print(f"[extract] store {store}: +{len(done)} clips indexed, {len(failed)} failed", flush=True)
    return done


def videos_from_folder(root: str | Path, source: str) -> pd.DataFrame:
    """<root>/<word>/<video> tree (word = parent folder)."""
    root = Path(root)
    rows = [{"video_path": str(p), "video_rel": p.relative_to(root).as_posix(),
             "word": p.parent.name, "source": source}
            for p in sorted(root.rglob("*")) if p.suffix.lower() in VIDEO_EXTS]
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos-csv")
    g.add_argument("--folder")
    ap.add_argument("--source", default="custom")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None, help="frame stride (default: ~15 fps)")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--task-path", default=None)
    ap.add_argument("--shard", default=None, help="i/n: extract only shard i of n (0-based)")
    ap.add_argument("--time-budget-h", type=float, default=None)
    a = ap.parse_args(argv)
    if a.folder:
        vids = videos_from_folder(a.folder, a.source)
    else:
        vids = pd.read_csv(a.videos_csv, dtype=str, keep_default_na=False)
        if "source" not in vids:
            vids["source"] = a.source
    extract_to_store(vids, a.store, a.workers, a.stride, a.max_side, a.task_path,
                     a.time_budget_h * 3600 if a.time_budget_h else None, parse_shard(a.shard))


if __name__ == "__main__":
    sys.exit(main())

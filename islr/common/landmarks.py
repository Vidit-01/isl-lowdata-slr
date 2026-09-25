"""MediaPipe Holistic landmark extraction + positional normalization."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Holistic landmark counts
N_POSE = 33
N_HAND = 21
N_FACE = 468
FEAT_DIM = (N_POSE + 2 * N_HAND + N_FACE) * 3  # xyz only


def _lm_to_array(landmarks, n: int) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.float32)
    if landmarks is None:
        return out
    # Classic solutions: object with .landmark list
    # Tasks API: plain list of landmarks (or Nested under .landmark)
    seq = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    if seq is None:
        return out
    # Some Tasks results nest an extra list level for multi-face
    if len(seq) > 0 and isinstance(seq[0], (list, tuple)):
        seq = seq[0]
    for i, lm in enumerate(list(seq)[:n]):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def sample_frame_indices(n_frames: int, target: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(target, dtype=np.int64)
    if n_frames >= target:
        return np.linspace(0, n_frames - 1, target).astype(np.int64)
    # pad by repeating last
    idx = np.arange(n_frames)
    pad = np.full(target - n_frames, n_frames - 1, dtype=np.int64)
    return np.concatenate([idx, pad])


def normalize_landmarks(frame_xyz: np.ndarray) -> np.ndarray:
    """Positional independence: center on mid-hip, scale by shoulder width.

    frame_xyz: (N_LM, 3) concatenated pose|lh|rh|face in Holistic order.
    Pose indices: 11 L-shoulder, 12 R-shoulder, 23 L-hip, 24 R-hip.
    """
    pose = frame_xyz[:N_POSE]
    lh = frame_xyz[N_POSE : N_POSE + N_HAND]
    rh = frame_xyz[N_POSE + N_HAND : N_POSE + 2 * N_HAND]
    face = frame_xyz[N_POSE + 2 * N_HAND :]

    # Mid hip as origin (fallback to mid-shoulder)
    l_hip, r_hip = pose[23], pose[24]
    l_sh, r_sh = pose[11], pose[12]
    if np.any(l_hip) or np.any(r_hip):
        origin = 0.5 * (l_hip + r_hip)
    else:
        origin = 0.5 * (l_sh + r_sh)

    scale = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    if scale < 1e-3:
        scale = 1.0

    def norm_block(block: np.ndarray) -> np.ndarray:
        b = block.copy()
        present = np.any(b != 0, axis=1)
        b[present] = (b[present] - origin) / scale
        return b

    return np.concatenate(
        [norm_block(pose), norm_block(lh), norm_block(rh), norm_block(face)],
        axis=0,
    ).astype(np.float32)


def _silence_mediapipe_logs() -> None:
    import os
    import warnings

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", message=".*Feedback manager.*")


def frame_to_landmark_vec(
    frame_bgr: np.ndarray,
    holistic,
    max_side: int = 480,
) -> np.ndarray:
    """One BGR frame -> normalized landmark vector (FEAT_DIM,)."""
    frame = frame_bgr
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    pose = _lm_to_array(res.pose_landmarks, N_POSE)
    lh = _lm_to_array(res.left_hand_landmarks, N_HAND)
    rh = _lm_to_array(res.right_hand_landmarks, N_HAND)
    face = _lm_to_array(res.face_landmarks, N_FACE)
    raw = np.concatenate([pose, lh, rh, face], axis=0)
    return normalize_landmarks(raw).reshape(-1)


def landmarks_from_frames(
    frames_bgr: list[np.ndarray],
    num_frames: int = 30,
    max_side: int = 480,
    holistic=None,
) -> np.ndarray:
    """Build (T, FEAT_DIM) from a list of BGR frames (live camera buffer)."""
    _silence_mediapipe_logs()

    own = holistic is None
    if own:
        # Prefer app HolisticSession (solutions or Tasks); fallback to classic import
        try:
            import sys
            from pathlib import Path

            root = Path(__file__).resolve().parents[2]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from app.holistic import HolisticSession

            holistic = HolisticSession(model_complexity=0)
        except Exception:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    "MediaPipe Holistic unavailable. Install mediapipe<0.10.15 on Python 3.10–3.12, "
                    "or use app.holistic.HolisticSession (Tasks API)."
                )
            holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
    try:
        n = len(frames_bgr)
        if n == 0:
            return np.zeros((num_frames, FEAT_DIM), dtype=np.float32)
        wanted = sample_frame_indices(n, num_frames)
        seq = np.zeros((num_frames, FEAT_DIM), dtype=np.float32)
        last = np.zeros(FEAT_DIM, dtype=np.float32)
        cache: dict[int, np.ndarray] = {}
        for t, fi in enumerate(wanted.tolist()):
            fi = int(fi)
            if fi not in cache:
                cache[fi] = frame_to_landmark_vec(frames_bgr[fi], holistic, max_side=max_side)
            last = cache[fi]
            seq[t] = last
        return seq
    finally:
        if own and hasattr(holistic, "close"):
            holistic.close()


def extract_video_landmarks(
    video_path: str | Path,
    num_frames: int = 30,
    max_side: int = 480,
) -> np.ndarray:
    """Return (T, FEAT_DIM) normalized landmark sequence."""
    _silence_mediapipe_logs()

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return landmarks_from_frames(frames, num_frames=num_frames, max_side=max_side)


def _portable_path(video_path: str | Path) -> str:
    """Path from the dataset folder down (`ISL_DATASET*/word/clip.mp4`), so the key is the
    same on every machine. Outside such a folder: `word/clip.mp4`."""
    parts = Path(video_path).resolve().parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].upper().startswith("ISL_DATASET"):
            return "/".join(parts[i:])
    return "/".join(parts[-2:])


def legacy_cache_key(video_path: str | Path, num_frames: int) -> str:
    """v1 key hashed the absolute path, so caches broke when the repo moved."""
    p = Path(video_path).resolve().as_posix()
    h = hashlib.sha1(f"{p}|{num_frames}|v1".encode()).hexdigest()[:16]
    return f"{Path(video_path).stem}__T{num_frames}__{h}.npy"


def cache_key(video_path: str | Path, num_frames: int) -> str:
    h = hashlib.sha1(f"{_portable_path(video_path)}|{num_frames}|v2".encode()).hexdigest()[:16]
    return f"{Path(video_path).stem}__T{num_frames}__{h}.npy"


def load_or_extract(
    video_path: str | Path,
    cache_dir: Path,
    num_frames: int = 30,
    require_cache: bool = False,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / cache_key(video_path, num_frames)
    if out.exists():
        return np.load(out)
    legacy = cache_dir / legacy_cache_key(video_path, num_frames)
    if legacy.exists():  # v1 cache from this machine: keep it under the portable name
        arr = np.load(legacy)
        np.save(out, arr)
        return arr
    if require_cache:
        raise FileNotFoundError(
            f"Landmark cache missing for {video_path}. "
            f"Run first:\n"
            f"  python -m islr.fewshot.extract_landmarks --num-frames {num_frames}"
        )
    arr = extract_video_landmarks(video_path, num_frames=num_frames)
    np.save(out, arr)
    return arr

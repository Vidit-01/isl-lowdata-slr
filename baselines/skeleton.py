"""27-joint ISL skeleton (HWGAT layout) and spatial adjacency partitions.

Layout (MediaPipe Holistic indices into pose | left-hand | right-hand):
  0     nose
  1–2   L/R shoulder
  3–4   L/R elbow
  5–6   L/R wrist
  7–16  left hand  (10 keypoints)
  17–26 right hand (10 keypoints)
"""
from __future__ import annotations

import numpy as np

from common.landmarks import N_HAND, N_POSE

# MediaPipe pose indices kept for the upper body
POSE_KEEP = (0, 11, 12, 13, 14, 15, 16)
# MediaPipe hand: wrist, tips + MCPs (HWGAT-style 10-point hand)
HAND_KEEP = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)

N_UPPER = len(POSE_KEEP)
N_HAND_KEEP = len(HAND_KEEP)
N_JOINTS = N_UPPER + 2 * N_HAND_KEEP  # 27
IN_CHANNELS = 3  # xyz

PARTS = {
    "upper": tuple(range(0, N_UPPER)),
    "left_hand": tuple(range(N_UPPER, N_UPPER + N_HAND_KEEP)),
    "right_hand": tuple(range(N_UPPER + N_HAND_KEEP, N_JOINTS)),
}


def _hand_edges(offset: int) -> list[tuple[int, int]]:
    """Local 10-point hand graph, shifted into global joint indices."""
    w, th, im, it, mm, mt, rm, rt, pm, pt = range(offset, offset + 10)
    return [
        (w, th),
        (w, im),
        (w, mm),
        (w, rm),
        (w, pm),
        (im, it),
        (mm, mt),
        (rm, rt),
        (pm, pt),
        (im, mm),
        (mm, rm),
        (rm, pm),
    ]


# Undirected anatomical edges (no self-loops)
EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 2),  # nose / shoulders
    (1, 3),
    (3, 5),  # left arm
    (2, 4),
    (4, 6),  # right arm
    (5, PARTS["left_hand"][0]),
    (6, PARTS["right_hand"][0]),
    *_hand_edges(PARTS["left_hand"][0]),
    *_hand_edges(PARTS["right_hand"][0]),
)

ROOT_JOINT = 0  # nose, used for centripetal / centrifugal partition


def landmarks_to_joints(seq: np.ndarray) -> np.ndarray:
    """(T, FEAT_DIM) flat Holistic xyz -> (T, V, C) 27-joint tensor."""
    t = seq.shape[0]
    n_lm = seq.size // (t * 3)
    xyz = seq.reshape(t, n_lm, 3)
    pose = xyz[:, :N_POSE][:, list(POSE_KEEP)]
    lh = xyz[:, N_POSE : N_POSE + N_HAND][:, list(HAND_KEEP)]
    rh = xyz[:, N_POSE + N_HAND : N_POSE + 2 * N_HAND][:, list(HAND_KEEP)]
    return np.concatenate([pose, lh, rh], axis=1).astype(np.float32)


def pose_hands_vec(seq: np.ndarray) -> np.ndarray:
    """Drop face mesh: (T, FEAT_DIM) -> (T, (33+21+21)*3)."""
    t = seq.shape[0]
    n_lm = seq.size // (t * 3)
    xyz = seq.reshape(t, n_lm, 3)
    keep = xyz[:, : N_POSE + 2 * N_HAND]
    return keep.reshape(t, -1).astype(np.float32)


def adjacency_binary(n: int = N_JOINTS, edges: tuple[tuple[int, int], ...] = EDGES) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    np.fill_diagonal(a, 1.0)
    return a


def hop_distances(n: int = N_JOINTS, root: int = ROOT_JOINT) -> np.ndarray:
    """Unweighted hop distance from `root` on the anatomical graph."""
    a = adjacency_binary(n)
    dist = np.full(n, -1, dtype=np.int32)
    dist[root] = 0
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.where(a[u] > 0)[0]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(int(v))
        frontier = nxt
    dist[dist < 0] = dist.max() + 1
    return dist


def normalize_undirected(a: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} with a small floor on the degree."""
    deg = a.sum(axis=1)
    deg = np.maximum(deg, 1e-6)
    inv = np.diag(1.0 / np.sqrt(deg))
    return (inv @ a @ inv).astype(np.float32)


def spatial_partition_adjacency(n: int = N_JOINTS) -> np.ndarray:
    """Yan et al. spatial-configuration partitions: self / centripetal / centrifugal.

    Returns A with shape (3, V, V), each already degree-normalized.
    """
    hops = hop_distances(n)
    binary = adjacency_binary(n)
    self_a = np.eye(n, dtype=np.float32)
    centripetal = np.zeros((n, n), dtype=np.float32)
    centrifugal = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j or binary[i, j] == 0:
                continue
            if hops[j] < hops[i]:
                centripetal[i, j] = 1.0
            else:
                centrifugal[i, j] = 1.0
    stacked = np.stack(
        [
            normalize_undirected(self_a),
            normalize_undirected(centripetal),
            normalize_undirected(centrifugal),
        ],
        axis=0,
    )
    # empty partition (e.g. root has no centripetal neighbor as source) — keep zeros
    return stacked.astype(np.float32)


def bone_pairs(n: int = N_JOINTS) -> tuple[tuple[int, int], ...]:
    """Parent -> child along hop distance (for the bone stream)."""
    hops = hop_distances(n)
    pairs = []
    for i, j in EDGES:
        if hops[i] <= hops[j]:
            pairs.append((i, j))
        else:
            pairs.append((j, i))
    return tuple(pairs)


def joints_to_bones(x: np.ndarray, pairs: tuple[tuple[int, int], ...] | None = None) -> np.ndarray:
    """x: (T, V, C) or (C, T, V) -> bone displacements on the child joint."""
    if pairs is None:
        pairs = bone_pairs()
    if x.ndim != 3:
        raise ValueError(f"expected 3D skeleton, got {x.shape}")
    # Detect layout: (T, V, C) vs (C, T, V)
    if x.shape[-1] <= 3:
        t, v, c = x.shape
        bone = np.zeros_like(x)
        for parent, child in pairs:
            bone[:, child] = x[:, child] - x[:, parent]
        return bone
    c, t, v = x.shape
    bone = np.zeros_like(x)
    for parent, child in pairs:
        bone[:, :, child] = x[:, :, child] - x[:, :, parent]
    return bone

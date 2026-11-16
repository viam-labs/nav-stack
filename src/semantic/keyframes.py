"""Camera keyframes for lost-pose recovery.

A keyframe records where a camera frame was taken, what the whole picture
looked like (one pooled descriptor), and every patch of it: its descriptor
and its 3D point. Finding a place is two steps:

1. shortlist keyframes whose pooled descriptor is closest to the live frame;
2. reorder the shortlist by how many patches match one-to-one.

Measured on LILocBench (``tools/eval_vision.py``): the pooled descriptor alone
puts the true place first for 31-57% of frames; the patch reorder lifts that
to 66-71%. Patch descriptors are stored at 128 dimensions, which matched the
full 384 within a point or two of recall at a third of the size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..geom import conversions as conv
from .patches import backproject_patches
from .types import PatchToken, VisionFrame

KEYFRAMES_FILENAME = "vision_keyframes.npz"
_FORMAT_VERSION = 1
# Patch descriptors are reduced to this many dimensions once the map has
# enough keyframes to fit the reduction from.
TOKEN_DIM = 128
FIT_AFTER_KEYFRAMES = 64
_PCA_SAMPLE = 40_000
# Keyframes kept by the pooled descriptor before patches are compared.
SHORTLIST = 30
# Patches farther than this have depth too coarse to place.
MAX_POINT_RANGE_M = 6.0


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def mutual_matches(query_tokens: np.ndarray, keyframe_tokens: np.ndarray) -> np.ndarray:
    """``(M, 2)`` index pairs (query, keyframe) of patches that pick each other.

    A pair is kept when each patch is the other's most similar. Bland wall
    patches rarely agree both ways, so the count says how much of one picture
    is in the other.
    """
    sims = np.asarray(query_tokens, dtype=np.float32) @ np.asarray(
        keyframe_tokens, dtype=np.float32
    ).T
    if sims.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    best_kf = sims.argmax(axis=1)
    best_q = sims.argmax(axis=0)
    q_idx = np.nonzero(best_q[best_kf] == np.arange(len(best_kf)))[0]
    return np.column_stack([q_idx, best_kf[q_idx]]).astype(np.int64)


def token_points(frame: VisionFrame, uv: np.ndarray) -> np.ndarray:
    """``(T, 3)`` base_link point for each patch centre; NaN without usable depth."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    tokens = [
        PatchToken(row=0, col=0, u=float(u), v=float(v), descriptor=np.zeros(1, np.float32))
        for u, v in uv
    ]
    backproject_patches(frame, tokens, max_range_m=MAX_POINT_RANGE_M)
    out = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    for i, tok in enumerate(tokens):
        if tok.point_base is not None:
            out[i] = tok.point_base
    return out


@dataclass
class KeyframeHit:
    """One place guess for a live frame."""

    keyframe_id: int
    pose: conv.Pose2D
    # (M, 2) matched patches: live token index, keyframe token index.
    pairs: np.ndarray
    # (T, 3) keyframe patch points in that keyframe's base_link.
    points: np.ndarray


class KeyframeIndex:
    """Keyframes of one map."""

    def __init__(self, token_dim: int = TOKEN_DIM) -> None:
        self.token_dim = int(token_dim)
        self.keyframe_ids: List[int] = []
        self.keyframe_poses: List[conv.Pose2D] = []
        self._pooled: List[np.ndarray] = []
        self._tokens: List[np.ndarray] = []  # float16; raw until ``_basis`` exists
        self._points: List[np.ndarray] = []
        self._basis: Optional[tuple] = None  # (components (d, D), mean (D,))

    def __len__(self) -> int:
        return len(self.keyframe_ids)

    @property
    def compressed(self) -> bool:
        return self._basis is not None

    def _project(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.float32)
        if self._basis is None:
            return l2_normalize(tokens)
        components, mean = self._basis
        return l2_normalize((tokens - mean) @ components.T)

    def add(
        self,
        keyframe_id: int,
        pose: conv.Pose2D,
        tokens: np.ndarray,
        points: np.ndarray,
    ) -> None:
        """Store a frame: ``tokens`` ``(T, D)`` and their ``points`` ``(T, 3)``."""
        tokens = np.asarray(tokens, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[0] < 1:
            raise ValueError("tokens must be (T, D) with T >= 1")
        if points.shape != (tokens.shape[0], 3):
            raise ValueError("points must be (T, 3), one per token")
        self.keyframe_ids.append(int(keyframe_id))
        self.keyframe_poses.append(conv.Pose2D(float(pose.x), float(pose.y), float(pose.theta)))
        self._pooled.append(l2_normalize(tokens.mean(axis=0)))
        self._tokens.append(self._project(tokens).astype(np.float16))
        self._points.append(points)
        if self._basis is None and len(self) >= FIT_AFTER_KEYFRAMES:
            self.compress()

    def move(self, keyframe_id: int, pose: conv.Pose2D) -> bool:
        """Set a keyframe's pose (the pose graph moved the node it hangs from)."""
        try:
            i = self.keyframe_ids.index(int(keyframe_id))
        except ValueError:
            return False
        self.keyframe_poses[i] = conv.Pose2D(float(pose.x), float(pose.y), float(pose.theta))
        return True

    def compress(self, *, seed: int = 0) -> bool:
        """Reduce stored patch descriptors to ``token_dim``. Idempotent."""
        if self._basis is not None or not self._tokens:
            return False
        bank = np.vstack([t.astype(np.float32) for t in self._tokens])
        if bank.shape[1] <= self.token_dim or bank.shape[0] <= self.token_dim:
            return False
        rng = np.random.default_rng(seed)
        if bank.shape[0] > _PCA_SAMPLE:
            bank = bank[rng.choice(bank.shape[0], _PCA_SAMPLE, replace=False)]
        mean = bank.mean(axis=0)
        _u, _s, vt = np.linalg.svd(bank - mean, full_matrices=False)
        raw = self._tokens
        self._basis = (vt[: self.token_dim].astype(np.float32), mean.astype(np.float32))
        self._tokens = [self._project(t.astype(np.float32)).astype(np.float16) for t in raw]
        return True

    def rank(
        self,
        tokens: np.ndarray,
        *,
        top_k: int = 5,
        shortlist: int = SHORTLIST,
    ) -> List[KeyframeHit]:
        """Best place guesses for a live frame's ``(T, D)`` tokens, best first."""
        if not self.keyframe_ids:
            return []
        tokens = np.asarray(tokens, dtype=np.float32)
        pooled = l2_normalize(tokens.mean(axis=0))
        sims = np.stack(self._pooled) @ pooled
        short = np.argsort(-sims)[: max(1, int(shortlist))]
        live = self._project(tokens)
        scored = []
        for k in short:
            k = int(k)
            pairs = mutual_matches(live, self._tokens[k])
            scored.append((len(pairs), k, pairs))
        # Stable: ties keep the pooled-descriptor order.
        scored.sort(key=lambda row: -row[0])
        return [
            KeyframeHit(
                keyframe_id=self.keyframe_ids[k],
                pose=self.keyframe_poses[k],
                pairs=pairs,
                points=self._points[k],
            )
            for _count, k, pairs in scored[: max(1, int(top_k))]
        ]

    def status_dict(self) -> dict:
        return {
            "keyframes": len(self),
            "compressed": self.compressed,
            "token_dim": int(self._tokens[0].shape[1]) if self._tokens else 0,
            "size_mb": round(
                sum(t.nbytes + p.nbytes for t, p in zip(self._tokens, self._points)) / 1e6, 1
            ),
        }

    def save(self, map_dir: Path) -> Optional[Path]:
        """Write next to the map; removes the file when there is nothing to keep."""
        path = Path(map_dir) / KEYFRAMES_FILENAME
        if not self.keyframe_ids:
            if path.is_file():
                path.unlink()
            return None
        self.compress()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": np.int32(_FORMAT_VERSION),
            "ids": np.asarray(self.keyframe_ids, dtype=np.int32),
            "poses": np.array(
                [[p.x, p.y, p.theta] for p in self.keyframe_poses], dtype=np.float64
            ),
            "pooled": np.stack(self._pooled).astype(np.float32),
            "counts": np.asarray([len(t) for t in self._tokens], dtype=np.int32),
            "tokens": np.vstack(self._tokens).astype(np.float16),
            "points": np.vstack(self._points).astype(np.float32),
        }
        if self._basis is not None:
            payload["basis_components"], payload["basis_mean"] = self._basis
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(str(tmp), **payload)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, map_dir: Path) -> Optional["KeyframeIndex"]:
        path = Path(map_dir) / KEYFRAMES_FILENAME
        if not path.is_file():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            if int(data["version"]) != _FORMAT_VERSION:
                return None
            index = cls()
            index.keyframe_ids = [int(x) for x in data["ids"]]
            index.keyframe_poses = [
                conv.Pose2D(float(x), float(y), float(t)) for x, y, t in data["poses"]
            ]
            index._pooled = [row for row in np.asarray(data["pooled"], dtype=np.float32)]
            ends = np.cumsum(np.asarray(data["counts"], dtype=np.int64))
            tokens = np.asarray(data["tokens"], dtype=np.float16)
            points = np.asarray(data["points"], dtype=np.float32)
            index._tokens = np.split(tokens, ends[:-1])
            index._points = np.split(points, ends[:-1])
            if "basis_components" in data:
                index._basis = (
                    np.asarray(data["basis_components"], dtype=np.float32),
                    np.asarray(data["basis_mean"], dtype=np.float32),
                )
                index.token_dim = int(index._basis[0].shape[0])
        except Exception:  # noqa: BLE001
            return None
        if len(index._tokens) != len(index.keyframe_ids):
            return None
        return index


def keyframe_gap_reached(
    pose: conv.Pose2D,
    last: Optional[conv.Pose2D],
    *,
    gap_m: float,
    gap_deg: float,
) -> bool:
    """True when the robot has moved or turned enough for another keyframe."""
    if last is None:
        return True
    moved = float(np.hypot(pose.x - last.x, pose.y - last.y))
    turned = abs(float(np.degrees(conv.normalize_angle(pose.theta - last.theta))))
    return moved >= float(gap_m) or turned >= float(gap_deg)


def stack_tokens(patches: Sequence[PatchToken]) -> tuple:
    """``(tokens (T, D), uv (T, 2))`` from a frame's patch grid."""
    tokens = np.stack([np.asarray(p.descriptor, dtype=np.float32).reshape(-1) for p in patches])
    uv = np.array([[float(p.u), float(p.v)] for p in patches], dtype=np.float32)
    return tokens, uv

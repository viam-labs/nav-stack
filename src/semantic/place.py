"""Whole-image place descriptors (VLAD over patch tokens) and top-K retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass
class PlaceIndex:
    """Per-map PCA + k-means vocabulary + keyframe VLAD descriptors."""

    pca_components: Optional[np.ndarray] = None  # (D_out, D_in)
    pca_mean: Optional[np.ndarray] = None  # (D_in,)
    vocabulary: Optional[np.ndarray] = None  # (K, D_out)
    keyframe_ids: List[int] = None  # type: ignore[assignment]
    keyframe_poses: List[conv.Pose2D] = None  # type: ignore[assignment]
    keyframe_vlad: Optional[np.ndarray] = None  # (N, K*D_out)
    cosine_threshold: float = 0.4

    def __post_init__(self) -> None:
        if self.keyframe_ids is None:
            self.keyframe_ids = []
        if self.keyframe_poses is None:
            self.keyframe_poses = []

    def project(self, descriptors: np.ndarray) -> np.ndarray:
        d = np.asarray(descriptors, dtype=np.float32)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if self.pca_mean is not None and self.pca_components is not None:
            d = (d - self.pca_mean) @ self.pca_components.T
        return l2_normalize(d)

    def vlad(self, descriptors: np.ndarray) -> np.ndarray:
        """Compute a single VLAD vector from (N, D) descriptors."""
        d = self.project(descriptors)
        if self.vocabulary is None or self.vocabulary.size == 0:
            # Fallback: mean pool.
            return l2_normalize(d.mean(axis=0))
        vocab = self.vocabulary
        # Assign to nearest cluster.
        # (N, K) squared distances via ||a||^2 + ||b||^2 - 2 a·b; vocab is L2-normed.
        sim = d @ vocab.T
        assign = np.argmax(sim, axis=1)
        k, dim = vocab.shape
        out = np.zeros((k, dim), dtype=np.float32)
        for i in range(len(d)):
            c = int(assign[i])
            out[c] += d[i] - vocab[c]
        flat = out.reshape(-1)
        return l2_normalize(flat)

    def add_keyframe(
        self, keyframe_id: int, pose: conv.Pose2D, descriptors: np.ndarray
    ) -> np.ndarray:
        vec = self.vlad(descriptors)
        self.keyframe_ids.append(int(keyframe_id))
        self.keyframe_poses.append(pose)
        if self.keyframe_vlad is None:
            self.keyframe_vlad = vec.reshape(1, -1)
        else:
            # Allow dim growth only if empty previously with mean-pool dim.
            if self.keyframe_vlad.shape[1] != vec.shape[0]:
                # Reset store if vocabulary was fit later — caller should rebuild.
                self.keyframe_vlad = vec.reshape(1, -1)
                self.keyframe_ids = [int(keyframe_id)]
                self.keyframe_poses = [pose]
            else:
                self.keyframe_vlad = np.vstack([self.keyframe_vlad, vec.reshape(1, -1)])
        return vec

    def query(
        self, descriptors: np.ndarray, *, top_k: int = 5
    ) -> List[Tuple[int, conv.Pose2D, float]]:
        if self.keyframe_vlad is None or len(self.keyframe_ids) == 0:
            return []
        q = self.vlad(descriptors)
        sims = self.keyframe_vlad @ q
        k = min(int(top_k), len(sims))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out: List[Tuple[int, conv.Pose2D, float]] = []
        for i in idx:
            score = float(sims[i])
            if score < float(self.cosine_threshold):
                continue
            out.append((self.keyframe_ids[int(i)], self.keyframe_poses[int(i)], score))
        return out


def fit_pca(descriptors: np.ndarray, *, out_dim: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Return (components (out_dim, D), mean (D,))."""
    x = np.asarray(descriptors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("need at least 2 descriptors for PCA")
    mean = x.mean(axis=0)
    xc = x - mean
    # Economy SVD on covariance via X.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    dim = min(int(out_dim), vt.shape[0], x.shape[1])
    components = vt[:dim].astype(np.float32)
    return components, mean.astype(np.float32)


def fit_vocabulary(
    descriptors: np.ndarray,
    *,
    n_clusters: int = 32,
    seed: int = 0,
    iters: int = 20,
) -> np.ndarray:
    """Simple k-means; returns (K, D) L2-normalized centers."""
    x = l2_normalize(np.asarray(descriptors, dtype=np.float32))
    n = x.shape[0]
    k = min(int(n_clusters), n)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(max(1, iters)):
        sim = x @ centers.T
        assign = np.argmax(sim, axis=1)
        for c in range(k):
            members = x[assign == c]
            if len(members) == 0:
                centers[c] = x[int(rng.integers(0, n))]
            else:
                centers[c] = members.mean(axis=0)
        centers = l2_normalize(centers)
    return centers

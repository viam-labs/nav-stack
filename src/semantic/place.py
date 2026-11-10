"""Whole-image place descriptors (VLAD over patch tokens) and top-K retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    keyframe_ids: List[int] = field(default_factory=list)
    keyframe_poses: List[conv.Pose2D] = field(default_factory=list)
    keyframe_vlad: Optional[np.ndarray] = None  # (N, K*D_out)
    cosine_threshold: float = 0.4
    # Raw patch banks kept in-session so we can (re)fit PCA/vocab then rebuild.
    keyframe_descs: List[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def fitted(self) -> bool:
        return (
            self.vocabulary is not None
            and self.vocabulary.size > 0
            and self.pca_components is not None
            and self.pca_mean is not None
        )

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
            # Fallback: mean pool until vocabulary is fit.
            return l2_normalize(d.mean(axis=0))
        vocab = self.vocabulary
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
        descs = np.asarray(descriptors, dtype=np.float32)
        if descs.ndim != 2 or descs.shape[0] < 1:
            raise ValueError("descriptors must be (N, D) with N>=1")
        vec = self.vlad(descs)
        self.keyframe_ids.append(int(keyframe_id))
        self.keyframe_poses.append(pose)
        self.keyframe_descs.append(descs.copy())
        if self.keyframe_vlad is None:
            self.keyframe_vlad = vec.reshape(1, -1)
        else:
            if self.keyframe_vlad.shape[1] != vec.shape[0]:
                # Dim changed after a fit — rebuild from stored banks.
                self._rebuild_vlads()
            else:
                self.keyframe_vlad = np.vstack([self.keyframe_vlad, vec.reshape(1, -1)])
        return vec

    def _rebuild_vlads(self) -> None:
        if not self.keyframe_descs:
            self.keyframe_vlad = None
            return
        rows = [self.vlad(d).reshape(1, -1) for d in self.keyframe_descs]
        self.keyframe_vlad = np.vstack(rows)

    def fit(
        self,
        *,
        n_clusters: int = 32,
        pca_dim: int = 128,
        seed: int = 0,
        min_descriptors: int = 64,
    ) -> bool:
        """Fit PCA + k-means from stored keyframe banks; rebuild VLAD rows."""
        if len(self.keyframe_descs) < 2:
            return False
        bank = np.vstack(self.keyframe_descs)
        if bank.shape[0] < int(min_descriptors):
            return False
        comps, mean = fit_pca(bank, out_dim=int(pca_dim))
        projected = l2_normalize((bank - mean) @ comps.T)
        vocab = fit_vocabulary(
            projected, n_clusters=int(n_clusters), seed=int(seed)
        )
        self.pca_components = comps
        self.pca_mean = mean
        self.vocabulary = vocab
        self._rebuild_vlads()
        return True

    def query(
        self, descriptors: np.ndarray, *, top_k: int = 5
    ) -> List[Tuple[int, conv.Pose2D, float]]:
        if self.keyframe_vlad is None or len(self.keyframe_ids) == 0:
            return []
        descs = np.asarray(descriptors, dtype=np.float32)
        if descs.ndim != 2 or descs.shape[0] < 1:
            return []
        q = self.vlad(descs)
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

    def status_dict(self) -> dict:
        return {
            "keyframes": len(self.keyframe_ids),
            "fitted": bool(self.fitted),
            "vocab_k": (
                int(self.vocabulary.shape[0])
                if self.vocabulary is not None and self.vocabulary.size
                else 0
            ),
            "pca_dim": (
                int(self.pca_components.shape[0])
                if self.pca_components is not None
                else 0
            ),
            "vlad_dim": (
                int(self.keyframe_vlad.shape[1])
                if self.keyframe_vlad is not None
                else 0
            ),
        }


def fit_pca(descriptors: np.ndarray, *, out_dim: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Return (components (out_dim, D), mean (D,))."""
    x = np.asarray(descriptors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("need at least 2 descriptors for PCA")
    mean = x.mean(axis=0)
    xc = x - mean
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


def descriptors_from_patches(
    patches: Sequence, *, max_patches: int = 64
) -> Optional[np.ndarray]:
    """Stack patch token descriptors for VLAD; None if empty."""
    if not patches:
        return None
    rows = [np.asarray(p.descriptor, dtype=np.float32).reshape(-1) for p in patches]
    if not rows:
        return None
    stacked = np.stack(rows, axis=0)
    if stacked.shape[0] > int(max_patches):
        # Uniform subsample keeps coverage without chewing CPU.
        idx = np.linspace(0, stacked.shape[0] - 1, num=int(max_patches), dtype=int)
        stacked = stacked[idx]
    return stacked

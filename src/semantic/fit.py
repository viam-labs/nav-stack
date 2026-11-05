"""Weighted 2D rigid fit (Umeyama) with RANSAC for landmark pose fixes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv


@dataclass
class RigidFitResult:
    pose: conv.Pose2D
    inliers: np.ndarray  # bool mask over input pairs
    residual_rmse_m: float
    cov_xy_m2: float  # approximate isotropic XY variance


def _umeyama_2d(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Rigid transform mapping ``src`` → ``dst``: R, t (no scale).

    Returns ``(tx, ty, yaw)`` such that ``R @ src + t ≈ dst``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src/dst must be (N, 2) with matching shapes")
    n = src.shape[0]
    if n < 2:
        raise ValueError("need at least 2 points")
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError("weights length mismatch")
        w = np.maximum(w, 1e-12)
    w_sum = float(w.sum())
    mu_s = (w[:, None] * src).sum(axis=0) / w_sum
    mu_d = (w[:, None] * dst).sum(axis=0) / w_sum
    xs = src - mu_s
    xd = dst - mu_d
    # Covariance.
    cov = (w[:, None] * xd).T @ xs / w_sum
    u, _, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    t = mu_d - r @ mu_s
    yaw = math.atan2(r[1, 0], r[0, 0])
    return float(t[0]), float(t[1]), float(yaw)


def fit_rigid_2d(
    live_xy: np.ndarray,
    map_xy: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    min_inliers: int = 8,
    residual_m: float = 0.08,
    ransac_iters: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> Optional[RigidFitResult]:
    """Estimate robot pose that maps live base_link points onto map points.

    Convention: ``map_point ≈ R(yaw) @ live_point + t``, i.e. the robot pose
    in the map that explains the observation (live points are in base_link).
    """
    live = np.asarray(live_xy, dtype=float)
    maps = np.asarray(map_xy, dtype=float)
    if live.shape != maps.shape or live.ndim != 2 or live.shape[1] != 2:
        raise ValueError("live_xy/map_xy must be matching (N, 2)")
    n = live.shape[0]
    if n < 2:
        return None
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError("weights length mismatch")
    gen = rng if rng is not None else np.random.default_rng(0)
    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    best_pose: Optional[Tuple[float, float, float]] = None

    for _ in range(max(1, int(ransac_iters))):
        if n == 2:
            idx = np.array([0, 1], dtype=int)
        else:
            idx = gen.choice(n, size=2, replace=False)
        try:
            tx, ty, yaw = _umeyama_2d(live[idx], maps[idx], weights=w[idx])
        except Exception:  # noqa: BLE001
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        pred = np.column_stack(
            [
                c * live[:, 0] - s * live[:, 1] + tx,
                s * live[:, 0] + c * live[:, 1] + ty,
            ]
        )
        err = np.hypot(pred[:, 0] - maps[:, 0], pred[:, 1] - maps[:, 1])
        inliers = err <= float(residual_m)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_pose = (tx, ty, yaw)

    if best_inliers is None or best_count < int(min_inliers) or best_pose is None:
        return None

    # Refine on inliers.
    try:
        tx, ty, yaw = _umeyama_2d(
            live[best_inliers], maps[best_inliers], weights=w[best_inliers]
        )
    except Exception:  # noqa: BLE001
        tx, ty, yaw = best_pose
    c, s = math.cos(yaw), math.sin(yaw)
    pred = np.column_stack(
        [
            c * live[:, 0] - s * live[:, 1] + tx,
            s * live[:, 0] + c * live[:, 1] + ty,
        ]
    )
    err = np.hypot(pred[:, 0] - maps[:, 0], pred[:, 1] - maps[:, 1])
    inliers = err <= float(residual_m)
    if int(inliers.sum()) < int(min_inliers):
        return None
    rmse = float(np.sqrt(np.mean(err[inliers] ** 2))) if inliers.any() else float("inf")
    # Rough cov from residual + weight scale.
    cov = rmse * rmse if math.isfinite(rmse) else 0.05**2
    return RigidFitResult(
        pose=conv.Pose2D(tx, ty, yaw),
        inliers=inliers,
        residual_rmse_m=rmse,
        cov_xy_m2=cov,
    )

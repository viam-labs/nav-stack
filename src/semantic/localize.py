"""Match live patches to confirmed landmarks and fit a pose correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv
from .fit import RigidFitResult, fit_rigid_2d
from .landmarks import EmbeddingLandmark, EmbeddingLandmarkMap
from .patches import PatchObservation


def _best_desc_cosine(lm: EmbeddingLandmark, desc: np.ndarray) -> float:
    if not lm.descriptors:
        return -1.0
    return max(EmbeddingLandmarkMap._cosine(d, desc) for d in lm.descriptors)


@dataclass
class LandmarkFix:
    fit: RigidFitResult
    n_pairs: int
    score: float
    jump_m: float
    jump_deg: float


def match_live_to_landmarks(
    observations: Sequence[PatchObservation],
    visible: Sequence[Tuple[EmbeddingLandmark, np.ndarray]],
    *,
    robot_pose: conv.Pose2D,
    cosine_threshold: float,
    max_reproj_m: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy 1:1 match of live base_link patches to visible map landmarks.

    Returns ``(live_xy, map_xy, weights)`` with shape ``(N, 2)`` / ``(N,)``.
    Empty arrays when nothing matched.
    """
    if not observations or not visible:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, np.zeros((0,), dtype=float)

    c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
    used_lm: set = set()
    live_rows: List[np.ndarray] = []
    map_rows: List[np.ndarray] = []
    weights: List[float] = []

    # Prefer distinctive (higher best-vs-threshold margin) by sorting on
    # range uncertainty ascending — closer/cleaner patches first.
    ordered = sorted(observations, key=lambda o: float(o.sigma_m))
    for o in ordered:
        desc = o.token.descriptor
        pb = o.point_base
        pred_map = np.array(
            [
                robot_pose.x + c * float(pb[0]) - s * float(pb[1]),
                robot_pose.y + s * float(pb[0]) + c * float(pb[1]),
            ],
            dtype=float,
        )
        best_i = -1
        best_cos = float(cosine_threshold)
        best_map: Optional[np.ndarray] = None
        for i, (lm, mxyz) in enumerate(visible):
            if lm.landmark_id in used_lm:
                continue
            dist = math.hypot(float(mxyz[0]) - pred_map[0], float(mxyz[1]) - pred_map[1])
            if dist > float(max_reproj_m):
                continue
            cos = _best_desc_cosine(lm, desc)
            if cos > best_cos:
                best_cos = cos
                best_i = i
                best_map = np.asarray(mxyz[:2], dtype=float)
        if best_i < 0 or best_map is None:
            continue
        lm = visible[best_i][0]
        used_lm.add(lm.landmark_id)
        live_rows.append(np.asarray(pb[:2], dtype=float))
        map_rows.append(best_map)
        # Weight by cosine margin and inverse depth sigma.
        weights.append(max(0.05, best_cos) / max(0.01, float(o.sigma_m)))

    if not live_rows:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, np.zeros((0,), dtype=float)
    return (
        np.stack(live_rows, axis=0),
        np.stack(map_rows, axis=0),
        np.asarray(weights, dtype=float),
    )


def estimate_landmark_fix(
    observations: Sequence[PatchObservation],
    visible: Sequence[Tuple[EmbeddingLandmark, np.ndarray]],
    *,
    robot_pose: conv.Pose2D,
    cosine_threshold: float,
    min_inliers: int = 8,
    residual_m: float = 0.08,
    max_jump_m: float = 1.5,
    max_jump_deg: float = 25.0,
    max_reproj_m: float = 0.45,
    rng: Optional[np.random.Generator] = None,
) -> Optional[LandmarkFix]:
    """Pair → rigid fit → jump gate. Returns None when the fix is untrusted."""
    live_xy, map_xy, weights = match_live_to_landmarks(
        observations,
        visible,
        robot_pose=robot_pose,
        cosine_threshold=cosine_threshold,
        max_reproj_m=max_reproj_m,
    )
    if live_xy.shape[0] < int(min_inliers):
        return None
    fit = fit_rigid_2d(
        live_xy,
        map_xy,
        weights=weights,
        min_inliers=min_inliers,
        residual_m=residual_m,
        rng=rng,
    )
    if fit is None:
        return None
    jump_m = math.hypot(fit.pose.x - robot_pose.x, fit.pose.y - robot_pose.y)
    jump_deg = abs(
        math.degrees(conv.normalize_angle(fit.pose.theta - robot_pose.theta))
    )
    if jump_m > float(max_jump_m) or jump_deg > float(max_jump_deg):
        return None
    # Score: inlier fraction × residual quality (soft for apply_map_pose_correction).
    inlier_frac = float(fit.inliers.mean()) if fit.inliers.size else 0.0
    resid_q = max(0.0, 1.0 - float(fit.residual_rmse_m) / max(1e-6, float(residual_m)))
    score = max(0.05, min(0.95, 0.5 * inlier_frac + 0.5 * resid_q))
    return LandmarkFix(
        fit=fit,
        n_pairs=int(live_xy.shape[0]),
        score=score,
        jump_m=jump_m,
        jump_deg=jump_deg,
    )

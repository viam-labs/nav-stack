"""Match live patches to confirmed landmarks and fit a pose correction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


@dataclass
class LandmarkPoseAttempt:
    """Result of pair → fit → jump; ``fix`` set only when all gates pass."""

    fix: Optional[LandmarkFix] = None
    reason: Optional[str] = None  # few_pairs | rejected_residual | rejected_jump
    pairs: int = 0
    jump_m: Optional[float] = None
    jump_deg: Optional[float] = None
    paired_ids: List[int] = field(default_factory=list)


def match_live_to_landmarks(
    observations: Sequence[PatchObservation],
    visible: Sequence[Tuple[EmbeddingLandmark, np.ndarray]],
    *,
    robot_pose: conv.Pose2D,
    cosine_threshold: float,
    max_reproj_m: Optional[float] = 0.45,
    detector_cosine_threshold: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Greedy 1:1 match of live base_link patches to visible map landmarks.

    Returns ``(live_xy, map_xy, weights, paired_ids)``.
    Empty arrays / empty ids when nothing matched.

    ``max_reproj_m`` drops a landmark whose stored point is farther than that
    from the live point placed at ``robot_pose``. ``None`` skips that check
    and pairs by descriptor among ``visible``; the caller has already limited
    ``visible`` (image projection from a place guess) and the rigid fit is
    the geometry check.

    Detector-anchored landmarks use ``detector_cosine_threshold`` (aligned with
    survey association) so writeback is not harsher than re-ID.
    """
    if not observations or not visible:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, np.zeros((0,), dtype=float), []

    c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
    used_lm: set = set()
    live_rows: List[np.ndarray] = []
    map_rows: List[np.ndarray] = []
    weights: List[float] = []
    paired_ids: List[int] = []

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
        best_cos = -1.0
        best_map: Optional[np.ndarray] = None
        hard = float(cosine_threshold)
        soft = min(hard, 0.35)
        gate = None if max_reproj_m is None else float(max_reproj_m)
        soft_dist = 0.0 if gate is None else 0.7 * gate
        det_thr = float(detector_cosine_threshold)
        for i, (lm, mxyz) in enumerate(visible):
            if lm.landmark_id in used_lm:
                continue
            if gate is None:
                need = hard
            else:
                dist = math.hypot(
                    float(mxyz[0]) - pred_map[0], float(mxyz[1]) - pred_map[1]
                )
                if dist > gate:
                    continue
                need = soft if dist <= soft_dist else hard
            if bool(lm.from_detector):
                need = min(need, det_thr)
            cos = _best_desc_cosine(lm, desc)
            if cos < need:
                continue
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
        paired_ids.append(int(lm.landmark_id))
        # Weight by cosine margin and inverse depth sigma.
        weights.append(max(0.05, best_cos) / max(0.01, float(o.sigma_m)))

    if not live_rows:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, np.zeros((0,), dtype=float), []
    return (
        np.stack(live_rows, axis=0),
        np.stack(map_rows, axis=0),
        np.asarray(weights, dtype=float),
        paired_ids,
    )


def diagnose_landmark_match(
    observations: Sequence[PatchObservation],
    visible: Sequence[Tuple[EmbeddingLandmark, np.ndarray]],
    *,
    robot_pose: conv.Pose2D,
    cosine_threshold: float,
    max_reproj_m: Optional[float] = 0.45,
    detector_cosine_threshold: float = 0.30,
) -> dict:
    """Explain why writeback pairing failed (geometry vs descriptor)."""
    n_obs = len(observations)
    n_vis = len(visible)
    if n_obs == 0 or n_vis == 0:
        return {
            "n_obs": n_obs,
            "n_visible": n_vis,
            "n_geom_ok": 0,
            "n_cos_ok": 0,
            "best_cos": None,
            "best_dist_m": None,
            "cosine_thr": float(cosine_threshold),
            "detector_cosine_thr": float(detector_cosine_threshold),
            "max_reproj_m": None if max_reproj_m is None else float(max_reproj_m),
        }

    c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
    hard = float(cosine_threshold)
    soft = min(hard, 0.35)
    gate = None if max_reproj_m is None else float(max_reproj_m)
    soft_dist = 0.0 if gate is None else 0.7 * gate
    det_thr = float(detector_cosine_threshold)
    n_geom_ok = 0
    n_cos_ok = 0
    best_cos = -1.0
    best_dist = None
    for o in observations:
        pb = o.point_base
        pred_map = np.array(
            [
                robot_pose.x + c * float(pb[0]) - s * float(pb[1]),
                robot_pose.y + s * float(pb[0]) + c * float(pb[1]),
            ],
            dtype=float,
        )
        for lm, mxyz in visible:
            dist = math.hypot(float(mxyz[0]) - pred_map[0], float(mxyz[1]) - pred_map[1])
            if gate is not None and dist > gate:
                continue
            n_geom_ok += 1
            if gate is None:
                need = hard
            else:
                need = soft if dist <= soft_dist else hard
            if bool(lm.from_detector):
                need = min(need, det_thr)
            cos = _best_desc_cosine(lm, o.token.descriptor)
            if cos > best_cos:
                best_cos = cos
                best_dist = dist
            if cos >= need:
                n_cos_ok += 1
    return {
        "n_obs": n_obs,
        "n_visible": n_vis,
        "n_geom_ok": int(n_geom_ok),
        "n_cos_ok": int(n_cos_ok),
        "best_cos": None if best_cos < 0 else round(float(best_cos), 3),
        "best_dist_m": None if best_dist is None else round(float(best_dist), 3),
        "cosine_thr": round(float(cosine_threshold), 3),
        "detector_cosine_thr": round(float(det_thr), 3),
        "max_reproj_m": None if max_reproj_m is None else float(max_reproj_m),
    }


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
    max_reproj_m: Optional[float] = 0.45,
    rng: Optional[np.random.Generator] = None,
) -> LandmarkPoseAttempt:
    """Pair -> rigid fit -> jump gate. ``fix`` is set only when trusted."""
    live_xy, map_xy, weights, paired_ids = match_live_to_landmarks(
        observations,
        visible,
        robot_pose=robot_pose,
        cosine_threshold=cosine_threshold,
        max_reproj_m=max_reproj_m,
        detector_cosine_threshold=0.30,
    )
    pairs = int(live_xy.shape[0])
    if pairs < int(min_inliers):
        return LandmarkPoseAttempt(
            reason="few_pairs", pairs=pairs, paired_ids=list(paired_ids)
        )
    fit = fit_rigid_2d(
        live_xy,
        map_xy,
        weights=weights,
        min_inliers=min_inliers,
        residual_m=residual_m,
        rng=rng,
    )
    if fit is None:
        return LandmarkPoseAttempt(
            reason="rejected_residual", pairs=pairs, paired_ids=list(paired_ids)
        )
    jump_m = math.hypot(fit.pose.x - robot_pose.x, fit.pose.y - robot_pose.y)
    jump_deg = abs(
        math.degrees(conv.normalize_angle(fit.pose.theta - robot_pose.theta))
    )
    if jump_m > float(max_jump_m) or jump_deg > float(max_jump_deg):
        return LandmarkPoseAttempt(
            reason="rejected_jump",
            pairs=pairs,
            jump_m=float(jump_m),
            jump_deg=float(jump_deg),
            paired_ids=list(paired_ids),
        )
    # Score: inlier fraction x residual quality (soft for apply_map_pose_correction).
    inlier_frac = float(fit.inliers.mean()) if fit.inliers.size else 0.0
    resid_q = max(0.0, 1.0 - float(fit.residual_rmse_m) / max(1e-6, float(residual_m)))
    score = max(0.05, min(0.95, 0.5 * inlier_frac + 0.5 * resid_q))
    fix = LandmarkFix(
        fit=fit,
        n_pairs=pairs,
        score=score,
        jump_m=jump_m,
        jump_deg=jump_deg,
    )
    return LandmarkPoseAttempt(
        fix=fix,
        pairs=pairs,
        jump_m=float(jump_m),
        jump_deg=float(jump_deg),
        paired_ids=list(paired_ids),
    )

"""Recover a lost pose: the camera proposes, the lidar decides.

The camera ranks stored keyframes for the live frame and, where enough
matched patches have depth, fits its own pose against each. Every guess is
then handed to the lidar scan search. A pose is returned only when

- the lidar match around the camera's pose passes the usual score floors,
- the lidar and the camera land on the same pose (two sensors, one answer),
- and no other guess somewhere else matches the scan nearly as well.

Measured on LILocBench with each frame treated as a kidnapped robot
(``tools/eval_vision.py reloc``). Taking the best lidar match among camera
guesses placed 55-64% of frames but 12-20% wrongly, nearly all in the hallway
where look-alike doors and two parallel walls fool both sensors. The two
extra conditions bring wrong placements to 0-2% and keep 26-38% placed per
attempt where people or moved furniture have changed the scene. There the
whole-map lidar search alone places 14-27%. In a scene that still matches the
map the lidar alone places 99% and this path 74%, so the camera earns its
place only when the scene has changed. A lost robot retries on each new
frame, so the per-attempt rate is not the time to recover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..geom import conversions as conv
from .fit import fit_rigid_2d
from .keyframes import KeyframeIndex

# Camera pose fit: matched patches must sit at the same height, and at least
# this many must land within the residual of one rigid pose.
MAX_HEIGHT_GAP_M = 0.25
FIT_RESIDUAL_M = 0.15
MIN_INLIERS = 8
# Lidar search around a camera pose (tight) or a bare keyframe pose (wide).
TIGHT_RADIUS_M = 0.6
TIGHT_YAW_WINDOW_DEG = 40.0
WIDE_RADIUS_M = 1.5
WIDE_YAW_WINDOW_DEG = 120.0
# Same floors the SLAM service uses before applying a seeded scan match.
MIN_LIDAR_SCORE = 0.35
MAX_RAY_MAE_M = 1.0
# Camera and lidar must agree this closely. 0.4 m / 10° let through about 8%
# wrong poses on the rearranged-furniture run.
AGREE_M = 0.25
AGREE_DEG = 6.0
# Another guess this far away that scores within the margin is a rival
# place, and the recovery is refused.
RIVAL_DISTANCE_M = 1.0
RIVAL_SCORE_MARGIN = 0.10


@dataclass
class CameraFit:
    pose: conv.Pose2D
    inliers: int
    pairs: int
    rmse_m: float


@dataclass
class LidarMatch:
    pose: conv.Pose2D
    score: float
    ray_mae_m: float


# ``(hint, search_radius_m, yaw_window_deg)`` → best scan match, or None.
LidarSearch = Callable[[conv.Pose2D, float, float], Optional[LidarMatch]]


def fit_to_keyframe(
    live_points: np.ndarray,
    keyframe_points: np.ndarray,
    pairs: np.ndarray,
    keyframe_pose: conv.Pose2D,
    *,
    min_inliers: int = MIN_INLIERS,
    residual_m: float = FIT_RESIDUAL_M,
) -> Optional[CameraFit]:
    """Robot pose that lays the live frame's matched patch points on the keyframe's.

    ``live_points`` / ``keyframe_points`` are ``(T, 3)`` in each frame's own
    base_link (NaN where depth was unusable); ``pairs`` indexes into them.
    """
    if len(pairs) < int(min_inliers):
        return None
    live = np.asarray(live_points, dtype=float)[pairs[:, 0]]
    stored = np.asarray(keyframe_points, dtype=float)[pairs[:, 1]]
    ok = (
        np.isfinite(live[:, 0])
        & np.isfinite(stored[:, 0])
        & (np.abs(live[:, 2] - stored[:, 2]) <= MAX_HEIGHT_GAP_M)
    )
    if int(ok.sum()) < int(min_inliers):
        return None
    live, stored = live[ok], stored[ok]
    c, s = math.cos(keyframe_pose.theta), math.sin(keyframe_pose.theta)
    stored_map = np.column_stack(
        [
            keyframe_pose.x + c * stored[:, 0] - s * stored[:, 1],
            keyframe_pose.y + s * stored[:, 0] + c * stored[:, 1],
        ]
    )
    fit = fit_rigid_2d(
        live[:, :2],
        stored_map,
        min_inliers=int(min_inliers),
        residual_m=float(residual_m),
        ransac_iters=128,
    )
    if fit is None:
        return None
    return CameraFit(
        pose=fit.pose,
        inliers=int(fit.inliers.sum()),
        pairs=int(ok.sum()),
        rmse_m=float(fit.residual_rmse_m),
    )


def _gap(a: conv.Pose2D, b: conv.Pose2D) -> tuple:
    return (
        math.hypot(a.x - b.x, a.y - b.y),
        abs(math.degrees(conv.normalize_angle(a.theta - b.theta))),
    )


@dataclass
class Hypothesis:
    """One place guess after both sensors have looked at it."""

    keyframe_id: int
    camera: Optional[CameraFit]
    lidar: Optional[LidarMatch]

    @property
    def lidar_passes(self) -> bool:
        return (
            self.lidar is not None
            and self.lidar.score >= MIN_LIDAR_SCORE
            and self.lidar.ray_mae_m <= MAX_RAY_MAE_M
        )

    @property
    def sensors_agree(self) -> bool:
        if self.camera is None or self.lidar is None:
            return False
        gap_m, gap_deg = _gap(self.camera.pose, self.lidar.pose)
        return gap_m <= AGREE_M and gap_deg <= AGREE_DEG

    def as_dict(self) -> dict:
        out: dict = {"keyframe_id": int(self.keyframe_id)}
        if self.camera is not None:
            out["camera_inliers"] = int(self.camera.inliers)
        if self.lidar is not None:
            out["lidar_score"] = round(float(self.lidar.score), 3)
            out["lidar_ray_mae_m"] = round(float(self.lidar.ray_mae_m), 3)
        if self.camera is not None and self.lidar is not None:
            gap_m, gap_deg = _gap(self.camera.pose, self.lidar.pose)
            out["agree_m"] = round(gap_m, 3)
            out["agree_deg"] = round(gap_deg, 1)
        return out


@dataclass
class Recovery:
    """Outcome of one recovery attempt.

    ``status`` is ``accepted``, or why not: ``no_keyframes`` (empty index),
    ``no_agreement`` (no guess where the lidar passed and matched the camera),
    ``rival_place`` (a second place explains the scan nearly as well).
    """

    status: str
    pose: Optional[conv.Pose2D] = None
    winner: Optional[Hypothesis] = None
    hypotheses: List[Hypothesis] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.pose is not None

    def as_dict(self) -> dict:
        out: dict = {
            "status": self.status,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
        }
        if self.winner is not None:
            out["keyframe_id"] = int(self.winner.keyframe_id)
        if self.pose is not None:
            out["pose"] = {
                "x": round(float(self.pose.x), 3),
                "y": round(float(self.pose.y), 3),
                "theta_deg": round(math.degrees(float(self.pose.theta)), 1),
            }
        return out


def recover_pose(
    index: KeyframeIndex,
    tokens: np.ndarray,
    points: np.ndarray,
    lidar_search: LidarSearch,
    *,
    top_k: int = 5,
) -> Recovery:
    """Pose for a live frame with no prior, or the reason there is none.

    ``tokens`` ``(T, D)`` and ``points`` ``(T, 3)`` describe the live frame's
    patches. ``lidar_search`` must match the scan taken with that frame.
    """
    hits = index.rank(tokens, top_k=top_k)
    if not hits:
        return Recovery("no_keyframes")
    hypotheses: List[Hypothesis] = []
    for hit in hits:
        camera = fit_to_keyframe(points, hit.points, hit.pairs, hit.pose)
        if camera is not None:
            lidar = lidar_search(camera.pose, TIGHT_RADIUS_M, TIGHT_YAW_WINDOW_DEG)
        else:
            # No camera pose to agree with, so this guess cannot win. Its
            # scan match still counts as a rival for the ones that can.
            lidar = lidar_search(hit.pose, WIDE_RADIUS_M, WIDE_YAW_WINDOW_DEG)
        hypotheses.append(Hypothesis(hit.keyframe_id, camera, lidar))

    agreeing = [h for h in hypotheses if h.lidar_passes and h.sensors_agree]
    if not agreeing:
        return Recovery("no_agreement", hypotheses=hypotheses)
    winner = max(agreeing, key=lambda h: h.lidar.score)
    for other in hypotheses:
        if other is winner or other.lidar is None:
            continue
        apart = math.hypot(
            other.lidar.pose.x - winner.lidar.pose.x,
            other.lidar.pose.y - winner.lidar.pose.y,
        )
        if (
            apart > RIVAL_DISTANCE_M
            and other.lidar.score >= winner.lidar.score - RIVAL_SCORE_MARGIN
        ):
            return Recovery("rival_place", winner=winner, hypotheses=hypotheses)
    return Recovery(
        "accepted", pose=winner.lidar.pose, winner=winner, hypotheses=hypotheses
    )

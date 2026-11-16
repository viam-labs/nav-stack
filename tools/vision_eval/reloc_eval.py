"""Kidnapped-robot test: recover the pose of a frame with no prior.

Three ways, all judged against ground truth:

- ``lidar``: the stack's whole-map scan search, as it runs today.
- ``place+lidar``: the top place guesses from the camera, each handed to the
  scan search as a hint, best scan match wins.
- ``oracle+lidar``: a hint 1 m and 30° off the truth. The ceiling for any
  hint-based method: what the scan search can confirm when the guess is right.

"Accepted" uses the same score and ray-error floors the SLAM service applies
before it moves the pose, so an accepted wrong answer here is a wrong pose on
the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.geom import conversions as conv
from src.nav.localize.global_localize import (
    GlobalLocalizeResult,
    OccupancyMap,
    global_localize_scan,
)
from src.semantic.keyframes import KeyframeIndex
from src.semantic.relocalize import LidarMatch, recover_pose

from .features import FrameFeatures
from .lilocbench import LilocSequence
from .metric_eval import PointCache, fit_to_keyframe
from .place_eval import rank_named, select_keyframes

FIT_SUFFIX = "+fit"
# The runtime recovery path end to end (``src/semantic/relocalize.py``).
RUNTIME = "runtime"

# ``SlamService._global_localize`` defaults.
FULL_MIN_SCORE = 0.55
FULL_MAX_RAY_MAE_M = 0.55
LOCAL_MIN_SCORE = 0.35
LOCAL_MAX_RAY_MAE_M = 1.0
LOCAL_RAY_WEIGHT = 0.35

# A recovered pose this close is one the continuous matcher holds.
SUCCESS_M = 0.5
SUCCESS_DEG = 15.0


@dataclass
class Attempt:
    """One recovery attempt for one query frame."""

    err_m: float
    err_deg: float
    accepted: bool
    score: float

    @property
    def right(self) -> bool:
        return self.err_m <= SUCCESS_M and self.err_deg <= SUCCESS_DEG


def _errors(pose: conv.Pose2D, truth: conv.Pose2D) -> tuple:
    return (
        math.hypot(pose.x - truth.x, pose.y - truth.y),
        abs(math.degrees(conv.normalize_angle(pose.theta - truth.theta))),
    )


def _combined(result: GlobalLocalizeResult, ray_weight: float = LOCAL_RAY_WEIGHT) -> float:
    return float(result.score) + ray_weight * (2.0 * float(result.ray_score) - 1.0)


def lidar_full_map(occ: OccupancyMap, scan: conv.LaserScan2D, truth: conv.Pose2D) -> Attempt:
    result = global_localize_scan(occ, scan, full_map=True)
    err_m, err_deg = _errors(result.pose, truth)
    accepted = (
        result.score >= FULL_MIN_SCORE
        and result.ray_mae_m <= FULL_MAX_RAY_MAE_M
        and not result.ambiguous
    )
    return Attempt(err_m, err_deg, bool(accepted), float(result.score))


def lidar_from_hints(
    occ: OccupancyMap,
    scan: conv.LaserScan2D,
    hints: Sequence[conv.Pose2D],
    truth: conv.Pose2D,
    *,
    search_radius_m: float = 1.5,
    yaw_window_deg: float = 120.0,
) -> Optional[Attempt]:
    """Scan-search around each hint; keep the best match."""
    best: Optional[GlobalLocalizeResult] = None
    for hint in hints:
        try:
            result = global_localize_scan(
                occ,
                scan,
                hint=hint,
                full_map=False,
                search_radius_m=search_radius_m,
                local_yaw_window_deg=yaw_window_deg,
            )
        except (RuntimeError, ValueError):
            continue
        if best is None or _combined(result) > _combined(best):
            best = result
    if best is None:
        return None
    err_m, err_deg = _errors(best.pose, truth)
    accepted = best.score >= LOCAL_MIN_SCORE and best.ray_mae_m <= LOCAL_MAX_RAY_MAE_M
    return Attempt(err_m, err_deg, bool(accepted), float(best.score))


def runtime_recovery(
    occ: OccupancyMap,
    scan: conv.LaserScan2D,
    index: KeyframeIndex,
    tokens: np.ndarray,
    points: np.ndarray,
    truth: conv.Pose2D,
) -> Optional[Attempt]:
    """``recover_pose`` as the robot would run it, judged against ground truth."""

    def search(hint: conv.Pose2D, radius_m: float, yaw_window_deg: float) -> Optional[LidarMatch]:
        try:
            result = global_localize_scan(
                occ, scan, hint=hint, full_map=False,
                search_radius_m=radius_m, local_yaw_window_deg=yaw_window_deg,
            )
        except (RuntimeError, ValueError):
            return None
        return LidarMatch(result.pose, float(result.score), float(result.ray_mae_m))

    recovery = recover_pose(index, tokens, points, search)
    if recovery.winner is None or recovery.winner.lidar is None:
        return None
    err_m, err_deg = _errors(recovery.winner.lidar.pose, truth)
    return Attempt(err_m, err_deg, recovery.accepted, float(recovery.winner.lidar.score))


def summarize(attempts: Sequence[Optional[Attempt]]) -> Dict[str, float]:
    """Shares of all queries. ``accepted_wrong`` is the dangerous one."""
    n = max(1, len(attempts))
    done = [a for a in attempts if a is not None]
    return {
        "queries": len(attempts),
        "right": sum(a.right for a in done) / n,
        "accepted_right": sum(a.right and a.accepted for a in done) / n,
        "accepted_wrong": sum((not a.right) and a.accepted for a in done) / n,
        "refused": sum(not a.accepted for a in done) / n + (len(attempts) - len(done)) / n,
    }


def evaluate_reloc(
    occ: OccupancyMap,
    mapping: LilocSequence,
    map_feats: FrameFeatures,
    query: LilocSequence,
    query_feats: FrameFeatures,
    place_methods: Sequence[str],
    *,
    query_stride: int = 4,
    top_k: int = 5,
    seed: int = 0,
    log=print,
) -> Dict[str, dict]:
    keyframes = select_keyframes(mapping.poses)
    kf_poses = [mapping.poses[i] for i in keyframes]
    q_frames = [
        i
        for i in range(0, len(query), max(1, int(query_stride)))
        if query.scan(query.stamps[i]) is not None
    ]
    # ``<method>+fit`` refines each place guess into a camera pose (patch
    # matches with depth) before the lidar search; the keyframe pose is the
    # hint when no fit is found.
    rankings = {
        name: rank_named(
            name[: -len(FIT_SUFFIX)] if name.endswith(FIT_SUFFIX) else name,
            map_feats, keyframes, query_feats, q_frames, top=top_k,
        )
        for name in place_methods
        if name != RUNTIME
    }
    map_points = PointCache(mapping, map_feats)
    query_points = PointCache(query, query_feats)
    index: Optional[KeyframeIndex] = None
    if RUNTIME in place_methods:
        index = KeyframeIndex()
        for n, i in enumerate(keyframes):
            index.add(n, kf_poses[n], map_feats.tokens(i), map_points(i))
        index.compress()
    rng = np.random.default_rng(seed)
    runs: Dict[str, List[Optional[Attempt]]] = {"lidar": [], "oracle+lidar": []}
    for name in rankings:
        runs[f"{name}+lidar"] = []
    if index is not None:
        runs[RUNTIME] = []
    for n, i in enumerate(q_frames):
        truth = query.poses[i]
        scan = query.scan(query.stamps[i])
        runs["lidar"].append(lidar_full_map(occ, scan, truth))
        bearing = float(rng.uniform(-math.pi, math.pi))
        turn = math.radians(30.0) * (1.0 if rng.random() < 0.5 else -1.0)
        oracle = conv.Pose2D(
            truth.x + math.cos(bearing), truth.y + math.sin(bearing), truth.theta + turn
        )
        runs["oracle+lidar"].append(lidar_from_hints(occ, scan, [oracle], truth))
        for name, ranking in rankings.items():
            hints = [kf_poses[int(k)] for k in ranking.order[n, :top_k]]
            if name.endswith(FIT_SUFFIX):
                q_tokens = query_feats.tokens(i)
                for h, k in enumerate(ranking.order[n, :top_k]):
                    k = int(k)
                    fit = fit_to_keyframe(
                        q_tokens, query_points(i),
                        map_feats.tokens(keyframes[k]), map_points(keyframes[k]), kf_poses[k],
                    )
                    if fit is not None:
                        hints[h] = fit.pose
            runs[f"{name}+lidar"].append(lidar_from_hints(occ, scan, hints, truth))
        if index is not None:
            runs[RUNTIME].append(
                runtime_recovery(occ, scan, index, query_feats.tokens(i), query_points(i), truth)
            )
        if (n + 1) % 20 == 0 or n + 1 == len(q_frames):
            log(f"  {query.name}: relocalized {n + 1}/{len(q_frames)}")
    return {name: summarize(attempts) for name, attempts in runs.items()}

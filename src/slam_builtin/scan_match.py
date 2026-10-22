"""Local correlative scan-to-map matching for continuous tracking."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..nav.global_localize import (
    OccupancyMap,
    _inflate_occupied,
    _score_pose,
    scan_endpoints_base_link,
)
from ..geom import conversions as conv


def occupancy_map_from_int16(
    grid: np.ndarray,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> OccupancyMap:
    return OccupancyMap(
        grid=np.asarray(grid, dtype=np.int16),
        resolution=float(resolution),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
    )


def score_pose(
    occ_map: OccupancyMap,
    scan_xy: np.ndarray,
    pose: conv.Pose2D,
    *,
    occupied_lookup: np.ndarray,
) -> float:
    return _score_pose(
        occ_map,
        scan_xy,
        pose,
        occupied_lookup=occupied_lookup,
        min_in_map_points=20,
        min_in_map_ratio=0.2,
    ).score


def _required_improvement(
    prior_score: float, dist_m: float, dyaw_rad: float
) -> float:
    """Larger jumps and already-good priors need a clearer score win.

    Prevents A↔B peak flipping when two discrete poses score within noise.
    Kept mild: odom drift is real and must stay correctable — an overly
    strict bar let the pose run ahead of the robot (goals "reached" early)
    until a manual refine snapped it back.
    """
    base = 0.06 if prior_score >= 0.25 else 0.03
    return base + 0.10 * dist_m + 0.08 * (abs(dyaw_rad) / (math.pi / 4.0))


def _candidate_scores(
    occ_map: OccupancyMap,
    scan_xy: np.ndarray,
    cand_x: np.ndarray,
    cand_y: np.ndarray,
    yaw: float,
    *,
    occupied_lookup: np.ndarray,
    min_in_map_points: int = 20,
    min_in_map_ratio: float = 0.2,
) -> np.ndarray:
    """Vectorized ``_score_pose`` for many XY origins at one yaw.

    The engine calls this at ~10 Hz from a background thread; a Python loop
    over thousands of candidates starved the module event loop (nav drive
    RPCs were timing out), so all candidates are scored with numpy at once.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    rx = c * scan_xy[:, 0] - s * scan_xy[:, 1]
    ry = s * scan_xy[:, 0] + c * scan_xy[:, 1]

    # (n_candidates, n_points)
    wx = cand_x[:, None] + rx[None, :]
    wy = cand_y[:, None] + ry[None, :]

    grid = occ_map.grid
    height, width = grid.shape
    res = occ_map.resolution
    cols = np.floor((wx - occ_map.origin_x) / res).astype(np.int32)
    rows = np.floor((wy - occ_map.origin_y) / res).astype(np.int32)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows_c = np.clip(rows, 0, height - 1)
    cols_c = np.clip(cols, 0, width - 1)

    hits = occupied_lookup[rows_c, cols_c] & inside
    free = (grid[rows_c, cols_c] == 0) & inside

    total = scan_xy.shape[0]
    in_map = inside.sum(axis=1)
    denom = np.maximum(in_map, 1)
    hit_rate = hits.sum(axis=1) / denom
    free_rate = free.sum(axis=1) / denom
    out_of_map = 1.0 - in_map / float(total)

    scores = hit_rate - 0.7 * free_rate - 1.2 * out_of_map
    min_points = max(min_in_map_points, int(total * min_in_map_ratio))
    scores[in_map < min_points] = float("-inf")
    return scores


def shift_axes(
    current: conv.Pose2D, matched: conv.Pose2D
) -> Tuple[float, float, float]:
    """``(along_m, cross_m, dyaw_rad)`` of ``matched`` in ``current``'s frame.

    Along is forward (heading). Cross is left. A straight hallway constrains
    cross and leaves along repetitive, so callers can keep one and drop the other.
    """
    dx = matched.x - current.x
    dy = matched.y - current.y
    c = math.cos(current.theta)
    s = math.sin(current.theta)
    along = c * dx + s * dy
    cross = -s * dx + c * dy
    dyaw = conv.normalize_angle(matched.theta - current.theta)
    return along, cross, dyaw


def lateral_correction_pose(
    current: conv.Pose2D,
    matched: conv.Pose2D,
    *,
    max_along_m: float = 0.15,
    max_cross_m: float = 1.20,
    max_yaw_rad: float = math.radians(20.0),
    min_cross_m: float = 0.08,
) -> Optional[conv.Pose2D]:
    """Pose that moves ``current`` sideways toward ``matched``.

    ``None`` when the peak slides along the hall, turns far enough to be a
    different corridor, or jumps past the far wall of this one. Heading and
    the along-hall coordinate stay at ``current``; only the cross-track term
    moves. A modest heading error (the slip that walks a skid-steer into the
    wall) does not veto the sideways step.
    """
    along, cross, dyaw = shift_axes(current, matched)
    if abs(along) > max_along_m or abs(cross) > max_cross_m or abs(dyaw) > max_yaw_rad:
        return None
    if abs(cross) < min_cross_m:
        return None
    c = math.cos(current.theta)
    s = math.sin(current.theta)
    return conv.Pose2D(
        current.x - s * cross,
        current.y + c * cross,
        current.theta,
    )


def refine_pose(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    prior: conv.Pose2D,
    *,
    xy_half_m: float = 0.30,
    xy_step_m: float = 0.05,
    yaw_half_deg: float = 12.0,
    yaw_step_deg: float = 2.0,
    inflate_cells: int = 2,
    min_score: float = 0.08,
    max_scan_points: int = 120,
    along_half_m: Optional[float] = None,
    cross_half_m: Optional[float] = None,
) -> Tuple[Optional[conv.Pose2D], float, float]:
    """Search a local window around ``prior``.

    Returns ``(pose_or_none, best_score, prior_score)``. Rejects candidates that
    do not beat the prior by a distance-weighted margin (anti-oscillation).

    ``along_half_m`` and ``cross_half_m`` search in the prior heading frame
    instead of a square. Cross is the wide axis so a straight-hall drift stays
    visible; along stays short so the search cannot slide down the hall.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 20:
        return None, float("-inf"), float("-inf")
    if max_scan_points > 0 and scan_xy.shape[0] > max_scan_points:
        pick = np.linspace(0, scan_xy.shape[0] - 1, max_scan_points, dtype=np.int32)
        scan_xy = scan_xy[pick]

    occupied = occ_map.grid >= 50
    lookup = _inflate_occupied(occupied, inflate_cells)
    prior_score = float(
        _candidate_scores(
            occ_map,
            scan_xy,
            np.array([prior.x]),
            np.array([prior.y]),
            prior.theta,
            occupied_lookup=lookup,
        )[0]
    )

    # Shrink the square search a little when already well aligned (alias
    # guard). An explicit along/cross window is the caller's: shrinking the
    # cross axis is how a hallway drift larger than 0.2 m became invisible.
    anisotropic = along_half_m is not None and cross_half_m is not None
    if prior_score >= 0.35 and not anisotropic:
        xy_half_m = min(xy_half_m, 0.20)
        yaw_half_deg = min(yaw_half_deg, 8.0)

    if anisotropic:
        along_offsets = np.arange(-float(along_half_m), float(along_half_m) + 1e-9, xy_step_m)
        cross_offsets = np.arange(-float(cross_half_m), float(cross_half_m) + 1e-9, xy_step_m)
        aa, cc = np.meshgrid(along_offsets, cross_offsets)
        cth = math.cos(prior.theta)
        sth = math.sin(prior.theta)
        cand_x = prior.x + cth * aa.ravel() - sth * cc.ravel()
        cand_y = prior.y + sth * aa.ravel() + cth * cc.ravel()
    else:
        offsets = np.arange(-xy_half_m, xy_half_m + 1e-9, xy_step_m)
        ox, oy = np.meshgrid(offsets, offsets)
        cand_x = prior.x + ox.ravel()
        cand_y = prior.y + oy.ravel()

    yaw_half = math.radians(yaw_half_deg)
    yaw_step = math.radians(yaw_step_deg)
    n_yaw = max(1, int(round(2.0 * yaw_half / yaw_step)) + 1)

    best_pose: Optional[conv.Pose2D] = None
    best_score = float("-inf")
    for it in range(n_yaw):
        dth = -yaw_half + it * yaw_step
        yaw = conv.normalize_angle(prior.theta + dth)
        scores = _candidate_scores(
            occ_map, scan_xy, cand_x, cand_y, yaw, occupied_lookup=lookup
        )
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score = float(scores[idx])
            best_pose = conv.Pose2D(float(cand_x[idx]), float(cand_y[idx]), yaw)

    if best_pose is None or best_score < min_score:
        return None, best_score, prior_score

    dist = math.hypot(best_pose.x - prior.x, best_pose.y - prior.y)
    dyaw = abs(conv.normalize_angle(best_pose.theta - prior.theta))
    need = _required_improvement(prior_score, dist, dyaw)
    if best_score < prior_score + need:
        return None, best_score, prior_score
    return best_pose, best_score, prior_score

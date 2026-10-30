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


# Farther than this from the winning cell, a high score is a second wall
# rather than the same peak sampled on the next grid point.
_RIVAL_SEP_M = 0.40


def scan_map_fit(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    pose: conv.Pose2D,
    *,
    min_score: float = 0.5,
) -> str:
    """``open`` when the map cannot judge ``pose``, else ``fit`` or ``miss``.

    Unknown cells do not count. A heading that drops returns into empty space
    the map already knows is a miss, even if part of the scan is new ground.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 20:
        return "open"
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    wx = pose.x + c * scan_xy[:, 0] - s * scan_xy[:, 1]
    wy = pose.y + s * scan_xy[:, 0] + c * scan_xy[:, 1]
    grid = occ_map.grid
    height, width = grid.shape
    res = occ_map.resolution
    cols = np.floor((wx - occ_map.origin_x) / res).astype(np.int32)
    rows = np.floor((wy - occ_map.origin_y) / res).astype(np.int32)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if int(inside.sum()) < 20:
        return "open"
    rows_c = np.clip(rows, 0, height - 1)
    cols_c = np.clip(cols, 0, width - 1)
    known = (grid[rows_c, cols_c] >= 0) & inside
    if int(known.sum()) / float(inside.sum()) < 0.25:
        return "open"
    # Score only the points that fall inside the grid. Points past the edge
    # are how the map grows; they must not veto a scan whose overlap hits walls.
    lookup = _inflate_occupied(grid >= 50, 2)
    hits = lookup[rows_c, cols_c] & inside
    free = (grid[rows_c, cols_c] == 0) & inside
    denom = float(inside.sum())
    overlap = float(hits.sum()) / denom - 0.7 * float(free.sum()) / denom
    if overlap >= min_score:
        return "fit"
    return "miss"


def score_at_pose(
    occ_map: OccupancyMap, scan: conv.LaserScan2D, pose: conv.Pose2D
) -> float:
    """How well ``scan`` lands on ``occ_map`` at ``pose``.

    ``-inf`` when too little of the scan overlaps the map to judge. A high
    score means returns sit on walls already drawn; near or below zero they
    land in empty space.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 20:
        return float("-inf")
    if scan_xy.shape[0] > 120:
        pick = np.linspace(0, scan_xy.shape[0] - 1, 120, dtype=np.int32)
        scan_xy = scan_xy[pick]
    occupied = occ_map.grid >= 50
    lookup = _inflate_occupied(occupied, 2)
    return float(
        _candidate_scores(
            occ_map,
            scan_xy,
            np.array([pose.x]),
            np.array([pose.y]),
            pose.theta,
            occupied_lookup=lookup,
        )[0]
    )


def search_heading(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    pose: conv.Pose2D,
    *,
    yaw_step_deg: float = 4.0,
    min_score: float = 0.5,
) -> Tuple[Optional[conv.Pose2D], float]:
    """Best heading at ``pose``'s position. XY stays put.

    Used after an in-place spin, when wheel yaw can be tens of degrees off
    and the narrow tracking window would not see the walls.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 20:
        return None, float("-inf")
    if scan_xy.shape[0] > 120:
        pick = np.linspace(0, scan_xy.shape[0] - 1, 120, dtype=np.int32)
        scan_xy = scan_xy[pick]
    occupied = occ_map.grid >= 50
    lookup = _inflate_occupied(occupied, 2)
    step = math.radians(yaw_step_deg)
    n = max(1, int(round(2.0 * math.pi / step)))
    scored: list[tuple[float, float]] = []
    for i in range(n):
        yaw = conv.normalize_angle(pose.theta + i * step)
        score = float(
            _candidate_scores(
                occ_map,
                scan_xy,
                np.array([pose.x]),
                np.array([pose.y]),
                yaw,
                occupied_lookup=lookup,
            )[0]
        )
        scored.append((score, yaw))
    # A centered hall scores the same forwards and backwards. Among headings
    # that tie the best, stay with the one nearest the wheels.
    best_score = max(item[0] for item in scored)
    near = [
        item
        for item in scored
        if math.isfinite(item[0]) and item[0] >= best_score - 0.05
    ]
    if not near:
        return None, best_score
    _score, best_yaw = min(
        near, key=lambda item: abs(conv.normalize_angle(item[1] - pose.theta))
    )
    chosen = conv.Pose2D(pose.x, pose.y, best_yaw)
    # Points past the grid edge lower every heading by about the same amount.
    # Keep a heading whose overlap sits on the walls.
    if best_score < min_score and scan_map_fit(
        occ_map, scan, chosen, min_score=min_score
    ) != "fit":
        return None, best_score
    return chosen, best_score


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
) -> Tuple[Optional[conv.Pose2D], float, float, float]:
    """Search a square window around ``prior``.

    Returns ``(pose_or_none, best_score, prior_score, rival_score)``.
    ``rival_score`` is the best peak at least ``_RIVAL_SEP_M`` from the winner,
    or ``-inf`` when the window has only one blob. Rejects candidates that
    do not beat the prior by a distance-weighted margin (anti-oscillation).
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 20:
        return None, float("-inf"), float("-inf"), float("-inf")
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

    # Shrink the square search a little when already well aligned (alias guard).
    if prior_score >= 0.35:
        xy_half_m = min(xy_half_m, 0.20)
        yaw_half_deg = min(yaw_half_deg, 8.0)

    offsets = np.arange(-xy_half_m, xy_half_m + 1e-9, xy_step_m)
    ox, oy = np.meshgrid(offsets, offsets)
    cand_x = prior.x + ox.ravel()
    cand_y = prior.y + oy.ravel()

    yaw_half = math.radians(yaw_half_deg)
    yaw_step = math.radians(yaw_step_deg)
    n_yaw = max(1, int(round(2.0 * yaw_half / yaw_step)) + 1)
    # A cell next to the winner is the same blob. A rival is another pose
    # far enough away to be a different wall.
    rival_sep2 = _RIVAL_SEP_M * _RIVAL_SEP_M

    best_pose: Optional[conv.Pose2D] = None
    best_score = float("-inf")
    peaks: list[tuple[float, float, float]] = []
    for it in range(n_yaw):
        dth = -yaw_half + it * yaw_step
        yaw = conv.normalize_angle(prior.theta + dth)
        scores = _candidate_scores(
            occ_map, scan_xy, cand_x, cand_y, yaw, occupied_lookup=lookup
        )
        idx = int(np.argmax(scores))
        local_score = float(scores[idx])
        if not math.isfinite(local_score):
            continue
        peaks.append((local_score, float(cand_x[idx]), float(cand_y[idx])))
        dx = cand_x - cand_x[idx]
        dy = cand_y - cand_y[idx]
        far = (dx * dx + dy * dy) >= rival_sep2
        if bool(np.any(far)):
            far_scores = np.where(far, scores, float("-inf"))
            far_idx = int(np.argmax(far_scores))
            far_score = float(far_scores[far_idx])
            if math.isfinite(far_score):
                peaks.append(
                    (far_score, float(cand_x[far_idx]), float(cand_y[far_idx]))
                )
        if local_score > best_score:
            best_score = local_score
            best_pose = conv.Pose2D(float(cand_x[idx]), float(cand_y[idx]), yaw)

    rival_score = float("-inf")
    if best_pose is not None:
        for score, x, y in peaks:
            if (x - best_pose.x) ** 2 + (y - best_pose.y) ** 2 < rival_sep2:
                continue
            if score > rival_score:
                rival_score = score

    if best_pose is None or best_score < min_score:
        return None, best_score, prior_score, rival_score

    dist = math.hypot(best_pose.x - prior.x, best_pose.y - prior.y)
    dyaw = abs(conv.normalize_angle(best_pose.theta - prior.theta))
    need = _required_improvement(prior_score, dist, dyaw)
    if best_score < prior_score + need:
        return None, best_score, prior_score, rival_score
    return best_pose, best_score, prior_score, rival_score

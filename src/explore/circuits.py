"""Structure-phase geometry: wall-following hops and inner circuits."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .clients import Pose
from .grid import Candidate, OccGrid, clear_mask


def path_length(points: Sequence[Tuple[float, float]]) -> float:
    return float(
        sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    )


def _nearest_wall(grid: OccGrid, pose: Pose, search_m: float) -> Optional[Tuple[float, float]]:
    r0, c0 = grid.to_cell(pose.x, pose.y)
    radius = int(math.ceil(search_m / grid.resolution))
    r_lo, r_hi = max(0, r0 - radius), min(grid.rows, r0 + radius + 1)
    c_lo, c_hi = max(0, c0 - radius), min(grid.cols, c0 + radius + 1)
    window = grid.occupied_mask()[r_lo:r_hi, c_lo:c_hi]
    hits = np.argwhere(window)
    if hits.size == 0:
        return None
    xs = grid.x_min + (hits[:, 1] + c_lo + 0.5) * grid.resolution
    ys = grid.y_min + (hits[:, 0] + r_lo + 0.5) * grid.resolution
    d = np.hypot(xs - pose.x, ys - pose.y)
    i = int(np.argmin(d))
    if d[i] > search_m:
        return None
    return float(xs[i]), float(ys[i])


def facing_wall_theta(grid: OccGrid, pose: Pose, search_m: float = 3.0) -> Optional[float]:
    wall = _nearest_wall(grid, pose, search_m)
    if wall is None:
        return None
    return math.atan2(wall[1] - pose.y, wall[0] - pose.x)


def _segment_clear(grid: OccGrid, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    free = grid.free_mask()
    n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.resolution / 2.0)))
    for i in range(n + 1):
        t = i / n
        r, c = grid.to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not grid.in_bounds(r, c) or not free[r, c]:
            return False
    return True


def _wall_distance(grid: OccGrid, x: float, y: float, search_m: float) -> float:
    wall = _nearest_wall(grid, Pose(x, y, 0.0), search_m)
    return math.inf if wall is None else math.hypot(wall[0] - x, wall[1] - y)


def next_wall_hop(
    grid: OccGrid,
    pose: Pose,
    *,
    hop_m: float = 1.5,
    standoff_m: float = 0.8,
    search_m: float = 3.0,
    prev_travel: Optional[Tuple[float, float]] = None,
) -> Optional[Candidate]:
    wall = _nearest_wall(grid, pose, search_m)
    if wall is None:
        return None
    nx, ny = pose.x - wall[0], pose.y - wall[1]
    norm = math.hypot(nx, ny)
    if norm < 1e-9:
        return None
    nx, ny = nx / norm, ny / norm
    base = (wall[0] + nx * standoff_m, wall[1] + ny * standoff_m)
    along = (-ny, nx)
    if prev_travel is not None and along[0] * prev_travel[0] + along[1] * prev_travel[1] < 0:
        along = (-along[0], -along[1])
    attempts = [(along, base), ((along[1], -along[0]), (pose.x, pose.y)),
                ((-along[1], along[0]), (pose.x, pose.y))]
    for travel, origin in attempts:
        hop = hop_m
        while hop >= 0.5:
            target = (origin[0] + travel[0] * hop, origin[1] + travel[1] * hop)
            if _segment_clear(grid, (pose.x, pose.y), target) and (
                _wall_distance(grid, target[0], target[1], search_m) >= standoff_m - 0.05
            ):
                return Candidate(target[0], target[1], "circuit")
            hop -= 0.25
    return None


def circuit_closed(
    path_m: float,
    pose: Pose,
    start: Pose,
    *,
    min_path_m: float = 12.0,
    close_m: float = 1.5,
    far_m: float = 0.0,
    min_far_m: float = 3.0,
) -> bool:
    return (
        path_m >= min_path_m
        and far_m >= min_far_m
        and math.hypot(pose.x - start.x, pose.y - start.y) < close_m
    )


def inner_circuit_goals(
    grid: OccGrid, start: Pose, *, n: int = 3, clearance_m: float = 1.0
) -> List[Candidate]:
    clear = clear_mask(grid, clearance_m)
    step = max(1, int(round(0.25 / grid.resolution)))
    cells = [
        grid.to_world(r, c)
        for r in range(0, grid.rows, step)
        for c in range(0, grid.cols, step)
        if clear[r, c]
    ]
    if not cells:
        return [Candidate(start.x, start.y, "circuit")]
    pts = np.asarray(cells, dtype=float)
    dist = np.hypot(pts[:, 0] - start.x, pts[:, 1] - start.y)
    picks: List[Candidate] = []
    for _ in range(min(n, len(cells))):
        i = int(np.argmax(dist))
        picks.append(Candidate(float(pts[i, 0]), float(pts[i, 1]), "circuit"))
        dist = np.minimum(dist, np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]))
    picks.append(Candidate(start.x, start.y, "circuit"))
    return picks

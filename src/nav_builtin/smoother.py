"""Post-process global plans for the builtin path follower."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .planner import line_of_sight
from .types import OccupancyGrid, Path2D


def _resample_polyline(
    points: Sequence[Tuple[float, float]],
    spacing_m: float,
) -> List[Tuple[float, float]]:
    """Resample a polyline at roughly ``spacing_m`` intervals (keeps endpoints)."""
    if len(points) < 2:
        return list(points)
    spacing = max(float(spacing_m), 1e-3)
    out: List[Tuple[float, float]] = [points[0]]
    carry = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        dist = spacing - carry
        while dist <= seg:
            t = dist / seg
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            dist += spacing
        carry = max(0.0, spacing - (seg - (dist - spacing)))
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _shortcut_smooth(
    points: Sequence[Tuple[float, float]],
    costs: np.ndarray,
    occ: OccupancyGrid,
) -> List[Tuple[float, float]]:
    """Greedy string-pull using costmap line-of-sight (Nav2 SmoothPath-ish)."""
    if len(points) < 3:
        return list(points)
    pts = list(points)
    out = [pts[0]]
    i = 0
    n = len(pts)
    while i < n - 1:
        best_j = i + 1
        r0, c0 = occ.world_to_cell(pts[i][0], pts[i][1])
        if not occ.in_bounds(r0, c0):
            out.append(pts[i + 1])
            i += 1
            continue
        for j in range(n - 1, i, -1):
            r1, c1 = occ.world_to_cell(pts[j][0], pts[j][1])
            if not occ.in_bounds(r1, c1):
                continue
            if line_of_sight(costs, (r0, c0), (r1, c1)):
                best_j = j
                break
        out.append(pts[best_j])
        i = best_j
    return out


def smooth_path(
    path: Path2D,
    costs: np.ndarray,
    occ: OccupancyGrid,
    *,
    enabled: bool = True,
    sample_spacing_m: float = 0.10,
) -> Path2D:
    """Shortcut + resample a global plan on the inflated costmap."""
    if not enabled or path.empty or len(path.points) < 2:
        return path
    shortened = _shortcut_smooth(path.points, costs, occ)
    dense = _resample_polyline(shortened, sample_spacing_m)
    if len(dense) < 2:
        return path
    return Path2D(points=tuple(dense), goal_theta=path.goal_theta)


def smooth_plan_path(
    path: Path2D,
    map_data: dict,
    *,
    inflation_radius_m: float,
    robot_radius_m: float,
    cost_scaling_factor: float,
    enabled: bool = True,
    sample_spacing_m: float = 0.10,
) -> Path2D:
    """Convenience wrapper: build costmap from map dict then smooth."""
    if not enabled:
        return path
    from .costmap import build_costmap, occupancy_from_bridge_map

    occ = occupancy_from_bridge_map(map_data)
    costs = build_costmap(
        occ,
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
        cost_scaling_factor=cost_scaling_factor,
    )
    return smooth_path(
        path,
        costs,
        occ,
        enabled=True,
        sample_spacing_m=sample_spacing_m,
    )

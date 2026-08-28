"""Shared path geometry helpers for builtin nav."""
from __future__ import annotations

import math
from typing import Tuple

from .types import Path2D, Pose2D


def closest_point_on_path(
    current: Pose2D,
    path: Path2D,
) -> Tuple[float, float, int, float]:
    """Return (x, y, segment_index, distance_along_path_m) for the closest point."""
    pts = path.points
    if not pts:
        return current.x, current.y, 0, 0.0
    if len(pts) == 1:
        return pts[0][0], pts[0][1], 0, 0.0

    best_d2 = math.inf
    best_xy = pts[0]
    best_seg = 0
    best_along = 0.0
    along = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            t = 0.0
            px, py = x0, y0
            seg_len = 0.0
        else:
            t = ((current.x - x0) * dx + (current.y - y0) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            px = x0 + t * dx
            py = y0 + t * dy
            seg_len = math.sqrt(seg_len2)
        d2 = (current.x - px) ** 2 + (current.y - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_xy = (px, py)
            best_seg = i
            best_along = along + t * seg_len
        along += seg_len
    return best_xy[0], best_xy[1], best_seg, best_along

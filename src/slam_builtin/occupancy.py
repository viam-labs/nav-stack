"""Log-odds occupancy grid insert / convert."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .types import LogOddsGrid

# Inverse sensor model (typical values for lidar occupancy grids).
L_OCC = 0.85
L_FREE = -0.40
L_MIN = -4.0
L_MAX = 4.0
# Unobserved cells stay at 0; clamp after updates.
OCC_THRESH = 0.65  # P(occ) threshold -> 100
FREE_THRESH = 0.35  # P(occ) below -> 0


def _prob_to_logodds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def empty_grid(
    *,
    resolution: float = 0.05,
    size_m: float = 20.0,
    origin_x: Optional[float] = None,
    origin_y: Optional[float] = None,
) -> LogOddsGrid:
    cells = max(8, int(math.ceil(size_m / resolution)))
    half = 0.5 * cells * resolution
    ox = -half if origin_x is None else float(origin_x)
    oy = -half if origin_y is None else float(origin_y)
    return LogOddsGrid(
        log_odds=np.zeros((cells, cells), dtype=np.float32),
        resolution=float(resolution),
        origin_x=ox,
        origin_y=oy,
    )


def from_occupancy_int16(
    grid: np.ndarray,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> LogOddsGrid:
    """Seed log-odds from ROS-style int16 occupancy (-1/0/100)."""
    g = np.asarray(grid, dtype=np.int16)
    lo = np.zeros(g.shape, dtype=np.float32)
    lo[g >= 50] = _prob_to_logodds(0.9)
    lo[(g >= 0) & (g < 50)] = _prob_to_logodds(0.1)
    return LogOddsGrid(
        log_odds=lo,
        resolution=float(resolution),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
    )


def to_occupancy_int16(grid: LogOddsGrid) -> np.ndarray:
    """Convert log-odds to ROS OccupancyGrid values (-1 unknown, 0 free, 100 occ)."""
    lo = grid.log_odds
    out = np.full(lo.shape, -1, dtype=np.int16)
    # Treat near-zero as unknown until observed.
    observed = np.abs(lo) > 0.05
    p = 1.0 / (1.0 + np.exp(-lo))
    out[observed & (p > OCC_THRESH)] = 100
    out[observed & (p < FREE_THRESH)] = 0
    # Mid-probability observed cells -> free-ish for nav (costmap treats >0 as occupied)
    mid = observed & (p >= FREE_THRESH) & (p <= OCC_THRESH)
    out[mid] = np.clip((p[mid] * 100.0).astype(np.int16), 1, 99)
    return out


def ensure_contains(
    grid: LogOddsGrid,
    x_m: float,
    y_m: float,
    *,
    margin_m: float = 2.0,
) -> LogOddsGrid:
    """Grow the grid so ``(x,y)`` plus margin fits; returns possibly new grid."""
    res = grid.resolution
    row, col = grid.world_to_cell(x_m, y_m)
    margin = int(math.ceil(margin_m / res))
    pad_bottom = max(0, margin - row)
    pad_left = max(0, margin - col)
    pad_top = max(0, row + margin + 1 - grid.height)
    pad_right = max(0, col + margin + 1 - grid.width)
    if pad_bottom == 0 and pad_left == 0 and pad_top == 0 and pad_right == 0:
        return grid
    new_lo = np.pad(
        grid.log_odds,
        ((pad_bottom, pad_top), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )
    return LogOddsGrid(
        log_odds=new_lo.astype(np.float32, copy=False),
        resolution=res,
        origin_x=grid.origin_x - pad_left * res,
        origin_y=grid.origin_y - pad_bottom * res,
    )


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[Tuple[int, int]]:
    """Inclusive Bresenham line of (row, col) cells."""
    cells: list[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def insert_scan(
    grid: LogOddsGrid,
    pose_x: float,
    pose_y: float,
    pose_theta: float,
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    *,
    range_min: float,
    range_max: float,
    max_beams: int = 180,
) -> LogOddsGrid:
    """Ray-cast a lidar scan into the log-odds grid (mutates + may expand)."""
    ranges = np.asarray(ranges, dtype=float)
    if ranges.size == 0:
        return grid

    idx = np.arange(ranges.size)
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(range_min))
        & (ranges <= float(range_max))
    )
    if not np.any(valid):
        return grid
    idx = idx[valid]
    if max_beams > 0 and idx.size > max_beams:
        pick = np.linspace(0, idx.size - 1, max_beams, dtype=np.int32)
        idx = idx[pick]

    # Expand so all endpoints fit.
    max_r = float(np.max(ranges[idx]))
    grid = ensure_contains(grid, pose_x, pose_y, margin_m=max_r + 1.0)

    lo = grid.log_odds
    h, w = lo.shape
    for i in idx:
        r = float(ranges[i])
        ang = float(angle_min + i * angle_increment)
        ca = math.cos(pose_theta + ang)
        sa = math.sin(pose_theta + ang)
        ex = pose_x + ca * r
        ey = pose_y + sa * r
        grid = ensure_contains(grid, ex, ey, margin_m=0.5)
        lo = grid.log_odds
        h, w = lo.shape

        r0, c0 = grid.world_to_cell(pose_x, pose_y)
        r1, c1 = grid.world_to_cell(ex, ey)
        if not (0 <= r0 < h and 0 <= c0 < w):
            continue
        line = _bresenham(r0, c0, r1, c1)
        if not line:
            continue
        # Free along ray (excluding endpoint).
        for rr, cc in line[:-1]:
            if 0 <= rr < h and 0 <= cc < w:
                lo[rr, cc] = float(np.clip(lo[rr, cc] + L_FREE, L_MIN, L_MAX))
        er, ec = line[-1]
        if 0 <= er < h and 0 <= ec < w:
            lo[er, ec] = float(np.clip(lo[er, ec] + L_OCC, L_MIN, L_MAX))

    return grid

"""Log-odds occupancy grid insert / convert."""

from __future__ import annotations

import math
from typing import Optional

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


def copy_grid(grid: LogOddsGrid) -> LogOddsGrid:
    """Deep copy so a paint can mutate without touching the published map."""
    return LogOddsGrid(
        log_odds=np.array(grid.log_odds, dtype=np.float32, copy=True),
        resolution=float(grid.resolution),
        origin_x=float(grid.origin_x),
        origin_y=float(grid.origin_y),
    )


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


# Log-odds equivalents of the probability thresholds. The sigmoid is
# monotonic, so comparing log-odds against these picks the same cells as
# comparing probabilities, without an ``exp`` over the whole grid.
_OCC_THRESH_LO = _prob_to_logodds(OCC_THRESH)
_FREE_THRESH_LO = _prob_to_logodds(FREE_THRESH)


def to_occupancy_int16(grid: LogOddsGrid) -> np.ndarray:
    """Convert log-odds to ROS OccupancyGrid values (-1 unknown, 0 free, 100 occ).

    Called for every ``get_map`` and every scan-match cache rebuild, so the
    full-grid work is three comparisons; only the (small) mid-probability band
    pays for the sigmoid.
    """
    lo = grid.log_odds
    out = np.full(lo.shape, -1, dtype=np.int16)
    # Treat near-zero as unknown until observed (both thresholds sit well
    # outside the +-0.05 unobserved band).
    out[lo > _OCC_THRESH_LO] = 100
    out[lo < _FREE_THRESH_LO] = 0
    # Mid-probability observed cells -> free-ish for nav (costmap treats >0 as occupied)
    mid = (np.abs(lo) > 0.05) & (lo >= _FREE_THRESH_LO) & (lo <= _OCC_THRESH_LO)
    if mid.any():
        p = 1.0 / (1.0 + np.exp(-lo[mid]))
        out[mid] = np.clip((p * 100.0).astype(np.int16), 1, 99)
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


def clear_disk(
    grid: LogOddsGrid,
    x_m: float,
    y_m: float,
    radius_m: float,
    *,
    free_log_odds: Optional[float] = None,
) -> int:
    """Paint a disk of free space into the log-odds grid (mutates in place).

    Returns the number of cells written. Cells outside the current grid are
    ignored (the grid is not expanded — erase only edits existing map).
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    res = float(grid.resolution)
    if res <= 0:
        raise ValueError("grid resolution must be > 0")

    # Strong free so a few later rays don't immediately re-occupy the patch.
    target = float(L_MIN) if free_log_odds is None else float(free_log_odds)
    target = float(np.clip(target, L_MIN, L_MAX))

    r_cells = max(1, int(math.ceil(radius_m / res)))
    cr, cc = grid.world_to_cell(x_m, y_m)
    lo = grid.log_odds
    h, w = lo.shape
    r0 = max(0, cr - r_cells)
    r1 = min(h - 1, cr + r_cells)
    c0 = max(0, cc - r_cells)
    c1 = min(w - 1, cc + r_cells)
    if r1 < r0 or c1 < c0:
        return 0

    # Compare cell centers to the erase center (meters).
    rows = np.arange(r0, r1 + 1, dtype=np.int32)[:, None]
    cols = np.arange(c0, c1 + 1, dtype=np.int32)[None, :]
    cx = grid.origin_x + (cols + 0.5) * res
    cy = grid.origin_y + (rows + 0.5) * res
    mask = (cx - x_m) ** 2 + (cy - y_m) ** 2 <= float(radius_m) ** 2
    cleared = int(np.count_nonzero(mask))
    if cleared:
        lo[r0 : r1 + 1, c0 : c1 + 1][mask] = target
    return cleared


def clear_rect(
    grid: LogOddsGrid,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    free_log_odds: Optional[float] = None,
) -> int:
    """Paint an axis-aligned box of free space (mutates in place).

    Returns the number of cells written. Cells outside the current grid are
    ignored. ``x_min``/``y_min`` need not be ordered relative to max.
    """
    res = float(grid.resolution)
    if res <= 0:
        raise ValueError("grid resolution must be > 0")
    xa, xb = sorted((float(x_min), float(x_max)))
    ya, yb = sorted((float(y_min), float(y_max)))
    if xb <= xa or yb <= ya:
        raise ValueError("clear_rect needs a non-empty box")

    target = float(L_MIN) if free_log_odds is None else float(free_log_odds)
    target = float(np.clip(target, L_MIN, L_MAX))

    r0, c0 = grid.world_to_cell(xa, ya)
    r1, c1 = grid.world_to_cell(xb, yb)
    lo = grid.log_odds
    h, w = lo.shape
    # world_to_cell can flip when axes disagree with array layout — normalize.
    rr0, rr1 = sorted((r0, r1))
    cc0, cc1 = sorted((c0, c1))
    rr0 = max(0, rr0)
    rr1 = min(h - 1, rr1)
    cc0 = max(0, cc0)
    cc1 = min(w - 1, cc1)
    if rr1 < rr0 or cc1 < cc0:
        return 0

    # Keep cells whose centers lie inside the world box (matches clear_disk).
    rows = np.arange(rr0, rr1 + 1, dtype=np.int32)[:, None]
    cols = np.arange(cc0, cc1 + 1, dtype=np.int32)[None, :]
    cx = grid.origin_x + (cols + 0.5) * res
    cy = grid.origin_y + (rows + 0.5) * res
    mask = (cx >= xa) & (cx <= xb) & (cy >= ya) & (cy <= yb)
    cleared = int(np.count_nonzero(mask))
    if cleared:
        lo[rr0 : rr1 + 1, cc0 : cc1 + 1][mask] = target
    return cleared


def mark_disk(
    grid: LogOddsGrid,
    x_m: float,
    y_m: float,
    radius_m: float,
    *,
    occupied_log_odds: Optional[float] = None,
) -> int:
    """Paint a disk of occupied space into the log-odds grid (mutates in place).

    Returns the number of cells written. Cells outside the current grid are
    ignored (the grid is not expanded).
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    res = float(grid.resolution)
    if res <= 0:
        raise ValueError("grid resolution must be > 0")

    target = float(L_MAX) if occupied_log_odds is None else float(occupied_log_odds)
    target = float(np.clip(target, L_MIN, L_MAX))

    r_cells = max(1, int(math.ceil(radius_m / res)))
    cr, cc = grid.world_to_cell(x_m, y_m)
    lo = grid.log_odds
    h, w = lo.shape
    r0 = max(0, cr - r_cells)
    r1 = min(h - 1, cr + r_cells)
    c0 = max(0, cc - r_cells)
    c1 = min(w - 1, cc + r_cells)
    if r1 < r0 or c1 < c0:
        return 0

    rows = np.arange(r0, r1 + 1, dtype=np.int32)[:, None]
    cols = np.arange(c0, c1 + 1, dtype=np.int32)[None, :]
    cx = grid.origin_x + (cols + 0.5) * res
    cy = grid.origin_y + (rows + 0.5) * res
    mask = (cx - x_m) ** 2 + (cy - y_m) ** 2 <= float(radius_m) ** 2
    marked = int(np.count_nonzero(mask))
    if marked:
        lo[r0 : r1 + 1, c0 : c1 + 1][mask] = target
    return marked


# Unobserved band used by ``to_occupancy_int16`` and extend-only paint.
_UNKNOWN_ABS_LO = 0.05


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
    extend_only: bool = False,
) -> LogOddsGrid:
    """Ray-cast a lidar scan into the log-odds grid (mutates + may expand).

    With ``extend_only``, only unobserved cells are written — already free or
    occupied cells stay put. Used when the resumed map baseline is untrusted
    so a bad pose cannot thicken a twin wall or carve through a good one.
    """
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

    # Expand once so every endpoint fits. Per-beam growth used to copy the
    # whole grid inside the ray loop.
    max_r = float(np.max(ranges[idx]))
    grid = ensure_contains(grid, pose_x, pose_y, margin_m=max_r + 1.0)

    beam = idx.astype(np.float64)
    beam_ranges = ranges[idx].astype(np.float64)
    angles = float(angle_min) + beam * float(angle_increment)
    ex = float(pose_x) + np.cos(float(pose_theta) + angles) * beam_ranges
    ey = float(pose_y) + np.sin(float(pose_theta) + angles) * beam_ranges
    res = float(grid.resolution)
    cols1 = np.floor((ex - grid.origin_x) / res).astype(np.int32)
    rows1 = np.floor((ey - grid.origin_y) / res).astype(np.int32)
    row0, col0 = grid.world_to_cell(float(pose_x), float(pose_y))
    _paint_rays(grid.log_odds, row0, col0, rows1, cols1, extend_only=bool(extend_only))
    return grid


def _paint_rays(
    lo: np.ndarray,
    row0: int,
    col0: int,
    rows1: np.ndarray,
    cols1: np.ndarray,
    *,
    extend_only: bool = False,
) -> None:
    """Mark Bresenham-like rays free, and their endpoints occupied.

    All beams are rasterized together so the work runs in numpy (the GIL is
    released) instead of a Python loop over every cell.
    """
    n = int(rows1.shape[0])
    if n == 0:
        return
    dr = rows1.astype(np.int32) - int(row0)
    dc = cols1.astype(np.int32) - int(col0)
    steps = np.maximum(np.abs(dr), np.abs(dc))
    max_steps = int(steps.max())
    h, w = lo.shape

    def _add(rows: np.ndarray, cols: np.ndarray, delta: float) -> None:
        if rows.size == 0:
            return
        if extend_only:
            keep = np.abs(lo[rows, cols]) <= _UNKNOWN_ABS_LO
            rows, cols = rows[keep], cols[keep]
            if rows.size == 0:
                return
        np.add.at(lo, (rows, cols), np.float32(delta))

    if max_steps <= 0:
        inside = (rows1 >= 0) & (rows1 < h) & (cols1 >= 0) & (cols1 < w)
        if np.any(inside):
            _add(rows1[inside], cols1[inside], L_OCC)
            np.clip(lo, L_MIN, L_MAX, out=lo)
        return

    t = np.arange(max_steps + 1, dtype=np.float64)
    denom = np.maximum(steps, 1).astype(np.float64)
    alpha = np.minimum(t[None, :] / denom[:, None], 1.0)
    rr = np.rint(int(row0) + dr.astype(np.float64)[:, None] * alpha).astype(np.int32)
    cc = np.rint(int(col0) + dc.astype(np.float64)[:, None] * alpha).astype(np.int32)
    valid = t[None, :] <= steps.astype(np.float64)[:, None]
    dup = (rr[:, 1:] == rr[:, :-1]) & (cc[:, 1:] == cc[:, :-1])
    valid[:, 1:] &= ~dup
    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    valid &= inside
    # Endpoint is occupied, not free — even when the last sample repeats it.
    is_hit = (rr == rows1[:, None]) & (cc == cols1[:, None])
    free = valid & ~is_hit
    if np.any(free):
        _add(rr[free], cc[free], L_FREE)
    hit = (rows1 >= 0) & (rows1 < h) & (cols1 >= 0) & (cols1 < w)
    if np.any(hit):
        _add(rows1[hit], cols1[hit], L_OCC)
    np.clip(lo, L_MIN, L_MAX, out=lo)

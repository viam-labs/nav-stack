"""Live occupancy-map health while mapping.

Detects early signs that the map is tearing (ghost parallel walls, inconsistent
wall angles across tiles) so mapping can tighten loop/revisit behavior before
the whole floor is doubled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Ghost = two occupied ridges with a thin free gap between them (0.15–0.55 m).
_GHOST_SEP_MIN_M = 0.15
_GHOST_SEP_MAX_M = 0.55
# Tile size for local wall-angle / hotspot estimates.
_TILE_M = 2.0
# Status thresholds on the occupied-cell ghost fraction and angle std (deg).
_WATCH_GHOST = 0.12
_DEGRADING_GHOST = 0.22
_WATCH_ANGLE_STD = 12.0
_DEGRADING_ANGLE_STD = 20.0
DEGRADING_ANGLE_STD = _DEGRADING_ANGLE_STD
# Ignore maps that are still too small to judge.
_MIN_OCC_CELLS = 400


@dataclass(frozen=True)
class MapHealth:
    status: str  # ok | watch | degrading | cold
    ghost_frac: float
    thick_frac: float
    angle_std_deg: float
    occupied_cells: int
    hotspots: Tuple[Tuple[float, float, float], ...]  # (frac, x_m, y_m)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "ghost_frac": round(self.ghost_frac, 3),
            "thick_frac": round(self.thick_frac, 3),
            "angle_std_deg": round(self.angle_std_deg, 1),
            "occupied_cells": int(self.occupied_cells),
            "hotspots": [
                {"frac": round(f, 2), "x": round(x, 1), "y": round(y, 1)}
                for f, x, y in self.hotspots
            ],
        }


def assess_occupancy(
    grid: np.ndarray,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
    baseline: Optional[np.ndarray] = None,
    max_hotspots: int = 8,
    max_side: int = 800,
) -> MapHealth:
    """Score a Nav2-style occupancy grid (-1 unknown, 0 free, ≥50 occupied).

    With ``baseline`` (same shape and frame as ``grid``, e.g. the map a mapping
    session resumed on), only walls absent from the baseline are scored, so
    damage inherited from the loaded map does not mask or fake new tearing.

    Large maps are downsampled so a live watchdog stays cheap (~tens of ms).
    """
    cells = np.asarray(grid)
    if cells.ndim != 2:
        raise ValueError("occupancy grid must be 2D")
    res = float(resolution)
    if res <= 0.0:
        raise ValueError("resolution must be positive")
    occ = cells >= 50
    free = (cells >= 0) & (cells < 50)
    if baseline is None:
        focus = occ
    else:
        base = np.asarray(baseline)
        if base.shape != cells.shape:
            raise ValueError("baseline must match the grid shape")
        # One-cell slack so re-observing a known wall is not "new".
        focus = occ & ~_dilate(base >= 50)
    n_occ = int(focus.sum())
    if n_occ < _MIN_OCC_CELLS:
        return MapHealth(
            status="cold",
            ghost_frac=0.0,
            thick_frac=0.0,
            angle_std_deg=0.0,
            occupied_cells=n_occ,
            hotspots=(),
        )

    ys, xs = np.where(focus)
    pad = max(2, int(round((_GHOST_SEP_MAX_M + 0.1) / res)))
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(cells.shape[0], int(ys.max()) + pad + 1)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(cells.shape[1], int(xs.max()) + pad + 1)
    sub = occ[r0:r1, c0:c1]
    free_sub = free[r0:r1, c0:c1]
    focus_sub = focus[r0:r1, c0:c1]
    work_res = res
    # Coarsen big maps; keep world hotspot coords via the same origins.
    step = 1
    h, w = sub.shape
    if max(h, w) > max_side:
        step = int(math.ceil(max(h, w) / float(max_side)))
        sub = sub[::step, ::step]
        free_sub = free_sub[::step, ::step]
        focus_sub = focus_sub[::step, ::step]
        work_res = res * step

    # Ghost pairs may straddle a baseline wall and a new one; count the new side.
    ghost = _ghost_mask(sub, free_sub, work_res) & focus_sub
    thick = _thick_mask(sub) & focus_sub
    n_sub = max(1, int(focus_sub.sum()))
    ghost_frac = float(ghost.sum()) / float(n_sub)
    thick_frac = float(thick.sum()) / float(n_sub)
    angle_std = _angle_std_deg(focus_sub, work_res)
    hotspots = _ghost_hotspots(
        ghost,
        focus_sub,
        res=work_res,
        origin_x=origin_x + c0 * res,
        origin_y=origin_y + r0 * res,
        row0=0,
        col0=0,
        max_hotspots=max_hotspots,
    )
    status = _status(ghost_frac, angle_std)
    return MapHealth(
        status=status,
        ghost_frac=ghost_frac,
        thick_frac=thick_frac,
        angle_std_deg=angle_std,
        occupied_cells=n_occ,
        hotspots=tuple(hotspots),
    )


def _status(ghost_frac: float, angle_std: float) -> str:
    if ghost_frac >= _DEGRADING_GHOST or angle_std >= _DEGRADING_ANGLE_STD:
        return "degrading"
    if ghost_frac >= _WATCH_GHOST or angle_std >= _WATCH_ANGLE_STD:
        return "watch"
    return "ok"


def _ghost_mask(occ: np.ndarray, free: np.ndarray, res: float) -> np.ndarray:
    """Occupied cells that face another occupied ridge across a thin free gap."""
    sep_min = max(2, int(round(_GHOST_SEP_MIN_M / res)))
    sep_max = max(sep_min + 1, int(round(_GHOST_SEP_MAX_M / res)))
    out = np.zeros_like(occ, dtype=bool)
    h, w = occ.shape
    for r in range(h):
        cols = np.flatnonzero(occ[r])
        fr = free[r]
        for i, c in enumerate(cols):
            for c2 in cols[i + 1 :]:
                gap = int(c2 - c)
                if gap < sep_min:
                    continue
                if gap > sep_max:
                    break
                mid = fr[c + 1 : c2]
                if mid.size >= 2 and float(mid.mean()) > 0.6:
                    out[r, c] = True
                    out[r, c2] = True
                    break
    for c in range(w):
        rows = np.flatnonzero(occ[:, c])
        fr = free[:, c]
        for i, r in enumerate(rows):
            for r2 in rows[i + 1 :]:
                gap = int(r2 - r)
                if gap < sep_min:
                    continue
                if gap > sep_max:
                    break
                mid = fr[r + 1 : r2]
                if mid.size >= 2 and float(mid.mean()) > 0.6:
                    out[r, c] = True
                    out[r2, c] = True
                    break
    return out


def align_grid(
    cells: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    shape: Tuple[int, int],
    to_origin_x: float,
    to_origin_y: float,
    resolution: float,
) -> np.ndarray:
    """Re-index ``cells`` into another grid frame of the same resolution.

    Cells outside ``cells`` come back unknown (-1).
    """
    src = np.asarray(cells)
    out = np.full(shape, -1, dtype=src.dtype)
    dc = int(round((float(origin_x) - float(to_origin_x)) / float(resolution)))
    dr = int(round((float(origin_y) - float(to_origin_y)) / float(resolution)))
    r0, c0 = max(0, dr), max(0, dc)
    r1 = min(shape[0], dr + src.shape[0])
    c1 = min(shape[1], dc + src.shape[1])
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = src[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _thick_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied cells with dense 4-neighborhood (cheap smear proxy)."""
    up = np.zeros_like(occ, dtype=np.int16)
    down = np.zeros_like(occ, dtype=np.int16)
    left = np.zeros_like(occ, dtype=np.int16)
    right = np.zeros_like(occ, dtype=np.int16)
    up[1:, :] = occ[:-1, :]
    down[:-1, :] = occ[1:, :]
    left[:, 1:] = occ[:, :-1]
    right[:, :-1] = occ[:, 1:]
    return occ & ((up + down + left + right) >= 3)


def _tile_angle_deg(block: np.ndarray) -> Optional[float]:
    ys, xs = np.where(block)
    if ys.size < 40:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    pts -= pts.mean(axis=0)
    cov = (pts.T @ pts) / float(len(pts))
    _evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, -1]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap45(deg: float) -> float:
    return ((deg + 45.0) % 90.0) - 45.0


def _angle_std_deg(occ: np.ndarray, res: float) -> float:
    tile = max(8, int(round(_TILE_M / res)))
    h, w = occ.shape
    vals: List[float] = []
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            ang = _tile_angle_deg(occ[i : i + tile, j : j + tile])
            if ang is None:
                continue
            vals.append(_wrap45(ang))
    if len(vals) < 3:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64)))


def _ghost_hotspots(
    ghost: np.ndarray,
    occ: np.ndarray,
    *,
    res: float,
    origin_x: float,
    origin_y: float,
    row0: int,
    col0: int,
    max_hotspots: int,
) -> List[Tuple[float, float, float]]:
    tile = max(8, int(round(_TILE_M / res)))
    h, w = ghost.shape
    scored: List[Tuple[float, float, float]] = []
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            g = ghost[i : i + tile, j : j + tile]
            o = occ[i : i + tile, j : j + tile]
            n_occ = int(o.sum())
            if n_occ < 30:
                continue
            frac = float(g.sum()) / float(n_occ)
            if frac < 0.08:
                continue
            wx = origin_x + (col0 + j + tile * 0.5) * res
            wy = origin_y + (row0 + i + tile * 0.5) * res
            scored.append((frac, float(wx), float(wy)))
    scored.sort(reverse=True)
    return scored[: max(0, int(max_hotspots))]

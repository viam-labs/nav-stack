"""Inflated costmap from an occupancy grid (pure numpy, no ROS)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .types import OccupancyGrid

# Cost layers (uint8): 0 free … 253 inscribed, 254 lethal, 255 unknown.
LETHAL = 254
INSCRIBED = 253
UNKNOWN = 255
FREE = 0


def occupancy_from_bridge_map(map_data: dict) -> OccupancyGrid:
    """Build OccupancyGrid from BridgeNode.get_map() dict."""
    grid = np.asarray(map_data["grid"], dtype=np.int16)
    if grid.ndim != 2:
        raise ValueError(f"occupancy grid must be 2D, got shape {grid.shape}")
    return OccupancyGrid(
        grid=grid,
        resolution=float(map_data["resolution"]),
        origin_x=float(map_data["origin_x"]),
        origin_y=float(map_data["origin_y"]),
    )


def _disk_offsets(radius_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    if radius_cells <= 0:
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
    ys, xs = [], []
    r2 = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= r2:
                ys.append(dy)
                xs.append(dx)
    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)


def build_costmap(
    occ: OccupancyGrid,
    *,
    inflation_radius_m: float,
    robot_radius_m: float = 0.0,
    occupied_threshold: int = 50,
    cost_scaling_factor: float = 4.0,
) -> np.ndarray:
    """Return (H, W) uint8 costmap.

    Occupied / unknown cells become lethal. Free cells near obstacles get
    exponentially decaying cost within ``inflation_radius_m``. Cells inside
    ``robot_radius_m`` of a lethal cell are marked inscribed (treated as blocked
    by the planner).
    """
    h, w = occ.height, occ.width
    costs = np.full((h, w), FREE, dtype=np.uint8)

    raw = occ.grid
    lethal_mask = (raw >= occupied_threshold) | (raw < 0)
    costs[lethal_mask] = LETHAL
    costs[raw < 0] = UNKNOWN  # unknown stays unknown; planner treats as lethal

    res = max(float(occ.resolution), 1e-6)
    inflate_cells = max(0, int(math.ceil(inflation_radius_m / res)))
    inscribed_cells = max(0, int(math.ceil(robot_radius_m / res)))
    if inflate_cells == 0 and inscribed_cells == 0:
        return costs

    # Seed from occupied (not unknown-only) so unknown voids don't inflate.
    seed = (raw >= occupied_threshold).astype(np.uint8)
    if not seed.any():
        return costs

    # Approximate distance transform via layered dilation of seeds.
    # dist_cells[y,x] = min cell distance to an occupied cell (or large).
    dist = np.full((h, w), 1_000_000, dtype=np.int32)
    seed_y, seed_x = np.nonzero(seed)
    dist[seed_y, seed_x] = 0

    max_r = max(inflate_cells, inscribed_cells)
    for r in range(1, max_r + 1):
        dys, dxs = _disk_offsets(r)
        # Only paint the ring at exactly this radius for speed.
        ring = []
        r2_lo = (r - 1) * (r - 1)
        r2_hi = r * r
        for dy, dx in zip(dys.tolist(), dxs.tolist()):
            d2 = dy * dy + dx * dx
            if r2_lo < d2 <= r2_hi:
                ring.append((dy, dx))
        for dy, dx in ring:
            yy = seed_y + dy
            xx = seed_x + dx
            valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            yy, xx = yy[valid], xx[valid]
            # Only update free/unknown cells that aren't already closer.
            closer = dist[yy, xx] > r
            yy, xx = yy[closer], xx[closer]
            dist[yy, xx] = r

    # Apply inflation costs on free cells.
    free = costs == FREE
    for y in range(h):
        row_free = free[y]
        if not row_free.any():
            continue
        d = dist[y]
        for x in np.flatnonzero(row_free):
            dc = int(d[x])
            if dc > inflate_cells:
                continue
            if dc <= inscribed_cells:
                costs[y, x] = INSCRIBED
                continue
            # Exponential decay like Nav2: cost = 253 * exp(-factor * dist_m)
            dist_m = dc * res
            c = int(round(INSCRIBED * math.exp(-cost_scaling_factor * dist_m)))
            costs[y, x] = max(1, min(INSCRIBED - 1, c))

    return costs


def is_traversable(cost: int, *, allow_unknown: bool = False) -> bool:
    if cost >= LETHAL:
        return False
    if cost == UNKNOWN:
        return allow_unknown
    if cost >= INSCRIBED:
        return False
    return True


def costs_to_occupancy_viz(costs: np.ndarray) -> np.ndarray:
    """Convert layered uint8 costs to OccupancyGrid-style int16 for nav-camera.

    Nav2 / nav_view colouring expects: -1 unknown, 0 free, 1..98 inflation,
    99 inscribed, 100 lethal.
    """
    c = np.asarray(costs)
    out = np.zeros(c.shape, dtype=np.int16)
    out[c == FREE] = 0
    out[c == UNKNOWN] = -1
    out[c == LETHAL] = 100
    out[c == INSCRIBED] = 99
    mid = (c > FREE) & (c < INSCRIBED)
    if mid.any():
        # Map 1..252 → 1..98.
        scaled = np.clip(
            np.rint(c[mid].astype(np.float32) * (98.0 / float(INSCRIBED - 1))),
            1,
            98,
        ).astype(np.int16)
        out[mid] = scaled
    return out


def costmap_viz_dict(
    occ: OccupancyGrid,
    costs: np.ndarray,
) -> dict:
    """Bridge-style map dict the nav-camera can render as an inflated costmap."""
    return {
        "grid": costs_to_occupancy_viz(costs),
        "resolution": float(occ.resolution),
        "origin_x": float(occ.origin_x),
        "origin_y": float(occ.origin_y),
    }


def nearest_free_cell(
    costs: np.ndarray,
    row: int,
    col: int,
    *,
    max_radius_cells: int = 40,
) -> Optional[Tuple[int, int]]:
    """Find a traversable cell near ``(row, col)`` (inclusive of itself)."""
    h, w = costs.shape
    if 0 <= row < h and 0 <= col < w and is_traversable(int(costs[row, col])):
        return row, col
    for r in range(1, max_radius_cells + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dy), abs(dx)) != r:
                    continue
                yy, xx = row + dy, col + dx
                if 0 <= yy < h and 0 <= xx < w and is_traversable(int(costs[yy, xx])):
                    return yy, xx
    return None

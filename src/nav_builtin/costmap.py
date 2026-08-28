"""Inflated costmap from an occupancy grid (pure numpy, no ROS)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..ros import conversions as conv
from .types import OccupancyGrid

# Cost layers (uint8): 0 free … 253 inscribed, 254 lethal, 255 unknown.
LETHAL = 254
INSCRIBED = 253
UNKNOWN = 255
FREE = 0


def scan_world_points(
    pose: conv.Pose2D,
    scan: conv.LaserScan2D,
    *,
    max_beams: int = 360,
) -> np.ndarray:
    """Map-frame XY hits from a lidar scan (finite ranges only)."""
    pts = scan.to_points()
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if max_beams > 0 and pts.shape[0] > max_beams:
        pick = np.linspace(0, pts.shape[0] - 1, max_beams, dtype=np.int32)
        pts = pts[pick]
    scan_pose = scan.capture_pose
    if scan_pose is not None:
        dtheta = abs(conv.normalize_angle(pose.theta - scan_pose.theta))
        dist = math.hypot(pose.x - scan_pose.x, pose.y - scan_pose.y)
        if dtheta > math.radians(8.0) or dist > 0.08:
            pts3 = np.column_stack([pts, np.zeros(len(pts))])
            pts = conv.transform_points_between_poses(pts3, scan_pose, pose)[:, :2]
    cth = math.cos(pose.theta)
    sth = math.sin(pose.theta)
    wx = pose.x + cth * pts[:, 0] - sth * pts[:, 1]
    wy = pose.y + sth * pts[:, 0] + cth * pts[:, 1]
    finite = np.isfinite(wx) & np.isfinite(wy)
    return np.column_stack([wx[finite], wy[finite]])


def mark_scan_on_occupancy(
    occ: OccupancyGrid,
    pose: conv.Pose2D,
    scan: conv.LaserScan2D,
) -> OccupancyGrid:
    """Copy ``occ`` with live scan hits marked occupied (for dynamic replanning)."""
    grid = occ.grid.copy()
    hits = scan_world_points(pose, scan)
    for wx, wy in hits:
        row, col = occ.world_to_cell(float(wx), float(wy))
        if occ.in_bounds(row, col):
            grid[row, col] = 100
    return OccupancyGrid(
        grid=grid,
        resolution=occ.resolution,
        origin_x=occ.origin_x,
        origin_y=occ.origin_y,
    )


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

    Occupied / unknown cells become lethal. Cells within
    ``max(inflation_radius_m, robot_radius_m)`` of a lethal cell are marked
    inscribed (non-traversable for global planning). ``robot_radius_m`` is
    still used for footprint collision checks in the local planner.
    """
    h, w = occ.height, occ.width
    costs = np.full((h, w), FREE, dtype=np.uint8)

    raw = occ.grid
    lethal_mask = (raw >= occupied_threshold) | (raw < 0)
    costs[lethal_mask] = LETHAL
    costs[raw < 0] = UNKNOWN  # unknown stays unknown; planner treats as lethal

    res = max(float(occ.resolution), 1e-6)
    # Full inflation radius is hard-blocked for planning (Lazy Theta* treats
    # soft costs as free and would shortcut through a partial halo).
    clearance_m = max(float(inflation_radius_m), float(robot_radius_m))
    inscribed_cells = max(0, int(math.ceil(clearance_m / res)))
    inflate_cells = max(
        inscribed_cells,
        max(0, int(math.ceil(float(inflation_radius_m) / res))),
    )
    if inscribed_cells == 0 and inflate_cells == 0:
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

    # Apply inflation costs on free cells (vectorized).
    free = costs == FREE
    within = free & (dist <= inflate_cells)
    if within.any():
        inscribed = within & (dist <= inscribed_cells)
        costs[inscribed] = INSCRIBED
        soft = within & ~inscribed
        if soft.any():
            dist_m = dist[soft].astype(np.float64) * res
            soft_costs = np.rint(
                INSCRIBED * np.exp(-cost_scaling_factor * dist_m)
            ).astype(np.int32)
            costs[soft] = np.clip(soft_costs, 1, INSCRIBED - 1).astype(np.uint8)

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


def local_view_viz_dict(view) -> dict:
    """Rolling local costmap for operator UIs (already in map frame)."""
    return {
        "grid": costs_to_occupancy_viz(view.costs),
        "resolution": float(view.occ.resolution),
        "origin_x": float(view.origin_x),
        "origin_y": float(view.origin_y),
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


def footprint_max_cost(
    costs: np.ndarray,
    occ: OccupancyGrid,
    x_m: float,
    y_m: float,
    *,
    robot_radius_m: float,
) -> int:
    """Maximum layered cost under a circular robot footprint."""
    res = occ.resolution
    cells = max(1, int(math.ceil(float(robot_radius_m) / res)))
    row, col = occ.world_to_cell(x_m, y_m)
    h, w = costs.shape
    r2 = cells * cells
    worst = FREE
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            rr, cc = row + dy, col + dx
            if not (0 <= rr < h and 0 <= cc < w):
                return LETHAL
            worst = max(worst, int(costs[rr, cc]))
    return worst


def footprint_traversable(
    costs: np.ndarray,
    occ: OccupancyGrid,
    x_m: float,
    y_m: float,
    *,
    robot_radius_m: float,
) -> bool:
    """True when the full robot disk at ``(x_m, y_m)`` is traversable."""
    return is_traversable(
        footprint_max_cost(
            costs, occ, x_m, y_m, robot_radius_m=robot_radius_m
        )
    )


def nearest_free_pose(
    costs: np.ndarray,
    occ: OccupancyGrid,
    x_m: float,
    y_m: float,
    *,
    robot_radius_m: float,
    max_radius_cells: int = 40,
) -> Optional[Tuple[float, float]]:
    """Find a nearby pose whose full footprint is traversable."""
    if footprint_traversable(
        costs, occ, x_m, y_m, robot_radius_m=robot_radius_m
    ):
        return x_m, y_m
    row, col = occ.world_to_cell(x_m, y_m)
    h, w = costs.shape
    for r in range(1, max_radius_cells + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dy), abs(dx)) != r:
                    continue
                yy, xx = row + dy, col + dx
                if not (0 <= yy < h and 0 <= xx < w):
                    continue
                wx, wy = occ.cell_to_world(yy, xx)
                if footprint_traversable(
                    costs, occ, wx, wy, robot_radius_m=robot_radius_m
                ):
                    return wx, wy
    return None

"""Rolling local costmap for builtin navigation (map frame, no ROS)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..ros import conversions as conv
from .costmap import FREE, INSCRIBED, LETHAL, build_costmap, is_traversable
from .types import OccupancyGrid


@dataclass
class LocalCostmapConfig:
    width_m: float = 4.0
    height_m: float = 4.0
    resolution: float = 0.05
    inflation_radius_m: float = 0.25
    robot_radius_m: float = 0.22
    cost_scaling_factor: float = 4.0
    # Include static lethal/inscribed cells from the global map in the window.
    use_global_static: bool = True
    # Subsample dense lidar beams when marking (keeps update cheap).
    max_scan_beams: int = 180


@dataclass
class LocalCostmapView:
    """Snapshot passed to the local planner / controller."""

    costs: np.ndarray
    occ: OccupancyGrid
    origin_x: float
    origin_y: float

    def world_to_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return self.occ.world_to_cell(x_m, y_m)

    def in_bounds(self, row: int, col: int) -> bool:
        return self.occ.in_bounds(row, col)

    def cost_at_world(self, x_m: float, y_m: float) -> int:
        row, col = self.world_to_cell(x_m, y_m)
        if not self.in_bounds(row, col):
            return LETHAL
        return int(self.costs[row, col])


class LocalCostmap:
    """Rolling window centered on the robot; marks live lidar hits each update."""

    def __init__(self, cfg: LocalCostmapConfig):
        self._cfg = cfg
        res = max(float(cfg.resolution), 1e-3)
        self._w = max(8, int(math.ceil(cfg.width_m / res)))
        self._h = max(8, int(math.ceil(cfg.height_m / res)))
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._occ = OccupancyGrid(
            grid=np.zeros((self._h, self._w), dtype=np.int16),
            resolution=res,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )
        self._raw = np.zeros((self._h, self._w), dtype=np.int16)

    def _recenter(self, pose: conv.Pose2D) -> None:
        res = self._occ.resolution
        half_w = 0.5 * self._w * res
        half_h = 0.5 * self._h * res
        self._origin_x = float(pose.x) - half_w
        self._origin_y = float(pose.y) - half_h
        self._occ = OccupancyGrid(
            grid=self._raw,
            resolution=res,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )

    def _project_global_costs(
        self,
        global_occ: OccupancyGrid,
        global_costs: np.ndarray,
    ) -> np.ndarray:
        """Copy already-inflated global costs into the local window (no re-inflate)."""
        costs = np.zeros((self._h, self._w), dtype=np.uint8)
        res = self._occ.resolution
        cols = np.arange(self._w, dtype=np.float64)
        rows = np.arange(self._h, dtype=np.float64)
        xs = self._origin_x + (cols + 0.5) * res
        ys = self._origin_y + (rows + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        gcols = np.floor((xx - global_occ.origin_x) / global_occ.resolution).astype(
            np.int32
        )
        grows = np.floor((yy - global_occ.origin_y) / global_occ.resolution).astype(
            np.int32
        )
        gh, gw = global_costs.shape
        inside = (grows >= 0) & (grows < gh) & (gcols >= 0) & (gcols < gw)
        grows_c = np.clip(grows, 0, gh - 1)
        gcols_c = np.clip(gcols, 0, gw - 1)
        costs[inside] = global_costs[grows_c, gcols_c][inside]
        return costs

    def _mark_scan(self, pose: conv.Pose2D, scan: conv.LaserScan2D) -> None:
        pts = scan.to_points()
        if pts.size == 0:
            return
        max_beams = int(self._cfg.max_scan_beams)
        if max_beams > 0 and pts.shape[0] > max_beams:
            pick = np.linspace(0, pts.shape[0] - 1, max_beams, dtype=np.int32)
            pts = pts[pick]
        scan_pose = scan.capture_pose
        if scan_pose is not None:
            dtheta = abs(conv.normalize_angle(pose.theta - scan_pose.theta))
            dist = math.hypot(pose.x - scan_pose.x, pose.y - scan_pose.y)
            if dtheta > math.radians(8.0) or dist > 0.08:
                pts3 = np.column_stack([pts, np.zeros(len(pts))])
                pts = conv.transform_points_between_poses(pts3, scan_pose, pose)[
                    :, :2
                ]
        cth = math.cos(pose.theta)
        sth = math.sin(pose.theta)
        wx = pose.x + cth * pts[:, 0] - sth * pts[:, 1]
        wy = pose.y + sth * pts[:, 0] + cth * pts[:, 1]
        cols = np.floor((wx - self._origin_x) / self._occ.resolution).astype(np.int32)
        rows = np.floor((wy - self._origin_y) / self._occ.resolution).astype(np.int32)
        inside = (rows >= 0) & (rows < self._h) & (cols >= 0) & (cols < self._w)
        self._raw[rows[inside], cols[inside]] = 100

    def update(
        self,
        pose: conv.Pose2D,
        scan: Optional[conv.LaserScan2D],
        *,
        global_occ: Optional[OccupancyGrid] = None,
        global_costs: Optional[np.ndarray] = None,
    ) -> LocalCostmapView:
        self._recenter(pose)
        self._raw.fill(0)
        costs = np.zeros((self._h, self._w), dtype=np.uint8)
        if (
            global_occ is not None
            and global_costs is not None
            and self._cfg.use_global_static
        ):
            costs = self._project_global_costs(global_occ, global_costs)
        if scan is not None:
            self._mark_scan(pose, scan)
        scan_occ = OccupancyGrid(
            grid=self._raw,
            resolution=self._occ.resolution,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )
        scan_costs = build_costmap(
            scan_occ,
            inflation_radius_m=self._cfg.inflation_radius_m,
            robot_radius_m=self._cfg.robot_radius_m,
            cost_scaling_factor=self._cfg.cost_scaling_factor,
        )
        costs = np.maximum(costs, scan_costs)
        occ = OccupancyGrid(
            grid=self._raw,
            resolution=self._occ.resolution,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )
        return LocalCostmapView(
            costs=costs,
            occ=occ,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )


def max_cost_along_segment(
    view: LocalCostmapView,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    sample_step_m: float = 0.05,
) -> int:
    """Maximum local cost along a world-frame segment."""
    seg = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(math.ceil(seg / max(sample_step_m, 1e-3))))
    worst = FREE
    for k in range(n + 1):
        t = k / n
        c = view.cost_at_world(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        worst = max(worst, c)
    return worst


def footprint_collides(
    view: LocalCostmapView,
    x_m: float,
    y_m: float,
    *,
    robot_radius_m: float,
) -> bool:
    """True when a circular footprint at ``(x,y)`` hits inscribed/lethal cost."""
    res = view.occ.resolution
    cells = max(1, int(math.ceil(robot_radius_m / res)))
    row, col = view.world_to_cell(x_m, y_m)
    h, w = view.costs.shape
    r2 = cells * cells
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            rr, cc = row + dy, col + dx
            if not (0 <= rr < h and 0 <= cc < w):
                return True
            if not is_traversable(int(view.costs[rr, cc])):
                return True
    return False

"""Rolling local costmap for builtin navigation (map frame)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..geom import conversions as conv
from .costmap import FREE, INSCRIBED, LETHAL, build_costmap, is_traversable
from .types import OccupancyGrid


@dataclass
class LocalCostmapConfig:
    width_m: float = 4.0
    height_m: float = 4.0
    resolution: float = 0.05
    # Soft outer radius for live scan hits (absolute, from the obstacle). At or
    # below ``robot_radius_m`` there is no soft band: only the footprint itself
    # is lethal, so ``path_cost_ahead`` means "route inside a live return".
    inflation_radius_m: float = 0.0
    robot_radius_m: float = 0.22
    cost_scaling_factor: float = 4.0
    # Include static lethal/inscribed cells from the global map in the window.
    use_global_static: bool = True
    # Subsample dense lidar beams when marking (keeps update cheap).
    max_scan_beams: int = 180
    # Live scan inflation is tighter than the global soft ring so sparse /
    # jittery hits don't paint a wide soft field that flips DWA left/right.
    scan_inflation_radius_m: Optional[float] = None
    # Decay applied to previous live marks each update (0–100 occupancy).
    # Fresh hits are 100; at the default local rate (~5 Hz) decay 12 keeps
    # ankle/side obstacles for ~1.5 s after depth/lidar lose them — long
    # enough that a rotate-to-heading cannot swing through a just-seen blob.
    # 0 disables persistence (legacy clear-every-tick behaviour).
    scan_persist_decay: int = 12


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

    def _carry_persisted_marks(self, old_raw: np.ndarray, old_ox: float, old_oy: float) -> None:
        """Reproject previous live hits into the recentered window with decay."""
        decay = max(0, int(self._cfg.scan_persist_decay))
        if decay <= 0 or old_raw.size == 0:
            return
        ys, xs = np.nonzero(old_raw > 0)
        if ys.size == 0:
            return
        res = self._occ.resolution
        wx = old_ox + (xs.astype(np.float64) + 0.5) * res
        wy = old_oy + (ys.astype(np.float64) + 0.5) * res
        cols = np.floor((wx - self._origin_x) / res).astype(np.int32)
        rows = np.floor((wy - self._origin_y) / res).astype(np.int32)
        inside = (rows >= 0) & (rows < self._h) & (cols >= 0) & (cols < self._w)
        if not inside.any():
            return
        vals = np.maximum(0, old_raw[ys[inside], xs[inside]].astype(np.int32) - decay)
        keep = vals > 0
        if not keep.any():
            return
        rr = rows[inside][keep]
        cc = cols[inside][keep]
        vv = vals[keep].astype(np.int16)
        # max so a carried mark is not overwritten by a weaker neighbour seed
        existing = self._raw[rr, cc]
        self._raw[rr, cc] = np.maximum(existing, vv)

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
        old_raw = self._raw
        old_ox, old_oy = self._origin_x, self._origin_y
        self._recenter(pose)
        self._raw = np.zeros((self._h, self._w), dtype=np.int16)
        self._carry_persisted_marks(old_raw, old_ox, old_oy)
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
        # Soft outer radius for live hits: explicit override first, else the
        # configured inflation radius (clamped to the hard disk).
        scan_inflation = (
            float(self._cfg.scan_inflation_radius_m)
            if self._cfg.scan_inflation_radius_m is not None
            else max(
                float(self._cfg.robot_radius_m),
                float(self._cfg.inflation_radius_m),
            )
        )
        scan_costs = build_costmap(
            scan_occ,
            inflation_radius_m=scan_inflation,
            robot_radius_m=self._cfg.robot_radius_m,
            cost_scaling_factor=self._cfg.cost_scaling_factor,
            # The clearance-preference band is a planner-only routing bias; it
            # has no business in the layer used for collision / DWA thresholds.
            clearance_preference_m=0.0,
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


def overlay_local_costs_on_costmap(
    costs: np.ndarray,
    occ: OccupancyGrid,
    view: LocalCostmapView,
    *,
    min_cost: int = 200,
) -> np.ndarray:
    """Raise global planning costs where the live local window is blocked.

    Does not edit occupancy (avoids a second inflation pass on static walls).
    ``np.maximum`` keeps static cells unchanged and makes novel live blobs
    non-traversable so replans cannot peel through a second obstacle that was
    already visible locally.
    """
    local = np.asarray(view.costs)
    if local.size == 0:
        return costs
    ys, xs = np.nonzero(local >= int(min_cost))
    if ys.size == 0:
        return costs
    out = np.array(costs, copy=True, dtype=np.uint8)
    res = float(view.occ.resolution)
    wx = view.origin_x + (xs.astype(np.float64) + 0.5) * res
    wy = view.origin_y + (ys.astype(np.float64) + 0.5) * res
    for x, y, lc in zip(wx, wy, local[ys, xs]):
        row, col = occ.world_to_cell(float(x), float(y))
        if not occ.in_bounds(row, col):
            continue
        out[row, col] = max(int(out[row, col]), int(lc))
    return out


def max_cost_along_segment(
    view: LocalCostmapView,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    sample_step_m: float = 0.05,
    margin_m: float = 0.0,
) -> int:
    """Maximum local cost along a world-frame segment.

    ``margin_m > 0`` widens each sample to a disc of that radius so the check
    covers a band around the segment instead of a one-cell line.
    """
    seg = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(math.ceil(seg / max(sample_step_m, 1e-3))))
    worst = FREE
    for k in range(n + 1):
        t = k / n
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if margin_m > 0.0:
            c = footprint_max_cost(view, x, y, robot_radius_m=margin_m)
        else:
            c = view.cost_at_world(x, y)
        worst = max(worst, c)
        if worst >= LETHAL:
            break
    return worst


def footprint_max_cost(
    view: LocalCostmapView,
    x_m: float,
    y_m: float,
    *,
    robot_radius_m: float,
) -> int:
    """Maximum cost under a circular footprint (``LETHAL`` when out of map)."""
    res = view.occ.resolution
    cells = max(1, int(math.ceil(robot_radius_m / res)))
    row, col = view.world_to_cell(x_m, y_m)
    h, w = view.costs.shape
    r2 = cells * cells
    worst = FREE
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            rr, cc = row + dy, col + dx
            if not (0 <= rr < h and 0 <= cc < w):
                return LETHAL
            worst = max(worst, int(view.costs[rr, cc]))
    return worst


def reverse_backup_feasible(
    view: LocalCostmapView,
    x_m: float,
    y_m: float,
    theta_rad: float,
    *,
    robot_radius_m: float,
    distance_m: float,
    sample_step_m: float = 0.05,
) -> bool:
    """True when straight reverse stays traversable and ends in lower local cost."""
    start_cost = footprint_max_cost(
        view, x_m, y_m, robot_radius_m=robot_radius_m
    )
    if not reverse_path_clear(
        view,
        x_m,
        y_m,
        theta_rad,
        robot_radius_m=robot_radius_m,
        distance_m=distance_m,
        sample_step_m=sample_step_m,
    ):
        return False
    step = max(float(sample_step_m), 1e-3)
    steps = max(1, int(math.ceil(float(distance_m) / step)))
    cth = math.cos(theta_rad)
    sth = math.sin(theta_rad)
    x = x_m - cth * step * steps
    y = y_m - sth * step * steps
    end_cost = footprint_max_cost(view, x, y, robot_radius_m=robot_radius_m)
    return end_cost < start_cost


def reverse_path_clear(
    view: LocalCostmapView,
    x_m: float,
    y_m: float,
    theta_rad: float,
    *,
    robot_radius_m: float,
    distance_m: float,
    sample_step_m: float = 0.05,
    ignore_ahead: bool = False,
) -> bool:
    """True when a straight reverse of ``distance_m`` stays footprint-clear.

    Used to gate short unstick reverses: lidar rear can look open while the
    local costmap (persisted hits / inflation) already occupies the path.

    When ``ignore_ahead`` is set, inscribed cells still *in front of the start
    pose* (the blob we are reversing away from) do not fail the check — only
    costs in the rear half-plane relative to the start heading count.
    """
    step = max(float(sample_step_m), 1e-3)
    steps = max(1, int(math.ceil(float(distance_m) / step)))
    cth = math.cos(theta_rad)
    sth = math.sin(theta_rad)
    res = view.occ.resolution
    # Costs are footprint-inflated: a point/small pad along the path is enough.
    # Re-applying the full robot radius double-counts and refuses valid reverse
    # between two obstacles.
    check_r = max(res, min(0.08, 0.25 * float(robot_radius_m)))
    cells = max(1, int(math.ceil(check_r / res)))
    r2 = cells * cells
    h, w = view.costs.shape
    x, y = float(x_m), float(y_m)
    start_x, start_y = x, y
    for _ in range(steps):
        x -= cth * step
        y -= sth * step
        row, col = view.world_to_cell(x, y)
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                if dx * dx + dy * dy > r2:
                    continue
                rr, cc = row + dy, col + dx
                if not (0 <= rr < h and 0 <= cc < w):
                    return False
                if is_traversable(int(view.costs[rr, cc])):
                    continue
                if ignore_ahead:
                    wx = view.origin_x + (cc + 0.5) * res
                    wy = view.origin_y + (rr + 0.5) * res
                    bx = cth * (wx - start_x) + sth * (wy - start_y)
                    # Skip the blob under/ahead of the start pose — that is
                    # what we are reversing *away* from. Only obstacles further
                    # behind (outside the start footprint pad) can refuse.
                    start_pad_m = max(0.10, 2.0 * res)
                    if bx > 0.05 or math.hypot(wx - start_x, wy - start_y) <= start_pad_m:
                        continue
                return False
    return True


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


def spin_disc_blocked(
    view: LocalCostmapView,
    x_m: float,
    y_m: float,
    *,
    spin_radius_m: float,
    inscribed_radius_m: float,
) -> bool:
    """True when an in-place spin would sweep through inflation.

    Local costs are already footprint-inflated by ``inscribed_radius_m``, so
    checking a disc of the full circumscribed radius would double-count and
    falsely block squeezable gaps (live: ~1 m lidar gap looked impassable).
    Only the *extra* radius beyond the inscribed inflation is tested.
    """
    from .costmap import INSCRIBED

    inscribed = max(0.0, float(inscribed_radius_m))
    spin = max(inscribed, float(spin_radius_m))
    extra = spin - inscribed
    if extra <= 1e-6:
        return int(view.cost_at_world(x_m, y_m)) >= INSCRIBED
    return footprint_collides(view, x_m, y_m, robot_radius_m=extra)

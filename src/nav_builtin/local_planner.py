"""Lightweight DWA-style local planner on the rolling local costmap."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..nav.simple_motion import DriveCommand, apply_velocity_floor
from ..geom import conversions as conv
from .path_utils import closest_point_on_path
from .costmap import is_traversable
from .local_costmap import LocalCostmapView, footprint_collides, max_cost_along_segment
from .types import Path2D, Pose2D

# Set to a dict (tests / offline debugging) to capture per-tick scoring.
_DEBUG: Optional[dict] = None


@dataclass
class LocalPlannerConfig:
    # Soft inflation on a mapped wall is normal — only wake DWA when the path
    # ahead is nearly blocked (live obstacle or tight squeeze).
    activate_cost_threshold: int = 200
    deactivate_cost_threshold: int = 120
    path_clearance_lookahead_m: float = 1.2
    # Sample a disc of this radius around each path point (not just the
    # centerline). Live hits are inflated by exactly robot_radius, so a
    # centerline-only check had zero margin: an obstacle 1 cm outside the
    # footprint read as free, and any pose error became a collision.
    path_clearance_margin_m: float = 0.18
    path_weight: float = 2.0
    goal_weight: float = 1.0
    speed_weight: float = 0.5
    obstacle_weight: float = 3.0
    # Reverse samples (~0.15 m/s) so DWA can back out of nose-first blocks.
    max_vel_x_reverse_m: float = 0.15
    reverse_speed_weight: float = 0.85
    spin_penalty: float = 1.0  # prefer translate (incl. reverse) over rotate-only
    # Prefer continuity with the previous DWA command so vθ doesn't flip each tick.
    continuity_weight: float = 0.35
    # Penalise end-of-rollout heading error to the path ahead (rad). Breaks the
    # tie between +/- rotation so a blocked robot turns toward the route, not
    # whichever way the continuity term happened to lock in.
    heading_weight: float = 0.4
    heading_lookahead_m: float = 1.0
    # Rollout collision check radius. The local costmap is already inflated by
    # robot_radius (inscribed), so checking a full robot disc on top demanded
    # 2x clearance — with a 0.45 m robot nothing within 0.9 m was drivable,
    # every rollout "collided" and DWA fell back to a fixed +0.5*max spin.
    collision_margin_m: float = 0.05
    # Lane the detour field keeps off the blocked region (on top of the
    # inflated footprint). Too small hugs the blob and stalls; too large
    # swings room-scale. ~0.12 m is a middle peel for typical bins.
    detour_clearance_m: float = 0.12
    # Soft pull back toward the global path while following the detour field
    # (0 = field-only wide arcs; ~1 = path_weight scale).
    detour_path_bias: float = 0.55
    # When heading error to the detour/path exceeds this, refuse forward samples
    # (rotate first). force_local used to creep at 1 cm/s with vθ=0 while facing
    # 100° off — into clutter that was not in the nose cone.
    max_translate_heading_err_rad: float = math.radians(70.0)
    vx_samples: int = 5
    vtheta_samples: int = 5
    sim_time_s: float = 1.2
    sim_dt_s: float = 0.15
    enabled: bool = True


def _simulate(
    pose: Pose2D,
    vx: float,
    vtheta: float,
    *,
    sim_time_s: float,
    sim_dt_s: float,
) -> list[Pose2D]:
    dt = max(float(sim_dt_s), 1e-3)
    steps = max(1, int(round(sim_time_s / dt)))
    out = [pose]
    x, y, th = pose.x, pose.y, pose.theta
    for _ in range(steps):
        c = math.cos(th)
        s = math.sin(th)
        x += (c * vx) * dt
        y += (s * vx) * dt
        th = conv.normalize_angle(th + vtheta * dt)
        out.append(Pose2D(x, y, th))
    return out


def _rollout_point_collides(
    view: LocalCostmapView,
    start: Pose2D,
    p: Pose2D,
    *,
    margin_m: float,
) -> bool:
    if math.hypot(p.x - start.x, p.y - start.y) <= margin_m:
        return not is_traversable(view.cost_at_world(p.x, p.y))
    return footprint_collides(view, p.x, p.y, robot_radius_m=margin_m)


def _path_distance_m(path: Path2D, x: float, y: float) -> float:
    if path.empty:
        return 0.0
    px, py, _, _ = closest_point_on_path(Pose2D(x, y, 0.0), path)
    return math.hypot(x - px, y - py)


def path_point_ahead(path: Path2D, x: float, y: float, ahead_m: float) -> Tuple[float, float]:
    """Point on ``path`` ``ahead_m`` past the closest projection of (x, y)."""
    pts = path.points
    if not pts:
        return x, y
    if len(pts) == 1:
        return pts[0][0], pts[0][1]
    _, _, _, along = closest_point_on_path(Pose2D(x, y, 0.0), path)
    target = along + max(0.0, float(ahead_m))
    cum = 0.0
    for i in range(len(pts) - 1):
        seg = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if cum + seg >= target:
            t = 0.0 if seg < 1e-9 else (target - cum) / seg
            t = max(0.0, min(1.0, t))
            return (
                pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                pts[i][1] + t * (pts[i + 1][1] - pts[i][1]),
            )
        cum += seg
    return pts[-1][0], pts[-1][1]


@dataclass
class DetourField:
    """Traversable distance (m) to the route beyond a blockage, on a coarse grid.

    Euclidean goal distance says "straight at the bin"; this says "around it".
    Built by a vectorised wavefront over the local costmap (cells below
    inscribed cost), seeded with route samples past the blocked stretch.
    """

    dist_m: np.ndarray  # (h, w) float32, inf = unreachable / blocked
    origin_x: float
    origin_y: float
    resolution: float
    rejoin_xy: Tuple[float, float]

    def at(self, x: float, y: float) -> float:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        h, w = self.dist_m.shape
        if not (0 <= row < h and 0 <= col < w):
            return math.inf
        return float(self.dist_m[row, col])

    def descent_heading(self, x: float, y: float, *, ring_m: float = 0.28) -> Optional[float]:
        """Heading (rad) toward the lowest field value on a ring around (x, y).

        This is the local geodesic direction around the blockage — sideways
        when the robot is nose-to-bin, diagonal from further back. A modest
        ring (not 0.4+ m) avoids locking onto the far-out lane early.
        """
        h, w = self.dist_m.shape
        res = self.resolution
        r_cells = max(1, int(round(ring_m / res)))
        col0 = int(math.floor((x - self.origin_x) / res))
        row0 = int(math.floor((y - self.origin_y) / res))
        best_d = math.inf
        best_xy: Optional[Tuple[float, float]] = None
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                rr = dr * dr + dc * dc
                if rr > r_cells * r_cells or rr < (r_cells - 1) ** 2:
                    continue
                row, col = row0 + dr, col0 + dc
                if not (0 <= row < h and 0 <= col < w):
                    continue
                d = float(self.dist_m[row, col])
                if d < best_d:
                    best_d = d
                    best_xy = (
                        self.origin_x + (col + 0.5) * res,
                        self.origin_y + (row + 0.5) * res,
                    )
        if best_xy is None or not math.isfinite(best_d):
            return None
        return math.atan2(best_xy[1] - y, best_xy[0] - x)


def _wavefront(trav: np.ndarray, seeds: np.ndarray, *, max_iter: int) -> np.ndarray:
    """8-connected Jacobi wavefront: cell distance from any seed through ``trav``."""
    inf = np.float32(np.inf)
    d = np.full(trav.shape, inf, dtype=np.float32)
    d[seeds & trav] = 0.0
    if not np.isfinite(d).any():
        return d
    r2 = np.float32(math.sqrt(2.0))
    one = np.float32(1.0)
    for _ in range(max_iter):
        n = d.copy()
        np.minimum(n[1:, :], d[:-1, :] + one, out=n[1:, :])
        np.minimum(n[:-1, :], d[1:, :] + one, out=n[:-1, :])
        np.minimum(n[:, 1:], d[:, :-1] + one, out=n[:, 1:])
        np.minimum(n[:, :-1], d[:, 1:] + one, out=n[:, :-1])
        np.minimum(n[1:, 1:], d[:-1, :-1] + r2, out=n[1:, 1:])
        np.minimum(n[1:, :-1], d[:-1, 1:] + r2, out=n[1:, :-1])
        np.minimum(n[:-1, 1:], d[1:, :-1] + r2, out=n[:-1, 1:])
        np.minimum(n[:-1, :-1], d[1:, 1:] + r2, out=n[:-1, :-1])
        n[~trav] = inf
        if np.array_equal(n, d):
            break
        d = n
    return d


def build_detour_field(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    *,
    block_threshold: int,
    margin_m: float,
    clearance_m: float = 0.2,
    coarse_factor: int = 2,
    sample_step_m: float = 0.1,
    max_seed_span_m: float = 3.0,
) -> Optional[DetourField]:
    """Distance field to the first clear stretch of route past the blockage.

    Returns None when no blocked sample is found ahead or the route past it
    leaves the local window before clearing (caller falls back to Euclidean).
    """
    pts = path.points
    if len(pts) < 2:
        return None
    _, _, _, along0 = closest_point_on_path(pose, path)
    seg_lens = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg_lens.append(L)
        cum.append(cum[-1] + L)
    total = cum[-1]

    def _at(d: float) -> Tuple[float, float]:
        for i in range(len(pts) - 1):
            if cum[i + 1] + 1e-9 < d:
                continue
            seg = seg_lens[i]
            t = 0.0 if seg < 1e-9 else (d - cum[i]) / seg
            t = max(0.0, min(1.0, t))
            return (
                pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                pts[i][1] + t * (pts[i + 1][1] - pts[i][1]),
            )
        return pts[-1]

    step = max(0.05, float(sample_step_m))
    d = along0
    seen_block = False
    rejoin_along: Optional[float] = None
    while d <= total + 1e-9:
        x, y = _at(d)
        row, col = view.world_to_cell(x, y)
        if not view.in_bounds(row, col):
            break
        c = _disc_max_cost_in_bounds(view, x, y, margin_m)
        blocked = c >= block_threshold
        if blocked:
            seen_block = True
        elif seen_block:
            rejoin_along = d
            break
        d += step
    if rejoin_along is None and not seen_block:
        return None
    # Path never cleared inside the window (large footprint / block near the
    # edge): seed past the last blocked sample with free cells beside the route.
    if rejoin_along is None:
        rejoin_along = min(total, along0 + 1.2)

    f = max(1, int(coarse_factor))
    costs = view.costs
    h, w = costs.shape
    hh, ww = h // f, w // f
    if hh < 2 or ww < 2:
        return None
    coarse = costs[: hh * f, : ww * f].reshape(hh, f, ww, f).max(axis=(1, 3))
    res = float(view.occ.resolution) * f
    # Same traversability rule the rollout filter applies (cost below the
    # activation threshold), then pushed out by ``clearance_m`` so the
    # geodesic runs a comfortable lane around the blob. Hugging the <200
    # contour (~6 cm off inscribed) put the robot in a corner no 5-sample
    # rollout set could get out of.
    base_trav = coarse < block_threshold
    seed_cells: list[Tuple[int, int]] = []
    first_xy: Optional[Tuple[float, float]] = None
    d = rejoin_along
    end_along = min(total, rejoin_along + max_seed_span_m)
    while d <= end_along + 1e-9:
        x, y = _at(d)
        # Path yaw for lateral seeds when the centerline is still lethal.
        if d + 0.05 <= total:
            x2, y2 = _at(min(total, d + 0.2))
        else:
            x2, y2 = _at(max(0.0, d - 0.2))
            x2, y2 = x - (x2 - x), y - (y2 - y)
        path_yaw = math.atan2(y2 - y, x2 - x)
        nx, ny = -math.sin(path_yaw), math.cos(path_yaw)
        candidates = [(x, y)]
        for side in (0.35, 0.55, 0.75, -0.35, -0.55, -0.75):
            candidates.append((x + side * nx, y + side * ny))
        for sx, sy in candidates:
            col = int(math.floor((sx - view.origin_x) / res))
            row = int(math.floor((sy - view.origin_y) / res))
            if 0 <= row < hh and 0 <= col < ww and base_trav[row, col]:
                seed_cells.append((row, col))
                if first_xy is None:
                    first_xy = (sx, sy)
        d += step
    if first_xy is None:
        return None

    pcol = int(math.floor((pose.x - view.origin_x) / res))
    prow = int(math.floor((pose.y - view.origin_y) / res))
    ring = max(1, int(round(0.4 / res)))
    margin_cells = max(0, int(round(float(clearance_m) / res)))
    for m in (margin_cells, 0):
        trav = _erode(base_trav, m) if m > 0 else base_trav
        seeds = np.zeros((hh, ww), dtype=bool)
        for row, col in seed_cells:
            if trav[row, col]:
                seeds[row, col] = True
        if not seeds.any():
            continue
        dist = _wavefront(trav, seeds, max_iter=hh + ww)
        r0, r1 = max(0, prow - ring), min(hh, prow + ring + 1)
        c0, c1 = max(0, pcol - ring), min(ww, pcol + ring + 1)
        if r1 > r0 and c1 > c0 and np.isfinite(dist[r0:r1, c0:c1]).any():
            return DetourField(
                dist_m=dist * np.float32(res),
                origin_x=float(view.origin_x),
                origin_y=float(view.origin_y),
                resolution=res,
                rejoin_xy=first_xy,
            )
    return None


def _disc_max_cost_in_bounds(
    view: LocalCostmapView, x: float, y: float, radius_m: float
) -> int:
    """Max cost on a disc, ignoring cells outside the window.

    ``footprint_max_cost`` returns LETHAL when the disc touches the window
    edge, which made the route look blocked all the way out and the field
    unbuildable whenever the rejoin point sat near the edge.
    """
    res = view.occ.resolution
    cells = max(0, int(math.ceil(radius_m / res)))
    row, col = view.world_to_cell(x, y)
    h, w = view.costs.shape
    r2 = cells * cells
    worst = 0
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            rr, cc = row + dy, col + dx
            if 0 <= rr < h and 0 <= cc < w:
                worst = max(worst, int(view.costs[rr, cc]))
    return worst


def _erode(trav: np.ndarray, cells: int) -> np.ndarray:
    """Shrink the traversable mask by ``cells`` (8-connected)."""
    out = trav.copy()
    for _ in range(cells):
        n = out.copy()
        n[1:, :] &= out[:-1, :]
        n[:-1, :] &= out[1:, :]
        n[:, 1:] &= out[:, :-1]
        n[:, :-1] &= out[:, 1:]
        n[1:, 1:] &= out[:-1, :-1]
        n[1:, :-1] &= out[:-1, 1:]
        n[:-1, 1:] &= out[1:, :-1]
        n[:-1, :-1] &= out[1:, 1:]
        out = n
    return out


def path_block_distance_m(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    *,
    threshold: int,
    lookahead_m: float,
    margin_m: float = 0.0,
    sample_step_m: float = 0.08,
) -> Optional[float]:
    """Distance along the path from the robot to the first blocked sample.

    ``None`` when nothing at/above ``threshold`` is found within ``lookahead_m``.
    """
    if path.empty:
        return 0.0
    _, _, _, along0 = closest_point_on_path(pose, path)
    pts = path.points
    seg_lens: list[float] = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        length = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg_lens.append(length)
        cum.append(cum[-1] + length)
    total = cum[-1]
    target = min(total, along0 + max(lookahead_m, sample_step_m))
    step = max(float(sample_step_m), 0.05)
    d = along0
    while d <= target + 1e-9:
        # Locate segment.
        for i in range(len(pts) - 1):
            if cum[i + 1] + 1e-9 < d:
                continue
            seg = seg_lens[i]
            t = 0.0 if seg < 1e-9 else (d - cum[i]) / seg
            t = max(0.0, min(1.0, t))
            x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
            y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
            c = max_cost_along_segment(view, x, y, x, y, margin_m=margin_m)
            if c >= threshold:
                return max(0.0, d - along0)
            break
        d += step
    return None


def path_cost_ahead(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    *,
    lookahead_m: float,
    margin_m: float = 0.0,
) -> int:
    """Max local cost on the global path segment ahead of the robot.

    With ``margin_m > 0`` each sample checks a disc of that radius around the
    centerline, so obstacles just outside the (already inflated) footprint
    still register.
    """
    if path.empty:
        return 0
    _, _, _, along = closest_point_on_path(pose, path)
    pts = path.points
    seg_lens = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg_lens.append(L)
        cum.append(cum[-1] + L)
    total = cum[-1]
    target = min(total, along + max(lookahead_m, 0.05))
    worst = 0
    for i in range(len(pts) - 1):
        if cum[i + 1] + 1e-9 < along:
            continue
        t0 = 0.0 if cum[i] < along else (along - cum[i]) / max(seg_lens[i], 1e-9)
        t1 = 1.0 if cum[i + 1] > target else (target - cum[i]) / max(seg_lens[i], 1e-9)
        t0 = max(0.0, min(1.0, t0))
        t1 = max(0.0, min(1.0, t1))
        x0 = pts[i][0] + t0 * (pts[i + 1][0] - pts[i][0])
        y0 = pts[i][1] + t0 * (pts[i + 1][1] - pts[i][1])
        x1 = pts[i][0] + t1 * (pts[i + 1][0] - pts[i][0])
        y1 = pts[i][1] + t1 * (pts[i + 1][1] - pts[i][1])
        worst = max(
            worst,
            max_cost_along_segment(view, x0, y0, x1, y1, margin_m=margin_m),
        )
        if cum[i + 1] >= target:
            break
    return worst


def should_use_local_planner(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    cfg: LocalPlannerConfig,
    *,
    currently_active: bool = False,
) -> bool:
    if not cfg.enabled:
        return False
    threshold = (
        cfg.deactivate_cost_threshold
        if currently_active
        else cfg.activate_cost_threshold
    )
    if threshold <= 0:
        return True
    ahead = path_cost_ahead(
        pose,
        path,
        view,
        lookahead_m=cfg.path_clearance_lookahead_m,
        margin_m=cfg.path_clearance_margin_m,
    )
    return ahead >= threshold


def compute_local_command(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    *,
    cfg: LocalPlannerConfig,
    max_vel_x: float,
    max_vel_theta: float,
    robot_radius_m: float,
    min_cmd_vel_x: float = 0.0,
    min_cmd_vel_theta: float = 0.0,
    local_planner_active: bool = False,
    prev_cmd: Optional[DriveCommand] = None,
) -> Optional[DriveCommand]:
    """Sample (vx, vtheta) rollouts; return best safe command or None if not needed."""
    if not should_use_local_planner(
        pose, path, view, cfg, currently_active=local_planner_active
    ):
        return None

    goal = path.points[-1]
    gx, gy = goal[0], goal[1]
    ahead_cost = path_cost_ahead(
        pose,
        path,
        view,
        lookahead_m=cfg.path_clearance_lookahead_m,
        margin_m=cfg.path_clearance_margin_m,
    )
    path_blocked_ahead = ahead_cost >= cfg.activate_cost_threshold
    block_dist = (
        path_block_distance_m(
            pose,
            path,
            view,
            threshold=cfg.activate_cost_threshold,
            lookahead_m=cfg.path_clearance_lookahead_m,
            margin_m=cfg.path_clearance_margin_m,
        )
        if path_blocked_ahead
        else None
    )
    # Costs already encode the footprint; ``robot_radius_m`` is kept for API
    # compatibility and only bounds the margin.
    collision_r = max(
        float(view.occ.resolution),
        min(float(cfg.collision_margin_m), float(robot_radius_m)),
    )
    max_reverse = min(float(cfg.max_vel_x_reverse_m), float(max_vel_x))
    best: Optional[Tuple[float, float, float]] = None
    n_vx = max(3, int(cfg.vx_samples))
    n_vt = max(3, int(cfg.vtheta_samples))
    prev_vx = float(prev_cmd.vx) if prev_cmd is not None else None
    prev_vt = float(prev_cmd.vtheta) if prev_cmd is not None else None
    # Blocked route: score progress as traversable distance to the route past
    # the blockage instead of Euclidean distance to the goal. Euclidean pulls
    # straight at the bin (creep, then spin at the stop bubble); the field's
    # gradient goes around it.
    detour: Optional[DetourField] = None
    if path_blocked_ahead:
        detour = build_detour_field(
            pose,
            path,
            view,
            block_threshold=cfg.activate_cost_threshold,
            margin_m=cfg.path_clearance_margin_m,
            clearance_m=cfg.detour_clearance_m,
        )
        # The robot's own cell may be inside the soft ring (>= threshold) and
        # read inf; rollouts that step into finite cells still score properly.
    # Heading reference: the field's descent direction when detouring (sideways
    # when nose-to-bin, so rotate-in-place picks the right way), else the
    # route ahead (not the far goal, which can sit behind the obstacle).
    heading_ref: Optional[float] = None
    if detour is not None:
        heading_ref = detour.descent_heading(pose.x, pose.y)
        hx, hy = detour.rejoin_xy
    else:
        hx, hy = path_point_ahead(path, pose.x, pose.y, cfg.heading_lookahead_m)
    if heading_ref is None:
        if math.hypot(hx - pose.x, hy - pose.y) < 0.05:
            hx, hy = gx, gy
        heading_ref = math.atan2(hy - pose.y, hx - pose.x)
    if _DEBUG is not None:
        _DEBUG.clear()
        _DEBUG.update(
            heading_ref=heading_ref,
            detour=detour is not None,
            dd_pose=detour.at(pose.x, pose.y) if detour else None,
            blocked=path_blocked_ahead,
        )
    field_far_m = (
        float(max(detour.dist_m.shape)) * detour.resolution if detour else 0.0
    )
    path_dist_now = _path_distance_m(path, pose.x, pose.y)
    goal_dist_now = math.hypot(pose.x - gx, pose.y - gy)
    pose_heading_err = abs(conv.normalize_angle(heading_ref - pose.theta))
    translate_ok = pose_heading_err <= float(cfg.max_translate_heading_err_rad)

    for i in range(n_vx):
        if n_vx == 1:
            vx = 0.0
        else:
            vx = -max_reverse + (max_vel_x + max_reverse) * i / (n_vx - 1)
        # When the path ahead is blocked, still allow shallow reverse to unstick;
        # deep reverse is for backup recovery, not DWA.
        if path_blocked_ahead and vx < -0.08:
            continue
        # Facing way off the clear direction: rotate in place first. Tiny
        # forward creeps with vθ=0 were how the robot walked into side clutter
        # with nose_clear still true (obstacle not in the forward cone).
        if not translate_ok and vx > 0.02:
            continue
        for j in range(n_vt):
            vtheta = -max_vel_theta + (2.0 * max_vel_theta) * j / (n_vt - 1)
            rollout = _simulate(
                pose,
                vx,
                vtheta,
                sim_time_s=cfg.sim_time_s,
                sim_dt_s=cfg.sim_dt_s,
            )
            # Where the rollout *goes* needs the margin; where the robot already
            # *is* only needs its own cell drivable — otherwise standing one
            # cell from the inscribed ring rejects even rotate-in-place and we
            # fall through to the blind fallback spin.
            if any(
                _rollout_point_collides(
                    view, pose, p, margin_m=collision_r
                )
                for p in rollout[1:]
            ):
                continue
            # A lethal cell on the *route* must not veto every forward rollout —
            # that left DWA with rotate-only choices (spin, 57 failed replans)
            # while the flanks were open. Each rollout is judged on the cells it
            # actually crosses; the reactive stop bubble backs this up.
            if path_blocked_ahead and vx > 0 and any(
                view.cost_at_world(p.x, p.y) >= cfg.activate_cost_threshold
                for p in rollout[1:]
            ):
                continue
            end = rollout[-1]
            path_dist = _path_distance_m(path, end.x, end.y)
            # Straight-on toward a block beyond the short rollout horizon used
            # to score as "free + on path" at full speed (cmd_vx=0.5 with
            # path_cost_ahead=254). On the corridor, demand a peel or real
            # detour-field *decrease* — not "inf → finite" which any free cell
            # satisfies.
            if path_blocked_ahead and vx > 0.08:
                peeling = path_dist >= 0.28
                field_progress = False
                if detour is not None:
                    dd_end = detour.at(end.x, end.y)
                    dd_now = detour.at(pose.x, pose.y)
                    if (
                        math.isfinite(dd_now)
                        and math.isfinite(dd_end)
                        and dd_end < dd_now - 0.05
                    ):
                        field_progress = True
                if not peeling and not field_progress:
                    continue
            heading_err = abs(conv.normalize_angle(heading_ref - end.theta))
            # Max (not summed) cost along the rollout: summing made every
            # motion from inside an inflation ring look equally bad, so the
            # "do nothing" sample won.
            max_c = 0
            for p in rollout[1:]:
                c = view.cost_at_world(p.x, p.y)
                if c > max_c:
                    max_c = c
            obs_pen = max_c / 253.0
            if vx >= 0.0:
                speed_term = cfg.speed_weight * vx
                if path_blocked_ahead and vx > 1e-3:
                    speed_term *= 0.15
            elif path_blocked_ahead:
                speed_term = cfg.speed_weight * cfg.reverse_speed_weight * abs(vx)
            else:
                speed_term = cfg.speed_weight * 0.35 * abs(vx)
            if detour is not None:
                # Field progress around the bin, plus a soft path bias so the
                # peel stays near the route instead of a room-scale arc.
                dd = detour.at(end.x, end.y)
                if not math.isfinite(dd):
                    dd = field_far_m
                path_dist = _path_distance_m(path, end.x, end.y)
                path_pen = (
                    cfg.path_weight * cfg.detour_path_bias * path_dist
                )
                goal_pen = (cfg.path_weight + cfg.goal_weight) * dd
            else:
                # Progress relative to where we are, so every sample is on the
                # same scale. The old 0.2x reverse discount compared "80% off
                # the goal distance" against full price for forward, and a
                # 1 cm reverse creep beat every forward arc whenever the route
                # was blocked.
                goal_dist = math.hypot(end.x - gx, end.y - gy)
                path_pen = cfg.path_weight * (path_dist - path_dist_now)
                goal_pen = cfg.goal_weight * (goal_dist - goal_dist_now)
            score = (
                -path_pen
                - goal_pen
                + speed_term
                - cfg.obstacle_weight * obs_pen
                - cfg.heading_weight * heading_err
            )
            if abs(vx) < 1e-3 and abs(vtheta) > 1e-3 and path_blocked_ahead:
                score -= cfg.spin_penalty
            if prev_vx is not None and prev_vt is not None and max_vel_x > 1e-6:
                # Stick to the last DWA choice so noisy costmaps don't chatter.
                # Only while translating: a rotate-in-place that turned out to
                # be the wrong way must stay cheap to reverse, or the spin
                # locks itself in.
                prev_moving = abs(prev_vx) >= 0.05
                cont_w = cfg.continuity_weight * (1.0 if prev_moving else 0.5)
                dvx = abs(vx - prev_vx) / max(max_vel_x, 1e-3)
                dvt = abs(vtheta - prev_vt) / max(max_vel_theta, 1e-3)
                score -= cont_w * (dvx + dvt)
                if (
                    prev_moving
                    and abs(prev_vt) > 0.05
                    and abs(vtheta) > 0.05
                    and (prev_vt > 0) != (vtheta > 0)
                ):
                    score -= cont_w * 1.5
            if _DEBUG is not None:
                _DEBUG.setdefault("candidates", []).append(
                    (round(vx, 3), round(vtheta, 3), round(score, 3),
                     round(goal_pen, 3), round(heading_err, 2), max_c)
                )
            if best is None or score > best[2]:
                best = (vx, vtheta, score)

    if best is None:
        # Every rollout collides: rotate in place toward the route ahead.
        turn = conv.normalize_angle(heading_ref - pose.theta)
        direction = 1.0 if turn >= 0.0 else -1.0
        return DriveCommand(0.0, 0.0, direction * max_vel_theta * 0.5, False)

    vx, vtheta, _ = best
    cmd = DriveCommand(vx, 0.0, vtheta, False)
    from ..nav.simple_motion import SimpleMotionConfig

    motion = SimpleMotionConfig(
        max_linear_mps=max_vel_x,
        max_angular_rad_s=max_vel_theta,
        min_linear_mps=min_cmd_vel_x,
        min_angular_rad_s=min_cmd_vel_theta,
    )
    return apply_velocity_floor(cmd, motion)

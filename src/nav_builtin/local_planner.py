"""Lightweight DWA-style local planner on the rolling local costmap."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..nav.simple_motion import DriveCommand, apply_velocity_floor
from ..ros import conversions as conv
from .path_utils import closest_point_on_path
from .local_costmap import LocalCostmapView, footprint_collides, max_cost_along_segment
from .types import Path2D, Pose2D


@dataclass
class LocalPlannerConfig:
    # Soft inflation on a mapped wall is normal — only wake DWA when the path
    # ahead is nearly blocked (live obstacle or tight squeeze).
    activate_cost_threshold: int = 200
    path_clearance_lookahead_m: float = 1.2
    path_weight: float = 2.0
    goal_weight: float = 1.0
    speed_weight: float = 0.5
    obstacle_weight: float = 3.0
    vx_samples: int = 4
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


def _path_distance_m(path: Path2D, x: float, y: float) -> float:
    if path.empty:
        return 0.0
    px, py, _, _ = closest_point_on_path(Pose2D(x, y, 0.0), path)
    return math.hypot(x - px, y - py)


def path_cost_ahead(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    *,
    lookahead_m: float,
) -> int:
    """Max local cost on the global path segment ahead of the robot."""
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
            max_cost_along_segment(view, x0, y0, x1, y1),
        )
        if cum[i + 1] >= target:
            break
    return worst


def should_use_local_planner(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    cfg: LocalPlannerConfig,
) -> bool:
    if not cfg.enabled:
        return False
    if cfg.activate_cost_threshold <= 0:
        return True
    ahead = path_cost_ahead(
        pose,
        path,
        view,
        lookahead_m=cfg.path_clearance_lookahead_m,
    )
    return ahead >= cfg.activate_cost_threshold


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
) -> Optional[DriveCommand]:
    """Sample (vx, vtheta) rollouts; return best safe command or None if not needed."""
    if not should_use_local_planner(pose, path, view, cfg):
        return None

    goal = path.points[-1]
    gx, gy = goal[0], goal[1]
    best: Optional[Tuple[float, float, float]] = None
    n_vx = max(2, int(cfg.vx_samples))
    n_vt = max(3, int(cfg.vtheta_samples))

    for i in range(n_vx):
        vx = max_vel_x * i / (n_vx - 1)
        for j in range(n_vt):
            vtheta = -max_vel_theta + (2.0 * max_vel_theta) * j / (n_vt - 1)
            rollout = _simulate(
                pose,
                vx,
                vtheta,
                sim_time_s=cfg.sim_time_s,
                sim_dt_s=cfg.sim_dt_s,
            )
            if any(
                footprint_collides(view, p.x, p.y, robot_radius_m=robot_radius_m)
                for p in rollout
            ):
                continue
            end = rollout[-1]
            path_dist = _path_distance_m(path, end.x, end.y)
            goal_dist = math.hypot(end.x - gx, end.y - gy)
            obs_pen = 0.0
            for p in rollout[1:]:
                c = view.cost_at_world(p.x, p.y)
                if c > 0:
                    obs_pen += c / 253.0
            score = (
                -cfg.path_weight * path_dist
                - cfg.goal_weight * goal_dist
                + cfg.speed_weight * vx
                - cfg.obstacle_weight * obs_pen
            )
            if best is None or score > best[2]:
                best = (vx, vtheta, score)

    if best is None:
        # Blocked: rotate toward clearer heading.
        return DriveCommand(0.0, 0.0, max_vel_theta * 0.5, False)

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

"""Path-following controller built on simple_motion P-control + obstacle cone."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..nav.simple_motion import (
    DriveCommand,
    ObstacleConfig,
    SimpleMotionConfig,
    apply_obstacle_avoidance,
    apply_velocity_floor,
    compute_drive_command,
    distance_m,
)
from ..ros import conversions as conv
from .types import Path2D, Pose2D


@dataclass
class FollowerConfig:
    lookahead_m: float = 0.6
    waypoint_tolerance_m: float = 0.15
    motion: SimpleMotionConfig = field(default_factory=SimpleMotionConfig)
    obstacle: Optional[ObstacleConfig] = None


def _path_length(path: Path2D) -> float:
    pts = path.points
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


def lookahead_pose(
    current: Pose2D,
    path: Path2D,
    *,
    lookahead_m: float,
    waypoint_tolerance_m: float,
) -> Tuple[Pose2D, int, bool]:
    """Return (target_pose, waypoint_index, is_final).

    Walks ``lookahead_m`` along the polyline starting from the closest vertex
    (or the next one if we are already within ``waypoint_tolerance_m`` of it).
    """
    pts = path.points
    if not pts:
        return current, 0, True
    if len(pts) == 1:
        return Pose2D(pts[0][0], pts[0][1], path.goal_theta), 0, True

    # Closest vertex.
    best_i = 0
    best_d = math.inf
    for i, (x, y) in enumerate(pts):
        d = math.hypot(x - current.x, y - current.y)
        if d < best_d:
            best_d = d
            best_i = i

    # Start walking from this vertex; if we're already on it, begin at the
    # outgoing segment without discarding remaining lookahead budget.
    i = best_i
    if best_d <= waypoint_tolerance_m and i < len(pts) - 1:
        i += 1

    # Include distance from robot to the start vertex as already-covered when
    # we did not skip past it.
    remaining = lookahead_m
    if i == best_i and best_d < lookahead_m:
        # Robot is between vertices conceptually; treat proximity as progress
        # only when still targeting best_i (not yet advanced).
        pass

    # Walk from current position toward pts[i], then along subsequent segments.
    # Seed the walk with the robot→first-target segment so lookahead is measured
    # from the robot, not from a skipped vertex.
    cursor_x, cursor_y = current.x, current.y
    target_i = i
    while target_i < len(pts) and remaining > 0:
        tx, ty = pts[target_i]
        seg = math.hypot(tx - cursor_x, ty - cursor_y)
        if seg <= 1e-9:
            target_i += 1
            continue
        if remaining <= seg:
            t = remaining / seg
            x = cursor_x + t * (tx - cursor_x)
            y = cursor_y + t * (ty - cursor_y)
            heading = math.atan2(ty - cursor_y, tx - cursor_x)
            is_final = target_i >= len(pts) - 1 and remaining >= seg - 1e-9
            return Pose2D(x, y, heading if not is_final else path.goal_theta), target_i, False
        remaining -= seg
        cursor_x, cursor_y = tx, ty
        target_i += 1

    gx, gy = pts[-1]
    return Pose2D(gx, gy, path.goal_theta), len(pts) - 1, True


def compute_path_command(
    current: Pose2D,
    path: Path2D,
    *,
    cfg: FollowerConfig,
    scan: Optional[conv.LaserScan2D] = None,
) -> Tuple[DriveCommand, dict]:
    """One control step along ``path``."""
    target, idx, is_final = lookahead_pose(
        current,
        path,
        lookahead_m=cfg.lookahead_m,
        waypoint_tolerance_m=cfg.waypoint_tolerance_m,
    )
    # Near the end, aim at the goal pose (xy + yaw).
    if is_final or distance_m(current, Pose2D(path.points[-1][0], path.points[-1][1], 0.0)) <= (
        cfg.motion.xy_tolerance_m * 2.0
    ):
        goal = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
        cmd = compute_drive_command(
            current, goal, cfg=cfg.motion, linear_mps=cfg.motion.max_linear_mps
        )
    else:
        cmd = compute_drive_command(
            current, target, cfg=cfg.motion, linear_mps=cfg.motion.max_linear_mps
        )

    obstacle_state = "clear"
    forward_clearance = math.inf
    if cfg.obstacle is not None and cfg.obstacle.enabled:
        cmd, obstacle_state, forward_clearance = apply_obstacle_avoidance(
            cmd, scan, cfg.obstacle, max_angular_rad_s=cfg.motion.max_angular_rad_s
        )
        if not cmd.done and (cmd.vx != 0.0 or cmd.vtheta != 0.0):
            cmd = apply_velocity_floor(cmd, cfg.motion)

    progress = {
        "waypoint_index": idx,
        "is_final": is_final,
        "distance_remaining_m": distance_m(
            current, Pose2D(path.points[-1][0], path.points[-1][1], 0.0)
        ),
        "path_length_m": _path_length(path),
        "obstacle": obstacle_state,
        "forward_clearance_m": None if math.isinf(forward_clearance) else forward_clearance,
        "cmd_vx_mps": cmd.vx,
        "cmd_vy_mps": cmd.vy,
        "cmd_vtheta_rad_s": cmd.vtheta,
    }
    return cmd, progress

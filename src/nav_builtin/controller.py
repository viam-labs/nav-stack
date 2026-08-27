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
    distance_m,
    heading_error_rad,
)
from ..ros import conversions as conv
from .types import Path2D, Pose2D


@dataclass
class FollowerConfig:
    lookahead_m: float = 1.0
    waypoint_tolerance_m: float = 0.15
    # Above this bearing error, stop translating and rotate in place. Below it,
    # keep moving while turning (needed for sparse Lazy Theta* paths).
    rotate_in_place_rad: float = math.radians(75.0)
    motion: SimpleMotionConfig = field(default_factory=SimpleMotionConfig)
    obstacle: Optional[ObstacleConfig] = None


def _path_length(path: Path2D) -> float:
    pts = path.points
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def closest_point_on_path(
    current: Pose2D,
    path: Path2D,
) -> Tuple[float, float, int, float]:
    """Return (x, y, segment_index, distance_along_path_m) for the closest point.

    ``segment_index`` is the index of the segment start vertex (``i`` of
    ``points[i] -> points[i+1]``).
    """
    pts = path.points
    if not pts:
        return current.x, current.y, 0, 0.0
    if len(pts) == 1:
        return pts[0][0], pts[0][1], 0, 0.0

    best_d2 = math.inf
    best_xy = pts[0]
    best_seg = 0
    best_along = 0.0
    along = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            t = 0.0
            px, py = x0, y0
            seg_len = 0.0
        else:
            t = ((current.x - x0) * dx + (current.y - y0) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            px = x0 + t * dx
            py = y0 + t * dy
            seg_len = math.sqrt(seg_len2)
        d2 = (current.x - px) ** 2 + (current.y - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_xy = (px, py)
            best_seg = i
            best_along = along + t * seg_len
        along += seg_len
    return best_xy[0], best_xy[1], best_seg, best_along


def lookahead_pose(
    current: Pose2D,
    path: Path2D,
    *,
    lookahead_m: float,
    waypoint_tolerance_m: float,
) -> Tuple[Pose2D, int, bool]:
    """Return (target_pose, waypoint_index, is_final).

    Projects the robot onto the polyline (closest point on any segment), then
    walks ``lookahead_m`` forward along the path. Vertex-only projection fails
    on sparse Lazy Theta* paths (2–4 waypoints spanning meters).
    """
    del waypoint_tolerance_m  # kept for call-site compatibility
    pts = path.points
    if not pts:
        return current, 0, True
    if len(pts) == 1:
        return Pose2D(pts[0][0], pts[0][1], path.goal_theta), 0, True

    # Build cumulative lengths for walking from the closest projection.
    seg_lens = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg_lens.append(L)
        cum.append(cum[-1] + L)
    total = cum[-1]
    if total < 1e-9:
        return Pose2D(pts[-1][0], pts[-1][1], path.goal_theta), len(pts) - 1, True

    _, _, seg_i, along = closest_point_on_path(current, path)
    target_along = min(total, along + max(lookahead_m, 0.05))

    # Walk to target_along.
    for i in range(len(pts) - 1):
        if cum[i + 1] + 1e-9 < target_along:
            continue
        seg = seg_lens[i]
        if seg < 1e-9:
            continue
        t = (target_along - cum[i]) / seg
        t = max(0.0, min(1.0, t))
        x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
        y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
        heading = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
        is_final = target_along >= total - 1e-6
        if is_final:
            return Pose2D(pts[-1][0], pts[-1][1], path.goal_theta), len(pts) - 1, True
        return Pose2D(x, y, heading), i, False

    return Pose2D(pts[-1][0], pts[-1][1], path.goal_theta), len(pts) - 1, True


def compute_follow_command(
    current: Pose2D,
    target: Pose2D,
    *,
    cfg: FollowerConfig,
    final_yaw: Optional[float] = None,
) -> DriveCommand:
    """Pure-pursuit-ish step toward ``target`` (map frame → body cmd_vel)."""
    motion = cfg.motion
    dist = distance_m(current, target)
    heading_to_target = math.atan2(target.y - current.y, target.x - current.x)
    bearing = heading_error_rad(current.theta, heading_to_target)

    # Final pose: hold XY then finish yaw.
    if final_yaw is not None and dist <= motion.xy_tolerance_m:
        yaw_err = heading_error_rad(current.theta, final_yaw)
        if abs(yaw_err) <= motion.yaw_tolerance_rad:
            return DriveCommand(0.0, 0.0, 0.0, True)
        return apply_velocity_floor(
            DriveCommand(0.0, 0.0, _clamp(yaw_err, motion.max_angular_rad_s), False),
            motion,
        )

    if dist <= motion.xy_tolerance_m * 0.5:
        return DriveCommand(0.0, 0.0, 0.0, True)

    max_linear = motion.max_linear_mps
    max_angular = motion.max_angular_rad_s

    # Large heading error: rotate in place first so skid-steers don't carve
    # circles. Threshold is intentionally wide so sparse any-angle paths still
    # translate while gently correcting.
    if abs(bearing) > cfg.rotate_in_place_rad:
        return apply_velocity_floor(
            DriveCommand(0.0, 0.0, _clamp(bearing * 1.5, max_angular), False),
            motion,
        )

    linear_cmd = _clamp(dist * 0.6, max_linear)
    if dist < motion.xy_tolerance_m * 3:
        linear_cmd = min(linear_cmd, max_linear * 0.4)
    # Scale linear with bearing so we don't plow sideways.
    bearing_scale = max(0.25, 1.0 - (abs(bearing) / cfg.rotate_in_place_rad) * 0.6)
    linear_cmd *= bearing_scale
    angular_cmd = _clamp(bearing * 1.8, max_angular)
    return apply_velocity_floor(
        DriveCommand(linear_cmd, 0.0, angular_cmd, False), motion
    )


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
    near_goal = distance_m(
        current, Pose2D(path.points[-1][0], path.points[-1][1], 0.0)
    ) <= (cfg.motion.xy_tolerance_m * 2.0)

    if is_final or near_goal:
        goal = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
        cmd = compute_follow_command(
            current, goal, cfg=cfg, final_yaw=path.goal_theta
        )
    else:
        cmd = compute_follow_command(current, target, cfg=cfg, final_yaw=None)

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
        "bearing_error_rad": heading_error_rad(
            current.theta, math.atan2(target.y - current.y, target.x - current.x)
        ),
        "cmd_vx_mps": cmd.vx,
        "cmd_vy_mps": cmd.vy,
        "cmd_vtheta_rad_s": cmd.vtheta,
    }
    return cmd, progress

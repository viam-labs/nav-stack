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
    cone_min_range,
    distance_m,
    heading_error_rad,
)
from ..ros import conversions as conv
from .local_costmap import LocalCostmapView
from .local_planner import LocalPlannerConfig, compute_local_command
from .path_utils import closest_point_on_path, signed_crosstrack_m
from .types import Path2D, Pose2D


@dataclass
class FollowerConfig:
    # Longer lookahead = gentler pure-pursuit arcs on skid-steer. The mugger
    # 0.25–0.7 m RPP band made mid-path corrections too sharp (small arcs → big).
    lookahead_m: float = 1.0
    min_lookahead_m: float = 0.7
    max_lookahead_m: float = 1.4
    approach_dist_m: float = 0.35
    waypoint_tolerance_m: float = 0.15
    # Above this bearing error, stop translating and rotate in place. Below it,
    # keep moving while turning (needed for sparse Lazy Theta* paths).
    # Kept tighter than 75° so we don't carve large translate+turn arcs.
    rotate_in_place_rad: float = math.radians(55.0)
    # Mid-path angular gain on path-tangent error (was 1.8 on point-bearing).
    heading_gain: float = 1.0
    # Cap |vθ| while translating so turn radius stays gentle.
    max_translate_yaw_rad_s: float = 0.55
    # Stanley crosstrack gain (1/m-ish via atan(k e / v)). Higher = snap back
    # to the polyline harder so pure-pursuit does not cut inside corners.
    crosstrack_gain: float = 1.25
    # Start scaling linear speed down once |crosstrack| exceeds this.
    crosstrack_slow_m: float = 0.12
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


def _near_goal_command(
    *,
    yaw_err: float,
    bearing: float,
    dist: float,
    motion: SimpleMotionConfig,
) -> DriveCommand:
    """Hold / crawl near the goal without bearing-RIP oscillation.

    Outside ``xy_tolerance`` the normal law tracks bearing-to-point; a 1–2 cm
    overshoot makes that bearing ≈ ±π and commands ±max_vel_theta, then the
    next tick (back inside the ball) flips to final-yaw with the opposite
    sign — classic goal-swing.

    Keep closing XY until a few centimetres from the point, then pure-spin for
    final yaw. Supervisor may still accept XY-only after a yaw align timeout
    once inside the acceptance ball. We do not stop translating at the outer
    ``xy_tolerance`` ball — that left ~0.1–0.25 m residuals while hunting heading.

    Translating cmds must survive ``ViamWorldIO`` base sanitizer: it zeros
    ``|vx| < 0.12`` when ``|vθ| > 0.25`` (and ``|vx| < 0.05`` always). Tiny
    reverse crawls with yaw hunt therefore become pure spin — end wiggle
    with no XY progress.

    Inside ``xy_tolerance``, never pure-spin to face the *point* (that flip-flops
    with final-yaw spin). Soft-crawl or, when already close enough, spin for
    goal θ only.
    """
    xy_tol = motion.xy_tolerance_m
    yaw_tol = motion.yaw_tolerance_rad
    # Nail the point before hunting goal θ (acceptance ball stays xy_tol).
    xy_settle_m = xy_settle_radius_m(xy_tol)
    # Once inside acceptance, prefer final yaw over point-facing RIP so we do
    # not oscillate: face-point → crawl → overshoot → final-yaw → drift → repeat.
    final_yaw_hand_off_m = max(0.12, min(xy_tol * 0.5, 0.15))
    yaw_cap = min(0.40, motion.max_angular_rad_s)
    # Keep |vθ| under the sanitizer's 0.25 cut when also translating.
    translate_yaw_cap = min(yaw_cap, 0.22)
    # Floor above sanitizer lin_eps (0.05) and the tiny+turn kill (0.12).
    crawl_floor = 0.12
    soft_floor = 0.05  # lin_eps; pair with small vθ so soft crawl survives

    def _spin_yaw() -> DriveCommand:
        vtheta = _clamp(yaw_err * 0.85, yaw_cap)
        if abs(yaw_err) > yaw_tol and abs(vtheta) < 0.10:
            vtheta = math.copysign(0.10, yaw_err)
        return DriveCommand(0.0, 0.0, vtheta, False)

    def _close_xy(*, soft: bool = False) -> DriveCommand:
        """Close distance to the goal point without pure-spin face-the-point RIP.

        Soft mode never stops to rotate in place: that was the end-of-path
        swing (spin to point bearing, crawl, then spin the other way for θ).
        """
        floor = soft_floor if soft else crawl_floor
        # Soft: scale speed with remaining gap so a 6 cm approach doesn't
        # blast through the settle radius at 0.12 m/s.
        if soft:
            max_crawl = min(0.10, max(floor, dist * 1.2))
        else:
            max_crawl = 0.18
        if abs(bearing) > math.radians(100.0):
            crawl = min(
                max_crawl, max(floor, motion.max_linear_mps * (0.20 if soft else 0.25))
            )
            rev_bearing = conv.normalize_angle(bearing + math.pi)
            # Soft: reverse with almost no yaw so sanitizer keeps vx.
            vth = (
                _clamp(rev_bearing * 0.6, 0.15)
                if soft
                else _clamp(rev_bearing * 1.2, translate_yaw_cap)
            )
            return apply_velocity_floor(
                DriveCommand(
                    -max(floor, min(crawl, dist * 0.8)),
                    0.0,
                    vth,
                    False,
                ),
                motion,
            )
        if abs(bearing) > math.radians(45.0) and not soft:
            # Outside acceptance: face the point first.
            return apply_velocity_floor(
                DriveCommand(0.0, 0.0, _clamp(bearing * 1.5, yaw_cap), False),
                motion,
            )
        crawl = min(
            max_crawl, max(floor, motion.max_linear_mps * (0.20 if soft else 0.30))
        )
        vx = max(floor, min(crawl, dist * (1.2 if soft else 0.8)))
        vth = _clamp(bearing * (1.0 if soft else 1.5), translate_yaw_cap)
        if soft:
            # Keep |vθ| low so soft |vx| survives the sanitizer — even with
            # large bearing (crawl while turning instead of RIP).
            vth = _clamp(bearing * 0.7, 0.15)
            if abs(bearing) > math.radians(60.0):
                # Prefer slow reverse/forward over a long in-place swing.
                vx = max(floor, min(vx, 0.08))
        elif abs(bearing) < math.radians(25.0):
            vth = _clamp(bearing * 0.8, 0.15)
        return apply_velocity_floor(
            DriveCommand(vx, 0.0, vth, False),
            motion,
        )

    # Tight on the point: only final yaw remains.
    if dist <= xy_settle_m:
        return _spin_yaw()

    # Inside acceptance ball: soft-close XY, or hand off to final yaw once
    # close enough that another face-point spin would start the end wiggle.
    if dist <= xy_tol:
        if abs(yaw_err) > yaw_tol and dist <= final_yaw_hand_off_m:
            return _spin_yaw()
        return _close_xy(soft=True)

    # Still outside XY tolerance: ignore final yaw and close on the point.
    return _close_xy(soft=False)


def xy_settle_radius_m(xy_tolerance_m: float) -> float:
    """Radius at which near-goal stops translating and spins for final yaw.

    Kept small so we park on the point when possible. Below ~3 cm, soft crawl
    (sanitizer lin_eps 0.05 m/s) mostly overshoots, so we spin instead.
    """
    del xy_tolerance_m  # acceptance ball is separate; settle is physical
    return 0.03


def _effective_lookahead(
    cfg: FollowerConfig,
    *,
    speed_mps: float,
    near_goal: bool = False,
) -> float:
    """Velocity-scaled lookahead — longer = gentler arcs on diff/skid bases."""
    lo = min(cfg.min_lookahead_m, cfg.max_lookahead_m)
    hi = max(cfg.min_lookahead_m, cfg.max_lookahead_m)
    if near_goal:
        return lo
    # Do not shrink lookahead when crawling — that pulls the pursuit target in
    # and caps linear speed in a slow/shimmy feedback loop.
    horizon = max(cfg.lookahead_m, abs(speed_mps) * 1.2)
    return max(lo, min(hi, horizon))


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
    crosstrack_m: Optional[float] = None,
    path_yaw: Optional[float] = None,
) -> DriveCommand:
    """Pure-pursuit-ish step toward ``target`` (map frame → body cmd_vel)."""
    motion = cfg.motion
    dist = distance_m(current, target)
    heading_to_target = math.atan2(target.y - current.y, target.x - current.x)
    bearing = heading_error_rad(current.theta, heading_to_target)
    xy_tol = motion.xy_tolerance_m

    if final_yaw is not None:
        yaw_err = heading_error_rad(current.theta, final_yaw)
        if dist <= xy_tol and abs(yaw_err) <= motion.yaw_tolerance_rad:
            return DriveCommand(0.0, 0.0, 0.0, True)
        # Always use the near-goal law once we are pursuing the goal pose
        # (final_yaw set). Gating on 2× xy_tol left a dead band (~0.5–0.8 m)
        # where is_final still targets the goal but rotate_in_place commanded
        # ±max_vel_theta — classic near-goal butt-wiggle with vx=0.
        return _near_goal_command(
            yaw_err=yaw_err,
            bearing=bearing,
            dist=dist,
            motion=motion,
        )

    # Intermediate pursuit target reached — not navigation complete.
    if final_yaw is None and dist <= xy_tol * 0.5:
        return DriveCommand(0.0, 0.0, 0.0, False)

    max_linear = motion.max_linear_mps
    max_angular = motion.max_angular_rad_s

    # Mid-path: path tangent + Stanley crosstrack (not chase-lookahead).
    # Pure point-pursuit cuts inside corners and rides under the planned
    # robot_radius clearance even when the green plan looked fine.
    yaw_ref = float(path_yaw) if path_yaw is not None else float(target.theta)
    heading_err = heading_error_rad(current.theta, yaw_ref)
    bearing_to_point = heading_error_rad(current.theta, heading_to_target)
    ct = float(crosstrack_m) if crosstrack_m is not None else 0.0
    # Positive crosstrack = robot left of path → turn right (negative vθ).
    speed_ref = max(0.20, max_linear * 0.5)
    stanley = math.atan(
        (float(cfg.crosstrack_gain) * ct) / speed_ref
    )
    steer = conv.normalize_angle(heading_err - stanley)
    along = math.cos(yaw_ref) * (target.x - current.x) + math.sin(yaw_ref) * (
        target.y - current.y
    )
    face = bearing_to_point if along < 0.05 else steer

    # Large heading / crosstrack error: rotate in place first.
    if abs(face) > cfg.rotate_in_place_rad or along < 0.05:
        return apply_velocity_floor(
            DriveCommand(0.0, 0.0, _clamp(face * 1.5, max_angular), False),
            motion,
        )

    linear_cmd = _clamp(dist * 0.75, max_linear)
    if abs(steer) < math.radians(25.0) and abs(ct) < cfg.crosstrack_slow_m:
        linear_cmd = max(max_linear * 0.55, min(max_linear, linear_cmd))
    bearing_scale = max(0.45, 1.0 - (abs(steer) / cfg.rotate_in_place_rad) * 0.45)
    linear_cmd *= bearing_scale
    # Slow when off the polyline so we stop cutting deeper into the corner.
    if abs(ct) > cfg.crosstrack_slow_m:
        slow_span = max(0.15, 0.45 - cfg.crosstrack_slow_m)
        ct_scale = max(
            0.25,
            1.0 - (abs(ct) - cfg.crosstrack_slow_m) / slow_span,
        )
        linear_cmd *= ct_scale
    yaw_cap = min(max_angular, float(cfg.max_translate_yaw_rad_s))
    angular_cmd = _clamp(steer * float(cfg.heading_gain), yaw_cap)
    return apply_velocity_floor(
        DriveCommand(linear_cmd, 0.0, angular_cmd, False), motion
    )


def compute_path_command(
    current: Pose2D,
    path: Path2D,
    *,
    cfg: FollowerConfig,
    scan: Optional[conv.LaserScan2D] = None,
    speed_mps: Optional[float] = None,
    local_view: Optional[LocalCostmapView] = None,
    local_planner: Optional[LocalPlannerConfig] = None,
    robot_radius_m: float = 0.22,
    min_cmd_vel_x: float = 0.0,
    min_cmd_vel_theta: float = 0.0,
    local_planner_active: bool = False,
    prev_local_cmd: Optional[DriveCommand] = None,
) -> Tuple[DriveCommand, dict]:
    """One control step along ``path``."""
    est_speed = cfg.motion.max_linear_mps * 0.5 if speed_mps is None else speed_mps
    goal_xy = Pose2D(path.points[-1][0], path.points[-1][1], 0.0)
    dist_goal = distance_m(current, goal_xy)
    # Keep local planner off and use final-yaw pursuit for the whole approach.
    near_goal = dist_goal <= max(
        cfg.motion.xy_tolerance_m * 2.0,
        cfg.approach_dist_m * 2.0,
        0.8,
    )
    lookahead = _effective_lookahead(
        cfg, speed_mps=est_speed, near_goal=near_goal
    )
    target, idx, is_final = lookahead_pose(
        current,
        path,
        lookahead_m=lookahead,
        waypoint_tolerance_m=cfg.waypoint_tolerance_m,
    )
    bearing = heading_error_rad(
        current.theta, math.atan2(target.y - current.y, target.x - current.x)
    )
    crosstrack, path_yaw_ref = signed_crosstrack_m(current, path)

    local_active = False
    if (
        local_view is not None
        and local_planner is not None
        and not near_goal
        and abs(bearing) <= cfg.rotate_in_place_rad
    ):
        local_cmd = compute_local_command(
            current,
            path,
            local_view,
            cfg=local_planner,
            max_vel_x=cfg.motion.max_linear_mps,
            max_vel_theta=cfg.motion.max_angular_rad_s,
            robot_radius_m=robot_radius_m,
            min_cmd_vel_x=min_cmd_vel_x,
            min_cmd_vel_theta=min_cmd_vel_theta,
            local_planner_active=local_planner_active,
            prev_cmd=prev_local_cmd if local_planner_active else None,
        )
        if local_cmd is not None:
            cmd = local_cmd
            cmd = apply_velocity_floor(cmd, cfg.motion)
            local_active = True

    if not local_active:
        if is_final or near_goal:
            goal = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
            cmd = compute_follow_command(
                current, goal, cfg=cfg, final_yaw=path.goal_theta
            )
            bearing = heading_error_rad(current.theta, path.goal_theta)
        else:
            cmd = compute_follow_command(
                current,
                target,
                cfg=cfg,
                final_yaw=None,
                crosstrack_m=crosstrack,
                path_yaw=path_yaw_ref,
            )

    obstacle_state = "clear"
    forward_clearance = math.inf
    # Inside the XY acceptance ball, reactive slow/stop fights the soft crawl
    # (scales vx under sanitizer lin_eps) and blocks the yaw handoff. Clearance
    # is still reported when we have a scan; we just don't reshape the cmd.
    apply_obstacle = (
        not local_active
        and cfg.obstacle is not None
        and cfg.obstacle.enabled
        and dist_goal > cfg.motion.xy_tolerance_m
    )
    if apply_obstacle:
        cmd, obstacle_state, forward_clearance = apply_obstacle_avoidance(
            cmd, scan, cfg.obstacle, max_angular_rad_s=cfg.motion.max_angular_rad_s
        )
        if not cmd.done and (cmd.vx != 0.0 or cmd.vtheta != 0.0):
            cmd = apply_velocity_floor(cmd, cfg.motion)
    elif local_active:
        obstacle_state = "local_planner"
    elif (
        cfg.obstacle is not None
        and cfg.obstacle.enabled
        and dist_goal <= cfg.motion.xy_tolerance_m
        and scan is not None
    ):
        half = cfg.obstacle.front_cone_half_rad
        forward_clearance = cone_min_range(scan, -half, half)
        obstacle_state = "clear"

    progress = {
        "waypoint_index": idx,
        "is_final": is_final,
        "local_planner": local_active,
        "distance_remaining_m": distance_m(
            current, Pose2D(path.points[-1][0], path.points[-1][1], 0.0)
        ),
        "path_length_m": _path_length(path),
        "obstacle": obstacle_state,
        "forward_clearance_m": None if math.isinf(forward_clearance) else forward_clearance,
        "bearing_error_rad": bearing,
        "crosstrack_m": crosstrack,
        "cmd_vx_mps": cmd.vx,
        "cmd_vy_mps": cmd.vy,
        "cmd_vtheta_rad_s": cmd.vtheta,
    }
    return cmd, progress

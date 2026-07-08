"""Closed-loop map-frame navigation via Viam base ``SetVelocity``.

Mirrors MiR manual-mode ``drive_to_pose`` but uses SLAM map poses (meters/radians)
instead of the MiR REST map. No Nav2 required.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from ..ros import conversions as conv


class SimpleMotionError(Exception):
    """Simple navigation failed (timeout, stall, no pose)."""


class SimpleMotionCanceled(SimpleMotionError):
    """Simple navigation was canceled."""


@dataclass
class SimpleMotionConfig:
    poll_interval_s: float = 0.1
    xy_tolerance_m: float = 0.075
    yaw_tolerance_rad: float = math.radians(5.0)
    default_linear_mps: float = 0.35
    max_linear_mps: float = 0.35
    max_angular_rad_s: float = 0.8
    stall_timeout_s: float = 4.0
    stall_progress_m: float = 0.025
    stall_progress_rad: float = math.radians(2.0)
    timeout_s: float = 120.0


@dataclass
class ObstacleConfig:
    """Reactive forward-obstacle avoidance for simple go_to_* motion.

    This is *not* a planner: it only slows, stops, or turns the robot away from
    returns in a forward cone. It cannot route around large obstacles — use Nav2
    (``navigate_to_*``) for that.
    """

    enabled: bool = True
    stop_distance_m: float = 0.4  # inside this: stop forward, turn to clearer side
    slow_distance_m: float = 1.0  # inside this: scale linear speed down
    front_cone_half_rad: float = math.radians(35.0)  # forward "will I hit it" cone
    side_cone_rad: float = math.radians(100.0)  # left/right span for turn decision
    # Ignore scans older than this. Generous by default: MiR rosbridge lidar
    # reads can lag, and a too-tight window makes get_base_scan return None so
    # avoidance silently no-ops (robot drives blind).
    max_age_s: float = 2.0
    # Seconds of continuous "no scan" tolerated before failing the move. When
    # avoidance is on but we cannot see, we must NOT drive forward blindly.
    no_scan_timeout_s: float = 3.0


def normalize_angle(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))


def cone_min_range(scan: conv.LaserScan2D, lo_rad: float, hi_rad: float) -> float:
    """Minimum finite in-range return whose bearing lies in ``[lo, hi]`` (radians).

    Bearings are measured in the scan frame (base_link), forward = 0. Returns
    ``inf`` when the cone has no valid returns (i.e. it is clear).
    """
    ranges = np.asarray(scan.ranges, dtype=float)
    n = len(ranges)
    if n == 0:
        return math.inf
    angles = scan.angle_min + np.arange(n) * scan.angle_increment
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    in_cone = (angles >= lo_rad) & (angles <= hi_rad)
    valid = in_cone & np.isfinite(ranges) & (ranges >= scan.range_min)
    if not valid.any():
        return math.inf
    return float(ranges[valid].min())


def apply_obstacle_avoidance(
    cmd: "DriveCommand",
    scan: Optional[conv.LaserScan2D],
    obs: ObstacleConfig,
    *,
    max_angular_rad_s: float,
) -> tuple["DriveCommand", str, float]:
    """Adjust a forward drive command for obstacles seen in ``scan``.

    Returns ``(command, state, forward_clearance_m)`` where state is one of
    ``clear`` / ``slow`` / ``avoid`` / ``no_scan``. Only forward motion
    (``vx > 0``) is affected; in-place rotation (final-heading, or an active
    avoid turn) passes through so the robot can still spin to safety.

    When avoidance is enabled but ``scan`` is None (no fresh data), forward
    motion is suppressed as a fail-safe — driving blind defeats the purpose of
    the feature and is how the robot ends up nosing into obstacles it "can't
    see". Rotation is preserved so the robot can still finish a final heading.
    """
    if not obs.enabled or cmd.done or cmd.vx <= 0.0:
        return cmd, "clear", math.inf

    if scan is None:
        return DriveCommand(0.0, 0.0, cmd.vtheta, False), "no_scan", math.inf

    half = obs.front_cone_half_rad
    forward = cone_min_range(scan, -half, half)
    if forward > obs.slow_distance_m:
        return cmd, "clear", forward

    if forward > obs.stop_distance_m:
        span = max(obs.slow_distance_m - obs.stop_distance_m, 1e-6)
        scale = (forward - obs.stop_distance_m) / span
        return DriveCommand(cmd.vx * scale, 0.0, cmd.vtheta, False), "slow", forward

    # Too close to keep going: stop forward motion and rotate toward whichever
    # side has more room. +vtheta (CCW) turns left (+y / positive bearings).
    left = cone_min_range(scan, 0.0, obs.side_cone_rad)
    right = cone_min_range(scan, -obs.side_cone_rad, 0.0)
    direction = 1.0 if left >= right else -1.0
    return DriveCommand(0.0, 0.0, direction * max_angular_rad_s, False), "avoid", forward


def heading_error_rad(current_rad: float, target_rad: float) -> float:
    return normalize_angle(target_rad - current_rad)


def distance_m(a: conv.Pose2D, b: conv.Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class DriveCommand:
    vx: float
    vy: float
    vtheta: float
    done: bool


def compute_drive_command(
    current: conv.Pose2D,
    goal: conv.Pose2D,
    *,
    cfg: SimpleMotionConfig,
    linear_mps: float,
) -> DriveCommand:
    """One control step toward ``goal`` in the map frame (ROS body-frame cmd)."""
    dist = distance_m(current, goal)
    heading_to_goal = math.atan2(goal.y - current.y, goal.x - current.x)
    bearing_error = heading_error_rad(current.theta, heading_to_goal)
    final_heading_error = heading_error_rad(current.theta, goal.theta)

    at_xy = dist <= cfg.xy_tolerance_m
    at_heading = abs(final_heading_error) <= cfg.yaw_tolerance_rad
    if at_xy and at_heading:
        return DriveCommand(0.0, 0.0, 0.0, True)

    max_linear = min(abs(linear_mps), cfg.max_linear_mps)
    max_angular = cfg.max_angular_rad_s

    if at_xy:
        angular_cmd = _clamp(final_heading_error, max_angular)
        return DriveCommand(0.0, 0.0, angular_cmd, False)

    linear_cmd = _clamp(dist * 0.5, max_linear)
    if dist < cfg.xy_tolerance_m * 3:
        linear_cmd = min(linear_cmd, max_linear * 0.35)
    angular_cmd = _clamp(bearing_error * 1.5, max_angular)
    if abs(bearing_error) > math.radians(45.0):
        linear_cmd = min(linear_cmd, max_linear * 0.4)
    return DriveCommand(linear_cmd, 0.0, angular_cmd, False)


def config_from_nav(
    *,
    max_vel_x: float,
    max_vel_theta: float,
    xy_tolerance_m: float = 0.075,
    yaw_tolerance_rad: float = math.radians(5.0),
    timeout_s: float = 120.0,
) -> SimpleMotionConfig:
    return SimpleMotionConfig(
        default_linear_mps=max_vel_x,
        max_linear_mps=max_vel_x,
        max_angular_rad_s=max_vel_theta,
        xy_tolerance_m=xy_tolerance_m,
        yaw_tolerance_rad=yaw_tolerance_rad,
        timeout_s=timeout_s,
    )


async def drive_to_pose(
    *,
    goal: conv.Pose2D,
    get_pose: Callable[[], Optional[conv.Pose2D]],
    set_velocity: Callable[[float, float, float], Awaitable[None]],
    stop: Callable[[], Awaitable[None]],
    cfg: SimpleMotionConfig,
    linear_mps: Optional[float] = None,
    cancel_event: Optional[asyncio.Event] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    get_scan: Optional[Callable[[], Optional[conv.LaserScan2D]]] = None,
    obstacle: Optional[ObstacleConfig] = None,
) -> None:
    """Drive to ``goal`` using a MiR-style bearing -> translate -> final-heading loop.

    When ``obstacle`` avoidance is enabled and ``get_scan`` is provided, each
    forward step is slowed, stopped, or redirected based on a live base_link
    scan (reactive only — no path planning).
    """
    speed = linear_mps if linear_mps is not None else cfg.default_linear_mps
    loop = asyncio.get_running_loop()
    deadline = loop.time() + cfg.timeout_s
    last_dist = 0.0
    last_yaw_progress = goal.theta
    last_progress_at = loop.time()
    no_scan_since: Optional[float] = None

    try:
        initial = await asyncio.to_thread(get_pose)
        if initial is None:
            raise SimpleMotionError("map pose unavailable")
        last_dist = distance_m(initial, goal)
        while loop.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise SimpleMotionCanceled("canceled")

            current = await asyncio.to_thread(get_pose)
            if current is None:
                raise SimpleMotionError("map pose unavailable")

            cmd = compute_drive_command(current, goal, cfg=cfg, linear_mps=speed)
            dist = distance_m(current, goal)
            bearing = heading_error_rad(
                current.theta, math.atan2(goal.y - current.y, goal.x - current.x)
            )

            if cmd.done:
                if on_progress is not None:
                    on_progress(
                        {
                            "distance_remaining_m": dist,
                            "bearing_error_rad": bearing,
                            "heading_error_rad": heading_error_rad(
                                current.theta, goal.theta
                            ),
                            "obstacle": "clear",
                        }
                    )
                return

            obstacle_state = "clear"
            forward_clearance = math.inf
            if obstacle is not None and obstacle.enabled and get_scan is not None:
                scan = await asyncio.to_thread(get_scan)
                cmd, obstacle_state, forward_clearance = apply_obstacle_avoidance(
                    cmd, scan, obstacle, max_angular_rad_s=cfg.max_angular_rad_s
                )
                # Fail closed: if avoidance is on but we have no fresh scan, we
                # suppressed forward motion above. Give up (rather than sit
                # blind) once we've been starved of scans past the tolerance.
                if obstacle_state == "no_scan":
                    if no_scan_since is None:
                        no_scan_since = loop.time()
                    elif loop.time() - no_scan_since >= obstacle.no_scan_timeout_s:
                        raise SimpleMotionError(
                            "obstacle avoidance enabled but no fresh lidar scan "
                            "available; refusing to drive blind (check the SLAM "
                            "lidar pipeline, or set simple_avoid_obstacles=false)"
                        )
                else:
                    no_scan_since = None

            if on_progress is not None:
                on_progress(
                    {
                        "distance_remaining_m": dist,
                        "bearing_error_rad": bearing,
                        "heading_error_rad": heading_error_rad(current.theta, goal.theta),
                        "obstacle": obstacle_state,
                        "forward_clearance_m": (
                            None
                            if math.isinf(forward_clearance)
                            else forward_clearance
                        ),
                    }
                )

            traveled = abs(last_dist - dist)
            turned = abs(heading_error_rad(current.theta, last_yaw_progress))
            if traveled >= cfg.stall_progress_m or turned >= cfg.stall_progress_rad:
                last_dist = dist
                last_yaw_progress = current.theta
                last_progress_at = loop.time()
            elif loop.time() - last_progress_at >= cfg.stall_timeout_s:
                raise SimpleMotionError(
                    "velocity motion stalled: robot did not move while receiving "
                    "SetVelocity (check MiR mode key / Resume)"
                )

            await set_velocity(cmd.vx, cmd.vy, cmd.vtheta)
            await asyncio.sleep(cfg.poll_interval_s)

        raise SimpleMotionError("timed out waiting for velocity navigation to complete")
    finally:
        await stop()

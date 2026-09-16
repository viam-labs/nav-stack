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

from ..geom import conversions as conv


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
    # Floor for nonzero commands. Proportional near-goal / turn slowdowns can
    # drop below skid-steer stiction (motors "thunk" but the cart does not move),
    # which then trips the stall detector. Zero means no floor.
    min_linear_mps: float = 0.0
    min_angular_rad_s: float = 0.0
    stall_timeout_s: float = 8.0
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
    # Body-width corridor ahead (|y| ≤ half width, 0 < x ≤ slow_distance). The
    # angular cone is narrower than a wide robot at short range (±35° at 0.6 m
    # is only ±0.35 m), so shoulder-height bins/chairs slid past it. ``None``
    # disables; builtin nav sets robot_radius + a small margin.
    footprint_half_width_m: Optional[float] = None
    side_cone_rad: float = math.radians(100.0)  # left/right span for turn decision
    # While spinning (vx≈0), freeze only for true nose collisions — NOT the full
    # stop_distance. Using stop_distance here froze rotate-to-heading whenever a
    # corridor wall swept through the front cone (~0.3–0.5 m), then soft loc
    # could resume translating while still misaligned toward an obstacle.
    spin_collision_m: float = 0.22
    # Ignore scans older than this. Generous by default: MiR rosbridge lidar
    # reads can lag, and a too-tight window makes get_base_scan return None so
    # avoidance silently no-ops (robot drives blind).
    max_age_s: float = 2.0
    # Seconds of continuous "no scan" tolerated before failing the move. When
    # avoidance is on but we cannot see, we must NOT drive forward blindly.
    no_scan_timeout_s: float = 3.0


def normalize_angle(rad: float) -> float:
    return conv.normalize_angle(rad)


def rear_clearance_m(
    scan: conv.LaserScan2D,
    *,
    half_cone_rad: float = math.radians(60.0),
) -> float:
    """Minimum range in the rear arc (scan frame, rear ≈ |bearing| > π − half)."""
    ranges = np.asarray(scan.ranges, dtype=float)
    n = len(ranges)
    if n == 0:
        return math.inf
    angles = scan.angle_min + np.arange(n) * scan.angle_increment
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    in_rear = np.abs(angles) >= (math.pi - half_cone_rad)
    valid = in_rear & np.isfinite(ranges) & (ranges >= scan.range_min)
    if not valid.any():
        return math.inf
    return float(ranges[valid].min())


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


def corridor_min_range(
    scan: conv.LaserScan2D,
    half_width_m: float,
    max_forward_m: float,
) -> float:
    """Nearest forward distance of any return inside the body-width corridor.

    Considers points with ``0 < x <= max_forward_m`` and ``|y| <= half_width_m``
    in the scan (base_link) frame and returns the smallest ``x``; ``inf`` when
    the corridor is clear. Complements ``cone_min_range``: an angular cone
    misses obstacles at the robot's shoulders when they are close.
    """
    pts = scan.to_points()
    if pts.size == 0:
        return math.inf
    x = pts[:, 0]
    y = pts[:, 1]
    inside = (x > 0.0) & (x <= max_forward_m) & (np.abs(y) <= half_width_m)
    inside &= np.isfinite(x) & np.isfinite(y)
    if not inside.any():
        return math.inf
    return float(x[inside].min())


def arc_clearance_m(
    scan: conv.LaserScan2D,
    *,
    curvature_1_m: float,
    half_width_m: float,
    max_forward_m: float,
) -> float:
    """Travel distance along the *commanded arc* before the body sweeps a return.

    A body-fixed forward cone measures the wrong thing while turning: rounding a
    corridor corner, the wall ahead is not on the path the robot will actually
    take, so cone-based slowing crawls with the route clear. Returns are tested
    against the swept band of the commanded arc (a circle of radius
    ``1/curvature``) and the result is the arc length to the nearest one;
    ``inf`` when the arc is clear within ``max_forward_m``.
    """
    half = max(0.0, float(half_width_m))
    reach = max(0.0, float(max_forward_m))
    kappa = float(curvature_1_m)
    if abs(kappa) < 1e-3:
        return corridor_min_range(scan, half, reach)
    pts = scan.to_points()
    if pts.size == 0:
        return math.inf
    x = np.asarray(pts[:, 0], dtype=float)
    y = np.asarray(pts[:, 1], dtype=float)
    radius = 1.0 / kappa  # signed: positive turns left
    r_abs = abs(radius)
    # Arc centre sits at (0, radius) in the body frame; the robot starts at the
    # origin. A return is swept only if it lands within the body-wide band.
    offset = np.abs(np.hypot(x, y - radius) - r_abs)
    swept = np.isfinite(offset) & (offset <= half)
    if not swept.any():
        return math.inf
    start = math.atan2(-radius, 0.0)
    angle = np.arctan2(y[swept] - radius, x[swept])
    # Angle travelled in the direction of motion (CCW turning left).
    delta = (angle - start) if radius > 0.0 else (start - angle)
    arc = r_abs * np.mod(delta, 2.0 * math.pi)
    ahead = arc[arc <= reach]
    if ahead.size == 0:
        return math.inf
    return float(ahead.min())


def forward_clearance_m(scan: conv.LaserScan2D, obs: ObstacleConfig) -> float:
    """Forward clearance = min(front cone, body-width corridor)."""
    half = float(obs.front_cone_half_rad)
    clearance = cone_min_range(scan, -half, half)
    if obs.footprint_half_width_m is not None and obs.footprint_half_width_m > 0.0:
        clearance = min(
            clearance,
            corridor_min_range(
                scan,
                float(obs.footprint_half_width_m),
                float(obs.slow_distance_m),
            ),
        )
    return clearance


def apply_obstacle_avoidance(
    cmd: "DriveCommand",
    scan: Optional[conv.LaserScan2D],
    obs: ObstacleConfig,
    *,
    max_angular_rad_s: float,
    prefer_bearing_rad: Optional[float] = None,
    prefer_min_clearance_m: Optional[float] = None,
) -> tuple["DriveCommand", str, float]:
    """Adjust a drive command for obstacles seen in ``scan``.

    Returns ``(command, state, forward_clearance_m)`` where state is one of
    ``clear`` / ``slow`` / ``avoid`` / ``hold`` / ``no_scan``.

    Forward motion (``vx > 0``): slow inside ``slow_distance``, and at
    ``stop_distance`` stop translating and turn (``avoid``). When
    ``prefer_bearing_rad`` is set (path/goal bearing, + = left), prefer that
    side only if its side-cone clearance is at least ``prefer_min_clearance_m``
    (default: ``stop_distance``); otherwise fall back to freer-flank.

    In-place rotation / reverse (``vx <= 0``): freeze (``hold``) only for a
    true nose collision (``spin_collision_m``) or when swinging the bumper into
    a near hit on the **turn-side flank** (outside the front cone). A wall at
    normal ``stop_distance`` straight ahead must **not** freeze rotate-to-heading
    — that is how corridor 90° turns work.

    When avoidance is enabled but ``scan`` is None (no fresh data), forward
    motion is suppressed as a fail-safe — driving blind defeats the purpose of
    the feature. Rotation is preserved when translating so a final heading can
    still finish; a pure spin with no scan is left alone (same as clear space).
    """
    if not obs.enabled or cmd.done:
        return cmd, "clear", math.inf

    if cmd.vx <= 0.0:
        if scan is None:
            return cmd, "clear", math.inf
        half = float(obs.front_cone_half_rad)
        forward = cone_min_range(scan, -half, half)
        nose = max(0.05, float(obs.spin_collision_m))
        stop = float(obs.stop_distance_m)
        if forward <= nose:
            # True collision bubble only (person / wall pressed against bumper).
            return DriveCommand(0.0, 0.0, 0.0, False), "hold", forward
        if abs(cmd.vtheta) > 1e-6:
            # Turn-side flank outside the front cone — don't use [0, side], which
            # re-includes straight ahead and freezes every corridor spin.
            side = float(obs.side_cone_rad)
            if cmd.vtheta > 0.0:
                flank = cone_min_range(scan, half, side)
            else:
                flank = cone_min_range(scan, -side, -half)
            if flank <= stop:
                clr = forward if math.isfinite(forward) else flank
                return DriveCommand(0.0, 0.0, 0.0, False), "hold", clr
        return cmd, "clear", forward

    if scan is None:
        return DriveCommand(0.0, 0.0, cmd.vtheta, False), "no_scan", math.inf

    forward = forward_clearance_m(scan, obs)
    if forward > obs.slow_distance_m:
        return cmd, "clear", forward

    if forward > obs.stop_distance_m:
        span = max(obs.slow_distance_m - obs.stop_distance_m, 1e-6)
        # Slow for what the commanded arc will actually reach. Measuring only
        # straight ahead crawled us to ~⅓ speed against the outside wall of
        # every corridor corner, with the planner reporting the route clear.
        # Relaxation only: never closer than the stop bubble below.
        measured = forward
        if (
            obs.footprint_half_width_m is not None
            and obs.footprint_half_width_m > 0.0
            and abs(cmd.vtheta) > 1e-6
        ):
            measured = max(
                forward,
                min(
                    arc_clearance_m(
                        scan,
                        curvature_1_m=cmd.vtheta / cmd.vx,
                        half_width_m=float(obs.footprint_half_width_m),
                        max_forward_m=float(obs.slow_distance_m),
                    ),
                    float(obs.slow_distance_m),
                ),
            )
        scale = min(1.0, max(0.0, (measured - obs.stop_distance_m) / span))
        if scale >= 1.0:
            return cmd, "slow", forward
        # Scale vθ with vx so the turn *radius* is preserved. Scaling vx alone
        # tightens the arc as the robot slows (κ = vθ/vx), which is exactly the
        # sharp swerve-near-walls behaviour a slow-down is supposed to prevent.
        return (
            DriveCommand(cmd.vx * scale, 0.0, cmd.vtheta * scale, False),
            "slow",
            forward,
        )

    # Too close to keep going: stop forward and rotate. Prefer the path/goal
    # side when that flank is actually open; otherwise freer left/right.
    left = cone_min_range(scan, 0.0, obs.side_cone_rad)
    right = cone_min_range(scan, -obs.side_cone_rad, 0.0)
    direction = 1.0 if left >= right else -1.0
    min_clear = float(
        prefer_min_clearance_m
        if prefer_min_clearance_m is not None
        else obs.stop_distance_m
    )
    min_clear = max(0.0, min_clear)
    prefer_eps = math.radians(12.0)
    if prefer_bearing_rad is not None and abs(prefer_bearing_rad) >= prefer_eps:
        prefer_left = prefer_bearing_rad > 0.0
        preferred_clear = left if prefer_left else right
        if preferred_clear >= min_clear:
            direction = 1.0 if prefer_left else -1.0
    return DriveCommand(0.0, 0.0, direction * max_angular_rad_s, False), "avoid", forward


def heading_error_rad(current_rad: float, target_rad: float) -> float:
    return normalize_angle(target_rad - current_rad)


def distance_m(a: conv.Pose2D, b: conv.Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _apply_min_speed(value: float, minimum: float, maximum: float) -> float:
    """Keep nonzero commands at least ``minimum`` (and at most ``maximum``)."""
    if value == 0.0 or minimum <= 0.0:
        return _clamp(value, maximum) if maximum > 0.0 else value
    floored = max(abs(value), minimum)
    return math.copysign(min(floored, maximum) if maximum > 0.0 else floored, value)


@dataclass(frozen=True)
class DriveCommand:
    vx: float
    vy: float
    vtheta: float
    done: bool


def apply_velocity_floor(cmd: DriveCommand, cfg: SimpleMotionConfig) -> DriveCommand:
    """Bump nonzero vx / vtheta up to the configured minimums (stiction floor).

    The angular floor only applies to pure rotation (``vx == 0``): that is the
    case where a skid-steer stalls below ``min_angular``. While translating,
    a small vθ is just a slightly-differential wheel speed, and flooring it
    turns every tiny heading correction into a ±min_angular zig-zag.
    """
    if cmd.done:
        return cmd
    ang_floor = cfg.min_angular_rad_s if cmd.vx == 0.0 else 0.0
    return DriveCommand(
        _apply_min_speed(cmd.vx, cfg.min_linear_mps, cfg.max_linear_mps),
        cmd.vy,
        _apply_min_speed(cmd.vtheta, ang_floor, cfg.max_angular_rad_s),
        False,
    )


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
        return apply_velocity_floor(
            DriveCommand(0.0, 0.0, angular_cmd, False), cfg
        )

    linear_cmd = _clamp(dist * 0.5, max_linear)
    if dist < cfg.xy_tolerance_m * 3:
        linear_cmd = min(linear_cmd, max_linear * 0.35)
    angular_cmd = _clamp(bearing_error * 1.5, max_angular)
    if abs(bearing_error) > math.radians(45.0):
        linear_cmd = min(linear_cmd, max_linear * 0.4)
    return apply_velocity_floor(
        DriveCommand(linear_cmd, 0.0, angular_cmd, False), cfg
    )


def config_from_nav(
    *,
    max_vel_x: float,
    max_vel_theta: float,
    xy_tolerance_m: float = 0.075,
    yaw_tolerance_rad: float = math.radians(5.0),
    timeout_s: float = 120.0,
    min_linear_mps: float = 0.0,
    min_angular_rad_s: float = 0.0,
) -> SimpleMotionConfig:
    return SimpleMotionConfig(
        default_linear_mps=max_vel_x,
        max_linear_mps=max_vel_x,
        max_angular_rad_s=max_vel_theta,
        min_linear_mps=min_linear_mps,
        min_angular_rad_s=min_angular_rad_s,
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
                    cmd,
                    scan,
                    obstacle,
                    max_angular_rad_s=cfg.max_angular_rad_s,
                    prefer_bearing_rad=bearing,
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

            # Re-apply stiction floor after obstacle scaling (which can crush
            # speed below what a skid-steer base can physically execute).
            if not cmd.done and (cmd.vx != 0.0 or cmd.vtheta != 0.0):
                cmd = apply_velocity_floor(cmd, cfg)

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
                        "cmd_vx_mps": cmd.vx,
                        "cmd_vy_mps": cmd.vy,
                        "cmd_vtheta_rad_s": cmd.vtheta,
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

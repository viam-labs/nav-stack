import asyncio
import math
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.nav.simple_motion import (
    DriveCommand,
    ObstacleConfig,
    SimpleMotionCanceled,
    SimpleMotionConfig,
    SimpleMotionError,
    apply_obstacle_avoidance,
    compute_drive_command,
    cone_min_range,
    drive_to_pose,
    heading_error_rad,
    normalize_angle,
)
from src.geom import conversions as conv


def _scan_with(range_by_angle: dict, num_bins: int = 360) -> conv.LaserScan2D:
    """Build a 360-deg base_link scan with given (angle_rad -> range_m) returns."""
    ranges = np.full(num_bins, np.inf)
    angle_min = -math.pi
    angle_increment = 2 * math.pi / num_bins
    for angle, rng in range_by_angle.items():
        b = int((angle - angle_min) / angle_increment) % num_bins
        ranges[b] = rng
    return conv.LaserScan2D(ranges, angle_min, angle_increment, range_min=0.05)


def test_normalize_angle_wraps():
    assert abs(normalize_angle(math.pi + 0.1) + math.pi - 0.1) < 1e-9


def test_heading_error_rad():
    assert abs(heading_error_rad(0.0, math.pi / 2) - math.pi / 2) < 1e-9
    assert abs(heading_error_rad(math.pi, -math.pi / 2) - math.pi / 2) < 1e-9


def test_compute_drive_command_done_at_goal():
    cfg = SimpleMotionConfig()
    current = conv.Pose2D(1.0, 2.0, 0.0)
    goal = conv.Pose2D(1.0, 2.0, 0.0)
    cmd = compute_drive_command(current, goal, cfg=cfg, linear_mps=0.3)
    assert cmd.done
    assert cmd.vx == 0.0 and cmd.vtheta == 0.0


def test_compute_drive_command_drives_toward_goal():
    cfg = SimpleMotionConfig(xy_tolerance_m=0.05, max_linear_mps=0.4, max_angular_rad_s=0.8)
    current = conv.Pose2D(0.0, 0.0, 0.0)
    goal = conv.Pose2D(2.0, 0.0, 0.0)
    cmd = compute_drive_command(current, goal, cfg=cfg, linear_mps=0.4)
    assert not cmd.done
    assert cmd.vx > 0.0
    assert cmd.vy == 0.0


def test_compute_drive_command_final_heading_only():
    cfg = SimpleMotionConfig(xy_tolerance_m=0.1, yaw_tolerance_rad=math.radians(5))
    current = conv.Pose2D(1.0, 1.0, 0.0)
    goal = conv.Pose2D(1.0, 1.0, math.pi / 2)
    cmd = compute_drive_command(current, goal, cfg=cfg, linear_mps=0.3)
    assert not cmd.done
    assert cmd.vx == 0.0
    assert cmd.vtheta > 0.0


def test_compute_drive_command_applies_stiction_floor():
    cfg = SimpleMotionConfig(
        xy_tolerance_m=0.05,
        max_linear_mps=0.75,
        max_angular_rad_s=1.2,
        min_linear_mps=0.2,
        min_angular_rad_s=0.4,
    )
    # Tiny remaining distance would otherwise yield dist*0.5 << min.
    current = conv.Pose2D(0.0, 0.0, 0.0)
    goal = conv.Pose2D(0.12, 0.0, 0.0)
    cmd = compute_drive_command(current, goal, cfg=cfg, linear_mps=0.75)
    assert not cmd.done
    assert cmd.vx == pytest.approx(0.2)


def test_compute_drive_command_min_floor_zero_disables():
    cfg = SimpleMotionConfig(
        xy_tolerance_m=0.05,
        max_linear_mps=0.75,
        min_linear_mps=0.0,
        min_angular_rad_s=0.0,
    )
    current = conv.Pose2D(0.0, 0.0, 0.0)
    goal = conv.Pose2D(0.12, 0.0, 0.0)
    cmd = compute_drive_command(current, goal, cfg=cfg, linear_mps=0.75)
    assert cmd.vx == pytest.approx(0.06)  # dist * 0.5, no floor


def test_drive_to_pose_reaches_goal():
    cfg = SimpleMotionConfig(
        poll_interval_s=0.01,
        xy_tolerance_m=0.05,
        yaw_tolerance_rad=math.radians(5),
        timeout_s=2.0,
    )
    poses = [
        conv.Pose2D(0.0, 0.0, 0.0),
        conv.Pose2D(0.5, 0.0, 0.0),
        conv.Pose2D(0.95, 0.0, 0.0),
        conv.Pose2D(1.0, 0.0, 0.0),
    ]
    goal = conv.Pose2D(1.0, 0.0, 0.0)

    def get_pose():
        return poses.pop(0) if poses else goal

    velocities: list[tuple[float, float, float]] = []

    async def set_velocity(vx, vy, vtheta):
        velocities.append((vx, vy, vtheta))

    stop = AsyncMock()

    async def _run():
        await drive_to_pose(
            goal=goal,
            get_pose=get_pose,
            set_velocity=set_velocity,
            stop=stop,
            cfg=cfg,
        )

    asyncio.run(_run())
    assert velocities
    stop.assert_awaited_once()


def test_drive_to_pose_cancel():
    cfg = SimpleMotionConfig(poll_interval_s=0.01, timeout_s=2.0)
    goal = conv.Pose2D(5.0, 0.0, 0.0)
    cancel = asyncio.Event()

    async def set_velocity(vx, vy, vtheta):
        cancel.set()

    stop = AsyncMock()

    async def _run():
        with pytest.raises(SimpleMotionCanceled):
            await drive_to_pose(
                goal=goal,
                get_pose=lambda: conv.Pose2D(0.0, 0.0, 0.0),
                set_velocity=set_velocity,
                stop=stop,
                cfg=cfg,
                cancel_event=cancel,
            )

    asyncio.run(_run())
    stop.assert_awaited_once()


def test_drive_to_pose_no_pose():
    cfg = SimpleMotionConfig(poll_interval_s=0.01, timeout_s=1.0)
    stop = AsyncMock()

    async def _run():
        with pytest.raises(SimpleMotionError, match="map pose unavailable"):
            await drive_to_pose(
                goal=conv.Pose2D(1.0, 0.0, 0.0),
                get_pose=lambda: None,
                set_velocity=AsyncMock(),
                stop=stop,
                cfg=cfg,
            )

    asyncio.run(_run())
    stop.assert_awaited_once()


def test_cone_min_range_forward():
    scan = _scan_with({0.0: 0.8, math.pi / 2: 0.2})
    half = math.radians(35)
    assert abs(cone_min_range(scan, -half, half) - 0.8) < 1e-6


def test_cone_min_range_empty_is_inf():
    scan = _scan_with({})
    assert math.isinf(cone_min_range(scan, -0.5, 0.5))


def test_avoidance_clear_when_far():
    cmd = DriveCommand(0.3, 0.0, 0.1, False)
    scan = _scan_with({0.0: 5.0})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "clear"
    assert out == cmd


def test_avoidance_slows_in_band():
    cmd = DriveCommand(0.3, 0.0, 0.1, False)
    scan = _scan_with({0.0: 0.7})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "slow"
    assert 0.0 < out.vx < cmd.vx
    # Curvature preserved: vθ scales with vx so the arc does not tighten.
    assert out.vtheta / out.vx == pytest.approx(cmd.vtheta / cmd.vx)


def test_arc_clearance_ignores_wall_the_turn_curves_away_from():
    """Corridor corner: wall ahead, tight turn — the arc misses it."""
    from src.nav.simple_motion import arc_clearance_m

    scan = _scan_with({0.0: 0.68})
    straight = arc_clearance_m(
        scan, curvature_1_m=0.0, half_width_m=0.28, max_forward_m=1.0
    )
    assert straight == pytest.approx(0.68, abs=0.02)
    # Turning left on a ~0.43 m radius (the robot's own commanded arc).
    turning = arc_clearance_m(
        scan, curvature_1_m=1.0 / 0.43, half_width_m=0.28, max_forward_m=1.0
    )
    assert math.isinf(turning)
    # A gentle turn still drives into the same wall.
    gentle = arc_clearance_m(
        scan, curvature_1_m=1.0 / 3.0, half_width_m=0.28, max_forward_m=1.0
    )
    assert gentle == pytest.approx(0.68, abs=0.05)


def test_avoidance_does_not_crawl_when_arc_clears_corner_wall():
    """The reported stall: 0.68 m to the corner wall throttled us to ~⅓ speed."""
    obs = ObstacleConfig(
        stop_distance_m=0.5, slow_distance_m=1.0, footprint_half_width_m=0.28
    )
    scan = _scan_with({0.0: 0.68})
    # Tight left turn (κ ≈ 2.3 /m) rounding the corner.
    cmd = DriveCommand(0.25, 0.0, 0.58, False)
    out, state, clr = apply_obstacle_avoidance(
        cmd, scan, obs, max_angular_rad_s=1.0
    )
    assert out.vx == pytest.approx(cmd.vx)
    assert clr == pytest.approx(0.68, abs=0.02)
    assert state == "slow"  # still reported as near something

    # Heading straight at the same wall must still slow down.
    ahead = DriveCommand(0.25, 0.0, 0.0, False)
    out2, state2, _ = apply_obstacle_avoidance(
        ahead, scan, obs, max_angular_rad_s=1.0
    )
    assert state2 == "slow"
    assert out2.vx < ahead.vx


def test_avoidance_slows_for_dead_end_but_not_for_corner():
    """Same 0.68 m reading: a corner the arc clears must not throttle; a wall
    spanning the front must slow exactly as before."""
    obs = ObstacleConfig(
        stop_distance_m=0.5, slow_distance_m=1.0, footprint_half_width_m=0.28
    )

    def _wall(span_deg: int) -> conv.LaserScan2D:
        """Flat wall 0.68 m ahead, spanning ±``span_deg``."""
        return _scan_with(
            {
                math.radians(deg): 0.68 / math.cos(math.radians(deg))
                for deg in range(-span_deg, span_deg + 1, 2)
            }
        )

    cmd = DriveCommand(0.25, 0.0, 0.58, False)  # tight left turn, κ ≈ 2.3 /m
    corner, _s1, _c1 = apply_obstacle_avoidance(
        cmd, _wall(12), obs, max_angular_rad_s=1.0
    )
    assert corner.vx == pytest.approx(cmd.vx)

    dead_end, _s2, _c2 = apply_obstacle_avoidance(
        cmd, _wall(60), obs, max_angular_rad_s=1.0
    )
    # Unchanged behaviour: scale = (0.68 - 0.5) / (1.0 - 0.5).
    assert dead_end.vx == pytest.approx(cmd.vx * 0.36, abs=0.01)
    assert dead_end.vtheta / dead_end.vx == pytest.approx(cmd.vtheta / cmd.vx)


def test_avoidance_stop_bubble_unchanged_by_arc_relaxation():
    """Inside stop_distance still stops and turns, however the arc curves."""
    obs = ObstacleConfig(
        stop_distance_m=0.5, slow_distance_m=1.0, footprint_half_width_m=0.28
    )
    scan = _scan_with({0.0: 0.35})
    cmd = DriveCommand(0.25, 0.0, 0.58, False)
    out, state, _ = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=1.0)
    assert state == "avoid"
    assert out.vx == 0.0


def test_velocity_floor_skips_angular_when_translating():
    from src.nav.simple_motion import apply_velocity_floor

    cfg = SimpleMotionConfig(
        max_linear_mps=0.6,
        max_angular_rad_s=1.0,
        min_linear_mps=0.1,
        min_angular_rad_s=0.15,
    )
    # Translating with a tiny correction: do not zig-zag at ±min_angular.
    moving = apply_velocity_floor(DriveCommand(0.3, 0.0, 0.03, False), cfg)
    assert moving.vtheta == pytest.approx(0.03)
    # Pure rotation: floor still applies (skid-steer stiction).
    spin = apply_velocity_floor(DriveCommand(0.0, 0.0, 0.03, False), cfg)
    assert spin.vtheta == pytest.approx(0.15)
    # Linear floor unchanged.
    crawl = apply_velocity_floor(DriveCommand(0.02, 0.0, 0.0, False), cfg)
    assert crawl.vx == pytest.approx(0.1)


def test_avoidance_stops_and_turns_to_clearer_side():
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    scan = _scan_with({0.0: 0.3, -math.radians(60): 0.3})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "avoid"
    assert out.vx == 0.0
    assert out.vtheta > 0.0


def test_avoidance_footprint_corridor_catches_shoulder_obstacle():
    """Wide robot: a bin at the shoulder sits outside the ±35° cone but inside
    the body's swept corridor. Cone-only says clear; corridor must stop."""
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    # Bearing 50°, range 0.5 → x≈0.32, y≈0.38: outside cone, inside a 0.48 m
    # half-width body.
    scan = _scan_with({math.radians(50): 0.5})
    cone_only = ObstacleConfig(stop_distance_m=0.5, slow_distance_m=1.0)
    out, state, _ = apply_obstacle_avoidance(cmd, scan, cone_only, max_angular_rad_s=0.8)
    assert state == "clear"

    wide = ObstacleConfig(
        stop_distance_m=0.5, slow_distance_m=1.0, footprint_half_width_m=0.48
    )
    out, state, clr = apply_obstacle_avoidance(cmd, scan, wide, max_angular_rad_s=0.8)
    assert state == "avoid"
    assert out.vx == 0.0
    assert clr == pytest.approx(0.5 * math.cos(math.radians(50)), abs=0.02)


def test_avoidance_footprint_corridor_ignores_points_beside_body():
    """Returns wider than the body (a wall you fit past) do not trip the corridor."""
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    scan = _scan_with({math.radians(70): 0.6})  # x≈0.21, y≈0.56
    wide = ObstacleConfig(
        stop_distance_m=0.5, slow_distance_m=1.0, footprint_half_width_m=0.48
    )
    out, state, _ = apply_obstacle_avoidance(cmd, scan, wide, max_angular_rad_s=0.8)
    assert state == "clear"
    assert out == cmd


def test_avoidance_prefers_path_side_when_that_flank_is_clear():
    """Object ahead; open space behind on the right, corridor on the left.

    Blind freer-flank would spin right (180° retreat). Path bearing left +
    left clearance above stop → turn left into the plan.
    """
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    scan = _scan_with(
        {
            0.0: 0.3,
            math.radians(50): 0.9,  # left corridor open
            -math.radians(50): 5.0,  # right freer (behind / open room)
        }
    )
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, _ = apply_obstacle_avoidance(
        cmd,
        scan,
        obs,
        max_angular_rad_s=0.8,
        prefer_bearing_rad=math.radians(30.0),
    )
    assert state == "avoid"
    assert out.vx == 0.0
    assert out.vtheta > 0.0  # left / path side


def test_avoidance_falls_back_when_path_side_is_unsafe():
    """Path wants left, but left flank is tighter than stop — freer-flank."""
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    scan = _scan_with(
        {
            0.0: 0.3,
            math.radians(50): 0.2,  # left blocked
            -math.radians(50): 2.0,  # right open
        }
    )
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, _ = apply_obstacle_avoidance(
        cmd,
        scan,
        obs,
        max_angular_rad_s=0.8,
        prefer_bearing_rad=math.radians(30.0),
    )
    assert state == "avoid"
    assert out.vtheta < 0.0  # freer right


def test_avoidance_holds_spin_when_front_occupied():
    """Rotate-to-heading must freeze for a true nose collision."""
    cmd = DriveCommand(0.0, 0.0, 0.5, False)
    scan = _scan_with({0.0: 0.1})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0, spin_collision_m=0.22)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "hold"
    assert out.vx == 0.0 and out.vtheta == 0.0
    assert clr == pytest.approx(0.1)


def test_avoidance_allows_corridor_spin_with_wall_at_stop_distance():
    """Wall at stop_distance ahead must not freeze rotate-to-heading."""
    cmd = DriveCommand(0.0, 0.0, 0.5, False)
    scan = _scan_with({0.0: 0.35})  # typical corridor wall during a 90° spin
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0, spin_collision_m=0.22)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "clear"
    assert out == cmd
    assert clr == pytest.approx(0.35)


def test_avoidance_allows_spin_in_open_space():
    cmd = DriveCommand(0.0, 0.0, 0.5, False)
    scan = _scan_with({0.0: 5.0})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "clear"
    assert out == cmd
    assert clr == pytest.approx(5.0)


def test_avoidance_holds_spin_into_near_flank():
    cmd = DriveCommand(0.0, 0.0, -0.5, False)  # CW → right flank outside front cone
    scan = _scan_with({0.0: 5.0, -math.radians(50): 0.2})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "hold"
    assert out.vtheta == 0.0


def test_avoidance_disabled_passthrough():
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    scan = _scan_with({0.0: 0.1})
    obs = ObstacleConfig(enabled=False)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert out == cmd


def test_avoidance_no_scan_suppresses_forward():
    cmd = DriveCommand(0.3, 0.0, 0.2, False)
    obs = ObstacleConfig(enabled=True)
    out, state, clr = apply_obstacle_avoidance(cmd, None, obs, max_angular_rad_s=0.8)
    assert state == "no_scan"
    assert out.vx == 0.0
    assert out.vtheta == cmd.vtheta  # rotation preserved


def test_avoidance_no_scan_ignored_when_disabled():
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    obs = ObstacleConfig(enabled=False)
    out, state, clr = apply_obstacle_avoidance(cmd, None, obs, max_angular_rad_s=0.8)
    assert state == "clear"
    assert out == cmd


def test_drive_to_pose_fails_closed_without_scan():
    cfg = SimpleMotionConfig(poll_interval_s=0.01, xy_tolerance_m=0.05, timeout_s=5.0)
    obs = ObstacleConfig(enabled=True, no_scan_timeout_s=0.2)
    stop = AsyncMock()

    async def _run():
        with pytest.raises(SimpleMotionError, match="no fresh lidar scan"):
            await drive_to_pose(
                goal=conv.Pose2D(5.0, 0.0, 0.0),
                get_pose=lambda: conv.Pose2D(0.0, 0.0, 0.0),
                set_velocity=AsyncMock(),
                stop=stop,
                cfg=cfg,
                get_scan=lambda: None,
                obstacle=obs,
            )

    asyncio.run(_run())
    stop.assert_awaited_once()


def test_drive_to_pose_avoids_obstacle():
    cfg = SimpleMotionConfig(poll_interval_s=0.01, xy_tolerance_m=0.05, timeout_s=1.0)
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    blocking_scan = _scan_with({0.0: 0.2, math.radians(60): 5.0})
    states: list[str] = []

    async def set_velocity(vx, vy, vtheta):
        pass

    def on_progress(p):
        states.append(p.get("obstacle", "clear"))

    stop = AsyncMock()

    async def _run():
        with pytest.raises(SimpleMotionError):
            await drive_to_pose(
                goal=conv.Pose2D(5.0, 0.0, 0.0),
                get_pose=lambda: conv.Pose2D(0.0, 0.0, 0.0),
                set_velocity=set_velocity,
                stop=stop,
                cfg=cfg,
                on_progress=on_progress,
                get_scan=lambda: blocking_scan,
                obstacle=obs,
            )

    asyncio.run(_run())
    assert "avoid" in states
    stop.assert_awaited_once()

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
from src.ros import conversions as conv


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
    assert out.vtheta == cmd.vtheta


def test_avoidance_stops_and_turns_to_clearer_side():
    cmd = DriveCommand(0.3, 0.0, 0.0, False)
    scan = _scan_with({0.0: 0.3, -math.radians(60): 0.3})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "avoid"
    assert out.vx == 0.0
    assert out.vtheta > 0.0


def test_avoidance_skips_when_not_driving_forward():
    cmd = DriveCommand(0.0, 0.0, 0.5, False)
    scan = _scan_with({0.0: 0.1})
    obs = ObstacleConfig(stop_distance_m=0.4, slow_distance_m=1.0)
    out, state, clr = apply_obstacle_avoidance(cmd, scan, obs, max_angular_rad_s=0.8)
    assert state == "clear"
    assert out == cmd


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

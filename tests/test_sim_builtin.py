"""Tests for builtin simulation (SimWorld, sim-base kinematics, SLAM smoke)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
from viam.components.base import Vector3

from src.config import (
    SlamConfig,
    ros_twist_to_viam_set_velocity,
    viam_set_velocity_to_ros_twist,
)
from src.models.sim_base import SimBase
from src.nav.maps import MapStore
from src.ros import conversions as conv
from src.sim import (
    SimSensors,
    SimWorld,
    ensure_sim_world_from_slam_cfg,
    make_builtin_corridor,
    register_sim_world,
    unregister_sim_world,
)
from src.slam_builtin import BuiltinSlamEngine, BuiltinSlamHost


def test_viam_velocity_roundtrip_viam_convention():
    lx, ly, ang = ros_twist_to_viam_set_velocity(0.4, 0.0, 0.5, "viam")
    vx, vy, vtheta = viam_set_velocity_to_ros_twist(lx, ly, ang, "viam")
    assert vx == pytest.approx(0.4, abs=1e-9)
    assert vy == pytest.approx(0.0, abs=1e-9)
    assert vtheta == pytest.approx(0.5, abs=1e-9)


def test_sim_world_raycast_hits_walls():
    # Start above the corridor gap so +x faces the interior wall.
    world = SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 3.0, 0.0))
    scan = world.get_scan()
    assert np.isfinite(scan.ranges).any()
    forward_i = int((0.0 - scan.angle_min) / scan.angle_increment) % len(scan.ranges)
    assert scan.ranges[forward_i] < 5.0


def test_sim_world_integrates_velocity_and_reset():
    world = SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 1.0, 0.0))
    world.set_velocity_ros(0.5, 0.0, 0.0)
    time.sleep(0.2)
    pose = world.get_pose()
    assert pose.x > 1.05
    world.reset()
    pose2 = world.get_pose()
    assert pose2.x == pytest.approx(1.0, abs=1e-6)
    assert pose2.y == pytest.approx(1.0, abs=1e-6)


def test_sim_config_enabled_skips_lidar_deps():
    cfg = SlamConfig.from_dict(
        {
            "base": "sim-base",
            "sim": {"enabled": True, "seed_x": 1.2, "seed_y": 1.1},
            "maps_dir": "/tmp/nav-stack-sim-test",
        }
    )
    assert cfg.uses_sim()
    assert cfg.required_dependencies() == ["sim-base"]
    assert cfg.lidars[0].name == "sim-lidar"
    # Seed pose is ground truth; do not fight it with startup global_localize.
    assert cfg.global_localize_on_start is False


def test_sim_world_collision_blocks_wall():
    # Drive +x into the interior wall at col=width/2 from a free cell.
    world = SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 3.0, 0.0))
    wall_x = (make_builtin_corridor().width // 2) * 0.05
    world.set_velocity_ros(2.0, 0.0, 0.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        time.sleep(0.05)
        pose = world.get_pose()
        if pose.x >= wall_x - 0.15:
            break
    pose = world.get_pose()
    assert pose.x < wall_x - 0.01, f"expected stop before wall at {wall_x}, got x={pose.x}"


def test_sim_world_teleport_syncs_pose():
    world = SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 1.0, 0.0))
    world.set_velocity_ros(0.5, 0.0, 0.0)
    time.sleep(0.1)
    world.stop()
    world.teleport(conv.Pose2D(2.5, 2.0, 1.2))
    pose = world.get_pose()
    assert pose.x == pytest.approx(2.5, abs=1e-6)
    assert pose.y == pytest.approx(2.0, abs=1e-6)
    assert pose.theta == pytest.approx(1.2, abs=1e-6)


def test_sim_slam_pose_tracks_world_not_teleport(tmp_path: Path):
    """SimWorld is ground truth: SLAM pose follows the body; localize must not yank it."""
    unregister_sim_world("pose-sync")
    cfg = SlamConfig.from_dict(
        {
            "base": "sim-base",
            "sim": {
                "enabled": True,
                "world_name": "pose-sync",
                "seed_x": 1.0,
                "seed_y": 1.0,
                "seed_theta": 0.0,
            },
            "maps_dir": str(tmp_path),
            "mode": "mapping",
        }
    )
    world = ensure_sim_world_from_slam_cfg(cfg)
    sensors = SimSensors(world)
    store = MapStore(str(tmp_path))
    store.get_or_create_map("default", resolution=0.05)
    store.set_active_map("default")
    engine = BuiltinSlamEngine(cfg, sensors, store, rate_hz=20.0)  # type: ignore[arg-type]
    engine.set_pose(conv.Pose2D(1.0, 1.0, 0.0))
    host = BuiltinSlamHost(engine)
    host.start()
    try:
        world.set_velocity_ros(0.4, 0.0, 0.0)
        time.sleep(0.25)
        world.stop()
        body = world.get_pose()
        slam_pose = host.get_pose_in_map()
        assert slam_pose is not None
        assert slam_pose.x == pytest.approx(body.x, abs=0.05)
        assert slam_pose.y == pytest.approx(body.y, abs=0.05)
        # Localize-style set_pose must not move the simulated body.
        engine.set_pose(conv.Pose2D(5.0, 5.0, 1.0))
        time.sleep(0.15)
        assert world.get_pose().x == pytest.approx(body.x, abs=0.08)
        assert world.get_pose().y == pytest.approx(body.y, abs=0.08)
        # Engine pose is pulled back to the body on the next tick.
        slam_pose2 = host.get_pose_in_map()
        assert slam_pose2 is not None
        assert slam_pose2.x == pytest.approx(world.get_pose().x, abs=0.05)
    finally:
        host.shutdown()
        unregister_sim_world("pose-sync")


def test_sim_config_requires_builtin_backend():
    with pytest.raises(ValueError, match="slam_toolbox|slam_backend=builtin|pre-ros-removal"):
        SlamConfig.from_dict(
            {
                "base": "sim-base",
                "sim": {"enabled": True},
                "slam_backend": "slam_toolbox",
            }
        )


def test_sim_base_set_velocity_moves_world():
    unregister_sim_world("test-base")
    world = SimWorld(
        make_builtin_corridor(),
        seed_pose=conv.Pose2D(1.0, 1.0, 0.0),
        name="test-base",
    )
    register_sim_world("test-base", world)
    base = SimBase("sim-base")
    base._world = world
    base._convention = "viam"
    base._world_name = "test-base"

    async def _drive():
        await base.set_velocity(
            Vector3(x=0.0, y=400.0, z=0.0),
            Vector3(x=0.0, y=0.0, z=0.0),
        )
        await asyncio.sleep(0.15)
        pose = world.get_pose()
        assert pose.x > 1.02
        await base.stop()
        resp = await base.do_command({"command": "reset"})
        assert resp["ok"] is True
        assert world.get_pose().x == pytest.approx(1.0, abs=1e-6)

    asyncio.run(_drive())
    unregister_sim_world("test-base")


def test_sim_slam_maps_while_driving(tmp_path: Path):
    unregister_sim_world("slam-map")
    cfg = SlamConfig.from_dict(
        {
            "base": "sim-base",
            "sim": {
                "enabled": True,
                "world_name": "slam-map",
                "seed_x": 1.0,
                "seed_y": 1.0,
                "seed_theta": 0.0,
            },
            "maps_dir": str(tmp_path),
            "mode": "mapping",
        }
    )
    world = ensure_sim_world_from_slam_cfg(cfg)
    sensors = SimSensors(world)
    store = MapStore(str(tmp_path))
    store.get_or_create_map("default", resolution=0.05)
    store.set_active_map("default")
    engine = BuiltinSlamEngine(cfg, sensors, store, rate_hz=20.0)  # type: ignore[arg-type]
    engine.set_pose(conv.Pose2D(1.0, 1.0, 0.0))
    host = BuiltinSlamHost(engine)
    host.start()
    try:
        world.set_velocity_ros(0.35, 0.0, 0.0)
        deadline = time.monotonic() + 3.0
        occupied = 0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            m = host.get_map()
            if m is not None and m.get("grid") is not None:
                occupied = int(np.sum(np.asarray(m["grid"]) >= 50))
                if occupied > 20:
                    break
        assert occupied > 20, f"expected mapped walls, got occupied={occupied}"
        # Repeatability: reset and drive again should still produce scans.
        world.reset()
        engine.set_pose(conv.Pose2D(1.0, 1.0, 0.0))
        scan = sensors.get_scan(fresh=True)
        assert scan is not None
        assert np.isfinite(scan.ranges).sum() > 10
    finally:
        host.shutdown()
        unregister_sim_world("slam-map")


def test_repeat_path_resets_pose_ten_times():
    unregister_sim_world("repeat")
    world = SimWorld(
        make_builtin_corridor(),
        seed_pose=conv.Pose2D(1.0, 1.5, 0.0),
        name="repeat",
    )
    register_sim_world("repeat", world)
    ends = []
    for _ in range(10):
        world.reset()
        world.set_velocity_ros(0.5, 0.0, 0.0)
        time.sleep(0.2)
        world.stop()
        ends.append(world.get_pose().x)
    unregister_sim_world("repeat")
    # All runs should advance similarly from the same seed.
    assert min(ends) > 1.05
    assert max(ends) - min(ends) < 0.15

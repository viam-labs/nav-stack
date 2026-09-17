"""Tests for Viam framesystem → mount / footprint resolution."""
from __future__ import annotations

import asyncio
import math

import pytest
from viam.proto.common import (
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
)
from viam.proto.robot import FrameSystemConfig

from src.config import NavConfig, SlamConfig
from src.viam_frames import (
    apply_framesystem_to_nav_cfg,
    apply_framesystem_to_slam_cfg,
    fetch_frame_system_config,
    footprint_from_base_geometry,
    pose_of_frame_in_destination,
)


def _tracer_like_frames():
    base = Transform(
        reference_frame="base",
        pose_in_observer_frame=PoseInFrame(
            reference_frame="world",
            pose=Pose(x=0, y=0, z=90, o_x=0, o_y=0, o_z=1, theta=0),
        ),
        physical_object=Geometry(
            box=RectangularPrism(dims_mm=Vector3(x=590, y=720, z=180))
        ),
    )
    lidar = Transform(
        reference_frame="rplidar",
        pose_in_observer_frame=PoseInFrame(
            reference_frame="base",
            pose=Pose(x=-50, y=240, z=150, o_x=0, o_y=0, o_z=1, theta=-90),
        ),
    )
    camera = Transform(
        reference_frame="camera",
        pose_in_observer_frame=PoseInFrame(
            reference_frame="base",
            pose=Pose(
                x=150,
                y=300,
                z=130,
                o_x=0.022545244531911285,
                o_y=0.9997458236716953,
                o_z=0.0,
                theta=-90.0,
            ),
        ),
    )
    return [
        FrameSystemConfig(frame=base),
        FrameSystemConfig(frame=lidar),
        FrameSystemConfig(frame=camera),
    ]


def test_pose_of_frame_in_base_converts_mm_and_ov():
    configs = _tracer_like_frames()
    m = pose_of_frame_in_destination(configs, "rplidar", "base")
    assert m is not None
    assert abs(m.x - (-0.05)) < 1e-9
    assert abs(m.y - 0.24) < 1e-9
    assert abs(m.z - 0.15) < 1e-9
    assert abs(m.theta - (-math.pi / 2)) < 1e-6


def test_footprint_from_box_uses_longer_side_as_length():
    configs = _tracer_like_frames()
    fp = footprint_from_base_geometry(configs, "base")
    assert fp is not None
    assert abs(fp.length_m - 0.72) < 1e-9
    assert abs(fp.width_m - 0.59) < 1e-9
    assert abs(fp.inscribed_radius_m - 0.295) < 1e-9


def test_apply_slam_fills_mount_when_omitted():
    configs = _tracer_like_frames()
    raw = {
        "base": "base",
        "movement_sensor": "odom",
        "lidars": [
            {"name": "rplidar", "scan_source": "point_cloud"},
            {
                "name": "camera",
                "scan_source": "point_cloud",
                "obstacles_only": True,
                "cloud_frame": "camera_optical",
            },
        ],
    }
    cfg = SlamConfig.from_dict(raw)
    assert cfg.lidars[0].x == 0.0
    cfg2, notes = apply_framesystem_to_slam_cfg(cfg, configs, raw_attrs=raw)
    assert any("rplidar" in n and "framesystem" in n for n in notes)
    rpl = next(l for l in cfg2.lidars if l.name == "rplidar")
    assert abs(rpl.x - (-0.05)) < 1e-9
    assert abs(rpl.y - 0.24) < 1e-9
    assert abs(rpl.theta - (-math.pi / 2)) < 1e-6
    cam = next(l for l in cfg2.lidars if l.name == "camera")
    assert abs(cam.x - 0.15) < 1e-9
    assert abs(cam.y - 0.30) < 1e-9
    # FS pose is the full component→base transform; optical remap must not stack.
    assert cam.cloud_frame == "sensor"
    assert any("cloud_frame sensor" in n for n in notes)


def test_apply_slam_keeps_explicit_mount():
    configs = _tracer_like_frames()
    raw = {
        "base": "base",
        "movement_sensor": "odom",
        "lidars": [
            {
                "name": "rplidar",
                "scan_source": "point_cloud",
                "mount": {"x": 0.23, "y": 0.12, "z": 0.29, "theta": 3.14159},
            }
        ],
    }
    cfg = SlamConfig.from_dict(raw)
    cfg2, notes = apply_framesystem_to_slam_cfg(cfg, configs, raw_attrs=raw)
    assert any("override" in n for n in notes)
    assert abs(cfg2.lidars[0].x - 0.23) < 1e-9
    assert abs(cfg2.lidars[0].theta - 3.14159) < 1e-6


def test_apply_nav_fills_footprint_when_omitted():
    configs = _tracer_like_frames()
    raw = {"slam_service": "slam", "base": "base"}
    cfg = NavConfig.from_dict(raw)
    assert cfg.footprint_length_m is None
    cfg2, notes = apply_framesystem_to_nav_cfg(cfg, configs, raw_attrs=raw)
    assert any("framesystem box" in n for n in notes)
    assert abs(cfg2.footprint_length_m - 0.72) < 1e-9
    assert abs(cfg2.footprint_width_m - 0.59) < 1e-9
    assert abs(cfg2.robot_radius - 0.295) < 1e-9


def test_apply_nav_keeps_explicit_footprint():
    configs = _tracer_like_frames()
    raw = {
        "slam_service": "slam",
        "base": "base",
        "footprint_length_m": 0.8,
        "footprint_width_m": 0.5,
        "robot_radius": 0.31,
    }
    cfg = NavConfig.from_dict(raw)
    cfg2, notes = apply_framesystem_to_nav_cfg(cfg, configs, raw_attrs=raw)
    assert any("from config" in n for n in notes)
    assert abs(cfg2.footprint_length_m - 0.8) < 1e-9
    assert abs(cfg2.footprint_width_m - 0.5) < 1e-9
    assert abs(cfg2.robot_radius - 0.31) < 1e-9


def test_apply_slam_mutates_lidar_in_place():
    """Live BuiltinSensors holds the same LidarConfig instances."""
    configs = _tracer_like_frames()
    raw = {
        "base": "base",
        "movement_sensor": "odom",
        "lidars": [{"name": "rplidar", "scan_source": "point_cloud"}],
    }
    cfg = SlamConfig.from_dict(raw)
    lidar = cfg.lidars[0]
    apply_framesystem_to_slam_cfg(cfg, configs, raw_attrs=raw)
    assert lidar is cfg.lidars[0]
    assert abs(lidar.x - (-0.05)) < 1e-9


@pytest.mark.asyncio
async def test_fetch_frame_system_config_times_out():
    class _SlowRobot:
        async def get_frame_system_config(self):
            await asyncio.sleep(10.0)
            return []

    with pytest.raises(asyncio.TimeoutError):
        await fetch_frame_system_config(_SlowRobot(), timeout_s=0.05)

"""Tests for Viam framesystem → mount / footprint resolution."""
from __future__ import annotations

import asyncio
import math

import numpy as np
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
            # Viam base: +Y forward, +X right. Lidar ahead/left in that frame.
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


def test_pose_raw_viam_axes_without_y_forward_convert():
    configs = _tracer_like_frames()
    m = pose_of_frame_in_destination(
        configs, "rplidar", "base", y_forward_base=False
    )
    assert m is not None
    assert abs(m.x - (-0.05)) < 1e-9
    assert abs(m.y - 0.24) < 1e-9
    assert abs(m.z - 0.15) < 1e-9
    # Viam OV θ=-90° → same matrix as our mount θ=+π/2 (yaw sign differs).
    assert abs(m.theta - (math.pi / 2)) < 1e-6


def test_pose_converts_viam_y_forward_base_to_ros_x_forward():
    """FS is Viam Y-forward; nav-stack mounts are ROS X-forward."""
    configs = _tracer_like_frames()
    m = pose_of_frame_in_destination(
        configs, "rplidar", "base", y_forward_base=True
    )
    assert m is not None
    # p_ros = (y_v, -x_v) = (0.24, 0.05)
    assert abs(m.x - 0.24) < 1e-9
    assert abs(m.y - 0.05) < 1e-9
    assert abs(m.z - 0.15) < 1e-9
    # Rz(-π/2) @ Rz(+π/2) = I → mount θ ≈ 0
    assert abs(m.theta) < 1e-6


def test_fs_mount_round_trips_viam_rotation_and_points():
    """Mount RPY must rebuild Viam's matrix so PCD lands where the 3D scene does."""
    from src.geom.conversions import _mount_rotation, transform_lidar_mount_to_base_link
    from src.lidar.rplidar_protocol import polar_to_xyz_m
    from src.viam_frames import _pose_to_matrix, _viam_y_forward_base_to_ros

    configs = _tracer_like_frames()
    m = pose_of_frame_in_destination(
        configs, "rplidar", "base", y_forward_base=True
    )
    assert m is not None
    pose = configs[1].frame.pose_in_observer_frame.pose
    T_ros = _viam_y_forward_base_to_ros(_pose_to_matrix(pose))
    R_mount = _mount_rotation(m.theta, m.pitch, m.roll)
    assert np.allclose(T_ros[:3, :3], R_mount, atol=1e-9)

    xs, ys, zs = polar_to_xyz_m(0.0, 1000.0)
    p_mm = np.array([xs * 1000.0, ys * 1000.0, zs * 1000.0, 1.0])
    via_T = (T_ros @ p_mm)[:3] / 1000.0
    via_mount = transform_lidar_mount_to_base_link(
        np.array([[xs, ys, zs]]),
        x=m.x,
        y=m.y,
        z=m.z,
        theta=m.theta,
        pitch=m.pitch,
        roll=m.roll,
    )[0]
    assert np.allclose(via_T, via_mount, atol=1e-9)
    # After Y→X convert, R≈I: angle 0 (X-flipped −X) stays on base −X.
    assert abs(via_mount[0] - (m.x - 1.0)) < 1e-9
    assert abs(via_mount[1] - m.y) < 1e-9


def test_camera_fs_mount_round_trips_rotation():
    from src.geom.conversions import _mount_rotation
    from src.viam_frames import _pose_to_matrix, _viam_y_forward_base_to_ros

    configs = _tracer_like_frames()
    m = pose_of_frame_in_destination(
        configs, "camera", "base", y_forward_base=True
    )
    assert m is not None
    T_ros = _viam_y_forward_base_to_ros(
        _pose_to_matrix(configs[2].frame.pose_in_observer_frame.pose)
    )
    assert np.allclose(T_ros[:3, :3], _mount_rotation(m.theta, m.pitch, m.roll), atol=1e-9)


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
    assert any("Y-forward" in n for n in notes)
    assert any("rplidar" in n and "framesystem" in n for n in notes)
    rpl = next(l for l in cfg2.lidars if l.name == "rplidar")
    # Y-forward → X-forward: (x,y)_v=(-0.05,0.24) → (0.24, 0.05)
    assert abs(rpl.x - 0.24) < 1e-9
    assert abs(rpl.y - 0.05) < 1e-9
    assert abs(rpl.theta) < 1e-6
    cam = next(l for l in cfg2.lidars if l.name == "camera")
    assert abs(cam.x - 0.30) < 1e-9
    assert abs(cam.y - (-0.15)) < 1e-9
    # RealSense PCD is optical even when FS carries a -90° roll; keep the
    # optical remap and ignore FS pitch/roll so hits are not z-filtered away.
    assert cam.cloud_frame == "camera_optical"
    assert abs(cam.pitch) < 1e-9
    assert abs(cam.roll) < 1e-9
    assert any("kept camera_optical" in n for n in notes)


def test_apply_slam_keeps_explicit_mount():
    configs = _tracer_like_frames()
    raw = {
        "base": "base",
        "movement_sensor": "odom",
        "lidars": [
            {
                "name": "rplidar",
                "mount": {"x": 0.09, "y": 0.05, "z": 0.12, "theta": 3.14159},
            }
        ],
    }
    cfg = SlamConfig.from_dict(raw)
    cfg2, notes = apply_framesystem_to_slam_cfg(cfg, configs, raw_attrs=raw)
    assert any("override" in n for n in notes)
    assert abs(cfg2.lidars[0].x - 0.09) < 1e-9
    assert abs(cfg2.lidars[0].theta - 3.14159) < 1e-5


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
    assert abs(lidar.x - 0.24) < 1e-9


def test_apply_nav_fills_footprint_from_box():
    configs = _tracer_like_frames()
    raw = {"slam_service": "slam", "base": "base"}
    cfg = NavConfig.from_dict(raw)
    cfg2, notes = apply_framesystem_to_nav_cfg(cfg, configs, raw_attrs=raw)
    assert any("framesystem box" in n for n in notes)
    assert abs(cfg2.footprint_length_m - 0.72) < 1e-9
    assert abs(cfg2.footprint_width_m - 0.59) < 1e-9


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


@pytest.mark.asyncio
async def test_fetch_frame_system_config_times_out():
    class _SlowRobot:
        async def get_frame_system_config(self):
            await asyncio.sleep(10.0)
            return []

    with pytest.raises(asyncio.TimeoutError):
        await fetch_frame_system_config(_SlowRobot(), timeout_s=0.05)

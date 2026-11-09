"""Intrinsics / FOV helpers for vision landmarks."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.geom.conversions import Pose2D
from src.semantic.frames import (
    intrinsics_from_camera_properties,
    intrinsics_from_size,
)
from src.semantic.patches import backproject_patches
from src.semantic.types import CameraIntrinsics, PatchToken, VisionFrame
from src.viz.landmarks_view import project_map_point_to_pixel


def test_intrinsics_from_camera_properties_scales_resolution():
    props = SimpleNamespace(
        intrinsic_parameters=SimpleNamespace(
            width_px=1280,
            height_px=720,
            focal_x_px=920.0,
            focal_y_px=918.0,
            center_x_px=640.0,
            center_y_px=360.0,
        )
    )
    K = intrinsics_from_camera_properties(props, image_width=640, image_height=360)
    assert K is not None
    assert K.width == 640 and K.height == 360
    assert K.fx == pytest.approx(460.0)
    assert K.fy == pytest.approx(459.0)
    assert K.cx == pytest.approx(320.0)
    assert K.cy == pytest.approx(180.0)


def test_intrinsics_from_camera_properties_rejects_empty():
    props = SimpleNamespace(
        intrinsic_parameters=SimpleNamespace(
            width_px=0,
            height_px=0,
            focal_x_px=0.0,
            focal_y_px=0.0,
            center_x_px=0.0,
            center_y_px=0.0,
        )
    )
    assert (
        intrinsics_from_camera_properties(props, image_width=640, image_height=480)
        is None
    )


def test_intrinsics_use_separate_vfov():
    K = intrinsics_from_size(640, 480, fov_deg=69.0, vfov_deg=42.0)
    assert K.fx != pytest.approx(K.fy, rel=0.01)
    # Wider HFOV → smaller fx than a square-pixel fy from narrower VFOV? 
    # 69° on 640 vs 42° on 480:
    fx_exp = (0.5 * 640) / math.tan(math.radians(34.5))
    fy_exp = (0.5 * 480) / math.tan(math.radians(21.0))
    assert K.fx == pytest.approx(fx_exp, rel=1e-6)
    assert K.fy == pytest.approx(fy_exp, rel=1e-6)


def test_wrong_fov_causes_reproject_drift_after_yaw():
    """Same bug users see on opaque walls when fov_deg is wrong.

    Measure with true K, then reproject with a wrong wider FOV after a spin —
    the pixel no longer sits on the feature.
    """
    true_k = CameraIntrinsics(fx=460, fy=460, cx=320, cy=240, width=640, height=480)
    wrong_k = intrinsics_from_size(640, 480, fov_deg=87.0)  # much wider
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    u0, v0 = 420.0, 240.0
    frame = VisionFrame(
        camera_name="cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=color,
        depth_m=depth,
        intrinsics=true_k,
        mount_x=0.1,
        mount_z=0.5,
    )
    tok = PatchToken(0, 0, u0, v0, np.ones(4, np.float32))
    obs = backproject_patches(frame, [tok])
    assert obs
    pb = obs[0].point_base
    pose0 = Pose2D(0.0, 0.0, 0.0)
    map_xyz = np.array([pose0.x + pb[0], pose0.y + pb[1], pb[2]])
    pose1 = Pose2D(0.0, 0.0, math.radians(25.0))
    # True intrinsics: reproject is geometrically consistent.
    good = project_map_point_to_pixel(
        map_xyz,
        robot_pose=pose1,
        intrinsics=true_k,
        mount_x=0.1,
        mount_y=0.0,
        mount_z=0.5,
        mount_theta=0.0,
    )
    bad = project_map_point_to_pixel(
        map_xyz,
        robot_pose=pose1,
        intrinsics=wrong_k,
        mount_x=0.1,
        mount_y=0.0,
        mount_z=0.5,
        mount_theta=0.0,
    )
    assert good is not None and bad is not None
    # Wrong FOV shifts the overlay by several pixels after a modest spin.
    assert abs(bad[0] - good[0]) > 8.0

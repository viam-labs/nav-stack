"""Unit tests for landmark overlay projection / vision-camera config."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.config import VisionCameraConfig
from src.geom import conversions as conv
from src.geom.conversions import Pose2D
from src.semantic.landmarks import EmbeddingLandmark, EmbeddingLandmarkMap
from src.semantic.types import CameraIntrinsics, LandmarkStatus
from src.viz.landmarks_view import (
    collect_landmark_map_points,
    project_map_point_to_pixel,
    render_landmarks_overlay,
    sensor_to_camera_optical,
)


def test_vision_camera_config_deps():
    cfg = VisionCameraConfig.from_dict({"slam": "slam", "source": "camera-top"})
    assert cfg.required_dependencies() == ["slam", "camera-top"]
    assert cfg.show_candidates is True


def test_sensor_optical_roundtrip():
    optical = np.array([[0.1, -0.2, 2.0]])
    sensor = conv.camera_optical_to_sensor_frame(optical)
    back = sensor_to_camera_optical(sensor)
    assert back[0, 0] == pytest.approx(0.1)
    assert back[0, 1] == pytest.approx(-0.2)
    assert back[0, 2] == pytest.approx(2.0)


def test_project_forward_point_to_centerish():
    K = CameraIntrinsics(fx=320, fy=320, cx=320, cy=240, width=640, height=480)
    map_xyz = np.array([2.0, 0.0, 0.5])
    pose = Pose2D(0.0, 0.0, 0.0)
    proj = project_map_point_to_pixel(
        map_xyz,
        robot_pose=pose,
        intrinsics=K,
        mount_x=0.0,
        mount_y=0.0,
        mount_z=0.5,
        mount_theta=0.0,
    )
    assert proj is not None
    u, v, depth = proj
    assert depth == pytest.approx(2.0, abs=0.05)
    assert abs(u - K.cx) < 5.0
    assert abs(v - K.cy) < 5.0


def test_render_overlay_draws_box():
    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (30, 30, 40)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    lm = EmbeddingLandmark(
        landmark_id=7,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 0.5]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        descriptors=[np.ones(4, dtype=np.float32)],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
    )
    png = render_landmarks_overlay(
        color,
        [(lm, np.array([2.0, 0.0, 0.5]))],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        intrinsics=K,
        mount_z=0.5,
    )
    img = Image.open(io.BytesIO(png))
    assert img.size == (320, 240)
    arr = np.asarray(img)
    assert arr[..., 1].max() > 40


def test_collect_prefers_confirmed():
    store = EmbeddingLandmarkMap()
    pose = Pose2D(0.0, 0.0, 0.0)
    for i, status in enumerate(
        [LandmarkStatus.CANDIDATE, LandmarkStatus.CONFIRMED, LandmarkStatus.MOVED]
    ):
        lm = EmbeddingLandmark(
            landmark_id=i + 1,
            anchor_node=0,
            rel_xyz=np.array([float(i), 0.0, 1.0]),
            normal_rel=np.array([-1.0, 0.0, 0.0]),
            descriptors=[],
            observations=3,
            status=status,
        )
        store._landmarks[lm.landmark_id] = lm  # noqa: SLF001
    pts = collect_landmark_map_points(
        store, {0: pose}, include_candidates=True, include_moved=True
    )
    assert pts[0][0].status == LandmarkStatus.CONFIRMED

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
    assert cfg.show_detections is True


def test_render_overlay_draws_yolo_detections():
    from src.semantic.types import Detection

    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (20, 20, 28)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    det = Detection(label="painting", conf=0.91, x0=40, y0=30, x1=200, y1=180)
    png = render_landmarks_overlay(
        color,
        [],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        intrinsics=K,
        detections=[det],
        detection_frame_size=(320, 240),
    )
    img = Image.open(io.BytesIO(png))
    arr = np.asarray(img)
    # Cyan detector outline bumps the blue channel.
    assert arr[..., 2].max() > 40


def test_match_detection_box_contains_center():
    from src.semantic.types import Detection
    from src.viz.landmarks_view import _match_detection_box

    det = Detection(label="painting", conf=0.4, x0=40, y0=30, x1=200, y1=180)
    boxes = [(40.0, 30.0, 200.0, 180.0, det)]
    matched = _match_detection_box(120.0, 100.0, boxes, used=set())
    assert matched is not None
    assert matched[4].label == "painting"


def test_render_nests_detector_landmark_inside_yolo():
    """Dc fill should sit in the YOLO rectangle, not a separate offset square."""
    from src.semantic.types import Detection

    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (20, 20, 28)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    # Landmark at map (2,0,0.5) with mount_z=0.5 projects near image center.
    lm = EmbeddingLandmark(
        landmark_id=3,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 0.5]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 0.5]),
        descriptors=[],
        observations=2,
        status=LandmarkStatus.CANDIDATE,
        from_detector=True,
    )
    det = Detection(label="painting", conf=0.5, x0=100, y0=60, x1=220, y1=180)
    png = render_landmarks_overlay(
        color,
        [(lm, np.array([2.0, 0.0, 0.5]))],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        intrinsics=K,
        mount_z=0.5,
        detections=[det],
        detection_frame_size=(320, 240),
        yolo_loaded=True,
    )
    arr = np.asarray(Image.open(io.BytesIO(png)))
    # Nested yellow fill inside the box should raise green+red in that region.
    roi = arr[70:170, 110:210]
    assert roi[..., 0].max() > 40 and roi[..., 1].max() > 40


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


def test_project_roundtrip_stable_across_yaw():
    """Fixed map_xyz must reproject to the geometrically correct pixel after a spin."""
    import math

    from src.semantic.patches import backproject_patches
    from src.semantic.types import PatchToken, VisionFrame

    K = CameraIntrinsics(fx=320, fy=320, cx=320, cy=240, width=640, height=480)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    # Pixel slightly off-center so yaw motion moves it.
    u0, v0 = 400.0, 240.0
    frame = VisionFrame(
        camera_name="cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=color,
        depth_m=depth,
        intrinsics=K,
        mount_x=0.0,
        mount_y=0.0,
        mount_z=0.5,
        mount_theta=0.0,
    )
    tok = PatchToken(row=0, col=0, u=u0, v=v0, descriptor=np.ones(4, dtype=np.float32))
    obs = backproject_patches(frame, [tok])
    assert len(obs) == 1
    pb = obs[0].point_base
    pose0 = Pose2D(0.0, 0.0, 0.0)
    c0, s0 = math.cos(pose0.theta), math.sin(pose0.theta)
    map_xyz = np.array(
        [
            pose0.x + c0 * pb[0] - s0 * pb[1],
            pose0.y + s0 * pb[0] + c0 * pb[1],
            pb[2],
        ]
    )
    # Spin in place; same map point must project back near the ray from pose1.
    pose1 = Pose2D(0.0, 0.0, math.radians(25.0))
    proj = project_map_point_to_pixel(
        map_xyz,
        robot_pose=pose1,
        intrinsics=K,
        mount_x=0.0,
        mount_y=0.0,
        mount_z=0.5,
        mount_theta=0.0,
    )
    assert proj is not None
    u1, v1, _d = proj
    # Expected pixel: transform map → base at pose1 → optical → K.
    c1, s1 = math.cos(pose1.theta), math.sin(pose1.theta)
    dx, dy = map_xyz[0] - pose1.x, map_xyz[1] - pose1.y
    base = np.array([c1 * dx + s1 * dy, -s1 * dx + c1 * dy, map_xyz[2]])
    sensor = conv.transform_base_link_to_lidar_mount(
        base.reshape(1, 3), x=0.0, y=0.0, z=0.5, theta=0.0
    )[0]
    optical = sensor_to_camera_optical(sensor.reshape(1, 3))[0]
    u_exp = K.fx * optical[0] / optical[2] + K.cx
    v_exp = K.fy * optical[1] / optical[2] + K.cy
    assert u1 == pytest.approx(u_exp, abs=0.5)
    assert v1 == pytest.approx(v_exp, abs=0.5)
    # And the spun projection must differ from the original pixel (not stuck).
    assert abs(u1 - u0) > 5.0


def test_render_overlay_draws_box():
    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (30, 30, 40)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    lm = EmbeddingLandmark(
        landmark_id=7,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 0.5]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 0.5]),
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
            map_xyz=np.array([float(i), 0.0, 1.0]),
            descriptors=[],
            observations=3,
            view_poses_xy=[(0.0, 0.0, 0.0), (0.6, 0.0, 0.0)],
            position_std_m=0.04,
            status=status,
        )
        store._landmarks[lm.landmark_id] = lm  # noqa: SLF001
    pts = collect_landmark_map_points(
        store, {0: pose}, include_candidates=True, include_moved=True
    )
    assert pts[0][0].status == LandmarkStatus.CONFIRMED


def test_collect_hides_single_view_candidates():
    store = EmbeddingLandmarkMap()
    pose = Pose2D(0.0, 0.0, 0.0)
    unstable = EmbeddingLandmark(
        landmark_id=1,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 1.0]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptors=[],
        observations=1,
        view_poses_xy=[(0.0, 0.0, 0.0)],
        position_std_m=float("inf"),
        status=LandmarkStatus.CANDIDATE,
    )
    store._landmarks[1] = unstable  # noqa: SLF001
    pts = collect_landmark_map_points(
        store, {0: pose}, include_candidates=True, stable_candidates_only=True
    )
    assert pts == []


def test_collect_shows_single_view_detector_candidates():
    store = EmbeddingLandmarkMap()
    pose = Pose2D(0.0, 0.0, 0.0)
    det_lm = EmbeddingLandmark(
        landmark_id=2,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 1.0]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptors=[],
        observations=1,
        view_poses_xy=[(0.0, 0.0, 0.0)],
        position_std_m=float("inf"),
        status=LandmarkStatus.CANDIDATE,
        from_detector=True,
    )
    store._landmarks[2] = det_lm  # noqa: SLF001
    pts = collect_landmark_map_points(
        store, {0: pose}, include_candidates=True, stable_candidates_only=True
    )
    assert len(pts) == 1
    assert pts[0][0].from_detector is True

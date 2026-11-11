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


def _detector_landmark(landmark_id: int, map_xyz) -> EmbeddingLandmark:
    return EmbeddingLandmark(
        landmark_id=landmark_id,
        anchor_node=0,
        rel_xyz=np.array(map_xyz, dtype=float),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array(map_xyz, dtype=float),
        descriptors=[],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
    )


def test_render_draws_detector_landmark_at_its_own_point_not_the_yolo_box():
    """A landmark inside a YOLO box must not paint that box with its ID.

    Filling the rectangle made a landmark stored metres in front of a painting
    look as if it sat on the painting.
    """
    from src.semantic.types import Detection

    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (20, 20, 28)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    # Projects to the image center (160, 120); half-size ~10 px at 2 m.
    lm = _detector_landmark(3, [2.0, 0.0, 0.5])
    det = Detection(label="painting", conf=0.5, x0=60, y0=40, x1=260, y1=200)
    png = render_landmarks_overlay(
        color,
        [(lm, lm.map_xyz)],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        intrinsics=K,
        mount_z=0.5,
        detections=[det],
        detection_frame_size=(320, 240),
        yolo_loaded=True,
    )
    arr = np.asarray(Image.open(io.BytesIO(png))).astype(int)
    green = (arr[..., 1] > 150) & (arr[..., 0] < 120) & (arr[..., 2] < 150)
    # Green landmark square around its own projection...
    assert green[105:135, 145:175].any()
    # ...and none along the YOLO rectangle or elsewhere inside it (the label
    # sits just above the square).
    green[94:136, 144:230] = False
    assert not green[36:204, 56:264].any()


def test_render_shows_which_boxes_were_used_and_which_were_skipped():
    """Used box: cyan, tied to its landmark. Skipped box: grey, no fill."""
    from src.semantic.types import Detection

    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (20, 20, 28)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    # Projects to the image center (160, 120); half-size ~10 px at 2 m.
    lm = _detector_landmark(3, [2.0, 0.0, 0.5])
    used = Detection(label="painting", conf=0.5, x0=20, y0=60, x1=100, y1=180)
    skipped = Detection(label="poster", conf=0.3, x0=220, y0=60, x1=300, y1=180)
    notes = [
        {"outcome": "used", "text": "=DC3", "landmark_id": 3, "landmark": "DC3"},
        {"outcome": "skipped_range", "text": "far 9.6m", "landmark_id": None},
    ]

    def render(detection_notes):
        png = render_landmarks_overlay(
            color,
            [(lm, lm.map_xyz)],
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            intrinsics=K,
            mount_z=0.5,
            detections=[used, skipped],
            detection_frame_size=(320, 240),
            yolo_loaded=True,
            detection_notes=detection_notes,
        )
        arr = np.asarray(Image.open(io.BytesIO(png))).astype(int)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cyan = (b > 220) & (g > 150) & (r < 120)
        grey = (abs(r - 170) < 12) & (abs(g - 170) < 12) & (abs(b - 170) < 12)
        green = (g > 150) & (r < 120) & (b < 150)
        return cyan, grey, green

    cyan, grey, green = render(notes)
    # Left edges of the two rectangles.
    assert cyan[60:181, 19:23].any() and not grey[60:181, 19:23].any()
    assert grey[60:181, 219:223].any() and not cyan[60:181, 219:223].any()
    # The skipped box is not filled.
    assert tuple(
        np.asarray(
            Image.open(
                io.BytesIO(
                    render_landmarks_overlay(
                        color,
                        [],
                        robot_pose=Pose2D(0.0, 0.0, 0.0),
                        intrinsics=K,
                        detections=[used, skipped],
                        detection_frame_size=(320, 240),
                        detection_notes=notes,
                    )
                )
            )
        )[120, 260, :3]
    ) == (20, 20, 28)
    # A line runs from the used box's centre (60, 120) to the landmark square.
    assert green[119:122, 105:145].sum() >= 40

    # Without notes (detections not surveyed yet) every box is plain cyan and
    # nothing is tied to a landmark.
    cyan, grey, green = render(None)
    assert cyan[60:181, 219:223].any() and not grey[60:181, 219:223].any()
    assert not green[119:122, 105:145].any()
    # Notes that do not line up with the detections are ignored, not misapplied.
    cyan, grey, _green = render(notes[:1])
    assert cyan[60:181, 219:223].any() and not grey.any()


def test_render_marks_seen_through_landmark_red():
    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (20, 20, 28)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    lm = _detector_landmark(9, [2.0, 0.0, 0.5])

    def render(seen_through):
        png = render_landmarks_overlay(
            color,
            [(lm, lm.map_xyz)],
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            intrinsics=K,
            mount_z=0.5,
            seen_through_ids=seen_through,
        )
        arr = np.asarray(Image.open(io.BytesIO(png))).astype(int)
        return (arr[..., 0] > 200) & (arr[..., 1] < 90) & (arr[..., 2] < 90)

    assert not render([])[100:140, 140:180].any()
    red = render([9])
    # Outline plus a cross through the center of the square.
    assert red[100:140, 140:180].sum() > 40
    assert red[118:123, 158:163].any()


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


def test_render_overlay_writeback_rings():
    """Paired landmarks get a magenta accent; pool-only get orange."""
    color = np.zeros((240, 320, 3), dtype=np.uint8)
    color[:] = (20, 20, 28)
    K = CameraIntrinsics(fx=160, fy=160, cx=160, cy=120, width=320, height=240)
    lm_pair = EmbeddingLandmark(
        landmark_id=7,
        anchor_node=0,
        rel_xyz=np.array([1.5, 0.0, 0.9]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([1.5, 0.0, 0.9]),
        descriptors=[],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
    )
    lm_vis = EmbeddingLandmark(
        landmark_id=8,
        anchor_node=0,
        rel_xyz=np.array([1.5, 0.3, 0.9]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([1.5, 0.3, 0.9]),
        descriptors=[],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
    )
    png = render_landmarks_overlay(
        color,
        [(lm_pair, lm_pair.map_xyz), (lm_vis, lm_vis.map_xyz)],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        intrinsics=K,
        mount_x=0.1,
        mount_z=0.4,
        writeback_visible_ids=[7, 8],
        writeback_paired_ids=[7],
    )
    arr = np.asarray(Image.open(io.BytesIO(png)))
    # Magenta ring bumps red+blue; orange bumps red+green.
    assert arr[..., 0].max() > 40
    assert arr[..., 2].max() > 40


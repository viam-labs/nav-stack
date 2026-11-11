"""Survey-only vision supervisor + landmark persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import SlamConfig
from src.geom.conversions import Pose2D
from src.semantic.persist import load_landmarks, save_landmarks
from src.semantic.supervisor import VisionSupervisor
from src.semantic.types import CameraIntrinsics, LandmarkStatus, VisionFrame
from src.semantic.worker import WorkerConfig


def _cfg(**extra) -> SlamConfig:
    d = {
        "base": "b",
        "lidars": [
            {"name": "front"},
            {
                "name": "depth-cam",
                "obstacles_only": True,
                "cloud_frame": "camera_optical",
                "scan_source": "point_cloud",
            },
        ],
        "vision_enabled": True,
        "vision_cameras": ["depth-cam"],
        "vision_synthetic": True,
        "vision_survey_only": True,
    }
    d.update(extra)
    return SlamConfig.from_dict(d)


def _frame() -> VisionFrame:
    h, w = 120, 160
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[::4, ::4] = (180, 90, 40)
    depth = np.full((h, w), 1.5, dtype=np.float32)
    K = CameraIntrinsics(
        fx=0.5 * w, fy=0.5 * h, cx=0.5 * w, cy=0.5 * h, width=w, height=h
    )
    return VisionFrame(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=color,
        depth_m=depth,
        intrinsics=K,
        mount_x=0.1,
        mount_z=0.4,
    )


def test_supervisor_ingest_builds_landmarks(tmp_path: Path):
    cfg = _cfg()
    pose = Pose2D(0.0, 0.0, 0.0)
    poses = {0: pose}

    def get_anchor():
        return 0, pose, poses

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=get_anchor,
    )
    # Force synthetic worker (no ONNX).
    vision._worker.cfg = WorkerConfig(synthetic=True, input_short_side=112)  # noqa: SLF001
    from src.semantic.worker import VisionWorker

    vision._worker = VisionWorker(WorkerConfig(synthetic=True, input_short_side=112))  # noqa: SLF001

    frame = _frame()
    # Multiple viewpoints for promotion.
    for dx in (0.0, 0.6, 1.2):
        p = Pose2D(dx, 0.0, 0.0)
        poses[0] = p

        def get_anchor_i(p=p):
            return 0, p, {0: p}

        vision._get_anchor = get_anchor_i  # noqa: SLF001
        vision._get_pose = lambda p=p: p  # noqa: SLF001
        vision._get_pose_at = lambda _t, p=p: p  # noqa: SLF001
        vision.ingest_frame(frame, mapping=True)

    status = vision.status()
    assert status["frames"] >= 3
    assert status["observations"] > 0
    assert status["landmarks"]["total"] >= 1
    # Survey path only — pose writeback stays off.
    assert status["survey_only"] is True

    path = vision.save_to_map_dir(tmp_path)
    assert path is not None
    loaded = load_landmarks(tmp_path)
    assert loaded is not None
    assert len(loaded) >= 1


def test_start_does_not_drop_dinov2_session():
    """Regression: start() used to close() the worker and force synthetic."""
    import asyncio

    cfg = _cfg(vision_synthetic=True)
    pose = Pose2D(0.0, 0.0, 0.0)
    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
    )
    # Pretend a real ONNX session is loaded.
    vision._worker.cfg.synthetic = False  # noqa: SLF001
    sentinel = object()
    vision._worker._dino = sentinel  # noqa: SLF001
    vision._worker.stats["provider"] = "CPUExecutionProvider"

    loop = asyncio.new_event_loop()
    try:
        vision.start(loop)
        assert vision._worker._dino is sentinel  # noqa: SLF001
        assert vision._worker.stats["provider"] == "CPUExecutionProvider"
        vision.stop()
        # Drain the cancelled vision task so pytest doesn't warn.
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()


def test_save_load_roundtrip(tmp_path: Path):
    from src.semantic.landmarks import EmbeddingLandmarkMap
    from src.semantic.place import l2_normalize

    store = EmbeddingLandmarkMap()
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    pose = Pose2D(1.0, 0.0, 0.0)
    for i, view in enumerate([(0.0, 0.0), (0.7, 0.0), (1.4, 0.0)]):
        store.observe(
            map_xyz=np.array([2.0, 0.005 * (i - 1), 1.0], dtype=float),
            descriptor=desc,
            normal_map=np.array([-1.0, 0.0, 0.0]),
            anchor_node=0,
            anchor_pose=pose,
            view_xy=view,
        )
    assert any(lm.status == LandmarkStatus.CONFIRMED for lm in store.all_landmarks())
    save_landmarks(tmp_path, store)
    loaded = load_landmarks(tmp_path)
    assert loaded is not None
    assert len(loaded.confirmed()) >= 1


def test_resolve_detector_classes_from_sidecar():
    from src.semantic.supervisor import _resolve_detector_classes

    models = Path(__file__).resolve().parents[1] / "src" / "semantic" / "models"
    classes_path = models / "yolov8s-worldv2-indoor.classes.json"
    assert classes_path.is_file()
    # Fake onnx path next to the committed classes.json.
    fake_onnx = str(models / "yolov8s-worldv2-indoor.onnx")
    got = _resolve_detector_classes(fake_onnx, configured=[])
    assert "painting" in got
    assert "door" in got
    # Appended so a rebuilt ONNX learns people without shifting older class ids.
    assert got[-1] == "person"
    # Config wins.
    assert _resolve_detector_classes(fake_onnx, configured=["tv", "clock"]) == (
        "tv",
        "clock",
    )


def test_person_box_is_handed_to_the_paint_gate():
    """A person blanks lidar returns. A painting does not."""
    from src.semantic.types import Detection, VisionResult

    cfg = _cfg(mode="mapping", vision_survey_only=False)
    pose = Pose2D(0.0, 0.0, 0.0)
    seen = {}

    def note(discs, captured_at):
        seen["n"] = len(discs)
        seen["t"] = captured_at
        seen["labels_x"] = [round(d.x, 2) for d in discs]

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
        note_paint_discs=note,
    )
    frame = _frame()
    frame.captured_at = 1.5
    person = Detection(label="person", conf=0.9, x0=40, y0=20, x1=120, y1=100)
    painting = Detection(label="painting", conf=0.9, x0=10, y0=10, x1=30, y1=30)
    result = VisionResult(
        camera_name=frame.camera_name,
        stamp=frame.stamp,
        captured_at=1.5,
        patches=[],
        detections=[person, painting],
        provider="test",
    )
    vision._survey(frame, result)  # noqa: SLF001
    assert seen["n"] == 1
    assert seen["t"] == 1.5
    assert seen["labels_x"][0] > 0.5
    assert vision.status()["paint_gate"]["last_discs"] == 1


def test_loop_anchor_needs_three_inliers_on_one_keyframe():
    from src.semantic.supervisor import dominant_loop_anchor

    # Three inliers on node 2, one on node 9.
    anchor = dominant_loop_anchor(
        [1, 2, 3, 4],
        [True, True, True, True],
        {1: 2, 2: 2, 3: 2, 4: 9},
        min_inliers=3,
    )
    assert anchor == 2
    # Split 2 and 2 does not close a loop.
    assert (
        dominant_loop_anchor(
            [1, 2, 3, 4],
            [True, True, True, True],
            {1: 2, 2: 2, 3: 9, 4: 9},
            min_inliers=3,
        )
        is None
    )


def test_survey_prefers_detector_boxes():
    """Injected YOLO boxes should yield fewer, box-pooled observations."""
    from src.semantic.types import Detection, VisionResult
    from src.semantic.worker import VisionWorker

    cfg = _cfg(
        vision_prefer_detections=True,
        vision_dense_fallback_patches=4,
        vision_max_patches_per_frame=64,
        vision_landmark_labels=["painting"],
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
    )
    vision._worker = VisionWorker(WorkerConfig(synthetic=True, input_short_side=112))  # noqa: SLF001

    frame = _frame()
    base = vision._worker.process(frame)  # noqa: SLF001
    # One painting covering most of the textured wall.
    det = Detection(label="painting", conf=0.95, x0=20.0, y0=20.0, x1=140.0, y1=100.0)
    result = VisionResult(
        camera_name=base.camera_name,
        stamp=base.stamp,
        captured_at=base.captured_at,
        patches=base.patches,
        detections=[det],
        place_descriptor=base.place_descriptor,
        infer_ms=base.infer_ms,
        provider=base.provider,
    )
    with vision._lock:  # noqa: SLF001
        vision._last_result = result  # noqa: SLF001
        vision._frames += 1  # noqa: SLF001
    vision._survey(frame, result)  # noqa: SLF001
    # Detector pooling + tiny dense fallback ≪ full dense grid.
    assert vision.status()["observations"] <= 12
    assert vision.status()["landmarks"]["total"] >= 1
    assert vision.status()["prefer_detections"] is True


def _surveyed_boxes(vision: VisionSupervisor):
    """Survey one frame with a usable box, a far box and a non-landmark box."""
    from src.semantic.types import Detection, VisionResult

    frame = _frame()
    # Right third of the wall is far beyond box range and noisy.
    rng = np.random.default_rng(5)
    frame.depth_m[:, 85:] = rng.uniform(7.5, 12.0, size=(120, 75))
    base = vision._worker.process(frame)  # noqa: SLF001
    dets = [
        Detection(label="person", conf=0.9, x0=40.0, y0=10.0, x1=120.0, y1=110.0),
        Detection(label="painting", conf=0.8, x0=10.0, y0=20.0, x1=70.0, y1=100.0),
        Detection(label="painting", conf=0.4, x0=95.0, y0=20.0, x1=155.0, y1=100.0),
    ]
    result = VisionResult(
        camera_name=base.camera_name,
        stamp=base.stamp,
        captured_at=base.captured_at,
        patches=base.patches,
        detections=dets,
        place_descriptor=base.place_descriptor,
        infer_ms=base.infer_ms,
        provider=base.provider,
    )
    with vision._lock:  # noqa: SLF001
        vision._last_result = result  # noqa: SLF001
        vision._last_camera_geom = {"width": 160, "height": 120}  # noqa: SLF001
    vision._survey(frame, result)  # noqa: SLF001
    return result


def test_survey_records_what_each_box_became():
    """The overlay and ``landmark_audit`` must be able to explain every box."""
    from viam.utils import dict_to_struct

    cfg = _cfg(
        vision_prefer_detections=True,
        vision_dense_fallback_patches=0,
        vision_landmark_labels=["painting"],
    )
    vision = _vision(cfg, Pose2D(0.0, 0.0, 0.0))
    result = _surveyed_boxes(vision)

    detections, size, boxes = vision.last_detections_with_outcomes()
    assert detections == result.detections and size == (160, 120)
    assert [b["outcome"] for b in boxes] == [
        "not_landmark_label",
        "used",
        "skipped_range",
    ]
    person, used, far = boxes
    assert person["text"] == "not a landmark class" and person["landmark"] is None
    lm = vision.landmarks.get(used["landmark_id"])
    assert lm is not None and lm.from_detector is True
    assert used["landmark"] == f"Dc{lm.landmark_id}"
    assert used["text"] == f"=Dc{lm.landmark_id}"
    assert abs(used["depth_m"] - 1.5) < 0.05
    assert far["text"].startswith("far ") and far["depth_m"] > 7.0
    assert far["landmark_id"] is None
    # The old two-value call still works for callers that only want boxes.
    assert vision.last_detections() == (detections, size)

    report = vision.landmark_audit()
    assert [b["label"] for b in report["boxes"]] == ["person", "painting", "painting"]
    assert [b["text"] for b in report["boxes"]] == [b["text"] for b in boxes]
    assert all("landmark_id" not in b for b in report["boxes"])
    dict_to_struct(report)

    box_depth = vision.status()["box_depth"]
    assert box_depth["by_label"] == {
        "painting": {"used": 1, "skipped_range": 1},
        "person": {"not_landmark_label": 1},
    }
    assert box_depth["used"] == 1 and box_depth["skipped_range"] == 1
    dict_to_struct(vision.status())


def test_box_outcomes_are_not_reported_for_unsurveyed_detections():
    """Newer detections than the last survey must not borrow its notes."""
    import dataclasses

    cfg = _cfg(
        vision_prefer_detections=True,
        vision_dense_fallback_patches=0,
        vision_landmark_labels=["painting"],
    )
    vision = _vision(cfg, Pose2D(0.0, 0.0, 0.0))
    result = _surveyed_boxes(vision)
    newer = dataclasses.replace(result, detections=list(result.detections))
    with vision._lock:  # noqa: SLF001
        vision._last_result = newer  # noqa: SLF001
    detections, _size, boxes = vision.last_detections_with_outcomes()
    assert len(detections) == 3 and boxes is None

    vision.clear_landmarks()
    assert vision.status()["box_depth"]["by_label"] == {}


# --- Depth audit, capture pose, clear_landmarks ---


def _vision(cfg: SlamConfig, pose: Pose2D, **kwargs) -> VisionSupervisor:
    from src.semantic.worker import VisionWorker

    args = {
        "get_pose": lambda: pose,
        "get_pose_at": lambda _t: pose,
        "get_anchor": lambda: (0, pose, {0: pose}),
    }
    args.update(kwargs)
    vision = VisionSupervisor(cfg=cfg, cameras={}, **args)
    vision._worker = VisionWorker(WorkerConfig(synthetic=True, input_short_side=112))  # noqa: SLF001
    return vision


def _air_landmark(vision: VisionSupervisor, lid: int = 500):
    """Confirmed landmark 0.8 m ahead of the lens; ``_frame()`` wall is at 1.5 m."""
    from src.semantic.landmarks import EmbeddingLandmark

    xyz = np.array([0.9, 0.0, 0.4])
    lm = EmbeddingLandmark(
        landmark_id=lid,
        anchor_node=0,
        rel_xyz=xyz.copy(),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=xyz.copy(),
        descriptors=[np.array([0.0, 0.0, 1.0], dtype=np.float32)],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
    )
    store = vision.landmarks
    store._landmarks[lid] = lm  # noqa: SLF001
    store._index_add(lm, xyz)  # noqa: SLF001
    store._next_id = max(store._next_id, lid + 1)  # noqa: SLF001
    return lm


def test_survey_drops_landmark_the_camera_sees_through():
    vision = _vision(_cfg(), Pose2D(0.0, 0.0, 0.0))
    _air_landmark(vision)
    frame = _frame()
    for _ in range(2):
        vision.ingest_frame(frame, mapping=True)
    assert vision.landmarks.get(500) is not None
    assert vision.seen_through_ids() == [500]
    report = vision.landmark_audit()
    row = report["rows"][0]  # worst first
    assert (row["id"], row["verdict"], row["strikes"]) == (500, "seen_through", 2)
    assert row["stored_depth_m"] < 1.0 < row["measured_depth_m"]
    assert report["removes"] is True

    vision.ingest_frame(frame, mapping=True)
    assert vision.landmarks.get(500) is None
    status = vision.status()
    assert status["depth_audit"]["removed_total"] == 1
    assert status["depth_audit"]["last"]["removed_ids"] == [500]
    assert status["landmarks"]["seen_through_removed"] == 1
    report = vision.landmark_audit()
    assert [r["id"] for r in report["recent_removed"]] == [500]
    # Landmarks the same frames built on the wall are untouched.
    assert status["landmarks"]["total"] >= 1
    others = [r for r in report["rows"] if r["id"] != 500]
    assert others and all(r["verdict"] == "agrees" for r in others)


def test_audit_runs_on_a_frame_with_nothing_to_add():
    """An air landmark has no box to match, so the audit cannot wait for one."""
    from src.semantic.types import VisionResult

    vision = _vision(_cfg(), Pose2D(0.0, 0.0, 0.0))
    _air_landmark(vision)
    frame = _frame()
    empty = VisionResult(
        camera_name=frame.camera_name,
        stamp=0.0,
        captured_at=0.0,
        patches=[],
        detections=[],
        place_descriptor=None,
        infer_ms=0.0,
        provider="test",
    )
    for _ in range(3):
        vision._survey(frame, empty)  # noqa: SLF001
    assert vision.landmarks.get(500) is None


def test_audit_only_reports_while_localizing():
    vision = _vision(
        _cfg(mode="localizing", vision_survey_only=False), Pose2D(0.0, 0.0, 0.0)
    )
    _air_landmark(vision)
    frame = _frame()
    for _ in range(6):
        vision.ingest_frame(frame, mapping=True)
    assert vision.landmarks.get(500) is not None
    status = vision.status()["depth_audit"]
    assert status["removes"] is False
    assert status["flagged_ids"] == [500]
    assert status["removed_total"] == 0


def test_clear_landmarks_drops_store_and_saved_file(tmp_path: Path):
    from src.semantic.persist import landmarks_path

    vision = _vision(_cfg(), Pose2D(0.0, 0.0, 0.0))
    frame = _frame()
    vision.ingest_frame(frame, mapping=True)
    assert vision.status()["landmarks"]["total"] >= 1
    assert vision.save_to_map_dir(tmp_path) is not None
    assert landmarks_path(tmp_path).is_file()

    out = vision.clear_landmarks(tmp_path)
    assert out["cleared"] >= 1 and out["saved_file_removed"] is True
    assert vision.status()["landmarks"]["total"] == 0
    assert not landmarks_path(tmp_path).is_file()
    assert vision.landmark_audit()["rows"] == []
    # Surveying starts again from an empty store.
    vision.ingest_frame(frame, mapping=True)
    assert vision.status()["landmarks"]["total"] >= 1


def _aged_frame(age_s: float) -> VisionFrame:
    import dataclasses
    import time

    return dataclasses.replace(_frame(), captured_at=time.monotonic() - age_s)


def test_capture_pose_prefers_lookup_then_grab_stamp():
    """A frame that aged out of odometry history keeps its grab-time pose."""
    now = Pose2D(5.0, 0.0, 0.0)
    at_grab = Pose2D(4.0, 0.0, 0.0)
    looked_up = Pose2D(4.2, 0.0, 0.0)
    answers = [looked_up, None]
    vision = _vision(_cfg(), now, get_pose_at=lambda _t: answers.pop(0))
    frame = _aged_frame(0.4)
    assert vision._capture_pose(frame, at_grab) is looked_up  # noqa: SLF001
    assert vision._capture_pose(frame, at_grab) is at_grab  # noqa: SLF001
    assert vision.status()["capture_pose"]["grab_time_fallbacks"] == 1


def test_stale_frame_without_capture_pose_is_dropped():
    now = Pose2D(5.0, 0.0, 0.0)
    vision = _vision(_cfg(), now, get_pose_at=lambda _t: None)
    # Fresh frame, no odometry history yet: the current pose stands in.
    assert vision._pose_at_grab(_aged_frame(0.1)) is now  # noqa: SLF001
    # Old frame: the robot may have moved since; do not place it at ``now``.
    assert vision._pose_at_grab(_aged_frame(1.5)) is None  # noqa: SLF001
    assert vision.status()["capture_pose"]["stale_frames_dropped"] == 1


def test_tick_skips_survey_for_a_stale_frame():
    import asyncio

    vision = _vision(_cfg(), Pose2D(0.0, 0.0, 0.0), get_pose_at=lambda _t: None)
    vision._cameras = {"depth-cam": object()}  # noqa: SLF001
    frames = [_aged_frame(1.5), _aged_frame(0.05)]

    async def grab(_cam, _lidar):
        return frames.pop(0)

    vision._grab_frame = grab  # noqa: SLF001
    asyncio.run(vision._tick_once())  # noqa: SLF001
    assert vision.status()["frames"] == 0
    assert vision.status()["capture_pose"]["stale_frames_dropped"] == 1
    asyncio.run(vision._tick_once())  # noqa: SLF001
    assert vision.status()["frames"] == 1
    assert vision.status()["observations"] > 0


def test_slam_do_commands_audit_and_clear_landmarks(tmp_path: Path):
    """``landmark_audit`` / ``clear_landmarks`` through the SLAM DoCommand."""
    import asyncio
    from unittest.mock import MagicMock

    from viam.utils import dict_to_struct

    from src.models.slam import SlamService
    from src.nav.stores.maps import MapStore
    from src.semantic.persist import landmarks_path

    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store  # noqa: SLF001
    slam._manager = MagicMock()  # noqa: SLF001
    slam._reset_live_slam = MagicMock()  # noqa: SLF001

    def do(name: str) -> dict:
        return asyncio.run(slam.do_command({"command": name}))

    assert do("landmark_audit") == {
        "enabled": False,
        "reason": "vision not running",
        "rows": [],
    }

    vision = _vision(_cfg(), Pose2D(0.0, 0.0, 0.0))
    slam._vision = vision  # noqa: SLF001
    frame = _frame()
    vision.ingest_frame(frame, mapping=True)
    root = store.active_handle().root
    assert vision.save_to_map_dir(root) is not None
    _air_landmark(vision)
    vision.ingest_frame(frame, mapping=True)

    report = do("audit_landmarks")  # alias
    assert report["summary"]["seen_through"] == 1
    assert report["rows"][0]["label"] == "DC500"
    # The reply must survive the trip through a protobuf Struct.
    dict_to_struct(report)
    dict_to_struct(do("vision_status"))

    out = do("clear_landmarks")
    assert out["status"] == "cleared" and out["cleared"] >= 2
    assert out["saved_file_removed"] is True
    assert not landmarks_path(root).is_file()
    assert len(vision.landmarks) == 0
    # The occupancy map is not touched.
    slam._reset_live_slam.assert_not_called()  # noqa: SLF001
    assert store.get_active_map_name() == "floor1"

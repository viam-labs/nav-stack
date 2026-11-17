"""Vision supervisor: frame handling, person mask, and what it keeps per map."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import SlamConfig
from src.geom.conversions import Pose2D
from src.semantic.supervisor import VisionSupervisor
from src.semantic.types import (
    CameraIntrinsics,
    Detection,
    PatchToken,
    VisionFrame,
    VisionResult,
)
from src.semantic.worker import VisionWorker, WorkerConfig


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


def _vision(cfg: SlamConfig, pose: Pose2D, **kwargs) -> VisionSupervisor:
    args = {
        "get_pose": lambda: pose,
        "get_pose_at": lambda _t: pose,
        "get_anchor": lambda: (0, pose, {0: pose}),
    }
    args.update(kwargs)
    vision = VisionSupervisor(cfg=cfg, cameras={}, **args)
    vision._worker = VisionWorker(WorkerConfig(synthetic=True, input_short_side=112))  # noqa: SLF001
    return vision


def _person_result(frame: VisionFrame) -> VisionResult:
    """A person in the middle of the picture and a painting beside them."""
    person = Detection(label="person", conf=0.9, x0=40, y0=20, x1=120, y1=100)
    painting = Detection(label="painting", conf=0.9, x0=10, y0=10, x1=30, y1=30)
    return VisionResult(
        camera_name=frame.camera_name,
        stamp=frame.stamp,
        captured_at=frame.captured_at,
        detections=[person, painting],
        provider="test",
    )


def _grid_result(frame: VisionFrame) -> VisionResult:
    rng = np.random.default_rng(0)
    desc = rng.normal(size=(48, 32)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    patches = [
        PatchToken(
            row=r,
            col=c,
            u=10.0 + 20.0 * c,
            v=10.0 + 20.0 * r,
            descriptor=desc[r * 8 + c],
        )
        for r in range(6)
        for c in range(8)
    ]
    return VisionResult(
        camera_name=frame.camera_name, stamp=0.0, captured_at=0.0, grid_patches=patches
    )


def _store_keyframe(vision: VisionSupervisor) -> None:
    """One recovery keyframe, as a frame from the real model would leave."""
    vision._worker.cfg.synthetic = False  # noqa: SLF001
    frame = _frame()
    vision._keyframe_tick(frame, _grid_result(frame), allow_keyframe=True)  # noqa: SLF001
    assert vision.status()["recovery"]["keyframes"] == 1


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
    seen = {}

    def note(discs, captured_at):
        seen["n"] = len(discs)
        seen["t"] = captured_at
        seen["x"] = [round(d.x, 2) for d in discs]

    vision = _vision(
        _cfg(mode="mapping", vision_survey_only=False),
        Pose2D(0.0, 0.0, 0.0),
        note_paint_discs=note,
    )
    frame = _frame()
    frame.captured_at = 1.5
    vision._publish_paint_discs(frame, _person_result(frame))  # noqa: SLF001
    assert seen["n"] == 1
    assert seen["t"] == 1.5
    assert seen["x"][0] > 0.5
    assert vision.status()["paint_gate"]["last_discs"] == 1


def test_people_are_masked_while_the_map_is_built_and_not_otherwise():
    calls = []
    vision = _vision(
        _cfg(mode="mapping", vision_survey_only=False),
        Pose2D(0.0, 0.0, 0.0),
        note_paint_discs=lambda discs, _t: calls.append(len(discs)),
    )
    frame = _frame()
    vision._worker.process = lambda f: _person_result(f)  # noqa: SLF001
    vision.ingest_frame(frame, mapping=True)
    assert calls == [1]
    vision.ingest_frame(frame, mapping=False)
    assert calls == [1]


def test_worker_keeps_only_the_boxes_the_mask_uses():
    worker = VisionWorker(WorkerConfig(synthetic=True))
    assert worker._keep_detection_label("person") is True  # noqa: SLF001
    assert worker._keep_detection_label("Bicycle") is True  # noqa: SLF001
    assert worker._keep_detection_label("painting") is False  # noqa: SLF001


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


def test_tick_skips_a_stale_frame_and_masks_people_in_a_fresh_one():
    import asyncio

    calls = []
    vision = _vision(
        _cfg(),
        Pose2D(0.0, 0.0, 0.0),
        get_pose_at=lambda _t: None,
        note_paint_discs=lambda discs, _t: calls.append(len(discs)),
    )
    vision._cameras = {"depth-cam": object()}  # noqa: SLF001
    vision._worker.process = lambda f: _person_result(f)  # noqa: SLF001
    frames = [_aged_frame(1.5), _aged_frame(0.05)]

    async def grab(_cam, _lidar):
        return frames.pop(0)

    vision._grab_frame = grab  # noqa: SLF001
    asyncio.run(vision._tick_once())  # noqa: SLF001
    assert vision.status()["frames"] == 0
    assert vision.status()["capture_pose"]["stale_frames_dropped"] == 1
    assert calls == []
    asyncio.run(vision._tick_once())  # noqa: SLF001
    assert vision.status()["frames"] == 1
    assert calls == [1]


def test_save_writes_the_keyframes_and_reset_forgets_them(tmp_path: Path):
    vision = _vision(
        _cfg(mode="mapping", vision_survey_only=False), Pose2D(1.0, 2.0, 0.0)
    )
    assert vision.save_to_map_dir(tmp_path) is None  # nothing stored yet
    _store_keyframe(vision)
    path = vision.save_to_map_dir(tmp_path)
    assert path is not None and Path(path).name == "vision_keyframes.npz"
    assert Path(path).is_file()

    vision.reset()
    assert vision.status()["recovery"]["keyframes"] == 0
    # The saved file is the map store's to remove; a restart would load it.
    again = _vision(
        _cfg(mode="mapping", vision_survey_only=False), Pose2D(0.0, 0.0, 0.0)
    )
    assert again.load_from_map_dir(tmp_path) is True
    assert again.status()["recovery"]["keyframes"] == 1


def test_slam_reports_vision_status_and_clear_map_drops_the_keyframes(tmp_path: Path):
    """``vision_status`` through the SLAM DoCommand, and what ``clear_map`` clears."""
    import asyncio
    from unittest.mock import MagicMock

    from viam.utils import dict_to_struct

    from src.models.slam import SlamService
    from src.nav.stores.maps import MapStore

    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store  # noqa: SLF001
    slam._manager = MagicMock()  # noqa: SLF001
    slam._reset_live_slam = MagicMock()  # noqa: SLF001

    def do(name: str) -> dict:
        return asyncio.run(slam.do_command({"command": name}))

    assert do("vision_status") == {"enabled": False, "reason": "vision not running"}

    vision = _vision(
        _cfg(mode="mapping", vision_survey_only=False), Pose2D(0.0, 0.0, 0.0)
    )
    slam._vision = vision  # noqa: SLF001
    _store_keyframe(vision)
    root = store.active_handle().root
    saved = vision.save_to_map_dir(root)
    assert saved is not None and Path(saved).is_file()

    status = do("get_vision_status")  # alias
    assert status["recovery"]["keyframes"] == 1
    # The reply must survive the trip through a protobuf Struct.
    dict_to_struct(status)

    assert do("clear_map")["status"] == "cleared"
    assert vision.status()["recovery"]["keyframes"] == 0
    assert not Path(saved).is_file()

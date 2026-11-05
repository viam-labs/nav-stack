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

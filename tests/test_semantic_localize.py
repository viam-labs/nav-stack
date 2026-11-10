"""Unit tests for landmark pairing + gated pose writeback."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom.conversions import Pose2D
from src.semantic.landmarks import EmbeddingLandmark
from src.semantic.localize import (
    diagnose_landmark_match,
    estimate_landmark_fix,
    match_live_to_landmarks,
)
from src.semantic.patches import PatchObservation
from src.semantic.place import l2_normalize
from src.semantic.supervisor import VisionSupervisor
from src.semantic.types import CameraIntrinsics, LandmarkStatus, PatchToken, VisionFrame


def _obs(x: float, y: float, z: float, desc: np.ndarray, sigma: float = 0.02):
    return PatchObservation(
        token=PatchToken(row=0, col=0, u=0.0, v=0.0, descriptor=desc),
        point_base=np.array([x, y, z], dtype=float),
        normal_base=np.array([-1.0, 0.0, 0.0], dtype=float),
        sigma_m=sigma,
    )


def test_match_and_fit_recovers_pose():
    rng = np.random.default_rng(0)
    true = Pose2D(1.2, -0.4, math.radians(15.0))
    c, s = math.cos(true.theta), math.sin(true.theta)
    live = rng.uniform(0.5, 3.0, size=(20, 2))
    map_xy = np.column_stack(
        [
            c * live[:, 0] - s * live[:, 1] + true.x,
            s * live[:, 0] + c * live[:, 1] + true.y,
        ]
    )
    descs = [l2_normalize(rng.standard_normal(16).astype(np.float32)) for _ in range(20)]
    obs = [
        _obs(float(live[i, 0]), float(live[i, 1]), 1.0, descs[i]) for i in range(20)
    ]
    visible = []
    for i in range(20):
        lm = EmbeddingLandmark(
            landmark_id=i + 1,
            anchor_node=0,
            rel_xyz=np.array([map_xy[i, 0], map_xy[i, 1], 1.0]),
            normal_rel=np.array([-1.0, 0.0, 0.0]),
            descriptors=[descs[i]],
            observations=5,
            status=LandmarkStatus.CONFIRMED,
        )
        visible.append((lm, np.array([map_xy[i, 0], map_xy[i, 1], 1.0])))

    # Wrong seed pose (drifted).
    seed = Pose2D(1.0, -0.2, math.radians(5.0))
    live_xy, map_out, w = match_live_to_landmarks(
        obs, visible, robot_pose=seed, cosine_threshold=0.5, max_reproj_m=1.0
    )
    assert live_xy.shape[0] >= 8
    fix = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=seed,
        cosine_threshold=0.5,
        min_inliers=8,
        max_jump_m=2.0,
        max_reproj_m=1.0,
        rng=rng,
    )
    assert fix is not None
    assert fix.fit.pose.x == pytest.approx(true.x, abs=0.08)
    assert fix.fit.pose.y == pytest.approx(true.y, abs=0.08)
    assert fix.fit.pose.theta == pytest.approx(true.theta, abs=0.08)


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


def test_writeback_inactive_while_mapping():
    """Defaults enable writeback, but mapping mode must not apply corrections."""
    cfg = _cfg(mode="mapping", vision_survey_only=False, vision_pose_writeback=True)
    pose = Pose2D(0.0, 0.0, 0.0)

    def apply(p, *, score=None, force=False):
        return {"applied": True, "status": "pose_only"}

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
        apply_pose_correction=apply,
    )
    assert vision.status()["pose_writeback"] is True
    assert vision.status()["pose_writeback_active"] is False
    assert vision._writeback_enabled() is False  # noqa: SLF001


def test_writeback_applies_when_enabled(tmp_path):
    cfg = _cfg(
        mode="localizing",
        vision_survey_only=False,
        vision_pose_writeback=True,
        vision_pose_writeback_min_inliers=3,
        vision_pose_writeback_cooldown_s=0.0,
        landmark_fix_max_jump_m=2.0,
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    called = []

    def apply(p, *, score=None, force=False):
        called.append({"pose": p, "score": score, "force": force})
        return {"applied": True, "status": "pose_only", "shift_m": 0.2, "shift_deg": 2.0}

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
        apply_pose_correction=apply,
    )
    # Synthetic blocks writeback even if flags are on.
    assert vision._writeback_enabled() is False  # noqa: SLF001

    # Pretend real DINOv2 is loaded.
    vision._worker.cfg.synthetic = False  # noqa: SLF001
    vision._worker.stats["provider"] = "CPUExecutionProvider"
    assert vision._writeback_enabled() is True  # noqa: SLF001

    # Seed confirmed landmarks + matching observations at a drifted pose.
    rng = np.random.default_rng(1)
    true = Pose2D(0.3, 0.0, 0.0)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(10)]
    store = vision.landmarks
    store.require_descriptor_match = True
    for i in range(10):
        live = np.array([1.0 + 0.1 * i, 0.05 * (i % 3 - 1), 1.0])
        map_xy = np.array([true.x + live[0], true.y + live[1], 1.0])
        lm = EmbeddingLandmark(
            landmark_id=i + 1,
            anchor_node=0,
            rel_xyz=map_xy.copy(),
            normal_rel=np.array([-1.0, 0.0, 0.0]),
            descriptors=[descs[i]],
            observations=5,
            status=LandmarkStatus.CONFIRMED,
        )
        store._landmarks[lm.landmark_id] = lm  # noqa: SLF001
        store._index_add(lm, map_xy)  # noqa: SLF001

    obs = [
        _obs(1.0 + 0.1 * i, 0.05 * (i % 3 - 1), 1.0, descs[i]) for i in range(10)
    ]
    frame = VisionFrame(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=100, fy=100, cx=4, cy=4, width=8, height=8),
        mount_x=0.1,
        mount_z=0.4,
    )
    # Seed pose is origin; true landmark geometry implies +0.3 m.
    vision._try_pose_writeback(frame, obs, pose, {0: pose})  # noqa: SLF001
    assert called, vision.status()["writeback"]
    assert called[0]["force"] is False
    assert called[0]["score"] is not None
    assert abs(called[0]["pose"].x - 0.3) < 0.08
    assert vision.status()["writeback"]["applied"] >= 1


def test_diagnose_match_reports_weak_cosine_vs_geom():
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.2, 0.98, 0.0, 0.0], dtype=np.float32))
    assert float(np.dot(d0, d1)) < 0.30
    lm = EmbeddingLandmark(
        landmark_id=1,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 1.0]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptors=[d0],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    obs = [_obs(2.05, 0.02, 1.0, d1)]
    diag = diagnose_landmark_match(
        obs,
        [(lm, lm.map_xyz)],
        robot_pose=pose,
        cosine_threshold=0.55,
    )
    assert diag["n_geom_ok"] >= 1
    assert diag["n_cos_ok"] == 0
    assert diag["best_cos"] is not None
    assert diag["best_cos"] < 0.30


def test_detector_writeback_match_uses_soft_cosine():
    """Survey associates at ~0.30; writeback must not demand 0.55."""
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.32, 0.947, 0.0, 0.0], dtype=np.float32))
    cos = float(np.dot(d0, d1))
    assert 0.30 <= cos < 0.55
    lm = EmbeddingLandmark(
        landmark_id=1,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 1.0]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptors=[d0],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
    )
    live, _, _ = match_live_to_landmarks(
        [_obs(2.0, 0.0, 1.0, d1)],
        [(lm, lm.map_xyz)],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
    )
    assert live.shape[0] == 1

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
from src.semantic.types import (
    CameraIntrinsics,
    LandmarkStatus,
    PatchToken,
    VisionFrame,
    VisionResult,
)


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
    live_xy, map_out, w, _ids = match_live_to_landmarks(
        obs, visible, robot_pose=seed, cosine_threshold=0.5, max_reproj_m=1.0
    )
    assert live_xy.shape[0] >= 8
    attempt = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=seed,
        cosine_threshold=0.5,
        min_inliers=8,
        max_jump_m=2.0,
        max_reproj_m=1.0,
        rng=rng,
    )
    assert attempt.fix is not None, attempt
    fix = attempt.fix
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


def test_place_retrieval_skipped_while_tracking():
    cfg = _cfg(
        mode="localizing",
        vision_survey_only=False,
        vision_place_retrieval=True,
        vision_pose_writeback=True,
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    called = []

    def apply(p, *, score=None, force=False):
        called.append(p)
        return {"applied": True, "status": "pose_only"}

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
        apply_pose_correction=apply,
        is_localization_lost=lambda: False,
    )
    descs = l2_normalize(np.eye(8, dtype=np.float32))
    vision._place.add_keyframe(1, pose, descs)  # noqa: SLF001
    vision._place.add_keyframe(2, Pose2D(1.0, 0.0, 0.0), descs)  # noqa: SLF001
    assert vision._place_retrieval_enabled() is False  # noqa: SLF001
    vision._try_place_retrieval(descs, pose)  # noqa: SLF001
    assert vision.status()["place"]["last_retrieval"]["status"] == "skipped_tracking"
    assert called == []


def test_place_retrieval_applies_when_lost_large_jump():
    """Lost: relative jump cap dropped; strong score may teleport."""
    cfg = _cfg(
        mode="localizing",
        vision_survey_only=False,
        vision_place_retrieval=True,
        vision_pose_writeback=True,
        vision_place_cooldown_s=0.0,
        vision_place_min_score=0.55,
        vision_place_large_jump_m=0.40,
        vision_place_large_jump_min_score=0.75,
        vision_place_min_score_margin=0.08,
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    target = Pose2D(2.5, 0.0, 0.0)
    called = []

    def apply(p, *, score=None, force=False):
        called.append({"pose": p, "score": score, "force": force})
        return {"applied": True, "status": "pose_only", "shift_m": 2.5, "shift_deg": 0.0}

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
        apply_pose_correction=apply,
        is_localization_lost=lambda: True,
    )
    # Synthetic blocks apply-allowed even if flags are on.
    vision._worker.cfg.synthetic = False  # noqa: SLF001

    # Distinct means so unfitted mean-pool VLAD ranks the far keyframe first.
    rng = np.random.default_rng(7)
    near = l2_normalize(rng.standard_normal((16, 8)).astype(np.float32))
    far = l2_normalize(rng.standard_normal((16, 8)).astype(np.float32) + 3.0)
    vision._place.add_keyframe(1, pose, near)  # noqa: SLF001
    vision._place.add_keyframe(2, target, far)  # noqa: SLF001
    assert vision._place_retrieval_enabled() is True  # noqa: SLF001
    vision._try_place_retrieval(far, pose)  # noqa: SLF001
    last = vision.status()["place"]["last_retrieval"]
    assert last["status"] == "applied", last
    assert last["jump_m"] == pytest.approx(2.5, abs=0.01)
    assert len(called) == 1
    assert called[0]["force"] is False
    assert called[0]["pose"].x == pytest.approx(2.5)


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
    # Realistic image size so landmarks at z≈1 m project into-frame for
    # image_visible_set (tiny 8x8 FOV puts them above the image).
    frame = VisionFrame(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((240, 320, 3), dtype=np.uint8),
        depth_m=np.ones((240, 320), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=300, fy=300, cx=160, cy=120, width=320, height=240
        ),
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
    live, _, _, _ids = match_live_to_landmarks(
        [_obs(2.0, 0.0, 1.0, d1)],
        [(lm, lm.map_xyz)],
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
    )
    assert live.shape[0] == 1


def _confirmed_lm(lid: int, xyz, desc: np.ndarray) -> EmbeddingLandmark:
    xyz = np.asarray(xyz, dtype=float)
    return EmbeddingLandmark(
        landmark_id=lid,
        anchor_node=0,
        rel_xyz=xyz.copy(),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=xyz.copy(),
        descriptors=[desc],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
    )


def test_estimate_two_pairs_is_few_pairs_with_min_inliers_3():
    """SE(2) writeback requires ≥3 pairs; 2-point fits must not apply."""
    rng = np.random.default_rng(3)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(2)]
    visible = [
        (_confirmed_lm(1, [1.0, 0.0, 1.0], descs[0]), np.array([1.0, 0.0, 1.0])),
        (_confirmed_lm(2, [1.0, 0.4, 1.0], descs[1]), np.array([1.0, 0.4, 1.0])),
    ]
    obs = [_obs(1.0, 0.0, 1.0, descs[0]), _obs(1.0, 0.4, 1.0, descs[1])]
    attempt = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
        min_inliers=3,
    )
    assert attempt.fix is None
    assert attempt.reason == "few_pairs"
    assert attempt.pairs == 2


def test_estimate_landmark_fix_rejected_jump():
    rng = np.random.default_rng(4)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(3)]
    # Map landmarks imply robot at ~(+2, 0); seed pose at origin → large jump.
    true = Pose2D(2.0, 0.0, 0.0)
    visible = []
    obs = []
    for i, d in enumerate(descs):
        live = np.array([1.0 + 0.15 * i, 0.1 * (i - 1), 1.0])
        map_xy = np.array([true.x + live[0], true.y + live[1], 1.0])
        visible.append((_confirmed_lm(i + 1, map_xy, d), map_xy))
        obs.append(_obs(float(live[0]), float(live[1]), 1.0, d))
    attempt = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
        min_inliers=3,
        max_jump_m=0.5,
        max_jump_deg=10.0,
        max_reproj_m=2.5,
    )
    assert attempt.fix is None, attempt
    assert attempt.reason == "rejected_jump", attempt
    assert attempt.jump_m is not None and attempt.jump_m > 0.5


def test_estimate_accepts_a_12cm_fit_and_rejects_it_at_8cm():
    """Office landmarks confirm with ~12 cm of scatter; 8 cm never agreed."""
    rng = np.random.default_rng(6)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(4)]
    # Each stored point is 12 cm off the live one, alternating sign, so no
    # pose puts three of them inside 8 cm and the 15 cm gate still fits.
    visible = []
    obs = []
    for i, d in enumerate(descs):
        live = np.array([1.2 + 0.5 * i, 0.4 * (i - 1.5), 1.0])
        sign = 1.0 if i % 2 == 0 else -1.0
        stored = live + np.array([sign * 0.12, 0.0, 0.0])
        visible.append((_confirmed_lm(i + 1, stored, d), stored))
        obs.append(_obs(float(live[0]), float(live[1]), 1.0, d))
    tight = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
        min_inliers=3,
        residual_m=0.08,
        max_reproj_m=0.45,
    )
    loose = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
        min_inliers=3,
        residual_m=0.15,
        max_reproj_m=0.45,
    )
    assert tight.fix is None and tight.reason == "rejected_residual"
    assert loose.fix is not None
    assert loose.fix.n_pairs == 4


def test_estimate_landmark_fix_applies_with_three_pairs():
    rng = np.random.default_rng(5)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(3)]
    true = Pose2D(0.25, 0.0, 0.0)
    visible = []
    obs = []
    for i, d in enumerate(descs):
        live = np.array([1.0 + 0.2 * i, 0.15 * (i - 1), 1.0])
        map_xy = np.array([true.x + live[0], true.y + live[1], 1.0])
        visible.append((_confirmed_lm(i + 1, map_xy, d), map_xy))
        obs.append(_obs(float(live[0]), float(live[1]), 1.0, d))
    attempt = estimate_landmark_fix(
        obs,
        visible,
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        cosine_threshold=0.55,
        min_inliers=3,
        max_jump_m=1.5,
    )
    assert attempt.fix is not None, attempt
    assert attempt.fix.n_pairs >= 3
    assert abs(attempt.fix.fit.pose.x - 0.25) < 0.08


def test_writeback_dense_match_not_surveyed():
    """Writeback may use dense patches; survey with dense_fallback=0 must not grow map from them."""
    cfg = _cfg(
        mode="mapping",
        vision_survey_only=False,
        vision_prefer_detections=True,
        vision_dense_fallback_patches=0,
        vision_pose_writeback=False,
        vision_max_patches_per_frame=64,
    )
    pose = Pose2D(0.0, 0.0, 0.0)
    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: pose,
        get_pose_at=lambda _t: pose,
        get_anchor=lambda: (0, pose, {0: pose}),
    )
    # Synthetic frame with patches + one painting detection.
    from src.semantic.types import Detection, VisionResult

    h = w = 64
    rng = np.random.default_rng(9)
    patches = []
    for r in range(4):
        for c in range(4):
            patches.append(
                PatchToken(
                    row=r,
                    col=c,
                    u=8.0 + 12.0 * c,
                    v=8.0 + 12.0 * r,
                    descriptor=l2_normalize(rng.standard_normal(8).astype(np.float32)),
                )
            )
    # Depth so backproject keeps tokens in the wall band.
    depth = np.full((h, w), 1.2, dtype=np.float32)
    frame = VisionFrame(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=100, fy=100, cx=32, cy=32, width=w, height=h),
        mount_x=0.1,
        mount_z=0.4,
    )
    det = Detection(label="painting", conf=0.9, x0=5, y0=5, x1=20, y1=20)
    result = VisionResult(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        patches=patches,
        detections=[det],
        provider="test",
        infer_ms=1.0,
    )
    before = len(vision.landmarks)
    vision._survey(frame, result)  # noqa: SLF001
    # Detector-only survey: at most one new landmark from the painting box.
    after = len(vision.landmarks)
    assert after - before <= 1
    # Cap constant is wired for writeback pairing.
    from src.semantic import supervisor as sup

    assert int(sup._WRITEBACK_DENSE_MATCH_CAP) == 24


def _seed_frame() -> VisionFrame:
    return VisionFrame(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((240, 320, 3), dtype=np.uint8),
        depth_m=np.ones((240, 320), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=300, fy=300, cx=160, cy=120, width=320, height=240
        ),
        mount_x=0.1,
        mount_z=0.4,
    )


def _lost_supervisor():
    cfg = _cfg(
        mode="localizing",
        vision_survey_only=False,
        vision_place_retrieval=True,
        vision_pose_writeback=True,
        vision_pose_writeback_min_inliers=3,
        vision_pose_writeback_cooldown_s=0.0,
        vision_place_cooldown_s=0.0,
        landmark_fix_max_jump_m=1.5,
        landmark_fix_residual_m=0.15,
        vision_place_min_score=0.55,
        vision_place_min_score_margin=0.08,
    )
    lost = Pose2D(0.0, 0.0, 0.0)
    called = []

    def apply(p, *, score=None, force=False):
        called.append({"pose": p, "score": score, "force": force})
        return {
            "applied": True,
            "status": "pose_only",
            "shift_m": 1.0,
            "shift_deg": 0.0,
        }

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=lambda: lost,
        get_pose_at=lambda _t: lost,
        get_anchor=lambda: (0, lost, {0: lost}),
        apply_pose_correction=apply,
        is_localization_lost=lambda: True,
    )
    vision._worker.cfg.synthetic = False  # noqa: SLF001
    return vision, called, lost


def _confirm_at(store, landmark_id, true, live, desc):
    c, s = math.cos(true.theta), math.sin(true.theta)
    mxyz = np.array(
        [
            true.x + c * live[0] - s * live[1],
            true.y + s * live[0] + c * live[1],
            live[2],
        ]
    )
    lm = EmbeddingLandmark(
        landmark_id=landmark_id,
        anchor_node=0,
        rel_xyz=mxyz.copy(),
        map_xyz=mxyz.copy(),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        descriptors=[desc],
        observations=8,
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
    )
    store._landmarks[lm.landmark_id] = lm  # noqa: SLF001


def _place_result(desc: np.ndarray) -> VisionResult:
    patches = [
        PatchToken(row=0, col=i, u=0.0, v=0.0, descriptor=desc) for i in range(12)
    ]
    return VisionResult(
        camera_name="depth-cam", stamp=0.0, captured_at=0.0, patches=patches
    )


_SEED_LIVES = (
    (2.0, -0.30, 1.0),
    (2.4, 0.35, 1.0),
    (2.9, 0.05, 1.0),
)


def test_place_seed_applies_landmark_fit_when_vlads_tie():
    """A tied place does not teleport. Three landmarks from one guess do."""
    vision, called, lost = _lost_supervisor()
    true = Pose2D(4.0, 0.0, 0.0)
    rng = np.random.default_rng(3)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(3)]
    for i, live in enumerate(_SEED_LIVES):
        _confirm_at(vision.landmarks, i + 1, true, live, descs[i])
    place = l2_normalize(np.array([1.0, 0.2, 0.0, 0.0, 0, 0, 0, 0], dtype=np.float32))
    bank = np.stack([place] * 12, axis=0)
    # Two guesses of the real place, both offset past the 0.45 m writeback gate,
    # plus a far twin with the same picture and no landmarks.
    vision._place.add_keyframe(1, Pose2D(4.7, 0.0, 0.0), bank)  # noqa: SLF001
    vision._place.add_keyframe(2, Pose2D(3.6, 0.0, 0.0), bank)  # noqa: SLF001
    vision._place.add_keyframe(3, Pose2D(4.0, 7.0, 0.0), bank)  # noqa: SLF001
    obs = [_obs(*live, descs[i]) for i, live in enumerate(_SEED_LIVES)]
    result = _place_result(place)
    vision._try_place_retrieval(  # noqa: SLF001
        np.stack([place] * 12, axis=0), lost
    )
    assert vision.status()["place"]["last_retrieval"]["status"] == "rejected_ambiguous"
    assert called == []
    applied = vision._try_place_seeded_fix(  # noqa: SLF001
        _seed_frame(), result, obs, lost, {0: lost}
    )
    assert applied, vision.status()["place"]["last_seed"]
    assert len(called) == 1
    assert called[0]["force"] is False
    assert called[0]["pose"].x == pytest.approx(true.x, abs=0.08)
    assert called[0]["pose"].y == pytest.approx(true.y, abs=0.08)
    assert abs(called[0]["pose"].x - 4.7) > 0.3
    last = vision.status()["place"]["last_seed"]
    assert last["status"] == "applied"
    assert last["inliers"] >= 3
    assert vision.status()["place"]["seed_applied"] == 1


def test_place_seed_refuses_two_landmarks():
    vision, called, lost = _lost_supervisor()
    true = Pose2D(4.0, 0.0, 0.0)
    rng = np.random.default_rng(4)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(2)]
    for i, live in enumerate(_SEED_LIVES[:2]):
        _confirm_at(vision.landmarks, i + 1, true, live, descs[i])
    place = l2_normalize(np.ones(8, dtype=np.float32))
    bank = np.stack([place] * 12, axis=0)
    vision._place.add_keyframe(1, Pose2D(4.5, 0.0, 0.0), bank)  # noqa: SLF001
    obs = [_obs(*live, descs[i]) for i, live in enumerate(_SEED_LIVES[:2])]
    applied = vision._try_place_seeded_fix(  # noqa: SLF001
        _seed_frame(), _place_result(place), obs, lost, {0: lost}
    )
    assert applied is False
    assert called == []
    assert vision.status()["place"]["last_seed"]["status"] == "no_fit"


def test_place_seed_refuses_two_poses_that_both_fit():
    vision, called, lost = _lost_supervisor()
    pose_a = Pose2D(4.0, 0.0, 0.0)
    pose_b = Pose2D(4.0, 7.0, 0.0)
    rng = np.random.default_rng(5)
    descs = [l2_normalize(rng.standard_normal(8).astype(np.float32)) for _ in range(3)]
    for i, live in enumerate(_SEED_LIVES):
        _confirm_at(vision.landmarks, i + 1, pose_a, live, descs[i])
        _confirm_at(vision.landmarks, i + 4, pose_b, live, descs[i])
    place = l2_normalize(np.array([0.0, 1.0, 0.3, 0, 0, 0, 0, 0], dtype=np.float32))
    bank = np.stack([place] * 12, axis=0)
    vision._place.add_keyframe(1, pose_a, bank)  # noqa: SLF001
    vision._place.add_keyframe(2, pose_b, bank)  # noqa: SLF001
    obs = [_obs(*live, descs[i]) for i, live in enumerate(_SEED_LIVES)]
    applied = vision._try_place_seeded_fix(  # noqa: SLF001
        _seed_frame(), _place_result(place), obs, lost, {0: lost}
    )
    assert applied is False
    assert called == []
    last = vision.status()["place"]["last_seed"]
    assert last["status"] == "rejected_ambiguous_fits"
    assert len(last["fits"]) == 2


def test_place_seed_skipped_while_tracking():
    vision, called, lost = _lost_supervisor()
    vision._is_localization_lost = lambda: False  # noqa: SLF001
    place = l2_normalize(np.ones(8, dtype=np.float32))
    vision._place.add_keyframe(1, Pose2D(4.0, 0.0, 0.0), np.stack([place] * 8))  # noqa: SLF001
    applied = vision._try_place_seeded_fix(  # noqa: SLF001
        _seed_frame(), _place_result(place), [], lost, {0: lost}
    )
    assert applied is False
    assert called == []
    assert vision.status()["place"]["last_seed"]["status"] == "skipped_tracking"


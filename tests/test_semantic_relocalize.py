"""Camera keyframes and lost-pose recovery (camera proposes, lidar decides)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.geom import conversions as conv
from src.semantic.fit import fit_rigid_2d
from src.semantic.keyframes import (
    KeyframeIndex,
    keyframe_gap_reached,
    mutual_matches,
    token_points,
)
from src.semantic.relocalize import (
    AGREE_M,
    LidarMatch,
    fit_to_keyframe,
    recover_pose,
)
from src.semantic.types import CameraIntrinsics, VisionFrame


def _unit(rng: np.random.Generator, n: int, dim: int = 32) -> np.ndarray:
    x = rng.normal(size=(n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _scene(rng: np.random.Generator, n: int = 40) -> np.ndarray:
    """Patch points in front of the robot, base_link, spread in x, y and height."""
    return np.column_stack(
        [rng.uniform(1.0, 4.0, n), rng.uniform(-2.0, 2.0, n), rng.uniform(0.3, 1.8, n)]
    ).astype(np.float32)


def _seen_from(
    points: np.ndarray, keyframe: conv.Pose2D, live: conv.Pose2D
) -> np.ndarray:
    """The same world points, expressed in the live robot's base_link."""
    c, s = math.cos(keyframe.theta), math.sin(keyframe.theta)
    wx = keyframe.x + c * points[:, 0] - s * points[:, 1]
    wy = keyframe.y + s * points[:, 0] + c * points[:, 1]
    c, s = math.cos(live.theta), math.sin(live.theta)
    dx, dy = wx - live.x, wy - live.y
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, points[:, 2]]).astype(
        np.float32
    )


def test_mutual_matches_keeps_only_pairs_that_pick_each_other():
    rng = np.random.default_rng(0)
    kf = _unit(rng, 6)
    # Live patches 0..3 are keyframe patches 2..5; the last live patch is new.
    live = np.vstack([kf[2:6], _unit(rng, 1)])
    pairs = mutual_matches(live, kf)
    assert [tuple(p) for p in pairs[:4]] == [(0, 2), (1, 3), (2, 4), (3, 5)]
    assert mutual_matches(np.zeros((0, 32)), kf).shape == (0, 2)


def test_rank_reorders_the_pooled_shortlist_by_matched_patches():
    rng = np.random.default_rng(1)
    shared = _unit(rng, 30)
    index = KeyframeIndex()
    # Keyframe 7 holds the live frame's patches plus clutter that drags its
    # pooled descriptor away; keyframe 8 is nearly the live frame's average
    # repeated, so it pools closer but shares no single patch.
    index.add(
        7,
        conv.Pose2D(1.0, 2.0, 0.0),
        np.vstack([shared, 3.0 * _unit(rng, 30)]),
        np.zeros((60, 3)),
    )
    blur = np.tile(shared.mean(axis=0), (60, 1)) + 0.05 * _unit(rng, 60)
    index.add(8, conv.Pose2D(9.0, 9.0, 0.0), blur, np.zeros((60, 3)))
    pooled = np.stack(index._pooled) @ (
        shared.mean(axis=0) / np.linalg.norm(shared.mean(axis=0))
    )  # noqa: SLF001
    assert pooled[1] > pooled[0]
    hits = index.rank(shared, top_k=2)
    assert [h.keyframe_id for h in hits] == [7, 8]
    assert len(hits[0].pairs) == 30


def test_compressing_and_reloading_keeps_the_ranking(tmp_path: Path):
    rng = np.random.default_rng(2)
    index = KeyframeIndex(token_dim=8)
    frames = [_unit(rng, 20) for _ in range(6)]
    for i, tokens in enumerate(frames):
        index.add(i, conv.Pose2D(float(i), 0.0, 0.1 * i), tokens, _scene(rng, 20))
    assert not index.compressed
    path = index.save(tmp_path)  # saving compresses
    assert path is not None and index.compressed
    assert index.status_dict()["token_dim"] == 8
    loaded = KeyframeIndex.load(tmp_path)
    assert loaded is not None and len(loaded) == 6 and loaded.compressed
    for i, tokens in enumerate(frames):
        assert loaded.rank(tokens, top_k=1)[0].keyframe_id == i
    assert loaded.keyframe_poses[3] == conv.Pose2D(3.0, 0.0, 0.30000000000000004)
    # A keyframe added after loading is stored in the same reduced space.
    loaded.add(6, conv.Pose2D(6.0, 0.0, 0.0), _unit(rng, 20), _scene(rng, 20))
    assert loaded.status_dict()["token_dim"] == 8


def test_saving_an_empty_index_removes_the_file(tmp_path: Path):
    rng = np.random.default_rng(3)
    index = KeyframeIndex()
    index.add(0, conv.Pose2D(0.0, 0.0, 0.0), _unit(rng, 5), _scene(rng, 5))
    assert index.save(tmp_path) is not None
    assert KeyframeIndex().save(tmp_path) is None
    assert KeyframeIndex.load(tmp_path) is None


def test_keyframe_gap_is_distance_or_turning():
    last = conv.Pose2D(0.0, 0.0, 0.0)
    assert keyframe_gap_reached(
        conv.Pose2D(0.1, 0.0, 0.0), None, gap_m=0.75, gap_deg=25.0
    )
    assert not keyframe_gap_reached(
        conv.Pose2D(0.5, 0.0, 0.2), last, gap_m=0.75, gap_deg=25.0
    )
    assert keyframe_gap_reached(
        conv.Pose2D(0.8, 0.0, 0.0), last, gap_m=0.75, gap_deg=25.0
    )
    assert keyframe_gap_reached(
        conv.Pose2D(0.0, 0.0, 0.5), last, gap_m=0.75, gap_deg=25.0
    )


def test_patch_points_need_depth():
    k = CameraIntrinsics(fx=200.0, fy=200.0, cx=64.0, cy=48.0, width=128, height=96)
    depth = np.zeros((96, 128), dtype=np.float32)
    depth[:, :64] = 2.0  # left half is a wall 2 m away; right half has no depth
    frame = VisionFrame(
        camera_name="cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((96, 128, 3), np.uint8),
        depth_m=depth,
        intrinsics=k,
        mount_z=1.0,
    )
    points = token_points(frame, np.array([[30.0, 48.0], [100.0, 48.0]]))
    assert abs(points[0, 0] - 2.0) < 1e-3 and abs(points[0, 2] - 1.0) < 1e-3
    assert np.isnan(points[1]).all()


def test_camera_fit_recovers_the_pose_and_ignores_matches_at_the_wrong_height():
    rng = np.random.default_rng(4)
    keyframe = conv.Pose2D(3.0, -1.0, 0.4)
    live = conv.Pose2D(3.5, -0.6, 0.7)
    stored = _scene(rng)
    seen = _seen_from(stored, keyframe, live)
    pairs = np.column_stack([np.arange(40), np.arange(40)])
    fit = fit_to_keyframe(seen, stored, pairs, keyframe)
    assert fit is not None and fit.inliers == 40
    assert abs(fit.pose.x - live.x) < 1e-3 and abs(fit.pose.y - live.y) < 1e-3
    assert abs(conv.normalize_angle(fit.pose.theta - live.theta)) < 1e-3

    # Too few matches with depth on both sides is no fit.
    seen_few = seen.copy()
    seen_few[5:] = np.nan
    assert fit_to_keyframe(seen_few, stored, pairs, keyframe) is None
    # A floor patch matched to a shelf is not the same surface.
    lifted = seen.copy()
    lifted[:, 2] += 1.0
    assert fit_to_keyframe(lifted, stored, pairs, keyframe) is None


class _World:
    """Two keyframes; the live frame shows the first one's patches."""

    def __init__(self) -> None:
        rng = np.random.default_rng(5)
        self.live_pose = conv.Pose2D(2.3, 1.2, 0.25)
        self.home = conv.Pose2D(2.0, 1.0, 0.1)
        self.twin = conv.Pose2D(9.5, 1.0, 0.1)
        self.tokens = _unit(rng, 40)
        stored = _scene(rng)
        self.points = _seen_from(stored, self.home, self.live_pose)
        self.index = KeyframeIndex()
        self.index.add(1, self.home, self.tokens, stored)
        # The twin shares half the patches, with no depth: no camera pose.
        twin_tokens = np.vstack([self.tokens[:20], _unit(rng, 20)])
        self.index.add(2, self.twin, twin_tokens, np.full((40, 3), np.nan, np.float32))


def _lidar(at_home: LidarMatch, at_twin: LidarMatch):
    def search(hint: conv.Pose2D, radius_m: float, yaw_window_deg: float):
        return at_home if hint.x < 5.0 else at_twin

    return search


def test_recovery_is_accepted_when_camera_and_lidar_agree_and_nothing_rivals_it():
    w = _World()
    home = LidarMatch(conv.Pose2D(2.32, 1.21, 0.26), score=0.62, ray_mae_m=0.3)
    twin = LidarMatch(w.twin, score=0.20, ray_mae_m=1.4)
    out = recover_pose(w.index, w.tokens, w.points, _lidar(home, twin))
    assert out.accepted and out.pose == home.pose
    assert out.winner.keyframe_id == 1 and out.winner.camera.inliers == 40
    assert out.as_dict()["status"] == "accepted"


def test_recovery_is_refused_when_the_lidar_lands_away_from_the_camera_pose():
    w = _World()
    slid = conv.Pose2D(w.live_pose.x + AGREE_M + 0.15, w.live_pose.y, w.live_pose.theta)
    home = LidarMatch(slid, score=0.62, ray_mae_m=0.3)
    twin = LidarMatch(w.twin, score=0.20, ray_mae_m=1.4)
    out = recover_pose(w.index, w.tokens, w.points, _lidar(home, twin))
    assert not out.accepted and out.status == "no_agreement" and out.pose is None


def test_recovery_is_refused_when_the_scan_match_is_too_weak():
    w = _World()
    home = LidarMatch(w.live_pose, score=0.20, ray_mae_m=0.3)
    twin = LidarMatch(w.twin, score=0.05, ray_mae_m=1.4)
    out = recover_pose(w.index, w.tokens, w.points, _lidar(home, twin))
    assert out.status == "no_agreement"


def test_recovery_is_refused_when_another_place_explains_the_scan_nearly_as_well():
    w = _World()
    home = LidarMatch(w.live_pose, score=0.60, ray_mae_m=0.3)
    # The twin has no camera pose, so it cannot win, but its scan match is a rival.
    twin = LidarMatch(w.twin, score=0.55, ray_mae_m=0.4)
    out = recover_pose(w.index, w.tokens, w.points, _lidar(home, twin))
    assert out.status == "rival_place" and not out.accepted and out.pose is None
    assert out.winner.keyframe_id == 1


def test_recovery_with_no_keyframes_says_so():
    out = recover_pose(
        KeyframeIndex(), np.zeros((4, 32)), np.zeros((4, 3)), lambda *a: None
    )
    assert out.status == "no_keyframes" and not out.accepted


def test_fit_rigid_recovers_pose_with_outliers():
    rng = np.random.default_rng(1)
    # True pose: translate (1, 0.5), yaw 20 deg.
    yaw = np.deg2rad(20.0)
    c, s = np.cos(yaw), np.sin(yaw)
    live = rng.uniform(-2, 2, size=(40, 2))
    map_xy = np.column_stack(
        [c * live[:, 0] - s * live[:, 1] + 1.0, s * live[:, 0] + c * live[:, 1] + 0.5]
    )
    # 30% outliers.
    n_out = 12
    map_xy[:n_out] += rng.normal(0, 1.5, size=(n_out, 2))
    result = fit_rigid_2d(
        live,
        map_xy,
        min_inliers=8,
        residual_m=0.1,
        ransac_iters=80,
        rng=rng,
    )
    assert result is not None
    assert result.pose.x == pytest.approx(1.0, abs=0.08)
    assert result.pose.y == pytest.approx(0.5, abs=0.08)
    assert result.pose.theta == pytest.approx(yaw, abs=0.08)
    assert int(result.inliers.sum()) >= 20


# --- Supervisor: keyframes while mapping, recovery while lost -----------------

from src.config import SlamConfig  # noqa: E402
from src.semantic.supervisor import VisionSupervisor  # noqa: E402
from src.semantic.types import PatchToken, VisionResult  # noqa: E402


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
        "vision_recovery_cooldown_s": 0.0,
    }
    d.update(extra)
    return SlamConfig.from_dict(d)


def _camera_frame() -> VisionFrame:
    h, w = 120, 160
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[:, w // 2 :] = 3.0  # a step, so the points are not one straight wall
    k = CameraIntrinsics(fx=80.0, fy=80.0, cx=80.0, cy=60.0, width=w, height=h)
    return VisionFrame(
        camera_name="depth-cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((h, w, 3), np.uint8),
        depth_m=depth,
        intrinsics=k,
        mount_x=0.1,
        mount_z=0.6,
    )


def _grid_result(seed: int = 0) -> VisionResult:
    rng = np.random.default_rng(seed)
    desc = _unit(rng, 48)
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
        camera_name="depth-cam", stamp=0.0, captured_at=0.0, grid_patches=patches
    )


class _Rig:
    """A supervisor with fake SLAM hooks. The worker is told it has a real model."""

    def __init__(self, mode: str, **cfg) -> None:
        self.pose = conv.Pose2D(4.0, 2.0, 0.5)
        self.node_pose = conv.Pose2D(3.5, 2.0, 0.5)
        self.lost = True
        self.stopped = True
        self.searches: list = []
        self.applied: list = []
        self.lidar_offset = (0.0, 0.0)
        self.vision = VisionSupervisor(
            cfg=_cfg(mode=mode, **cfg),
            cameras={},
            get_pose=lambda: self.pose,
            get_pose_at=lambda _t: self.pose,
            get_anchor=lambda: (3, self.node_pose, {3: self.node_pose}),
            is_localization_lost=lambda: self.lost,
            lidar_search=self._search,
            apply_relocalization=self._apply,
            is_base_stopped=lambda: self.stopped,
        )
        self.vision._worker.cfg.synthetic = False  # noqa: SLF001

    def _search(self, hint, radius_m, yaw_window_deg):
        self.searches.append((hint, radius_m, yaw_window_deg))
        dx, dy = self.lidar_offset
        return LidarMatch(conv.Pose2D(hint.x + dx, hint.y + dy, hint.theta), 0.6, 0.3)

    def _apply(self, pose) -> bool:
        self.applied.append(pose)
        return True

    def tick(self, mapping: bool, seed: int = 0) -> None:
        self.vision._keyframe_tick(  # noqa: SLF001
            _camera_frame(), _grid_result(seed), allow_keyframe=mapping, pose=self.pose
        )


def _mapped(tmp_path: Path) -> conv.Pose2D:
    """Map one keyframe and save it; returns where it was taken."""
    rig = _Rig("mapping")
    rig.tick(mapping=True)
    assert rig.vision.status()["recovery"]["keyframes"] == 1
    rig.vision.save_to_map_dir(tmp_path)
    return rig.pose


def test_mapping_stores_a_keyframe_per_gap_and_never_searches():
    rig = _Rig("mapping")
    rig.tick(mapping=True)
    rig.tick(mapping=True)  # has not moved
    assert rig.vision.status()["recovery"]["keyframes"] == 1
    rig.pose = conv.Pose2D(5.0, 2.0, 0.5)
    rig.tick(mapping=True, seed=1)
    status = rig.vision.status()["recovery"]
    assert status["keyframes"] == 2 and status["active"] is False
    assert rig.searches == [] and rig.applied == []


def test_a_keyframe_follows_its_pose_graph_node():
    rig = _Rig("mapping")
    rig.tick(mapping=True)
    moved = conv.Pose2D(3.5, 3.0, 0.5)  # loop closure shifts node 3 by +1 m in y
    rig.vision.follow_anchor_motion({3: rig.node_pose}, {3: moved})
    pose = rig.vision._keyframes.keyframe_poses[0]  # noqa: SLF001
    assert abs(pose.x - 4.0) < 1e-9 and abs(pose.y - 3.0) < 1e-9


def test_a_lost_still_robot_is_placed_where_camera_and_lidar_agree(tmp_path: Path):
    taken_at = _mapped(tmp_path)
    rig = _Rig("localizing")
    assert rig.vision.load_from_map_dir(tmp_path)
    rig.pose = conv.Pose2D(-9.0, -9.0, 2.0)  # the lost pose is not consulted
    rig.tick(mapping=False)
    assert len(rig.applied) == 1
    placed = rig.applied[0]
    assert abs(placed.x - taken_at.x) < 0.02 and abs(placed.y - taken_at.y) < 0.02
    status = rig.vision.status()["recovery"]
    assert status["applied"] == 1 and status["last"]["status"] == "accepted"
    assert status["last"]["applied"] is True


def test_recovery_waits_while_the_base_is_moving(tmp_path: Path):
    _mapped(tmp_path)
    rig = _Rig("localizing")
    rig.vision.load_from_map_dir(tmp_path)
    rig.stopped = False
    rig.tick(mapping=False)
    assert rig.searches == [] and rig.applied == []
    assert rig.vision.status()["recovery"]["last"] == {"status": "skipped_moving"}


def test_recovery_does_not_place_a_pose_the_lidar_puts_elsewhere(tmp_path: Path):
    _mapped(tmp_path)
    rig = _Rig("localizing")
    rig.vision.load_from_map_dir(tmp_path)
    rig.lidar_offset = (0.5, 0.0)
    rig.tick(mapping=False)
    assert rig.searches and rig.applied == []
    assert rig.vision.status()["recovery"]["last"]["status"] == "no_agreement"


def test_vision_does_no_work_while_localizing_with_a_tracking_pose(tmp_path: Path):
    _mapped(tmp_path)
    rig = _Rig("localizing")
    rig.vision.load_from_map_dir(tmp_path)
    rig.lost = False
    assert rig.vision._tick_needed() is False  # noqa: SLF001
    rig.tick(mapping=False)
    assert rig.searches == []
    rig.lost = True
    assert rig.vision._tick_needed() is True  # noqa: SLF001
    assert _Rig("mapping").vision._tick_needed() is True  # noqa: SLF001


def test_recovery_needs_the_real_model_and_can_be_switched_off(tmp_path: Path):
    rig = _Rig("mapping")
    rig.vision._worker.cfg.synthetic = True  # noqa: SLF001
    rig.tick(mapping=True)
    assert rig.vision.status()["recovery"]["keyframes"] == 0
    off = _Rig("mapping", vision_recovery=False)
    off.tick(mapping=True)
    assert off.vision.status()["recovery"]["enabled"] is False
    assert off.vision.status()["recovery"]["keyframes"] == 0

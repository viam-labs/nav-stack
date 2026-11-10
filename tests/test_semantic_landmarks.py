"""Unit tests for embedding landmark map and rigid fit."""

from __future__ import annotations

import numpy as np
import pytest

from src.geom.conversions import Pose2D
from src.semantic.fit import fit_rigid_2d
from src.semantic.landmarks import EmbeddingLandmark, EmbeddingLandmarkMap
from src.semantic.place import PlaceIndex, fit_pca, fit_vocabulary, l2_normalize
from src.semantic.types import CameraIntrinsics, LandmarkStatus


def test_spatial_merge_without_descriptor_match():
    """Synthetic mode: nearby patches collapse by distance alone."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        cell_m=0.30,
        require_descriptor_match=False,
        promote_min_obs=99,  # stay candidate
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    # Two unrelated descriptors within 10 cm → one landmark.
    for i, desc in enumerate(
        [
            l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            l2_normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        ]
    ):
        store.observe(
            map_xyz=np.array([2.0, 0.05 * i, 1.0], dtype=float),
            descriptor=desc,
            normal_map=normal,
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=(0.0, 0.0),
        )
    assert len(store) == 1
    assert store.all_landmarks()[0].observations == 2


def test_landmark_promotion_and_demotion():
    store = EmbeddingLandmarkMap(
        promote_min_obs=3,
        promote_min_views=2,
        promote_min_view_gap_m=0.5,
        promote_max_std_m=0.12,
        demote_offset_m=0.15,
        demote_passes=3,
    )
    anchor = Pose2D(1.0, 2.0, 0.0)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    # Three views of the same wall point.
    views = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)]
    lm = None
    for i, view in enumerate(views):
        # Small noise within promote std.
        noise = np.array([0.01 * (i - 1), 0.0, 0.0])
        map_xyz = np.array([3.0, 2.0, 1.0], dtype=float) + noise
        lm = store.observe(
            map_xyz=map_xyz,
            descriptor=desc,
            normal_map=normal,
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=view,
        )
    assert lm is not None
    assert lm.status == LandmarkStatus.CONFIRMED
    assert len(store) == 1

    # Three bad re-observations → moved (same anchor pose for expected point).
    for _ in range(3):
        lm = store.observe(
            map_xyz=np.array([3.0, 2.0, 1.0], dtype=float) + np.array([0.4, 0.0, 0.0]),
            descriptor=desc,
            normal_map=normal,
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=(0.0, 0.0, 0.0),
        )
    assert lm.status == LandmarkStatus.MOVED


def test_promote_tolerates_depth_noise_above_5cm():
    """RealSense multi-view std often sits ~6–10 cm; 5 cm was too strict."""
    store = EmbeddingLandmarkMap(
        promote_min_obs=3,
        promote_min_views=2,
        promote_min_view_gap_m=0.5,
        promote_max_std_m=0.12,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    # Multi-axis noise → mean axis-std ~7 cm (above old 5 cm gate, under 12 cm).
    noises = (
        np.array([-0.10, 0.10, 0.05]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.10, -0.10, -0.05]),
    )
    views = ((0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0))
    lm = None
    for noise, view in zip(noises, views):
        lm = store.observe(
            map_xyz=np.array([2.0, 0.0, 1.0], dtype=float) + noise,
            descriptor=desc,
            normal_map=normal,
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=view,
        )
    assert lm is not None
    assert lm.position_std_m > 0.05
    assert lm.position_std_m <= 0.12
    assert lm.status == LandmarkStatus.CONFIRMED


def test_map_xyz_stable_when_virtual_anchor_pose_spins():
    """Overlay/association must not reinterpret rel under a new live yaw."""
    import math

    store = EmbeddingLandmarkMap(require_descriptor_match=True, promote_min_obs=99)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    pose0 = Pose2D(0.0, 0.0, 0.0)
    # Feature 2 m ahead of pose0 → map (2, 0, 1).
    lm = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=desc,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=pose0,
        view_xy=(0.0, 0.0, 0.0),
    )
    assert lm.map_xyz is not None
    np.testing.assert_allclose(lm.map_xyz, [2.0, 0.0, 1.0], atol=1e-6)
    # Live/virtual anchor table now claims node 0 is a spun pose — old bug
    # would rotate the landmark with it.
    spun = Pose2D(0.0, 0.0, math.radians(40.0))
    resolved = EmbeddingLandmarkMap.resolved_map_xyz(lm, {0: spun})
    np.testing.assert_allclose(resolved, [2.0, 0.0, 1.0], atol=1e-6)
    # Re-observe the same map point after the spin; must associate, not mint.
    lm2 = store.observe(
        map_xyz=np.array([2.02, 0.0, 1.0]),
        descriptor=desc,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=spun,
        view_xy=(0.0, 0.0, math.radians(40.0)),
    )
    assert lm2.landmark_id == lm.landmark_id
    assert len(store) == 1
    # Second observe must not crawl the locked map_xyz toward the new sample.
    np.testing.assert_allclose(lm.map_xyz, [2.0, 0.0, 1.0], atol=1e-6)


def test_promote_on_spin_in_place_yaw_gap():
    """Pure rotation used to never promote (view gap was XY-only)."""
    import math

    store = EmbeddingLandmarkMap(
        promote_min_obs=3,
        promote_min_views=2,
        promote_min_view_gap_m=0.5,  # XY alone would fail
        promote_min_view_gap_deg=25.0,
        promote_max_std_m=0.12,
        require_descriptor_match=True,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    lm = None
    for i, yaw_deg in enumerate((0.0, 15.0, 35.0)):
        lm = store.observe(
            map_xyz=np.array([2.0, 0.01 * (i - 1), 1.0], dtype=float),
            descriptor=desc,
            normal_map=normal,
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=(0.0, 0.0, math.radians(yaw_deg)),
        )
    assert lm is not None
    assert lm.status == LandmarkStatus.CONFIRMED


def test_confirmed_soft_match_banks_new_viewpoint_descriptor():
    """Drive-by: confirmed should re-associate under angle change when close."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        cosine_threshold=0.55,
        candidate_cosine_threshold=0.35,
        confirmed_soft_match_frac=0.7,
        max_descriptors=6,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # Cosine ≈ 0.45 — below confirmed gate, above soft gate.
    d1 = l2_normalize(np.array([0.45, 0.893, 0.0, 0.0], dtype=np.float32))
    assert 0.35 <= float(np.dot(d0, d1)) < 0.55
    lm = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
    )
    lm.status = LandmarkStatus.CONFIRMED
    lm.map_xyz = np.array([2.0, 0.0, 1.0])
    lm2 = store.observe(
        map_xyz=np.array([2.05, 0.02, 1.0]),
        descriptor=d1,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.4, 0.0, 0.3),
    )
    assert lm2.landmark_id == lm.landmark_id
    assert len(store) == 1
    assert len(lm.descriptors) == 2
    assert store.status_dict()["assoc_soft_hits"] >= 1


def test_detector_reassociates_across_angle_and_depth_jitter():
    """YOLO-pooled centers wobble; must not mint a parallel ID each pass."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        candidate_cosine_threshold=0.35,
        cosine_threshold=0.55,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.40, 0.9165, 0.0, 0.0], dtype=np.float32))
    assert 0.30 <= float(np.dot(d0, d1)) < 0.55
    lm = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
        from_detector=True,
    )
    lm2 = store.observe(
        map_xyz=np.array([2.35, 0.08, 1.05]),  # outside old 0.25 m gate
        descriptor=d1,
        normal_map=normal,
        anchor_node=1,
        anchor_pose=Pose2D(0.4, 0.0, 0.2),
        view_xy=(0.4, 0.0, 0.2),
        from_detector=True,
    )
    assert lm2.landmark_id == lm.landmark_id
    assert len(store) == 1
    assert lm.from_detector is True


def test_confirmed_footprint_overlap_absorbs_instead_of_parallel_candidate():
    """Weak second box whose footprint overlaps confirmed must not mint a twin."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        candidate_cosine_threshold=0.35,
        cosine_threshold=0.55,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # Nearly orthogonal — fails every cosine gate.
    d1 = l2_normalize(np.array([0.05, 0.9987, 0.0, 0.0], dtype=np.float32))
    assert float(np.dot(d0, d1)) < 0.20
    lm = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
        from_detector=True,
        extent_m=0.30,
    )
    lm.status = LandmarkStatus.CONFIRMED
    lm.map_xyz = np.array([2.0, 0.0, 1.0])
    before_desc = len(lm.descriptors)
    # 18 cm offset: each center lies inside the other's 30 cm footprint.
    lm2 = store.observe(
        map_xyz=np.array([2.18, 0.05, 1.02]),
        descriptor=d1,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.3, 0.0, 0.2),
        from_detector=True,
        extent_m=0.28,
    )
    assert lm2.landmark_id == lm.landmark_id
    assert len(store) == 1
    assert len(lm.descriptors) == before_desc  # weak crop not banked
    assert store.status_dict()["assoc_overlap_absorbs"] >= 1


def test_near_but_non_overlapping_footprints_still_mint():
    """Centers 35 cm apart with 15 cm footprints: near, but not overlapping."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    lm = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
        from_detector=True,
        extent_m=0.15,
    )
    lm.status = LandmarkStatus.CONFIRMED
    lm.map_xyz = np.array([2.0, 0.0, 1.0])
    lm2 = store.observe(
        map_xyz=np.array([2.35, 0.0, 1.0]),
        descriptor=d1,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
        from_detector=True,
        extent_m=0.15,
    )
    assert lm2.landmark_id != lm.landmark_id
    assert len(store) == 2


def test_merge_overlapping_confirmed_detector_twins():
    """DC1 + DC12 stacked on one object should collapse to a single ID."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    a = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
        from_detector=True,
        extent_m=0.30,
    )
    # Force a parallel twin (bypass observe absorb) to simulate two promotions.
    b = EmbeddingLandmark(
        landmark_id=store._next_id,  # noqa: SLF001
        anchor_node=0,
        rel_xyz=np.array([2.05, 0.02, 1.0]),
        normal_rel=normal.copy(),
        map_xyz=np.array([2.05, 0.02, 1.0]),
        descriptors=[d1.copy()],
        observations=5,
        view_poses_xy=[(0.0, 0.0, 0.0), (0.4, 0.0, 0.3)],
        status=LandmarkStatus.CONFIRMED,
        from_detector=True,
        extent_m=0.28,
    )
    store._next_id += 1  # noqa: SLF001
    store._landmarks[b.landmark_id] = b  # noqa: SLF001
    store._index_add(b, b.map_xyz)  # noqa: SLF001
    a.status = LandmarkStatus.CONFIRMED
    a.map_xyz = np.array([2.0, 0.0, 1.0])
    a.observations = 8
    assert len(store) == 2
    n = store.merge_overlapping_detector_landmarks()
    assert n >= 1
    assert len(store) == 1
    kept = next(iter(store.all_landmarks()))
    assert kept.landmark_id == a.landmark_id  # more obs wins
    assert len(kept.descriptors) >= 2
    assert store.status_dict()["overlap_merges"] >= 1


def test_detector_revive_moved_instead_of_minting():
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    lm = store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=desc,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
        from_detector=True,
    )
    lm.status = LandmarkStatus.MOVED
    lm2 = store.observe(
        map_xyz=np.array([2.05, 0.02, 1.0]),
        descriptor=desc,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.2, 0.0, 0.0),
        from_detector=True,
    )
    assert lm2.landmark_id == lm.landmark_id
    assert lm.status == LandmarkStatus.CANDIDATE
    assert len(store) == 1


def test_candidate_rejects_orthogonal_nearby_descriptor():
    """Wrong surface nearby must not steal the same candidate ID (overlay jump)."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        candidate_cosine_threshold=0.35,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))  # orthogonal
    store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
    )
    store.observe(
        map_xyz=np.array([2.05, 0.0, 1.0]),
        descriptor=d1,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.3, 0.0, 0.0),
    )
    assert len(store) == 2
    assert all(lm.observations == 1 for lm in store.all_landmarks())


def test_candidate_soft_cosine_allows_mild_descriptor_drift():
    """Same feature under approach: cosine can drop below confirmed gate."""
    store = EmbeddingLandmarkMap(
        associate_radius_m=0.25,
        require_descriptor_match=True,
        cosine_threshold=0.55,
        candidate_cosine_threshold=0.35,
        promote_min_obs=99,
    )
    anchor = Pose2D(0.0, 0.0, 0.0)
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    # Cosine ≈ 0.45 — fails confirmed gate, passes candidate gate.
    d0 = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    d1 = l2_normalize(np.array([0.45, 0.893, 0.0, 0.0], dtype=np.float32))
    assert 0.35 <= float(np.dot(d0, d1)) < 0.55
    store.observe(
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptor=d0,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.0, 0.0, 0.0),
    )
    lm = store.observe(
        map_xyz=np.array([2.05, 0.0, 1.0]),
        descriptor=d1,
        normal_map=normal,
        anchor_node=0,
        anchor_pose=anchor,
        view_xy=(0.3, 0.0, 0.0),
    )
    assert len(store) == 1
    assert lm.observations == 2
    assert lm.status == LandmarkStatus.CANDIDATE


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


def test_place_index_query_top_k():
    rng = np.random.default_rng(0)
    descs = l2_normalize(rng.standard_normal((200, 32)).astype(np.float32))
    comps, mean = fit_pca(descs, out_dim=16)
    vocab = fit_vocabulary(descs @ comps.T, n_clusters=8, seed=0)
    index = PlaceIndex(pca_components=comps, pca_mean=mean, vocabulary=vocab)
    # Two keyframes with distinct descriptor sets.
    a = descs[:50]
    b = descs[50:100]
    index.add_keyframe(0, Pose2D(0.0, 0.0, 0.0), a)
    index.add_keyframe(1, Pose2D(5.0, 0.0, 0.0), b)
    hits = index.query(a[:20], top_k=2)
    assert hits
    assert hits[0][0] == 0


def test_image_visible_includes_facing_away():
    """Overlay-style projection keeps landmarks that facing-cull drops."""
    store = EmbeddingLandmarkMap()
    lm = EmbeddingLandmark(
        landmark_id=1,
        anchor_node=0,
        rel_xyz=np.array([2.0, 0.0, 1.0]),
        # Faces +x (away from robot at origin looking +x) → fails facing gate.
        normal_rel=np.array([1.0, 0.0, 0.0]),
        map_xyz=np.array([2.0, 0.0, 1.0]),
        descriptors=[l2_normalize(np.ones(8, dtype=np.float32))],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
    )
    store._landmarks[1] = lm  # noqa: SLF001
    store._index_add(lm, lm.map_xyz)  # noqa: SLF001
    pose = Pose2D(0.0, 0.0, 0.0)
    nodes = {0: Pose2D(0.0, 0.0, 0.0)}
    K = CameraIntrinsics(fx=200, fy=200, cx=160, cy=120, width=320, height=240)
    frustum = store.visible_set(
        pose, nodes, camera_xy=(0.1, 0.0), camera_yaw=0.0, max_range_m=4.0
    )
    image = store.image_visible_set(
        pose,
        nodes,
        intrinsics=K,
        mount_x=0.1,
        mount_z=0.4,
        max_range_m=4.0,
    )
    assert frustum == []
    assert len(image) == 1
    assert image[0][0].landmark_id == 1


def test_image_visible_excludes_behind_camera():
    store = EmbeddingLandmarkMap()
    lm = EmbeddingLandmark(
        landmark_id=2,
        anchor_node=0,
        rel_xyz=np.array([-2.0, 0.0, 1.0]),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array([-2.0, 0.0, 1.0]),
        descriptors=[l2_normalize(np.ones(8, dtype=np.float32))],
        observations=5,
        status=LandmarkStatus.CONFIRMED,
    )
    store._landmarks[2] = lm  # noqa: SLF001
    store._index_add(lm, lm.map_xyz)  # noqa: SLF001
    pose = Pose2D(0.0, 0.0, 0.0)
    nodes = {0: Pose2D(0.0, 0.0, 0.0)}
    K = CameraIntrinsics(fx=200, fy=200, cx=160, cy=120, width=320, height=240)
    image = store.image_visible_set(
        pose,
        nodes,
        intrinsics=K,
        mount_x=0.1,
        mount_z=0.4,
        max_range_m=4.0,
    )
    assert image == []


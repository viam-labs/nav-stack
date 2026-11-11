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


def _observe_detector(store, *, n, views, offsets):
    anchor = Pose2D(0.0, 0.0, 0.0)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    lm = None
    for i in range(n):
        lm = store.observe(
            map_xyz=np.array([2.0, 0.0, 1.0], dtype=float) + offsets[i % len(offsets)],
            descriptor=desc,
            normal_map=normal,
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=views[i % len(views)],
            from_detector=True,
        )
    return lm


def test_detector_landmark_confirms_on_the_eighth_observation():
    """Three observations used to confirm a box landmark on one short pass."""
    store = EmbeddingLandmarkMap(
        promote_min_obs=3,
        promote_min_views=2,
        promote_min_view_gap_m=0.5,
        promote_max_std_m=0.12,
    )
    views = ((0.0, 0.0, 0.0), (0.6, 0.0, 0.0))
    offsets = (np.zeros(3), np.array([0.01, 0.0, 0.0]))
    lm = _observe_detector(store, n=7, views=views, offsets=offsets)
    assert lm is not None
    assert lm.status == LandmarkStatus.CANDIDATE
    assert lm.observations == 7
    lm = _observe_detector(store, n=1, views=views[1:], offsets=offsets)
    assert lm.observations == 8
    assert lm.status == LandmarkStatus.CONFIRMED


def test_detector_landmark_stays_candidate_above_12cm_scatter():
    """The old 22 cm allowance confirmed landmarks that wandered 14–16 cm."""
    store = EmbeddingLandmarkMap(
        promote_min_obs=3,
        promote_min_views=2,
        promote_min_view_gap_m=0.5,
        promote_max_std_m=0.12,
        associate_radius_m=0.35,
    )
    # Around the first point, inside the association radius, noisy on every axis.
    offsets = (
        np.zeros(3),
        np.array([0.20, 0.10, 0.08]),
        np.array([-0.20, -0.10, -0.08]),
        np.array([0.10, 0.20, -0.08]),
        np.array([-0.10, -0.20, 0.08]),
        np.array([0.08, -0.08, 0.20]),
        np.array([-0.08, 0.08, -0.20]),
        np.array([0.15, -0.15, 0.10]),
    )
    views = ((0.0, 0.0, 0.0), (0.6, 0.0, 0.0))
    lm = _observe_detector(store, n=8, views=views, offsets=offsets)
    assert lm is not None
    assert len(store) == 1
    assert 0.12 < lm.position_std_m < 0.22
    assert lm.status == LandmarkStatus.CANDIDATE


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



# --- Depth audit: does a surface exist where each landmark is stored? ---

_AUDIT_K = CameraIntrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)
_AUDIT_MOUNT = {"mount_x": 0.1, "mount_z": 0.4}


def _stored(store: EmbeddingLandmarkMap, lid: int, map_xyz, *, status=LandmarkStatus.CONFIRMED):
    lm = EmbeddingLandmark(
        landmark_id=lid,
        anchor_node=0,
        rel_xyz=np.array(map_xyz, dtype=float),
        normal_rel=np.array([-1.0, 0.0, 0.0]),
        map_xyz=np.array(map_xyz, dtype=float),
        descriptors=[],
        observations=5,
        status=status,
        from_detector=True,
    )
    store._landmarks[lid] = lm  # noqa: SLF001
    store._index_add(lm, lm.map_xyz)  # noqa: SLF001
    return lm


def _audit(store, depth, *, remove, pose=Pose2D(0.0, 0.0, 0.0)):
    return store.audit_against_depth(
        depth,
        robot_pose=pose,
        pose_of_node={},
        intrinsics=_AUDIT_K,
        remove=remove,
        **_AUDIT_MOUNT,
    )


def _row(audit: dict, lid: int) -> dict:
    return next(r for r in audit["rows"] if r["id"] == lid)


def test_audit_verdicts_on_a_wall_at_three_metres():
    """Camera at base x=0.1 looking along +x at a wall 3 m from the lens."""
    store = EmbeddingLandmarkMap()
    _stored(store, 1, [3.1, 0.0, 0.4])  # on the wall
    _stored(store, 2, [1.6, 0.0, 0.4])  # in the air, 1.5 m in front of it
    _stored(store, 3, [4.1, 0.0, 0.4])  # behind the wall
    depth = np.full((240, 320), 3.0, dtype=np.float32)
    audit = _audit(store, depth, remove=False)
    assert audit["pose_suspect"] is False
    on_wall, in_air, behind = (_row(audit, i) for i in (1, 2, 3))
    assert on_wall["verdict"] == "agrees"
    assert in_air["verdict"] == "seen_through"
    assert in_air["label"] == "DC2"
    assert in_air["stored_depth_m"] == pytest.approx(1.5, abs=0.01)
    assert in_air["measured_depth_m"] == pytest.approx(3.0, abs=0.01)
    assert in_air["strikes"] == 1
    assert behind["verdict"] == "occluded"


def test_audit_removes_after_consecutive_seen_through_frames():
    store = EmbeddingLandmarkMap()
    _stored(store, 1, [3.1, 0.0, 0.4])
    _stored(store, 2, [1.6, 0.0, 0.4])
    _stored(store, 5, [1.6, 0.3, 0.4], status=LandmarkStatus.CANDIDATE)
    depth = np.full((240, 320), 3.0, dtype=np.float32)
    for _ in range(2):
        audit = _audit(store, depth, remove=True)
        assert not any(r["removed"] for r in audit["rows"])
    assert store.seen_through_ids() == []
    assert store.seen_through_ids(min_strikes=1) == [2, 5]
    audit = _audit(store, depth, remove=True)
    assert _row(audit, 2)["removed"] and _row(audit, 5)["removed"]
    assert store.get(2) is None and store.get(5) is None
    assert store.get(1) is not None
    assert store.status_dict()["seen_through_removed"] == 2
    # Gone from the spatial index too: nothing left to associate with.
    assert all(1 in ids or not ids for ids in store._cell_index.values())  # noqa: SLF001


def test_audit_keeps_a_well_seen_landmark_unless_the_gap_is_large():
    """477 observations at a 1.1 m gap were being deleted; a 1.5 m gap still goes."""
    store = EmbeddingLandmarkMap()
    kept = _stored(store, 2, [1.6, 0.0, 0.4])
    _stored(store, 5, [1.6, 0.4, 0.4], status=LandmarkStatus.CANDIDATE)
    kept.observations = 40
    # Stored depth is 1.5 m. 2.3 m is seen through (past the 0.4 m margin)
    # but short of the 1.2 m gap a well-observed landmark needs.
    near = np.full((240, 320), 2.3, dtype=np.float32)
    for _ in range(3):
        audit = _audit(store, near, remove=True)
    row = _row(audit, 2)
    assert row["verdict"] == "seen_through"
    assert row["strikes"] == store.see_through_strikes
    assert row["removed"] is False
    assert store.get(2) is not None
    assert store.seen_through_ids() == [2]
    # A young landmark at the same disagreement is still deleted.
    assert store.get(5) is None

    gone = _stored(store, 7, [1.6, -0.4, 0.4])
    gone.observations = 40
    far = np.full((240, 320), 3.0, dtype=np.float32)
    for _ in range(3):
        audit = _audit(store, far, remove=True)
    assert _row(audit, 7)["removed"] is True
    assert store.get(7) is None


def test_audit_agreeing_frame_clears_strikes():
    store = EmbeddingLandmarkMap()
    _stored(store, 2, [1.6, 0.0, 0.4])
    far = np.full((240, 320), 3.0, dtype=np.float32)
    here = np.full((240, 320), 1.5, dtype=np.float32)
    _audit(store, far, remove=True)
    _audit(store, far, remove=True)
    assert _row(_audit(store, here, remove=True), 2)["strikes"] == 0
    _audit(store, far, remove=True)
    _audit(store, far, remove=True)
    assert store.get(2) is not None


def test_audit_report_only_flags_but_keeps_the_landmark():
    store = EmbeddingLandmarkMap()
    _stored(store, 2, [1.6, 0.0, 0.4])
    depth = np.full((240, 320), 3.0, dtype=np.float32)
    for _ in range(5):
        audit = _audit(store, depth, remove=False)
    assert _row(audit, 2)["strikes"] == store.see_through_strikes
    assert _row(audit, 2)["removed"] is False
    assert store.get(2) is not None
    assert store.seen_through_ids() == [2]
    assert store.status_dict()["seen_through_flagged"] == 1


def test_audit_missing_depth_and_occluders_never_strike():
    store = EmbeddingLandmarkMap()
    _stored(store, 2, [1.6, 0.0, 0.4])
    no_depth = np.zeros((240, 320), dtype=np.float32)
    blocked = np.full((240, 320), 0.6, dtype=np.float32)
    for depth, verdict in ((no_depth, "no_depth"), (blocked, "occluded")):
        for _ in range(4):
            row = _row(_audit(store, depth, remove=True), 2)
            assert row["verdict"] == verdict
            assert row["strikes"] == 0
    assert store.get(2) is not None


def test_audit_thin_object_in_front_of_far_background_agrees():
    """A pole fills a fraction of the window; it is still there."""
    store = EmbeddingLandmarkMap()
    _stored(store, 2, [1.6, 0.0, 0.4])
    depth = np.full((240, 320), 3.5, dtype=np.float32)
    depth[:, 156:164] = 1.5  # 8 px wide at the image center
    assert _row(_audit(store, depth, remove=True), 2)["verdict"] == "agrees"


def test_audit_blames_the_pose_when_most_landmarks_are_seen_through():
    """Robot is really 0.8 m further from the wall than its pose says."""
    store = EmbeddingLandmarkMap()
    for i, y in enumerate((-0.6, -0.3, 0.0, 0.3, 0.6)):
        _stored(store, i + 1, [3.1, y, 0.4])
    depth = np.full((240, 320), 3.8, dtype=np.float32)
    for _ in range(5):
        audit = _audit(store, depth, remove=True)
        assert audit["pose_suspect"] is True
        assert all(r["verdict"] == "seen_through" for r in audit["rows"])
        assert not any(r["removed"] or r["strikes"] for r in audit["rows"])
    assert len(store) == 5


def test_confirm_keeps_rel_xyz_with_the_refined_map_xyz():
    """Confirming moves map_xyz to the sample mean; rel_xyz must follow."""
    store = EmbeddingLandmarkMap()
    anchor = Pose2D(1.0, 2.0, 0.5)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    lm = None
    for i, view in enumerate([(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)]):
        lm = store.observe(
            map_xyz=np.array([3.0 + 0.06 * i, 2.0, 1.0]),
            descriptor=desc,
            normal_map=np.array([-1.0, 0.0, 0.0]),
            anchor_node=0,
            anchor_pose=anchor,
            view_xy=view,
        )
    assert lm is not None and lm.status == LandmarkStatus.CONFIRMED
    assert lm.map_xyz[0] == pytest.approx(3.06)
    np.testing.assert_allclose(
        EmbeddingLandmarkMap.map_point_from_rel(anchor, lm.rel_xyz), lm.map_xyz, atol=1e-9
    )


def test_update_from_another_anchor_does_not_rewrite_rel_xyz():
    """rel_xyz is relative to the landmark's own anchor, not the current one."""
    store = EmbeddingLandmarkMap(promote_min_obs=99)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    first_anchor = Pose2D(1.0, 0.0, 0.0)
    common = {
        "map_xyz": np.array([3.0, 0.0, 1.0]),
        "descriptor": desc,
        "normal_map": np.array([-1.0, 0.0, 0.0]),
    }
    lm = store.observe(anchor_node=0, anchor_pose=first_anchor, view_xy=(0.0, 0.0), **common)
    rel_before = lm.rel_xyz.copy()
    # Robot has driven on: the current keyframe is node 7, and the pose table
    # no longer holds node 0.
    later_anchor = Pose2D(2.5, 0.4, 1.0)
    again = store.observe_with_poses(
        anchor_node=7,
        anchor_pose=later_anchor,
        pose_of_node={7: later_anchor},
        view_xy=(1.0, 0.0),
        **common,
    )
    assert again.landmark_id == lm.landmark_id
    assert lm.anchor_node == 0
    np.testing.assert_allclose(lm.rel_xyz, rel_before)


def test_audit_ignores_incoherent_far_depth_on_a_real_frame():
    """Real camera-top frame: part of a painting 3.7 m away reads 4-10 m.

    The stereo matcher returns a steep ramp of far depths on the painting's
    left side. That is a sensor artefact, not a view past the painting, and
    it once deleted a correctly placed landmark (DC101).
    """
    from pathlib import Path

    from src.semantic.landmarks import _depth_verdict

    path = Path(__file__).parent / "data" / "camera_top_depth_424x240.npz"
    depth = np.load(path)["depth_mm"].astype(float) / 1000.0
    fx = 303.845

    def verdict(u, v, stored):
        return _depth_verdict(
            depth, u, v, stored, half_px=0.10 * fx / stored, focal_px=fx
        )

    # Landmark on the painting's left side, where the depth is garbage.
    kind, measured = verdict(94.1, 65.9, 3.617)
    assert measured > 6.0
    assert kind == "unclear"
    # Landmark on the painting's well-measured centre.
    assert verdict(122.1, 67.0, 3.684)[0] == "agrees"
    # Landmark in the air 1.8 m in front of the painting: the wall behind.
    kind, measured = verdict(138.1, 55.3, 1.797)
    assert kind == "seen_through"
    assert measured == pytest.approx(3.65, abs=0.1)
    # Landmark in the air with a chair and a wall behind it: two real
    # surfaces, still seen through.
    assert verdict(322.6, 148.4, 1.589)[0] == "seen_through"


def test_audit_sees_through_to_a_floor_or_side_wall():
    """Backgrounds steep in one image direction are real surfaces."""
    from src.semantic.landmarks import _depth_verdict

    rows = np.arange(240, dtype=float)[:, None] * np.ones((1, 320))
    cols = np.ones((240, 1)) * np.arange(320, dtype=float)[None, :]
    # 0.06 m per pixel at 3 m with f=300 px is a slope of 6 (about 80 degrees).
    up = 0.06 * (120.0 - rows)
    across = 0.06 * (cols - 160.0)
    floor = 3.0 + up
    side_wall = 3.0 + across
    garbage = 3.0 + up + across
    for depth, expected in (
        (floor, "seen_through"),
        (side_wall, "seen_through"),
        (garbage, "unclear"),
    ):
        kind, _ = _depth_verdict(
            depth, 160.0, 120.0, 1.0, half_px=12.0, focal_px=300.0
        )
        assert kind == expected

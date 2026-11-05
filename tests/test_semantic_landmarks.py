"""Unit tests for embedding landmark map and rigid fit."""

from __future__ import annotations

import numpy as np
import pytest

from src.geom.conversions import Pose2D
from src.semantic.fit import fit_rigid_2d
from src.semantic.landmarks import EmbeddingLandmarkMap
from src.semantic.place import PlaceIndex, fit_pca, fit_vocabulary, l2_normalize
from src.semantic.types import LandmarkStatus


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
        promote_max_std_m=0.05,
        demote_offset_m=0.15,
        demote_passes=3,
    )
    anchor = Pose2D(1.0, 2.0, 0.0)
    desc = l2_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    normal = np.array([-1.0, 0.0, 0.0], dtype=float)
    # Three views of the same wall point.
    views = [(0.0, 0.0), (0.6, 0.0), (1.2, 0.0)]
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
            view_xy=(0.0, 0.0),
        )
    assert lm.status == LandmarkStatus.MOVED


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

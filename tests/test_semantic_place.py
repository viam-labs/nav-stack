"""VLAD place index: fit, query, persist, supervisor keyframes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom.conversions import Pose2D
from src.semantic.persist import load_place_index, save_place_index
from src.semantic.place import PlaceIndex, descriptors_from_patches, fit_pca, fit_vocabulary, l2_normalize
from src.semantic.supervisor import VisionSupervisor
from src.semantic.types import CameraIntrinsics, PatchToken, VisionFrame


def test_place_index_fit_improves_same_view_rank():
    rng = np.random.default_rng(0)
    descs = l2_normalize(rng.standard_normal((400, 48)).astype(np.float32))
    index = PlaceIndex(cosine_threshold=0.2)
    a = descs[:40]
    b = descs[40:80]
    c = descs[80:120]
    index.add_keyframe(0, Pose2D(0.0, 0.0, 0.0), a)
    index.add_keyframe(1, Pose2D(3.0, 0.0, 0.0), b)
    index.add_keyframe(2, Pose2D(6.0, 0.0, 0.0), c)
    # Add more banks so fit has enough rows.
    for i in range(3, 10):
        chunk = descs[i * 30 : (i + 1) * 30]
        index.add_keyframe(i, Pose2D(float(i), 0.0, 0.0), chunk)
    assert index.fit(n_clusters=8, pca_dim=16, min_descriptors=64)
    assert index.fitted
    hits = index.query(a[:20], top_k=3)
    assert hits
    assert hits[0][0] == 0


def test_save_load_place_index(tmp_path: Path):
    rng = np.random.default_rng(1)
    descs = l2_normalize(rng.standard_normal((200, 32)).astype(np.float32))
    comps, mean = fit_pca(descs, out_dim=8)
    vocab = fit_vocabulary((descs - mean) @ comps.T, n_clusters=4, seed=0)
    index = PlaceIndex(
        pca_components=comps,
        pca_mean=mean,
        vocabulary=vocab,
        cosine_threshold=0.3,
    )
    index.add_keyframe(7, Pose2D(1.0, 2.0, 0.5), descs[:25])
    index.add_keyframe(8, Pose2D(4.0, 0.0, -0.2), descs[25:50])
    path = save_place_index(tmp_path, index)
    assert path is not None
    loaded = load_place_index(tmp_path)
    assert loaded is not None
    assert loaded.fitted
    assert loaded.keyframe_ids == [7, 8]
    assert loaded.keyframe_poses[0].x == pytest.approx(1.0)
    hits = loaded.query(descs[:25], top_k=1)
    assert hits and hits[0][0] == 7


def test_descriptors_from_patches_subsamples():
    patches = [
        PatchToken(
            row=i,
            col=0,
            u=float(i),
            v=0.0,
            descriptor=l2_normalize(np.ones(8, dtype=np.float32) * (i + 1)),
        )
        for i in range(100)
    ]
    stacked = descriptors_from_patches(patches, max_patches=16)
    assert stacked is not None
    assert stacked.shape == (16, 8)


def _base_lidars():
    return [
        {"name": "front-lidar", "scan_source": "get_laser_scan"},
        {
            "name": "depth-cam",
            "scan_source": "point_cloud",
            "obstacles_only": True,
            "cloud_frame": "camera_optical",
            "mount": {"x": 0.15, "z": 0.4},
        },
    ]


def test_supervisor_builds_place_keyframes(tmp_path: Path):
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": _base_lidars(),
            "vision_enabled": True,
            "vision_synthetic": True,
            "vision_survey_only": True,
            "vision_place_enabled": True,
            "vision_place_keyframe_min_gap_m": 0.5,
            "vision_place_keyframe_min_gap_deg": 15.0,
            "vision_place_min_keyframes_to_fit": 4,
            "vision_place_vocab_size": 4,
            "vision_place_pca_dim": 16,
            "vision_cameras": ["depth-cam"],
        }
    )
    poses = [Pose2D(0.0, 0.0, 0.0)]

    def get_pose():
        return poses[0]

    def get_pose_at(_t):
        return poses[0]

    def get_anchor():
        return 0, poses[0], {0: poses[0]}

    vision = VisionSupervisor(
        cfg=cfg,
        cameras={},
        get_pose=get_pose,
        get_pose_at=get_pose_at,
        get_anchor=get_anchor,
    )
    K = CameraIntrinsics(fx=300, fy=300, cx=80, cy=60, width=160, height=120)
    for i in range(6):
        poses[0] = Pose2D(float(i) * 0.6, 0.0, math.radians(i * 20.0))
        rgb = np.full((120, 160, 3), fill_value=(40 + i * 10) % 200, dtype=np.uint8)
        # Distinct texture so synthetic descriptors differ by pose/frame.
        rgb[20:40, 30:50] = (i * 40) % 255
        depth = np.full((120, 160), 1.5, dtype=np.float32)
        frame = VisionFrame(
            camera_name="cam",
            stamp=float(i),
            captured_at=float(i),
            color_rgb=rgb,
            depth_m=depth,
            intrinsics=K,
        )
        vision.ingest_frame(frame, mapping=True)

    st = vision.status()["place"]
    assert st["enabled"] is True
    assert st["keyframes"] >= 4
    assert st["last_query"] is not None
    path = vision.save_to_map_dir(tmp_path)
    # landmarks may be empty with synthetic+height gates; place index should save
    loaded = load_place_index(tmp_path)
    assert loaded is not None
    assert len(loaded.keyframe_ids) >= 4

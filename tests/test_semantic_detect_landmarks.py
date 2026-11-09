"""Detector-box → pooled DINOv2 landmark observations."""

from __future__ import annotations

import numpy as np
import pytest

from src.semantic.detect_landmarks import (
    observations_from_detections,
    pool_descriptor,
)
from src.semantic.types import CameraIntrinsics, Detection, PatchToken, VisionFrame


def _frame_with_wall(*, depth_m: float = 1.5) -> VisionFrame:
    h, w = 240, 320
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:, :] = (30, 60, 90)
    # Textured painting region.
    color[60:180, 100:220] = (180, 140, 40)
    color[70:170:4, 110:210:4] = (220, 200, 80)
    depth = np.full((h, w), depth_m, dtype=np.float32)
    K = CameraIntrinsics(
        fx=0.5 * w, fy=0.5 * h, cx=0.5 * w, cy=0.5 * h, width=w, height=h
    )
    return VisionFrame(
        camera_name="cam",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=color,
        depth_m=depth,
        intrinsics=K,
        mount_x=0.2,
        mount_z=0.5,
    )


def _grid_tokens(frame: VisionFrame, *, dim: int = 16) -> list[PatchToken]:
    h, w = frame.color_rgb.shape[:2]
    tokens: list[PatchToken] = []
    for r, v in enumerate(range(20, h, 28)):
        for c, u in enumerate(range(20, w, 28)):
            # Distinct descriptors by location so pooling is meaningful.
            desc = np.zeros(dim, dtype=np.float32)
            desc[0] = u / w
            desc[1] = v / h
            desc[2] = 1.0
            n = float(np.linalg.norm(desc))
            desc /= n
            tokens.append(
                PatchToken(row=r, col=c, u=float(u), v=float(v), descriptor=desc)
            )
    return tokens


def test_pool_descriptor_l2_unit():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = pool_descriptor([a, b])
    assert out.shape == (3,)
    assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-5)


def test_observations_from_detections_pools_box():
    frame = _frame_with_wall()
    tokens = _grid_tokens(frame)
    det = Detection(
        label="painting",
        conf=0.9,
        x0=100.0,
        y0=60.0,
        x1=220.0,
        y1=180.0,
    )
    obs = observations_from_detections(
        frame,
        tokens,
        [det],
        landmark_labels=["painting", "tv"],
        z_min_m=0.25,
        z_max_m=2.0,
        min_patches=2,
    )
    assert len(obs) == 1
    o = obs[0]
    assert float(np.linalg.norm(o.token.descriptor)) == pytest.approx(1.0, abs=1e-5)
    # Box center ~ (160, 120); depth 1.5 + mount_x 0.2 → base x ≈ 1.7
    assert o.point_base[0] == pytest.approx(1.7, abs=0.15)
    assert abs(o.point_base[1]) < 0.25


def test_observations_skips_non_landmark_labels():
    frame = _frame_with_wall()
    tokens = _grid_tokens(frame)
    det = Detection(
        label="person",
        conf=0.99,
        x0=100.0,
        y0=60.0,
        x1=220.0,
        y1=180.0,
    )
    obs = observations_from_detections(
        frame,
        tokens,
        [det],
        landmark_labels=["painting"],
        min_patches=2,
    )
    assert obs == []


def test_soft_label_match_picture_frame():
    frame = _frame_with_wall()
    tokens = _grid_tokens(frame)
    det = Detection(
        label="picture frame",
        conf=0.8,
        x0=100.0,
        y0=60.0,
        x1=220.0,
        y1=180.0,
    )
    obs = observations_from_detections(
        frame,
        tokens,
        [det],
        landmark_labels=["picture"],
        min_patches=2,
    )
    assert len(obs) == 1

"""Unit tests for RGB-D patch back-projection and paint gating."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.semantic.paint_mask import Disc, filter_scan_endpoints
from src.semantic.patches import (
    backproject_patches,
    depth_sigma_m,
    distinctiveness_mask,
    lateral_sigma_m,
)
from src.semantic.types import CameraIntrinsics, PatchToken, VisionFrame
from src.semantic.worker import VisionWorker, WorkerConfig


def _wall_frame(*, depth_m: float = 2.0, hole: bool = False) -> VisionFrame:
    h, w = 240, 320
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:, :] = (40, 80, 120)
    color[::8, ::8] = (200, 200, 50)
    depth = np.full((h, w), depth_m, dtype=np.float32)
    if hole:
        depth[100:140, 140:180] = 0.0
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
        mount_theta=0.0,
    )


def test_backproject_round_trip_forward_mount():
    frame = _wall_frame(depth_m=2.0)
    tok = PatchToken(
        row=0,
        col=0,
        u=frame.intrinsics.cx,
        v=frame.intrinsics.cy,
        descriptor=np.ones(8, dtype=np.float32),
    )
    obs = backproject_patches(frame, [tok])
    assert len(obs) == 1
    p = obs[0].point_base
    # Optical Z=2 → sensor X=2; mount_x=0.2 → base x ≈ 2.2
    assert p[0] == pytest.approx(2.2, abs=0.05)
    assert abs(p[1]) < 0.05
    assert math.isfinite(obs[0].sigma_m)
    assert obs[0].sigma_m > 0.0


def test_backproject_rejects_holes():
    frame = _wall_frame(depth_m=2.0, hole=True)
    tok_hole = PatchToken(
        row=0, col=0, u=160.0, v=120.0, descriptor=np.ones(4, dtype=np.float32)
    )
    obs = backproject_patches(frame, [tok_hole], window_half=3)
    assert obs == []


def test_sigma_grows_with_range():
    assert depth_sigma_m(4.0) > depth_sigma_m(1.0)
    assert lateral_sigma_m(4.0, fx=300.0) > lateral_sigma_m(1.0, fx=300.0)


def test_filter_scan_endpoints_drops_disc():
    xy = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)
    out = filter_scan_endpoints(xy, [Disc(x=2.0, y=0.0, radius_m=0.5)])
    assert len(out) == 2
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(3.0)


def test_distinctiveness_keeps_unique_descriptors():
    a = np.array([1.0, 0.0, 0.0], dtype=float)
    b = np.array([0.0, 1.0, 0.0], dtype=float)
    desc = np.stack([a, a, a, b], axis=0)
    keep = distinctiveness_mask(desc, max_ratio=0.9)
    assert bool(keep[3])
    assert not all(keep[:3])


def test_synthetic_worker_runs():
    frame = _wall_frame()
    worker = VisionWorker(WorkerConfig(synthetic=True, input_short_side=112, patch=14))
    result = worker.process(frame)
    assert result.provider == "synthetic"
    assert len(result.patches) > 0
    assert result.place_descriptor is not None
    assert result.infer_ms >= 0.0
    obs = backproject_patches(frame, result.patches)
    assert len(obs) > 0

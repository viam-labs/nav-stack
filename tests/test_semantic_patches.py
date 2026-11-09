"""Unit tests for RGB-D patch back-projection and paint gating."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.semantic.paint_mask import Disc, filter_scan_endpoints
from src.semantic.patches import (
    annotate_patch_texture,
    backproject_patches,
    depth_sigma_m,
    distinctiveness_mask,
    filter_textured_tokens,
    lateral_sigma_m,
    local_luminance_std,
    select_survey_tokens,
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
    keep = distinctiveness_mask(desc, max_ratio=0.92)
    assert bool(keep[3])
    assert not all(keep[:3])


def test_texture_filter_prefers_painting_over_white_wall():
    h, w = 120, 160
    color = np.full((h, w, 3), 240, dtype=np.uint8)  # flat white wall
    # High-contrast "painting" region on the right.
    yy, xx = np.mgrid[20:100, 90:150]
    color[20:100, 90:150, 0] = (xx * 3) % 255
    color[20:100, 90:150, 1] = (yy * 5) % 255
    color[20:100, 90:150, 2] = ((xx + yy) * 7) % 255
    wall = PatchToken(row=0, col=0, u=40.0, v=60.0, descriptor=np.ones(4))
    art = PatchToken(row=0, col=1, u=120.0, v=60.0, descriptor=np.ones(4))
    annotate_patch_texture(color, [wall, art])
    assert local_luminance_std(color, art.u, art.v) > local_luminance_std(
        color, wall.u, wall.v
    )
    kept = filter_textured_tokens([wall, art], min_texture=8.0)
    assert art in kept
    assert wall not in kept


def test_select_survey_keeps_painting_interior_despite_similar_neighbors():
    """Hard distinctiveness drops art interiors; texture-first must keep them."""
    h, w = 120, 160
    color = np.full((h, w, 3), 240, dtype=np.uint8)
    yy, xx = np.mgrid[20:100, 90:150]
    color[20:100, 90:150, 0] = (xx * 3) % 255
    color[20:100, 90:150, 1] = (yy * 5) % 255
    color[20:100, 90:150, 2] = ((xx + yy) * 7) % 255
    # Three near-duplicate descriptors on the painting (as DINOv2 neighbors).
    art_desc = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    art_desc = art_desc / np.linalg.norm(art_desc)
    wall_desc = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    arts = [
        PatchToken(row=0, col=i, u=100.0 + 5 * i, v=60.0, descriptor=art_desc.copy())
        for i in range(3)
    ]
    wall = PatchToken(row=1, col=0, u=40.0, v=60.0, descriptor=wall_desc)
    kept = select_survey_tokens(color, arts + [wall], min_texture=8.0, max_keep=4)
    assert wall not in kept
    assert sum(1 for t in kept if t in arts) >= 2


def test_subsample_prefers_higher_texture():
    from src.semantic.patches import PatchObservation, subsample_observations

    wall = PatchObservation(
        token=PatchToken(
            row=0, col=0, u=0.0, v=0.0, descriptor=np.ones(2), texture=3.0
        ),
        point_base=np.array([1.0, 0.0, 1.0]),
        normal_base=np.array([-1.0, 0.0, 0.0]),
        sigma_m=0.01,  # closer / lower uncertainty than the painting
    )
    art = PatchObservation(
        token=PatchToken(
            row=0, col=1, u=1.0, v=0.0, descriptor=np.ones(2), texture=40.0
        ),
        point_base=np.array([1.5, 0.0, 1.0]),
        normal_base=np.array([-1.0, 0.0, 0.0]),
        sigma_m=0.05,
    )
    out = subsample_observations([wall, art], 1)
    assert len(out) == 1
    assert out[0] is art


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


def test_backproject_height_gate():
    frame = _wall_frame(depth_m=2.0)  # mount_z=0.5 → center hit z≈0.5
    tok = PatchToken(
        row=0,
        col=0,
        u=frame.intrinsics.cx,
        v=frame.intrinsics.cy,
        descriptor=np.ones(8, dtype=np.float32),
    )
    kept = backproject_patches(frame, [tok], z_min_m=0.25, z_max_m=2.0)
    assert len(kept) == 1
    assert kept[0].point_base[2] == pytest.approx(0.5, abs=0.05)
    # Treat the wall hit as "floor" by raising z_min above mount height.
    dropped = backproject_patches(frame, [tok], z_min_m=0.6, z_max_m=2.0)
    assert dropped == []


def test_subsample_observations_caps_count():
    from src.semantic.patches import PatchObservation, subsample_observations

    toks = [
        PatchObservation(
            token=PatchToken(row=i, col=0, u=float(i), v=0.0, descriptor=np.ones(2)),
            point_base=np.array([1.0, 0.0, 1.0]),
            normal_base=np.array([-1.0, 0.0, 0.0]),
            sigma_m=0.01 * (i + 1),
        )
        for i in range(20)
    ]
    out = subsample_observations(toks, 5)
    assert len(out) == 5


def test_dinov2_onnx_letterbox_smoke():
    """If the exported ViT-S ONNX is present, run one real inference frame."""
    from pathlib import Path

    onnx = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "semantic"
        / "models"
        / "dinov2_vits14_patches_448.onnx"
    )
    if not onnx.is_file():
        pytest.skip("dinov2 onnx not present (fetch/export first)")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed")

    # Richer texture than _wall_frame so distinctiveness keeps some patches.
    frame = _wall_frame(depth_m=2.0)
    h, w = frame.color_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    frame.color_rgb[:] = np.stack(
        [
            (xx * 3) % 255,
            (yy * 5) % 255,
            ((xx + yy) * 7) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    worker = VisionWorker(
        WorkerConfig(
            dinov2_onnx=str(onnx),
            synthetic=False,
            input_short_side=448,
            cpu_threads=1,
        )
    )
    assert worker.stats["provider"] != "synthetic"
    result = worker.process(frame)
    assert result.provider != "synthetic"
    assert len(result.patches) > 0
    # Patch UVs must land inside the source image (letterbox undone).
    for tok in result.patches[:20]:
        assert 0.0 <= tok.u < w
        assert 0.0 <= tok.v < h
    obs = backproject_patches(frame, result.patches, z_min_m=0.25, z_max_m=2.0)
    assert len(obs) > 0
    worker.close()

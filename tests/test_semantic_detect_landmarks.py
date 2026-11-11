"""Detector-box → pooled DINOv2 landmark observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.semantic.detect_landmarks import (
    BOX_MAX_RANGE_M,
    BOX_NOT_LANDMARK_LABEL,
    BOX_USED,
    SKIP_HEIGHT,
    SKIP_MIXED_DEPTH,
    SKIP_RANGE,
    SKIP_SPARSE_DEPTH,
    box_surface_point,
    describe_box_outcome,
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


def test_tokens_inside_detections_keeps_box_centers():
    from src.semantic.detect_landmarks import tokens_inside_detections

    frame = _frame_with_wall()
    tokens = _grid_tokens(frame)
    det = Detection(label="painting", conf=0.4, x0=100, y0=60, x1=220, y1=180)
    inside = tokens_inside_detections(tokens, [det])
    assert len(inside) >= 2
    assert all(100 <= t.u <= 220 and 60 <= t.v <= 180 for t in inside)


def test_detection_obs_with_single_token():
    """Small YOLO boxes must still form a landmark from one in-box patch."""
    frame = _frame_with_wall()
    tok = PatchToken(
        row=0,
        col=0,
        u=160.0,
        v=120.0,
        descriptor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    det = Detection(label="painting", conf=0.3, x0=140, y0=100, x1=180, y1=140)
    obs = observations_from_detections(
        frame,
        [tok],
        [det],
        landmark_labels=["painting"],
        z_min_m=0.25,
        z_max_m=2.0,
        min_patches=1,
    )
    assert len(obs) == 1


def _box(frame: VisionFrame, x0, y0, x1, y1):
    return box_surface_point(frame, x0, y0, x1, y1, z_min_m=0.25, z_max_m=2.0)


def test_box_surface_takes_the_majority_surface():
    """A painting with a strip of nearer clutter still lands on the painting."""
    frame = _frame_with_wall(depth_m=3.0)
    # A chair back pokes into the bottom fifth of the box core.
    frame.depth_m[150:180, 100:220] = 1.2
    surface, why = _box(frame, 100, 60, 220, 180)
    assert why is None and surface is not None
    assert surface.depth_m == pytest.approx(3.0, abs=0.02)
    assert surface.point_base[0] == pytest.approx(3.2, abs=0.05)
    assert 0.6 <= surface.support < 1.0
    # Wall faces the camera: normal points back along -x in base_link.
    assert surface.normal_base[0] < -0.9


def test_box_surface_skips_two_surfaces_without_majority():
    """Half object, half background: a median would be a point on neither."""
    frame = _frame_with_wall(depth_m=3.0)
    frame.depth_m[60:180, 100:160] = 1.6
    surface, why = _box(frame, 100, 60, 220, 180)
    assert surface is None
    assert why == SKIP_MIXED_DEPTH


def test_box_surface_skips_sparse_depth():
    """Object pixels with no depth must not hand the box to what little is left."""
    frame = _frame_with_wall(depth_m=3.0)
    frame.depth_m[60:180, 100:220] = 0.0
    # A sliver of background wall is all the depth the box has.
    frame.depth_m[60:180, 190:200] = 3.9
    surface, why = _box(frame, 100, 60, 220, 180)
    assert surface is None
    assert why == SKIP_SPARSE_DEPTH


def test_box_surface_rejects_out_of_range():
    frame = _frame_with_wall(depth_m=7.0)
    surface, why = _box(frame, 100, 60, 220, 180)
    assert surface is None
    assert why == SKIP_RANGE


def test_box_surface_accepts_a_clean_surface_out_to_four_and_a_half_metres():
    """A box averages hundreds of pixels, so it reaches past the 4 m patch limit."""
    frame = _frame_with_wall(depth_m=4.4)
    surface, why = _box(frame, 100, 60, 220, 180)
    assert why is None and surface is not None
    assert surface.depth_m == pytest.approx(4.4, abs=0.02)
    assert BOX_MAX_RANGE_M == pytest.approx(4.5)


def test_box_surface_rejects_a_clean_surface_past_four_and_a_half_metres():
    """5.5 m repeated from one pose, then failed the next viewpoint."""
    frame = _frame_with_wall(depth_m=5.5)
    surface, why = _box(frame, 100, 60, 220, 180)
    assert surface is None
    assert why == SKIP_RANGE


def test_far_noisy_box_is_called_range_not_mixed_depth():
    """Beyond range the depth scatters; say "too far", not "two surfaces"."""
    frame = _frame_with_wall(depth_m=3.0)
    rng = np.random.default_rng(3)
    frame.depth_m[60:180, 100:220] = rng.uniform(7.0, 14.0, size=(120, 120))
    det = Detection(label="painting", conf=0.9, x0=100.0, y0=60.0, x1=220.0, y1=180.0)
    stats: dict = {}
    outcomes: list = []
    obs = observations_from_detections(
        frame,
        _grid_tokens(frame),
        [det],
        landmark_labels=["painting"],
        stats=stats,
        outcomes=outcomes,
    )
    assert obs == []
    assert stats == {SKIP_RANGE: 1}
    assert outcomes[0]["outcome"] == SKIP_RANGE
    assert 7.0 < outcomes[0]["depth_m"] < 14.0
    assert describe_box_outcome(SKIP_RANGE, outcomes[0]["depth_m"]).startswith("far ")


def test_describe_box_outcome_texts():
    assert describe_box_outcome(SKIP_RANGE, 9.62) == "far 9.6m"
    assert describe_box_outcome(SKIP_RANGE, 0.31) == "near 0.3m"
    assert describe_box_outcome(SKIP_RANGE) == "out of range"
    assert describe_box_outcome(SKIP_MIXED_DEPTH) == "mixed depth"
    assert describe_box_outcome(SKIP_SPARSE_DEPTH) == "no depth"
    assert describe_box_outcome(SKIP_HEIGHT, 2.0) == "height"
    assert describe_box_outcome(BOX_NOT_LANDMARK_LABEL) == "not a landmark class"
    assert describe_box_outcome(BOX_USED, 2.0) == "used"


def test_outcomes_has_one_entry_per_detection_in_order():
    frame = _frame_with_wall(depth_m=3.0)
    # Left box: half near, half far. Right box: clean wall.
    frame.depth_m[40:120, 20:50] = 1.6
    dets = [
        Detection(label="person", conf=0.9, x0=100.0, y0=60.0, x1=220.0, y1=180.0),
        Detection(label="painting", conf=0.9, x0=20.0, y0=40.0, x1=80.0, y1=120.0),
        Detection(label="painting", conf=0.9, x0=230.0, y0=60.0, x1=310.0, y1=180.0),
    ]
    outcomes: list = []
    obs = observations_from_detections(
        frame,
        _grid_tokens(frame),
        dets,
        landmark_labels=["painting"],
        outcomes=outcomes,
    )
    assert [o["outcome"] for o in outcomes] == [
        BOX_NOT_LANDMARK_LABEL,
        SKIP_MIXED_DEPTH,
        BOX_USED,
    ]
    assert [o["label"] for o in outcomes] == ["person", "painting", "painting"]
    assert len(obs) == 1
    assert outcomes[2]["obs_index"] == 0
    assert outcomes[2]["depth_m"] == pytest.approx(3.0, abs=0.02)
    assert outcomes[0]["obs_index"] is None and outcomes[1]["obs_index"] is None


def test_box_surface_keeps_an_oblique_plane_together():
    """Depth ramps across a painting seen at an angle; still one surface."""
    frame = _frame_with_wall(depth_m=3.0)
    ramp = np.linspace(2.85, 3.15, 120, dtype=np.float32)
    frame.depth_m[60:180, 100:220] = ramp[None, :]
    surface, why = _box(frame, 100, 60, 220, 180)
    assert why is None and surface is not None
    assert surface.depth_m == pytest.approx(3.0, abs=0.03)
    assert surface.support == pytest.approx(1.0)


def test_mixed_box_makes_no_observation_and_is_counted():
    frame = _frame_with_wall(depth_m=3.0)
    frame.depth_m[60:180, 100:160] = 1.6
    det = Detection(label="painting", conf=0.9, x0=100.0, y0=60.0, x1=220.0, y1=180.0)
    stats: dict = {}
    obs = observations_from_detections(
        frame,
        _grid_tokens(frame),
        [det],
        landmark_labels=["painting"],
        stats=stats,
    )
    assert obs == []
    assert stats == {SKIP_MIXED_DEPTH: 1}


# One real camera-top frame (424x240, depth in mm) from the office: a painting
# on a wall 3.67 m away, a table with trays at 2.4-3.0 m in front of a wall at
# 3.9 m, and an armchair at 1.2 m in the foreground.
_REAL_DEPTH = Path(__file__).parent / "data" / "camera_top_depth_424x240.npz"


# A second frame looking down the long side of the office: a poster on a
# pillar 5.5 m away, a framed picture on the far wall ~9.5 m away, and an
# armchair under a desk in the foreground.
_REAL_DEPTH_FAR = Path(__file__).parent / "data" / "camera_top_depth_far_wall_424x240.npz"


def real_camera_top_frame(path: Path = _REAL_DEPTH) -> VisionFrame:
    depth = np.load(path)["depth_mm"].astype(np.float32) / 1000.0
    h, w = depth.shape
    return VisionFrame(
        camera_name="camera-top",
        stamp=0.0,
        captured_at=0.0,
        color_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=CameraIntrinsics(
            fx=303.845, fy=303.684, cx=219.518, cy=126.626, width=w, height=h
        ),
        mount_x=0.276,
        mount_y=0.015,
        mount_z=0.745,
    )


def test_real_frame_painting_box_lands_on_the_painting():
    frame = real_camera_top_frame()
    for box in ((85, 38, 145, 98), (75, 28, 155, 108)):
        surface, why = _box(frame, *box)
        assert why is None and surface is not None
        assert surface.depth_m == pytest.approx(3.67, abs=0.08)


def test_real_frame_table_box_is_skipped_not_put_on_the_wall_behind():
    """Trays + table: the per-patch median used to return the wall at 3.9 m."""
    frame = real_camera_top_frame()
    surface, why = _box(frame, 150, 60, 280, 200)
    assert surface is None
    assert why == SKIP_MIXED_DEPTH


def test_real_frame_poster_at_five_and_a_half_metres_is_too_far():
    """Repeatable from one pose; landmarks stored here died on the next look."""
    frame = real_camera_top_frame(_REAL_DEPTH_FAR)
    det = Detection(label="poster", conf=0.3, x0=236, y0=66, x1=277, y1=108)
    tok = PatchToken(
        row=0, col=0, u=256.0, v=87.0, descriptor=np.array([1.0, 0.0], dtype=np.float32)
    )
    outcomes: list = []
    obs = observations_from_detections(
        frame,
        [tok],
        [det],
        landmark_labels=["poster"],
        z_min_m=0.25,
        z_max_m=2.0,
        outcomes=outcomes,
    )
    assert obs == []
    assert outcomes[0]["outcome"] == SKIP_RANGE
    assert outcomes[0]["depth_m"] == pytest.approx(5.5, abs=0.15)
    assert describe_box_outcome(SKIP_RANGE, outcomes[0]["depth_m"]).startswith("far ")


def test_real_frame_picture_on_the_far_wall_is_out_of_range():
    frame = real_camera_top_frame(_REAL_DEPTH_FAR)
    det = Detection(label="picture frame", conf=0.3, x0=97, y0=82, x1=121, y1=112)
    tok = PatchToken(
        row=0, col=0, u=109.0, v=97.0, descriptor=np.array([1.0, 0.0], dtype=np.float32)
    )
    outcomes: list = []
    obs = observations_from_detections(
        frame,
        [tok],
        [det],
        landmark_labels=["picture"],
        z_min_m=0.25,
        z_max_m=2.0,
        outcomes=outcomes,
    )
    assert obs == []
    assert outcomes[0]["outcome"] == SKIP_RANGE
    assert outcomes[0]["depth_m"] > 8.0

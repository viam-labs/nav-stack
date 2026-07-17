"""Tests for multi-height-slice revisit verification (src/nav/slice_match.py)."""
import math

import numpy as np
import pytest

from src.nav import slice_match
from src.ros import conversions as conv


def _wall_points(x0, y0, x1, y1, n=80):
    """XY points along a segment, as an (n, 2) array."""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1)


def test_parse_bands_valid_and_invalid():
    bands = slice_match.SliceBand.parse_bands([[0.15, 0.45], [1.6, 2.4]])
    assert bands[0].z_min == pytest.approx(0.15)
    assert bands[1].z_max == pytest.approx(2.4)
    with pytest.raises(ValueError):
        slice_match.SliceBand.parse_bands([[1.0, 0.5]])


def test_slice_points_by_bands_splits_and_subsamples():
    z_lo = np.column_stack([np.arange(500.0), np.zeros(500), np.full(500, 0.3)])
    z_hi = np.column_stack([np.arange(10.0), np.zeros(10), np.full(10, 2.0)])
    pts = np.vstack([z_lo, z_hi])
    bands = slice_match.SliceBand.parse_bands([[0.15, 0.45], [1.6, 2.4]])
    lo, hi = slice_match.slice_points_by_bands(pts, bands, max_points_per_band=200)
    assert lo.shape == (200, 2)  # subsampled
    assert hi.shape == (10, 2)
    empty = slice_match.slice_points_by_bands(np.empty((0, 3)), bands)
    assert all(a.shape[0] == 0 for a in empty)


def test_slice_grid_hit_rate_same_place_high_shifted_low():
    grid = slice_match.SliceGrid(resolution_m=0.15)
    wall = _wall_points(2.0, -3.0, 2.0, 3.0)  # wall ahead in base_link
    origin = conv.Pose2D(0.0, 0.0, 0.0)
    grid.add_points(wall, origin)

    same = grid.hit_rate(wall, origin)
    assert same is not None and same > 0.9

    # 2 m shift perpendicular to the wall: points land away from stored cells.
    shifted = grid.hit_rate(wall, conv.Pose2D(2.0, 0.0, 0.0))
    assert shifted is not None and shifted < 0.2


def test_slice_grid_hit_rate_rotation_aware():
    grid = slice_match.SliceGrid(resolution_m=0.15)
    wall = _wall_points(2.0, -3.0, 2.0, 3.0)
    # Record from a rotated vantage: wall at bearing 90 deg.
    pose = conv.Pose2D(0.0, 0.0, math.pi / 2)
    grid.add_points(wall, pose)
    # Querying with the same pose reproduces the geometry.
    rate = grid.hit_rate(wall, pose)
    assert rate is not None and rate > 0.9
    # Un-rotated query misses.
    rate = grid.hit_rate(wall, conv.Pose2D(0.0, 0.0, 0.0))
    assert rate is not None and rate < 0.2


def test_library_verify_passes_at_recorded_pose_and_vetoes_elsewhere():
    bands = slice_match.SliceBand.parse_bands([[0.15, 0.45], [1.6, 2.4]])
    lib = slice_match.SliceLibrary(
        bands, min_query_points=10, min_reference_cells_near=10
    )
    low_wall = _wall_points(1.5, -2.0, 1.5, 2.0)
    head_shelf = _wall_points(-1.0, 1.2, 1.0, 1.2)
    pose_a = conv.Pose2D(0.0, 0.0, 0.0)
    lib.record([low_wall, head_shelf], pose_a)
    assert lib.scans_recorded == 1

    ok = lib.verify([low_wall, head_shelf], pose_a)
    assert ok["pass"] is True
    assert ok["bands_checked"] == 2
    assert ok["bands_failed"] == 0

    # Same low-band silhouette but different head-height geometry — the
    # desk-clutter aliasing case this feature exists to catch.
    other_head = _wall_points(-1.0, -1.8, 1.0, -1.8)
    bad = lib.verify([low_wall, other_head], pose_a)
    assert bad["pass"] is False
    assert bad["bands_failed"] == 1
    statuses = {e["band"][0]: e["status"] for e in bad["per_band"]}
    assert statuses[0.15] == "passed"
    assert statuses[1.6] == "failed"


def test_library_verify_skips_bands_without_data():
    bands = slice_match.SliceBand.parse_bands([[0.15, 0.45], [1.6, 2.4]])
    lib = slice_match.SliceLibrary(
        bands, min_query_points=10, min_reference_cells_near=10
    )
    wall = _wall_points(1.5, -2.0, 1.5, 2.0)

    # Empty library: nothing can be checked, verdict must be None (no veto).
    verdict = lib.verify([wall, wall], conv.Pose2D(0.0, 0.0, 0.0))
    assert verdict["pass"] is None
    assert verdict["bands_checked"] == 0

    # Record only near the origin; a pose in unexplored space has no
    # reference data and must not be vetoed.
    lib.record([wall, wall], conv.Pose2D(0.0, 0.0, 0.0))
    far = lib.verify([wall, wall], conv.Pose2D(50.0, 50.0, 0.0))
    assert far["pass"] is None
    assert all(e["status"] == "no_reference_data" for e in far["per_band"])

    # Too few query points in a band: that band is skipped, not failed.
    sparse = lib.verify([wall, wall[:3]], conv.Pose2D(0.0, 0.0, 0.0))
    assert sparse["per_band"][1]["status"] == "too_few_points"
    assert sparse["pass"] is True  # low band still checked and passed

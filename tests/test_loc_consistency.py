"""Scan-vs-map consistency used for mid-nav localization refine."""
from __future__ import annotations

import numpy as np

from src.geom import conversions as conv
from src.nav_builtin.loc_consistency import localization_looks_bad
from src.nav_builtin.types import OccupancyGrid, Pose2D


def _scan(ranges: np.ndarray) -> conv.LaserScan2D:
    n = len(ranges)
    return conv.LaserScan2D(
        ranges=np.asarray(ranges, dtype=float),
        angle_min=-np.pi,
        angle_increment=(2.0 * np.pi) / n,
        range_min=0.05,
        range_max=25.0,
    )


def _open_scan(range_m: float = 4.0, n: int = 360) -> conv.LaserScan2D:
    return _scan(np.full(n, range_m))


def _left_wall_occ() -> OccupancyGrid:
    grid = np.zeros((80, 80), dtype=np.int16)
    # Wall at y=1.55 m — 0.55 m to the left of (1.0, 1.0) facing +X.
    grid[31, 10:50] = 100
    return OccupancyGrid(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)


def _room_occ() -> OccupancyGrid:
    """Small room around (1.0, 1.0): walls ~0.6 m away on all sides."""
    grid = np.zeros((80, 80), dtype=np.int16)
    grid[8, 8:33] = 100
    grid[32, 8:33] = 100
    grid[8:33, 8] = 100
    grid[8:33, 32] = 100
    return OccupancyGrid(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)


def test_disagree_when_map_walls_do_not_match_scan():
    pose = Pose2D(1.0, 1.0, 0.0)
    verdict = localization_looks_bad(pose, _open_scan(), _left_wall_occ())
    assert verdict.disagree is True
    assert verdict.reason == "scan_map"
    assert verdict.disagree_frac >= 0.30
    assert verdict.compared_beams >= 6
    assert verdict.worst is not None
    assert verdict.worst.name == "left"


def test_agree_when_scan_matches_nearby_map():
    pose = Pose2D(1.0, 1.0, 0.0)
    # Lidar range matches the one mapped wall; other directions have no map claim.
    verdict = localization_looks_bad(pose, _open_scan(0.55), _left_wall_occ())
    assert verdict.disagree is False


def test_no_trigger_without_scan_or_map_structure():
    pose = Pose2D(1.0, 1.0, 0.0)
    occ = _left_wall_occ()
    assert localization_looks_bad(pose, None, occ).disagree is False
    empty = OccupancyGrid(
        grid=np.zeros((80, 80), dtype=np.int16),
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    assert localization_looks_bad(pose, _open_scan(), empty).disagree is False
    assert localization_looks_bad(pose, _open_scan(), None).disagree is False


def test_few_map_hits_do_not_trigger():
    """A couple of occupied cells are not enough to call localization bad."""
    grid = np.zeros((80, 80), dtype=np.int16)
    grid[31, 20] = 100
    occ = OccupancyGrid(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)
    verdict = localization_looks_bad(Pose2D(1.0, 1.0, 0.0), _open_scan(), occ)
    assert verdict.disagree is False
    assert verdict.compared_beams < 6


def test_disagree_when_surrounding_room_does_not_match_scan():
    pose = Pose2D(1.0, 1.0, 0.0)
    bad = localization_looks_bad(pose, _open_scan(), _room_occ())
    assert bad.disagree is True
    assert bad.compared_beams >= 6
    good = localization_looks_bad(pose, _open_scan(0.60), _room_occ())
    assert good.disagree is False

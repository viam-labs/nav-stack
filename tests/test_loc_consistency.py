"""Scan-vs-map consistency used for mid-nav localization refine."""
from __future__ import annotations

import numpy as np

from src.geom import conversions as conv
from src.nav_builtin.loc_consistency import (
    LocDisagreement,
    loc_blocks_forward,
    localization_looks_bad,
    match_scores_say_lost,
    published_pose_misregistered,
    residual_is_lost,
    slam_pose_is_trusted,
    small_local_match_worth_applying,
)
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


def test_residual_is_lost_ignores_thin_leftover():
    thin = LocDisagreement(
        disagree=True,
        compared_beams=12,
        disagree_beams=4,
        disagree_frac=0.333,
    )
    assert residual_is_lost(thin) is False
    bad = LocDisagreement(
        disagree=True,
        compared_beams=12,
        disagree_beams=8,
        disagree_frac=0.667,
    )
    assert residual_is_lost(bad) is True


def test_loc_blocks_forward_live_wall_case():
    """~53% disagree + negative scores must refuse forward motion."""
    # Under residual_is_lost's 0.55 / 16-beam bars, but still "this bad".
    live = LocDisagreement(
        disagree=True,
        compared_beams=12,
        disagree_beams=6,
        disagree_frac=0.529,
    )
    assert residual_is_lost(live) is False
    assert loc_blocks_forward(live) is True
    assert match_scores_say_lost(
        {"previous_score": -0.24, "tick_match_score": -0.04}
    )
    assert loc_blocks_forward(
        LocDisagreement(disagree=False),
        {"status": "ok", "previous_score": -0.1},
    )
    # Live tick recovered; the stale negative prior must not keep holding.
    assert loc_blocks_forward(
        LocDisagreement(disagree=False),
        {
            "status": "refused_along",
            "previous_score": -0.03,
            "tick_match_score": 0.37,
        },
    ) is False
    assert loc_blocks_forward(
        LocDisagreement(disagree=False),
        {"status": "refused_large"},
    )
    thin = LocDisagreement(
        disagree=True,
        compared_beams=12,
        disagree_beams=4,
        disagree_frac=0.333,
    )
    assert loc_blocks_forward(thin) is False


def test_trusted_slam_pose_does_not_block_on_beam_residual():
    """Live stuck case: good_match + shift 0 must not permanent-hold."""
    residual = LocDisagreement(
        disagree=True,
        compared_beams=45,
        disagree_beams=26,
        disagree_frac=0.578,
    )
    trusted = {
        "status": "ok",
        "corrected": False,
        "good_match": True,
        "score": 0.66,
        "previous_score": 0.66,
        "tick_match_score": 0.70,
        "shift_m": 0.0,
        "shift_deg": 0.0,
        "ray_mae_m": 0.63,
    }
    assert slam_pose_is_trusted(trusted) is True
    assert loc_blocks_forward(residual, trusted) is False
    # Negative prior still blocks even with a decent candidate score.
    assert slam_pose_is_trusted(
        {**trusted, "previous_score": -0.1, "good_match": True}
    ) is False


def test_published_pose_misregistered_on_a_refused_large_jump():
    """The live hallway: 2.6 m / 74° refused, ray error still about a meter."""
    refused = {
        "status": "low_quality",
        "corrected": False,
        "large_jump": True,
        "shift_m": 2.582,
        "shift_deg": 74.0,
        "ray_mae_m": 1.078,
        "good_match": False,
        "score": 0.29,
    }
    assert published_pose_misregistered(refused) is True
    # Applied, or a weak score with no shift (map change): keep the map.
    assert published_pose_misregistered({**refused, "corrected": True}) is False
    assert (
        published_pose_misregistered(
            {
                "status": "low_quality",
                "corrected": False,
                "large_jump": False,
                "shift_m": 0.0,
                "shift_deg": 0.0,
                "ray_mae_m": 0.9,
            }
        )
        is False
    )
    assert published_pose_misregistered(None) is False


def test_small_local_match_worth_applying_live_dock_case():
    assert small_local_match_worth_applying(
        {
            "status": "nav_hold",
            "corrected": False,
            "good_match": False,
            "match_mode": "local",
            "shift_m": 0.32,
            "shift_deg": 20.0,
            "score": 0.47,
            "previous_score": -0.17,
            "large_jump": False,
        }
    )
    # Under the 1 m cap even if SLAM marked large_jump (0.75 m).
    assert small_local_match_worth_applying(
        {
            "corrected": False,
            "match_mode": "local",
            "shift_m": 0.9,
            "shift_deg": 8.0,
            "score": 0.40,
            "previous_score": 0.05,
            "large_jump": True,
        }
    )
    # 2.2 m / 0.17 twin — over the cap and below the score floor.
    assert not small_local_match_worth_applying(
        {
            "corrected": False,
            "match_mode": "local",
            "shift_m": 2.229,
            "shift_deg": 2.0,
            "score": 0.166,
            "previous_score": -0.229,
            "large_jump": True,
        }
    )


def test_early_gap_triggers_before_fraction_is_high():
    """A few unmatched beams with a ~0.5 m gap should refine before 22%."""
    pose = Pose2D(1.0, 1.0, 0.0)
    occ = _room_occ()
    ranges = np.full(360, 0.60)
    # Open a 25° left wedge so ~5 samples (every 5°) disagree vs the 0.6 m wall.
    angles = -np.pi + np.arange(360) * (2.0 * np.pi / 360)
    wrapped = (angles - (np.pi / 2.0) + np.pi) % (2.0 * np.pi) - np.pi
    ranges[np.abs(wrapped) <= np.radians(12.0)] = 4.0
    verdict = localization_looks_bad(
        pose, _scan(ranges), occ, min_frac=0.50, early_beams=4
    )
    assert verdict.disagree is True
    assert verdict.disagree_frac < 0.50


def test_early_gap_trigger_is_off_by_default():
    """A few open-door beams must not stop nav on their own."""
    pose = Pose2D(1.0, 1.0, 0.0)
    occ = _room_occ()
    ranges = np.full(360, 0.60)
    angles = -np.pi + np.arange(360) * (2.0 * np.pi / 360)
    wrapped = (angles - (np.pi / 2.0) + np.pi) % (2.0 * np.pi) - np.pi
    ranges[np.abs(wrapped) <= np.radians(12.0)] = 4.0
    verdict = localization_looks_bad(pose, _scan(ranges), occ)
    assert verdict.disagree is False

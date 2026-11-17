"""The mapping paint check with lidars at more than one height.

A scan is painted only when it lands on walls the map already has. With two
lidars the merged scan keeps the nearest return in each direction, so new
furniture at one height used to veto a pose that the walls at the other
height confirmed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import occupancy as occ
from src.slam_builtin import scan_match
from src.slam_builtin.engine import _HEIGHT_VOTE_MIN_SCORE, BuiltinSlamEngine

from .test_loop_closure import _Sensors

HALF_X, HALF_Y = 3.0, 2.0
BEAMS = 360
LOW_Z, TOP_Z = 0.21, 1.20
LOW_BAND, TOP_BAND = 0, 4


def _room_ranges(pose: conv.Pose2D) -> np.ndarray:
    """Ranges to the walls of a 6 m x 4 m room, seen from ``pose``."""
    angles = -math.pi + np.arange(BEAMS) * (2 * math.pi / BEAMS)
    dx, dy = np.cos(angles + pose.theta), np.sin(angles + pose.theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (HALF_X - pose.x) / dx, (-HALF_X - pose.x) / dx)
        ty = np.where(dy > 0, (HALF_Y - pose.y) / dy, (-HALF_Y - pose.y) / dy)
    return np.minimum(
        np.where(np.isfinite(tx), tx, np.inf), np.where(np.isfinite(ty), ty, np.inf)
    )


def _scan(ranges: np.ndarray) -> conv.LaserScan2D:
    return conv.LaserScan2D(ranges, -math.pi, 2 * math.pi / BEAMS, 0.05, 25.0)


def _cluttered(ranges: np.ndarray, share: float) -> np.ndarray:
    """``ranges`` with that share of the beams stopping on something 0.9 m away."""
    out = ranges.copy()
    step = max(1, int(round(1.0 / share)))
    out[::step] = 0.9
    return out


def _room_grid() -> occ.LogOddsGrid:
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    grid = occ.empty_grid(resolution=0.05, size_m=10.0)
    for _ in range(4):
        grid = occ.insert_scan(
            grid,
            truth.x,
            truth.y,
            truth.theta,
            _room_ranges(truth),
            -math.pi,
            2 * math.pi / BEAMS,
            range_min=0.1,
            range_max=10.0,
        )
    return grid


def _room_map() -> scan_match.OccupancyMap:
    grid = _room_grid()
    return scan_match.occupancy_map_from_int16(
        occ.to_occupancy_int16(grid),
        resolution=grid.resolution,
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
    )


def _bands(low: np.ndarray, top: np.ndarray) -> dict:
    return {LOW_BAND: _scan(low), TOP_BAND: _scan(top)}


def _best(votes) -> float:
    return max(vote.score for vote in votes)


def test_walls_at_one_height_confirm_the_pose_when_the_other_sees_new_furniture():
    room = _room_map()
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    low = _cluttered(walls, share=0.34)  # a third of the low beams hit chairs
    votes = {
        v.band: v for v in scan_match.height_votes(room, _bands(low, walls), truth)
    }
    assert votes[TOP_BAND].score == pytest.approx(1.0)
    assert votes[LOW_BAND].score < _HEIGHT_VOTE_MIN_SCORE
    assert _best(votes.values()) >= _HEIGHT_VOTE_MIN_SCORE
    # The merged scan keeps the nearer return, so the chairs replace the
    # walls behind them and it reads as a miss at the same, correct pose.
    merged = _scan(np.minimum(low, walls))
    assert scan_match.scan_map_fit(room, merged, truth) == "miss"


def test_a_wrong_pose_fails_at_every_height():
    room = _room_map()
    walls = _room_ranges(conv.Pose2D(0.0, 0.0, 0.0))
    # Shifts across the long walls and a turn. (A shift along them is not
    # visible in an empty room: those returns still land on the same walls.)
    for wrong in (
        conv.Pose2D(0.0, 0.25, 0.0),
        conv.Pose2D(0.25, 0.25, 0.0),
        conv.Pose2D(0.0, 0.0, math.radians(8.0)),
    ):
        votes = scan_match.height_votes(room, _bands(walls, walls), wrong)
        assert votes and _best(votes) < _HEIGHT_VOTE_MIN_SCORE, wrong


def test_a_return_one_cell_from_the_wall_counts_and_one_three_cells_out_does_not():
    room = _room_map()
    walls = _scan(_room_ranges(conv.Pose2D(0.0, 0.0, 0.0)))
    on, judged = scan_match.wall_agreement(room, walls, conv.Pose2D(0.0, 0.0, 0.0))
    near, _ = scan_match.wall_agreement(room, walls, conv.Pose2D(0.04, 0.04, 0.0))
    far, _ = scan_match.wall_agreement(room, walls, conv.Pose2D(0.15, 0.15, 0.0))
    assert on == pytest.approx(1.0) and judged > 300
    assert near > 0.9
    assert far < 0.0


def test_a_height_that_saw_little_of_the_map_does_not_vote():
    room = _room_map()
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    sliver = np.full(BEAMS, np.inf)
    sliver[170:210] = walls[170:210]  # one slice of a depth camera: 40 beams
    votes = scan_match.height_votes(
        room, {LOW_BAND: _scan(walls), 3: _scan(sliver)}, truth
    )
    assert [v.band for v in votes] == [LOW_BAND]
    # A height looking only at unmapped ground cannot vote either.
    beyond = np.full(BEAMS, 20.0)
    votes = scan_match.height_votes(
        room, {LOW_BAND: _scan(walls), 3: _scan(beyond)}, truth
    )
    assert [v.band for v in votes] == [LOW_BAND]


def _cloud(ranges: np.ndarray, z: float) -> np.ndarray:
    angles = -math.pi + np.arange(BEAMS) * (2 * math.pi / BEAMS)
    ok = np.isfinite(ranges)
    return np.column_stack(
        [
            ranges[ok] * np.cos(angles[ok]),
            ranges[ok] * np.sin(angles[ok]),
            np.full(ok.sum(), z),
        ]
    )


def _engine(tmp_path) -> BuiltinSlamEngine:
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": str(tmp_path),
            "mode": "mapping",
            "scan_bins": BEAMS,
            "builtin_repair_on_save": False,
        }
    )
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore(str(tmp_path)), rate_hz=5.0)  # type: ignore[arg-type]
    with engine._lock:  # noqa: SLF001
        engine._grid = _room_grid()  # noqa: SLF001
        engine._invalidate_occ_cache()  # noqa: SLF001
    return engine


def test_the_engine_paints_when_one_lidar_height_lines_up(tmp_path):
    from viam.utils import dict_to_struct

    engine = _engine(tmp_path)
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    low = _cluttered(walls, share=0.34)
    merged = _scan(np.minimum(low, walls))
    frames = [
        SimpleNamespace(cloud_xyz=_cloud(low, LOW_Z), range_min=0.05, range_max=25.0),
        SimpleNamespace(cloud_xyz=_cloud(walls, TOP_Z), range_min=0.05, range_max=25.0),
    ]
    engine._sensors.mapping_frames = lambda: frames  # noqa: SLF001

    assert engine._scan_fits_map(merged, truth) is True  # noqa: SLF001
    fit = engine.diagnostics()["paint"]["fit"]
    assert fit["rule"] == "best_height" and fit["state"] == "fit"
    assert (
        fit["pass_mark"] == _HEIGHT_VOTE_MIN_SCORE and fit["score"] >= fit["pass_mark"]
    )
    assert [(h["z_min_m"], h["z_max_m"]) for h in fit["heights"]] == [
        (0.0, 0.25),
        (1.0, 1.25),
    ]
    assert fit["heights"][0]["score"] < fit["pass_mark"] <= fit["heights"][1]["score"]
    dict_to_struct(fit)

    # The same two scans at a pose 30 cm off: neither height lines up.
    assert engine._scan_fits_map(merged, conv.Pose2D(0.0, 0.30, 0.0)) is False  # noqa: SLF001
    assert engine.diagnostics()["paint"]["fit"]["state"] == "miss"


def test_one_lidar_height_is_checked_exactly_as_before(tmp_path):
    engine = _engine(tmp_path)
    room = _room_map()
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    for ranges, pose in (
        (walls, truth),
        (_cluttered(walls, share=0.34), truth),
        (walls, conv.Pose2D(0.0, 0.0, math.radians(8.0))),
    ):
        scan = _scan(ranges)
        # No per-sensor frames, or frames all in one plane: the merged scan decides.
        for frames in (
            [],
            [
                SimpleNamespace(
                    cloud_xyz=_cloud(ranges, LOW_Z), range_min=0.05, range_max=25.0
                )
            ],
        ):
            engine._sensors.mapping_frames = lambda frames=frames: frames  # noqa: SLF001
            want, score = scan_match.scan_map_overlap(room, scan, pose)
            assert engine._scan_map_fit(scan, pose) == want  # noqa: SLF001
            fit = engine.diagnostics()["paint"]["fit"]
            assert fit["rule"] == "merged_scan" and fit["pass_mark"] == 0.5
            assert fit["score"] == pytest.approx(score, abs=1e-3)


# --- The pose match and the last few centimetres --------------------------------


def _two_height_engine(tmp_path, low: np.ndarray, top: np.ndarray) -> BuiltinSlamEngine:
    """Engine on the room map whose sensors report these two heights."""
    engine = _engine(tmp_path)
    frames = [
        SimpleNamespace(cloud_xyz=_cloud(low, LOW_Z), range_min=0.05, range_max=25.0),
        SimpleNamespace(cloud_xyz=_cloud(top, TOP_Z), range_min=0.05, range_max=25.0),
    ]
    engine._sensors.mapping_frames = lambda: frames  # noqa: SLF001
    return engine


def _map_of(engine: BuiltinSlamEngine) -> scan_match.OccupancyMap:
    return engine._occupancy_for_match()  # noqa: SLF001


def _far(pose: conv.Pose2D, other: conv.Pose2D) -> float:
    return math.hypot(pose.x - other.x, pose.y - other.y)


def test_the_height_the_map_explains_takes_over_the_pose_match(tmp_path):
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    low = _cluttered(walls, share=0.5)  # half the low beams stop on furniture
    merged = _scan(np.minimum(low, walls))
    engine = _two_height_engine(tmp_path, low, walls)
    room = _map_of(engine)

    off = conv.Pose2D(0.0, 0.20, 0.0)
    found, score, _prior, _rival, used = engine._match_pose(room, merged, off)  # noqa: SLF001
    detail = engine.diagnostics()["match"]
    assert detail["rule"] == "best_height"
    assert detail["decided_by"] == {"z_min_m": 1.0, "z_max_m": 1.25}
    assert detail["merged_score"] + engine._MATCH_HEIGHT_MARGIN <= score  # noqa: SLF001
    assert found is not None and _far(found, truth) < 0.06
    # The yaw-only fix that follows uses the same scan as the match.
    assert np.array_equal(
        used.ranges, engine._band_scans(engine._mapping_frames())[TOP_BAND].ranges
    )  # noqa: SLF001

    # At the right pose that height has nothing better to offer, so the pose
    # stays, whatever the cluttered height would ask for.
    stay = engine._match_pose(room, merged, truth)  # noqa: SLF001
    assert stay[0] is None and engine.diagnostics()["match"]["rule"] == "best_height"


def test_the_merged_scan_keeps_the_pose_match_where_every_height_agrees(tmp_path):
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    engine = _two_height_engine(tmp_path, walls, walls)
    room = _map_of(engine)
    off = conv.Pose2D(0.0, 0.20, 0.0)
    found = engine._match_pose(room, _scan(walls), off)  # noqa: SLF001
    want = scan_match.refine_pose(room, _scan(walls), off)
    detail = engine.diagnostics()["match"]
    assert detail["rule"] == "merged_scan" and len(detail["heights"]) == 2
    assert found[0] is not None and want[0] is not None
    assert (found[0].x, found[0].y, found[0].theta) == (
        want[0].x,
        want[0].y,
        want[0].theta,
    )


def test_one_lidar_height_is_matched_exactly_as_before(tmp_path):
    engine = _engine(tmp_path)
    room = _map_of(engine)
    walls = _scan(_room_ranges(conv.Pose2D(0.0, 0.0, 0.0)))
    off = conv.Pose2D(0.10, 0.15, math.radians(4.0))
    found = engine._match_pose(room, walls, off)  # noqa: SLF001
    want = scan_match.refine_pose(room, walls, off)
    assert found[4] is walls and engine.diagnostics()["match"] == {
        "rule": "merged_scan"
    }
    assert found[1:4] == want[1:4]
    assert (found[0].x, found[0].y, found[0].theta) == (
        want[0].x,
        want[0].y,
        want[0].theta,
    )
    # And there is nothing to polish.
    assert engine._paint_pose_by_heights(off) == ("one_height", None)  # noqa: SLF001


def test_polish_closes_a_few_centimetres_and_says_when_the_fit_is_farther_away():
    room = _room_map()
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _scan(_room_ranges(truth))

    start = conv.Pose2D(0.11, 0.11, 0.0)
    here, _judged = scan_match.wall_agreement(room, walls, start)
    found, score, edge_gain = scan_match.polish_on_walls(room, walls, start)
    assert found is not None and score > here and score > 0.9
    assert _far(found, truth) < _far(start, truth) and edge_gain < 0.05

    # 20 cm off across the long walls: the fit keeps improving out to the
    # edge of the search, so this is not the polish's to fix.
    _found, _score, edge_gain = scan_match.polish_on_walls(
        room, walls, conv.Pose2D(0.0, 0.20, 0.0)
    )
    assert edge_gain >= 0.05


def test_polish_does_not_slide_the_pose_along_a_corridor():
    # Two long walls and nothing across them: every position along the
    # corridor fits the same.
    angles = -math.pi + np.arange(BEAMS) * (2 * math.pi / BEAMS)
    with np.errstate(divide="ignore"):
        ranges = np.abs(1.0 / np.sin(angles))
    ranges[ranges > 6.0] = np.inf
    corridor = _scan(ranges)
    grid = occ.empty_grid(resolution=0.05, size_m=30.0)
    for x in np.arange(-6.0, 6.01, 0.25):
        grid = occ.insert_scan(
            grid,
            float(x),
            0.0,
            0.0,
            ranges,
            -math.pi,
            2 * math.pi / BEAMS,
            range_min=0.1,
            range_max=10.0,
        )
    hall = scan_match.occupancy_map_from_int16(
        occ.to_occupancy_int16(grid),
        resolution=grid.resolution,
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
    )
    pose = conv.Pose2D(0.4, 0.0, 0.0)
    found, score, edge_gain = scan_match.polish_on_walls(hall, corridor, pose)
    assert found is not None and score > 0.9 and edge_gain < 0.05
    assert _far(found, pose) < 1e-9 and found.theta == pose.theta


def test_a_marginal_scan_is_painted_at_the_polished_pose(tmp_path):
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    low = _cluttered(walls, share=0.5)
    engine = _two_height_engine(tmp_path, low, walls)
    merged = _scan(np.minimum(low, walls))

    # 10 cm off across the long walls: the best height's vote is short of
    # confident, and a polish can still reach where its walls fit.
    off = conv.Pose2D(0.0, 0.10, 0.0)
    engine.set_pose(off)
    kind, polished = engine._paint_pose_by_heights(off)  # noqa: SLF001
    assert (
        kind == "moved"
        and polished is not None
        and _far(polished, truth) < _far(off, truth)
    )
    updates = engine._updates  # noqa: SLF001
    engine._maybe_insert_scan(merged, now=10.0)  # noqa: SLF001
    assert engine._updates == updates + 1  # noqa: SLF001
    assert engine.diagnostics()["pose_polishes"] == 1
    assert _far(engine.get_pose(), truth) < _far(off, truth)

    # 20 cm off: no pose within the polish passes, so nothing is painted.
    far = conv.Pose2D(0.0, 0.20, 0.0)
    engine.set_pose(far)
    assert engine._paint_pose_by_heights(far)[0] == "refuse"  # noqa: SLF001
    updates = engine._updates  # noqa: SLF001
    refused = engine.diagnostics()["paint"]["skips_unmatched"]
    engine._maybe_insert_scan(merged, now=20.0)  # noqa: SLF001
    assert engine._updates == updates  # noqa: SLF001
    paint = engine.diagnostics()["paint"]
    assert (
        paint["skips_unmatched"] == refused + 1 and paint["state"] == "held_unmatched"
    )


def test_a_heading_lost_in_a_spin_is_found_by_the_height_that_fits(tmp_path):
    truth = conv.Pose2D(0.0, 0.0, 0.0)
    walls = _room_ranges(truth)
    low = _cluttered(walls, share=0.5)
    engine = _two_height_engine(tmp_path, low, walls)
    spun = conv.Pose2D(0.0, 0.0, math.radians(40.0))
    found, agreement = engine._search_heading_for_scan(
        _scan(np.minimum(low, walls)), spun
    )  # noqa: SLF001
    assert found is not None and agreement >= _HEIGHT_VOTE_MIN_SCORE
    assert abs(conv.normalize_angle(found.theta)) < math.radians(5.0)

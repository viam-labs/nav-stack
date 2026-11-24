"""Finding the robot again when no scan has fit the map for a while.

While mapping, a scan that misses the map is not painted. The tick recovers a
lost heading and a slip of up to about half a meter; past that nothing was
painted again until a person switched to localizing and refined the pose. On
one robot the pose was 2.2 m off with the heading right, and the search that
should have caught it used both lidars merged into one scan.
"""

from __future__ import annotations

import math
import time

import numpy as np

from src.geom import conversions as conv
from src.slam_builtin import occupancy as occ
from src.slam_builtin import position_recover as pr
from src.slam_builtin import scan_match

from .test_paint_fit_heights import (
    BEAMS,
    _bands,
    _cluttered,
    _room_map,
    _room_ranges,
    _scan,
    _two_height_engine,
)

TRUTH = conv.Pose2D(0.6, -0.4, 0.0)
# Where the robot believes it is: 1.8 m from where it really is.
SLIPPED = conv.Pose2D(TRUTH.x + 1.5, TRUTH.y - 1.0, 0.0)


def _far(a: conv.Pose2D, b: conv.Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _two_heights():
    """Low lidar with a third of its beams on furniture, top lidar on walls."""
    walls = _room_ranges(TRUTH)
    low = _cluttered(walls, share=0.34)
    return low, walls, _scan(np.minimum(low, walls))


def test_a_slip_of_two_meters_is_found_and_every_height_agrees():
    room = _room_map()
    low, top, merged = _two_heights()
    bands = _bands(low, top)
    assert max(v.score for v in scan_match.height_votes(room, bands, SLIPPED)) < 0.0

    found = pr.find_position(room, merged, bands, SLIPPED)

    assert found.reason == "recovered" and found.pose is not None
    assert _far(found.pose, TRUTH) < 0.06
    assert abs(conv.normalize_angle(found.pose.theta)) < math.radians(2.0)
    assert found.shift_m > 1.7 and found.radius_m == 3.0
    assert found.score_before < 0.0 and found.score_after > 0.9


def test_a_pose_that_is_right_stays_put_when_the_room_has_changed():
    """Scans miss because of new furniture, not because of the pose."""
    room = _room_map()
    walls = _room_ranges(TRUTH)
    low = _cluttered(walls, share=0.5)
    top = _cluttered(walls, share=0.5)
    bands = _bands(low, top)
    assert max(v.score for v in scan_match.height_votes(room, bands, TRUTH)) < 0.65

    found = pr.find_position(room, _scan(np.minimum(low, top)), bands, TRUTH)

    assert found.pose is None and found.reason == "no_better_place"


def test_one_height_cannot_move_the_pose_against_one_that_fits_where_it_is():
    """The top lidar fits 1.8 m away; the low lidar fits where the robot is."""
    room = _room_map()
    top = _room_ranges(TRUTH)
    low = _room_ranges(SLIPPED)

    found = pr.find_position(
        room, _scan(np.minimum(low, top)), _bands(low, top), SLIPPED
    )

    assert found.pose is None
    assert found.reason == "heights_disagree"


def test_the_best_height_decides_when_another_sees_what_the_map_does_not():
    """Live tracer2a, 1.7 m off after a restart.

    The 1.2 m lidar sat on walls on four sides at the right place. The 0.21 m
    lidar sees furniture standing in front of those walls, and scored worse at
    the right place than at the wrong one, where it happened to line up with
    one wall. Asking every height to agree left the robot lost.
    """
    room = _room_map()
    top = _room_ranges(TRUTH)
    # Fits the wrong pose a little (well short of the paint check) and the
    # right pose not at all.
    low = _cluttered(_room_ranges(SLIPPED), share=0.4)
    bands = _bands(low, top)
    votes = {v.band: v.score for v in scan_match.height_votes(room, bands, SLIPPED)}
    assert 0.2 < max(votes.values()) < pr.MIN_SCORE_ALONE

    found = pr.find_position(room, _scan(np.minimum(low, top)), bands, SLIPPED)

    assert found.reason == "recovered" and found.pose is not None
    assert _far(found.pose, TRUTH) < 0.06 and found.score_after > 0.9


def test_a_height_deciding_alone_must_pass_the_paint_check():
    from src.slam_builtin.engine import _HEIGHT_VOTE_MIN_SCORE

    assert pr.MIN_SCORE_ALONE == _HEIGHT_VOTE_MIN_SCORE


def test_two_places_that_look_the_same_are_refused():
    """Two identical rooms side by side, and a robot between them."""
    walls = _room_ranges(conv.Pose2D(0.0, 0.0, 0.0))
    grid = occ.empty_grid(resolution=0.05, size_m=20.0)
    for center_x in (-3.5, 3.5):
        for _ in range(4):
            grid = occ.insert_scan(
                grid,
                center_x,
                0.0,
                0.0,
                walls,
                -math.pi,
                2 * math.pi / BEAMS,
                range_min=0.1,
                range_max=10.0,
            )
    twins = scan_match.occupancy_map_from_int16(
        occ.to_occupancy_int16(grid),
        resolution=grid.resolution,
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
    )
    between = conv.Pose2D(0.0, 0.0, 0.0)

    found = pr.find_position(twins, _scan(walls), _bands(walls, walls), between)

    assert found.pose is None and found.reason == "ambiguous"


# --- In the engine --------------------------------------------------------


def _held_engine(tmp_path, held_s: float):
    low, top, merged = _two_heights()
    engine = _two_height_engine(tmp_path, low, top)
    with engine._lock:  # noqa: SLF001
        engine._pose = SLIPPED  # noqa: SLF001
        engine._last_scan_obj = merged  # noqa: SLF001
    engine._note_paint("held_unmatched", time.monotonic() - held_s, fit="miss")  # noqa: SLF001
    return engine, merged


def test_the_engine_moves_the_live_pose_and_nothing_else(tmp_path):
    engine, merged = _held_engine(tmp_path, held_s=10.0)
    grid = engine._grid  # noqa: SLF001
    nodes = len(engine._graph)  # noqa: SLF001
    assert engine._scan_fits_map(merged, SLIPPED) is False  # noqa: SLF001
    assert engine.paint_hold()["held_s"] >= 10.0

    outcome = engine.recover_position_if_held()

    assert outcome is not None and outcome["recovered"] is True
    pose = engine.get_pose()
    assert _far(pose, TRUTH) < 0.06
    assert engine._grid is grid and len(engine._graph) == nodes  # noqa: SLF001
    # The scan that was being refused now passes the paint check.
    assert engine._scan_fits_map(merged, pose) is True  # noqa: SLF001
    status = engine.diagnostics()["paint"]
    assert status["recover"]["recovered"] == 1
    assert status["recover"]["last"]["shift_m"] > 1.7


def test_the_engine_waits_before_searching_and_between_searches(tmp_path):
    engine, _merged = _held_engine(tmp_path, held_s=1.0)
    assert engine.recover_position_if_held() is None
    assert _far(engine.get_pose(), SLIPPED) < 1e-9

    engine, _merged = _held_engine(tmp_path, held_s=10.0)
    assert engine.recover_position_if_held() is not None
    # Straight after an attempt the next one waits.
    assert engine.recover_position_if_held() is None


def test_a_painted_scan_ends_the_hold_and_a_turn_does_not(tmp_path):
    engine, _merged = _held_engine(tmp_path, held_s=10.0)
    now = time.monotonic()
    engine._note_paint("held_turning", now)  # noqa: SLF001
    assert engine.paint_hold()["held_s"] >= 10.0
    engine._note_paint("extend_only", now, fit="fit")  # noqa: SLF001
    assert engine.paint_hold()["held_s"] == 0.0
    assert engine.recover_position_if_held() is None


def test_no_search_while_the_robot_is_turning(tmp_path):
    engine, _merged = _held_engine(tmp_path, held_s=10.0)
    engine._last_yaw_rate = math.radians(40.0)  # noqa: SLF001
    assert engine.recover_position_if_held() is None
    assert engine.paint_hold()["recover"]["attempts"] == 0


def test_searches_that_place_nothing_come_less_often_until_the_robot_moves(tmp_path):
    """A robot standing still gets the same answer each time, at a second or
    two of matching per search on its own computer."""
    engine, _merged = _held_engine(tmp_path, held_s=10.0)
    # Nothing nearby is better than where it is: two identical views.
    walls = _room_ranges(SLIPPED)
    frames = engine._sensors.mapping_frames()  # noqa: SLF001
    for frame in frames:
        frame.cloud_xyz = np.column_stack(
            [
                walls * np.cos(-math.pi + np.arange(BEAMS) * (2 * math.pi / BEAMS)),
                walls * np.sin(-math.pi + np.arange(BEAMS) * (2 * math.pi / BEAMS)),
                frame.cloud_xyz[: len(walls), 2],
            ]
        )
    with engine._lock:  # noqa: SLF001
        engine._last_scan_obj = _scan(walls)  # noqa: SLF001

    def later(seconds: float):
        engine._recover_last_at -= seconds  # noqa: SLF001
        return engine.recover_position_if_held()

    first = engine.recover_position_if_held()
    assert first is not None and first["recovered"] is False
    assert later(3.0) is None  # the wait doubled to 4 s
    assert later(2.0) is not None  # 5 s since the last one
    assert later(5.0) is None  # and doubled again, to 8 s
    # A meter driven is a new view: no need to wait the rest out.
    engine._paint_miss_travel_m += 1.2  # noqa: SLF001
    assert engine.recover_position_if_held() is not None
    assert engine.paint_hold()["recover"]["attempts"] == 3

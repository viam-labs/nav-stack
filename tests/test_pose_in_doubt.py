"""Painting waits while the mapping revisit check has the robot somewhere else.

Live tracer2a: the first look of that check had the scan fitting a place
3.38 m from the pose. Scans went on being painted where the robot thought it
was, the second look found nothing wrong there, and a hallway was mapped a
second time beside the first.
"""

from __future__ import annotations

from src.geom import conversions as conv
from src.slam_builtin.engine import BuiltinSlamEngine

from .test_loop_closure import _raycast
from .test_rebuild_keeps_pose import TRUTH, _engine


def _stand_at(engine: BuiltinSlamEngine, pose: conv.Pose2D) -> None:
    """The robot is at ``pose``, standing, and its lidar sees the room from there."""
    with engine._lock:  # noqa: SLF001
        engine._pose = pose  # noqa: SLF001
    engine._sensors.get_scan = lambda max_age_s=2.0: _raycast(TRUTH, pose)  # noqa: SLF001
    engine._sensors.get_odom = lambda: conv.OdomReading(0.0, 0.0, 0.0, pose=pose)  # noqa: SLF001


def _painted(engine: BuiltinSlamEngine) -> int:
    return int(engine.diagnostics()["map_updates"])


def test_nothing_is_painted_while_the_pose_is_in_doubt(tmp_path):
    engine = _engine(tmp_path)
    # 0.6 m on from the last scan painted: far enough for the next one.
    _stand_at(engine, conv.Pose2D(0.6 * 8, 0.0, 0.0))
    before = _painted(engine)

    engine.hold_paint_while_pose_in_doubt(True)
    engine._tick()  # noqa: SLF001
    assert _painted(engine) == before
    assert engine.diagnostics()["paint"]["state"] == "held_pose_in_doubt"

    engine.hold_paint_while_pose_in_doubt(False)
    engine._tick()  # noqa: SLF001
    assert _painted(engine) == before + 1
    assert engine.diagnostics()["paint"]["state"] == "painting"


def test_a_doubt_that_never_settles_does_not_stop_the_map_for_good(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stand_at(engine, conv.Pose2D(0.6 * 8, 0.0, 0.0))
    before = _painted(engine)
    monkeypatch.setattr(BuiltinSlamEngine, "_POSE_IN_DOUBT_MAX_S", 0.0)

    # The check reports the same doubt again and again; the hold has run out.
    engine.hold_paint_while_pose_in_doubt(True)
    engine.hold_paint_while_pose_in_doubt(True)
    engine._tick()  # noqa: SLF001
    assert _painted(engine) == before + 1


def test_repeating_the_doubt_does_not_extend_the_hold(tmp_path):
    engine = _engine(tmp_path)
    engine.hold_paint_while_pose_in_doubt(True)
    until = engine._pose_in_doubt_until  # noqa: SLF001
    engine.hold_paint_while_pose_in_doubt(True)
    assert engine._pose_in_doubt_until == until  # noqa: SLF001
    # A new doubt, after this one was settled, gets its own time.
    engine.hold_paint_while_pose_in_doubt(False)
    engine.hold_paint_while_pose_in_doubt(True)
    assert engine._pose_in_doubt_until >= until  # noqa: SLF001

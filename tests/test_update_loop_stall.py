"""What SLAM does when its update loop falls behind.

On tracer2a (1.3.1-rc11) the loop stalled seven times in a five-minute explore
run, 1.1 to 4.0 s each, with no background job running. One stall of 2.8 s
came in the middle of a turn at 32 deg/s. The 91 degrees the loop saw when it
came back were refused as an odometry jump, the heading was left a quarter
turn out, and every meter driven after that took the position with it.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from src.geom import conversions as conv
from src.slam_builtin.engine import BuiltinSlamEngine

from .test_paint_fit_heights import _engine, _room_ranges, _scan

TURN = math.radians(-32.0)  # rad/s, as on the robot


def _reading(heading_deg: float, yaw_rate: float) -> conv.OdomReading:
    heading = math.radians(heading_deg)
    return conv.OdomReading(
        0.0,
        0.0,
        yaw_rate,
        pose=conv.Pose2D(0.0, 0.0, heading),
        heading_rad=heading,
    )


def _heading_after(engine: BuiltinSlamEngine, first, then, gap_s: float) -> float:
    with engine._lock:  # noqa: SLF001
        engine._pose = engine._predict(first, 100.0)  # noqa: SLF001
        start = engine._pose.theta  # noqa: SLF001
        engine._pose = engine._predict(then, 100.0 + gap_s)  # noqa: SLF001
        return math.degrees(conv.normalize_angle(engine._pose.theta - start))  # noqa: SLF001


def test_a_turn_under_way_through_a_stall_is_kept(tmp_path):
    engine = _engine(tmp_path)
    turned = _heading_after(
        engine, _reading(0.0, TURN), _reading(-91.0, TURN), gap_s=2.8
    )
    assert turned == pytest.approx(-91.0, abs=0.5)


def test_a_turn_that_began_during_the_stall_is_kept(tmp_path):
    """Standing before the gap, turning after it: the gyro still shows it."""
    engine = _engine(tmp_path)
    turned = _heading_after(
        engine, _reading(0.0, 0.0), _reading(-80.0, TURN), gap_s=2.8
    )
    assert turned == pytest.approx(-80.0, abs=0.5)


def test_a_heading_flip_on_a_parked_robot_is_still_refused(tmp_path):
    """No turn rate before or after: a magnetometer snap, however late the read."""
    engine = _engine(tmp_path)
    said = []
    engine._log = said.append  # noqa: SLF001
    turned = _heading_after(engine, _reading(0.0, 0.0), _reading(-91.0, 0.0), gap_s=2.8)
    assert turned == pytest.approx(0.0, abs=0.5)
    assert any("rejected odom jump" in line for line in said)


def test_a_snap_larger_than_the_turn_could_make_is_refused(tmp_path):
    """Turning at 32 deg/s for a tenth of a second is 3 degrees, not 40."""
    engine = _engine(tmp_path)
    turned = _heading_after(
        engine, _reading(0.0, TURN), _reading(-40.0, TURN), gap_s=0.1
    )
    assert turned == pytest.approx(0.0, abs=0.5)


def test_a_turn_is_painted_from_a_handful_of_its_scans(tmp_path):
    """Forty scans held through a turn used to be painted in one update."""
    engine = _engine(tmp_path)
    headings = np.radians(np.linspace(-170.0, 0.0, 40))
    held = []
    for k, theta in enumerate(headings[:-1]):
        pose = conv.Pose2D(0.0, 0.0, float(theta))
        held.append((_scan(_room_ranges(pose)), pose, [], 10.0 + 0.1 * k))
    last = conv.Pose2D(0.0, 0.0, float(headings[-1]))
    with engine._lock:  # noqa: SLF001
        engine._pose = last  # noqa: SLF001
        engine._spin_hold = held  # noqa: SLF001
    painted = []
    paint = engine._paint_mapping_scan  # noqa: SLF001

    def record(scan, pose, now, **kw):
        painted.append(round(math.degrees(pose.theta)))
        return paint(scan, pose, now, **kw)

    engine._paint_mapping_scan = record  # noqa: SLF001
    engine._flush_spin_hold(_scan(_room_ranges(last)), 14.0, None)  # noqa: SLF001

    assert len(painted) == BuiltinSlamEngine._SPIN_PAINT_MAX  # noqa: SLF001
    # Spread over the whole turn, and the stopped scan at its end is one.
    assert painted[0] == -170 and painted[-1] == 0
    assert all(b - a >= 20 for a, b in zip(painted, painted[1:]))
    assert engine._spin_hold == []  # noqa: SLF001


def test_a_short_turn_paints_every_scan_it_held(tmp_path):
    engine = _engine(tmp_path)
    poses = [conv.Pose2D(0.0, 0.0, math.radians(a)) for a in (-30.0, -15.0, 0.0)]
    with engine._lock:  # noqa: SLF001
        engine._pose = poses[-1]  # noqa: SLF001
        engine._spin_hold = [  # noqa: SLF001
            (_scan(_room_ranges(p)), p, [], 10.0 + k) for k, p in enumerate(poses[:-1])
        ]
    before = engine._updates  # noqa: SLF001
    engine._flush_spin_hold(_scan(_room_ranges(poses[-1])), 13.0, None)  # noqa: SLF001
    assert engine._updates - before == 3  # noqa: SLF001


def test_a_stall_reports_where_the_loop_was(tmp_path):
    engine = _engine(tmp_path)
    said = []
    engine._log = said.append  # noqa: SLF001
    engine._stall_where = engine._where(sys._getframe())  # noqa: SLF001
    engine._note_stall(2.8, 1000.0)  # noqa: SLF001
    assert "stalled 2.8 s" in said[0]
    assert "the loop was in test_update_loop_stall.py:" in said[0]
    assert "test_a_stall_reports_where_the_loop_was" in said[0]
    # Said once: the next stall starts with nothing noted.
    engine._note_stall(1.5, 1010.0)  # noqa: SLF001
    assert "the loop was in" not in said[1]

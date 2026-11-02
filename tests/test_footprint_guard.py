"""Unit tests for the rectangular footprint collision guard."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.nav_builtin.footprint_guard import (
    FootprintGuard,
    GuardConfig,
    open_corridor_yaw,
)
from src.nav_builtin.types import Pose2D

L, W = 0.72, 0.59
HL, HW = L / 2, W / 2


def _guard(**kw) -> FootprintGuard:
    return FootprintGuard(GuardConfig(length_m=L, width_m=W, padding_m=0.04, **kw))


def _wall(x0, y0, x1, y1, step=0.02) -> np.ndarray:
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


O = Pose2D(0.0, 0.0, 0.0)


def test_sub_deadband_crawl_is_not_sent():
    """A 1.5 cm/s crawl is a stop on the base. Don't report it as motion."""
    g = _guard()
    far = _wall(4.0, -0.3, 4.0, 0.3)
    res = g.guard(O, 0.015, 0.025, far, allow_steer=False)
    assert res.vx == 0.0 and res.vtheta == 0.0
    assert res.state == "blocked"
    # Room ahead and a target off-axis: steer or yaw at a rate the base accepts.
    aimed = g.guard(
        O, 0.015, 0.025, far, target=Pose2D(1.0, 1.0, 0.0), allow_steer=True
    )
    assert aimed.state in ("steer", "rotate")
    assert abs(aimed.vx) >= 0.05 or abs(aimed.vtheta) >= 0.08


def test_obstacle_ahead_regulates_speed_to_a_stop():
    g = _guard()
    pts = _wall(HL + 0.5, -0.3, HL + 0.5, 0.3)
    free = g.free_distance(O, 0.3, 0.0, pts, 0.8)
    assert free == pytest.approx(0.5 - 0.04, abs=0.03)
    far = g.guard(O, 0.4, 0.0, pts, allow_steer=False)
    assert 0.0 < far.vx < 0.4 and far.state == "slow"
    close = g.guard(
        O, 0.4, 0.0, _wall(HL + 0.05, -0.3, HL + 0.05, 0.3), allow_steer=False
    )
    assert close.vx == 0.0 and close.state == "blocked"


def test_doorway_jambs_beside_body_do_not_block():
    """rc23 stopped at every 0.85 m door: jambs sat inside a padded lidar
    corridor. The rectangle clears them by 13 cm per side."""
    g = _guard()
    jambs = np.concatenate(
        [_wall(0.5, 0.425, 0.6, 0.425), _wall(0.5, -0.425, 0.6, -0.425)]
    )
    assert math.isinf(g.free_distance(O, 0.3, 0.0, jambs, 1.5))
    res = g.guard(O, 0.3, 0.0, jambs, allow_steer=False)
    assert res.vx == pytest.approx(0.3) and res.state == "clear"


def test_bumper_contact_does_not_drive_straight():
    """A steer whose yaw the base drops is a straight push into the bumper.

    Turn in place instead when that sweep is free.
    """
    g = _guard()
    # Just off the front-left of the nose: straight is blocked, spin is not.
    blob = np.array([[HL + 0.06, 0.12]])
    res = g.guard(O, 0.3, 0.0, blob, target=Pose2D(1.0, -0.8, 0.0), allow_steer=True)
    assert not (res.vx > 0.05 and abs(res.vtheta) < 0.08)
    assert abs(res.vtheta) >= 0.08


def test_shoulder_obstacle_inside_body_width_blocks():
    """The rc24 bin: at the shoulder, lidar cone ahead empty, body would hit it."""
    g = _guard()
    bin_pts = _wall(HL + 0.3, HW - 0.1, HL + 0.3, HW + 0.3)
    assert g.free_distance(O, 0.3, 0.0, bin_pts, 0.8) < 0.3
    res = g.guard(O, 0.3, 0.0, bin_pts, target=Pose2D(3.0, 0.0, 0.0), allow_steer=True)
    # Steers right (away from the bin on the left), never reverses.
    assert res.state == "steer" and res.vx > 0.0 and res.vtheta < 0.0


def test_self_returns_inside_body_are_ignored():
    g = _guard()
    mast = np.array([[0.1, 0.05], [-0.2, -0.1]])
    assert math.isinf(g.free_distance(O, 0.3, 0.0, mast, 0.8))
    assert math.isinf(g.free_rotation(O, 1.0, mast))


def test_rotation_blocked_when_corner_sweep_hits():
    """Half-diagonal 0.465 m: a wall 0.40 m off the side is clear to drive
    along but a spin swings the corner into it."""
    g = _guard()
    wall = _wall(-1.0, 0.40, 1.0, 0.40)
    assert math.isinf(g.free_distance(O, 0.3, 0.0, wall, 0.8))
    assert g.free_rotation(O, 1.0, wall) < 0.3
    res = g.guard(O, 0.0, 0.6, wall, allow_steer=False)
    assert res.vtheta == 0.0 and res.rotation_blocked


def test_forward_nominal_never_becomes_reverse():
    g = _guard()
    box = np.concatenate(
        [
            _wall(HL + 0.06, -1.0, HL + 0.06, 1.0),
            _wall(-1.0, HW + 0.06, 1.0, HW + 0.06),
            _wall(-1.0, -HW - 0.06, 1.0, -HW - 0.06),
        ]
    )
    res = g.guard(O, 0.3, 0.2, box, target=Pose2D(3.0, 0.0, 0.0))
    assert res.vx >= 0.0
    assert res.state == "blocked"


def test_low_speed_arc_survives_base_sanitizer():
    """vx < 0.12 with |w| > 0.25 is turned into a spin by the base."""
    g = _guard()
    res = g.guard(O, 0.08, 0.6, np.empty((0, 2)), allow_steer=False)
    assert abs(res.vtheta) <= 0.25 + 1e-9


def test_leg_inside_padding_may_be_passed_but_not_approached():
    """Unmapped table leg 5 cm off the right side (inside the 10 cm band):
    driving past is fine, curving onto it is not (live grazes, rc27)."""
    g = FootprintGuard(GuardConfig(length_m=L, width_m=W, padding_m=0.10))
    leg = np.array([[0.1, -(HW + 0.05)]])
    straight = g.free_distance(O, 0.3, 0.0, leg, 0.8)
    assert math.isinf(straight)
    toward = g.free_distance(O, 0.3, -0.6, leg, 0.8)
    assert toward < 0.3
    res = g.guard(O, 0.3, -0.6, leg, allow_steer=False)
    assert res.vx < 0.3 or abs(res.vtheta) < 0.6


def _nose_cluster() -> np.ndarray:
    """Blob ~20 cm past the nose, slightly right of center (live F4 pose)."""
    xs = np.linspace(0.56, 0.77, 16)
    ys = np.linspace(-0.18, -0.01, 12)
    xx, yy = np.meshgrid(xs, ys)
    cluster = np.stack([xx.ravel(), yy.ravel()], axis=1)
    angs = np.radians(np.arange(70, 110, 3))
    left = np.stack([0.9 * np.cos(angs), 0.9 * np.sin(angs)], axis=1)
    return np.vstack([cluster, left])


def test_open_corridor_yaw_aims_ccw_through_the_notch():
    """Straight is blocked. The open heading nearest the goal is counter-clockwise."""
    pts = _nose_cluster()
    hl, hw = 0.36, 0.365
    # 0° travel is ~0.2 m, so the search must leave the blocked heading.
    aimed = open_corridor_yaw(
        pts,
        half_length_m=hl,
        half_width_m=hw,
        prefer_rad=0.0,
        min_travel_m=0.8,
    )
    assert aimed is not None and aimed > math.radians(20)
    prefer = math.radians(46.0)
    yaw = open_corridor_yaw(
        pts,
        half_length_m=hl,
        half_width_m=hw,
        prefer_rad=prefer,
        min_travel_m=0.8,
    )
    assert yaw is not None
    assert abs(yaw - prefer) < math.radians(8)
    mirrored = pts.copy()
    mirrored[:, 1] *= -1.0
    other = open_corridor_yaw(
        mirrored,
        half_length_m=hl,
        half_width_m=hw,
        prefer_rad=-prefer,
        min_travel_m=0.8,
    )
    assert other is not None and other < 0
    assert abs(other + prefer) < math.radians(8)


def test_open_corridor_yaw_none_when_every_heading_is_short():
    angs = np.linspace(-math.pi, math.pi, 80, endpoint=False)
    ring = np.stack([0.7 * np.cos(angs), 0.7 * np.sin(angs)], axis=1)
    assert (
        open_corridor_yaw(
            ring,
            half_length_m=0.36,
            half_width_m=0.365,
            prefer_rad=math.radians(46),
            min_travel_m=0.8,
        )
        is None
    )


def test_turning_away_does_not_swing_tail_into_leg_beside_rear():
    """Turning left with a leg beside the rear-right swings the tail into it."""
    g = FootprintGuard(GuardConfig(length_m=L, width_m=W, padding_m=0.10))
    leg = np.array([[-0.25, -(HW + 0.05)]])
    assert math.isinf(g.free_distance(O, 0.3, 0.0, leg, 0.8))
    assert g.free_distance(O, 0.3, 0.9, leg, 0.8) < 0.2
    assert g.free_rotation(O, 1.0, leg) < 0.1

"""Footprint guard when the body is almost touching an obstacle.

Live tracer2a (1.3.1-rc0): something within 2 cm of the body blocked every
motion, so the robot could not even reverse away from it.
"""

import math

import numpy as np
import pytest

from src.nav.control.footprint_guard import FootprintGuard, GuardConfig
from src.nav.types import Pose2D

POSE = Pose2D(0.0, 0.0, 0.0)
HL, HW = 0.36, 0.295


def _guard():
    return FootprintGuard(GuardConfig(length_m=2 * HL, width_m=2 * HW, padding_m=0.04))


def _target(bearing_deg):
    b = math.radians(bearing_deg)
    return Pose2D(3.0 * math.cos(b), 3.0 * math.sin(b), 0.0)


@pytest.mark.parametrize("gap", [0.005, 0.015])
def test_can_leave_a_point_closer_than_the_minimum_gap(gap):
    g = _guard()
    flank = np.array([[0.0, HW + gap]])
    # Sliding past it does not close the gap; before, both were 0.
    assert g.free_distance(POSE, 0.2, 0.0, flank, 1.0) > 0.5
    assert g.free_distance(POSE, -0.12, 0.0, flank, 0.5) >= 0.5
    nose = np.array([[HL + gap, 0.1]])
    assert g.free_distance(POSE, -0.12, 0.0, nose, 0.5) >= 0.5
    # Driving into it is still refused.
    assert g.free_distance(POSE, 0.2, 0.0, nose, 1.0) == 0.0


@pytest.mark.parametrize("gap", [0.005, 0.015])
def test_may_not_get_closer_to_a_touching_point(gap):
    g = _guard()
    # A wall of points along the left flank: turning either way swings a
    # corner into it.
    wall = np.array([[x, HW + gap] for x in np.linspace(-0.6, 0.6, 25)])
    assert g.free_rotation(POSE, 1.0, wall) == 0.0
    assert g.free_rotation(POSE, -1.0, wall) == 0.0


def test_boxed_in_on_every_side_stays_blocked():
    g = _guard()
    desk = [[HL + 0.015, y] for y in np.linspace(-0.6, 0.6, 25)]
    behind = [[-HL - 0.05, y] for y in np.linspace(-0.6, 0.6, 25)]
    res = g.guard(
        POSE, 0.0, 0.5, np.array(desk + behind), target=_target(84), max_dist_m=3.0
    )
    assert res.state == "blocked" and res.vx == 0.0 and res.vtheta == 0.0

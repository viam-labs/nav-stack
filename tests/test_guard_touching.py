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


@pytest.mark.parametrize("bearing", [84, -84, 150])
def test_nose_against_a_desk_backs_out_whatever_the_bearing(bearing):
    g = _guard()
    desk = np.array([[HL + 0.015, y] for y in np.linspace(-0.6, 0.6, 25)])
    res = g.guard(
        POSE,
        0.0,
        math.copysign(0.5, bearing),
        desk,
        target=_target(bearing),
        max_dist_m=3.0,
    )
    assert res.state == "back_out" and res.vx < 0.0 and res.vtheta == 0.0


def test_boxed_in_on_every_side_stays_blocked():
    g = _guard()
    desk = [[HL + 0.015, y] for y in np.linspace(-0.6, 0.6, 25)]
    behind = [[-HL - 0.05, y] for y in np.linspace(-0.6, 0.6, 25)]
    res = g.guard(
        POSE, 0.0, 0.5, np.array(desk + behind), target=_target(84), max_dist_m=3.0
    )
    assert res.state == "blocked" and res.vx == 0.0 and res.vtheta == 0.0


def _drive(g, pts, target, pose, ticks=400, dt=0.05):
    """Ask for a left turn every tick and integrate what the guard allows."""
    flips, last, xs = 0, 0, []
    for _ in range(ticks):
        res = g.guard(pose, 0.0, 0.5, pts, target=target, max_dist_m=3.0)
        sign = (res.vx > 1e-3) - (res.vx < -1e-3)
        if sign and last and sign != last:
            flips += 1
        last = sign or last
        th = pose.theta + res.vtheta * dt
        pose = Pose2D(
            pose.x + res.vx * math.cos(th) * dt, pose.y + res.vx * math.sin(th) * dt, th
        )
        xs.append(pose.x)
    return flips, xs, pose


@pytest.mark.parametrize(
    "desk_m,leg_x,leg_gap,bearing",
    [(0.2, -0.05, 0.06, 80), (0.3, 0.2, 0.03, 40), (0.1, -0.05, 0.06, 40)],
)
def test_making_room_to_turn_commits_to_a_direction(desk_m, leg_x, leg_gap, bearing):
    # Live tracer2a (1.3.1-rc1): a leg a few cm off the right flank blocks
    # the turn and a desk ahead leaves a short forward arc. The guard chose
    # forward or reverse afresh every tick, flipped about every 0.3 s and
    # stayed inside 9 cm for 30 s. These layouts flipped 46-76 times in 20 s.
    g = _guard()
    leg = [[leg_x, -HW - leg_gap]]
    desk = [[HL + desk_m, y] for y in np.linspace(-0.8, 0.8, 33)]
    flips, xs, pose = _drive(g, np.array(leg + desk), _target(bearing), POSE)
    assert flips <= 2, flips
    # It makes room and turns toward the route.
    assert pose.theta > math.radians(bearing - 15.0)

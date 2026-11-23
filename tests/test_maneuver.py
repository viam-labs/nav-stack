"""Multi-point manoeuvres for a turn the body cannot make where it stands."""

import math
import time

import numpy as np
import pytest

from src.nav.control.footprint_guard import FootprintGuard, GuardConfig, _rect_dist
from src.nav.control.maneuver import BACK, FORWARD, LEFT, RIGHT, plan_maneuver
from src.nav.types import Pose2D

HL, HW = 0.36, 0.295
START = Pose2D(0.0, 0.0, 0.0)


def _guard():
    return FootprintGuard(GuardConfig(length_m=2 * HL, width_m=2 * HW, padding_m=0.04))


def _target(bearing_deg, dist=1.0):
    b = math.radians(bearing_deg)
    return Pose2D(dist * math.cos(b), dist * math.sin(b), 0.0)


def _line(x0, y0, x1, y1):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.03))
    return [
        [x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)] for i in range(n)
    ]


def _boxed_like_tracer2a():
    """Furniture 6 cm off the right-rear flank, a cabinet 22 cm past the nose
    on the right, a wall 25 cm behind; open to the left."""
    return np.array(
        _line(-0.35, -HW - 0.06, -0.10, -HW - 0.06)
        + _line(HL + 0.22, -0.5, HL + 0.22, 0.05)
        + _line(-HL - 0.25, -0.5, -HL - 0.25, 0.3)
    )


def _pocket(width=0.85):
    """Nose-in: walls each side and across the nose, open 0.3 m behind centre."""
    h = width / 2
    return np.array(
        _line(-0.3, h, 1.2, h) + _line(-0.3, -h, 1.2, -h) + _line(1.2, -h, 1.2, h)
    )


def _run(pts, bearing, ticks=1600, dt=0.05):
    """Follow the guard's answers to a blocked turn with ideal kinematics.

    Returns (final pose, closest approach of the body to any point, seconds).
    """
    g = _guard()
    target = _target(bearing)
    pose = START
    closest = math.inf
    for i in range(ticks):
        want = conv_bearing(pose, target)
        if abs(want) <= math.radians(20.0):
            return pose, closest, i * dt
        res = g.guard(
            pose,
            0.0,
            math.copysign(0.6, want),
            pts,
            target=target,
            max_dist_m=3.0,
            plan=True,
        )
        th = pose.theta + res.vtheta * dt
        pose = Pose2D(
            pose.x + res.vx * math.cos(th) * dt, pose.y + res.vx * math.sin(th) * dt, th
        )
        here = np.array([[pose.x, pose.y, pose.theta]])
        closest = min(closest, float(_rect_dist(here, pts, HL, HW).min()))
    return pose, closest, ticks * dt


def conv_bearing(pose, target):
    a = math.atan2(target.y - pose.y, target.x - pose.x) - pose.theta
    return math.atan2(math.sin(a), math.cos(a))


def test_leg_beside_the_flank_backs_clear_then_turns():
    pts = np.array([[0.05, -HW - 0.08]])
    plan = plan_maneuver(_guard(), START, pts, _target(130))
    assert [s.kind for s in plan] == [BACK, LEFT]
    assert plan[0].amount <= 0.5 and plan[1].amount >= math.radians(90)


def test_nose_against_a_desk_backs_off_before_turning():
    desk = np.array(_line(HL + 0.015, -0.6, HL + 0.015, 0.6))
    for bearing in (84, -84, 150):
        plan = plan_maneuver(_guard(), START, desk, _target(bearing))
        assert plan[0].kind == BACK, bearing
        assert plan[-1].kind == (LEFT if bearing > 0 else RIGHT), bearing


def test_pocket_reverses_out_and_does_not_turn_inside():
    plan = plan_maneuver(_guard(), START, _pocket(), _target(180))
    assert plan[0].kind == BACK and plan[0].amount >= 0.5
    assert all(s.kind != FORWARD for s in plan)


def test_boxed_in_like_tracer2a_finds_a_multi_point_way_out():
    # Live tracer2a (1.3.1-rc3): 30 s shuffling 13 cm forward and back, no
    # turn at all, with open floor on the left.
    plan = plan_maneuver(_guard(), START, _boxed_like_tracer2a(), _target(94))
    assert plan is not None
    kinds = {s.kind for s in plan}
    # It has to turn the "wrong" way and back up at some point.
    assert RIGHT in kinds and BACK in kinds and plan[-1].kind == LEFT


def test_no_plan_when_boxed_on_every_side():
    box = np.array(
        _line(-0.5, 0.36, 0.5, 0.36)
        + _line(-0.5, -0.36, 0.5, -0.36)
        + _line(0.44, -0.36, 0.44, 0.36)
        + _line(-0.44, -0.36, -0.44, 0.36)
    )
    assert plan_maneuver(_guard(), START, box, _target(90)) is None


def test_search_is_fast_enough_for_the_control_loop():
    # Worst case: room to shuffle but closed all round, so the whole reachable
    # space is searched and nothing is found.
    w, length = 1.0, 1.6
    box = np.array(
        _line(-length / 2, w / 2, length / 2, w / 2)
        + _line(-length / 2, -w / 2, length / 2, -w / 2)
        + _line(length / 2, -w / 2, length / 2, w / 2)
        + _line(-length / 2, -w / 2, -length / 2, w / 2)
    )
    g = _guard()
    plan_maneuver(g, START, box, _target(90, 3.0))  # warm numpy
    t = time.perf_counter()
    for pts in (box, _boxed_like_tracer2a(), _pocket()):
        plan_maneuver(g, START, pts, _target(90, 3.0))
    assert time.perf_counter() - t < 1.5  # ~0.1 s on a laptop


@pytest.mark.parametrize(
    "name,bearing",
    [("flank", 130), ("desk", 84), ("pocket", 180), ("boxed", 94), ("boxed", 150)],
)
def test_following_the_plan_turns_to_the_route_without_touching(name, bearing):
    pts = {
        "flank": np.array([[0.05, -HW - 0.08]]),
        "desk": np.array(_line(HL + 0.015, -0.6, HL + 0.015, 0.6)),
        "pocket": _pocket(),
        "boxed": _boxed_like_tracer2a(),
    }[name]
    start_gap = float(_rect_dist(np.array([[0.0, 0.0, 0.0]]), pts, HL, HW).min())
    pose, closest, seconds = _run(pts, bearing)
    assert abs(conv_bearing(pose, _target(bearing))) <= math.radians(20.0), seconds
    # Never closer than it started, less 1 cm, and never touching.
    assert closest >= min(start_gap, 0.04) - 0.01
    assert seconds < 60.0


def test_veto_calls_do_not_disturb_a_running_manoeuvre():
    pts = _boxed_like_tracer2a()
    g = _guard()
    target = _target(94)
    first = g.guard(START, 0.0, 0.6, pts, target=target, max_dist_m=3.0, plan=True)
    assert first.state == "maneuver" and g.maneuver.active
    left = g.maneuver.plan
    # The final gate re-checks commands with plan=False, forward ones too.
    g.guard(START, 0.12, 0.0, pts, max_dist_m=3.0, allow_steer=False)
    g.guard(START, 0.0, 0.3, pts, max_dist_m=3.0, allow_steer=False)
    assert g.maneuver.active and g.maneuver.plan == left
    # The follower driving forward again ends it.
    g.guard(START, 0.2, 0.0, np.empty((0, 2)), target=target, plan=True)
    assert not g.maneuver.active


def test_turn_direction_is_kept_when_the_route_is_nearly_behind():
    # Live tracer2a and the boxed-in sim: with the lookahead point almost
    # straight behind, its bearing flipped between +160 and -153 degrees and
    # the follower reversed its turn each time, rocking 2 degrees for good.
    from src.nav.control.controller import FollowerConfig, pursuit_command
    from src.nav.motion.simple_motion import DriveCommand

    cfg = FollowerConfig()
    here = Pose2D(0.0, 0.0, 0.0)
    turning_left = DriveCommand(0.0, 0.0, 0.5, False)
    behind_right = Pose2D(-1.0, -0.4, 0.0)  # bearing about -158 degrees
    cmd, rotating = pursuit_command(
        here, behind_right, cfg=cfg, rotate_active=True, prev_cmd=turning_left
    )
    assert rotating and cmd.vx == 0.0 and cmd.vtheta > 0.0
    # From rest, or for a turn under 135 degrees, the nearer way is taken.
    cmd, _ = pursuit_command(here, behind_right, cfg=cfg, rotate_active=True)
    assert cmd.vtheta < 0.0
    side_right = Pose2D(0.0, -1.0, 0.0)  # bearing -90 degrees
    cmd, _ = pursuit_command(
        here, side_right, cfg=cfg, rotate_active=True, prev_cmd=turning_left
    )
    assert cmd.vtheta < 0.0


def test_route_passing_close_to_an_obstacle_still_gets_a_manoeuvre():
    # Live tracer2a (1.3.1-rc4): nose 7.6 cm from furniture, route 74 degrees
    # to the right through a gap, open floor behind. A check that refused to
    # plan toward a lookahead point within half a body width of any obstacle
    # (it was 20 cm from one) left the robot at zero command for 30 s.
    ahead = _line(HL + 0.076, -0.5, HL + 0.076, 0.5)
    target = _target(-74, 0.6)
    beside_route = [[target.x + 0.2, target.y]]
    pts = np.array(ahead + beside_route)
    g = _guard()
    res = g.guard(START, 0.0, -0.6, pts, target=target, max_dist_m=4.0, plan=True)
    assert res.state == "maneuver" and g.maneuver.active
    assert res.vx < 0.0  # backs away from the furniture first
    pose, closest = START, math.inf
    for _ in range(1200):
        want = conv_bearing(pose, target)
        if abs(want) <= math.radians(20.0):
            break
        res = g.guard(
            pose, 0.0, math.copysign(0.6, want), pts, target=target, plan=True
        )
        th = pose.theta + res.vtheta * 0.05
        pose = Pose2D(
            pose.x + res.vx * math.cos(th) * 0.05,
            pose.y + res.vx * math.sin(th) * 0.05,
            th,
        )
        here = np.array([[pose.x, pose.y, pose.theta]])
        closest = min(closest, float(_rect_dist(here, pts, HL, HW).min()))
    assert abs(conv_bearing(pose, target)) <= math.radians(20.0)
    assert closest >= 0.03

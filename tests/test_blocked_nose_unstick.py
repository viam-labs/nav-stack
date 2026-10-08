"""Blocked-nose unstick: reverse judged by the guard sweep, then yaw away.

Live 2026-10-08 (tracer2a): wedged on a desk corner at the front-left, the
supervisor never reversed in 84 s and turned under 20 deg, because the rear
check used a 120 deg lidar cone (the desk just left sits inside it) plus the
inflated costmap, and the yaw sign was a constant.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.nav_builtin.footprint_guard import FootprintGuard, GuardConfig
from src.nav_builtin.types import Pose2D
from tests.test_recovery_policies import _sup

HERE = Pose2D(0.0, 0.0, 0.0)
FRONT_LEFT_CORNER = (0.33, 0.275)  # the live desk point, body frame == world at HERE


def _sup_with_guard(**kw):
    sup, world = _sup(backup_dist_m=0.5, **kw)
    if sup._guard is None:  # noqa: SLF001
        sup._guard = FootprintGuard(GuardConfig(length_m=0.72, width_m=0.59))  # noqa: SLF001
    return sup, world


def test_rear_open_uses_guard_sweep_not_rear_cone():
    sup, _ = _sup_with_guard()
    pts = np.array([FRONT_LEFT_CORNER])
    rear_open, rev = sup._unstick_rear_open(HERE, None, None, pts)  # noqa: SLF001
    assert rear_open is True
    assert rev is not None and rev.vx < 0.0 and abs(rev.vtheta) < 1e-9


def test_rear_closed_when_something_sits_behind_the_body():
    sup, _ = _sup_with_guard()
    pts = np.array([FRONT_LEFT_CORNER, (-0.41, 0.0)])  # 5 cm behind the tail
    rear_open, rev = sup._unstick_rear_open(HERE, None, None, pts)  # noqa: SLF001
    assert rear_open is False and rev is None


def test_rear_open_without_guard_needs_scan_and_costmap():
    sup, _ = _sup_with_guard()
    sup._guard = None  # noqa: SLF001
    assert sup._unstick_rear_open(HERE, None, None, None) == (False, None)  # noqa: SLF001


@pytest.mark.parametrize(
    "point,expected",
    [((0.33, 0.275), -1.0), ((0.33, -0.275), 1.0), ((0.5, 0.0), 1.0)],
)
def test_yaw_sign_turns_away_from_the_nearest_obstacle(point, expected):
    sup, _ = _sup_with_guard()
    assert sup._unstick_yaw_sign(HERE, np.array([point]), None) == expected  # noqa: SLF001


def test_yaw_sign_default_left_without_information():
    sup, _ = _sup_with_guard()
    sup._guard = None  # noqa: SLF001
    assert sup._unstick_yaw_sign(HERE, None, None) == 1.0  # noqa: SLF001


def test_spin_clear_false_without_scan():
    sup, _ = _sup_with_guard()
    assert sup._unstick_spin_clear(None, None, HERE) is False  # noqa: SLF001


def test_unstick_yaw_knob_is_read_and_floored():
    sup, _ = _sup(unstick_yaw_rad=1.2)
    assert sup._unstick_yaw_rad == pytest.approx(1.2)  # noqa: SLF001
    sup, _ = _sup(unstick_yaw_rad=-3.0)
    assert sup._unstick_yaw_rad == 0.0  # noqa: SLF001
    sup, _ = _sup()
    assert sup._unstick_yaw_rad == pytest.approx(0.8)  # noqa: SLF001


def test_decision_table_prefers_reverse_then_turn():
    sup, _ = _sup()
    f = sup._blocked_nose_unstick  # noqa: SLF001
    assert f(failed_replans=0, nose_clear=False, rear_open=True, spin_clear=True) == "hold"
    assert f(failed_replans=1, nose_clear=True, rear_open=True, spin_clear=True) == "hold"
    assert f(failed_replans=1, nose_clear=False, rear_open=True, spin_clear=True) == "reverse"
    assert f(failed_replans=1, nose_clear=False, rear_open=False, spin_clear=True) == "turn"
    assert f(failed_replans=1, nose_clear=False, rear_open=False, spin_clear=False) == "hold"


def test_config_default_and_kwargs_plumbing():
    from src.config import NavConfig
    from src.nav_builtin.runtime_kwargs import builtin_nav_runtime_kwargs

    cfg = NavConfig.from_dict({"slam_service": "s", "base": "b", "builtin": {"unstick_yaw_rad": 1.0}})
    assert cfg.builtin.unstick_yaw_rad == pytest.approx(1.0)
    assert builtin_nav_runtime_kwargs(cfg)["unstick_yaw_rad"] == pytest.approx(1.0)
    assert math.isclose(NavConfig.from_dict({"slam_service": "s", "base": "b"}).builtin.unstick_yaw_rad, 0.8)

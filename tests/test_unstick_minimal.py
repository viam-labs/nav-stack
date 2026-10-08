"""Minimal unstick changes (2026-10-08): commit to the reverse, paint depth
memory into replans, turn toward the freer side."""
from __future__ import annotations

import math

import numpy as np
import pytest

import tests.tight_space_sim as sim
from src.geom import conversions as conv
from src.nav_builtin.footprint_guard import FootprintGuard, GuardConfig
from src.nav_builtin.supervisor import NavSupervisor
from src.nav_builtin.types import Pose2D
from tests.test_nav_builtin import _FakeWorld, _empty_map


def _sup(**kw):
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map())
    sup = NavSupervisor(world, **kw)
    # tracer2a-sized guard regardless of the fake world's defaults.
    sup._guard = FootprintGuard(GuardConfig(length_m=0.72, width_m=0.59))  # noqa: SLF001
    return sup, world


def test_depth_memory_is_painted_into_the_plan_scan():
    sup, _ = _sup()
    pose = Pose2D(2.0, 3.0, math.pi / 2)  # facing +y
    # Remembered point 0.5 m ahead of the robot in the world (= +y).
    sup._depth_memory._cells[(0, 0)] = (2.0, 3.5, 0.0)  # noqa: SLF001
    scan = sup._scan_with_depth_memory(None, pose)  # noqa: SLF001
    assert scan is not None
    pts = scan.to_points()
    assert len(pts) == 1
    assert pts[0] == pytest.approx((0.5, 0.0), abs=0.02)  # straight ahead in base frame


def test_depth_memory_merges_with_existing_scan():
    sup, _ = _sup()
    pose = Pose2D(0.0, 0.0, 0.0)
    base = conv.points_to_scan(np.array([(1.0, 0.0)]), num_bins=360, range_min=0.0, range_max=10.0)
    sup._depth_memory._cells[(0, 0)] = (0.0, 0.6, 0.0)  # noqa: SLF001  # left of the robot
    merged = sup._scan_with_depth_memory(base, pose)  # noqa: SLF001
    pts = merged.to_points()
    assert len(pts) == 2
    assert sorted(round(float(p[0]), 2) for p in pts) == [0.0, 1.0]


def test_no_depth_memory_leaves_scan_untouched():
    sup, _ = _sup()
    base = conv.points_to_scan(np.array([(1.0, 0.0)]), num_bins=360, range_min=0.0, range_max=10.0)
    assert sup._scan_with_depth_memory(base, Pose2D(0, 0, 0)) is base  # noqa: SLF001
    assert sup._scan_with_depth_memory(None, Pose2D(0, 0, 0)) is None  # noqa: SLF001


@pytest.mark.parametrize("side,expected", [(0.33, -1.0), (-0.33, 1.0)])
def test_turn_sign_prefers_the_freer_side(side, expected):
    sup, _ = _sup()
    # A desk edge running forward from beside a front corner (the live
    # desk-corner case): rotating toward that side swings the front corner
    # into the edge within a few degrees, the other way swings it clear.
    xs = np.linspace(0.30, 0.90, 13)
    pts = np.column_stack([xs, np.full_like(xs, side)])
    assert sup._unstick_turn_sign(Pose2D(0, 0, 0), pts) == expected  # noqa: SLF001


def test_turn_sign_defaults_ccw_without_information():
    sup, _ = _sup()
    assert sup._unstick_turn_sign(Pose2D(0, 0, 0), None) == 1.0  # noqa: SLF001
    assert sup._unstick_turn_sign(Pose2D(0, 0, 0), np.empty((0, 2))) == 1.0  # noqa: SLF001


def test_low_pallet_only_the_camera_sees_is_routed_around(monkeypatch):
    """Live 2026-10-08: an unmapped cardboard pallet below the lidar plane.
    The guard (depth memory) refused, the planner never saw it and returned
    the same route after every short reverse: a 0.3 s reverse / forward /
    rotate loop. Expect: back off, replan around it, reach the goal."""
    g = sim.blank(12.0, 6.0)
    pallet = (4.0, 2.2, 4.6, 3.8)  # 0.6 m deep, 1.6 m wide, across the route
    r = sim.run(
        g,
        Pose2D(1.0, 3.0, 0.0),
        Pose2D(10.0, 3.0, 0.0),
        body_only_obstacles=[pallet],
        monkeypatch=monkeypatch,
        max_sim_s=90.0,
        clearance_m=0.1,
        lidar_offset=(0.24, 0.05),
        lidar_min_range_m=0.1,
        depth_cam=(0.30, -0.15, 87.0, 0.2, 2.0),
    )
    assert r.contacts == 0, (r.state, r.error, r.obstacle_states)
    assert r.state == "succeeded", (r.state, r.error, r.sim_s, r.obstacle_states)
    assert r.sim_s < 75.0, (r.sim_s, r.obstacle_states)

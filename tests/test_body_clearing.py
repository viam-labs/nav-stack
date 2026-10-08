"""Reactive obstacle measures must ignore returns inside the robot's own body.

Live 2026-10-07 (tracer2a, run 7): a mapped desk edge 1 cm inside the left
flank was counted by ``corridor_min_range`` as an obstacle 8 mm ahead. Forward
clearance read 0, the reactive layer went to ``avoid``, the turn-away was
vetoed by the neighbouring cells, and the robot could not move in any direction
for two goals. The footprint guard already clears in-body points; the reactive
layer did not.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.geom import conversions as conv
from src.nav.simple_motion import (
    ObstacleConfig,
    arc_clearance_m,
    corridor_min_range,
    forward_clearance_m,
    rear_clearance_m,
    spin_clearance_m,
)

HL, HW = 0.36, 0.295  # tracer2a body: 0.72 x 0.59 m


def _scan(points) -> conv.LaserScan2D:
    return conv.points_to_scan(
        np.asarray(points, dtype=float),
        angle_min=-math.pi,
        angle_max=math.pi,
        num_bins=720,
        range_min=0.0,
        range_max=10.0,
    )


def test_corridor_ignores_return_inside_body_but_keeps_one_ahead():
    scan = _scan([(0.008, 0.285), (0.8, 0.0)])
    # Old behaviour: the in-body point reads as 8 mm ahead.
    assert corridor_min_range(scan, 0.415, 1.0) == pytest.approx(0.008, abs=0.02)
    # With the body rectangle known it is dropped and the real obstacle wins.
    assert corridor_min_range(
        scan, 0.415, 1.0, body_half_length_m=HL, body_half_width_m=HW
    ) == pytest.approx(0.8, abs=0.03)


def test_shoulder_obstacle_just_outside_body_still_counts():
    # 2 cm outside the flank, *alongside* the body (x inside the body length):
    # a real shoulder graze. This is the case a wrong half-width would hide.
    scan = _scan([(0.20, 0.315)])
    d = corridor_min_range(scan, 0.415, 1.0, body_half_length_m=HL, body_half_width_m=HW)
    assert d == pytest.approx(0.20, abs=0.03)
    # With an inflated half-width (the 0.365 hard-clearance radius) it would
    # vanish — pin that the real dims are what callers must pass.
    assert corridor_min_range(
        scan, 0.415, 1.0, body_half_length_m=HL, body_half_width_m=0.365
    ) == math.inf


def test_forward_clearance_uses_body_dims_from_config():
    scan = _scan([(0.008, 0.285), (0.8, 0.0)])
    plain = ObstacleConfig(footprint_half_width_m=0.415, slow_distance_m=1.0)
    cleared = ObstacleConfig(
        footprint_half_width_m=0.415,
        slow_distance_m=1.0,
        body_half_length_m=HL,
        body_half_width_m=HW,
    )
    assert forward_clearance_m(scan, plain) == pytest.approx(0.008, abs=0.02)
    assert forward_clearance_m(scan, cleared) == pytest.approx(0.8, abs=0.03)


def test_spin_and_rear_clearance_ignore_body_returns():
    # Rear point slightly off exact 180° so it lands inside the scan's bin range.
    scan = _scan([(0.008, 0.285), (-0.1, 0.1), (-1.0, 0.05), (0.9, 0.0)])
    assert spin_clearance_m(scan) == pytest.approx(0.14, abs=0.03)  # in-body point wins
    assert spin_clearance_m(
        scan, body_half_length_m=HL, body_half_width_m=HW
    ) == pytest.approx(0.9, abs=0.03)
    assert rear_clearance_m(scan) == pytest.approx(0.14, abs=0.03)
    assert rear_clearance_m(
        scan, body_half_length_m=HL, body_half_width_m=HW
    ) == pytest.approx(1.0, abs=0.03)


def test_arc_clearance_ignores_body_returns():
    scan = _scan([(0.008, 0.285), (0.7, 0.2)])
    blind = arc_clearance_m(
        scan, curvature_1_m=0.3, half_width_m=0.415, max_forward_m=1.0
    )
    seeing = arc_clearance_m(
        scan,
        curvature_1_m=0.3,
        half_width_m=0.415,
        max_forward_m=1.0,
        body_half_length_m=HL,
        body_half_width_m=HW,
    )
    assert blind < 0.1
    assert seeing > 0.5


def test_no_body_dims_keeps_legacy_behaviour():
    scan = _scan([(0.008, 0.285)])
    assert corridor_min_range(scan, 0.415, 1.0) == pytest.approx(0.008, abs=0.02)
    assert spin_clearance_m(scan) == pytest.approx(0.285, abs=0.03)

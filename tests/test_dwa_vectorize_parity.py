"""Parity checks for vectorized DWA helpers vs scalar conventions."""

from __future__ import annotations

import math

import numpy as np

from src.geom import conversions as conv
from src.nav.planning.local_planner import _normalize_angles, _simulate, _simulate_batch
from src.nav.types import OccupancyGrid, Pose2D


def test_normalize_angles_matches_conv_at_pi_boundary():
    vals = np.array(
        [
            math.pi,
            -math.pi,
            3.0 * math.pi,
            -3.0 * math.pi,
            0.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            3.0 * math.pi / 2.0,
            -3.0 * math.pi / 2.0,
            math.pi + 1e-12,
            -math.pi - 1e-12,
        ],
        dtype=np.float64,
    )
    got = _normalize_angles(vals)
    for v, g in zip(vals.tolist(), got.tolist()):
        assert math.isclose(g, conv.normalize_angle(v), abs_tol=1e-12)


def test_batch_simulate_matches_scalar_at_pi_heading():
    pose = Pose2D(0.0, 0.0, math.pi)
    batch = _simulate_batch(
        pose,
        np.asarray([0.2], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        sim_time_s=0.3,
        sim_dt_s=0.1,
    )
    scalar = _simulate(pose, 0.2, 0.0, sim_time_s=0.3, sim_dt_s=0.1)
    for p, s in zip(batch[0], scalar):
        assert p[0] == s.x and p[1] == s.y
        assert p[2] == s.theta
        assert p[2] == conv.normalize_angle(p[2])


def test_world_to_cell_matches_occupancy_grid_truncation():
    occ = OccupancyGrid(
        grid=np.zeros((10, 10), dtype=np.int16),
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    xs = np.array([-0.01, -0.049, -0.05, -0.051, 0.0, 0.049, 0.05], dtype=np.float64)
    ys = xs.copy()
    # Same cast the DWA path uses.
    cols = ((xs - occ.origin_x) / occ.resolution).astype(np.int32)
    rows = ((ys - occ.origin_y) / occ.resolution).astype(np.int32)
    for x, y, r, c in zip(xs, ys, rows, cols):
        er, ec = occ.world_to_cell(float(x), float(y))
        assert int(r) == er and int(c) == ec

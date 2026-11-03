"""Golden checks for disk-offset fancy-index + scan cache + viz vectorize."""
from __future__ import annotations

import math

import numpy as np

from src.geom.conversions import LaserScan2D, Pose2D, costmap_frame_to_map
from src.nav.planning.costmap import (
    FREE,
    HARD_BUFFER,
    LETHAL,
    footprint_max_cost,
    nearest_free_cell,
)
from src.nav.control.footprint_guard import obstacle_points, scan_to_world_points
from src.nav.types import OccupancyGrid


def test_footprint_max_cost_center_lethal_and_oob():
    costs = np.zeros((20, 20), dtype=np.uint8)
    costs[10, 10] = LETHAL
    occ = OccupancyGrid(grid=costs, resolution=0.1, origin_x=0.0, origin_y=0.0)
    # Cell center (10,10) → world 1.05, 1.05
    assert footprint_max_cost(
        costs, occ, 1.05, 1.05, robot_radius_m=0.05
    ) == LETHAL
    # Near map edge: disk reaches outside → LETHAL
    assert footprint_max_cost(
        costs, occ, 0.05, 0.05, robot_radius_m=0.2
    ) == LETHAL
    # Clear interior far from lethal
    assert footprint_max_cost(
        costs, occ, 0.55, 0.55, robot_radius_m=0.05
    ) == FREE


def test_footprint_max_cost_soft_ring():
    costs = np.zeros((20, 20), dtype=np.uint8)
    costs[10, 10] = HARD_BUFFER - 1  # soft, still traversable as a cell
    occ = OccupancyGrid(grid=costs, resolution=0.1, origin_x=0.0, origin_y=0.0)
    assert footprint_max_cost(
        costs, occ, 1.05, 1.05, robot_radius_m=0.05
    ) == HARD_BUFFER - 1


def test_nearest_free_cell_snaps_off_lethal():
    costs = np.full((11, 11), FREE, dtype=np.uint8)
    costs[5, 5] = LETHAL
    cell = nearest_free_cell(costs, 5, 5, max_radius_cells=3)
    assert cell is not None
    assert cell != (5, 5)
    assert costs[cell] < HARD_BUFFER


def test_obstacle_points_scan_world_pts_cache_equivalence():
    pose = Pose2D(1.0, 2.0, math.radians(30.0))
    ranges = np.full(8, 1.5, dtype=np.float64)
    ranges[3] = float("inf")
    scan = LaserScan2D(
        ranges=ranges,
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 7,
        range_min=0.05,
        range_max=8.0,
        capture_pose=pose,
    )
    cached = scan_to_world_points(pose, scan)
    fresh = obstacle_points(pose, scan, None, radius_m=5.0)
    reused = obstacle_points(
        pose, scan, None, radius_m=5.0, scan_world_pts=cached
    )
    assert fresh.shape == reused.shape
    np.testing.assert_allclose(fresh, reused, atol=1e-9)
    # Passing empty cache must not fall back to re-transforming the scan.
    empty = obstacle_points(
        pose,
        scan,
        None,
        radius_m=5.0,
        scan_world_pts=np.empty((0, 2)),
    )
    assert empty.shape == (0, 2)


def test_costmap_frame_to_map_identity_and_yaw():
    grid = np.full((4, 5), -1, dtype=np.int16)
    grid[1, 2] = 40
    grid[2, 3] = 80
    cm = {
        "grid": grid,
        "resolution": 0.1,
        "origin_x": 1.0,
        "origin_y": 2.0,
    }
    identity = costmap_frame_to_map(cm, Pose2D(0.0, 0.0, 0.0))
    assert identity["resolution"] == 0.1
    assert np.max(identity["grid"]) == 80
    assert np.count_nonzero(identity["grid"] >= 0) == 2

    yawed = costmap_frame_to_map(cm, Pose2D(0.5, -0.25, math.radians(15.0)))
    assert yawed["resolution"] == 0.1
    assert np.max(yawed["grid"]) == 80
    assert np.count_nonzero(yawed["grid"] >= 0) >= 1

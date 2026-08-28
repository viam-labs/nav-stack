"""Tests for path smoother, local costmap, and local planner."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.nav_builtin.controller import FollowerConfig, compute_path_command
from src.nav_builtin.costmap import build_costmap, occupancy_from_bridge_map
from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig, footprint_collides
from src.nav_builtin.local_planner import LocalPlannerConfig, compute_local_command
from src.nav_builtin.smoother import smooth_path
from src.nav_builtin.types import OccupancyGrid, Path2D, Pose2D
from src.ros import conversions as conv


def _empty_map(size: int = 40, resolution: float = 0.05) -> dict:
    return {
        "grid": np.zeros((size, size), dtype=np.int16),
        "resolution": resolution,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }


def test_smooth_path_shortens_zigzag_astar():
    m = _empty_map(size=60, resolution=0.05)
    occ = occupancy_from_bridge_map(m)
    costs = build_costmap(
        occ, inflation_radius_m=0.15, robot_radius_m=0.05, cost_scaling_factor=3.0
    )
    # Deliberately jagged polyline.
    jagged = Path2D(
        points=(
            (0.25, 0.25),
            (0.25, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (2.5, 2.0),
        ),
        goal_theta=0.0,
    )

    def _len(path: Path2D) -> float:
        pts = path.points
        return sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )

    smooth = smooth_path(jagged, costs, occ, sample_spacing_m=0.10)
    assert len(smooth.points) >= 2
    assert _len(smooth) <= _len(jagged) + 1e-6
    assert len(smooth.points) > len(jagged.points) // 2


def test_local_costmap_marks_scan_hit():
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=2.0,
            height_m=2.0,
            resolution=0.05,
            inflation_radius_m=0.10,
            robot_radius_m=0.05,
            use_global_static=False,
        )
    )
    pose = Pose2D(1.0, 1.0, 0.0)
    scan = conv.LaserScan2D(
        angle_min=-math.pi,
        angle_increment=math.pi / 4.0,
        range_min=0.05,
        range_max=10.0,
        ranges=np.array(
            [1.0, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        ),
    )
    view = lc.update(pose, scan)
    # Forward hit at x=2.0 should be marked lethal/inscribed after inflation.
    assert view.cost_at_world(2.0, 1.0) > 0


def test_local_planner_avoids_marked_obstacle():
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=2.0,
            height_m=2.0,
            resolution=0.05,
            inflation_radius_m=0.15,
            robot_radius_m=0.08,
            use_global_static=False,
        )
    )
    pose = Pose2D(0.5, 1.0, 0.0)
    scan = conv.LaserScan2D(
        angle_min=-0.2,
        angle_increment=0.1,
        range_min=0.05,
        range_max=10.0,
        ranges=np.array([0.45, 0.45, 0.45, 0.45, 0.45]),
    )
    view = lc.update(pose, scan)
    path = Path2D(points=((0.5, 1.0), (2.0, 1.0)), goal_theta=0.0)
    cfg = LocalPlannerConfig(enabled=True, activate_cost_threshold=1, sim_time_s=2.0)
    cmd = compute_local_command(
        pose,
        path,
        view,
        cfg=cfg,
        max_vel_x=0.4,
        max_vel_theta=1.0,
        robot_radius_m=0.08,
    )
    assert cmd is not None
    # Straight full-speed forward should hit the wall within the rollout horizon.
    assert cmd.vx < 0.35 or abs(cmd.vtheta) > 0.1


def test_compute_path_command_defers_local_planner_when_misaligned():
    """DWA forward creep with zero turn while |bearing| > 75° causes shimmy loops."""
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=2.0,
            height_m=2.0,
            resolution=0.05,
            inflation_radius_m=0.15,
            robot_radius_m=0.08,
            use_global_static=False,
        )
    )
    # Facing ~129° while the path runs east — same failure mode as large bearing error.
    pose = Pose2D(0.5, 1.0, 2.26)
    scan = conv.LaserScan2D(
        angle_min=-0.1,
        angle_increment=0.1,
        range_min=0.05,
        range_max=10.0,
        ranges=np.array([0.8]),
    )
    view = lc.update(pose, scan)
    path = Path2D(points=((0.5, 1.0), (2.0, 1.0)), goal_theta=0.0)
    cmd, progress = compute_path_command(
        pose,
        path,
        cfg=FollowerConfig(),
        local_view=view,
        local_planner=LocalPlannerConfig(enabled=True, activate_cost_threshold=1),
        robot_radius_m=0.08,
    )
    assert progress.get("local_planner") is False
    assert abs(progress["bearing_error_rad"]) > math.radians(75.0)
    assert cmd.vx == 0.0
    assert abs(cmd.vtheta) > 0.05


def test_compute_path_command_uses_local_planner_when_blocked():
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=2.0,
            height_m=2.0,
            resolution=0.05,
            inflation_radius_m=0.15,
            robot_radius_m=0.08,
            use_global_static=False,
        )
    )
    pose = Pose2D(0.5, 1.0, 0.0)
    scan = conv.LaserScan2D(
        angle_min=-0.1,
        angle_increment=0.1,
        range_min=0.05,
        range_max=10.0,
        ranges=np.array([0.8]),
    )
    view = lc.update(pose, scan)
    path = Path2D(points=((0.5, 1.0), (2.0, 1.0)), goal_theta=0.0)
    cmd, progress = compute_path_command(
        pose,
        path,
        cfg=FollowerConfig(),
        local_view=view,
        local_planner=LocalPlannerConfig(enabled=True, activate_cost_threshold=1),
        robot_radius_m=0.08,
    )
    assert progress.get("local_planner") is True
    assert cmd.vx >= 0.0


def test_footprint_collides_outside_map():
    occ = OccupancyGrid(
        grid=np.zeros((10, 10), dtype=np.int16),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
    )
    costs = build_costmap(
        occ, inflation_radius_m=0.0, robot_radius_m=0.05, cost_scaling_factor=3.0
    )
    from src.nav_builtin.local_costmap import LocalCostmapView

    view = LocalCostmapView(costs=costs, occ=occ, origin_x=0.0, origin_y=0.0)
    assert footprint_collides(view, -1.0, 0.5, robot_radius_m=0.05) is True

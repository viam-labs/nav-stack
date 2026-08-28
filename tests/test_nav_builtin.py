"""Unit tests for ROS-free builtin costmap + A* planner."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.nav_builtin.controller import FollowerConfig, compute_path_command, lookahead_pose
from src.nav_builtin.costmap import (
    LETHAL,
    build_costmap,
    footprint_traversable,
    is_traversable,
    nearest_free_pose,
)
from src.nav_builtin.navigator import BuiltinNavigator
from src.nav_builtin.planner import (
    connect_plan_start,
    paths_meaningfully_differ,
    plan_on_costmap,
    plan_path,
)
from src.nav_builtin.types import OccupancyGrid, Path2D, Pose2D
from src.ros import conversions as conv


def _empty_map(size: int = 40, resolution: float = 0.05) -> dict:
    grid = np.zeros((size, size), dtype=np.int16)
    return {
        "grid": grid,
        "resolution": resolution,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }


def _wall_map() -> dict:
    """Free space with a vertical wall that forces a detour."""
    grid = np.zeros((40, 40), dtype=np.int16)
    grid[5:35, 20] = 100  # wall down the middle, with gaps at top/bottom
    grid[0:5, 20] = 0
    grid[35:40, 20] = 0
    return {
        "grid": grid,
        "resolution": 0.1,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }


def test_nearest_free_pose_requires_clear_footprint():
    grid = np.zeros((40, 40), dtype=np.int16)
    grid[20, 20] = 100
    occ = OccupancyGrid(grid=grid, resolution=0.1, origin_x=0.0, origin_y=0.0)
    costs = build_costmap(
        occ, inflation_radius_m=0.25, robot_radius_m=0.22, cost_scaling_factor=4.0
    )
    # Center cell is lethal; center point overlaps inscribed halo at r=0.22 m.
    assert not footprint_traversable(
        costs, occ, 2.05, 2.05, robot_radius_m=0.22
    )
    free = nearest_free_pose(
        costs, occ, 2.05, 2.05, robot_radius_m=0.22, max_radius_cells=12
    )
    assert free is not None
    assert footprint_traversable(
        costs, occ, free[0], free[1], robot_radius_m=0.22
    )


def test_connect_plan_start_prepends_escape_from_blocked_pose():
    m = _empty_map(size=60, resolution=0.05)
    grid = m["grid"]
    grid[30, 30] = 100
    start = Pose2D(1.52, 1.52, 0.0)  # inside pillar inflation
    goal = Pose2D(2.5, 1.52, 0.0)
    main = plan_path(
        m, start, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert main.feasible
    connected = connect_plan_start(
        m,
        start,
        main,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
    )
    assert connected.feasible
    assert len(connected.path.points) >= len(main.path.points)
    occ = OccupancyGrid(
        grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0
    )
    costs = build_costmap(
        occ, inflation_radius_m=0.25, robot_radius_m=0.22, cost_scaling_factor=4.0
    )
    sx, sy = connected.path.points[0]
    assert footprint_traversable(
        costs, occ, sx, sy, robot_radius_m=0.22
    )


def test_plan_path_marks_scan_for_dynamic_replan():
    m = _empty_map(size=60, resolution=0.05)
    start = Pose2D(0.5, 1.5, 0.0)
    goal = Pose2D(2.5, 1.5, 0.0)
    baseline = plan_path(
        m, start, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert baseline.feasible
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 0.9
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    blocked = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        scan=scan,
        scan_pose=conv.Pose2D(0.5, 1.5, 0.0),
    )
    assert blocked.feasible
    assert paths_meaningfully_differ(baseline.path, blocked.path)


def test_mark_path_ahead_forces_detour_on_replan():
    m = _empty_map(size=80, resolution=0.05)
    start = Pose2D(0.5, 2.0, 0.0)
    goal = Pose2D(3.5, 2.0, 0.0)
    baseline = plan_path(
        m, start, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert baseline.feasible
    mid = Pose2D(2.0, 2.0, 0.0)
    blocked = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        blocked_path=baseline.path,
        blocked_path_pose=mid,
    )
    assert blocked.feasible
    assert paths_meaningfully_differ(baseline.path, blocked.path)


def test_build_costmap_inflates_obstacles():
    occ = OccupancyGrid(
        grid=np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 100, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int16,
        ),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
    )
    costs = build_costmap(
        occ, inflation_radius_m=0.25, robot_radius_m=0.1, cost_scaling_factor=3.0
    )
    assert costs[1, 2] == LETHAL
    # Neighbors should be non-free.
    assert costs[1, 1] > 0
    assert costs[1, 3] > 0


def test_costmap_viz_dict_shows_inflation_gradient():
    from src.nav_builtin.costmap import costmap_viz_dict, costs_to_occupancy_viz

    occ = OccupancyGrid(
        grid=np.zeros((21, 21), dtype=np.int16),
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    occ.grid[10, 10] = 100
    costs = build_costmap(
        occ, inflation_radius_m=0.35, robot_radius_m=0.12, cost_scaling_factor=3.0
    )
    viz = costs_to_occupancy_viz(costs)
    assert viz[10, 10] == 100
    # Halo around obstacle should be 1..99, not raw free zeros.
    assert (viz[8:13, 8:13] > 0).sum() > 5
    d = costmap_viz_dict(occ, costs)
    assert d["grid"].shape == (21, 21)
    assert d["resolution"] == 0.05


def test_plan_respects_inflation_radius():
    """Lazy Theta* must not shortcut through the soft inflation halo."""
    grid = np.zeros((40, 40), dtype=np.int16)
    grid[20, 20] = 100  # pillar at map center
    m = {"grid": grid, "resolution": 0.1, "origin_x": 0.0, "origin_y": 0.0}
    start = Pose2D(0.5, 2.0, 0.0)
    goal = Pose2D(3.5, 2.0, 0.0)
    occ = OccupancyGrid(
        grid=grid, resolution=0.1, origin_x=0.0, origin_y=0.0
    )
    costs = build_costmap(
        occ, inflation_radius_m=0.35, robot_radius_m=0.05, cost_scaling_factor=3.0
    )
    # With the fix, 0.35 m halo is inscribed — path cannot pass through x≈2, y≈2.
    result = plan_on_costmap(
        occ, costs, start, goal, algorithm="lazy_theta_star"
    )
    assert result.feasible
    for x, y in result.path.points:
        r, c = occ.world_to_cell(x, y)
        assert is_traversable(int(costs[r, c]))


def test_plan_straight_line_on_empty_map():
    m = _empty_map()
    start = Pose2D(0.25, 0.25, 0.0)
    goal = Pose2D(1.5, 1.5, 0.0)
    result = plan_path(
        m, start, goal, inflation_radius_m=0.2, robot_radius_m=0.1
    )
    assert result.feasible
    assert len(result.path.points) >= 2
    assert result.path.points[0][0] == pytest.approx(start.x, abs=0.15)
    assert result.path.points[-1][0] == pytest.approx(goal.x, abs=0.15)
    preview = result.to_preview_dict(goal=(goal.x, goal.y, goal.theta), start=start)
    assert preview["feasible"] is True
    assert preview["point_count"] >= 2
    assert preview["planner_id"] == "LazyThetaStar"


def test_lazy_theta_star_shorter_or_smoother_than_astar():
    m = _empty_map(size=60, resolution=0.05)
    start = Pose2D(0.25, 0.25, 0.0)
    goal = Pose2D(2.75, 2.25, 0.0)
    astar = plan_path(
        m, start, goal, inflation_radius_m=0.15, robot_radius_m=0.05, algorithm="astar"
    )
    theta = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.15,
        robot_radius_m=0.05,
        algorithm="lazy_theta_star",
    )
    assert astar.feasible and theta.feasible
    # Any-angle should use fewer waypoints than grid A* on open space.
    assert len(theta.path.points) <= len(astar.path.points)
    # And path length should be no worse than A* (within tiny float slack).
    def _len(path):
        pts = path.points
        return sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )

    assert _len(theta.path) <= _len(astar.path) + 1e-6


def test_lazy_theta_star_detours_around_wall():
    m = _wall_map()
    start = Pose2D(0.5, 2.0, 0.0)
    goal = Pose2D(3.5, 2.0, 0.0)
    result = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.15,
        robot_radius_m=0.05,
        algorithm="lazy_theta_star",
    )
    assert result.feasible
    ys = [p[1] for p in result.path.points]
    assert min(ys) < 1.0 or max(ys) > 3.0


def test_plan_detours_around_wall():
    m = _wall_map()
    start = Pose2D(0.5, 2.0, 0.0)  # left of wall
    goal = Pose2D(3.5, 2.0, 0.0)  # right of wall
    result = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.15,
        robot_radius_m=0.05,
        algorithm="astar",
    )
    assert result.feasible
    xs = [p[0] for p in result.path.points]
    # Path must go around (not straight through x=2.0 wall column).
    # At least one point should be near the gap (y near 0 or 3.5+).
    ys = [p[1] for p in result.path.points]
    assert min(ys) < 1.0 or max(ys) > 3.0
    assert max(xs) > 3.0


def test_builtin_planner_config_alias():
    from src.config import NavConfig

    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "builtin": {"planner": "LazyThetaStar"},
        }
    )
    assert cfg.builtin.planner == "lazy_theta_star"


def test_plan_fails_when_goal_in_lethal():
    # Fully occupied map: nowhere to snap the goal.
    grid = np.full((20, 20), 100, dtype=np.int16)
    m = {
        "grid": grid,
        "resolution": 0.1,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }
    start = Pose2D(0.5, 0.5, 0.0)
    goal = Pose2D(1.5, 1.5, 0.0)
    result = plan_path(
        m, start, goal, inflation_radius_m=0.2, robot_radius_m=0.1
    )
    assert result.feasible is False
    assert result.error_code != 0


def test_lookahead_advances_along_path():
    path = Path2D(points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), goal_theta=0.0)
    pose = Pose2D(0.0, 0.0, 0.0)
    target, idx, is_final = lookahead_pose(
        pose, path, lookahead_m=0.5, waypoint_tolerance_m=0.1
    )
    assert not is_final
    assert target.x == pytest.approx(0.5, abs=0.05)


def test_lookahead_projects_onto_sparse_segment():
    """Robot mid-segment should look ahead along the line, not jump to a vertex."""
    path = Path2D(points=((0.0, 0.0), (5.0, 0.0)), goal_theta=0.0)
    pose = Pose2D(2.0, 0.1, 0.0)
    target, idx, is_final = lookahead_pose(
        pose, path, lookahead_m=1.0, waypoint_tolerance_m=0.1
    )
    assert not is_final
    assert target.x == pytest.approx(3.0, abs=0.15)
    assert abs(target.y) < 0.2


def test_follow_command_translates_while_gently_turning():
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    current = Pose2D(0.0, 0.0, 0.0)
    target = Pose2D(2.0, 0.5, 0.0)  # ~14 deg bearing — should still drive
    cmd = compute_follow_command(current, target, cfg=cfg)
    assert not cmd.done
    assert cmd.vx > 0.05
    assert cmd.vtheta != 0.0


def test_follow_command_rotates_in_place_when_facing_away():
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    current = Pose2D(0.0, 0.0, 0.0)
    target = Pose2D(-2.0, 0.0, 0.0)  # 180 deg behind
    cmd = compute_follow_command(current, target, cfg=cfg)
    assert cmd.vx == 0.0
    assert abs(cmd.vtheta) > 0.0


def test_compute_path_command_drives_forward():
    path = Path2D(points=((0.0, 0.0), (2.0, 0.0)), goal_theta=0.0)
    current = Pose2D(0.0, 0.0, 0.0)
    cmd, progress = compute_path_command(current, path, cfg=FollowerConfig())
    assert not cmd.done
    assert cmd.vx > 0.0
    assert progress["distance_remaining_m"] > 0.0


class _FakeWorld:
    def __init__(self, pose: Pose2D, map_data: dict):
        self.pose = pose
        self.map_data = map_data
        self.cmds = []
        self.stopped = False

    def get_map(self):
        return self.map_data

    def get_pose(self):
        return self.pose

    def get_scan(self, max_age_s: float = 2.0):
        return None

    def set_velocity(self, vx, vy, vtheta):
        self.cmds.append((vx, vy, vtheta))
        # Nudge pose toward +x for a trivial follow.
        if vx > 0:
            self.pose = Pose2D(self.pose.x + 0.05, self.pose.y, self.pose.theta)

    def stop(self):
        self.stopped = True

    def set_viz_plan(self, path_xy, goal=None):
        pass

    def set_viz_costmap(self, costmap):
        self.costmap = costmap


def test_builtin_navigator_compute_path_and_status():
    world = _FakeWorld(Pose2D(0.2, 0.2, 0.0), _empty_map())
    nav = BuiltinNavigator(
        world,
        inflation_radius_m=0.15,
        robot_radius_m=0.05,
        avoid_obstacles=False,
        xy_tolerance_m=0.1,
        timeout_s=5.0,
    )
    preview = nav.compute_path(1.5, 0.2, 0.0)
    assert preview["feasible"] is True
    assert nav.last_preview_plan() is not None
    status = nav.nav_status()
    assert status["motion"] == "builtin"
    assert status["active"] is False


def test_builtin_navigator_cancel_sets_status():
    world = _FakeWorld(Pose2D(0.2, 0.2, 0.0), _empty_map())
    nav = BuiltinNavigator(world, avoid_obstacles=False)
    nav.cancel()
    assert nav.nav_status()["state"] == "canceled"

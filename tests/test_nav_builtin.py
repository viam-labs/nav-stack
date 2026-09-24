"""Unit tests for ROS-free builtin costmap + A* planner."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.nav_builtin.controller import FollowerConfig, compute_path_command, lookahead_pose
from src.nav_builtin.costmap import (
    FREE,
    HARD_BUFFER,
    INSCRIBED,
    LETHAL,
    build_costmap,
    footprint_traversable,
    is_hard,
    is_traversable,
    nearest_free_pose,
)
from src.nav_builtin.navigator import BuiltinNavigator
from src.nav_builtin.supervisor import NavSupervisor
from src.nav_builtin.planner import (
    connect_plan_start,
    path_blocked,
    paths_meaningfully_differ,
    plan_on_costmap,
    plan_path,
)
from src.nav_builtin.types import OccupancyGrid, Path2D, Pose2D
from src.geom import conversions as conv


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
    # Costs are already inflated by robot_radius: the start cell (plus one cell
    # of slack) being traversable means the robot can stand there.
    assert footprint_traversable(costs, occ, sx, sy, robot_radius_m=0.05)
    assert math.hypot(sx - 1.52, sy - 1.52) >= 0.22


def test_plan_start_next_to_live_obstacle_stays_feasible():
    """Start 0.5 m from a bin with a 0.45 m robot: must not be 'start in lethal'."""
    m = _empty_map(size=100, resolution=0.05)
    grid = m["grid"]
    grid[48:52, 60:64] = 100  # ~0.2 m bin around (3.1, 2.5)
    start = Pose2D(2.5, 2.5, 0.0)  # 0.5 m from the bin face
    goal = Pose2D(4.5, 2.5, 0.0)
    res = plan_path(m, start, goal, inflation_radius_m=0.35, robot_radius_m=0.45)
    assert res.feasible, res.error_msg
    sx, sy = res.path.points[0]
    assert math.hypot(sx - start.x, sy - start.y) < 0.15


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
        max_goal_snap_m=1.0,
    )
    assert blocked.feasible
    assert paths_meaningfully_differ(baseline.path, blocked.path)


def test_mark_path_ahead_keeps_robot_footprint_free():
    """Painting the corridor must start ahead of the robot, not under it.

    A 0.45 m robot in a 2.6 m-wide room, replanning from *on* its own route:
    painting from the robot's position made the start lethal and every retry
    infeasible (57 failed replans while spinning).
    """
    grid = np.zeros((80, 120), dtype=np.int16)
    grid[0:2, :] = 100  # walls: room is y in [0.1, 3.9] -> effectively 2.6 m
    grid[78:80, :] = 100
    grid[0:14, :] = 100
    grid[66:80, :] = 100
    m = {"grid": grid, "resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0}
    r = 0.45
    start = Pose2D(1.0, 2.0, 0.0)
    goal = Pose2D(5.0, 2.0, 0.0)
    baseline = plan_path(m, start, goal, inflation_radius_m=0.35, robot_radius_m=r)
    assert baseline.feasible
    from src.nav_builtin.costmap import mark_path_ahead_on_occupancy, occupancy_from_map_dict

    occ = occupancy_from_map_dict(m)
    painted = mark_path_ahead_on_occupancy(
        occ,
        baseline.path,
        start,
        radius_m=0.12,
        lookahead_m=1.5,
        start_offset_m=r + 0.12 + 0.1,
    )
    costs = build_costmap(painted, inflation_radius_m=0.35, robot_radius_m=r)
    # Robot cell itself stays traversable (no lethal within robot_radius).
    row, col = painted.world_to_cell(start.x, start.y)
    assert is_traversable(int(costs[row, col]))
    # And the corridor ahead really is painted.
    row2, col2 = painted.world_to_cell(start.x + 1.2, start.y)
    assert int(painted.grid[row2, col2]) >= 50
    replanned = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.35,
        robot_radius_m=r,
        blocked_path=baseline.path,
        blocked_path_pose=start,
    )
    assert replanned.feasible, replanned.error_msg
    assert paths_meaningfully_differ(baseline.path, replanned.path)


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


def test_costmap_soft_outer_matches_inflation_radius():
    """Soft halo ends at inflation_radius; preference past that is planner-only."""
    from src.nav_builtin.costmap import costs_to_occupancy_viz

    occ = OccupancyGrid(
        grid=np.zeros((81, 81), dtype=np.int16),
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    occ.grid[40, 40] = 100
    robot_r = 0.22
    inflate_r = 0.35
    costs = build_costmap(
        occ,
        inflation_radius_m=inflate_r,
        robot_radius_m=robot_r,
        cost_scaling_factor=4.0,
        clearance_preference_m=0.35,
    )
    cx, cy = 2.0, 2.0  # world center of obstacle cell
    # Just inside soft outer edge: non-zero soft cost (viz-visible).
    r_in, c_in = occ.world_to_cell(cx + inflate_r - 0.03, cy)
    assert int(costs[r_in, c_in]) >= 50
    # Just outside configured inflation: preference cost, not drawn as soft.
    r_out, c_out = occ.world_to_cell(cx + inflate_r + 0.08, cy)
    assert 0 < int(costs[r_out, c_out]) < 50
    viz = costs_to_occupancy_viz(costs)
    assert int(viz[r_out, c_out]) == 0
    # Inside footprint: inscribed.
    r_hard, c_hard = occ.world_to_cell(cx + robot_r * 0.5, cy)
    assert int(costs[r_hard, c_hard]) == INSCRIBED
    # Past preference band: free.
    r_far, c_far = occ.world_to_cell(cx + inflate_r + 0.45, cy)
    assert int(costs[r_far, c_far]) == FREE


def test_costmap_hard_buffer_ring_is_blocked_and_lighter_in_viz():
    """Body vs clearance_m are two viz rings; both stay non-traversable."""
    from src.nav_builtin.costmap import costs_to_occupancy_viz

    occ = OccupancyGrid(
        grid=np.zeros((81, 81), dtype=np.int16),
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    occ.grid[40, 40] = 100
    body_r = 0.25
    hard_r = 0.45  # 0.20 m clearance past the body
    costs = build_costmap(
        occ,
        inflation_radius_m=hard_r,
        robot_radius_m=hard_r,
        body_radius_m=body_r,
        cost_scaling_factor=4.0,
        clearance_preference_m=0.0,
    )
    cx, cy = 2.0, 2.0
    r_body, c_body = occ.world_to_cell(cx + body_r * 0.4, cy)
    r_buf, c_buf = occ.world_to_cell(cx + (body_r + hard_r) * 0.5, cy)
    assert int(costs[r_body, c_body]) == INSCRIBED
    assert int(costs[r_buf, c_buf]) == HARD_BUFFER
    assert not is_traversable(int(costs[r_body, c_body]))
    assert not is_traversable(int(costs[r_buf, c_buf]))
    assert is_hard(int(costs[r_buf, c_buf]))
    viz = costs_to_occupancy_viz(costs)
    assert int(viz[r_body, c_body]) == 99
    assert int(viz[r_buf, c_buf]) == 90


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
    # Soft halo out to inflation_radius; path may graze soft costs but must
    # stay traversable (outside inscribed/lethal).
    result = plan_on_costmap(
        occ, costs, start, goal, algorithm="lazy_theta_star"
    )
    assert result.feasible
    for x, y in result.path.points:
        r, c = occ.world_to_cell(x, y)
        assert is_traversable(int(costs[r, c]))


def test_planner_prefers_clear_lane_over_inflation_hug():
    """When a clear detour exists, do not hug the soft inflation of a wall."""
    import numpy as np

    grid = np.zeros((100, 140), dtype=np.int16)
    grid[0:28, 25:115] = 100  # solid block for y in [0, 1.4)
    occ = OccupancyGrid(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)
    costs = build_costmap(
        occ, inflation_radius_m=0.40, robot_radius_m=0.22, cost_scaling_factor=4.0
    )
    start = Pose2D(0.6, 1.55, 0.0)
    goal = Pose2D(6.4, 1.55, 0.0)
    result = plan_on_costmap(
        occ, costs, start, goal, algorithm="lazy_theta_star", robot_radius_m=0.22
    )
    assert result.feasible
    # Sample the polyline across the wall span — must climb into free space
    # (~y≥1.8) rather than ride the inscribed/soft edge at y≈1.55.
    ys: list[float] = []
    peak_cost = 0
    pts = result.path.points
    for i in range(len(pts) - 1):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[i + 1][0], pts[i + 1][1]
        for t in np.linspace(0.0, 1.0, 60):
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if 1.5 < x < 5.5:
                ys.append(y)
                r, c = occ.world_to_cell(x, y)
                peak_cost = max(peak_cost, int(costs[r, c]))
    assert ys
    assert min(ys) >= 1.65
    # Mid-route should stay out of preference / soft (LOS soft cap 30).
    assert peak_cost <= 30


def _gap_map(gap_m: float, resolution: float = 0.05) -> OccupancyGrid:
    """Room split by a wall with a ``gap_m`` doorway, open at both far ends."""
    h = int(8.0 / resolution)
    w = int(12.0 / resolution)
    grid = np.zeros((h, w), dtype=np.int16)
    wall = int(4.0 / resolution)
    grid[wall : wall + 3, :] = 100
    centre = int(6.0 / resolution)
    half = int((gap_m / 2.0) / resolution)
    grid[wall : wall + 3, centre - half : centre + half] = 0
    grid[wall : wall + 3, :6] = 0  # detour around the left end
    grid[wall : wall + 3, w - 6 :] = 0  # and the right end
    return OccupancyGrid(
        grid=grid, resolution=resolution, origin_x=0.0, origin_y=0.0
    )


def test_half_diagonal_radius_seals_gap_the_robot_fits_through():
    """A 0.59x0.72 m robot fits an 0.84 m doorway; one disc of its half-diagonal
    (0.465) does not, and sealed it with no feasible route."""
    occ = _gap_map(0.84)
    start = Pose2D(6.0, 3.0, 0.0)
    goal = Pose2D(6.0, 5.0, 0.0)

    circumscribed = math.hypot(0.72 / 2.0, 0.59 / 2.0)
    assert circumscribed == pytest.approx(0.465, abs=0.005)
    sealed_costs = build_costmap(
        occ,
        inflation_radius_m=0.25,
        robot_radius_m=circumscribed,
        cost_scaling_factor=4.0,
    )
    row, col = occ.world_to_cell(6.0, 4.075)
    assert int(sealed_costs[row, col]) >= 253  # inscribed: no cell to plan through
    sealed = plan_on_costmap(
        occ, sealed_costs, start, goal, robot_radius_m=circumscribed
    )
    assert not sealed.feasible

    # Half-width (what actually has to fit) leaves the doorway open.
    inscribed = 0.59 / 2.0
    costs = build_costmap(
        occ,
        inflation_radius_m=0.25,
        robot_radius_m=inscribed,
        cost_scaling_factor=4.0,
    )
    assert int(costs[row, col]) < 253
    through = plan_on_costmap(occ, costs, start, goal, robot_radius_m=inscribed)
    assert through.feasible, through.error_msg
    # Straight through the doorway, not around either end of the wall.
    assert all(abs(p[0] - 6.0) < 1.5 for p in through.path.points)


def test_supervisor_wires_footprint_derived_clearances():
    """Footprint dims must reach the follower: bumper stop, body-width corridor,
    drive radius on the half-width, spin radius on the half-diagonal."""
    from src.config import NavConfig
    from src.nav_builtin.supervisor import NavSupervisor

    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "robot_radius": 0.45,
            "footprint_width_m": 0.59,
            "footprint_length_m": 0.72,
            "simple_stop_distance": 0.4,
        }
    )
    sup = NavSupervisor(
        _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map()),
        robot_radius_m=cfg.inscribed_radius_m(),
        spin_radius_m=cfg.circumscribed_radius_m(),
        nose_offset_m=cfg.nose_offset_m(),
        wheel_half_track_m=cfg.wheel_half_track_m(),
        stop_distance_m=cfg.simple_stop_distance,
    )
    assert sup._robot_radius == pytest.approx(0.295)
    assert sup._spin_radius == pytest.approx(0.4654, abs=0.001)
    follower = sup._follower
    # Bumper is 0.36 m out, so stop at 0.41 — not a 0.5 m half-diagonal disc.
    assert follower.obstacle.stop_distance_m == pytest.approx(0.41)
    assert follower.obstacle.footprint_half_width_m == pytest.approx(0.415)
    assert follower.wheel_half_track_m == pytest.approx(0.2655)


def test_supervisor_uses_hard_clearance_for_plan_and_local_costmap():
    """clearance_m is the same inscribed radius on both costmaps."""
    from src.config import NavConfig
    from src.nav_builtin.supervisor import NavSupervisor

    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "footprint_width_m": 0.59,
            "footprint_length_m": 0.72,
        }
    )
    sup = NavSupervisor(
        _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map()),
        cfg,
    )
    assert cfg.hard_clearance_radius_m() == pytest.approx(0.495)
    assert sup._robot_radius == pytest.approx(0.495)  # noqa: SLF001
    local = sup._local_costmap  # noqa: SLF001
    assert local is not None
    assert local._cfg.robot_radius_m == pytest.approx(0.495)  # noqa: SLF001
    assert local._cfg.inflation_radius_m == pytest.approx(0.495)  # noqa: SLF001


def test_corner_path_stays_out_of_soft_halo():
    """Repro: round a pillar tip in clear space, not through the soft glow."""
    import numpy as np

    from src.nav_builtin.smoother import smooth_path

    grid = np.zeros((80, 80), dtype=np.int16)
    grid[20:55, 35:50] = 100  # vertical bar
    occ = OccupancyGrid(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)
    costs = build_costmap(
        occ, inflation_radius_m=0.35, robot_radius_m=0.22, cost_scaling_factor=4.0
    )
    start = Pose2D(1.0, 2.5, 0.0)
    goal = Pose2D(3.2, 1.0, 0.0)
    result = plan_on_costmap(
        occ, costs, start, goal, algorithm="lazy_theta_star", robot_radius_m=0.22
    )
    assert result.feasible
    path = smooth_path(result.path, costs, occ, enabled=True, sample_spacing_m=0.10)
    peak = 0
    pts = path.points
    for i in range(len(pts) - 1):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[i + 1][0], pts[i + 1][1]
        for t in np.linspace(0.0, 1.0, 40):
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            r, c = occ.world_to_cell(x, y)
            if occ.in_bounds(r, c):
                peak = max(peak, int(costs[r, c]))
    # Open space around the tip: stay in free / near-free, not soft glow.
    assert peak <= 5


def test_t_pillar_tip_prefers_clear_swing():
    """Screenshot-like stem tip: only under-tip route; stay outside soft glow."""
    grid = np.zeros((120, 100), dtype=np.int16)
    # Stem from y=2.0 up to map top so the only detour is under the tip.
    grid[40:120, 48:55] = 100
    occ = OccupancyGrid(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)
    costs = build_costmap(
        occ, inflation_radius_m=0.35, robot_radius_m=0.22, cost_scaling_factor=4.0
    )
    start = Pose2D(3.6, 2.4, 0.0)
    goal = Pose2D(1.4, 2.4, 0.0)
    result = plan_on_costmap(
        occ, costs, start, goal, algorithm="lazy_theta_star", robot_radius_m=0.22
    )
    assert result.feasible
    ys: list[float] = []
    peak = 0
    pts = result.path.points
    for i in range(len(pts) - 1):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[i + 1][0], pts[i + 1][1]
        for t in np.linspace(0.0, 1.0, 50):
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if 2.2 < x < 2.9:
                ys.append(y)
                r, c = occ.world_to_cell(x, y)
                if occ.in_bounds(r, c):
                    peak = max(peak, int(costs[r, c]))
    assert ys
    # Tip at y=2.0; soft outer ≈ 1.65; preference outer ≈ 1.30. Prefer clear
    # swing outside the visible glow (and ideally past preference).
    assert max(ys) <= 1.40
    assert peak <= 5


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

    def _len(path):
        pts = path.points
        return sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )

    # World densify may add waypoints for collision safety; length still matters.
    assert _len(theta.path) <= _len(astar.path) + 1e-6
    assert (
        path_blocked(
            m, theta.path, inflation_radius_m=0.15, robot_radius_m=0.05, from_pose=start
        )
        is False
    )

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


def test_plan_rejects_large_goal_snap():
    """Live inflation must not silently move the goal a metre away."""
    from src.nav_builtin.planner import plan_on_costmap
    from src.nav_builtin.costmap import build_costmap, occupancy_from_map_dict

    m = _empty_map(size=60, resolution=0.1)
    # Block a wide region around the requested goal so nearest free is far.
    m["grid"][20:40, 20:45] = 100
    occ = occupancy_from_map_dict(m)
    costs = build_costmap(occ, inflation_radius_m=0.2, robot_radius_m=0.15)
    start = Pose2D(0.5, 0.5, 0.0)
    goal = Pose2D(3.0, 3.0, 0.0)  # inside the blocked blob
    bad = plan_on_costmap(
        occ, costs, start, goal, robot_radius_m=0.15, max_goal_snap_m=0.5
    )
    assert bad.feasible is False
    assert "goal snap" in (bad.error_msg or "")


def test_path_blocked_on_costmap_matches_path_blocked():
    """Cached-costmap check must agree with a fresh inflate (control-loop path)."""
    from src.nav_builtin.costmap import build_costmap, occupancy_from_map_dict
    from src.nav_builtin.planner import path_blocked_on_costmap

    m = _empty_map(size=40, resolution=0.1)
    m["grid"][20, 15] = 100
    path = Path2D(points=((0.5, 2.0), (3.5, 2.0)), goal_theta=0.0)
    pose = Pose2D(0.5, 2.0, 0.0)
    fresh = path_blocked(
        m,
        path,
        inflation_radius_m=0.2,
        robot_radius_m=0.15,
        from_pose=pose,
        ahead_m=3.0,
    )
    occ = occupancy_from_map_dict(m)
    costs = build_costmap(occ, inflation_radius_m=0.2, robot_radius_m=0.15)
    cached = path_blocked_on_costmap(
        occ, costs, path, robot_radius_m=0.15, from_pose=pose, ahead_m=3.0
    )
    assert fresh is True
    assert cached is True


def test_path_blocked_horizon_ignores_far_obstacle():
    """Long routes must not fail static checks on far-ahead map changes."""
    m = _empty_map(size=80, resolution=0.1)
    # Lethal pillar ~1.5 m along +x — outside a short horizon from origin.
    m["grid"][10, 15] = 100
    path = Path2D(
        points=tuple((0.1 * i, 1.0) for i in range(40)),
        goal_theta=0.0,
    )
    assert path_blocked(
        m,
        path,
        inflation_radius_m=0.2,
        robot_radius_m=0.1,
        from_pose=Pose2D(0.0, 1.0, 0.0),
        ahead_m=0.8,
    ) is False
    assert path_blocked(
        m,
        path,
        inflation_radius_m=0.2,
        robot_radius_m=0.1,
        from_pose=Pose2D(0.0, 1.0, 0.0),
        ahead_m=2.0,
    ) is True


def test_lazy_theta_plan_survives_path_blocked_on_corridor():
    """Planned world polyline must not immediately fail static path_blocked."""
    from src.sim.world import make_builtin_corridor

    sm = make_builtin_corridor()
    m = {
        "grid": sm.grid.copy(),
        "resolution": sm.resolution,
        "origin_x": sm.origin_x,
        "origin_y": sm.origin_y,
    }
    start = Pose2D(5.69, 1.63, -3.1)
    goal = Pose2D(1.77, 2.34, -0.13)
    result = plan_path(
        m, start, goal, inflation_radius_m=0.35, robot_radius_m=0.22
    )
    assert result.feasible
    assert (
        path_blocked(
            m,
            result.path,
            inflation_radius_m=0.35,
            robot_radius_m=0.22,
            from_pose=start,
            ahead_m=4.0,
        )
        is False
    )


def test_plan_with_scan_does_not_seal_mapped_corridor():
    """Lidar hits on already-mapped walls must not double-inflate the gap shut."""
    from src.sim.world import SimWorld, make_builtin_corridor

    sm = make_builtin_corridor()
    m = {
        "grid": sm.grid.copy(),
        "resolution": sm.resolution,
        "origin_x": sm.origin_x,
        "origin_y": sm.origin_y,
    }
    start = Pose2D(5.69, 1.63, -3.1)
    goal = Pose2D(1.77, 2.34, -0.13)
    world = SimWorld(sm, seed_pose=conv.Pose2D(start.x, start.y, start.theta))
    scan = world.get_scan()
    result = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.35,
        robot_radius_m=0.22,
        scan=scan,
        scan_pose=start,
        dynamic_obstacle_radius_m=0.45,
    )
    assert result.feasible


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
    path_yaw = math.atan2(0.5, 2.0)
    target = Pose2D(2.0, 0.5, path_yaw)
    cmd = compute_follow_command(current, target, cfg=cfg)
    assert not cmd.done
    assert cmd.vx > 0.05
    assert cmd.vtheta > 0.0
    # Pure pursuit: ω = v·κ with κ = 2y/L² → gentle for a far, slightly-off point.
    assert abs(cmd.vtheta) <= 0.3


def test_follow_command_cruises_when_aligned():
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.max_linear_mps = 0.6
    path_yaw = math.atan2(-2.96, -0.38)
    current = Pose2D(0.0, 0.0, path_yaw)
    target = Pose2D(-0.38, -2.96, path_yaw)
    cmd = compute_follow_command(current, target, cfg=cfg, final_yaw=None)
    assert not cmd.done
    assert cmd.vx >= 0.33
    assert abs(cmd.vtheta) < 0.15


def test_pursuit_is_geometric_omega_scales_with_speed():
    """ω = v·κ: halving max speed halves ω for the same lookahead point."""
    from src.nav_builtin.controller import pursuit_command

    current = Pose2D(0.0, 0.0, 0.0)
    target = Pose2D(0.6, 0.1, 0.0)  # gentle: κ = 2·0.1/0.37 ≈ 0.54 (r ≈ 1.85 m)
    fast = FollowerConfig()
    fast.motion.max_linear_mps = 0.6
    slow = FollowerConfig()
    slow.motion.max_linear_mps = 0.3
    cmd_f, rot_f = pursuit_command(current, target, cfg=fast)
    cmd_s, rot_s = pursuit_command(current, target, cfg=slow)
    assert not rot_f and not rot_s
    assert cmd_f.vx == pytest.approx(0.6)
    assert cmd_s.vx == pytest.approx(0.3)
    assert cmd_f.vtheta == pytest.approx(2 * cmd_s.vtheta, rel=1e-6)
    assert cmd_f.vtheta / cmd_f.vx == pytest.approx(cmd_s.vtheta / cmd_s.vx, rel=1e-6)


def test_pursuit_regulates_speed_by_curvature():
    """Tight lookahead arc → slow down (r/r_min), never below the regulated floor."""
    from src.nav_builtin.controller import pursuit_command

    cfg = FollowerConfig()
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(0.0, 0.0, 0.0)
    # 40° bearing at L=0.5 → κ = 2 sin(40°)/0.5 ≈ 2.57 → r ≈ 0.39 m < 0.7.
    target = Pose2D(0.5 * math.cos(math.radians(40)), 0.5 * math.sin(math.radians(40)), 0.0)
    cmd, rotating = pursuit_command(current, target, cfg=cfg)
    assert not rotating
    assert cfg.regulated_min_speed_mps <= cmd.vx < 0.6 * 0.6
    assert cmd.vtheta > 0.0
    # Still a drivable arc for the Viam base sanitizer (not a spin).
    assert cmd.vx >= 0.12


def test_pursuit_corrects_crosstrack_toward_path():
    """Robot left of a straight path turns right back onto it, without Stanley."""
    from src.nav_builtin.controller import pursuit_command

    cfg = FollowerConfig()
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(0.0, 0.35, 0.0)
    target = Pose2D(0.6, 0.0, 0.0)
    cmd, rotating = pursuit_command(current, target, cfg=cfg)
    assert not rotating
    assert cmd.vtheta < 0.0
    assert cmd.vx > 0.12


def test_pursuit_crosstrack_deadband_zeros_kappa():
    """Sub-deadband lateral error must not command a turn (pose-noise floor)."""
    from src.nav_builtin.controller import pursuit_command

    cfg = FollowerConfig()
    cfg.crosstrack_deadband_m = 0.06
    cfg.curvature_smoothing = 1.0  # no EMA; raw κ only
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(0.0, 0.0, 0.0)
    # |y_l| = 0.04 < deadband → κ = 0 → straight cruise.
    cmd, rotating = pursuit_command(current, Pose2D(1.0, 0.04, 0.0), cfg=cfg)
    assert not rotating
    assert cmd.vx > 0.12
    assert abs(cmd.vtheta) < 1e-9
    # Just outside the deadband: still corrects.
    cmd2, _ = pursuit_command(current, Pose2D(1.0, 0.07, 0.0), cfg=cfg)
    assert cmd2.vtheta > 0.0


def test_pursuit_rotate_to_heading_has_hysteresis():
    from src.nav_builtin.controller import pursuit_command

    cfg = FollowerConfig()
    current = Pose2D(0.0, 0.0, 0.0)

    def _at(deg: float) -> Pose2D:
        return Pose2D(0.6 * math.cos(math.radians(deg)), 0.6 * math.sin(math.radians(deg)), 0.0)

    # Beyond the enter threshold: rotate-to-heading.
    far = _at(math.degrees(cfg.rotate_in_place_rad) + 15.0)
    cmd, rotating = pursuit_command(current, far, cfg=cfg)
    assert rotating and cmd.vx == 0.0 and cmd.vtheta > 0.0
    assert abs(cmd.vtheta) <= cfg.rotate_vel_rad_s + 1e-9
    # Between exit and enter: keep rotating only if we already were.
    mid = _at(0.5 * math.degrees(cfg.rotate_in_place_rad + cfg.rotate_exit_rad))
    cmd_stay, still = pursuit_command(current, mid, cfg=cfg, rotate_active=True)
    assert still and cmd_stay.vx == 0.0
    cmd_go, fresh = pursuit_command(current, mid, cfg=cfg, rotate_active=False)
    assert not fresh and cmd_go.vx > 0.0
    # Under the exit threshold: leave rotate-to-heading.
    close = _at(math.degrees(cfg.rotate_exit_rad) - 10.0)
    cmd_exit, done_rot = pursuit_command(current, close, cfg=cfg, rotate_active=True)
    assert not done_rot and cmd_exit.vx > 0.0


def test_keep_arc_drivable_preserves_curvature_at_crawl():
    from src.nav_builtin.controller import DriveCommand, keep_arc_drivable

    cfg = FollowerConfig()  # half_track 0.27, wheel_min 0.06, r_min 0.42
    # Slow-down produced 0.05 m/s with 0.10 rad/s (r = 0.5 m). Same arc, but
    # fast enough that the inner wheel (vx - |ω|·half_track) stays ≥ 0.06.
    out = keep_arc_drivable(DriveCommand(0.05, 0.0, 0.10, False), cfg)
    assert out.vtheta / out.vx == pytest.approx(2.0)
    assert out.vx >= 0.125
    assert out.vx - abs(out.vtheta) * cfg.wheel_half_track_m >= cfg.wheel_min_speed_mps - 1e-9
    # Tighter than the min turn radius (r = 0.27 m): widen to r_min, not spin.
    tight = keep_arc_drivable(DriveCommand(0.08, 0.0, 0.30, False), cfg)
    assert tight.vx > 0.0
    assert tight.vx / abs(tight.vtheta) == pytest.approx(cfg.effective_min_turn_radius_m())
    assert tight.vx - abs(tight.vtheta) * cfg.wheel_half_track_m >= cfg.wheel_min_speed_mps - 1e-9
    # Already drivable / not translating: untouched.
    ok = DriveCommand(0.3, 0.0, 0.3, False)
    assert keep_arc_drivable(ok, cfg) == ok
    spin = DriveCommand(0.0, 0.0, 0.5, False)
    assert keep_arc_drivable(spin, cfg) == spin


def test_pursuit_respects_skid_steer_wheel_envelope():
    """No translating command may put the inner wheel under the base minimum."""
    from src.nav_builtin.controller import pursuit_command

    cfg = FollowerConfig()
    cfg.motion.max_linear_mps = 0.4
    current = Pose2D(0.0, 0.0, 0.0)
    for deg in range(-58, 59, 4):
        for L in (0.6, 0.9, 1.2, 1.5):
            tgt = Pose2D(L * math.cos(math.radians(deg)), L * math.sin(math.radians(deg)), 0.0)
            cmd, rotating = pursuit_command(current, tgt, cfg=cfg)
            assert not rotating
            inner = cmd.vx - abs(cmd.vtheta) * cfg.wheel_half_track_m
            assert inner >= cfg.wheel_min_speed_mps - 1e-9, (deg, L, cmd)
            assert cmd.vx >= 0.125
            if abs(cmd.vtheta) > 1e-9:
                assert cmd.vx / abs(cmd.vtheta) >= cfg.effective_min_turn_radius_m() - 1e-9


def test_follow_command_approach_cap_only_at_goal():
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.max_linear_mps = 0.6
    cfg.approach_dist_m = 0.35
    current = Pose2D(0.0, 0.0, 0.0)
    near = Pose2D(0.2, 0.0, 0.0)
    mid = compute_follow_command(current, near, cfg=cfg, final_yaw=None)
    goal = compute_follow_command(current, near, cfg=cfg, final_yaw=0.0)
    assert mid.vx > goal.vx


def test_follow_command_rotate_in_place_when_goal_behind():
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    current = Pose2D(0.0, 0.0, 0.0)
    target = Pose2D(-2.0, 0.0, 0.0)  # 180 deg behind
    cmd = compute_follow_command(current, target, cfg=cfg)
    assert cmd.vx == 0.0
    assert abs(cmd.vtheta) > 0.0


def test_follow_command_no_sign_flip_across_xy_tolerance():
    """XY jitter across settle must not reverse saturated turn direction."""
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.max_angular_rad_s = 1.5
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    goal = Pose2D(0.0, 0.0, 0.0)
    # Facing ~57°, need to turn CW (negative) to final yaw 0.
    # Inside settle (~3 cm): pure spin for final yaw.
    inside = Pose2D(0.02, 0.0, 1.0)
    # Slight overshoot past the goal — old law RIP'd CCW on ±π bearing.
    outside = Pose2D(-0.30, 0.0, 1.0)
    cmd_in = compute_follow_command(inside, goal, cfg=cfg, final_yaw=0.0)
    cmd_out = compute_follow_command(outside, goal, cfg=cfg, final_yaw=0.0)
    assert cmd_in.vtheta < 0.0
    assert cmd_out.vtheta < 0.0 or cmd_out.vx < 0.0  # reverse crawl or CW yaw
    assert abs(cmd_in.vtheta) <= 0.40 + 1e-6
    assert abs(cmd_out.vtheta) <= 0.40 + 1e-6


def test_follow_command_yaw_settled_closes_xy_without_wiggle():
    """Repro: yaw already at goal θ, 0.38 m out — must drive, not creep+stall."""
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_linear_mps = 0.6
    # Face goal yaw already; goal is ahead-left (~bearing not tiny).
    current = Pose2D(0.333, 0.631, -1.659)
    goal = Pose2D(0.694, 0.496, -1.606)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert abs(cmd.vx) >= 0.05 or abs(cmd.vtheta) >= 0.1
    # Must not be the old tiny crawl that stall ignores.
    if abs(cmd.vx) > 1e-6:
        assert abs(cmd.vx) >= 0.05


def test_follow_command_just_outside_2x_tol_no_full_spin():
    """Repro: ~0.53 m out, yaw nearly settled — must not ±max_vel_theta wiggle."""
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(-1.131, 1.022, -1.038)
    goal = Pose2D(-1.652, 0.918, -0.738)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    # Old bug: rotate_in_place at ±1.0 with vx=0 just outside the 2×tol ball.
    assert abs(cmd.vtheta) <= 0.40 + 1e-6
    # Translating cmds must clear the ViamWorldIO sanitizer.
    if abs(cmd.vx) > 1e-6:
        assert abs(cmd.vx) >= 0.12 - 1e-6
        assert abs(cmd.vtheta) <= 0.25 + 1e-6


def test_follow_command_dock_end_no_point_facing_rip():
    """Repro: ~9 cm out, ~57° off dock yaw — don't RIP to face the point.

    Status dump: spinning |vθ|=0.4 with vx=0 while hunting point bearing,
    then crawl, then final-yaw the other way — endless end swing.
    """
    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(-0.972, 1.663, -0.844)
    goal = Pose2D(-1.051, 1.622, 0.042)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    # Prefer final-yaw handoff (close enough) or soft crawl — never long
    # in-place spin solely to face the XY point.
    if cmd.vx == 0.0:
        # Final yaw spin: error is toward +0.042 from -0.844 → positive vθ.
        yaw_err = conv.normalize_angle(goal.theta - current.theta)
        assert cmd.vtheta * yaw_err > 0.0
    else:
        sx, _, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
        assert abs(sx) >= 0.05 - 1e-6
        assert abs(st) <= 0.25 + 1e-6


def test_follow_command_inside_tol_soft_close_no_rip():
    """Inside XY tol with mid bearing: crawl, do not pure-spin to face point."""
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    # ~0.18 m out (beyond final-yaw handoff), final yaw still off, bearing ~56°.
    current = Pose2D(0.0, 0.0, 0.0)
    goal = Pose2D(0.10, 0.15, 0.8)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    assert abs(cmd.vx) > 1e-6
    assert abs(cmd.vtheta) <= 0.22 + 1e-6


def test_follow_command_end_approach_no_tiny_reverse_yaw_hunt():
    """Repro: 0.29 m out, goal behind heading — don't emit vx=-0.06 + vθ=0.28."""
    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(1.745, 0.298, -2.887)
    goal = Pose2D(2.031, 0.340, -2.473)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    sx, _sy, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    # After sanitize, still making progress (translate or intentional pure spin).
    assert abs(sx) >= 0.12 - 1e-6 or (abs(sx) < 1e-9 and abs(st) >= 0.08)


def test_follow_command_half_metre_final_yaw_closes_xy():
    """Repro: ~0.5 m out with ~35° final yaw must not pure-spin (stall)."""
    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    # Matches status dump: goal south of robot, final yaw differs ~35°.
    current = Pose2D(2.513, 2.105, 2.45)
    goal = Pose2D(2.498, 1.610, 1.835)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    sx, _, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    # Old bug: spin for final yaw (|vθ|≈0.4, vx=0) while still 0.5 m out.
    assert abs(sx) >= 0.12 - 1e-6 or (abs(sx) < 1e-9 and abs(st) >= 0.08)
    assert not (cmd.vx == 0.0 and abs(cmd.vtheta) >= 0.30)


def test_follow_command_large_yaw_outside_tol_closes_xy():
    """~140° final yaw at 0.3 m: close XY first; do not spin for goal θ yet."""
    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(2.52, 1.40, 1.08)
    goal = Pose2D(2.79, 1.56, -2.76)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    sx, _, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    # Must make XY progress (or face the point), not pure-spin for final yaw.
    assert abs(sx) >= 0.12 - 1e-6 or (abs(sx) < 1e-9 and abs(st) >= 0.08)


def test_follow_command_edge_of_xy_tol_still_closes():
    """Inside xy_tol (~0.22 m) but outside settle: soft-close, don't yaw-spin."""
    import math

    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    goal = Pose2D(5.663, 4.573, 0.10)
    # Already facing the goal point; final yaw still far off.
    heading = math.atan2(goal.y - 4.402, goal.x - 5.826)
    current = Pose2D(5.826, 4.402, heading)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    sx, _, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    assert abs(sx) >= 0.05 - 1e-6
    # Must not be final-yaw-only spin at the outer edge of the ball.
    assert abs(cmd.vx) > 0.0
    assert abs(cmd.vtheta) < 0.25


def test_follow_command_large_yaw_inside_settle_spins():
    """Inside settle radius (~3 cm) with large final yaw: pure spin only."""
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    current = Pose2D(2.775, 1.550, 1.08)
    goal = Pose2D(2.79, 1.56, -2.76)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert cmd.vx == 0.0
    assert abs(cmd.vtheta) > 0.0
    assert abs(cmd.vtheta) <= 0.40 + 1e-6


def test_follow_command_near_settle_large_yaw_spins():
    """Repro: ~4 cm out with ~134° final yaw must spin, not soft-crawl forever."""
    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    # Status dump: dist≈0.044, yaw_err≈-2.35
    current = Pose2D(4.962, 2.218, 2.538)
    goal = Pose2D(4.928, 2.247, 0.191)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    assert cmd.vx == 0.0
    assert abs(cmd.vtheta) >= 0.10
    assert abs(cmd.vtheta) <= 0.40 + 1e-6


def test_follow_command_nine_cm_out_hands_off_to_final_yaw():
    """~9 cm out with final yaw still wrong: spin for θ, not soft-crawl forever."""
    import math

    from src.nav_builtin.controller import compute_follow_command

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    goal = Pose2D(4.349, 0.863, 0.624)
    current_xy = (4.272, 0.912)
    heading = math.atan2(goal.y - current_xy[1], goal.x - current_xy[0])
    current = Pose2D(current_xy[0], current_xy[1], heading)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    # Close enough for final-yaw handoff — pure spin toward goal θ.
    assert cmd.vx == 0.0
    yaw_err = conv.normalize_angle(goal.theta - current.theta)
    assert abs(yaw_err) > cfg.motion.yaw_tolerance_rad
    assert cmd.vtheta * yaw_err > 0.0


def test_follow_command_large_yaw_far_out_closes_xy_first():
    """Repro: ~0.75 m out with ~128° final yaw — must not spin in place forever."""
    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    cfg.motion.yaw_tolerance_rad = 0.35
    cfg.motion.max_angular_rad_s = 1.0
    cfg.motion.max_linear_mps = 0.6
    current = Pose2D(-1.282, 0.859, 2.155)
    goal = Pose2D(-2.025, 0.725, -0.108)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=goal.theta)
    assert not cmd.done
    sx, _, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    assert abs(sx) >= 0.12 - 1e-6 or (abs(sx) < 1e-9 and abs(st) >= 0.08)

def test_follow_command_overshoot_reverses_instead_of_spinning():
    from src.nav_builtin.controller import compute_follow_command
    from src.nav_builtin.viam_io import _sanitize_base_cmd

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    # Past the goal, yaw already aligned with goal θ.
    current = Pose2D(0.40, 0.0, 0.0)
    goal = Pose2D(0.0, 0.0, 0.0)
    cmd = compute_follow_command(current, goal, cfg=cfg, final_yaw=0.0)
    assert cmd.vx < 0.0
    assert abs(cmd.vtheta) <= 0.25 + 1e-6
    assert abs(cmd.vx) >= 0.12 - 1e-6
    sx, _, st = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    assert sx < 0.0
    assert not cmd.done

def test_compute_path_command_drives_forward():
    path = Path2D(points=((0.0, 0.0), (2.0, 0.0)), goal_theta=0.0)
    current = Pose2D(0.0, 0.0, 0.0)
    cmd, progress = compute_path_command(current, path, cfg=FollowerConfig())
    assert not cmd.done
    assert cmd.vx > 0.0
    assert progress["distance_remaining_m"] > 0.0


def test_compute_path_command_holds_rotate_into_person():
    """Rotate-to-heading with a body in the nose collision bubble must full-stop."""
    import numpy as np

    from src.geom import conversions as conv
    from src.nav.simple_motion import ObstacleConfig

    path = Path2D(points=((0.0, 0.0), (3.0, 0.0)), goal_theta=0.0)
    # Facing ~120° off the path → rotate-to-heading (vx=0).
    current = Pose2D(0.0, 0.0, math.radians(120.0))
    ranges = np.full(72, np.inf)
    # Forward in base frame ≈ bin at angle 0.
    angle_min = -math.pi
    angle_increment = 2 * math.pi / 72
    ranges[int((0.0 - angle_min) / angle_increment) % 72] = 0.15
    scan = conv.LaserScan2D(ranges, angle_min, angle_increment, range_min=0.05)
    cfg = FollowerConfig(
        obstacle=ObstacleConfig(
            stop_distance_m=0.4, slow_distance_m=1.0, spin_collision_m=0.22
        )
    )
    cmd, progress = compute_path_command(
        current, path, cfg=cfg, scan=scan, rotate_active=True
    )
    assert progress["obstacle"] == "hold"
    assert cmd.vx == 0.0 and cmd.vtheta == 0.0
    assert progress["forward_clearance_m"] == pytest.approx(0.15)


class _FakeWorld:
    def __init__(self, pose: Pose2D, map_data: dict):
        self.pose = pose
        self.map_data = map_data
        self.cmds = []
        self.stopped = False
        self.stop_calls = 0
        self.loc_hold = None
        self.scan = None
        self.lidar_scan = None
        self.scan_calls = []
        self.loc_checks = 0
        self.on_loc_check = None

    def get_map(self):
        return self.map_data

    def get_pose(self):
        return self.pose

    def get_scan(self, max_age_s: float = 2.0, *, include_obstacles_only: bool = True):
        self.scan_calls.append(include_obstacles_only)
        if not include_obstacles_only and self.lidar_scan is not None:
            return self.lidar_scan
        return self.scan

    def get_localization_hold(self):
        return self.loc_hold

    def check_localization(self, **kwargs):
        self.loc_checks += 1
        self.last_loc_kwargs = kwargs
        if callable(self.on_loc_check):
            return self.on_loc_check(self, **kwargs)
        return {"status": "ok", "corrected": False}

    def set_velocity(self, vx, vy, vtheta):
        self.cmds.append((vx, vy, vtheta))
        # Nudge pose toward +x for a trivial follow.
        if vx > 0:
            self.pose = Pose2D(self.pose.x + 0.05, self.pose.y, self.pose.theta)

    def stop(self):
        self.stopped = True
        self.stop_calls += 1
        self.cmds.append((0.0, 0.0, 0.0))

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


def test_try_replan_falls_back_to_scan_plan_and_records_reason():
    """Painted-corridor retry infeasible -> scan-only plan; reasons kept for status."""
    from src.nav_builtin.supervisor import NavSupervisor

    # 0.6 m-wide corridor: painting the route seals it, but a scan-only plan
    # past a bin hugging one wall still exists.
    grid = np.zeros((60, 120), dtype=np.int16)
    grid[:24, :] = 100
    grid[36:, :] = 100
    m = {"grid": grid, "resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0}
    world = _FakeWorld(Pose2D(1.0, 1.5, 0.0), m)
    sup = NavSupervisor(
        world,
        inflation_radius_m=0.15,
        robot_radius_m=0.1,
        avoid_obstacles=False,
        local_costmap_enabled=False,
        local_planner_enabled=False,
    )
    goal = Pose2D(5.0, 1.5, 0.0)
    base = sup.plan(goal, start=world.pose)
    assert base.feasible
    # Bin 0.8 m ahead, offset toward the +y wall, seen by the scan.
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2 + 2] = math.hypot(0.8, 0.15)  # beam at +10 deg -> (1.8, 1.65)
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    painted = sup.plan(
        goal,
        start=world.pose,
        scan=scan,
        blocked_path=base.path,
        blocked_path_pose=world.pose,
    )
    assert not painted.feasible  # corridor sealed by the painted route
    new = sup._try_replan(
        goal, world.pose, base.path, scan, failed_count=1, require_different=True
    )
    assert new is not None
    assert paths_meaningfully_differ(base.path, new)
    assert sup._last_replan_error == ""

    # No scan and nothing painted differently -> same route -> reason recorded.
    same = sup._try_replan(
        goal, world.pose, base.path, None, failed_count=0, require_different=True
    )
    assert same is None
    assert "same route" in sup._last_replan_error


def test_plan_path_marks_local_costmap_for_replan():
    """Local-costmap blob must force a different global path (scan can miss it)."""
    from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig

    m = _empty_map(size=80, resolution=0.05)
    start = Pose2D(0.5, 2.0, 0.0)
    goal = Pose2D(3.5, 2.0, 0.0)
    baseline = plan_path(
        m, start, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert baseline.feasible
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=4.0,
            height_m=4.0,
            resolution=0.05,
            inflation_radius_m=0.25,
            robot_radius_m=0.22,
            use_global_static=False,
        )
    )
    # Bin on the straight route, seen only via the local costmap.
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.0
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(start, scan)
    # Empty scan for plan_path — only local_view carries the obstacle.
    empty = conv.LaserScan2D(
        np.full(n, np.inf),
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    same = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        scan=empty,
        scan_pose=start,
    )
    assert same.feasible
    assert not paths_meaningfully_differ(baseline.path, same.path)
    detoured = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        local_view=view,
    )
    assert detoured.feasible, detoured.error_msg
    assert paths_meaningfully_differ(baseline.path, detoured.path)


def test_plan_path_overlay_avoids_second_local_obstacle():
    """Peeling around one live blob must not thread a second blob in-window."""
    from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig
    from src.nav_builtin.local_planner import path_cost_in_local_window

    m = _empty_map(size=120, resolution=0.05)
    start = Pose2D(1.0, 3.0, 0.0)
    goal = Pose2D(5.0, 3.0, 0.0)
    baseline = plan_path(
        m, start, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert baseline.feasible
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=6.0,
            height_m=6.0,
            resolution=0.05,
            inflation_radius_m=0.30,
            robot_radius_m=0.22,
            use_global_static=False,
            scan_inflation_radius_m=0.30,
        )
    )
    # Two bins: on the straight path, and on the natural +y peel.
    n = 180
    ranges = np.full(n, np.inf)
    # ~1.2 m ahead on centerline
    ranges[n // 2] = 1.2
    # ~1.2 m ahead, ~0.55 m to +y (peel corridor)
    ang = math.atan2(0.55, 1.2)
    idx = int((ang - (-math.pi)) / (2 * math.pi / n)) % n
    ranges[idx] = math.hypot(1.2, 0.55)
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(start, scan)
    # Straight path should be locally blocked.
    assert (
        path_cost_in_local_window(start, baseline.path, view, start_offset_m=0.22)
        >= 200
    )
    detoured = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        local_view=view,
        clearance_preference_m=0.0,
    )
    assert detoured.feasible, detoured.error_msg
    assert paths_meaningfully_differ(baseline.path, detoured.path)
    # Accepted path must stay clear of *both* live blobs inside the window.
    assert (
        path_cost_in_local_window(start, detoured.path, view, start_offset_m=0.22)
        < 200
    ), "replan still intersects a live local obstacle"


def test_try_replan_rejects_path_still_hitting_local_cost():
    """Safety net: do not accept a 'different' route that is still local-lethal."""
    from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig
    from src.nav_builtin.supervisor import NavSupervisor
    from src.nav_builtin.types import PlanResult

    m = _empty_map(size=120, resolution=0.05)
    world = _FakeWorld(Pose2D(1.0, 3.0, 0.0), m)
    sup = NavSupervisor(
        world,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        avoid_obstacles=False,
        local_costmap_enabled=False,
        local_planner_enabled=False,
        clearance_preference_m=0.0,
    )
    goal = Pose2D(5.0, 3.0, 0.0)
    base = plan_path(
        m, world.pose, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert base.feasible
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=6.0,
            height_m=6.0,
            resolution=0.05,
            inflation_radius_m=0.30,
            robot_radius_m=0.22,
            use_global_static=False,
            scan_inflation_radius_m=0.30,
        )
    )
    n = 180
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.2
    ang = math.atan2(0.55, 1.2)
    idx = int((ang - (-math.pi)) / (2 * math.pi / n)) % n
    ranges[idx] = math.hypot(1.2, 0.55)
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(world.pose, scan)

    # Force every plan attempt to return a peel that still hits the +y blob.
    bad_pts = [
        (1.0, 3.0),
        (1.6, 3.55),
        (2.2, 3.55),
        (3.0, 3.55),
        (4.0, 3.2),
        (5.0, 3.0),
    ]
    bad_path = Path2D(points=bad_pts, goal_theta=0.0)

    def sticky_bad(g, start=None, scan=None, **kwargs):
        return PlanResult(feasible=True, path=bad_path)

    sup.plan = sticky_bad  # type: ignore[method-assign]
    new = sup._try_replan(
        goal,
        world.pose,
        base.path,
        scan,
        failed_count=2,
        require_different=True,
        local_view=view,
    )
    assert new is None
    assert "still local-blocked" in (sup._last_replan_error or "")
    assert any(
        "still local-blocked" in str(a) for a in (sup._last_replan_info or {}).get("attempts", [])
    )


def test_builtin_navigator_cancel_sets_status():
    """Open room: scan peels around a bin; corridor paint must not replace it
    with a room-scale loop."""
    from src.nav_builtin.controller import _path_length
    from src.nav_builtin.supervisor import NavSupervisor

    m = _empty_map(size=120, resolution=0.05)
    world = _FakeWorld(Pose2D(1.0, 3.0, 0.0), m)
    # Mark a bin on the map so a scan-free paint has something to avoid too.
    m["grid"][58:62, 50:54] = 100  # ~ (2.6, 3.0)
    world.map_data = m
    sup = NavSupervisor(
        world,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        avoid_obstacles=False,
        local_costmap_enabled=False,
        local_planner_enabled=False,
        clearance_preference_m=0.0,
    )
    goal = Pose2D(5.0, 3.0, 0.0)
    base = plan_path(
        m, world.pose, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert base.feasible
    base_len = _path_length(base.path)
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.5
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    new = sup._try_replan(
        goal, world.pose, base.path, scan, failed_count=3, require_different=True
    )
    assert new is not None
    # Mild peel — not a multi-metre loop around the room.
    assert _path_length(new) < base_len * 1.6


def test_forced_side_detour_peels_around_blocked_sample():
    """Synthetic left/right via must leave the corridor without a room loop."""
    from src.nav_builtin.controller import _path_length
    from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig
    from src.nav_builtin.supervisor import NavSupervisor

    m = _empty_map(size=120, resolution=0.05)
    world = _FakeWorld(Pose2D(1.0, 3.0, 0.0), m)
    sup = NavSupervisor(
        world,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        avoid_obstacles=False,
        local_costmap_enabled=False,
        local_planner_enabled=False,
        clearance_preference_m=0.0,
    )
    goal = Pose2D(5.0, 3.0, 0.0)
    base = plan_path(
        m, world.pose, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert base.feasible
    base_len = _path_length(base.path)
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=4.0,
            height_m=4.0,
            resolution=0.05,
            inflation_radius_m=0.25,
            robot_radius_m=0.22,
            use_global_static=False,
        )
    )
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.2
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(world.pose, scan)
    forced = sup._forced_side_detour(goal, world.pose, base.path, scan, view)
    assert forced is not None, "expected a left/right via peel"
    new_path, _result, label = forced
    assert label.startswith("via")
    assert paths_meaningfully_differ(base.path, new_path, tol_m=0.12)
    assert _path_length(new_path) < base_len * 2.2


def test_try_replan_falls_to_forced_via_when_plans_identical():
    """If scan+local/paint keep returning the same path, use forced via."""
    from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig
    from src.nav_builtin.supervisor import NavSupervisor
    from src.nav_builtin.types import PlanResult

    m = _empty_map(size=120, resolution=0.05)
    world = _FakeWorld(Pose2D(1.0, 3.0, 0.0), m)
    sup = NavSupervisor(
        world,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        avoid_obstacles=False,
        local_costmap_enabled=False,
        local_planner_enabled=False,
        clearance_preference_m=0.0,
    )
    goal = Pose2D(5.0, 3.0, 0.0)
    base = plan_path(
        m, world.pose, goal, inflation_radius_m=0.25, robot_radius_m=0.22
    )
    assert base.feasible
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=4.0,
            height_m=4.0,
            resolution=0.05,
            inflation_radius_m=0.25,
            robot_radius_m=0.22,
            use_global_static=False,
        )
    )
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.2
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(world.pose, scan)

    real_plan = sup.plan

    def sticky_same_route(g, start=None, scan=None, **kwargs):
        # From the live pose to the final goal: pretend the planner is stuck
        # on the old corridor. Via hops (different goals / starts) plan for real.
        start_pose = start if start is not None else world.pose
        at_live = (
            abs(start_pose.x - world.pose.x) < 1e-6
            and abs(start_pose.y - world.pose.y) < 1e-6
        )
        to_goal = abs(g.x - goal.x) < 1e-6 and abs(g.y - goal.y) < 1e-6
        if at_live and to_goal:
            return PlanResult(feasible=True, path=base.path)
        return real_plan(g, start=start, scan=scan, **kwargs)

    sup.plan = sticky_same_route  # type: ignore[method-assign]
    new = sup._try_replan(
        goal,
        world.pose,
        base.path,
        scan,
        failed_count=2,
        require_different=True,
        local_view=view,
    )
    assert new is not None
    assert paths_meaningfully_differ(base.path, new, tol_m=0.12)


def test_builtin_navigator_cancel_sets_status():
    world = _FakeWorld(Pose2D(0.2, 0.2, 0.0), _empty_map())
    nav = BuiltinNavigator(world, avoid_obstacles=False)
    nav.cancel()
    assert nav.nav_status()["state"] == "canceled"


def test_nav_holds_drive_while_localization_awaiting_confirm():
    """Do not crawl/turn on a disputed pose while a large jump awaits confirm."""
    import threading
    import time

    world = _FakeWorld(Pose2D(0.2, 0.2, 0.0), _empty_map(size=80))
    world.loc_hold = {
        "status": "awaiting_confirm",
        "confirm_count": 1,
        "confirm_needed": 2,
        "jump_shift_m": 0.49,
        "jump_shift_deg": 42.0,
    }
    nav = BuiltinNavigator(
        world,
        inflation_radius_m=0.15,
        robot_radius_m=0.05,
        avoid_obstacles=False,
        xy_tolerance_m=0.1,
        timeout_s=4.0,
        local_costmap_enabled=False,
        local_planner_enabled=False,
    )

    def _run():
        nav.navigate(3.0, 0.2, 0.0)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.35)
    status = nav.nav_status()
    assert status.get("active") is True
    assert status.get("obstacle") == "loc_hold"
    # Stop once on hold entry — not every control tick.
    assert world.stop_calls == 1
    # No forward or turn commands while held (stops only).
    assert all(abs(vx) < 1e-9 and abs(vth) < 1e-9 for vx, _vy, vth in world.cmds)
    world.loc_hold = None
    time.sleep(0.25)
    nav.cancel()
    t.join(timeout=2.0)
    assert world.stopped


def test_nav_does_not_sticky_hold_leftover_nav_hold():
    """A refused large mid-nav jump must not sit in loc_hold forever."""
    import threading
    import time

    world = _FakeWorld(Pose2D(0.2, 0.2, 0.0), _empty_map(size=80))
    world.loc_hold = {
        "status": "nav_hold",
        "shift_m": 3.384,
        "shift_deg": 2.0,
        "score": 0.25,
        "large_jump": True,
    }
    nav = BuiltinNavigator(
        world,
        inflation_radius_m=0.15,
        robot_radius_m=0.05,
        avoid_obstacles=False,
        xy_tolerance_m=0.1,
        timeout_s=4.0,
        local_costmap_enabled=False,
        local_planner_enabled=False,
        nav_loc_refine_on_disagree=True,
    )

    def _run():
        nav.navigate(3.0, 0.2, 0.0)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.35)
    status = nav.nav_status()
    assert status.get("active") is True
    assert status.get("obstacle") != "loc_hold"
    assert any(abs(vx) > 1e-6 or abs(vth) > 1e-6 for vx, _vy, vth in world.cmds)
    nav.cancel()
    t.join(timeout=2.0)
    assert world.stopped


def _open_scan(range_m: float = 4.0, n: int = 360) -> conv.LaserScan2D:
    return conv.LaserScan2D(
        ranges=np.full(n, range_m, dtype=float),
        angle_min=-math.pi,
        angle_increment=(2.0 * math.pi) / n,
        range_min=0.05,
        range_max=25.0,
    )


def _left_wall_map() -> dict:
    """Free space with a wall ~0.55 m to the left of (1.0, 1.0) facing +X."""
    grid = np.zeros((80, 80), dtype=np.int16)
    grid[31, 10:50] = 100
    return {
        "grid": grid,
        "resolution": 0.05,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }


def _loc_refine_supervisor(world: _FakeWorld) -> NavSupervisor:
    return NavSupervisor(
        world,
        inflation_radius_m=0.15,
        robot_radius_m=0.08,
        avoid_obstacles=False,
        local_costmap_enabled=False,
        local_planner_enabled=False,
        xy_tolerance_m=0.15,
        timeout_s=6.0,
        poll_interval_s=0.02,
        nav_loc_refine_on_disagree=True,
        nav_loc_refine_settle_s=0.0,
        nav_loc_refine_period_s=0.0,
        nav_loc_refine_cooldown_s=0.0,
        nav_loc_refine_max_tries=2,
    )


def test_lidar_scan_for_loc_refine_excludes_obstacles_only():
    """Loc refine must read lidar-only; fused depth is for avoidance."""
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map())
    fused = _open_scan(0.3)
    lidar = _open_scan(4.0)
    world.scan = fused
    world.lidar_scan = lidar
    sup = _loc_refine_supervisor(world)
    got = sup._lidar_scan_for_loc_refine()  # noqa: SLF001
    assert got is lidar
    assert world.scan_calls == [False]


def test_follow_loop_uses_fused_scan_for_local_costmap():
    """Follow ticks request the fused scan; depth hits reach the local costmap."""
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map(size=80))
    n = 8
    inc = math.pi / 4.0
    fused_r = np.full(n, 2.0)
    fused_r[4] = 0.30
    world.scan = conv.LaserScan2D(
        fused_r,
        angle_min=-math.pi,
        angle_increment=inc,
        range_min=0.05,
        range_max=10.0,
    )
    world.lidar_scan = conv.LaserScan2D(
        np.full(n, 2.0),
        angle_min=-math.pi,
        angle_increment=inc,
        range_min=0.05,
        range_max=10.0,
    )
    sup = NavSupervisor(
        world,
        inflation_radius_m=0.10,
        robot_radius_m=0.05,
        avoid_obstacles=True,
        local_costmap_enabled=True,
        xy_tolerance_m=0.15,
        timeout_s=1.5,
        poll_interval_s=0.02,
        nav_loc_refine_on_disagree=False,
    )
    sup.run_goal(Pose2D(1.6, 1.0, 0.0))
    assert world.scan_calls
    assert all(world.scan_calls)
    view = sup._local_view_cache  # noqa: SLF001
    assert view is not None
    assert view.cost_at_world(1.30, 1.0) > 0


def test_nav_loc_refine_resumes_when_disagreement_clears():
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _left_wall_map())
    world.scan = _open_scan()

    def _fix(w: _FakeWorld, **_kw):
        w.scan = _open_scan(0.55)
        return {"status": "ok", "corrected": True}

    world.on_loc_check = _fix
    sup = _loc_refine_supervisor(world)
    sup.run_goal(Pose2D(1.6, 1.0, 0.0))
    assert world.loc_checks == 1
    assert sup.status().state == "succeeded"
    assert sup.status().error_msg == ""


def test_nav_loc_refine_continues_after_two_tries():
    """A leftover residual after two local tries must not abort the goal."""
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _left_wall_map())
    world.scan = _open_scan()
    sup = _loc_refine_supervisor(world)
    sup.run_goal(Pose2D(1.6, 1.0, 0.0))
    assert world.loc_checks == 2
    st = sup.status()
    assert st.state == "succeeded"
    assert st.error_msg == ""


def test_nav_loc_refine_continues_when_residual_is_thin():
    """A 4/12 leftover is not localization_lost — keep the goal."""
    from src.nav_builtin.loc_consistency import LocDisagreement

    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map(size=80))
    mild = LocDisagreement(
        disagree=True,
        reason="scan_map",
        compared_beams=12,
        disagree_beams=4,
        disagree_frac=0.333,
    )
    sup = _loc_refine_supervisor(world)
    calls = {"n": 0}

    def _measure(_pose):
        calls["n"] += 1
        if calls["n"] <= 4:
            return mild
        return LocDisagreement(disagree=False)

    sup._measure_loc_disagreement = _measure  # noqa: SLF001
    sup.run_goal(Pose2D(1.6, 1.0, 0.0))
    st = sup.status()
    assert st.state == "succeeded"
    assert st.error_msg == ""
    # Thin residual SLAM could not fix: one look, then keep driving.
    assert world.loc_checks == 1


def test_goal_timeout_scales_with_route_length():
    world = _FakeWorld(Pose2D(0.0, 0.0, 0.0), _empty_map())
    sup = NavSupervisor(world, timeout_s=300.0, max_vel_x=0.55)
    assert sup._goal_timeout_s(10.0) == pytest.approx(300.0)  # noqa: SLF001
    # 110 m at 0.55 m/s = 200 s of driving -> 600 s budget.
    assert sup._goal_timeout_s(110.0) == pytest.approx(600.0)  # noqa: SLF001


def test_loc_refine_replans_only_when_pose_moved():
    kind = NavSupervisor._loc_refine_resume_kind  # noqa: SLF001
    start = Pose2D(1.0, 1.0, 0.0)
    assert kind(start, Pose2D(1.03, 1.02, 0.01)) == "continue"
    assert kind(start, Pose2D(1.4, 1.0, 0.0)) == "resume"
    assert kind(start, Pose2D(1.0, 1.0, math.radians(8.0))) == "resume"
    assert kind(None, start) == "resume"


def test_nav_loc_refine_applies_small_improving_match():
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _left_wall_map())
    world.scan = _open_scan()
    applies = []

    def _check(w: _FakeWorld, **kw):
        applies.append(kw.get("apply"))
        if kw.get("apply"):
            w.scan = _open_scan(0.55)
            return {"status": "ok", "corrected": True, "match_mode": "local"}
        return {
            "status": "nav_hold",
            "corrected": False,
            "good_match": False,
            "match_mode": "local",
            "shift_m": 0.32,
            "shift_deg": 20.0,
            "score": 0.47,
            "previous_score": -0.17,
            "large_jump": False,
            "drifted": True,
        }

    world.on_loc_check = _check
    sup = _loc_refine_supervisor(world)
    sup.run_goal(Pose2D(1.6, 1.0, 0.0))
    assert True in applies
    assert sup.status().state == "succeeded"
    assert sup.status().error_msg == ""

"""Tests for path smoother, local costmap, and local planner."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.nav.simple_motion import DriveCommand, rear_clearance_m
from src.nav_builtin.controller import FollowerConfig, compute_path_command
from src.nav_builtin.costmap import build_costmap, occupancy_from_map_dict
from src.nav_builtin.local_costmap import (
    LocalCostmap,
    LocalCostmapConfig,
    footprint_collides,
    reverse_backup_feasible,
)
from src.nav_builtin.local_planner import LocalPlannerConfig, compute_local_command
from src.nav_builtin.smoother import smooth_path
from src.nav_builtin.types import OccupancyGrid, Path2D, Pose2D
from src.geom import conversions as conv


def _empty_map(size: int = 40, resolution: float = 0.05) -> dict:
    return {
        "grid": np.zeros((size, size), dtype=np.int16),
        "resolution": resolution,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }


def test_rear_clearance_m_ignores_forward_returns():
    n = 36
    ranges = np.full(n, 3.0)
    ranges[0] = 0.4  # rear beam (angle ≈ −π)
    ranges[n // 2] = 0.25  # forward beam — must not dominate rear reading
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    assert rear_clearance_m(scan) == pytest.approx(0.4)


def test_smooth_path_shortens_zigzag_astar():
    m = _empty_map(size=60, resolution=0.05)
    occ = occupancy_from_map_dict(m)
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


def test_local_costmap_respects_sensor_pose():
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
    pose = conv.Pose2D(1.0, 1.0, 0.0)
    n = 36
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.0  # angle 0 = robot +X
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
        sensor_pose=conv.Pose2D(0.5, 0.0, 0.0),
    )
    view = lc.update(pose, scan)
    # Hit should land ~1.5 m ahead in map (+X), not 1.0 m.
    assert view.cost_at_world(2.5, 1.0) > 0
    assert view.cost_at_world(1.0, 1.0) == 0


def test_local_costmap_syncs_stale_scan_pose():
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=4.0,
            height_m=4.0,
            resolution=0.05,
            inflation_radius_m=0.10,
            robot_radius_m=0.05,
            use_global_static=False,
        )
    )
    n = 72
    ranges = np.full(n, np.inf)
    ranges[n // 2] = 1.0  # angle 0 = robot +X
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
        capture_pose=conv.Pose2D(0.0, 0.0, 0.0),
    )
    current = conv.Pose2D(1.0, 0.0, 0.0)
    view = lc.update(current, scan)
    # Wall was at world (1, 0) when scanned; robot moved +1 m without re-scanning.
    assert view.cost_at_world(1.0, 0.0) > 0
    assert view.cost_at_world(2.0, 0.0) == 0


def test_local_costmap_does_not_reinflate_global_static():
    """Global static in the local window must not get a second inflation pass."""
    from src.nav_builtin.costmap import INSCRIBED, build_costmap, occupancy_from_map_dict

    m = _empty_map(size=60, resolution=0.05)
    occ = occupancy_from_map_dict(m)
    occ.grid[30, 30] = 100
    global_costs = build_costmap(
        occ, inflation_radius_m=0.25, robot_radius_m=0.22, cost_scaling_factor=4.0
    )
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=2.0,
            height_m=2.0,
            resolution=0.05,
            inflation_radius_m=0.25,
            robot_radius_m=0.22,
            use_global_static=True,
        )
    )
    pose = conv.Pose2D(1.5, 1.5, 0.0)
    view = lc.update(pose, None, global_occ=occ, global_costs=global_costs)
    # Just outside the global hard halo should remain free (double inflate would block).
    wx, wy = occ.cell_to_world(30, 24)
    assert int(global_costs[30, 24]) < INSCRIBED
    assert view.cost_at_world(wx, wy) == int(global_costs[30, 24])
    # Inside the halo, local must match global exactly — not a wider ring.
    wx2, wy2 = occ.cell_to_world(30, 28)
    assert view.cost_at_world(wx2, wy2) == int(global_costs[30, 28])


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


def test_path_cost_ahead_margin_catches_obstacle_just_outside_footprint():
    """Live hits inflate by robot_radius; a hit 1 cell past that reads free on
    the centerline. The margin disc must still flag it as blocked."""
    from src.nav_builtin.local_planner import path_cost_ahead
    from src.nav_builtin.planner import path_blocked_local

    robot_r = 0.20
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=3.0,
            height_m=3.0,
            resolution=0.05,
            inflation_radius_m=robot_r,
            robot_radius_m=robot_r,
            use_global_static=False,
        )
    )
    pose = Pose2D(1.5, 1.5, 0.0)
    # Path straight ahead along y=1.5; obstacle at (2.2, 1.5 + 0.27): 0.27 m
    # off the centerline, robot_r + 1 cell → centerline cost 0.
    dy = robot_r + 0.07
    bearing = math.atan2(dy, 0.7)
    rng = math.hypot(0.7, dy)
    n = 360
    ranges = np.full(n, math.inf)
    b = int((bearing + math.pi) / (2 * math.pi / n)) % n
    ranges[b] = rng
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(pose, scan)
    # Stay inside the 3 m local window (x < 3.0) so bounds never read lethal.
    path = Path2D(points=[(1.5, 1.5), (2.2, 1.5), (2.7, 1.5)], goal_theta=0.0)

    centerline = path_cost_ahead(pose, path, view, lookahead_m=1.2)
    with_margin = path_cost_ahead(pose, path, view, lookahead_m=1.2, margin_m=0.10)
    assert centerline < 200
    assert with_margin >= 200
    assert (
        path_blocked_local(pose, path, view, cost_threshold=200, lookahead_m=1.2)
        is False
    )
    assert path_blocked_local(
        pose, path, view, cost_threshold=200, lookahead_m=1.2, margin_m=0.10
    )


def test_reverse_backup_feasible_requires_cost_improvement():
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=3.0,
            height_m=3.0,
            resolution=0.05,
            inflation_radius_m=0.15,
            robot_radius_m=0.08,
            use_global_static=False,
        )
    )
    th = math.pi / 2
    n = 36
    ranges_ahead = np.full(n, 3.0)
    ranges_ahead[n // 2] = 0.25
    scan_ahead = conv.LaserScan2D(
        ranges_ahead,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(conv.Pose2D(1.5, 1.5, th), scan_ahead)
    assert reverse_backup_feasible(
        view, 1.5, 1.5, th, robot_radius_m=0.08, distance_m=0.35
    )
    # Wall behind robot — reverse would deepen overlap; must reject.
    ranges_rear = np.full(n, 3.0)
    ranges_rear[0] = 0.25
    scan_rear = conv.LaserScan2D(
        ranges_rear,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view_rear = lc.update(conv.Pose2D(1.5, 1.5, th), scan_rear)
    assert not reverse_backup_feasible(
        view_rear, 1.5, 1.5, th, robot_radius_m=0.08, distance_m=0.35
    )


def test_local_planner_prefers_reverse_when_blocked_ahead():
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=3.0,
            height_m=3.0,
            resolution=0.05,
            inflation_radius_m=0.15,
            robot_radius_m=0.08,
            use_global_static=False,
        )
    )
    pose = Pose2D(1.5, 1.5, 0.0)
    n = 36
    ranges = np.full(n, 3.0)
    ranges[n // 2] = 0.35
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(pose, scan)
    path = Path2D(points=((1.5, 1.5), (2.5, 1.5)), goal_theta=0.0)
    cfg = LocalPlannerConfig(
        enabled=True,
        activate_cost_threshold=1,
        sim_time_s=1.0,
        max_vel_x_reverse_m=0.15,
    )
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
    # Path blocked ahead → detour (turn) or hold; never drive straight in.
    assert cmd.vx <= 0.05 or abs(cmd.vtheta) > 0.1


def _open_view_with_bin(pose: Pose2D, bin_xy: tuple[float, float], *, robot_radius_m: float = 0.22):
    """4 m open window around ``pose`` with a ~0.3 m box marked at ``bin_xy``."""
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=4.0,
            height_m=4.0,
            resolution=0.05,
            inflation_radius_m=0.35,
            robot_radius_m=robot_radius_m,
            use_global_static=False,
        )
    )
    # Build a scan whose hits paint the bin faces, everything else far.
    n = 360
    ranges = np.full(n, 8.0)
    angle_min = -math.pi
    inc = 2 * math.pi / n
    bx, by = bin_xy
    for dx in np.linspace(-0.15, 0.15, 7):
        for dy in np.linspace(-0.15, 0.15, 7):
            wx, wy = bx + dx, by + dy
            rx = wx - pose.x
            ry = wy - pose.y
            ang = conv.normalize_angle(math.atan2(ry, rx) - pose.theta)
            k = int(round((ang - angle_min) / inc)) % n
            ranges[k] = min(ranges[k], math.hypot(rx, ry))
    scan = conv.LaserScan2D(
        ranges,
        angle_min=angle_min,
        angle_increment=inc,
        range_min=0.05,
        range_max=10.0,
    )
    return lc.update(pose, scan)


def test_local_planner_detours_around_bin_on_route():
    """Lethal cost *on the route* must not veto every forward rollout."""
    pose = Pose2D(0.0, 0.0, 0.0)
    path = Path2D(points=((0.0, 0.0), (3.0, 0.0)), goal_theta=0.0)
    view = _open_view_with_bin(pose, (1.0, 0.0))
    cfg = LocalPlannerConfig(enabled=True, sim_time_s=1.2)
    cmd = compute_local_command(
        pose,
        path,
        view,
        cfg=cfg,
        max_vel_x=0.4,
        max_vel_theta=1.0,
        robot_radius_m=0.22,
    )
    assert cmd is not None
    # Route is blocked, flanks are open: expect a forward arc, not rotate-only.
    assert cmd.vx > 0.05
    assert abs(cmd.vtheta) > 0.05


@pytest.mark.parametrize(
    "robot_r, bin_x",
    [(0.22, 1.0), (0.22, 1.2), (0.32, 1.0), (0.45, 1.2)],
)
def test_local_planner_closed_loop_detour_passes_bin(robot_r, bin_x):
    """Drive the DWA loop against a bin on the route: must get past, not creep/spin."""
    path = Path2D(points=((0.0, 0.0), (3.0, 0.0)), goal_theta=0.0)
    bin_xy = (bin_x, 0.0)
    pose = Pose2D(0.0, 0.0, 0.0)
    cfg = LocalPlannerConfig(enabled=True, sim_time_s=1.2)
    prev = None
    active = False
    dt = 0.1
    min_clear = math.inf
    max_abs_y = 0.0
    used_dwa = 0
    for _ in range(300):
        view = _open_view_with_bin(pose, bin_xy, robot_radius_m=robot_r)
        cmd = compute_local_command(
            pose,
            path,
            view,
            cfg=cfg,
            max_vel_x=0.4,
            max_vel_theta=1.0,
            robot_radius_m=robot_r,
            local_planner_active=active,
            prev_cmd=prev if active else None,
        )
        if cmd is None:
            # Route clear locally: plain pursuit toward a point 0.5 m ahead.
            tx, ty = path_point_ahead_for_test(path, pose, 0.5)
            bearing = conv.normalize_angle(
                math.atan2(ty - pose.y, tx - pose.x) - pose.theta
            )
            cmd = DriveCommand(0.3 if abs(bearing) < 1.0 else 0.0, 0.0, max(-1.0, min(1.0, 2.0 * bearing)), False)
            active = False
            prev = None
        else:
            active = True
            prev = cmd
            used_dwa += 1
        th = pose.theta + cmd.vtheta * dt
        pose = Pose2D(
            pose.x + math.cos(pose.theta) * cmd.vx * dt,
            pose.y + math.sin(pose.theta) * cmd.vx * dt,
            conv.normalize_angle(th),
        )
        clear = math.hypot(pose.x - bin_xy[0], pose.y - bin_xy[1])
        min_clear = min(min_clear, clear)
        max_abs_y = max(max_abs_y, abs(pose.y))
        if pose.x > bin_x + 0.6:
            break
    assert used_dwa > 0
    assert pose.x > bin_x + 0.6, f"never passed the bin: ended at {pose}"
    # Never overlap the bin (half-size 0.15) with the footprint.
    assert min_clear > 0.15 + robot_r, f"clipped the bin: {min_clear:.2f} m"
    # Middle peel: clear the footprint, but don't swing past ~1.1 m off path.
    need = 0.15 + robot_r
    assert max_abs_y > need * 0.85, f"too tight: max|y|={max_abs_y:.2f}"
    assert max_abs_y < 1.15, f"too wide: max|y|={max_abs_y:.2f}"


def path_point_ahead_for_test(path: Path2D, pose: Pose2D, ahead: float):
    from src.nav_builtin.local_planner import path_point_ahead

    return path_point_ahead(path, pose.x, pose.y, ahead)


def test_local_planner_blocked_turns_toward_route_not_away():
    """Facing ~106° off the route with the route blocked: rotate toward it."""
    # Route runs at -54° (south-east); robot faces +52°.
    theta_path = math.radians(-54.0)
    pose = Pose2D(0.0, 0.0, math.radians(52.0))
    pts = tuple(
        (k * 0.2 * math.cos(theta_path), k * 0.2 * math.sin(theta_path))
        for k in range(15)
    )
    path = Path2D(points=pts, goal_theta=theta_path)
    bin_xy = (0.7 * math.cos(theta_path), 0.7 * math.sin(theta_path))
    view = _open_view_with_bin(pose, bin_xy)
    cfg = LocalPlannerConfig(enabled=True, sim_time_s=1.2)
    # Seed continuity with the *wrong-way* spin the robot had locked into.
    prev = DriveCommand(0.0, 0.0, 0.75, False)
    cmd = compute_local_command(
        pose,
        path,
        view,
        cfg=cfg,
        max_vel_x=0.4,
        max_vel_theta=0.75,
        robot_radius_m=0.22,
        local_planner_active=True,
        prev_cmd=prev,
    )
    assert cmd is not None
    # Bearing to the route is negative (clockwise) → vθ must be negative.
    assert cmd.vtheta < 0.0


def test_local_planner_does_not_charge_blocked_corridor():
    """Aligned with a lethal cell ~1 m ahead: must peel/turn, not cmd_vx=max."""
    pose = Pose2D(0.0, 0.0, 0.0)
    path = Path2D(points=((0.0, 0.0), (3.0, 0.0)), goal_theta=0.0)
    view = _open_view_with_bin(pose, (1.0, 0.0), robot_radius_m=0.32)
    cfg = LocalPlannerConfig(enabled=True, sim_time_s=1.2)
    cmd = compute_local_command(
        pose,
        path,
        view,
        cfg=cfg,
        max_vel_x=0.5,
        max_vel_theta=1.0,
        robot_radius_m=0.32,
    )
    assert cmd is not None
    assert cmd.vx <= cfg.max_detour_forward_mps + 1e-9
    assert not (cmd.vx >= 0.35 and abs(cmd.vtheta) < 0.05), f"charged: {cmd}"


def test_local_replan_paints_raw_hits_not_merged_static_inflation():
    """Do not turn an already-inflated mapped wall into new occupancy."""
    from src.nav_builtin.costmap import mark_local_costmap_on_occupancy
    from src.nav_builtin.local_costmap import LocalCostmapView

    grid = np.zeros((40, 40), dtype=np.int16)
    grid[20, 20] = 100
    occ = OccupancyGrid(
        grid=grid,
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    raw = np.zeros_like(grid)
    raw[20, 20] = 100  # lidar return from the mapped wall: skip
    raw[20, 28] = 100  # novel box in mapped-free space: retain
    local_occ = OccupancyGrid(
        grid=raw,
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    merged_costs = np.zeros_like(grid, dtype=np.uint8)
    merged_costs[15:26, 15:26] = 253  # projected static inflation
    merged_costs[20, 28] = 253
    view = LocalCostmapView(
        costs=merged_costs,
        occ=local_occ,
        origin_x=0.0,
        origin_y=0.0,
    )

    marked = mark_local_costmap_on_occupancy(occ, view, radius_m=0.05)
    # Static inflation was not copied as a large raw obstacle.
    assert np.count_nonzero(marked.grid[15:26, 15:26]) == 1
    # The novel raw hit was copied.
    assert marked.grid[20, 28] == 100


def test_corridor_paint_preserves_requested_goal():
    """A short remaining route must keep paint/inflation away from its goal."""
    from src.nav_builtin.planner import plan_path

    m = _empty_map(size=80, resolution=0.05)
    start = Pose2D(0.5, 2.0, 0.0)
    goal = Pose2D(2.5, 2.0, 0.0)
    base = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
    )
    assert base.feasible
    detour = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        blocked_path=base.path,
        blocked_path_pose=start,
        paint_corridor=True,
    )
    assert detour.feasible, detour.error_msg
    end_x, end_y = detour.path.points[-1]
    assert math.hypot(end_x - goal.x, end_y - goal.y) < 0.1


def test_plan_seals_blocked_path_samples_from_local():
    """Fat seal on path samples the local map flags must change the route."""
    from src.nav_builtin.local_costmap import LocalCostmap, LocalCostmapConfig
    from src.nav_builtin.planner import plan_path, paths_meaningfully_differ

    m = {
        "grid": np.zeros((80, 80), dtype=np.int16),
        "resolution": 0.05,
        "origin_x": 0.0,
        "origin_y": 0.0,
    }
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
    sealed = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        local_view=view,
        blocked_path=baseline.path,
        blocked_path_pose=start,
        paint_corridor=False,
    )
    assert sealed.feasible, sealed.error_msg
    assert paths_meaningfully_differ(baseline.path, sealed.path)


def test_local_planner_avoids_marked_obstacle():
    """Fat seal on path samples the local map flags must change the route."""
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
    sealed = plan_path(
        m,
        start,
        goal,
        inflation_radius_m=0.25,
        robot_radius_m=0.22,
        local_view=view,
        blocked_path=baseline.path,
        blocked_path_pose=start,
        paint_corridor=False,
    )
    assert sealed.feasible, sealed.error_msg
    assert paths_meaningfully_differ(baseline.path, sealed.path)


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
    assert abs(progress["bearing_error_rad"]) > math.radians(55.0)
    assert cmd.vx == 0.0
    assert abs(cmd.vtheta) > 0.05

    # Path-blocked recovery must still run DWA despite the large bearing —
    # otherwise we spin in place forever while replans fail.
    cmd2, progress2 = compute_path_command(
        pose,
        path,
        cfg=FollowerConfig(),
        local_view=view,
        local_planner=LocalPlannerConfig(enabled=True, activate_cost_threshold=1),
        robot_radius_m=0.08,
        force_local_planner=True,
    )
    assert progress2.get("local_planner") is True
    # Even under force_local, do not creep forward while ~129° off heading.
    assert cmd2.vx <= 0.02
    assert abs(cmd2.vtheta) > 0.05


def test_costmap_hard_stop_blocks_translate_into_inscribed():
    """Lidar nose clear but costmap inscribed ahead — must not translate."""
    from src.nav.simple_motion import ObstacleConfig
    from src.nav_builtin.costmap import LETHAL
    from src.nav_builtin.local_costmap import LocalCostmapView
    from src.nav_builtin.types import OccupancyGrid

    costs = np.zeros((40, 40), dtype=np.uint8)
    # Inscribed wall directly ahead of pose at (1.0, 1.0) facing +x.
    costs[18:22, 28:32] = 253
    occ = OccupancyGrid(
        grid=np.zeros((40, 40), dtype=np.int16),
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )
    view = LocalCostmapView(costs=costs, occ=occ, origin_x=0.0, origin_y=0.0)
    pose = Pose2D(1.0, 1.0, 0.0)
    # Wide-open lidar — nose_clear would be true without the costmap check.
    scan = conv.LaserScan2D(
        np.full(36, 5.0),
        angle_min=-math.pi,
        angle_increment=2 * math.pi / 36,
        range_min=0.05,
        range_max=10.0,
    )
    path = Path2D(points=((1.0, 1.0), (2.5, 1.0)), goal_theta=0.0)
    cfg = FollowerConfig(
        obstacle=ObstacleConfig(
            enabled=True, stop_distance_m=0.45, slow_distance_m=0.9
        )
    )
    cmd, progress = compute_path_command(
        pose,
        path,
        cfg=cfg,
        scan=scan,
        local_view=view,
        local_planner=None,
        robot_radius_m=0.08,
    )
    assert cmd.vx <= 1e-9
    assert progress["obstacle"] == "avoid"


def test_hard_stop_blocks_local_planner_into_stop_bubble():
    """DWA must not translate forward through the reactive stop distance."""
    from src.nav.simple_motion import ObstacleConfig

    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=3.0,
            height_m=3.0,
            resolution=0.05,
            inflation_radius_m=0.05,
            robot_radius_m=0.08,
            use_global_static=False,
        )
    )
    pose = Pose2D(1.0, 1.5, 0.0)
    # Costmap: open path so DWA wants forward. Live scan: wall in stop bubble.
    open_scan = conv.LaserScan2D(
        np.full(72, 3.0),
        angle_min=-math.pi,
        angle_increment=2 * math.pi / 72,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(pose, open_scan)
    n = 72
    close = np.full(n, 3.0)
    close[n // 2] = 0.35
    live = conv.LaserScan2D(
        close,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    path = Path2D(points=((1.0, 1.5), (2.5, 1.5)), goal_theta=0.0)
    cfg = FollowerConfig(
        obstacle=ObstacleConfig(
            enabled=True, stop_distance_m=0.45, slow_distance_m=0.9
        )
    )
    cmd, progress = compute_path_command(
        pose,
        path,
        cfg=cfg,
        scan=live,
        local_view=view,
        local_planner=LocalPlannerConfig(
            enabled=True, activate_cost_threshold=250
        ),
        robot_radius_m=0.08,
    )
    assert cmd.vx <= 1e-9
    if progress["obstacle"] == "avoid":
        assert cmd.vx == 0.0
    else:
        # DWA may reverse out; it must not drive forward into the bubble.
        assert progress["obstacle"] == "local_planner"
        assert cmd.vx <= 0.0


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
    assert cmd.vx <= 0.35


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


def test_local_planner_continuity_prefers_previous_turn_sign():
    """Noisy soft costs should not flip vθ sign every tick when prev_cmd is set."""
    lc = LocalCostmap(
        LocalCostmapConfig(
            width_m=3.0,
            height_m=3.0,
            resolution=0.05,
            inflation_radius_m=0.2,
            robot_radius_m=0.1,
            use_global_static=False,
            scan_inflation_radius_m=0.1,
        )
    )
    pose = Pose2D(1.0, 1.5, 0.0)
    # Soft side obstacle: enough to wake DWA, not lethal in footprint.
    n = 48
    ranges = np.full(n, 3.0)
    ranges[n // 4] = 0.55  # ~+45°
    scan = conv.LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )
    view = lc.update(pose, scan)
    path = Path2D(points=((1.0, 1.5), (2.5, 1.5)), goal_theta=0.0)
    cfg = LocalPlannerConfig(
        enabled=True,
        activate_cost_threshold=1,
        continuity_weight=0.8,
        sim_time_s=1.0,
    )
    prev = DriveCommand(0.25, 0.0, 0.6, False)
    cmd = compute_local_command(
        pose,
        path,
        view,
        cfg=cfg,
        max_vel_x=0.4,
        max_vel_theta=1.0,
        robot_radius_m=0.1,
        local_planner_active=True,
        prev_cmd=prev,
    )
    assert cmd is not None
    assert cmd.vtheta >= 0.0 or abs(cmd.vtheta) < 0.05

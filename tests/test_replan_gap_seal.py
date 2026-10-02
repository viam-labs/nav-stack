"""Replan scan-paint seals a fit doorway that the static plan uses.

Also documents the runtime policy: with a clear nose, the supervisor keeps the
short path and peels with DWA for a grace window before escalating to replan
(so scan-paint room loops are not the first response).
"""
from __future__ import annotations

import math

import numpy as np

from src.geom.conversions import LaserScan2D, Pose2D
from src.nav_builtin.costmap import OccupancyGrid
from src.nav_builtin.planner import plan_path
from src.nav_builtin.supervisor import NavSupervisor


RES = 0.05
INSCRIBED = 0.59 / 2.0
INFLATION = INSCRIBED + 0.05
PREFER = 0.50
GAP = 1.08
DOOR_X = 10.0
WALL_Y = 8.0
WALL_END = 32.0


def _gap_map() -> OccupancyGrid:
    h, w = int(16.0 / RES), int(40.0 / RES)
    grid = np.zeros((h, w), dtype=np.int16)
    grid[:3, :] = 100
    grid[-3:, :] = 100
    grid[:, :3] = 100
    grid[:, -3:] = 100
    wr = int(WALL_Y / RES)
    grid[wr : wr + 3, : int(WALL_END / RES)] = 100
    half = int((GAP / 2.0) / RES)
    dc = int(DOOR_X / RES)
    grid[wr : wr + 3, dc - half : dc + half] = 0
    return OccupancyGrid(grid=grid, resolution=RES, origin_x=0.0, origin_y=0.0)


def _frame_scan(pose: Pose2D, *, inward_m: float) -> LaserScan2D:
    """Lidar that sees both door frames, optionally shifted into the opening."""
    n = 360
    ranges = np.full(n, math.inf)
    for edge in (-1.0, 1.0):
        ex = DOOR_X + edge * (GAP / 2.0 - inward_m)
        ey = WALL_Y
        dx, dy = ex - pose.x, ey - pose.y
        bearing = math.atan2(dy, dx) - pose.theta
        idx = int((bearing + math.pi) / (2 * math.pi / n)) % n
        ranges[idx] = math.hypot(dx, dy)
    return LaserScan2D(
        ranges,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / n,
        range_min=0.05,
        range_max=10.0,
    )


def test_local_block_policy_prefers_dwa_when_nose_clear():
    """Doorway pinch: clear nose → DWA; escalate only if peel makes no progress."""
    wait = 2.0
    assert (
        NavSupervisor._local_block_action(
            nose_clear=True,
            blocked_for_s=0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=True,
        )
        == "keep_dwa"
    )
    assert (
        NavSupervisor._local_block_action(
            nose_clear=False,
            blocked_for_s=0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=True,
        )
        == "wait"
    )
    # Clear nose still peels after grace — no immediate stop-replan stutter.
    assert (
        NavSupervisor._local_block_action(
            nose_clear=True,
            blocked_for_s=wait + 0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=True,
            peel_stuck_s=1.0,
        )
        == "keep_dwa"
    )
    # Peel stuck long enough → escalate once.
    assert (
        NavSupervisor._local_block_action(
            nose_clear=True,
            blocked_for_s=wait + 0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=True,
            peel_stuck_s=6.0,
        )
        == "replan"
    )
    assert (
        NavSupervisor._local_block_action(
            nose_clear=False,
            blocked_for_s=wait + 0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=True,
        )
        == "replan"
    )
    # Blocked nose, cooldown not ready: keep waiting.
    assert (
        NavSupervisor._local_block_action(
            nose_clear=False,
            blocked_for_s=wait + 0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=False,
        )
        == "wait"
    )
    # Peel stuck but cooldown not ready: keep peeling.
    assert (
        NavSupervisor._local_block_action(
            nose_clear=True,
            blocked_for_s=wait + 0.1,
            wait_before_replan_s=wait,
            replan_cooldown_ready=False,
            peel_stuck_s=10.0,
        )
        == "keep_dwa"
    )


def test_blocked_nose_unstick_after_a_failed_replan():
    """First wait holds. After a failed replan, reverse, else turn."""
    hold = dict(failed_replans=0, nose_clear=False, rear_open=True, spin_clear=True)
    assert NavSupervisor._blocked_nose_unstick(**hold) == "hold"
    assert (
        NavSupervisor._blocked_nose_unstick(
            failed_replans=1, nose_clear=False, rear_open=True, spin_clear=False
        )
        == "reverse"
    )
    assert (
        NavSupervisor._blocked_nose_unstick(
            failed_replans=1, nose_clear=False, rear_open=False, spin_clear=True
        )
        == "turn"
    )
    assert (
        NavSupervisor._blocked_nose_unstick(
            failed_replans=2, nose_clear=False, rear_open=False, spin_clear=False
        )
        == "hold"
    )
    # A clear nose stays on the peel path.
    assert (
        NavSupervisor._blocked_nose_unstick(
            failed_replans=2, nose_clear=True, rear_open=True, spin_clear=True
        )
        == "hold"
    )


def test_bumper_spin_backs_up_even_when_the_path_is_free():
    """Inch off an obstacle, room behind, path cost still low: reverse."""
    assert NavSupervisor._bumper_spin_reverse(
        nose_clear=False,
        spin_blocked=True,
        cmd_vx=0.0,
        cmd_vtheta=0.0,
        rear_open=True,
    )
    assert not NavSupervisor._bumper_spin_reverse(
        nose_clear=False,
        spin_blocked=True,
        cmd_vx=0.0,
        cmd_vtheta=0.0,
        rear_open=False,
    )
    assert not NavSupervisor._bumper_spin_reverse(
        nose_clear=True,
        spin_blocked=True,
        cmd_vx=0.0,
        cmd_vtheta=0.0,
        rear_open=True,
    )


def test_path_locally_blocked_ignores_inscribed_when_corridor_open():
    """Clear nose + inscribed path cost must not force DWA (doorway pecking)."""
    # Soft / inscribed on path, clear nose → not blocked (any clearance).
    assert not NavSupervisor._path_locally_blocked(
        path_ahead_cost=253,
        pose_cost=0,
        activate_cost=200,
        nose_clear=True,
        forward_clearance_m=0.45,
    )
    assert not NavSupervisor._path_locally_blocked(
        path_ahead_cost=253,
        pose_cost=0,
        activate_cost=200,
        nose_clear=True,
        forward_clearance_m=1.52,
    )
    # Blocked nose + soft path cost → blocked (wait / replan path).
    assert NavSupervisor._path_locally_blocked(
        path_ahead_cost=253,
        pose_cost=0,
        activate_cost=200,
        nose_clear=False,
        forward_clearance_m=0.2,
    )
    # True lethal always blocks.
    assert NavSupervisor._path_locally_blocked(
        path_ahead_cost=254,
        pose_cost=0,
        activate_cost=200,
        nose_clear=True,
        forward_clearance_m=2.0,
    )
    # Body already in hard cost blocks.
    assert NavSupervisor._path_locally_blocked(
        path_ahead_cost=0,
        pose_cost=253,
        activate_cost=200,
        nose_clear=True,
        forward_clearance_m=2.0,
    )


def test_static_plan_uses_doorway_but_scan_replan_takes_long_detour():
    """Mirrors live: plan_to_point ~4 m through gap; navigate replan ~47 m.

    Initial plan is scan-free. Every ``_try_replan`` paints live hits with a
    0.12 m disc then full footprint inflation — enough to seal a 1.08 m gap
    when the frames read even slightly inward — and the 1.8× length cap does
    not apply to that first scan+local attempt. Runtime policy now delays that
    replan while the nose is clear so DWA can keep the short route.
    """
    occ = _gap_map()
    start = Pose2D(DOOR_X, WALL_Y - 1.5, math.pi / 2)
    goal = Pose2D(DOOR_X, WALL_Y + 1.5, math.pi / 2)
    map_data = {
        "grid": occ.grid,
        "resolution": occ.resolution,
        "origin_x": occ.origin_x,
        "origin_y": occ.origin_y,
    }

    static = plan_path(
        map_data,
        start,
        goal,
        inflation_radius_m=INFLATION,
        robot_radius_m=INSCRIBED,
        clearance_preference_m=PREFER,
        algorithm="lazy_theta_star",
    )
    assert static.feasible
    static_pts = static.path.points
    static_len = sum(
        math.hypot(static_pts[i + 1][0] - static_pts[i][0], static_pts[i + 1][1] - static_pts[i][1])
        for i in range(len(static_pts) - 1)
    )
    assert static_len < 5.0
    assert max(p[0] for p in static_pts) < WALL_END - 1.0

    # Frames perceived 10 cm inside the opening (localization / beam thickness).
    scan = _frame_scan(start, inward_m=0.10)
    painted = plan_path(
        map_data,
        start,
        goal,
        inflation_radius_m=INFLATION,
        robot_radius_m=INSCRIBED,
        clearance_preference_m=PREFER,
        algorithm="lazy_theta_star",
        scan=scan,
        scan_pose=start,
        dynamic_obstacle_radius_m=0.12,
    )
    assert painted.feasible
    pts = painted.path.points
    painted_len = sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )
    assert painted_len > 20.0, f"expected long detour, got {painted_len:.1f} m"
    assert max(p[0] for p in pts) > WALL_END - 1.0

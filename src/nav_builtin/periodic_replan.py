"""Periodic / pose-jump / oscillation replan decision for one tick."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from ..nav.simple_motion import distance_m
from .planner import path_blocked, path_blocked_on_costmap
from .replan_controller import apply_periodic_replan
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .supervisor import NavSupervisor


@dataclass
class PeriodicReplanResult:
    path: Path2D
    pose: Pose2D
    abort: bool = False


def run_periodic_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    now: float,
    pose: Pose2D,
    goal: Pose2D,
    path: Path2D,
    scan,
    local_view,
    cmd,
    progress: Mapping[str, Any],
    local_blocked: bool,
    nose_clear: bool,
    corridor_yaw,
) -> PeriodicReplanResult:
    """Track vx oscillation, check static path, optionally replan or abort."""
    if abs(cmd.vx) > 0.05:
        sign = 1 if cmd.vx > 0 else -1
        ep.vx_sign_history.append((now, sign))
        ep.vx_sign_history = [
            (t, s) for t, s in ep.vx_sign_history if now - t <= 3.0
        ]
    oscillating = False
    if len(ep.vx_sign_history) >= 4:
        flips = sum(
            1
            for i in range(1, len(ep.vx_sign_history))
            if ep.vx_sign_history[i][1] != ep.vx_sign_history[i - 1][1]
        )
        oscillating = flips >= 3

    replan_due = now - ep.last_replan >= nav._replan_period  # noqa: SLF001
    static_blocked = False
    pose_jumped = ep.pending_loc_replan
    if ep.last_tick_pose is not None:
        pose_jumped = pose_jumped or (
            distance_m(pose, ep.last_tick_pose) >= ep.pose_jump_replan_m
        )
    ep.last_tick_pose = pose
    if replan_due or pose_jumped:
        # Prefer the already-inflated global cache (background thread).
        # Falling back to path_blocked() rebuilds the full map costmap
        # (~0.4 s on a large grid) on the control thread.
        with nav._global_cache_lock:  # noqa: SLF001
            cached_occ = nav._global_occ_cache  # noqa: SLF001
            cached_costs = nav._global_costs_cache  # noqa: SLF001
        if cached_occ is not None and cached_costs is not None:
            static_blocked = path_blocked_on_costmap(
                cached_occ,
                cached_costs,
                path,
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
                from_pose=pose,
                ahead_m=ep.path_block_horizon_m,
            )
        else:
            map_data = nav._world.get_map()  # noqa: SLF001
            masks = nav._zone_masks(map_data)  # noqa: SLF001
            keepout = None if masks is None else masks.keepout
            static_blocked = map_data is not None and path_blocked(
                map_data,
                path,
                inflation_radius_m=nav._inflation,  # noqa: SLF001
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
                from_pose=pose,
                ahead_m=ep.path_block_horizon_m,
                mapping=nav._slam_is_mapping(),  # noqa: SLF001
                keepout_mask=keepout,
            )
        # Large localization corrections used to force a replan even
        # when the polyline was still free — that stop+replan looped
        # forever under the detour ban (short-flip rejected every
        # shorter candidate, ep.pending_loc_replan never cleared).
        # Soft loc nudge with a clear path: keep following.
        if pose_jumped and not static_blocked:
            ep.pending_loc_replan = False

    backup_exhausted = (
        ep.backup_attempts >= nav._backup_max_attempts and local_blocked  # noqa: SLF001
    )
    # Sign-flip rock (narrow crawl ↔ reverse) with a *blocked* nose
    # still forces a replan. With a clear nose, the same forward/
    # back jig is normal DWA peel in a tight corridor — stop-replan
    # there was the "crazy jig then freeze" (live: oscillating=True,
    # static_blocked=False, nose_clear). peel_stuck covers escalate.
    osc_replan = bool(oscillating and local_blocked and not nose_clear)
    spin_rock = bool(
        progress.get("spin_blocked") and oscillating and not nose_clear
    )
    should_replan = (replan_due or pose_jumped) and (
        static_blocked
        or osc_replan
        or spin_rock
        or backup_exhausted
    )
    # Periodic check said "still clear" — still advance the timer.
    # Otherwise replan_due stays true and the expensive static check
    # (or a full inflate fallback) runs on every subsequent tick.
    if replan_due and not should_replan:
        ep.last_replan = now
    if should_replan and nav._defer_periodic_replan(  # noqa: SLF001
        static_blocked=static_blocked,
        nose_clear=nose_clear,
        corridor_open=corridor_yaw is not None,
        pose_jumped=pose_jumped,
        osc_replan=osc_replan,
        spin_rock=spin_rock,
        backup_exhausted=backup_exhausted,
    ):
        # Keep the controller running. The search this would
        # start has been returning "no feasible path" and the
        # stop is the whole stutter.
        ep.last_replan = now
        nav._last_replan_trigger = (  # noqa: SLF001
            "deferred static_blocked; scan can continue"
        )
        return PeriodicReplanResult(path=path, pose=pose)
    if not should_replan:
        return PeriodicReplanResult(path=path, pose=pose)

    # On static/pose-jump recovery, accept any feasible plan —
    # require_different would reject a valid near-identical route
    # and count it as "replan failed".
    outcome = apply_periodic_replan(
        nav,
        ep,
        goal=goal,
        pose=pose,
        path=path,
        scan=scan,
        local_view=local_view,
        local_blocked=local_blocked,
        static_blocked=static_blocked,
        pose_jumped=pose_jumped,
        oscillating=oscillating,
        now=now,
    )
    path, pose = outcome.path, outcome.pose
    if outcome.abort_msg == "static_replan_fail":
        clearance = progress.get("forward_clearance_m")
        has_room = clearance is None or float(clearance) >= 0.35
        obstacle = str(progress.get("obstacle") or "")
        # "wait"/"slow" still mean lidar sees space; only hard
        # stop / proximity hold / missing scan should force fail.
        lidar_open = (
            obstacle not in ("stop", "hold", "no_scan") and has_room
        )
        # Keep following while the robot can still see open space;
        # a mid-route localization jump often fails a few replans
        # before the map/pose settle.
        if (
            ep.failed_static_replan >= ep.static_replan_fail_limit
            and not lidar_open
        ):
            nav._world.stop()  # noqa: SLF001
            nav._set_status(  # noqa: SLF001
                state="failed",
                active=False,
                error_msg="replan failed (path blocked)",
            )
            return PeriodicReplanResult(path=path, pose=pose, abort=True)
    return PeriodicReplanResult(path=path, pose=pose)

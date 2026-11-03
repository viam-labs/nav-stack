"""Periodic / pose-jump / oscillation replan decision for one tick."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..motion.simple_motion import distance_m
from ..planning.planner import path_blocked, path_blocked_on_costmap
from .replan_controller import apply_periodic_replan

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


@dataclass
class PeriodicReplanResult:
    abort: bool = False


def run_periodic_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> PeriodicReplanResult:
    """Track vx oscillation, check static path, optionally replan or abort."""
    cmd = tick.cmd
    progress = tick.progress
    if abs(cmd.vx) > 0.05:
        sign = 1 if cmd.vx > 0 else -1
        ep.vx_sign_history.append((tick.now, sign))
        ep.vx_sign_history = [
            (t, s) for t, s in ep.vx_sign_history if tick.now - t <= 3.0
        ]
    oscillating = False
    if len(ep.vx_sign_history) >= 4:
        flips = sum(
            1
            for i in range(1, len(ep.vx_sign_history))
            if ep.vx_sign_history[i][1] != ep.vx_sign_history[i - 1][1]
        )
        oscillating = flips >= 3

    replan_due = tick.now - ep.last_replan >= nav._replan_period  # noqa: SLF001
    static_blocked = False
    pose_jumped = ep.pending_loc_replan
    if ep.last_tick_pose is not None:
        pose_jumped = pose_jumped or (
            distance_m(tick.pose, ep.last_tick_pose) >= ep.pose_jump_replan_m
        )
    ep.last_tick_pose = tick.pose
    if replan_due or pose_jumped:
        with nav._global_cache_lock:  # noqa: SLF001
            cached_occ = nav._global_occ_cache  # noqa: SLF001
            cached_costs = nav._global_costs_cache  # noqa: SLF001
        if cached_occ is not None and cached_costs is not None:
            static_blocked = path_blocked_on_costmap(
                cached_occ,
                cached_costs,
                tick.path,
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
                from_pose=tick.pose,
                ahead_m=ep.path_block_horizon_m,
            )
        else:
            map_data = nav._world.get_map()  # noqa: SLF001
            masks = nav._zone_masks(map_data)  # noqa: SLF001
            keepout = None if masks is None else masks.keepout
            static_blocked = map_data is not None and path_blocked(
                map_data,
                tick.path,
                inflation_radius_m=nav._inflation,  # noqa: SLF001
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
                from_pose=tick.pose,
                ahead_m=ep.path_block_horizon_m,
                mapping=nav._slam_is_mapping(),  # noqa: SLF001
                keepout_mask=keepout,
            )
        if pose_jumped and not static_blocked:
            ep.pending_loc_replan = False

    backup_exhausted = (
        ep.backup_attempts >= nav._backup_max_attempts and tick.local_blocked  # noqa: SLF001
    )
    osc_replan = bool(
        oscillating and tick.local_blocked and not tick.nose_clear
    )
    spin_rock = bool(
        progress.get("spin_blocked") and oscillating and not tick.nose_clear
    )
    should_replan = (replan_due or pose_jumped) and (
        static_blocked
        or osc_replan
        or spin_rock
        or backup_exhausted
    )
    if replan_due and not should_replan:
        ep.last_replan = tick.now
    if should_replan and nav._defer_periodic_replan(  # noqa: SLF001
        static_blocked=static_blocked,
        nose_clear=tick.nose_clear,
        corridor_open=tick.corridor_yaw is not None,
        pose_jumped=pose_jumped,
        osc_replan=osc_replan,
        spin_rock=spin_rock,
        backup_exhausted=backup_exhausted,
    ):
        ep.last_replan = tick.now
        nav._last_replan_trigger = (  # noqa: SLF001
            "deferred static_blocked; scan can continue"
        )
        return PeriodicReplanResult()
    if not should_replan:
        return PeriodicReplanResult()

    outcome = apply_periodic_replan(
        nav,
        ep,
        tick,
        static_blocked=static_blocked,
        pose_jumped=pose_jumped,
        oscillating=oscillating,
    )
    if outcome.abort_msg == "static_replan_fail":
        clearance = progress.get("forward_clearance_m")
        has_room = clearance is None or float(clearance) >= 0.35
        obstacle = str(progress.get("obstacle") or "")
        lidar_open = (
            obstacle not in ("stop", "hold", "no_scan") and has_room
        )
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
            return PeriodicReplanResult(abort=True)
    return PeriodicReplanResult()

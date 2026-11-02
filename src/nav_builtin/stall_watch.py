"""Stall detection + recovery replan for one ``run_goal`` tick."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..geom import conversions as conv
from ..nav.simple_motion import distance_m
from .replan_controller import apply_stall_replan

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


@dataclass
class StallResult:
    abort: bool = False


def run_stall_watch(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> StallResult:
    """Mark progress or stall-replan / fail the goal."""
    pose = tick.pose
    cmd = tick.cmd
    progress = tick.progress
    dist_goal = distance_m(pose, tick.goal)
    near_goal_stall = (
        dist_goal <= nav._follower.motion.xy_tolerance_m * 2.0  # noqa: SLF001
        or nav._xy_at_nav_goal(pose, tick.goal, tick.path)  # noqa: SLF001
    )
    translating_floor = 0.03 if near_goal_stall else 0.05
    stall_scale = 2.0 if near_goal_stall else 1.0
    if tick.local_blocked:
        stall_scale = max(stall_scale, 4.0)
    elif (
        tick.nose_clear
        and tick.path_ahead_cost >= nav._local_planner_activate_cost  # noqa: SLF001
    ):
        from .costmap import LETHAL

        if tick.path_ahead_cost < int(LETHAL):
            stall_scale = max(stall_scale, 4.0)
    stall_limit_s = nav._follower.motion.stall_timeout_s * stall_scale  # noqa: SLF001
    bearing_err = abs(float(progress.get("bearing_error_rad", 0.0)))
    spinning = (
        abs(float(cmd.vtheta)) >= 0.08
        and abs(float(cmd.vx)) < translating_floor
    )

    def _mark_progress() -> None:
        ep.last_progress_pose = pose
        ep.last_progress_at = tick.now
        ep.last_progress_dist = dist_goal
        ep.last_progress_bearing = bearing_err

    def _stall_replan_or_fail(error_msg: str) -> bool:
        _, should_return = apply_stall_replan(
            nav,
            ep,
            tick,
            dist_goal=dist_goal,
            bearing_err=bearing_err,
            error_msg=error_msg,
        )
        return should_return

    if ep.last_progress_pose is None:
        _mark_progress()
        return StallResult()

    moved = distance_m(pose, ep.last_progress_pose)
    closing = dist_goal < ep.last_progress_dist - 0.01
    translating = abs(float(cmd.vx)) >= translating_floor
    bearing_improved = bearing_err < ep.last_progress_bearing - math.radians(3.0)
    turned = abs(
        conv.normalize_angle(pose.theta - ep.last_progress_pose.theta)
    )
    if closing or (
        moved >= nav._follower.motion.stall_progress_m  # noqa: SLF001
        and (
            translating
            or moved >= nav._follower.motion.stall_progress_m * 2  # noqa: SLF001
        )
    ):
        _mark_progress()
    elif translating:
        if turned >= nav._follower.motion.stall_progress_rad:  # noqa: SLF001
            _mark_progress()
        elif tick.now - ep.last_progress_at >= stall_limit_s:
            if _stall_replan_or_fail("navigation stalled"):
                return StallResult(abort=True)
    elif spinning and (
        turned >= nav._follower.motion.stall_progress_rad  # noqa: SLF001
        or bearing_improved
    ):
        _mark_progress()
    elif tick.now - ep.last_progress_at >= stall_limit_s:
        if _stall_replan_or_fail("navigation stalled (no forward progress)"):
            return StallResult(abort=True)
    return StallResult()

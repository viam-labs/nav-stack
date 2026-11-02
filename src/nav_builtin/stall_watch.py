"""Stall detection + recovery replan for one ``run_goal`` tick."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from ..geom import conversions as conv
from ..nav.simple_motion import distance_m
from .replan_controller import apply_stall_replan
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .supervisor import NavSupervisor


@dataclass
class StallResult:
    path: Path2D
    abort: bool = False


def run_stall_watch(
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
    path_ahead_cost: int,
) -> StallResult:
    """Mark progress or stall-replan / fail the goal."""
    # Pure spin with no bearing improvement must not reset the timer
    # forever (stuck local-planner / RIP loops need to replan). But
    # intentional align spins that shrink |bearing| are real progress.
    # While the route is locally blocked, give recovery room —
    # stop/replan/forced-via look like "no progress" and were
    # aborting with cmd_vx still showing a corridor charge.
    dist_goal = distance_m(pose, goal)
    near_goal_stall = (
        dist_goal <= nav._follower.motion.xy_tolerance_m * 2.0  # noqa: SLF001
        or nav._xy_at_nav_goal(pose, goal, path)  # noqa: SLF001
    )
    # Near goal, tiny crawls are real progress — don't require 0.05 m/s.
    translating_floor = 0.03 if near_goal_stall else 0.05
    stall_scale = 2.0 if near_goal_stall else 1.0
    if local_blocked:
        stall_scale = max(stall_scale, 4.0)
    # Clear-nose C-space pinch (pathc=253, local_blocked=false after
    # rc22): slow pursuit must get the same grace or stall aborts
    # the goal while crawling a doorway.
    elif nose_clear and path_ahead_cost >= nav._local_planner_activate_cost:  # noqa: SLF001
        from .costmap import LETHAL

        if path_ahead_cost < int(LETHAL):
            stall_scale = max(stall_scale, 4.0)
    stall_limit_s = nav._follower.motion.stall_timeout_s * stall_scale  # noqa: SLF001
    bearing_err = abs(float(progress.get("bearing_error_rad", 0.0)))
    spinning = (
        abs(float(cmd.vtheta)) >= 0.08
        and abs(float(cmd.vx)) < translating_floor
    )

    def _mark_progress() -> None:
        ep.last_progress_pose = pose
        ep.last_progress_at = now
        ep.last_progress_dist = dist_goal
        ep.last_progress_bearing = bearing_err

    def _stall_replan_or_fail(error_msg: str) -> tuple[Path2D, bool]:
        outcome, should_return = apply_stall_replan(
            nav,
            ep,
            goal=goal,
            pose=pose,
            path=path,
            scan=scan,
            local_view=local_view,
            local_blocked=local_blocked,
            nose_clear=nose_clear,
            path_ahead_cost=path_ahead_cost,
            dist_goal=dist_goal,
            bearing_err=bearing_err,
            error_msg=error_msg,
            now=now,
            progress_spin_blocked=bool(progress.get("spin_blocked")),
            cmd_vx=float(cmd.vx),
        )
        return outcome.path, should_return

    cur_path = path
    if ep.last_progress_pose is None:
        _mark_progress()
        return StallResult(path=cur_path)

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
        elif now - ep.last_progress_at >= stall_limit_s:
            cur_path, should_return = _stall_replan_or_fail("navigation stalled")
            if should_return:
                return StallResult(path=cur_path, abort=True)
    elif spinning and (
        turned >= nav._follower.motion.stall_progress_rad  # noqa: SLF001
        or bearing_improved
    ):
        _mark_progress()
    elif now - ep.last_progress_at >= stall_limit_s:
        cur_path, should_return = _stall_replan_or_fail(
            "navigation stalled (no forward progress)"
        )
        if should_return:
            return StallResult(path=cur_path, abort=True)
    return StallResult(path=cur_path)

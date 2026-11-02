"""Wait vs replan vs DWA peel when the local route is blocked.

Owns corridor yaw commit and ``waiting_for_clear``. Does not call the planner —
the supervisor (or ``ReplanController``) acts on ``action == "replan"``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..geom import conversions as conv
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor

# Match supervisor: a few degrees is closed while a moving pursuit can finish.
_CORRIDOR_ALIGN_RAD = math.radians(3.0)


@dataclass
class BlockDecision:
    action: str  # "wait" | "replan" | "keep_dwa" | ""
    waiting_for_clear: bool
    corridor_yaw: Optional[float]


def run_block_policy(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
    *,
    goal: Pose2D,
    path: Path2D,
    loc_hold_motion: bool,
) -> BlockDecision:
    """Decide wait / replan / DWA; update episode timers and corridor heading."""
    now = tick.now
    pose = tick.pose
    scan = tick.scan
    nose_clear = tick.nose_clear
    local_blocked = tick.local_blocked
    path_ahead_cost = tick.path_ahead_cost

    wait_before_replan_s = max(
        nav._recovery_wait_duration_s,  # noqa: SLF001
        nav._replan_local_blocked_time_s,  # noqa: SLF001
    )
    waiting_for_clear = False
    corridor_yaw: Optional[float] = None
    action = ""

    if not local_blocked:
        ep.on_unblocked()
        nav._stuck_pose_refine_used = False  # noqa: SLF001
        return BlockDecision(
            action="", waiting_for_clear=False, corridor_yaw=None
        )

    if ep.local_blocked_since is None:
        ep.local_blocked_since = now
    blocked_for = now - ep.local_blocked_since
    cooldown_ready = now - ep.last_local_replan_at >= min(
        8.0,
        nav._replan_local_min_period_s  # noqa: SLF001
        * (2.0 ** min(ep.blocked_fail_count, 4)),
    )
    peel_stuck_s = (
        max(0.0, now - ep.last_progress_at) if nose_clear else 0.0
    )
    action = nav._local_block_action(  # noqa: SLF001
        nose_clear=nose_clear,
        blocked_for_s=blocked_for,
        wait_before_replan_s=wait_before_replan_s,
        replan_cooldown_ready=cooldown_ready,
        peel_stuck_s=peel_stuck_s,
    )
    # Path centerline free but "wait for nose" — prefer DWA peel.
    if (
        action == "wait"
        and path_ahead_cost < nav._local_planner_activate_cost  # noqa: SLF001
    ):
        action = "keep_dwa"
    if (
        action == "replan"
        and path_ahead_cost < nav._local_planner_activate_cost  # noqa: SLF001
        and nose_clear
    ):
        action = "keep_dwa"
    # Corridor: keep turning when a heading is already committed.
    if (
        (not nose_clear or ep.corridor_heading is not None)
        and not loc_hold_motion
        and scan is not None
        and nav._guard is not None  # noqa: SLF001
        and blocked_for >= wait_before_replan_s
    ):
        corridor_yaw = nav._corridor_escape_yaw(  # noqa: SLF001
            pose, scan, goal, path, commit_heading=ep.corridor_heading
        )
        ep.corridor_heading = (
            conv.normalize_angle(pose.theta + corridor_yaw)
            if corridor_yaw is not None
            else None
        )
    if corridor_yaw is not None:
        if abs(corridor_yaw) < _CORRIDOR_ALIGN_RAD and cooldown_ready:
            action = "replan"
        elif action == "replan":
            action = "wait"
    if action == "wait":
        waiting_for_clear = True

    return BlockDecision(
        action=action,
        waiting_for_clear=waiting_for_clear,
        corridor_yaw=corridor_yaw,
    )

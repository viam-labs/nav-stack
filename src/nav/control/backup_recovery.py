"""DWA spin-stuck backup state machine for ``run_goal``.

Owns backup active/start/cost/attempts/cooldown. On completion, asks
``apply_backup_done_replan`` for a fresh path. Mutates ``TickContext``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..motion.simple_motion import DriveCommand, distance_m, rear_clearance_m
from ..planning.local_costmap import footprint_max_cost, reverse_backup_feasible
from .replan_controller import apply_backup_done_replan

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


def run_backup_recovery(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> None:
    """Advance or start a reverse backup when DWA is spin-stuck."""
    progress = dict(tick.progress)
    cmd = tick.cmd
    allow_backup = (
        nav._backup_enabled  # noqa: SLF001
        and tick.scan is not None
        and tick.local_view is not None
        and progress.get("local_planner")
        and abs(cmd.vx) < 0.05
        and abs(cmd.vtheta) > 0.1
        and ep.backup_attempts < nav._backup_max_attempts  # noqa: SLF001
        and tick.now >= ep.backup_cooldown_until
        and not tick.waiting_for_clear
        and (not tick.local_blocked or ep.failed_replan_while_blocked >= 2)
    )

    if ep.backup_active and ep.backup_start is not None:
        backed_m = distance_m(tick.pose, ep.backup_start)
        pose_cost = (
            footprint_max_cost(
                tick.local_view,
                tick.pose.x,
                tick.pose.y,
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
            )
            if tick.local_view is not None
            else 0
        )
        rear = rear_clearance_m(tick.scan) if tick.scan is not None else math.inf
        worse = tick.local_view is not None and pose_cost > ep.backup_start_cost + 5
        if (
            backed_m >= nav._backup_dist_m  # noqa: SLF001
            or rear < nav._backup_rear_clear_m * 0.85  # noqa: SLF001
            or worse
        ):
            ep.backup_active = False
            ep.backup_start = None
            ep.spin_stuck_since = None
            ep.backup_cooldown_until = tick.now + nav._backup_cooldown_s  # noqa: SLF001
            apply_backup_done_replan(nav, ep, tick)
            tick.cmd = DriveCommand(0.0, 0.0, 0.0, False)
        else:
            tick.cmd = DriveCommand(
                -nav._backup_speed_mps,
                0.0,
                0.0,
                False,  # noqa: SLF001
            )
            tick.progress = {
                **progress,
                "local_planner": False,
                "obstacle": "backup",
                "cmd_vx_mps": tick.cmd.vx,
                "cmd_vtheta_rad_s": tick.cmd.vtheta,
            }
    elif allow_backup:
        if ep.spin_stuck_since is None:
            ep.spin_stuck_since = tick.now
        elif tick.now - ep.spin_stuck_since >= nav._backup_stuck_time_s:  # noqa: SLF001
            rear_ok = (
                rear_clearance_m(tick.scan) >= nav._backup_rear_clear_m  # noqa: SLF001
            )
            costmap_ok = reverse_backup_feasible(
                tick.local_view,
                tick.pose.x,
                tick.pose.y,
                tick.pose.theta,
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
                distance_m=nav._backup_dist_m,  # noqa: SLF001
            )
            if rear_ok and costmap_ok:
                ep.backup_active = True
                ep.backup_start = tick.pose
                ep.backup_start_cost = footprint_max_cost(
                    tick.local_view,
                    tick.pose.x,
                    tick.pose.y,
                    robot_radius_m=nav._robot_radius,  # noqa: SLF001
                )
                ep.backup_attempts += 1
                ep.spin_stuck_since = None
                tick.cmd = DriveCommand(
                    -nav._backup_speed_mps,
                    0.0,
                    0.0,
                    False,  # noqa: SLF001
                )
                tick.progress = {
                    **progress,
                    "local_planner": False,
                    "obstacle": "backup",
                    "cmd_vx_mps": tick.cmd.vx,
                    "cmd_vtheta_rad_s": tick.cmd.vtheta,
                }
            else:
                ep.spin_stuck_since = None
    else:
        ep.spin_stuck_since = None

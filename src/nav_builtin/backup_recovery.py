"""DWA spin-stuck backup state machine for ``run_goal``.

Owns backup active/start/cost/attempts/cooldown. On completion, asks
``apply_backup_done_replan`` for a fresh path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, TYPE_CHECKING

from ..nav.simple_motion import DriveCommand, distance_m, rear_clearance_m
from .local_costmap import footprint_max_cost, reverse_backup_feasible
from .replan_controller import apply_backup_done_replan
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .supervisor import NavSupervisor


@dataclass
class BackupResult:
    cmd: DriveCommand
    progress: Mapping[str, Any]
    path: Path2D
    pose: Pose2D


def run_backup_recovery(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    now: float,
    pose: Pose2D,
    goal: Pose2D,
    path: Path2D,
    scan,
    local_view,
    cmd: DriveCommand,
    progress: Mapping[str, Any],
    local_blocked: bool,
    waiting_for_clear: bool,
) -> BackupResult:
    """Advance or start a reverse backup when DWA is spin-stuck."""
    progress = dict(progress)
    allow_backup = (
        nav._backup_enabled  # noqa: SLF001
        and scan is not None
        and local_view is not None
        and progress.get("local_planner")
        and abs(cmd.vx) < 0.05
        and abs(cmd.vtheta) > 0.1
        and ep.backup_attempts < nav._backup_max_attempts  # noqa: SLF001
        and now >= ep.backup_cooldown_until
        and not waiting_for_clear
        and (
            not local_blocked
            or ep.failed_replan_while_blocked >= 2
        )
    )

    if ep.backup_active and ep.backup_start is not None:
        backed_m = distance_m(pose, ep.backup_start)
        pose_cost = (
            footprint_max_cost(
                local_view,
                pose.x,
                pose.y,
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
            )
            if local_view is not None
            else 0
        )
        rear = rear_clearance_m(scan) if scan is not None else math.inf
        worse = (
            local_view is not None
            and pose_cost > ep.backup_start_cost + 5
        )
        if (
            backed_m >= nav._backup_dist_m  # noqa: SLF001
            or rear < nav._backup_rear_clear_m * 0.85  # noqa: SLF001
            or worse
        ):
            ep.backup_active = False
            ep.backup_start = None
            ep.spin_stuck_since = None
            ep.backup_cooldown_until = now + nav._backup_cooldown_s  # noqa: SLF001
            outcome = apply_backup_done_replan(
                nav,
                ep,
                goal=goal,
                pose=pose,
                path=path,
                scan=scan,
                local_view=local_view,
                now=now,
            )
            path, pose = outcome.path, outcome.pose
            cmd = DriveCommand(0.0, 0.0, 0.0, False)
        else:
            cmd = DriveCommand(
                -nav._backup_speed_mps, 0.0, 0.0, False  # noqa: SLF001
            )
            progress = {
                **progress,
                "local_planner": False,
                "obstacle": "backup",
                "cmd_vx_mps": cmd.vx,
                "cmd_vtheta_rad_s": cmd.vtheta,
            }
    elif allow_backup:
        if ep.spin_stuck_since is None:
            ep.spin_stuck_since = now
        elif now - ep.spin_stuck_since >= nav._backup_stuck_time_s:  # noqa: SLF001
            rear_ok = rear_clearance_m(scan) >= nav._backup_rear_clear_m  # noqa: SLF001
            costmap_ok = reverse_backup_feasible(
                local_view,
                pose.x,
                pose.y,
                pose.theta,
                robot_radius_m=nav._robot_radius,  # noqa: SLF001
                distance_m=nav._backup_dist_m,  # noqa: SLF001
            )
            if rear_ok and costmap_ok:
                ep.backup_active = True
                ep.backup_start = pose
                ep.backup_start_cost = footprint_max_cost(
                    local_view,
                    pose.x,
                    pose.y,
                    robot_radius_m=nav._robot_radius,  # noqa: SLF001
                )
                ep.backup_attempts += 1
                ep.spin_stuck_since = None
                cmd = DriveCommand(
                    -nav._backup_speed_mps, 0.0, 0.0, False  # noqa: SLF001
                )
                progress = {
                    **progress,
                    "local_planner": False,
                    "obstacle": "backup",
                    "cmd_vx_mps": cmd.vx,
                    "cmd_vtheta_rad_s": cmd.vtheta,
                }
            else:
                ep.spin_stuck_since = None
    else:
        ep.spin_stuck_since = None

    return BackupResult(cmd=cmd, progress=progress, path=path, pose=pose)

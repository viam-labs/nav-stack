"""Path pursuit + progress enrich + EscapeMachine.tick_pre for one tick."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, TYPE_CHECKING

from ..nav.simple_motion import DriveCommand, distance_m
from .controller import compute_path_command
from .escape import EscapeSense
from .replan_controller import apply_escape_pre_replan
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


def _mono() -> float:
    from . import supervisor as _sup

    return _sup.time.monotonic()


@dataclass
class PathFollowResult:
    cmd: DriveCommand
    progress: Mapping[str, Any]
    path: Path2D
    pose: Pose2D
    abort: bool = False


def run_path_follow(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
    *,
    goal: Pose2D,
    path: Path2D,
    pose: Pose2D,
    waiting_for_clear: bool,
    corridor_yaw: Optional[float],
    loc_hold_motion: bool,
) -> PathFollowResult:
    """DWA/pursuit command, progress bag, then escape_pre (+ optional replan)."""
    now = tick.now
    scan = tick.scan
    mem_pts = tick.mem_pts
    local_view = tick.local_view
    guard_pts = tick.guard_pts
    nose_clear = tick.nose_clear
    path_ahead_cost = tick.path_ahead_cost
    pose_cost = tick.pose_cost
    local_blocked = tick.local_blocked
    map_untrusted = tick.map_untrusted
    collision_view = tick.collision_view

    # DWA only when the route is actually blocked (or after a failed
    # detour). Clear-nose C-space pinch: pursuit + reactive slow —
    # DWA reverse/spin samples were the doorway hunt-and-peck.
    allow_local_planner = (
        nav._local_costmap_enabled  # noqa: SLF001
        and not waiting_for_clear
        and (
            local_blocked
            or ep.failed_replan_while_blocked >= 1
        )
    )
    force_local = bool(local_blocked and allow_local_planner)
    # Reactive stop/slow, nose_clear, and the local costmap all use
    # the fused scan so depth can catch low / lidar-blind hits.
    # Loc refine stays lidar-only. Replan guards still refuse
    # avoid→replan death spirals when the nose and path are clear.
    cmd, progress = compute_path_command(
        pose,
        path,
        cfg=nav._follower,  # noqa: SLF001
        scan=scan,
        speed_mps=nav._last_cmd_vx,  # noqa: SLF001
        local_view=collision_view,
        local_planner=nav._local_planner if allow_local_planner else None,  # noqa: SLF001
        robot_radius_m=nav._robot_radius,  # noqa: SLF001
        spin_radius_m=nav._spin_radius,  # noqa: SLF001
        min_cmd_vel_x=nav._follower.motion.min_linear_mps,  # noqa: SLF001
        min_cmd_vel_theta=nav._follower.motion.min_angular_rad_s,  # noqa: SLF001
        local_planner_active=ep.local_planner_active,
        prev_local_cmd=ep.prev_local_cmd,
        rotate_active=ep.rotate_active,
        prev_cmd=ep.prev_cmd,
        force_local_planner=force_local,
        guard=nav._guard,  # noqa: SLF001
        guard_extra_pts=mem_pts,
    )
    ep.rotate_active = bool(progress.get("rotate_to_heading"))
    ep.local_planner_active = bool(progress.get("local_planner"))
    if ep.local_planner_active:
        ep.prev_local_cmd = cmd
    else:
        ep.prev_local_cmd = None

    obs_state = str(progress.get("obstacle") or "")
    if obs_state == "avoid":
        if ep.reactive_avoid_since is None:
            ep.reactive_avoid_since = now
    else:
        ep.reactive_avoid_since = None
    ep.last_obstacle_state = obs_state
    progress = {
        **progress,
        "local_blocked": bool(local_blocked),
        "path_cost_ahead": int(path_ahead_cost),
        "pose_cost": int(pose_cost),
        "failed_replan_while_blocked": int(ep.failed_replan_while_blocked),
        "last_replan_error": nav._last_replan_error,  # noqa: SLF001
        "last_replan_trigger": nav._last_replan_trigger,  # noqa: SLF001
        "last_replan_info": dict(nav._last_replan_info),  # noqa: SLF001
        **(
            {"stuck_pose_refine": dict(nav._stuck_pose_refine)}  # noqa: SLF001
            if nav._stuck_pose_refine  # noqa: SLF001
            else {}
        ),
        "nose_clear": bool(nose_clear),
        "map_pose_untrusted": bool(map_untrusted),
        "above_cart": (
            int(len(nav._above_memory))  # noqa: SLF001
            if nav._above_memory is not None  # noqa: SLF001
            else None
        ),
        "local_replan_cooldown_s": round(
            max(
                0.0,
                nav._replan_local_min_period_s  # noqa: SLF001
                - (_mono() - ep.last_local_replan_at),
            ),
            2,
        ),
        "distance_remaining_m": distance_m(pose, goal),
    }

    # Bumper adopt / narrow_reverse bound / wait inject / unstick.
    escape_pre = ep.escape_m.tick_pre(
        EscapeSense(
            now=now,
            pose=pose,
            cmd=cmd,
            progress=progress,
            nose_clear=nose_clear,
            waiting=waiting_for_clear,
            failed_replans=ep.failed_replan_while_blocked,
            spin_blocked=bool(progress.get("spin_blocked")),
            heading_open=corridor_yaw is not None,
            bearing_rad=float(progress.get("bearing_error_rad") or 0.0),
            scan=scan,
            local_view=local_view,
            guard=nav._guard,  # noqa: SLF001
            guard_pts=guard_pts,
            robot_radius_m=nav._robot_radius,  # noqa: SLF001
            spin_radius_m=nav._spin_radius,  # noqa: SLF001
            follower=nav._follower,  # noqa: SLF001
            backup_dist_m=float(nav._backup_dist_m),  # noqa: SLF001
            backup_cooldown_s=float(nav._backup_cooldown_s),  # noqa: SLF001
            loc_hold=loc_hold_motion,
            prefer_yaw_sign=(
                math.copysign(1.0, corridor_yaw)
                if corridor_yaw is not None
                else ep.escape_m.yaw_sign
            ),
        )
    )
    if escape_pre.replan_trigger is not None:
        outcome = apply_escape_pre_replan(
            nav,
            ep,
            goal=goal,
            pose=pose,
            path=path,
            scan=scan,
            local_view=local_view,
            trigger=escape_pre.replan_trigger,
            nose_clear=nose_clear,
            corridor_yaw=corridor_yaw,
            now=now,
        )
        path, pose = outcome.path, outcome.pose
        if outcome.abort:
            return PathFollowResult(
                cmd=cmd, progress=progress, path=path, pose=pose, abort=True
            )
        cmd = escape_pre.cmd or DriveCommand(0.0, 0.0, 0.0, False)
        progress = {
            **progress,
            "obstacle": escape_pre.obstacle or "planning",
            "local_planner": False,
            "nose_unstick": None,
            "cmd_vx_mps": cmd.vx,
            "cmd_vtheta_rad_s": cmd.vtheta,
        }
    elif escape_pre.cmd is not None:
        cmd = escape_pre.cmd
        progress = {
            **progress,
            "obstacle": escape_pre.obstacle or progress.get("obstacle"),
            "local_planner": False,
            "nose_unstick": escape_pre.active_mode,
            "cmd_vx_mps": cmd.vx,
            "cmd_vtheta_rad_s": cmd.vtheta,
        }
    if waiting_for_clear:
        ep.last_progress_at = now
        ep.last_progress_pose = pose
        ep.last_progress_dist = distance_m(pose, goal)
        ep.last_progress_bearing = float("inf")

    return PathFollowResult(cmd=cmd, progress=progress, path=path, pose=pose)

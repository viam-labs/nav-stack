"""Path pursuit + progress enrich + EscapeMachine.tick_pre for one tick."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..nav.simple_motion import DriveCommand, distance_m
from .controller import compute_path_command
from .replan_controller import apply_escape_pre_replan

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


def _mono() -> float:
    from . import supervisor as _sup

    return _sup.time.monotonic()


@dataclass
class PathFollowResult:
    abort: bool = False


def run_path_follow(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> PathFollowResult:
    """DWA/pursuit command, progress bag, then escape_pre (+ optional replan)."""
    allow_local_planner = (
        nav._local_costmap_enabled  # noqa: SLF001
        and not tick.waiting_for_clear
        and (
            tick.local_blocked
            or ep.failed_replan_while_blocked >= 1
        )
    )
    force_local = bool(tick.local_blocked and allow_local_planner)
    cmd, progress = compute_path_command(
        tick.pose,
        tick.path,
        cfg=nav._follower,  # noqa: SLF001
        scan=tick.scan,
        speed_mps=nav._last_cmd_vx,  # noqa: SLF001
        local_view=tick.collision_view,
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
        guard_extra_pts=tick.mem_pts,
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
            ep.reactive_avoid_since = tick.now
    else:
        ep.reactive_avoid_since = None
    ep.last_obstacle_state = obs_state
    tick.cmd = cmd
    tick.progress = {
        **progress,
        "local_blocked": bool(tick.local_blocked),
        "path_cost_ahead": int(tick.path_ahead_cost),
        "pose_cost": int(tick.pose_cost),
        "failed_replan_while_blocked": int(ep.failed_replan_while_blocked),
        "last_replan_error": nav._last_replan_error,  # noqa: SLF001
        "last_replan_trigger": nav._last_replan_trigger,  # noqa: SLF001
        "last_replan_info": dict(nav._last_replan_info),  # noqa: SLF001
        **(
            {"stuck_pose_refine": dict(nav._stuck_pose_refine)}  # noqa: SLF001
            if nav._stuck_pose_refine  # noqa: SLF001
            else {}
        ),
        "nose_clear": bool(tick.nose_clear),
        "map_pose_untrusted": bool(tick.map_untrusted),
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
        "distance_remaining_m": distance_m(tick.pose, tick.goal),
    }

    escape_pre = ep.escape_m.tick_pre(tick.escape_sense(nav, ep))
    if escape_pre.replan_trigger is not None:
        outcome = apply_escape_pre_replan(
            nav, ep, tick, trigger=escape_pre.replan_trigger
        )
        if outcome.abort:
            return PathFollowResult(abort=True)
        tick.cmd = escape_pre.cmd or DriveCommand(0.0, 0.0, 0.0, False)
        tick.progress = {
            **tick.progress,
            "obstacle": escape_pre.obstacle or "planning",
            "local_planner": False,
            "nose_unstick": None,
            "cmd_vx_mps": tick.cmd.vx,
            "cmd_vtheta_rad_s": tick.cmd.vtheta,
        }
    elif escape_pre.cmd is not None:
        tick.cmd = escape_pre.cmd
        tick.progress = {
            **tick.progress,
            "obstacle": escape_pre.obstacle or tick.progress.get("obstacle"),
            "local_planner": False,
            "nose_unstick": escape_pre.active_mode,
            "cmd_vx_mps": tick.cmd.vx,
            "cmd_vtheta_rad_s": tick.cmd.vtheta,
        }
    if tick.waiting_for_clear:
        ep.last_progress_at = tick.now
        ep.last_progress_pose = tick.pose
        ep.last_progress_dist = distance_m(tick.pose, tick.goal)
        ep.last_progress_bearing = float("inf")

    return PathFollowResult()

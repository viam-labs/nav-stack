"""Rate-limit → guard veto → same-arc → EscapeMachine.tick_post.

Last motion shaping before status / stall / set_velocity. Corridor yaw
override is applied here so the limited command is what the base sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, TYPE_CHECKING

from ..nav.simple_motion import DriveCommand, distance_m
from .escape import EscapeSense
from .footprint_guard import obstacle_points
from .replan_controller import apply_escape_post_replan
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .supervisor import NavSupervisor

# Match supervisor: a few degrees is closed while a moving pursuit can finish.
_CORRIDOR_ALIGN_RAD = math.radians(3.0)


@dataclass
class GateResult:
    cmd: DriveCommand
    progress: Mapping[str, Any]
    path: Path2D
    pose: Pose2D
    # Nearest-obstacle fields for the optional TRACE append (guard path only).
    traced: bool = False
    near_d: float = float("inf")
    near_x: float = float("nan")
    near_y: float = float("nan")


def run_final_gate(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    now: float,
    pose: Pose2D,
    goal: Pose2D,
    path: Path2D,
    scan,
    local_view,
    guard_pts,
    nose_clear: bool,
    cmd: DriveCommand,
    progress: Mapping[str, Any],
    corridor_yaw: Optional[float],
    waiting_for_clear: bool,
    loc_hold_motion: bool,
) -> GateResult:
    """Corridor override, slew limit, footprint veto, undrivable tick_post."""
    progress = dict(progress)
    traced = False
    near_d = float("inf")
    near_x = float("nan")
    near_y = float("nan")

    if corridor_yaw is not None and not loc_hold_motion:
        ep.escape_m.yaw_sign = math.copysign(1.0, corridor_yaw)
        # A moving pursuit command already closes a sub-3° error.
        # Replace it when the opening is off the nose, or when
        # the command has already stopped.
        if abs(corridor_yaw) >= _CORRIDOR_ALIGN_RAD or cmd.vx < 0.05:
            cmd, label = nav._corridor_motion(pose, scan, corridor_yaw)  # noqa: SLF001
            progress = {
                **progress,
                "obstacle": label,
                "local_planner": False,
                "corridor_yaw_rad": round(float(corridor_yaw), 3),
                "cmd_vx_mps": cmd.vx,
                "cmd_vtheta_rad_s": cmd.vtheta,
            }
    if loc_hold_motion:
        cmd = DriveCommand(0.0, 0.0, 0.0, False)
    pre_limit = cmd
    cmd = nav._rate_limited(cmd)  # noqa: SLF001
    if nav._guard is not None and not cmd.done:  # noqa: SLF001
        # Reduce-only veto on whatever is about to go to the base
        # (slew-limited pursuit, backup, recovery spins).
        veto_pts = guard_pts
        if (
            progress.get("obstacle") in ("yaw_open", "corridor")
            and scan is not None
        ):
            # Inflation of the blob on the nose fills the spin disc
            # even when the rectangle can rotate and then drive.
            # Judge this escape against the raw scan.
            veto_pts = obstacle_points(
                pose,
                scan,
                None,
                radius_m=nav._guard.cfg.obstacle_radius_m,  # noqa: SLF001
            )

        def _veto(c: DriveCommand):
            return nav._guard.guard(  # noqa: SLF001
                pose,
                c.vx,
                c.vtheta,
                veto_pts,
                max_dist_m=distance_m(pose, goal),
                allow_steer=False,
            )

        def _reduced(c: DriveCommand, v) -> bool:
            return abs(v.vx) < abs(c.vx) - 1e-9 or abs(v.vtheta) < abs(
                c.vtheta
            ) - 1e-9

        veto = _veto(cmd)
        if _reduced(cmd, veto) and abs(pre_limit.vx) > 1e-6:
            # The slew limiter ramps ω slower than v, which
            # straightens the arc the guard approved — through a
            # doorway that clips the jamb. Keep the approved
            # curvature at the ramped speed instead.
            same_arc = DriveCommand(
                cmd.vx,
                cmd.vy,
                pre_limit.vtheta * (cmd.vx / pre_limit.vx),
                cmd.done,
            )
            arc_veto = _veto(same_arc)
            if not _reduced(same_arc, arc_veto):
                cmd, veto = same_arc, arc_veto
                nav._last_sent_cmd = cmd  # noqa: SLF001
        if _reduced(cmd, veto):
            cmd = DriveCommand(veto.vx, cmd.vy, veto.vtheta, cmd.done)
            nav._last_sent_cmd = cmd  # noqa: SLF001
        # The veto can leave a crawl the base sanitizer turns into
        # a stop. That never ramps, so the bumper stays put. Yaw
        # onto the path when the sweep is free; otherwise back up.
        if not cmd.done:
            escape_post = ep.escape_m.tick_post(
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
                    bearing_rad=float(
                        progress.get("bearing_error_rad") or 0.0
                    ),
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
                    corridor_active=progress.get("obstacle")
                    in ("yaw_open", "corridor"),
                    prefer_yaw_sign=ep.escape_m.yaw_sign,
                )
            )
            if escape_post.replan_trigger is not None:
                outcome = apply_escape_post_replan(
                    nav,
                    ep,
                    goal=goal,
                    pose=pose,
                    path=path,
                    scan=scan,
                    local_view=local_view,
                    trigger=escape_post.replan_trigger,
                    now=now,
                )
                path, pose = outcome.path, outcome.pose
                cmd = escape_post.cmd or DriveCommand(0.0, 0.0, 0.0, False)
                nav._last_sent_cmd = cmd  # noqa: SLF001
                progress = {
                    **progress,
                    "obstacle": escape_post.obstacle or "narrow_reverse_done",
                    "local_planner": False,
                    "cmd_vx_mps": 0.0,
                    "cmd_vtheta_rad_s": 0.0,
                }
            elif escape_post.cmd is not None:
                cmd = escape_post.cmd
                nav._last_sent_cmd = cmd  # noqa: SLF001
                progress = {
                    **progress,
                    "obstacle": escape_post.obstacle
                    or progress.get("obstacle"),
                    "local_planner": False,
                    "cmd_vx_mps": cmd.vx,
                    "cmd_vtheta_rad_s": cmd.vtheta,
                }
        near_d, near_x, near_y = nav._guard.nearest(pose, guard_pts)  # noqa: SLF001
        traced = True

    return GateResult(
        cmd=cmd,
        progress=progress,
        path=path,
        pose=pose,
        traced=traced,
        near_d=near_d,
        near_x=near_x,
        near_y=near_y,
    )

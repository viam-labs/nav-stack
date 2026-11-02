"""Rate-limit → guard veto → same-arc → EscapeMachine.tick_post.

Last motion shaping before status / stall / set_velocity. Mutates the
``TickContext`` command and progress.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..nav.simple_motion import DriveCommand, distance_m
from .base_envelope import LIN_EPS_MPS
from .footprint_guard import obstacle_points
from .replan_controller import apply_escape_post_replan
from .thresholds import CORRIDOR_ALIGN_RAD

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


@dataclass
class GateResult:
    # Nearest-obstacle fields for the optional TRACE append (guard path only).
    traced: bool = False
    near_d: float = float("inf")
    near_x: float = float("nan")
    near_y: float = float("nan")


def run_final_gate(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> GateResult:
    """Corridor override, slew limit, footprint veto, undrivable tick_post."""
    progress = dict(tick.progress)
    cmd = tick.cmd
    corridor_yaw = tick.corridor_yaw
    traced = False
    near_d = float("inf")
    near_x = float("nan")
    near_y = float("nan")

    if corridor_yaw is not None and not tick.loc_hold_motion:
        ep.escape_m.yaw_sign = math.copysign(1.0, corridor_yaw)
        if abs(corridor_yaw) >= CORRIDOR_ALIGN_RAD or cmd.vx < LIN_EPS_MPS:
            cmd, label = nav._corridor_motion(  # noqa: SLF001
                tick.pose,
                tick.scan,
                corridor_yaw,
                scan_world_pts=tick.scan_world_pts,
            )
            progress = {
                **progress,
                "obstacle": label,
                "local_planner": False,
                "corridor_yaw_rad": round(float(corridor_yaw), 3),
                "cmd_vx_mps": cmd.vx,
                "cmd_vtheta_rad_s": cmd.vtheta,
            }
    if tick.loc_hold_motion:
        cmd = DriveCommand(0.0, 0.0, 0.0, False)
    pre_limit = cmd
    cmd = nav._rate_limited(cmd)  # noqa: SLF001
    tick.cmd = cmd
    tick.progress = progress
    if nav._guard is not None and not cmd.done:  # noqa: SLF001
        veto_pts = tick.guard_pts
        if (
            progress.get("obstacle") in ("yaw_open", "corridor")
            and tick.scan is not None
        ):
            veto_pts = obstacle_points(
                tick.pose,
                tick.scan,
                None,
                radius_m=nav._guard.cfg.obstacle_radius_m,  # noqa: SLF001
                scan_world_pts=tick.scan_world_pts,
            )

        def _veto(c: DriveCommand):
            return nav._guard.guard(  # noqa: SLF001
                tick.pose,
                c.vx,
                c.vtheta,
                veto_pts,
                max_dist_m=distance_m(tick.pose, tick.goal),
                allow_steer=False,
            )

        def _reduced(c: DriveCommand, v) -> bool:
            return abs(v.vx) < abs(c.vx) - 1e-9 or abs(v.vtheta) < abs(
                c.vtheta
            ) - 1e-9

        veto = _veto(cmd)
        if _reduced(cmd, veto) and abs(pre_limit.vx) > 1e-6:
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
        tick.cmd = cmd
        if not cmd.done:
            escape_post = ep.escape_m.tick_post(
                tick.escape_sense(
                    nav,
                    ep,
                    prefer_yaw_sign=ep.escape_m.yaw_sign,
                    corridor_active=tick.progress.get("obstacle")
                    in ("yaw_open", "corridor"),
                )
            )
            if escape_post.replan_trigger is not None:
                apply_escape_post_replan(
                    nav, ep, tick, trigger=escape_post.replan_trigger
                )
                tick.cmd = escape_post.cmd or DriveCommand(0.0, 0.0, 0.0, False)
                nav._last_sent_cmd = tick.cmd  # noqa: SLF001
                tick.progress = {
                    **tick.progress,
                    "obstacle": escape_post.obstacle or "narrow_reverse_done",
                    "local_planner": False,
                    "cmd_vx_mps": 0.0,
                    "cmd_vtheta_rad_s": 0.0,
                }
            elif escape_post.cmd is not None:
                tick.cmd = escape_post.cmd
                nav._last_sent_cmd = tick.cmd  # noqa: SLF001
                tick.progress = {
                    **tick.progress,
                    "obstacle": escape_post.obstacle
                    or tick.progress.get("obstacle"),
                    "local_planner": False,
                    "cmd_vx_mps": tick.cmd.vx,
                    "cmd_vtheta_rad_s": tick.cmd.vtheta,
                }
        near_d, near_x, near_y = nav._guard.nearest(  # noqa: SLF001
            tick.pose, tick.guard_pts
        )
        traced = True

    return GateResult(
        traced=traced, near_d=near_d, near_x=near_x, near_y=near_y
    )

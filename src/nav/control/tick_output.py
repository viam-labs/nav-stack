"""Post-gate status publish, TRACE, speed-zone, and base command for one tick."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ...geom import conversions as conv
from ..motion.simple_motion import DriveCommand
from .loc_hold_tick import overlay_loc_hold_progress
from .sense_tick import PHASE_MS_KEYS

if TYPE_CHECKING:
    from .final_gate import GateResult
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


@dataclass
class TickOutputResult:
    # Caller should continue (no_scan).
    skip_rest: bool = False
    # Caller should return (reached goal on cmd.done).
    succeeded: bool = False


_PROGRESS_KEYS = (
    "obstacle",
    "local_planner",
    "local_blocked",
    "path_cost_ahead",
    "failed_replan_while_blocked",
    "last_replan_error",
    "nose_clear",
    "map_pose_untrusted",
    "spin_blocked",
    "local_replan_cooldown_s",
    "forward_clearance_m",
    "cmd_vx_mps",
    "cmd_vtheta_rad_s",
    "bearing_error_rad",
    "corridor_yaw_rad",
    "distance_remaining_m",
    "waypoint_index",
    "localization_hold",
    *PHASE_MS_KEYS,
)


def finish_tick_output(
    nav: "NavSupervisor",
    tick: "TickContext",
    gate: "GateResult",
    *,
    holding_for_localize: bool,
    loc_hold: Optional[Mapping[str, Any]],
    tick_started: float,
) -> TickOutputResult:
    """Speed-zone, TRACE, status, no_scan stop, cmd.done goal check."""
    from . import supervisor as _sup

    if gate.traced:
        mem_pts = tick.mem_pts
        guard_pts = tick.guard_pts
        _sup._TRACE.append(
            {
                "t": round(time.time(), 3),
                "x": round(tick.pose.x, 3),
                "y": round(tick.pose.y, 3),
                "th": round(tick.pose.theta, 4),
                "vx": round(tick.cmd.vx, 3),
                "w": round(tick.cmd.vtheta, 3),
                "obs": tick.progress.get("obstacle"),
                "near_m": (
                    round(gate.near_d, 3) if math.isfinite(gate.near_d) else None
                ),
                "near_bx": (
                    round(gate.near_x, 3) if math.isfinite(gate.near_x) else None
                ),
                "near_by": (
                    round(gate.near_y, 3) if math.isfinite(gate.near_y) else None
                ),
                "pts": int(len(guard_pts)) if guard_pts is not None else 0,
                "mem": int(len(mem_pts)) if mem_pts is not None else 0,
                "wp": tick.progress.get("waypoint_index"),
            }
        )
    tick.cmd = nav._apply_speed_zone(tick.pose, tick.cmd)  # noqa: SLF001
    tick.cmd = nav._lift_sub_deadband(  # noqa: SLF001
        tick.cmd, tick.progress.get("forward_clearance_m")
    )
    tick.progress = {
        **tick.progress,
        "cmd_vx_mps": tick.cmd.vx,
        "cmd_vtheta_rad_s": tick.cmd.vtheta,
        **{k: tick.phase_ms[k] for k in PHASE_MS_KEYS if k in tick.phase_ms},
    }
    tick.progress = overlay_loc_hold_progress(
        tick.progress,
        holding_for_localize=holding_for_localize,
        loc_hold=loc_hold,
    )
    nav._note_phase_ms(tick.phase_ms)  # noqa: SLF001

    nav._set_status(  # noqa: SLF001
        pose={"x": tick.pose.x, "y": tick.pose.y, "theta": tick.pose.theta},
        progress={k: tick.progress[k] for k in _PROGRESS_KEYS if k in tick.progress},
    )
    if tick.progress.get("obstacle") == "no_scan":
        nav._world.set_velocity(0.0, 0.0, 0.0)  # noqa: SLF001
        nav._last_sent_cmd = DriveCommand(0.0, 0.0, 0.0, False)  # noqa: SLF001
        nav._last_sent_at = _sup.time.monotonic()  # noqa: SLF001
        nav._sleep_control_period(tick_started)  # noqa: SLF001
        return TickOutputResult(skip_rest=True)

    if tick.cmd.done:
        at_goal = nav._xy_at_nav_goal(tick.pose, tick.goal, tick.path) and (  # noqa: SLF001
            abs(conv.normalize_angle(tick.pose.theta - tick.goal.theta))
            <= nav._follower.motion.yaw_tolerance_rad  # noqa: SLF001
        )
        if at_goal:
            nav._world.stop()  # noqa: SLF001
            nav._set_status(state="succeeded", active=False, error_msg="")  # noqa: SLF001
            return TickOutputResult(succeeded=True)

    return TickOutputResult()


def send_tick_velocity(
    nav: "NavSupervisor",
    ep,
    tick: "TickContext",
    *,
    tick_started: float,
) -> bool:
    """Publish cmd_vel. True ⇒ caller should continue the outer loop."""
    from .controller import update_speed_estimate

    cmd = tick.cmd
    try:
        nav._world.set_velocity(cmd.vx, cmd.vy, cmd.vtheta)  # noqa: SLF001
        nav._io_timeout_streak = 0  # noqa: SLF001
    except TimeoutError:
        nav._io_timeout_streak += 1  # noqa: SLF001
        if nav._io_timeout_streak >= nav._drive_timeout_streak:  # noqa: SLF001
            raise
        nav._sleep_control_period(tick_started)  # noqa: SLF001
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "nearly 0" in msg or "rpm" in msg:
            try:
                nav._world.stop()  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass
            nav._note_base_stopped()  # noqa: SLF001
            nav._sleep_control_period(tick_started)  # noqa: SLF001
            return True
        if "goaway" in msg or "unavailable" in msg or "connection" in msg:
            nav._io_timeout_streak += 1  # noqa: SLF001
            if nav._io_timeout_streak >= max(8, nav._drive_timeout_streak):  # noqa: SLF001
                raise
            nav._sleep_control_period(tick_started)  # noqa: SLF001
            return True
        raise
    nav._last_cmd_vx = update_speed_estimate(nav._last_cmd_vx, cmd.vx)  # noqa: SLF001
    ep.prev_cmd = cmd
    nav._sleep_control_period(tick_started)  # noqa: SLF001
    return False

"""Post-gate status publish, TRACE, speed-zone, and base command for one tick."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, TYPE_CHECKING

from ..geom import conversions as conv
from ..nav.simple_motion import DriveCommand
from .loc_hold_tick import overlay_loc_hold_progress

if TYPE_CHECKING:
    from .final_gate import GateResult
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor
    from .types import Path2D, Pose2D


@dataclass
class TickOutputResult:
    cmd: DriveCommand
    progress: Mapping[str, Any]
    # Caller should continue (no_scan / transient drive reject).
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
)


def finish_tick_output(
    nav: "NavSupervisor",
    tick: "TickContext",
    gate: "GateResult",
    *,
    goal: "Pose2D",
    path: "Path2D",
    pose: "Pose2D",
    cmd: DriveCommand,
    progress: Mapping[str, Any],
    holding_for_localize: bool,
    loc_hold: Optional[Mapping[str, Any]],
    tick_started: float,
) -> TickOutputResult:
    """Speed-zone, TRACE, status, no_scan stop, cmd.done goal check."""
    from . import supervisor as _sup

    progress = dict(progress)
    if gate.traced:
        mem_pts = tick.mem_pts
        guard_pts = tick.guard_pts
        _sup._TRACE.append(
            {
                "t": round(time.time(), 3),
                "x": round(pose.x, 3),
                "y": round(pose.y, 3),
                "th": round(pose.theta, 4),
                "vx": round(cmd.vx, 3),
                "w": round(cmd.vtheta, 3),
                "obs": progress.get("obstacle"),
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
                "wp": progress.get("waypoint_index"),
            }
        )
    # Apply before status / lookahead feedback so the limited twist
    # is what we report and close the loop on.
    cmd = nav._apply_speed_zone(pose, cmd)  # noqa: SLF001
    cmd = nav._lift_sub_deadband(  # noqa: SLF001
        cmd, progress.get("forward_clearance_m")
    )
    progress = {
        **progress,
        "cmd_vx_mps": cmd.vx,
        "cmd_vtheta_rad_s": cmd.vtheta,
    }
    progress = overlay_loc_hold_progress(
        progress,
        holding_for_localize=holding_for_localize,
        loc_hold=loc_hold,
    )

    nav._set_status(  # noqa: SLF001
        pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
        progress={k: progress[k] for k in _PROGRESS_KEYS if k in progress},
    )
    if progress.get("obstacle") == "no_scan":
        # No fresh lidar frame: stop translation and yaw. The last
        # scan must not be driven or spun on.
        nav._world.set_velocity(0.0, 0.0, 0.0)  # noqa: SLF001
        nav._last_sent_cmd = DriveCommand(0.0, 0.0, 0.0, False)  # noqa: SLF001
        nav._last_sent_at = _sup.time.monotonic()  # noqa: SLF001
        nav._sleep_control_period(tick_started)  # noqa: SLF001
        return TickOutputResult(cmd=cmd, progress=progress, skip_rest=True)

    if cmd.done:
        at_goal = nav._xy_at_nav_goal(pose, goal, path) and (  # noqa: SLF001
            abs(conv.normalize_angle(pose.theta - goal.theta))
            <= nav._follower.motion.yaw_tolerance_rad  # noqa: SLF001
        )
        if at_goal:
            nav._world.stop()  # noqa: SLF001
            nav._set_status(state="succeeded", active=False, error_msg="")  # noqa: SLF001
            return TickOutputResult(cmd=cmd, progress=progress, succeeded=True)

    return TickOutputResult(cmd=cmd, progress=progress)


def send_tick_velocity(
    nav: "NavSupervisor",
    ep,
    cmd: DriveCommand,
    *,
    tick_started: float,
) -> bool:
    """Publish cmd_vel. True ⇒ caller should continue the outer loop."""
    from .controller import update_speed_estimate

    try:
        nav._world.set_velocity(cmd.vx, cmd.vy, cmd.vtheta)  # noqa: SLF001
        nav._io_timeout_streak = 0  # noqa: SLF001
    except TimeoutError:
        # Transient event-loop starvation — don't abort the goal on
        # a single missed cmd_vel (common when SLAM mapping + lidar
        # gRPC share the module loop with Base.SetVelocity).
        nav._io_timeout_streak += 1  # noqa: SLF001
        if nav._io_timeout_streak >= nav._drive_timeout_streak:  # noqa: SLF001
            raise
        nav._sleep_control_period(tick_started)  # noqa: SLF001
        return True
    except Exception as exc:  # noqa: BLE001
        # Motor "nearly 0 RPM" / similar drive rejects: skip tick.
        # Also treat gRPC GOAWAY / unavailable as transient while
        # resources flap (lidar USB reconnect storms).
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
    # Smoothed speed for the velocity-scaled lookahead (see
    # ``update_speed_estimate``): raw cmd feedback limit-cycles.
    nav._last_cmd_vx = update_speed_estimate(nav._last_cmd_vx, cmd.vx)  # noqa: SLF001
    ep.prev_cmd = cmd
    nav._sleep_control_period(tick_started)  # noqa: SLF001
    return False

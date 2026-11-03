"""Localization hold / mid-nav refine for one ``run_goal`` tick.

Pose-jump ``awaiting_confirm`` and ``_maybe_pause_and_refine_localization``
both force zero motion while the rest of the tick (sense / avoid) still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..types import Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .supervisor import NavSupervisor


def _mono() -> float:
    from . import supervisor as _sup

    return _sup.time.monotonic()


@dataclass
class LocHoldResult:
    now: float
    loc_hold_motion: bool
    holding_for_localize: bool
    loc_hold: Optional[Mapping[str, Any]]
    # Supervisor should abort the goal (refine failed).
    abort: bool = False


def run_loc_hold_tick(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    pose: Pose2D,
    dist_goal_m: float,
    tick_started: float,
) -> LocHoldResult:
    """Update episode loc-hold state; may extend the goal deadline."""
    loc_hold = None
    hold_fn = getattr(nav._world, "get_localization_hold", None)  # noqa: SLF001
    if callable(hold_fn):
        try:
            loc_hold = hold_fn()
        except Exception:  # noqa: BLE001
            loc_hold = None
    holding_for_localize = isinstance(loc_hold, dict)
    entering_loc_hold = holding_for_localize and not ep.was_loc_holding
    if ep.was_loc_holding and not holding_for_localize:
        ep.pending_loc_replan = True
    ep.was_loc_holding = holding_for_localize

    now = _mono()
    if holding_for_localize:
        hold_status = str((loc_hold or {}).get("status") or "")
        # A leftover mid-nav ``nav_hold`` (refused hallway yank)
        # must not skip refine forever. ``awaiting_confirm`` still
        # stops until SLAM applies or rejects.
        if hold_status == "nav_hold" and nav._nav_loc_refine:  # noqa: SLF001
            holding_for_localize = False
            entering_loc_hold = False

    # Localization never skips the rest of the tick. A match runs
    # on its own thread; this loop still reads the scan, runs
    # obstacle avoidance, and sends a command every period.
    loc_hold_motion = False
    if holding_for_localize:
        if entering_loc_hold:
            nav._world.stop()  # noqa: SLF001
            nav._note_base_stopped()  # noqa: SLF001
        loc_hold_motion = True
        ep.last_progress_at = now
        ep.last_progress_pose = pose
        ep.last_progress_dist = dist_goal_m
        ep.last_progress_bearing = float("inf")
        ep.spin_stuck_since = None
        ep.deadline += _mono() - tick_started
        nav._set_status(  # noqa: SLF001
            pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
            progress={
                "obstacle": "loc_hold",
                "local_planner": False,
                "forward_clearance_m": None,
                "cmd_vx_mps": 0.0,
                "cmd_vtheta_rad_s": 0.0,
                "bearing_error_rad": 0.0,
                "distance_remaining_m": dist_goal_m,
                "waypoint_index": 0,
                "localization_hold": {
                    "status": loc_hold.get("status"),
                    "confirm_count": loc_hold.get("confirm_count"),
                    "confirm_needed": loc_hold.get("confirm_needed"),
                    "shift_m": loc_hold.get("shift_m") or loc_hold.get("jump_shift_m"),
                    "shift_deg": loc_hold.get("shift_deg")
                    or loc_hold.get("jump_shift_deg"),
                },
            },
        )
        return LocHoldResult(
            now=now,
            loc_hold_motion=loc_hold_motion,
            holding_for_localize=holding_for_localize,
            loc_hold=loc_hold,
        )

    now = _mono()
    loc_outcome = nav._maybe_pause_and_refine_localization(  # noqa: SLF001
        pose, now, dist_goal_m
    )
    if loc_outcome == "fail":
        return LocHoldResult(
            now=now,
            loc_hold_motion=False,
            holding_for_localize=False,
            loc_hold=None,
            abort=True,
        )
    if loc_outcome == "resume":
        ep.pending_loc_replan = True
    if loc_outcome == "hold":
        loc_hold_motion = True
        ep.last_progress_at = now
        ep.last_progress_pose = pose
        ep.last_progress_dist = dist_goal_m
        ep.last_progress_bearing = float("inf")
        ep.spin_stuck_since = None
    # Time spent waiting on localization does not count against the goal timeout.
    if loc_outcome in ("hold", "resume", "continue"):
        ep.deadline += _mono() - tick_started
    return LocHoldResult(
        now=now,
        loc_hold_motion=loc_hold_motion,
        holding_for_localize=False,
        loc_hold=None,
    )


def overlay_loc_hold_progress(
    progress: dict,
    *,
    holding_for_localize: bool,
    loc_hold: Optional[Mapping[str, Any]],
) -> dict:
    """Relabel avoidance progress as loc_hold when the base is held for SLAM."""
    if not holding_for_localize or progress.get("obstacle") == "no_scan":
        return progress
    if not isinstance(loc_hold, dict):
        return progress
    out = dict(progress)
    out["obstacle"] = "loc_hold"
    out["localization_hold"] = {
        "status": loc_hold.get("status"),
        "confirm_count": loc_hold.get("confirm_count"),
        "confirm_needed": loc_hold.get("confirm_needed"),
        "shift_m": loc_hold.get("shift_m") or loc_hold.get("jump_shift_m"),
        "shift_deg": loc_hold.get("shift_deg") or loc_hold.get("jump_shift_deg"),
    }
    return out

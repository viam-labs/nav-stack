"""Typed replan + episode reset packs used by ``run_goal``.

Centralizes ``_try_replan`` call sites so each trigger applies the matching
``GoalEpisode`` reset method. Call sites pass a ``TickContext``; path/pose
updates land back on the tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .sense_tick import TickContext
    from .supervisor import NavSupervisor


def _mono() -> float:
    # Use supervisor's ``time`` so the scenario harness FakeClock applies
    # (it monkeypatches ``nav_builtin.supervisor.time``, not stdlib).
    from . import supervisor as _sup

    return _sup.time.monotonic()


@dataclass
class ReplanOutcome:
    # Supervisor should abort the goal (blocked-replan limit / static fail).
    abort: bool = False
    abort_msg: str = ""


def apply_blocked_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> ReplanOutcome:
    """Policy action == replan. Mutates tick.path/pose/corridor_yaw."""
    nose_clear = tick.nose_clear
    path_ahead_cost = tick.path_ahead_cost
    _trig = f"local_blocked cost={path_ahead_cost} nose_clear={nose_clear}"
    nav._stop_before_replan(_trig)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        tick.goal,
        tick.pose,
        tick.path,
        tick.scan,
        failed_count=max(1, ep.failed_replan_while_blocked),
        require_different=ep.failed_replan_while_blocked < 3,
        local_view=tick.local_view,
        trigger=_trig,
    )
    ep.note_replan_finished(_mono())
    if new_path is None:
        recovered = nav._recover_unreachable_start(  # noqa: SLF001
            tick.goal,
            tick.pose,
            tick.path,
            tick.scan,
            failed_count=max(1, ep.failed_replan_while_blocked),
            local_view=tick.local_view,
            trigger=_trig,
        )
        if recovered is not None:
            new_path = recovered
            fresh = nav._world.get_pose()  # noqa: SLF001
            if fresh is not None:
                tick.pose = fresh
    if new_path is not None:
        tick.path = new_path
        ep.on_blocked_replan_ok()
        tick.corridor_yaw = None
        return ReplanOutcome()
    from ..nav.simple_motion import distance_m

    ep.on_blocked_replan_fail(goal_dist_m=distance_m(tick.pose, tick.goal))
    abort = tick.corridor_yaw is None and nav._abort_blocked_replans(  # noqa: SLF001
        ep.blocked_fail_count, nose_clear=nose_clear
    )
    return ReplanOutcome(abort=abort, abort_msg="blocked_replan")


def apply_escape_pre_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
    *,
    trigger: str,
) -> ReplanOutcome:
    """EscapeMachine.tick_pre asked for a replan after finishing a motion."""
    nav._stop_before_replan(trigger)  # noqa: SLF001
    if trigger == "blocked_nose_unstick":
        new_path = nav._try_replan(  # noqa: SLF001
            tick.goal,
            tick.pose,
            tick.path,
            tick.scan,
            failed_count=max(1, ep.failed_replan_while_blocked),
            require_different=ep.failed_replan_while_blocked < 3,
            local_view=tick.local_view,
            trigger=trigger,
        )
    else:
        new_path = nav._try_replan(  # noqa: SLF001
            tick.goal,
            tick.pose,
            tick.path,
            tick.scan,
            failed_count=ep.failed_replan_while_blocked,
            local_view=tick.local_view,
            trigger=trigger,
        )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        tick.path = new_path
        ep.last_progress_at = tick.now
        ep.on_escape_pre_replan_ok(unstick=(trigger == "blocked_nose_unstick"))
        return ReplanOutcome()
    if trigger == "blocked_nose_unstick":
        from ..nav.simple_motion import distance_m

        ep.on_unstick_replan_fail(goal_dist_m=distance_m(tick.pose, tick.goal))
        abort = (
            tick.corridor_yaw is None
            and nav._abort_blocked_replans(  # noqa: SLF001
                ep.blocked_fail_count, nose_clear=tick.nose_clear
            )
        )
        return ReplanOutcome(abort=abort, abort_msg="unstick_replan")
    return ReplanOutcome()


def apply_backup_done_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
) -> ReplanOutcome:
    nav._stop_before_replan("backup_done")  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        tick.goal,
        tick.pose,
        tick.path,
        tick.scan,
        failed_count=ep.failed_replan_while_blocked,
        local_view=tick.local_view,
        trigger="backup_done",
    )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        tick.path = new_path
        ep.last_progress_at = tick.now
        ep.on_backup_done_ok()
    return ReplanOutcome()


def apply_escape_post_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
    *,
    trigger: str,
) -> ReplanOutcome:
    nav._stop_before_replan(trigger)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        tick.goal,
        tick.pose,
        tick.path,
        tick.scan,
        failed_count=ep.failed_replan_while_blocked,
        local_view=tick.local_view,
        trigger=trigger,
    )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        tick.path = new_path
        ep.last_progress_at = tick.now
        ep.on_escape_post_ok()
    return ReplanOutcome()


def apply_periodic_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
    *,
    static_blocked: bool,
    pose_jumped: bool,
    oscillating: bool,
) -> ReplanOutcome:
    """Periodic / pose-jump / osc static replan."""
    _trig = (
        f"periodic static_blocked={static_blocked} "
        f"pose_jumped={pose_jumped} "
        f"oscillating={oscillating}"
    )
    nav._stop_before_replan(_trig)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        tick.goal,
        tick.pose,
        tick.path,
        tick.scan,
        require_different=not static_blocked,
        failed_count=ep.failed_replan_while_blocked if tick.local_blocked else 0,
        local_view=tick.local_view,
        trigger=_trig,
        allow_lift_ban=True,
        force_lift_short_flip=bool(static_blocked),
    )
    ep.note_replan_finished(_mono())
    ep.pending_loc_replan = False
    if new_path is not None:
        tick.path = new_path
        ep.last_progress_at = tick.now
        ep.on_periodic_replan_ok()
        return ReplanOutcome()
    if static_blocked:
        ep.failed_static_replan += 1
        return ReplanOutcome(abort_msg="static_replan_fail")
    ep.last_replan = _mono()
    return ReplanOutcome()


def apply_stall_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    tick: "TickContext",
    *,
    dist_goal: float,
    bearing_err: float,
    error_msg: str,
) -> tuple[ReplanOutcome, bool]:
    """Stall recovery replan. Returns (outcome, caller_should_return)."""
    from .costmap import LETHAL

    if tick.nose_clear and tick.path_ahead_cost < int(LETHAL):
        ep.last_progress_at = tick.now
        return ReplanOutcome(), False
    if (
        not tick.nose_clear
        and bool(tick.progress.get("spin_blocked"))
        and tick.cmd.vx < -1e-6
    ):
        return ReplanOutcome(), False
    _trig = f"stall:{error_msg}"
    nav._stop_before_replan(_trig)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        tick.goal,
        tick.pose,
        tick.path,
        tick.scan,
        failed_count=max(2, ep.failed_replan_while_blocked),
        local_view=tick.local_view,
        trigger=_trig,
    )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        tick.path = new_path
        ep.last_progress_at = tick.now
        ep.last_progress_dist = dist_goal
        ep.last_progress_bearing = bearing_err
        ep.on_stall_replan_ok()
        return ReplanOutcome(), False
    if tick.local_blocked and ep.failed_replan_while_blocked < 5:
        ep.failed_replan_while_blocked += 1
        ep.last_progress_at = tick.now
        return ReplanOutcome(), False
    nav._world.stop()  # noqa: SLF001
    nav._set_status(  # noqa: SLF001
        state="failed",
        active=False,
        error_msg=error_msg,
    )
    return ReplanOutcome(abort=True, abort_msg="stall"), True

"""Typed replan + episode reset packs used by ``run_goal``.

Centralizes ``_try_replan`` call sites so each trigger applies the matching
``GoalEpisode`` reset method instead of copy-pasted counter clears.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .supervisor import NavSupervisor


def _mono() -> float:
    # Use supervisor's ``time`` so the scenario harness FakeClock applies
    # (it monkeypatches ``nav_builtin.supervisor.time``, not stdlib).
    from . import supervisor as _sup

    return _sup.time.monotonic()


@dataclass
class ReplanOutcome:
    path: Path2D
    pose: Pose2D
    # Supervisor should abort the goal (blocked-replan limit / static fail).
    abort: bool = False
    abort_msg: str = ""


def apply_blocked_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    goal: Pose2D,
    pose: Pose2D,
    path: Path2D,
    scan,
    local_view,
    nose_clear: bool,
    path_ahead_cost: int,
    corridor_yaw: Optional[float],
) -> tuple[ReplanOutcome, Optional[float]]:
    """Policy action == replan. Returns (outcome, cleared_corridor_yaw_or_same)."""
    _trig = f"local_blocked cost={path_ahead_cost} nose_clear={nose_clear}"
    nav._stop_before_replan(_trig)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        goal,
        pose,
        path,
        scan,
        failed_count=max(1, ep.failed_replan_while_blocked),
        require_different=ep.failed_replan_while_blocked < 3,
        local_view=local_view,
        trigger=_trig,
    )
    ep.note_replan_finished(_mono())
    if new_path is None:
        recovered = nav._recover_unreachable_start(  # noqa: SLF001
            goal,
            pose,
            path,
            scan,
            failed_count=max(1, ep.failed_replan_while_blocked),
            local_view=local_view,
            trigger=_trig,
        )
        if recovered is not None:
            new_path = recovered
            fresh = nav._world.get_pose()  # noqa: SLF001
            if fresh is not None:
                pose = fresh
    out_yaw = corridor_yaw
    if new_path is not None:
        path = new_path
        ep.on_blocked_replan_ok()
        out_yaw = None
        return ReplanOutcome(path=path, pose=pose), out_yaw
    from ..nav.simple_motion import distance_m

    ep.on_blocked_replan_fail(goal_dist_m=distance_m(pose, goal))
    abort = corridor_yaw is None and nav._abort_blocked_replans(  # noqa: SLF001
        ep.blocked_fail_count, nose_clear=nose_clear
    )
    return (
        ReplanOutcome(path=path, pose=pose, abort=abort, abort_msg="blocked_replan"),
        out_yaw,
    )


def apply_escape_pre_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    goal: Pose2D,
    pose: Pose2D,
    path: Path2D,
    scan,
    local_view,
    trigger: str,
    nose_clear: bool,
    corridor_yaw: Optional[float],
    now: float,
) -> ReplanOutcome:
    """EscapeMachine.tick_pre asked for a replan after finishing a motion."""
    nav._stop_before_replan(trigger)  # noqa: SLF001
    if trigger == "blocked_nose_unstick":
        new_path = nav._try_replan(  # noqa: SLF001
            goal,
            pose,
            path,
            scan,
            failed_count=max(1, ep.failed_replan_while_blocked),
            require_different=ep.failed_replan_while_blocked < 3,
            local_view=local_view,
            trigger=trigger,
        )
    else:
        new_path = nav._try_replan(  # noqa: SLF001
            goal,
            pose,
            path,
            scan,
            failed_count=ep.failed_replan_while_blocked,
            local_view=local_view,
            trigger=trigger,
        )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        path = new_path
        ep.last_progress_at = now
        ep.on_escape_pre_replan_ok(unstick=(trigger == "blocked_nose_unstick"))
        return ReplanOutcome(path=path, pose=pose)
    if trigger == "blocked_nose_unstick":
        from ..nav.simple_motion import distance_m

        ep.on_unstick_replan_fail(goal_dist_m=distance_m(pose, goal))
        abort = corridor_yaw is None and nav._abort_blocked_replans(  # noqa: SLF001
            ep.blocked_fail_count, nose_clear=nose_clear
        )
        return ReplanOutcome(
            path=path, pose=pose, abort=abort, abort_msg="unstick_replan"
        )
    return ReplanOutcome(path=path, pose=pose)


def apply_backup_done_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    goal: Pose2D,
    pose: Pose2D,
    path: Path2D,
    scan,
    local_view,
    now: float,
) -> ReplanOutcome:
    nav._stop_before_replan("backup_done")  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        goal,
        pose,
        path,
        scan,
        failed_count=ep.failed_replan_while_blocked,
        local_view=local_view,
        trigger="backup_done",
    )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        path = new_path
        ep.last_progress_at = now
        ep.on_backup_done_ok()
    return ReplanOutcome(path=path, pose=pose)


def apply_escape_post_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    goal: Pose2D,
    pose: Pose2D,
    path: Path2D,
    scan,
    local_view,
    trigger: str,
    now: float,
) -> ReplanOutcome:
    nav._stop_before_replan(trigger)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        goal,
        pose,
        path,
        scan,
        failed_count=ep.failed_replan_while_blocked,
        local_view=local_view,
        trigger=trigger,
    )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        path = new_path
        ep.last_progress_at = now
        ep.on_escape_post_ok()
    return ReplanOutcome(path=path, pose=pose)


def apply_periodic_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    goal: Pose2D,
    pose: Pose2D,
    path: Path2D,
    scan,
    local_view,
    local_blocked: bool,
    static_blocked: bool,
    pose_jumped: bool,
    oscillating: bool,
    now: float,
) -> ReplanOutcome:
    """Periodic / pose-jump / osc static replan."""
    _trig = (
        f"periodic static_blocked={static_blocked} "
        f"pose_jumped={pose_jumped} "
        f"oscillating={oscillating}"
    )
    nav._stop_before_replan(_trig)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        goal,
        pose,
        path,
        scan,
        require_different=not static_blocked,
        failed_count=ep.failed_replan_while_blocked if local_blocked else 0,
        local_view=local_view,
        trigger=_trig,
        allow_lift_ban=True,
        force_lift_short_flip=bool(static_blocked),
    )
    ep.note_replan_finished(_mono())
    ep.pending_loc_replan = False
    if new_path is not None:
        path = new_path
        ep.last_progress_at = now
        ep.on_periodic_replan_ok()
        return ReplanOutcome(path=path, pose=pose)
    if static_blocked:
        ep.failed_static_replan += 1
        return ReplanOutcome(path=path, pose=pose, abort_msg="static_replan_fail")
    ep.last_replan = _mono()
    return ReplanOutcome(path=path, pose=pose)


def apply_stall_replan(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    goal: Pose2D,
    pose: Pose2D,
    path: Path2D,
    scan,
    local_view,
    local_blocked: bool,
    nose_clear: bool,
    path_ahead_cost: int,
    dist_goal: float,
    bearing_err: float,
    error_msg: str,
    now: float,
    progress_spin_blocked: bool,
    cmd_vx: float,
) -> tuple[ReplanOutcome, bool]:
    """Stall recovery replan. Returns (outcome, caller_should_return)."""
    from .costmap import LETHAL

    if nose_clear and path_ahead_cost < int(LETHAL):
        ep.last_progress_at = now
        return ReplanOutcome(path=path, pose=pose), False
    if (
        not nose_clear
        and progress_spin_blocked
        and cmd_vx < -1e-6
    ):
        return ReplanOutcome(path=path, pose=pose), False
    _trig = f"stall:{error_msg}"
    nav._stop_before_replan(_trig)  # noqa: SLF001
    new_path = nav._try_replan(  # noqa: SLF001
        goal,
        pose,
        path,
        scan,
        failed_count=max(2, ep.failed_replan_while_blocked),
        local_view=local_view,
        trigger=_trig,
    )
    ep.note_replan_finished(_mono())
    if new_path is not None:
        path = new_path
        ep.last_progress_at = now
        ep.last_progress_dist = dist_goal
        ep.last_progress_bearing = bearing_err
        ep.on_stall_replan_ok()
        return ReplanOutcome(path=path, pose=pose), False
    if local_blocked and ep.failed_replan_while_blocked < 5:
        ep.failed_replan_while_blocked += 1
        ep.last_progress_at = now
        return ReplanOutcome(path=path, pose=pose), False
    nav._world.stop()  # noqa: SLF001
    nav._set_status(  # noqa: SLF001
        state="failed",
        active=False,
        error_msg=error_msg,
    )
    return ReplanOutcome(path=path, pose=pose, abort=True, abort_msg="stall"), True

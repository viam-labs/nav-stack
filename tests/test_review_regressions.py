"""Regressions from the adversarial review of PR #68 (2026-10-08).

Each test pins a behaviour the review found wrong in the first cut of the
recovery policies; see docs/investigation-2026-10-07-localization.md.
"""
from __future__ import annotations

import time

import pytest

import src.nav_builtin.supervisor as sup_mod
from src.nav_builtin.planner import connect_plan_start
from src.nav_builtin.types import Path2D, PlanResult, Pose2D
from tests.test_recovery_policies import _sup


def _path(*pts):
    return Path2D(points=tuple(pts), goal_theta=0.0)


# --- goal-blocked finish ----------------------------------------------------


def test_plain_no_feasible_path_is_not_goal_blocked():
    """The blocked-corridor paint seals the robot's own corridor by design and
    the planner then says "no feasible path"; that is not a goal verdict."""
    sup, _ = _sup()
    sup._last_replan_error = "scan+local: same route (1.0 m); blocked-corridor: no feasible path"
    assert sup._replan_says_goal_blocked() is False
    sup._last_replan_error = "scan+local: goal snap 0.80 m exceeds max_goal_snap_m=0.50 (goal blocked / over-inflated)"
    assert sup._replan_says_goal_blocked() is True
    sup._last_replan_error = "scan+local: goal pose is in lethal / unknown space"
    assert sup._replan_says_goal_blocked() is True


def test_paint_attempt_no_feasible_path_does_not_read_as_goal_blocked(monkeypatch):
    sup, _ = _sup(replan_budget_s=100.0)
    path = _path((1.0, 1.0), (2.0, 1.0))

    def fake_plan(goal, start=None, scan=None, *, blocked_path=None,
                  blocked_path_pose=None, local_view=None, paint_corridor=True):
        if paint_corridor:
            return PlanResult(feasible=False, error_code=3, error_msg="no feasible path")
        return PlanResult(feasible=True, error_code=0, error_msg="", path=path)

    monkeypatch.setattr(sup, "plan", fake_plan)
    out = sup._try_replan(Pose2D(2.0, 1.0, 0.0), Pose2D(1.0, 1.0, 0.0), path, None,
                          failed_count=1, require_different=True)
    assert out is None
    assert "no feasible path" in sup._last_replan_error
    assert sup._replan_says_goal_blocked() is False


def test_goal_blocked_timer_needs_a_fresh_replan():
    """A stale verdict from a block minutes ago must not pre-arm the timer."""
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=15.0)
    sup._last_replan_error = "scan+local: goal pose is in lethal / unknown space"
    sup._last_replan_at = time.monotonic() - 60.0
    goal = Pose2D(0.0, 0.0, 0.0)
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0, 0), goal, 100.0, pose_cost=0) is False
    assert sup._goal_blocked_since is None


def test_goal_blocked_timer_rearms_after_the_robot_moves_on():
    """Timer armed on a transient block; the robot drives 1 m; a later block
    gets the full grace period again instead of finishing at once."""
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=15.0)
    sup._last_replan_error = "scan+local: goal pose is in lethal / unknown space"
    goal = Pose2D(0.0, 0.0, 0.0)
    sup._last_replan_at = time.monotonic()
    assert sup._maybe_finish_goal_blocked(Pose2D(2.0, 0, 0), goal, 100.0, pose_cost=0) is False
    assert sup._goal_blocked_since == 100.0
    # ... drives on, blocks again 1 m further along, replan says blocked again.
    sup._last_replan_at = time.monotonic()
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0, 0), goal, 200.0, pose_cost=0) is False
    assert sup._goal_blocked_since == 200.0  # re-armed, not finished
    assert sup.status().to_dict()["state"] != "succeeded"
    # Same episode, same spot, grace elapsed: now it finishes.
    sup._last_replan_at = time.monotonic()
    assert sup._maybe_finish_goal_blocked(Pose2D(1.05, 0, 0), goal, 216.0, pose_cost=0) is True
    st = sup.status().to_dict()
    assert st["state"] == "succeeded" and st["goal_blocked"] is True


# --- cancel semantics ---------------------------------------------------------


@pytest.mark.parametrize(
    "code,msg",
    [(9, "planning aborted"), (8, "cannot reach plan start from current pose")],
)
def test_cancel_during_initial_plan_reports_canceled_not_failed(code, msg):
    sup, _ = _sup()

    def fake_plan(goal, *a, **k):
        sup.request_cancel()  # cancel lands while the plan is running
        return PlanResult(feasible=False, error_code=code, error_msg=msg)

    sup.plan = fake_plan
    sup.run_goal(Pose2D(5.0, 5.0, 0.0))
    st = sup.status().to_dict()
    assert st["state"] == "canceled"
    assert st["error_msg"] == "canceled"
    assert st["active"] is False


def test_failed_status_is_canceled_once_cancel_is_set():
    sup, _ = _sup()
    sup.request_cancel()
    sup._set_status(state="failed", active=False, error_msg="replan failed (path blocked)")
    st = sup.status().to_dict()
    assert st["state"] == "canceled" and st["error_msg"] == "canceled"


def test_connect_plan_start_keeps_abort_verdict(monkeypatch):
    """An aborted bridge plan must not be relabelled 'cannot reach plan start'."""
    import src.nav_builtin.planner as planner_mod

    aborted = PlanResult(feasible=False, error_code=9, error_msg="planning aborted")
    monkeypatch.setattr(planner_mod, "plan_path", lambda *a, **k: aborted)
    route = PlanResult(
        feasible=True, error_code=0, error_msg="", path=_path((5.0, 5.0), (6.0, 5.0))
    )
    from tests.test_nav_builtin import _empty_map

    out = connect_plan_start(
        _empty_map(),
        Pose2D(1.0, 1.0, 0.0),
        route,
        inflation_radius_m=0.3,
        robot_radius_m=0.3,
        should_abort=lambda: True,
    )
    assert out.feasible is False and out.error_code == 9


# --- localization yield -------------------------------------------------------


def test_yield_pause_returns_at_once_on_cancel():
    sup, world = _sup(loc_yield_enabled=True, loc_yield_after_s=0.0, loc_yield_wait_s=100.0)

    def check(w, **k):
        sup.request_cancel()  # cancel arrives during the first attempt
        return {"status": "awaiting_confirm", "corrected": False}

    world.on_loc_check = check
    t0 = time.monotonic()
    out = sup._maybe_yield_for_localization(
        Pose2D(5, 1, 0), Pose2D(1, 1, 0), _path((1.0, 1.0), (5.0, 1.0)), None, None,
        pose_cost=254, stuck_s=10.0, trigger="t",
    )
    assert out is None
    assert world.loc_checks == 1
    assert time.monotonic() - t0 < 1.0  # no 2 s sleep behind a cancel


def test_loc_yield_is_off_by_default():
    sup, world = _sup(loc_yield_after_s=0.0)
    out = sup._maybe_yield_for_localization(
        Pose2D(5, 1, 0), Pose2D(1, 1, 0), _path((1.0, 1.0), (5.0, 1.0)), None, None,
        pose_cost=254, stuck_s=10.0, trigger="t",
    )
    assert out is None and world.loc_checks == 0


# --- replan budget ------------------------------------------------------------


def test_lift_ban_recursion_shares_the_replan_budget(monkeypatch):
    """With the budget spent, lifting the detour ban must not start a fresh
    budget and run another escalation chain."""
    sup, _ = _sup(replan_budget_s=0.0)
    path = _path((1.0, 1.0), (2.0, 1.0))
    sup._detour_ban_path = _path((1.0, 1.0), (1.5, 1.2), (2.0, 1.0))
    sup._detour_min_length_m = 0.0
    calls: list = []

    def fake_plan(goal, start=None, scan=None, **k):
        calls.append(bool(k.get("paint_corridor")))
        return PlanResult(feasible=False, error_code=3, error_msg="no feasible path")

    monkeypatch.setattr(sup, "plan", fake_plan)
    out = sup._try_replan(Pose2D(2.0, 1.0, 0.0), Pose2D(1.0, 1.0, 0.0), path, None,
                          failed_count=1, local_view=None)
    assert out is None
    # First attempt always runs; after the ban lift only one more unbudgeted
    # first attempt is allowed, never the full chain.
    assert len(calls) <= 2
    assert "budget" in sup._last_replan_error

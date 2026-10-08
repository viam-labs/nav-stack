"""Recovery policies added after the 2026-10-07 tracer2a live session.

* cancel interrupts a replan storm (planner abort hook, replan budget)
* goal blocked → finish ``succeeded`` + ``goal_blocked`` nearby
* localization yield → nav pauses so SLAM may apply a large correction
"""
from __future__ import annotations

import time

import pytest

from src import runtime
from src.nav_builtin import planner as planner_mod
from src.nav_builtin import supervisor as sup_mod
from src.nav_builtin.costmap import LETHAL
from src.nav_builtin.planner import plan_path
from src.nav_builtin.supervisor import NavSupervisor
from src.nav_builtin.types import Path2D, PlanResult, Pose2D
from tests.test_nav_builtin import _FakeWorld, _empty_map


def _sup(**kw):
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map())
    return NavSupervisor(world, **kw), world


def _path(*pts):
    return Path2D(points=tuple(pts), goal_theta=0.0)


# -- cancel interrupts replanning ---------------------------------------------


def test_plan_path_abort_hook_ends_search_early(monkeypatch):
    monkeypatch.setattr(planner_mod, "_ABORT_POLL_EVERY", 1)
    big = _empty_map(size=200, resolution=0.05)
    res = plan_path(
        big,
        Pose2D(0.5, 0.5, 0.0),
        Pose2D(9.5, 9.5, 0.0),
        inflation_radius_m=0.3,
        robot_radius_m=0.2,
        should_abort=lambda: True,
    )
    assert not res.feasible
    assert res.error_code == 9
    assert "aborted" in res.error_msg
    # Same plan without the hook is fine.
    ok = plan_path(
        big,
        Pose2D(0.5, 0.5, 0.0),
        Pose2D(9.5, 9.5, 0.0),
        inflation_radius_m=0.3,
        robot_radius_m=0.2,
    )
    assert ok.feasible


def test_try_replan_bails_before_planning_when_cancel_requested(monkeypatch):
    sup, _ = _sup()
    calls: list = []
    monkeypatch.setattr(
        sup,
        "plan",
        lambda *a, **k: calls.append(k) or PlanResult(feasible=False, error_code=3, error_msg="x"),
    )
    sup.request_cancel()
    out = sup._try_replan(  # noqa: SLF001
        Pose2D(2.0, 1.0, 0.0), Pose2D(1.0, 1.0, 0.0), _path((1.0, 1.0), (2.0, 1.0)), None
    )
    assert out is None
    assert calls == []
    assert "cancel" in sup._last_replan_error  # noqa: SLF001


def test_replan_budget_stops_escalation_after_first_attempt(monkeypatch):
    sup, _ = _sup(replan_budget_s=0.0)
    paints: list = []

    def fake_plan(*a, **k):
        paints.append(bool(k.get("paint_corridor")))
        return PlanResult(feasible=False, error_code=3, error_msg="no feasible path")

    monkeypatch.setattr(sup, "plan", fake_plan)
    out = sup._try_replan(  # noqa: SLF001
        Pose2D(2.0, 1.0, 0.0),
        Pose2D(1.0, 1.0, 0.0),
        _path((1.0, 1.0), (2.0, 1.0)),
        None,
        failed_count=1,  # would normally add the blocked-corridor attempt
        local_view=object(),  # would normally trigger forced side vias
    )
    assert out is None
    assert paints == [False]  # first attempt only
    err = sup._last_replan_error  # noqa: SLF001
    assert "budget" in err
    assert "forced-via: skipped" in err


# -- goal blocked → arrived nearby ----------------------------------------------


def test_goal_blocked_finishes_nearby_after_grace():
    sup, world = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=10.0)
    sup._last_replan_error = (  # noqa: SLF001
        "scan+local: goal snap 0.57 m exceeds max_goal_snap_m=0.50 (goal blocked / over-inflated)"
    )
    pose, goal = Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0)
    sup._last_replan_at = time.monotonic()  # verdict is from this episode  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(pose, goal, 100.0, pose_cost=0) is False  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(pose, goal, 105.0, pose_cost=0) is False  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(pose, goal, 111.0, pose_cost=0) is True  # noqa: SLF001
    st = sup.status().to_dict()
    assert st["state"] == "succeeded"
    assert st["active"] is False
    assert st["goal_blocked"] is True
    assert st["goal_offset_m"] == pytest.approx(1.0, abs=0.01)
    assert world.stop_calls >= 1


def test_goal_blocked_not_applied_when_far_or_start_lethal_or_other_error():
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=0.0)
    goal = Pose2D(0.0, 0.0, 0.0)
    sup._last_replan_error = "scan+local: no feasible path"  # noqa: SLF001
    # Too far from the goal.
    assert sup._maybe_finish_goal_blocked(Pose2D(3.0, 0.0, 0.0), goal, 1.0, pose_cost=0) is False  # noqa: SLF001
    # Start cell lethal: that is a localization problem, not a blocked goal.
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0.0, 0.0), goal, 1.0, pose_cost=int(LETHAL)) is False  # noqa: SLF001
    # Unrelated planner error.
    sup._last_replan_error = "cannot reach plan start from current pose"  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0.0, 0.0), goal, 1.0, pose_cost=0) is False  # noqa: SLF001
    assert sup.status().to_dict()["state"] != "succeeded"


def test_goal_blocked_disabled_with_zero_radius():
    sup, _ = _sup(goal_blocked_accept_m=0.0, goal_blocked_after_s=0.0)
    sup._last_replan_error = "scan+local: no feasible path"  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(Pose2D(0.5, 0, 0), Pose2D(0, 0, 0), 1.0, pose_cost=0) is False  # noqa: SLF001


# -- localization yield -----------------------------------------------------------


def test_any_navigation_active_ignores_yielding_host():
    class Host:
        def __init__(self, st):
            self.st = st

        def nav_status(self):
            return self.st

    name = "test-yield-host"
    try:
        runtime.register_nav_host(name, Host({"active": True, "localization_yield": True}))
        assert runtime.any_navigation_active() is False
        runtime.register_nav_host(name, Host({"active": True, "localization_yield": False}))
        assert runtime.any_navigation_active() is True
    finally:
        runtime.unregister_nav_host(name)


def test_loc_yield_pauses_goal_and_replans_from_corrected_pose(monkeypatch):
    sup, world = _sup(loc_yield_enabled=True, loc_yield_after_s=1.0, loc_yield_wait_s=10.0)
    monkeypatch.setattr(sup_mod.time, "sleep", lambda s: None)
    seen_flags: list = []

    def on_check(w, **kwargs):
        seen_flags.append(
            (kwargs.get("allow_during_navigation"), sup.status().to_dict()["localization_yield"])
        )
        if len(seen_flags) == 1:
            # Idle-mode large jump: first match only arms the confirm.
            return {"status": "awaiting_confirm", "corrected": False, "shift_m": 1.3}
        w.pose = Pose2D(2.3, 1.0, 0.0)
        return {"status": "ok", "corrected": True, "shift_m": 1.3, "score": 0.54}

    world.on_loc_check = on_check
    sentinel = _path((2.3, 1.0), (5.0, 1.0))
    seen_replan: dict = {}

    def fake_replan(goal, pose, path, scan, **kw):
        seen_replan["pose"] = pose
        seen_replan["kw"] = kw
        return sentinel

    monkeypatch.setattr(sup, "_try_replan", fake_replan)
    out = sup._maybe_yield_for_localization(  # noqa: SLF001
        Pose2D(5.0, 1.0, 0.0),
        Pose2D(1.0, 1.0, 0.0),
        _path((1.0, 1.0), (5.0, 1.0)),
        None,
        None,
        pose_cost=int(LETHAL),
        stuck_s=5.0,
        trigger="test",
    )
    assert out is sentinel
    assert world.loc_checks == 2
    # SLAM was asked in idle mode and saw nav as yielding both times.
    assert all(allow is False for allow, _ in seen_flags)
    assert all(flag is True for _, flag in seen_flags)
    # Flag is cleared afterwards; replan started from the corrected pose and
    # may keep the same corridor.
    assert sup.status().to_dict()["localization_yield"] is False
    assert seen_replan["pose"].x == pytest.approx(2.3)
    assert seen_replan["kw"]["require_different"] is False
    assert world.cmds and world.cmds[-1] == (0.0, 0.0, 0.0)  # stopped before yielding
    info = sup._loc_yield_info  # noqa: SLF001
    assert info["corrected"] is True and len(info["attempts"]) == 2


def test_loc_yield_requires_lethal_start_and_stuck_time():
    sup, world = _sup(loc_yield_enabled=True, loc_yield_after_s=8.0)
    args = (Pose2D(5, 1, 0), Pose2D(1, 1, 0), _path((1.0, 1.0), (5.0, 1.0)), None, None)
    # Free start cell: not a localization deadlock.
    assert sup._maybe_yield_for_localization(*args, pose_cost=0, stuck_s=30.0, trigger="t") is None  # noqa: SLF001
    # Lethal start but not stuck long enough.
    assert sup._maybe_yield_for_localization(*args, pose_cost=int(LETHAL), stuck_s=2.0, trigger="t") is None  # noqa: SLF001
    assert world.loc_checks == 0
    assert sup.status().to_dict()["localization_yield"] is False


def test_loc_yield_respects_per_goal_cap_and_cooldown(monkeypatch):
    sup, world = _sup(loc_yield_enabled=True, loc_yield_after_s=0.0, loc_yield_max_per_goal=1, loc_yield_cooldown_s=1000.0)
    monkeypatch.setattr(sup_mod.time, "sleep", lambda s: None)
    world.on_loc_check = lambda w, **k: {"status": "low_quality", "corrected": False}
    args = (Pose2D(5, 1, 0), Pose2D(1, 1, 0), _path((1.0, 1.0), (5.0, 1.0)), None, None)
    assert sup._maybe_yield_for_localization(*args, pose_cost=int(LETHAL), stuck_s=10.0, trigger="t") is None  # noqa: SLF001
    first = world.loc_checks
    assert first >= 1
    # Second call in the same goal: capped (and on cooldown) → no SLAM call.
    assert sup._maybe_yield_for_localization(*args, pose_cost=int(LETHAL), stuck_s=10.0, trigger="t") is None  # noqa: SLF001
    assert world.loc_checks == first

"""Cancel latency: a cancel must not wait behind a replan storm.

Live (tracer2a, 2026-10-07): a ``cancel`` took more than 10 s to register
because the control thread was inside back-to-back full-map replans (84
planning ticks). The planner now polls a ``should_abort`` hook inside its
search loops, and the supervisor checks the cancel event before every
planning / escalation step and reports the result as ``canceled``.
"""
from __future__ import annotations

import pytest

from src.nav_builtin import planner as planner_mod
from src.nav_builtin.costmap import build_costmap
from src.nav_builtin.planner import (
    PLANNER_ASTAR,
    PLANNER_LAZY_THETA,
    _astar,
    _lazy_theta_star,
    connect_plan_start,
    plan_on_costmap,
    plan_path,
)
from src.nav_builtin.supervisor import NavSupervisor
from src.nav_builtin.types import OccupancyGrid, Path2D, PlanResult, Pose2D
from tests.test_nav_builtin import _FakeWorld, _empty_map


def _sup(**kw):
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map())
    return NavSupervisor(world, **kw), world


def _path(*pts):
    return Path2D(points=tuple(pts), goal_theta=0.0)


def _costmap(size: int = 40, resolution: float = 0.05):
    m = _empty_map(size=size, resolution=resolution)
    occ = OccupancyGrid(
        grid=m["grid"], resolution=resolution, origin_x=0.0, origin_y=0.0
    )
    costs = build_costmap(
        occ, inflation_radius_m=0.25, robot_radius_m=0.2, cost_scaling_factor=4.0
    )
    return occ, costs


# -- planner: should_abort hook -------------------------------------------------


@pytest.mark.parametrize("search", [_astar, _lazy_theta_star], ids=["astar", "theta"])
def test_search_returns_no_cells_when_abort_fires(monkeypatch, search):
    monkeypatch.setattr(planner_mod, "_ABORT_POLL_EVERY", 1)
    occ, costs = _costmap()
    start = occ.world_to_cell(0.5, 0.5)
    goal = occ.world_to_cell(1.5, 1.5)
    plain = search(costs, start, goal)
    assert plain  # sanity: the map is solvable
    assert search(costs, start, goal, should_abort=lambda: True) is None
    # A hook that never fires changes nothing.
    assert search(costs, start, goal, should_abort=lambda: False) == plain


def test_search_polls_the_hook_during_expansion(monkeypatch):
    """The hook is polled as the search runs, not only once up front."""
    monkeypatch.setattr(planner_mod, "_ABORT_POLL_EVERY", 1)
    occ, costs = _costmap()
    start = occ.world_to_cell(0.5, 0.5)
    goal = occ.world_to_cell(1.5, 1.5)
    polls: list = []
    assert _astar(costs, start, goal, should_abort=lambda: polls.append(1) or len(polls) >= 4) is None
    assert len(polls) == 4


@pytest.mark.parametrize("algorithm", [PLANNER_ASTAR, PLANNER_LAZY_THETA])
def test_plan_on_costmap_reports_code_9_on_abort(monkeypatch, algorithm):
    monkeypatch.setattr(planner_mod, "_ABORT_POLL_EVERY", 1)
    occ, costs = _costmap()
    start, goal = Pose2D(0.5, 0.5, 0.0), Pose2D(1.5, 1.5, 0.0)
    kw = dict(robot_radius_m=0.2, algorithm=algorithm)
    aborted = plan_on_costmap(occ, costs, start, goal, should_abort=lambda: True, **kw)
    assert not aborted.feasible
    assert aborted.error_code == 9
    assert aborted.error_msg == "planning aborted"
    assert aborted.path.empty
    # Identical result with a hook that never fires and with no hook at all.
    base = plan_on_costmap(occ, costs, start, goal, **kw)
    same = plan_on_costmap(occ, costs, start, goal, should_abort=lambda: False, **kw)
    assert base.feasible and same.feasible
    assert same.path.points == base.path.points
    assert same.error_code == base.error_code == 0


def test_plan_path_abort_hook_ends_search_early(monkeypatch):
    monkeypatch.setattr(planner_mod, "_ABORT_POLL_EVERY", 1)
    big = _empty_map(size=200, resolution=0.05)
    kw = dict(inflation_radius_m=0.3, robot_radius_m=0.2)
    res = plan_path(big, Pose2D(0.5, 0.5, 0.0), Pose2D(9.5, 9.5, 0.0), should_abort=lambda: True, **kw)
    assert not res.feasible
    assert res.error_code == 9
    assert "aborted" in res.error_msg
    ok = plan_path(big, Pose2D(0.5, 0.5, 0.0), Pose2D(9.5, 9.5, 0.0), **kw)
    assert ok.feasible


def test_connect_plan_start_keeps_aborted_bridge_code(monkeypatch):
    """An aborted bridge plan stays code 9, not 'cannot reach plan start' (8)."""
    m = _empty_map(size=60, resolution=0.05)
    m["grid"][30, 30] = 100
    start = Pose2D(1.52, 1.52, 0.0)  # inside the pillar's inflation
    goal = Pose2D(2.5, 1.52, 0.0)
    main = plan_path(m, start, goal, inflation_radius_m=0.25, robot_radius_m=0.22)
    assert main.feasible
    kw = dict(inflation_radius_m=0.25, robot_radius_m=0.22)
    ok = connect_plan_start(m, start, main, should_abort=lambda: False, **kw)
    assert ok.feasible
    assert len(ok.path.points) >= len(main.path.points)  # a bridge was planned
    monkeypatch.setattr(planner_mod, "_ABORT_POLL_EVERY", 1)
    aborted = connect_plan_start(m, start, main, should_abort=lambda: True, **kw)
    assert not aborted.feasible
    assert aborted.error_code == 9
    assert aborted.error_msg == "planning aborted"


# -- supervisor: cancel short-circuits replanning ---------------------------------


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
    assert sup._last_replan_error == "scan+local: skipped (cancel requested)"  # noqa: SLF001
    assert sup._last_replan_info["accepted"] is None  # noqa: SLF001


def test_try_replan_skips_forced_via_when_cancel_requested(monkeypatch):
    sup, _ = _sup()
    monkeypatch.setattr(sup, "plan", lambda *a, **k: pytest.fail("plan must not run"))
    monkeypatch.setattr(
        sup, "_forced_side_detour", lambda *a, **k: pytest.fail("forced via must not run")
    )
    sup.request_cancel()
    out = sup._try_replan(  # noqa: SLF001
        Pose2D(2.0, 1.0, 0.0),
        Pose2D(1.0, 1.0, 0.0),
        _path((1.0, 1.0), (2.0, 1.0)),
        None,
        failed_count=1,  # would normally add the blocked-corridor attempt
        local_view=object(),  # would normally trigger forced side vias
    )
    assert out is None
    assert sup._last_replan_error == (  # noqa: SLF001
        "scan+local: skipped (cancel requested); forced-via: skipped (cancel requested)"
    )


def test_try_replan_does_not_lift_detour_ban_when_cancel_requested(monkeypatch):
    sup, _ = _sup()
    ban = _path((1.0, 1.0), (3.0, 1.0))
    sup._detour_ban_path = ban  # noqa: SLF001
    sup._detour_min_length_m = 2.0  # noqa: SLF001
    monkeypatch.setattr(sup, "plan", lambda *a, **k: pytest.fail("plan must not run"))
    sup.request_cancel()
    out = sup._try_replan(  # noqa: SLF001
        Pose2D(3.0, 1.0, 0.0), Pose2D(1.0, 1.0, 0.0), _path((1.0, 1.0), (3.0, 1.0)), None
    )
    assert out is None
    assert sup._detour_ban_path is ban  # noqa: SLF001
    assert "detour-ban: lifting" not in sup._last_replan_error  # noqa: SLF001


def test_forced_side_detour_stops_on_cancel(monkeypatch):
    sup, _ = _sup()
    calls: list = []

    def plan_then_cancel(*a, **k):
        calls.append(k)
        sup.request_cancel()
        return PlanResult(feasible=False, error_code=9, error_msg="planning aborted")

    monkeypatch.setattr(sup, "plan", plan_then_cancel)
    path = _path((1.0, 1.0), (2.0, 1.0), (3.0, 1.0))
    out = sup._forced_side_detour(Pose2D(3.0, 1.0, 0.0), Pose2D(1.0, 1.0, 0.0), path, None, None)  # noqa: SLF001
    assert out is None
    assert len(calls) == 1  # the first via's plan set cancel; no further vias tried
    # Already canceled: no via is planned at all.
    calls.clear()
    out = sup._forced_side_detour(Pose2D(3.0, 1.0, 0.0), Pose2D(1.0, 1.0, 0.0), path, None, None)  # noqa: SLF001
    assert out is None
    assert calls == []


def test_recover_unreachable_start_returns_none_on_cancel(monkeypatch):
    sup, _ = _sup()
    sup._last_replan_error = "scan+local: cannot reach plan start from current pose"  # noqa: SLF001
    assert sup._replan_blocked_at_start()  # noqa: SLF001 - would otherwise recover
    monkeypatch.setattr(sup, "_refine_stuck_pose", lambda: pytest.fail("must not refine"))
    monkeypatch.setattr(sup, "_try_replan", lambda *a, **k: pytest.fail("must not replan"))
    sup.request_cancel()
    out = sup._recover_unreachable_start(  # noqa: SLF001
        Pose2D(3.0, 1.0, 0.0),
        Pose2D(1.0, 1.0, 0.0),
        _path((1.0, 1.0), (3.0, 1.0)),
        None,
        failed_count=1,
        local_view=None,
        trigger="t",
    )
    assert out is None
    assert sup._stuck_pose_refine_used is False  # noqa: SLF001 - attempt not consumed


# -- supervisor: a cancel during planning ends ``canceled``, not ``failed`` ----


@pytest.mark.parametrize(
    "code,msg",
    [(9, "planning aborted"), (8, "cannot reach plan start from current pose")],
)
def test_run_goal_cancel_during_initial_plan_ends_canceled(monkeypatch, code, msg):
    sup, world = _sup()

    def plan_then_cancel(goal, *a, **k):
        sup.request_cancel()
        return PlanResult(feasible=False, error_code=code, error_msg=msg)

    monkeypatch.setattr(sup, "plan", plan_then_cancel)
    sup.run_goal(Pose2D(3.0, 1.0, 0.0))
    st = sup.status().to_dict()
    assert st["state"] == "canceled"
    assert st["error_msg"] == "canceled"
    assert st["active"] is False


def test_set_status_failed_becomes_canceled_once_cancel_is_set():
    sup, _ = _sup()
    sup._set_status(state="failed", active=False, error_msg="no feasible path")  # noqa: SLF001
    assert sup.status().to_dict()["state"] == "failed"
    sup.request_cancel()
    sup._set_status(state="failed", active=False, error_msg="no feasible path")  # noqa: SLF001
    st = sup.status().to_dict()
    assert st["state"] == "canceled"
    assert st["error_msg"] == "canceled"
    assert st["active"] is False
    # Only ``failed`` is rewritten; other states pass through unchanged.
    sup._set_status(state="succeeded", error_msg="")  # noqa: SLF001
    assert sup.status().to_dict()["state"] == "succeeded"

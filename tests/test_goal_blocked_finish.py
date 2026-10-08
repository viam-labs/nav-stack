"""Goal-blocked finish ("arrived nearby"), opt-in via ``goal_blocked_accept_m``.

Live motivation (tracer2a, 2026-10-07/08): a chair on the goal made every
replan fail for three minutes. With the policy on, the goal finished
``succeeded`` with ``goal_blocked: true`` at 2.4 m after 56 s, and a desk
whose goal cell sits inside inflation finished at 0.98 m.
"""
from __future__ import annotations

import time

import pytest

from src.config import NavConfig
from src.nav_builtin.costmap import LETHAL
from src.nav_builtin.runtime_kwargs import builtin_nav_runtime_kwargs
from src.nav_builtin.supervisor import NavSupervisor
from src.nav_builtin.types import NavStatus, Path2D, PlanResult, Pose2D
from tests.test_nav_builtin import _FakeWorld, _empty_map

_GOAL_BLOCKED_SNAP = (
    "scan+local: goal snap 0.57 m exceeds max_goal_snap_m=0.50 (goal blocked / over-inflated)"
)
_GOAL_IN_LETHAL = "scan+local: goal pose is in lethal / unknown space"


def _sup(**kw):
    world = _FakeWorld(Pose2D(1.0, 1.0, 0.0), _empty_map())
    return NavSupervisor(world, **kw), world


def _path(*pts):
    return Path2D(points=tuple(pts), goal_theta=0.0)


# --- config → kwargs → supervisor plumbing ------------------------------------


def test_goal_blocked_is_off_by_default_and_plumbed_through_config():
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    assert cfg.builtin.goal_blocked_accept_m == pytest.approx(0.0)
    assert cfg.builtin.goal_blocked_after_s == pytest.approx(15.0)
    kw = builtin_nav_runtime_kwargs(cfg)
    assert kw["goal_blocked_accept_m"] == pytest.approx(0.0)
    assert kw["goal_blocked_after_s"] == pytest.approx(15.0)

    tuned = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "builtin": {"goal_blocked_accept_m": 2.5, "goal_blocked_after_s": 20.0},
        }
    )
    kw = builtin_nav_runtime_kwargs(tuned)
    assert kw["goal_blocked_accept_m"] == pytest.approx(2.5)
    assert kw["goal_blocked_after_s"] == pytest.approx(20.0)
    sup = NavSupervisor(_FakeWorld(Pose2D(0, 0, 0), _empty_map()), tuned)
    assert sup._goal_blocked_accept_m == pytest.approx(2.5)  # noqa: SLF001
    assert sup._goal_blocked_after_s == pytest.approx(20.0)  # noqa: SLF001


def test_status_dict_carries_goal_blocked_fields():
    d = NavStatus().to_dict()
    assert d["goal_blocked"] is False
    assert "goal_offset_m" not in d
    d = NavStatus(goal_blocked=True, goal_offset_m=1.23456).to_dict()
    assert d["goal_blocked"] is True
    assert d["goal_offset_m"] == pytest.approx(1.235)


# --- verdict classification ------------------------------------------------------


def test_plain_no_feasible_path_is_not_goal_blocked():
    """The blocked-corridor paint seals the robot's own corridor by design and
    the planner then says "no feasible path"; that is not a goal verdict."""
    sup, _ = _sup()
    sup._last_replan_error = "scan+local: same route (1.0 m); blocked-corridor: no feasible path"
    assert sup._replan_says_goal_blocked() is False
    sup._last_replan_error = ""
    assert sup._replan_says_goal_blocked() is False
    sup._last_replan_error = "scan+local: goal snap 0.80 m exceeds max_goal_snap_m=0.50 (goal blocked / over-inflated)"
    assert sup._replan_says_goal_blocked() is True
    sup._last_replan_error = _GOAL_IN_LETHAL
    assert sup._replan_says_goal_blocked() is True


def test_paint_attempt_no_feasible_path_does_not_read_as_goal_blocked(monkeypatch):
    sup, _ = _sup()
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
    # A failed replan stamps the verdict time for the freshness check.
    assert time.monotonic() - sup._last_replan_at < 5.0


# --- finish policy ----------------------------------------------------------------


def test_goal_blocked_finishes_nearby_after_grace():
    sup, world = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=10.0)
    sup._last_replan_error = _GOAL_BLOCKED_SNAP  # noqa: SLF001
    pose, goal = Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0)
    sup._last_replan_at = time.monotonic()  # verdict is from this episode  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(pose, goal, 100.0, pose_cost=0) is False  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(pose, goal, 105.0, pose_cost=0) is False  # noqa: SLF001
    assert sup.status().to_dict()["state"] != "succeeded"
    assert sup._maybe_finish_goal_blocked(pose, goal, 111.0, pose_cost=0) is True  # noqa: SLF001
    st = sup.status().to_dict()
    assert st["state"] == "succeeded"
    assert st["active"] is False
    assert st["error_msg"] == ""
    assert st["goal_blocked"] is True
    assert st["goal_offset_m"] == pytest.approx(1.0, abs=0.01)
    assert world.stop_calls >= 1


def test_goal_blocked_not_applied_when_far_or_start_lethal_or_other_error():
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=0.0)
    goal = Pose2D(0.0, 0.0, 0.0)
    sup._last_replan_error = _GOAL_IN_LETHAL  # noqa: SLF001
    sup._last_replan_at = time.monotonic()  # noqa: SLF001
    # Too far from the goal.
    assert sup._maybe_finish_goal_blocked(Pose2D(3.0, 0.0, 0.0), goal, 1.0, pose_cost=0) is False  # noqa: SLF001
    assert sup._goal_blocked_since is None  # noqa: SLF001
    # Start cell lethal: that is a localization problem, not a blocked goal.
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0.0, 0.0), goal, 1.0, pose_cost=int(LETHAL)) is False  # noqa: SLF001
    assert sup._goal_blocked_since is None  # noqa: SLF001
    # Unrelated planner error.
    sup._last_replan_error = "cannot reach plan start from current pose"  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0.0, 0.0), goal, 1.0, pose_cost=0) is False  # noqa: SLF001
    assert sup.status().to_dict()["state"] != "succeeded"


def test_goal_blocked_timer_needs_a_fresh_replan():
    """A stale verdict from a block minutes ago must not pre-arm the timer."""
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=15.0)
    sup._last_replan_error = _GOAL_IN_LETHAL
    sup._last_replan_at = time.monotonic() - 60.0
    goal = Pose2D(0.0, 0.0, 0.0)
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0, 0), goal, 100.0, pose_cost=0) is False
    assert sup._goal_blocked_since is None


def test_goal_blocked_timer_rearms_after_the_robot_moves_on():
    """Timer armed on a transient block; the robot drives 1 m; a later block
    gets the full grace period again instead of finishing at once."""
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=15.0)
    sup._last_replan_error = _GOAL_IN_LETHAL
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


def test_accepted_replan_disarms_the_timer():
    sup, _ = _sup(goal_blocked_accept_m=2.5, goal_blocked_after_s=15.0)
    sup._last_replan_error = _GOAL_IN_LETHAL
    sup._last_replan_at = time.monotonic()
    goal = Pose2D(0.0, 0.0, 0.0)
    assert sup._maybe_finish_goal_blocked(Pose2D(1.0, 0, 0), goal, 100.0, pose_cost=0) is False
    assert sup._goal_blocked_since == 100.0
    old = _path((1.0, 0.0), (0.0, 0.0))
    new = _path((1.0, 0.0), (0.5, 0.5), (0.0, 0.0))
    sup._commit_replan_path(
        old, new, 1.5, 1.0, "scan+local", [], require_different=False,
        result=PlanResult(feasible=True, error_code=0, error_msg="", path=new),
        goal=goal, pose=Pose2D(1.0, 0, 0),
    )
    assert sup._goal_blocked_since is None
    assert sup._last_replan_error == ""


def test_default_zero_radius_never_finishes_goal_blocked():
    """With the default ``goal_blocked_accept_m=0`` the policy is inert even
    when the verdict is fresh, goal-side, nearby and the grace has elapsed."""
    sup, world = _sup()
    assert sup._goal_blocked_accept_m == 0.0  # noqa: SLF001
    sup._last_replan_error = _GOAL_BLOCKED_SNAP  # noqa: SLF001
    goal = Pose2D(0.0, 0.0, 0.0)
    for t in (1.0, 100.0, 1000.0):
        sup._last_replan_at = time.monotonic()  # noqa: SLF001
        assert sup._maybe_finish_goal_blocked(Pose2D(0.5, 0, 0), goal, t, pose_cost=0) is False  # noqa: SLF001
    assert sup._goal_blocked_since is None  # noqa: SLF001
    st = sup.status().to_dict()
    assert st["state"] != "succeeded"
    assert st["goal_blocked"] is False
    assert "goal_offset_m" not in st
    assert world.stop_calls == 0

    # Same with an explicit zero and a zero grace.
    sup, _ = _sup(goal_blocked_accept_m=0.0, goal_blocked_after_s=0.0)
    sup._last_replan_error = _GOAL_BLOCKED_SNAP  # noqa: SLF001
    sup._last_replan_at = time.monotonic()  # noqa: SLF001
    assert sup._maybe_finish_goal_blocked(Pose2D(0.5, 0, 0), goal, 1.0, pose_cost=0) is False  # noqa: SLF001

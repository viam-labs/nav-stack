"""Unit tests for optional TypeSafe/Jev nav policy (stubbed client)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.nav_builtin.jev_policy import (
    ACTION_KEEP_DWA,
    ACTION_REPLAN,
    ACTION_WAIT,
    JevNavPolicy,
    LocalBlockContext,
    ObstacleSample,
    map_jev_to_action,
    normalize_nav_policy,
    obstacle_motion_features,
)


def test_normalize_nav_policy():
    assert normalize_nav_policy("heuristic") == "heuristic"
    assert normalize_nav_policy("SHADOW") == "shadow"
    assert normalize_nav_policy("typesafe") == "jev"
    with pytest.raises(ValueError):
        normalize_nav_policy("gpt")


def test_obstacle_motion_features_mover_vs_fixed():
    t0 = 100.0
    mover = [
        ObstacleSample(
            t=t0 + i * 0.2,
            nose_clear=False,
            forward_clearance_m=0.4 - i * 0.05,
            path_ahead_cost=220,
            obstacle_state="avoid",
            nearest_range_m=0.8 - i * 0.08,
            nearest_bearing_rad=-0.1 + i * 0.08,
        )
        for i in range(6)
    ]
    fixed = [
        ObstacleSample(
            t=t0 + i * 0.2,
            nose_clear=False,
            forward_clearance_m=0.35,
            path_ahead_cost=240,
            obstacle_state="avoid",
            nearest_range_m=0.55,
            nearest_bearing_rad=0.05,
        )
        for i in range(6)
    ]
    m = obstacle_motion_features(mover)
    f = obstacle_motion_features(fixed)
    assert m["motion_score"] > f["motion_score"]
    assert m["likely_mover"] is True
    assert f["likely_mover"] is False


def test_map_jev_soft_overrides():
    assert (
        map_jev_to_action(
            choice=ACTION_REPLAN,
            temporary_mover=0.9,
            gap_worth_trying=0.1,
            heuristic_action=ACTION_WAIT,
        )
        == ACTION_WAIT
    )
    assert (
        map_jev_to_action(
            choice=ACTION_REPLAN,
            temporary_mover=0.1,
            gap_worth_trying=0.9,
            heuristic_action=ACTION_KEEP_DWA,
        )
        == ACTION_KEEP_DWA
    )


def _stub_result(*, choice: str, confidence: float, mover: float, gap: float):
    return SimpleNamespace(
        choices={
            "action": SimpleNamespace(
                choice=choice,
                confidence=confidence,
                probabilities={choice: confidence},
            )
        },
        nouls={
            "temporary_mover": SimpleNamespace(noul=mover),
            "gap_worth_trying": SimpleNamespace(noul=gap),
        },
        scores={"replan_urgency": SimpleNamespace(score=1.0, confidence=0.5)},
    )


def _ctx(heuristic: str = ACTION_REPLAN) -> LocalBlockContext:
    return LocalBlockContext(
        heuristic_action=heuristic,
        nose_clear=False,
        blocked_for_s=2.5,
        wait_before_replan_s=2.0,
        replan_cooldown_ready=True,
        path_ahead_cost=240,
        obstacle_state="avoid",
        forward_clearance_m=0.3,
        failed_replan_while_blocked=1,
        remaining_path_m=4.0,
    )


def test_mode_heuristic_never_queries():
    calls = []

    def query_fn(state, questions):
        calls.append(1)
        return _stub_result(
            choice=ACTION_WAIT, confidence=0.9, mover=0.8, gap=0.2
        )

    policy = JevNavPolicy(mode="heuristic", query_fn=query_fn, min_period_s=0)
    d = policy.decide(_ctx(ACTION_REPLAN))
    assert d.applied_action == ACTION_REPLAN
    assert d.queried is False
    assert calls == []


def test_mode_shadow_logs_but_applies_heuristic():
    def query_fn(state, questions):
        assert "obstacle_motion" in state
        assert "action" in questions
        return _stub_result(
            choice=ACTION_WAIT, confidence=0.95, mover=0.9, gap=0.1
        )

    policy = JevNavPolicy(mode="shadow", query_fn=query_fn, min_period_s=0)
    d = policy.decide(_ctx(ACTION_REPLAN))
    assert d.queried is True
    assert d.heuristic_action == ACTION_REPLAN
    assert d.jev_action == ACTION_WAIT
    assert d.applied_action == ACTION_REPLAN
    assert d.fallback_reason == "shadow"
    assert d.confidence == pytest.approx(0.95)


def test_mode_jev_applies_when_confident():
    policy = JevNavPolicy(
        mode="jev",
        query_fn=lambda s, q: _stub_result(
            choice=ACTION_KEEP_DWA, confidence=0.8, mover=0.1, gap=0.85
        ),
        min_confidence=0.45,
        min_period_s=0,
    )
    d = policy.decide(_ctx(ACTION_REPLAN))
    assert d.applied_action == ACTION_KEEP_DWA
    assert d.jev_action == ACTION_KEEP_DWA
    assert d.fallback_reason == ""


def test_mode_jev_falls_back_on_low_confidence():
    policy = JevNavPolicy(
        mode="jev",
        query_fn=lambda s, q: _stub_result(
            choice=ACTION_WAIT, confidence=0.2, mover=0.5, gap=0.5
        ),
        min_confidence=0.45,
        min_period_s=0,
    )
    d = policy.decide(_ctx(ACTION_REPLAN))
    assert d.applied_action == ACTION_REPLAN
    assert d.jev_action == ACTION_WAIT
    assert "low_confidence" in d.fallback_reason


def test_mode_jev_falls_back_on_query_error():
    def boom(_s, _q):
        raise RuntimeError("network down")

    policy = JevNavPolicy(mode="jev", query_fn=boom, min_period_s=0)
    d = policy.decide(_ctx(ACTION_WAIT))
    assert d.applied_action == ACTION_WAIT
    assert d.fallback_reason == "query_error"
    assert "network" in d.error


def test_builtin_nav_config_accepts_nav_policy():
    from src.config import BuiltinNavConfig

    cfg = BuiltinNavConfig.from_dict({"nav_policy": "shadow", "jev_timeout_s": 0.9})
    assert cfg.nav_policy == "shadow"
    assert cfg.jev_timeout_s == pytest.approx(0.9)
    with pytest.raises(ValueError):
        BuiltinNavConfig.from_dict({"nav_policy": "chatgpt"})

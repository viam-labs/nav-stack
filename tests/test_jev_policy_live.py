"""Optional live check against the real TypeSafe/Jev API.

Skipped unless ``TYPESAFE_API_KEY`` is set:

    TYPESAFE_API_KEY=... pytest tests/test_jev_policy_live.py -q
"""
from __future__ import annotations

import os

import pytest

from src.nav_builtin.jev_policy import (
    ACTION_KEEP_DWA,
    ACTION_REPLAN,
    ACTION_WAIT,
    ACTIONS,
    JevNavPolicy,
    LocalBlockContext,
    ObstacleSample,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY not set",
)


def test_live_jev_local_block_consult():
    policy = JevNavPolicy(
        mode="shadow",
        min_period_s=0,
        timeout_s=8.0,
        min_confidence=0.3,
    )
    t0 = 1_000.0
    for i in range(5):
        policy.record_obstacle(
            ObstacleSample(
                t=t0 + i * 0.25,
                nose_clear=False,
                forward_clearance_m=0.45,
                path_ahead_cost=230,
                obstacle_state="avoid",
                nearest_range_m=0.7 - i * 0.05,
                nearest_bearing_rad=-0.05 + i * 0.04,
            )
        )
    decision = policy.decide(
        LocalBlockContext(
            heuristic_action=ACTION_REPLAN,
            nose_clear=False,
            blocked_for_s=2.0,
            wait_before_replan_s=1.5,
            replan_cooldown_ready=True,
            path_ahead_cost=230,
            obstacle_state="avoid",
            forward_clearance_m=0.35,
            failed_replan_while_blocked=0,
            remaining_path_m=6.0,
        )
    )
    assert decision.queried is True
    assert decision.error == ""
    assert decision.jev_action in ACTIONS
    assert decision.applied_action == ACTION_REPLAN  # shadow
    assert decision.confidence is not None
    assert "temporary_mover" in decision.answers
    assert "gap_worth_trying" in decision.answers

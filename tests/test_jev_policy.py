"""Unit tests for optional TypeSafe/Jev nav policy (stubbed client)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.nav_builtin.jev_policy import (
    ACTION_ABORT,
    ACTION_BACKUP,
    ACTION_KEEP_DWA,
    ACTION_REPLAN,
    ACTION_WAIT,
    ACTION_WIDE_REPLAN,
    JevNavPolicy,
    LocalBlockContext,
    ObstacleSample,
    backup_denial_reason,
    build_policy_state,
    classify_block_reasons,
    map_jev_to_action,
    normalize_nav_policy,
    obstacle_motion_features,
    stuck_progress_features,
)


def test_normalize_nav_policy():
    assert normalize_nav_policy("heuristic") == "heuristic"
    assert normalize_nav_policy("SHADOW") == "shadow"
    assert normalize_nav_policy("typesafe") == "jev"
    assert normalize_nav_policy("random") == "random"
    with pytest.raises(ValueError):
        normalize_nav_policy("gpt")


def test_available_policy_actions_respects_gates():
    from src.nav_builtin.jev_policy import (
        CONSULT_SOFT,
        available_policy_actions,
    )

    assert available_policy_actions() == [
        ACTION_WAIT,
        ACTION_KEEP_DWA,
        ACTION_REPLAN,
    ]
    assert available_policy_actions(backup_feasible=True)[-1] == ACTION_BACKUP
    full = available_policy_actions(
        backup_feasible=True,
        wide_replan_available=True,
        abort_available=True,
    )
    assert full == [
        ACTION_WAIT,
        ACTION_KEEP_DWA,
        ACTION_REPLAN,
        ACTION_BACKUP,
        ACTION_WIDE_REPLAN,
        ACTION_ABORT,
    ]
    soft = available_policy_actions(
        backup_feasible=True,
        wide_replan_available=True,
        abort_available=True,
        consult_kind=CONSULT_SOFT,
    )
    assert soft == [ACTION_WAIT, ACTION_KEEP_DWA, ACTION_REPLAN]


def test_corridor_gap_features_open_vs_sealed():
    from src.nav_builtin.costmap import INSCRIBED
    from src.nav_builtin.jev_policy import corridor_gap_features

    class _Open:
        def cost_at_world(self, x_m, y_m):
            return 0

    class _Sealed:
        def cost_at_world(self, x_m, y_m):
            # Wall across x>=0.6, free behind the robot.
            return INSCRIBED if x_m >= 0.6 else 0

    open_g = corridor_gap_features(_Open(), x=0.0, y=0.0, theta=0.0)
    assert open_g["min_gap_m"] >= 2.9
    assert open_g["sealed_count"] == 0

    sealed = corridor_gap_features(_Sealed(), x=0.0, y=0.0, theta=0.0)
    assert sealed["min_gap_m"] < 0.35
    assert sealed["sealed_count"] >= 1


def test_soft_consult_heuristic_and_gates():
    from src.nav_builtin.jev_policy import (
        CONSULT_SOFT,
        map_jev_to_action,
        soft_consult_heuristic,
    )

    assert (
        soft_consult_heuristic(
            likely_mover=True,
            nose_clear=True,
            path_ahead_cost=120,
            soft_cost=110,
            activate_cost=200,
        )
        == ACTION_WAIT
    )
    assert (
        soft_consult_heuristic(
            likely_mover=False,
            nose_clear=True,
            path_ahead_cost=160,
            soft_cost=110,
            activate_cost=200,
        )
        == ACTION_REPLAN
    )
    assert (
        soft_consult_heuristic(
            likely_mover=False,
            nose_clear=True,
            path_ahead_cost=120,
            soft_cost=110,
            activate_cost=200,
        )
        == ACTION_KEEP_DWA
    )
    # Soft consult must not apply backup/abort even if Jev asks.
    assert (
        map_jev_to_action(
            choice=ACTION_BACKUP,
            heuristic_action=ACTION_KEEP_DWA,
            backup_feasible=True,
            consult_kind=CONSULT_SOFT,
        )
        == ACTION_KEEP_DWA
    )
    assert (
        map_jev_to_action(
            choice=ACTION_ABORT,
            heuristic_action=ACTION_WAIT,
            abort_available=True,
            consult_kind=CONSULT_SOFT,
        )
        == ACTION_WAIT
    )


def test_builtin_nav_config_accepts_soft_consult():
    from src.config import BuiltinNavConfig

    cfg = BuiltinNavConfig.from_dict(
        {"nav_policy": "jev", "jev_soft_consult": True, "jev_soft_path_cost": 100}
    )
    assert cfg.jev_soft_consult is True
    assert cfg.jev_soft_path_cost == 100
    assert BuiltinNavConfig.from_dict({}).jev_soft_path_cost is None


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


def test_map_jev_safety_gates_only():
    # Jev's choice is applied as-is...
    assert (
        map_jev_to_action(
            choice=ACTION_KEEP_DWA,
            heuristic_action=ACTION_REPLAN,
            backup_feasible=False,
        )
        == ACTION_KEEP_DWA
    )
    assert (
        map_jev_to_action(
            choice=ACTION_REPLAN,
            heuristic_action=ACTION_WAIT,
        )
        == ACTION_REPLAN
    )
    # ...except infeasible backup → heuristic (or wait).
    assert (
        map_jev_to_action(
            choice=ACTION_BACKUP,
            heuristic_action=ACTION_KEEP_DWA,
            backup_feasible=False,
        )
        == ACTION_KEEP_DWA
    )
    assert (
        map_jev_to_action(
            choice=ACTION_BACKUP,
            heuristic_action=ACTION_WAIT,
            backup_feasible=True,
        )
        == ACTION_BACKUP
    )
    # Unknown choice → heuristic.
    assert (
        map_jev_to_action(choice="teleport", heuristic_action=ACTION_REPLAN)
        == ACTION_REPLAN
    )
    # Jev-only escalations pass through only when the supervisor says so.
    assert (
        map_jev_to_action(
            choice=ACTION_WIDE_REPLAN,
            heuristic_action=ACTION_REPLAN,
            wide_replan_available=True,
        )
        == ACTION_WIDE_REPLAN
    )
    assert (
        map_jev_to_action(
            choice=ACTION_WIDE_REPLAN,
            heuristic_action=ACTION_KEEP_DWA,
            wide_replan_available=False,
        )
        == ACTION_KEEP_DWA
    )
    assert (
        map_jev_to_action(
            choice=ACTION_ABORT, heuristic_action=ACTION_WAIT, abort_available=True
        )
        == ACTION_ABORT
    )
    assert (
        map_jev_to_action(
            choice=ACTION_ABORT, heuristic_action=ACTION_WAIT, abort_available=False
        )
        == ACTION_WAIT
    )


def test_run_stats_survive_block_flicker_and_report_progress():
    policy = JevNavPolicy(
        mode="jev",
        query_fn=lambda s, q: _stub_result(
            choice=ACTION_KEEP_DWA, confidence=0.9, mover=0.1, gap=0.5
        ),
        min_period_s=0,
    )
    policy.start_run()
    t0 = 1000.0
    # 30 s of ticks: robot creeps 0.02 m/s toward goal, blocked half the time.
    for i in range(600):
        t = t0 + i * 0.05
        policy.note_tick(
            now=t,
            x=i * 0.001,
            y=0.0,
            dist_to_goal=4.0 - i * 0.001,
            blocked=(i // 40) % 2 == 0,
            dt=0.05,
        )
    policy.note_block_episode_started()
    policy.note_block_episode_started()
    policy.note_replan(accepted=False)
    policy.note_replan(accepted=False)
    policy.note_replan(accepted=True, wide=True)
    policy.note_backup_started()
    run = policy.run_stats.to_dict(t0 + 600 * 0.05)
    assert run["block_episodes"] == 2
    # 15 windows of 40 ticks (2 s); indices 0,2,..,14 blocked → 8 × 2 s.
    assert run["blocked_total_s"] == pytest.approx(16.0, abs=0.5)
    assert run["replans_failed"] == 2
    assert run["replans_accepted"] == 1
    assert run["wide_replans"] == 1
    assert run["backups_started"] == 1
    assert run["progress_30s"]["toward_goal_m"] == pytest.approx(0.6, abs=0.05)
    assert run["progress_10s"]["toward_goal_m"] == pytest.approx(0.2, abs=0.05)
    # decide() ships run facts to Jev and into features / the decision log.
    ctx = _ctx(ACTION_REPLAN)
    ctx.abort_available = True
    ctx.wide_replan_available = True
    captured = {}

    def q(state, questions):
        captured["state"] = state
        return _stub_result(choice=ACTION_ABORT, confidence=0.9, mover=0.0, gap=0.0)

    policy._query_fn = q
    d = policy.decide(ctx)
    assert captured["state"]["run"]["block_episodes"] == 2
    assert captured["state"]["abort"]["available"] is True
    assert ACTION_WIDE_REPLAN in captured["state"]["actions"]
    assert d.applied_action == ACTION_ABORT
    assert d.features["run"]["replans_failed"] == 2
    # New run wipes the counters.
    policy.start_run()
    assert policy.run_stats.block_episodes == 0


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
        failed_replan_while_blocked=0,
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
        min_confidence=0.7,
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
            choice=ACTION_WAIT, confidence=0.55, mover=0.5, gap=0.5
        ),
        min_confidence=0.7,
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


def test_decision_log_records_pose_and_survives_clear_obstacle_history():
    policy = JevNavPolicy(
        mode="shadow",
        query_fn=lambda s, q: _stub_result(
            choice=ACTION_WAIT, confidence=0.9, mover=0.8, gap=0.2
        ),
        min_period_s=0,
    )
    run_id = policy.start_run()
    ctx = _ctx(ACTION_REPLAN)
    ctx.pose_xy = (1.25, 2.5)
    ctx.goal_xy = (3.0, 4.0)
    d = policy.decide(ctx)
    assert d.queried is True
    log = policy.decision_log()
    assert len(log) == 1
    assert log[0]["run_id"] == run_id
    assert log[0]["pose"] == {"x": 1.25, "y": 2.5}
    assert log[0]["heuristic_action"] == ACTION_REPLAN
    assert log[0]["jev_action"] == ACTION_WAIT
    assert log[0]["applied_action"] == ACTION_REPLAN  # shadow
    policy.clear_history()  # obstacle track only
    assert len(policy.decision_log()) == 1
    policy.clear_decision_log()
    assert policy.decision_log() == []


def test_builtin_nav_config_accepts_nav_policy():
    from src.config import BuiltinNavConfig

    cfg = BuiltinNavConfig.from_dict({"nav_policy": "shadow", "jev_timeout_s": 0.9})
    assert cfg.nav_policy == "shadow"
    assert cfg.jev_timeout_s == pytest.approx(0.9)
    assert cfg.jev_min_confidence == pytest.approx(0.7)
    assert BuiltinNavConfig.from_dict({"nav_policy": "random"}).nav_policy == "random"
    with pytest.raises(ValueError):
        BuiltinNavConfig.from_dict({"nav_policy": "chatgpt"})


def test_mode_random_picks_executable_action():
    import random

    from src.nav_builtin.jev_policy import available_policy_actions

    rng = random.Random(0)
    policy = JevNavPolicy(mode="random", min_period_s=0, rng=rng)
    ctx = LocalBlockContext(
        heuristic_action=ACTION_WAIT,
        nose_clear=False,
        blocked_for_s=2.0,
        wait_before_replan_s=2.0,
        replan_cooldown_ready=True,
        path_ahead_cost=250,
        obstacle_state="avoid",
        forward_clearance_m=0.4,
        backup_feasible=True,
        wide_replan_available=False,
        abort_available=False,
    )
    expected = set(
        available_policy_actions(
            backup_feasible=True,
            wide_replan_available=False,
            abort_available=False,
        )
    )
    seen = set()
    for _ in range(40):
        d = policy.decide(ctx)
        assert d.mode == "random"
        assert d.applied_action in expected
        assert d.jev_action == d.applied_action
        assert d.fallback_reason == "random"
        assert d.queried is False
        seen.add(d.applied_action)
    assert len(seen) >= 2


def test_mode_random_rate_limit_reuses_choice():
    import random

    policy = JevNavPolicy(
        mode="random", min_period_s=60.0, rng=random.Random(1)
    )
    ctx = LocalBlockContext(
        heuristic_action=ACTION_REPLAN,
        nose_clear=False,
        blocked_for_s=1.0,
        wait_before_replan_s=2.0,
        replan_cooldown_ready=True,
        path_ahead_cost=250,
        obstacle_state="avoid",
        forward_clearance_m=0.5,
    )
    first = policy.decide(ctx)
    second = policy.decide(ctx)
    assert first.applied_action == second.applied_action
    assert second.fallback_reason == "rate_limited_reuse_random"
    assert second.queried is False


def test_stuck_progress_reports_raw_motion_only():
    t0 = 50.0
    samples = [
        ObstacleSample(
            t=t0 + i * 0.25,
            nose_clear=False,
            forward_clearance_m=0.35,
            path_ahead_cost=0,
            obstacle_state="avoid",
            pose_x=0.0 + i * 0.01,
            pose_y=0.0,
            pose_theta=i * 0.25,
        )
        for i in range(10)
    ]
    stuck = stuck_progress_features(samples)
    assert stuck["progress_m"] < 0.15
    assert stuck["yaw_delta_rad"] >= 0.45
    assert stuck["window_s"] == pytest.approx(2.25)
    # No verdict keys — Jev decides what "stuck" means.
    assert "likely_peel_loop" not in stuck
    assert "spinning_in_place" not in stuck


def test_classify_block_reasons_and_backup_denial():
    assert classify_block_reasons(
        path_ahead_cost=0,
        activate_cost=200,
        pose_cost=0,
        inscribed_cost=253,
        obstacle_state="avoid",
        reactive_avoid_for_s=1.2,
    ) == ["reactive_avoid"]
    assert classify_block_reasons(
        path_ahead_cost=254,
        activate_cost=200,
        pose_cost=253,
        inscribed_cost=253,
        obstacle_state="clear",
    ) == ["path_cost", "pose_inscribed"]
    assert backup_denial_reason(
        enabled=True,
        feasible=False,
        attempts_remaining=2,
        cooldown_ready=True,
        has_scan=True,
        rear_clearance_m=0.1,
        rear_clear_min_m=0.35,
        reverse_footprint_ok=None,
    ) == "rear_blocked"
    assert (
        backup_denial_reason(
            enabled=True,
            feasible=True,
            attempts_remaining=2,
            cooldown_ready=True,
            has_scan=True,
            rear_clearance_m=1.0,
            rear_clear_min_m=0.35,
            reverse_footprint_ok=True,
        )
        == ""
    )


def test_build_policy_state_is_facts_only():
    ctx = LocalBlockContext(
        heuristic_action=ACTION_REPLAN,
        nose_clear=False,
        blocked_for_s=3.0,
        wait_before_replan_s=2.0,
        replan_cooldown_ready=True,
        path_ahead_cost=0,
        obstacle_state="avoid",
        forward_clearance_m=0.37,
        block_reasons=["reactive_avoid"],
        pose_cost=10,
        cmd_vx_mps=0.0,
        cmd_vtheta_rad_s=0.4,
        bearing_error_rad=0.7,
        backup_denial_reason="rear_blocked",
        last_replan_attempts=[
            "scan+local: still local-blocked (cost=254)",
            "viaR0.65: still local-blocked (cost=253)",
        ],
        last_replan_error="scan+local: still local-blocked",
        last_replan_age_s=22.3,
    )
    motion = {
        "samples": 5,
        "motion_score": 0.0,
        "likely_mover": False,
        "progress_m": 0.05,
        "yaw_delta_rad": 0.9,
        "window_s": 2.5,
    }
    recent = {"current": "wait", "current_for_s": 12.0, "time_share_s": {"wait": 12.0}}
    state = build_policy_state(ctx, motion, recent_actions=recent)
    assert state["block_reasons"] == ["reactive_avoid"]
    assert state["motion_while_blocked"]["progress_m"] == pytest.approx(0.05)
    assert state["recent_actions"]["current"] == "wait"
    assert state["motion_cmd"]["vtheta_rad_s"] == pytest.approx(0.4)
    assert "still local-blocked" in state["last_replan"]["attempts"][0]
    assert state["last_replan"]["age_s"] == pytest.approx(22.3)
    assert state["backup"]["denial_reason"] == "rear_blocked"
    # No editorial hints / verdicts.
    assert "efficiency_hint" not in state
    assert "stuck" not in state
    blob = repr(state).lower()
    assert "prefer " not in blob
    assert "likely_peel_loop" not in blob


def test_decide_applies_jev_choice_and_tracks_recent_actions():
    captured = {}

    def query_fn(state, questions):
        captured["state"] = state
        return _stub_result(
            choice=ACTION_KEEP_DWA, confidence=0.85, mover=0.05, gap=0.7
        )

    policy = JevNavPolicy(mode="jev", query_fn=query_fn, min_period_s=0)
    ctx = _ctx(ACTION_REPLAN)
    ctx.last_replan_attempts = [
        "scan+local: still local-blocked (cost=254)",
        "blocked-corridor: still local-blocked (cost=254)",
    ]
    ctx.failed_replan_while_blocked = 1
    d = policy.decide(ctx)
    # Jev said keep_dwa; code does not second-guess it.
    assert d.jev_action == ACTION_KEEP_DWA
    assert d.applied_action == ACTION_KEEP_DWA
    assert "replan_looks_futile" not in d.features
    assert captured["state"]["last_replan"]["attempts"]
    d2 = policy.decide(ctx)
    assert d2.features["recent_actions"]["current"] == ACTION_KEEP_DWA
    assert captured["state"]["recent_actions"]["current"] == ACTION_KEEP_DWA


def test_rate_limited_reuse_requeries_when_situation_changes():
    calls = []

    def query_fn(state, questions):
        calls.append(state["nose_clear"])
        return _stub_result(
            choice=ACTION_WAIT, confidence=0.9, mover=0.1, gap=0.1
        )

    policy = JevNavPolicy(mode="jev", query_fn=query_fn, min_period_s=60.0)
    ctx = _ctx(ACTION_REPLAN)
    ctx.nose_clear = False
    d1 = policy.decide(ctx)
    assert d1.queried is True
    d2 = policy.decide(ctx)
    assert d2.queried is False
    assert d2.fallback_reason == "rate_limited_reuse_jev"
    # Nose opens up → do not keep replaying the stale wait.
    ctx.nose_clear = True
    d3 = policy.decide(ctx)
    assert d3.queried is True
    assert d3.fallback_reason == "requeried_state_change"
    assert calls == [False, True]

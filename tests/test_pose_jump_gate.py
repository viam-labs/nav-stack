"""Unit tests for large automatic pose-jump confirmation."""
from __future__ import annotations

import math

from src.nav.pose_jump_gate import PoseJumpGate, should_hold_drive_for_pose_jump
from src.geom import conversions as conv


def test_should_hold_drive_for_awaiting_confirm():
    assert should_hold_drive_for_pose_jump({"status": "awaiting_confirm"})
    assert should_hold_drive_for_pose_jump({"status": "nav_hold"})
    assert not should_hold_drive_for_pose_jump({"status": "previous_better"})
    assert not should_hold_drive_for_pose_jump({"status": "corrected"})
    assert not should_hold_drive_for_pose_jump({"status": "ok"})
    assert not should_hold_drive_for_pose_jump({"status": "low_quality"})
    assert not should_hold_drive_for_pose_jump({"status": "refused_large"})
    assert not should_hold_drive_for_pose_jump({"status": "soft_loc_resume"})
    assert not should_hold_drive_for_pose_jump(None)
    assert not should_hold_drive_for_pose_jump("awaiting_confirm")


def test_candidate_beats_previous_requires_margin_on_large_shift():
    from src.nav.pose_jump_gate import candidate_beats_previous

    # 7.7 m false peak with similar/worse score must not win.
    assert not candidate_beats_previous(
        previous_score=0.55,
        candidate_score=0.43,
        shift_m=7.7,
        shift_deg=1.0,
        previous_ray_mae_m=0.4,
        candidate_ray_mae_m=1.4,
    )
    # Clear win at distance still applies.
    assert candidate_beats_previous(
        previous_score=0.2,
        candidate_score=1.2,
        shift_m=2.0,
        shift_deg=5.0,
    )


def test_small_jump_applies_immediately():
    gate = PoseJumpGate(confirm_count=2, large_m=0.75, large_deg=25.0)
    current = conv.Pose2D(0.0, 0.0, 0.0)
    decision = gate.evaluate(current, conv.Pose2D(0.4, 0.0, 0.0))
    assert decision.should_apply is True
    assert decision.status == "apply_small"
    assert decision.large_jump is False


def test_large_jump_needs_two_agreeing_matches():
    gate = PoseJumpGate(confirm_count=2, large_m=0.75, agree_m=0.4)
    current = conv.Pose2D(0.0, 0.0, 0.0)
    cand = conv.Pose2D(1.9, 0.0, math.pi)
    first = gate.evaluate(current, cand)
    assert first.should_apply is False
    assert first.status == "awaiting_confirm"
    assert first.confirm_count == 1
    second = gate.evaluate(current, conv.Pose2D(1.95, 0.05, math.pi))
    assert second.should_apply is True
    assert second.status == "apply_confirmed"


def test_disagreeing_large_jump_resets_confirm():
    gate = PoseJumpGate(confirm_count=2, large_m=0.75, agree_m=0.4)
    current = conv.Pose2D(0.0, 0.0, 0.0)
    gate.evaluate(current, conv.Pose2D(2.0, 0.0, 0.0))
    again = gate.evaluate(current, conv.Pose2D(0.0, 2.0, 0.0))
    assert again.should_apply is False
    assert again.confirm_count == 1


def test_force_bypasses_confirm():
    gate = PoseJumpGate(confirm_count=3, large_m=0.75)
    current = conv.Pose2D(0.0, 0.0, 0.0)
    decision = gate.evaluate(
        current, conv.Pose2D(3.0, 0.0, 0.0), force=True
    )
    assert decision.should_apply is True
    assert decision.status == "forced"

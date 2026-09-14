"""Unit tests for large automatic pose-jump confirmation."""
from __future__ import annotations

import math

from src.nav.pose_jump_gate import PoseJumpGate, should_hold_drive_for_pose_jump
from src.ros import conversions as conv


def test_should_hold_drive_for_awaiting_confirm():
    assert should_hold_drive_for_pose_jump({"status": "awaiting_confirm"})
    assert not should_hold_drive_for_pose_jump({"status": "corrected"})
    assert not should_hold_drive_for_pose_jump({"status": "ok"})
    assert not should_hold_drive_for_pose_jump(None)
    assert not should_hold_drive_for_pose_jump("awaiting_confirm")


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

"""Shared base envelope + nose-escape policy."""

from __future__ import annotations

from dataclasses import replace

from src.nav.control import escape
from src.nav.control.controller import FollowerConfig
from src.nav.io.viam_io import _sanitize_base_cmd
from src.nav.motion import base_envelope as env
from src.nav.motion.simple_motion import DriveCommand
from src.nav.types import Pose2D


def test_sanitize_matches_envelope_constants():
    assert _sanitize_base_cmd(0.03, 0.0, 0.5) == env.sanitize_base_cmd(0.03, 0.0, 0.5)
    vx, _, vt = env.sanitize_base_cmd(0.087, 0.0, 0.49)
    assert vx == 0.0
    assert abs(vt - 0.49) < 1e-9
    vx, _, vt = env.sanitize_base_cmd(0.25, 0.0, 0.1)
    assert vx == 0.25


def test_lift_sub_deadband_raises_a_tiny_reverse():
    lifted = env.lift_sub_deadband(DriveCommand(-0.03, 0.0, 0.0, False))
    assert lifted.vx <= -0.10
    assert env.lift_sub_deadband(DriveCommand(0.0, 0.0, 0.0, False)).vx == 0.0
    assert env.lift_sub_deadband(DriveCommand(0.015, 0.0, 0.025, False)).vx == 0.0
    turned = env.lift_sub_deadband(
        DriveCommand(0.13, 0.0, 0.0325, False),
        forward_clearance_m=0.36,
        body_length_m=0.72,
    )
    assert turned.vx == 0.0
    assert abs(turned.vtheta - env.BLOCKED_NOSE_TURN_RAD_S) < 1e-9


def test_escape_choose_unstick_priority():
    assert (
        escape.choose(
            reason="unstick",
            nose_clear=False,
            rear_open=True,
            spin_clear=False,
            heading_open=True,
            failed_replans=1,
        )
        == escape.TURN
    )
    assert (
        escape.choose(
            reason="unstick",
            nose_clear=False,
            rear_open=True,
            spin_clear=False,
            failed_replans=1,
        )
        == escape.REVERSE
    )
    assert (
        escape.choose(
            reason="bumper",
            nose_clear=False,
            rear_open=True,
            spin_blocked=True,
            cmd_vx=0.0,
            cmd_vtheta=0.0,
        )
        == escape.REVERSE
    )
    assert (
        escape.choose(
            reason="undrivable",
            nose_clear=False,
            rear_open=True,
            spin_clear=True,
            bearing_rad=0.73,
            cmd_vx=0.015,
            cmd_vtheta=0.025,
        )
        == escape.TURN
    )


def test_reverse_command_clears_the_sanitizer():
    cmd = escape.reverse_command(FollowerConfig())
    sx, _, st = env.sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
    assert sx < -env.LIN_EPS_MPS
    assert abs(st) < 1e-9


def test_keep_arc_inside_envelope_caps_yaw():
    v, w = env.keep_arc_inside_envelope(0.08, 0.40)
    assert abs(v - 0.08) < 1e-9
    assert abs(w) == env.SPIN_W_CAP_RAD_S
    v, w = env.keep_arc_inside_envelope(0.20, 0.40)
    assert abs(w - 0.40) < 1e-9


def _sense(**kwargs):
    base = escape.EscapeSense(
        now=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        cmd=DriveCommand(0.0, 0.0, 0.0, False),
        progress={},
        nose_clear=False,
        waiting=False,
        failed_replans=0,
        spin_blocked=True,
        heading_open=False,
        bearing_rad=0.0,
        scan=None,
        local_view=None,
        guard=None,
        guard_pts=None,
        robot_radius_m=0.3,
        spin_radius_m=0.4,
        follower=FollowerConfig(),
        backup_dist_m=0.35,
        backup_cooldown_s=2.0,
    )
    return replace(base, **kwargs) if kwargs else base


def test_escape_machine_bumper_reverse_then_replan(monkeypatch):
    monkeypatch.setattr(escape, "bumper_wants_reverse", lambda **k: True)
    monkeypatch.setattr(escape, "reverse_clear", lambda **k: True)
    monkeypatch.setattr(
        escape.EscapeMachine, "_rear_still_open", lambda self, s, remain: True
    )
    m = escape.EscapeMachine()
    r = m.tick_pre(_sense())
    assert r.active_mode == escape.MODE_REVERSE
    assert r.cmd is not None and r.cmd.vx < 0.0

    r2 = m.tick_pre(
        _sense(
            now=1.0,
            pose=Pose2D(-0.40, 0.0, 0.0),
            cmd=r.cmd,
            progress={"obstacle": "narrow_reverse"},
        )
    )
    assert r2.replan_trigger == "narrow_reverse_done"
    assert m.mode == escape.IDLE
    assert m.cooldown_until > 1.0


def test_escape_machine_wait_freeze_and_unstick_turn(monkeypatch):
    monkeypatch.setattr(escape, "try_reverse", lambda *a, **k: None)
    monkeypatch.setattr(escape, "bumper_wants_reverse", lambda **k: False)
    monkeypatch.setattr(escape.EscapeMachine, "_spin_clear", lambda self, s: False)
    m = escape.EscapeMachine()
    # Idle wait with no unstick path → freeze.
    r = m.tick_pre(
        _sense(
            waiting=True,
            failed_replans=0,
            spin_blocked=False,
            progress={"obstacle": "wait"},
        )
    )
    assert r.obstacle == "wait"
    assert r.cmd is not None and abs(r.cmd.vx) < 1e-9

    # After a failed replan with heading open → turn.
    r2 = m.tick_pre(
        _sense(
            waiting=True,
            failed_replans=2,
            heading_open=True,
            prefer_yaw_sign=-1.0,
            scan=object(),
            local_view=object(),
            progress={"obstacle": "wait"},
        )
    )
    assert r2.active_mode == escape.MODE_TURN
    assert r2.cmd is not None and r2.cmd.vtheta < 0.0

    r3 = m.tick_pre(
        _sense(
            now=1.0,
            waiting=True,
            failed_replans=2,
            pose=Pose2D(0.0, 0.0, -escape.UNSTICK_YAW_RAD),
            cmd=r2.cmd,
            scan=object(),
            local_view=object(),
            progress={"obstacle": "wait_turn"},
        )
    )
    assert r3.replan_trigger == "blocked_nose_unstick"


def test_escape_machine_wait_unstick_despite_follower_crawl(monkeypatch):
    """Wait must unstick even when the follower still offers a slow forward."""
    rev = DriveCommand(-0.12, 0.0, 0.0, False)
    monkeypatch.setattr(escape, "try_reverse", lambda *a, **k: rev)
    monkeypatch.setattr(escape, "bumper_wants_reverse", lambda **k: False)
    monkeypatch.setattr(escape.EscapeMachine, "_spin_clear", lambda self, s: False)
    m = escape.EscapeMachine()
    r = m.tick_pre(
        _sense(
            waiting=True,
            failed_replans=2,
            nose_clear=False,
            cmd=DriveCommand(0.11, 0.0, 0.0, False),
            scan=object(),
            local_view=object(),
            progress={"obstacle": "slow"},
        )
    )
    assert r.active_mode == escape.MODE_REVERSE
    assert r.obstacle == "wait_reverse"
    assert r.cmd is not None and r.cmd.vx < 0.0


def test_escape_machine_undrivable_post(monkeypatch):
    monkeypatch.setattr(escape, "undrivable_nose", lambda **k: escape.TURN)
    monkeypatch.setattr(escape, "rectangle_rear_open", lambda *a, **k: False)
    monkeypatch.setattr(
        escape.EscapeMachine, "_guard_spin_clear", lambda self, s, b: True
    )
    m = escape.EscapeMachine()
    r = m.tick_post(
        _sense(
            cmd=DriveCommand(0.02, 0.0, 0.02, False),
            bearing_rad=0.5,
            guard=object(),
            guard_pts=object(),
        )
    )
    assert r.active_mode == escape.MODE_TURN
    assert r.cmd is not None and r.cmd.vtheta > 0.0

    r2 = m.tick_post(
        _sense(
            now=1.0,
            pose=Pose2D(0.0, 0.0, escape.UNSTICK_YAW_RAD),
            cmd=r.cmd,
            bearing_rad=0.5,
            guard=object(),
            guard_pts=object(),
        )
    )
    assert r2.replan_trigger == "undrivable_nose"

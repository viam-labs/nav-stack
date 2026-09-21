"""Tests for localize spin-recovery helper."""
from __future__ import annotations

import math

from src.geom import conversions as conv
from src.nav.localize_spin_recovery import (
    LocalizeSpinConfig,
    LocalizeSpinRecovery,
    localization_needs_spin,
    localization_recovered,
)


def test_localization_needs_spin_for_awaiting_confirm_and_bad_prior():
    assert localization_needs_spin({"status": "awaiting_confirm", "shift_m": 6.0})
    assert localization_needs_spin({"status": "nav_hold"})
    assert localization_needs_spin(
        {"status": "low_quality", "previous_score": -0.3, "score": 0.5}
    )
    assert localization_needs_spin(
        {"status": "low_quality", "drifted": True, "large_jump": True, "score": 0.5}
    )
    assert not localization_needs_spin({"status": "ok", "good_match": True})
    assert not localization_needs_spin({"status": "low_quality", "previous_score": 0.8})


def test_localization_recovered_statuses():
    assert localization_recovered({"status": "corrected", "corrected": True})
    assert localization_recovered({"status": "ok", "good_match": True})
    assert localization_recovered({"good_match": True, "drifted": False, "status": "x"})
    assert not localization_recovered({"status": "awaiting_confirm"})


def test_spin_recovery_turns_then_pauses_and_kicks():
    cfg = LocalizeSpinConfig(
        enabled=True,
        step_deg=20.0,
        vel_rad_s=0.5,
        pause_s=0.3,
        max_total_yaw_deg=90.0,
        cooldown_s=1.0,
        min_spin_clearance_m=0.0,
    )
    spin = LocalizeSpinRecovery(cfg)
    check = {"status": "awaiting_confirm", "shift_m": 5.0, "previous_score": -0.2}
    assert spin.maybe_start(check, now=0.0)

    open_scan = conv.LaserScan2D(
        ranges=[3.0] * 8,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / 8,
        range_min=0.05,
        range_max=10.0,
    )
    cmd = spin.tick(check=check, pose_theta=0.0, scan=open_scan, now=0.0)
    assert cmd.phase == "turning"
    assert cmd.vtheta > 0

    # Complete one step by advancing yaw.
    step = math.radians(20.0)
    cmd = spin.tick(check=check, pose_theta=step, scan=open_scan, now=1.0)
    assert cmd.phase == "pausing"
    assert abs(cmd.vtheta) < 1e-9

    cmd = spin.tick(check=check, pose_theta=step, scan=open_scan, now=1.1)
    assert cmd.phase == "pausing"
    assert not cmd.kick_check

    cmd = spin.tick(check=check, pose_theta=step, scan=open_scan, now=1.4)
    assert cmd.kick_check

    # Recovered → done + cooldown.
    cmd = spin.tick(
        check={"status": "corrected", "corrected": True},
        pose_theta=step,
        scan=open_scan,
        now=1.5,
    )
    assert cmd.phase == "done"
    assert not spin.active()
    assert not spin.maybe_start(check, now=1.6)  # cooldown
    assert spin.maybe_start(check, now=3.0)


def test_spin_recovery_disabled():
    spin = LocalizeSpinRecovery(LocalizeSpinConfig(enabled=False))
    assert not spin.maybe_start({"status": "awaiting_confirm"})

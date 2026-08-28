"""Tests for ViamWorldIO cmd sanitization."""
from __future__ import annotations

from src.nav_builtin.viam_io import _is_near_zero_rpm_error, _sanitize_base_cmd


def test_sanitize_snaps_tiny_linear_to_zero():
    vx, vy, vt = _sanitize_base_cmd(0.03, 0.0, 0.5)
    assert vx == 0.0
    assert vt == 0.5


def test_sanitize_prefers_spin_when_tiny_vx_large_omega():
    # Matches the failure mode: vx~0.09 + vtheta~0.49 → one wheel ~0 RPM.
    vx, vy, vt = _sanitize_base_cmd(0.087, 0.0, 0.49)
    assert vx == 0.0
    assert abs(vt - 0.49) < 1e-9


def test_sanitize_keeps_normal_drive():
    vx, vy, vt = _sanitize_base_cmd(0.25, 0.0, 0.1)
    assert vx == 0.25
    assert abs(vt - 0.1) < 1e-9


def test_near_zero_rpm_error_detection():
    assert _is_near_zero_rpm_error(
        Exception("(<Status.UNKNOWN: 2>, 'Cannot move motor at an RPM that is nearly 0', None)")
    )
    assert not _is_near_zero_rpm_error(Exception("timeout"))

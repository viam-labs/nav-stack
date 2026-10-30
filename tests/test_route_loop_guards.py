"""Unit tests for the pure guards in scripts/route_loop.py.

These gate the live test harness: no goal may be sent on an untrusted SLAM
pose, a nav-only status error is never "idle", and deliberate holds do not
count as stall. No robot connection is involved.
"""
import pytest

from scripts.route_loop import (
    STALL_EXEMPT_OBS,
    UNTRUSTED_LC_STATUS,
    localization_trusted,
    nav_idle,
    recovery_trusted,
    row_blind,
    stall_exempt,
)

MIN = 0.2


def _slam(**over):
    """A Monitor.sample()['slam'] row as published on a healthy robot."""
    row = {"lc_status": "ok", "lc_score": 0.6, "lc_tick_match_score": 0.55,
           "lc_ray_mae_m": 0.4, "lc_shift_m": 0.02, "lc_corrected": False}
    row.update(over)
    return row


# -- localization_trusted ----------------------------------------------------

def test_trusted_on_healthy_row():
    assert localization_trusted(_slam(), min_tick_score=MIN) == (True, "ok")


@pytest.mark.parametrize("status", sorted(UNTRUSTED_LC_STATUS))
def test_untrusted_status_refuses(status):
    ok, why = localization_trusted(_slam(lc_status=status), min_tick_score=MIN)
    assert not ok
    assert status in why


def test_untrusted_status_set_is_exactly_the_reviewed_one():
    assert UNTRUSTED_LC_STATUS == {"ambiguous", "low_quality", "error",
                                   "awaiting_confirm", "rejected_shift", "refused_large"}


@pytest.mark.parametrize("status", ["ok", "corrected", "skipped", "idle", "unconfigured", "correction_failed"])
def test_other_statuses_pass(status):
    assert localization_trusted(_slam(lc_status=status), min_tick_score=MIN)[0]


def test_low_tick_score_refuses_even_with_ok_status():
    ok, why = localization_trusted(_slam(lc_tick_match_score=0.1), min_tick_score=MIN)
    assert not ok
    assert "tick_score" in why and "0.10" in why


def test_tick_score_at_threshold_passes():
    assert localization_trusted(_slam(lc_tick_match_score=MIN), min_tick_score=MIN)[0]


def test_zero_tick_score_is_a_real_number_not_missing():
    assert not localization_trusted(_slam(lc_tick_match_score=0.0), min_tick_score=MIN)[0]


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), "0.1"])
def test_missing_or_non_finite_tick_score_is_not_evidence(score):
    assert localization_trusted(_slam(lc_tick_match_score=score), min_tick_score=MIN)[0]


def test_missing_fields_pass():
    # Older module builds / no check yet: every key None, or no keys at all.
    assert localization_trusted({"lc_status": None, "lc_tick_match_score": None}, min_tick_score=MIN)[0]
    assert localization_trusted({}, min_tick_score=MIN)[0]


def test_status_takes_precedence_over_a_good_score():
    ok, why = localization_trusted(_slam(lc_status="ambiguous", lc_tick_match_score=0.9), min_tick_score=MIN)
    assert not ok and why == "lc_status=ambiguous"


def test_does_not_mutate_input():
    row = _slam(lc_status="low_quality")
    before = dict(row)
    localization_trusted(row, min_tick_score=MIN)
    assert row == before


# -- recovery_trusted --------------------------------------------------------

def test_recovery_corrected_check_vouches_for_pose():
    ok, why = recovery_trusted({"status": "corrected", "corrected": True}, _slam(lc_status="low_quality"),
                               min_tick_score=MIN)
    assert ok and why == "check_corrected"


def test_recovery_ok_check_vouches_for_pose():
    assert recovery_trusted({"status": "ok", "corrected": False}, _slam(lc_status="ambiguous"),
                            min_tick_score=MIN)[0]


@pytest.mark.parametrize("check", [None, {}, {"status": "skipped", "reason": "moving"},
                                   {"status": "awaiting_confirm", "corrected": False},
                                   {"status": "ambiguous", "corrected": False}])
def test_recovery_falls_back_to_live_status(check):
    assert recovery_trusted(check, _slam(), min_tick_score=MIN)[0]
    ok, why = recovery_trusted(check, _slam(lc_status="ambiguous"), min_tick_score=MIN)
    assert not ok and why == "lc_status=ambiguous"


def test_recovery_low_live_score_blocks_without_a_good_check():
    assert not recovery_trusted({"status": "low_quality"}, _slam(lc_tick_match_score=0.05),
                                min_tick_score=MIN)[0]


# -- row_blind / nav_idle ----------------------------------------------------

def _row(active, **over):
    row = {"t": 0.0, "nav": {"state": "active" if active else "succeeded", "active": active}, "slam": _slam()}
    row.update(over)
    return row


def test_idle_row():
    assert nav_idle(_row(False))
    assert not row_blind(_row(False))


def test_active_row_is_not_idle():
    assert not nav_idle(_row(True))


def test_conn_lost_row_is_blind_not_idle():
    row = _row(False, nav_error="closed", slam_error="closed", conn_lost=True)
    assert row_blind(row)
    assert not nav_idle(row)


def test_nav_only_error_is_blind_not_idle():
    # nav get_status failed while slam answered: Monitor leaves nav fields None.
    row = {"t": 0.0, "nav": {"state": None, "active": None}, "slam": _slam(), "nav_error": "deadline exceeded"}
    assert row_blind(row)
    assert not nav_idle(row)


def test_slam_only_error_does_not_blind_nav():
    row = _row(False, slam_error="boom")
    assert not row_blind(row)
    assert nav_idle(row)


def test_active_none_without_error_counts_as_idle():
    # A row from a module build that omits ``active``: nav answered, no goal.
    assert nav_idle({"t": 0.0, "nav": {"state": "idle", "active": None}, "slam": {}})
    assert nav_idle({"t": 0.0, "nav": {}, "slam": {}})


# -- stall_exempt ------------------------------------------------------------

@pytest.mark.parametrize("obs", sorted(STALL_EXEMPT_OBS))
def test_hold_states_are_exempt(obs):
    assert stall_exempt({"obstacle": obs, "active": True})


def test_exempt_set_is_exactly_the_reviewed_one():
    assert STALL_EXEMPT_OBS == {"loc_hold", "loc_refine", "loc_yield", "wait"}


@pytest.mark.parametrize("obs", ["planning", "backup", "narrow_reverse", "", None, "blocked"])
def test_moving_and_replanning_states_are_not_exempt(obs):
    # "planning" stays counted on purpose: a replan storm is what stall catches.
    assert not stall_exempt({"obstacle": obs, "localization_yield": False})


def test_localization_yield_flag_is_exempt_regardless_of_obstacle():
    assert stall_exempt({"obstacle": "planning", "localization_yield": True})
    assert not stall_exempt({"obstacle": "planning", "localization_yield": None})

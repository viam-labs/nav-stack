import math

from src.nav.localize.loc_state import (
    LOC_DEGRADED,
    LOC_LOST,
    LOC_MAP_MISMATCH,
    LOC_TRACKING,
    LOC_UNKNOWN,
    LocalizationStateTracker,
    describe_state,
)


def _tracker():
    clock = {"t": 1000.0}
    tracker = LocalizationStateTracker(clock=lambda: clock["t"])
    return tracker, clock


def test_two_bad_cycles_make_lost_and_skips_keep_it():
    tracker, clock = _tracker()
    assert tracker.state == LOC_UNKNOWN
    tracker.observe({"status": "ambiguous", "score": -0.03, "shift_m": 3.274, "shift_deg": 23.96})
    assert tracker.state == LOC_DEGRADED
    clock["t"] += 20.0
    tracker.observe({"status": "ambiguous", "score": -0.01, "shift_m": 3.1, "shift_deg": 22.0})
    assert tracker.state == LOC_LOST
    # The robot starts driving; skipped checks say nothing about the pose.
    clock["t"] += 74.0
    tracker.observe({"status": "skipped", "reason": "navigation_active"})
    snap = tracker.snapshot()
    assert snap["state"] == LOC_LOST
    assert snap["since_s"] == 74.0
    assert snap["last_attempt"]["status"] == "ambiguous"
    assert snap["last_attempt"]["age_s"] == 74.0
    assert "initial pose" in snap["suggested_action"]
    text = tracker.describe()
    assert text.startswith("lost for 74 s (ambiguous, best match -0.01 at 3.1 m / 22 deg)")


def test_correction_or_seed_returns_to_tracking():
    tracker, clock = _tracker()
    tracker.observe({"status": "low_quality", "score": 0.2})
    tracker.observe({"status": "low_quality", "score": 0.2})
    assert tracker.state == LOC_LOST
    tracker.observe({"status": "ok", "corrected": True, "good_match": True, "score": 0.7})
    assert tracker.state == LOC_TRACKING
    assert tracker.snapshot()["bad_cycles"] == 0
    tracker.observe({"status": "refused_large", "score": 0.3})
    assert tracker.state == LOC_DEGRADED
    tracker.seeded("set_initial_pose")
    assert tracker.state == LOC_TRACKING
    assert tracker.reason == "set_initial_pose"


def test_tick_prior_already_good_clears_lost():
    """Engine hold when the published pose already fits must not stay lost."""
    tracker, _ = _tracker()
    tracker.observe({"status": "nav_hold", "score": 0.4})
    tracker.observe({"status": "nav_hold", "score": 0.4})
    assert tracker.state == LOC_LOST
    tracker.observe(
        {"status": "previous_better", "score": 0.52, "prior_score": 0.58}
    )
    assert tracker.state == LOC_TRACKING
    assert tracker.reason == "previous_better"
    assert tracker.snapshot()["bad_cycles"] == 0


def test_weak_unshifted_match_is_degraded_not_lost():
    tracker, _ = _tracker()
    for _ in range(5):
        tracker.observe({"status": "ok", "good_match": False, "drifted": False, "score": 0.4})
    assert tracker.state == LOC_DEGRADED
    assert tracker.snapshot()["bad_cycles"] == 0


def test_full_map_that_fits_nowhere_is_a_map_mismatch():
    tracker, _ = _tracker()
    tracker.observe({"status": "low_quality", "match_mode": "full_map", "score": 0.18})
    assert tracker.state == LOC_MAP_MISMATCH
    assert "remap" in tracker.snapshot()["suggested_action"]


def test_describe_state_from_a_published_snapshot():
    assert describe_state(None) is None
    assert describe_state({}) is None
    text = describe_state(
        {
            "state": "lost",
            "since_s": 94.2,
            "last_attempt": {"status": "ambiguous", "score": 0.31, "shift_m": 3.3, "shift_deg": 24.0},
            "suggested_action": "set an initial pose near the robot",
        }
    )
    assert text == (
        "lost for 94 s (ambiguous, best match 0.31 at 3.3 m / 24 deg); "
        "set an initial pose near the robot"
    )
    assert math.isfinite(94.2)

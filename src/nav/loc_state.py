"""One operator-facing localization state built from the watchdog's checks.

The relocalize cycle publishes a fresh dict every run (``ok``, ``ambiguous``,
``low_quality``, ``skipped`` ...). Each one describes a single attempt; none
says how long the robot has been unable to place itself or what to do about
it. This tracker keeps that across cycles so ``get_status`` and the goal
refusal can say "lost for 94 s, best match 0.31 at 3.3 m: set an initial pose".
"""
from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Optional

LOC_UNKNOWN = "unknown"
LOC_TRACKING = "tracking"
LOC_DEGRADED = "degraded"
LOC_LOST = "lost"
LOC_MAP_MISMATCH = "map_mismatch"

# Results that say nothing about the pose: keep the current state.
_NO_CHANGE = frozenset({"skipped", "unconfigured", "idle", "error", ""})
# The pose explains the scan, or a correction landed.
_GOOD = frozenset({"corrected", "previous_better"})
# The scan does not explain the published pose and nothing was applied.
_BAD = frozenset(
    {"ambiguous", "refused_large", "nav_hold", "awaiting_confirm", "low_quality"}
)
# A full-map best under this fits nowhere: the map is stale or wrong, and
# another pose search will not help.
MAP_MISMATCH_SCORE = 0.30

_ATTEMPT_KEYS = (
    "status",
    "match_mode",
    "score",
    "ray_mae_m",
    "shift_m",
    "shift_deg",
    "second_best_score",
    "tick_match_score",
)


def _finite(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class LocalizationStateTracker:
    """Fold watchdog results into ``tracking`` / ``degraded`` / ``lost``."""

    def __init__(
        self,
        *,
        lost_after: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lost_after = max(1, int(lost_after))
        self._clock = clock
        self.state = LOC_UNKNOWN
        self.reason: Optional[str] = None
        self._since = clock()
        self._bad_streak = 0
        self._last_attempt: Optional[dict] = None

    # -- inputs -----------------------------------------------------------
    def seeded(self, reason: str = "seeded") -> None:
        """A pose was placed (manual seed, applied localize, loaded map)."""
        self._bad_streak = 0
        self._set(LOC_TRACKING, reason)

    def observe(self, result: Mapping) -> None:
        status = str(result.get("status") or "")
        if status in _NO_CHANGE:
            return
        attempt = {k: result.get(k) for k in _ATTEMPT_KEYS if result.get(k) is not None}
        attempt["at"] = self._clock()
        self._last_attempt = attempt

        if bool(result.get("corrected")) or status in _GOOD:
            self._bad_streak = 0
            self._set(LOC_TRACKING, status or "corrected")
            return
        if status in ("ok", "soft_loc_resume"):
            self._bad_streak = 0
            if bool(result.get("good_match")):
                self._set(LOC_TRACKING, "good_match")
            else:
                self._set(LOC_DEGRADED, "weak_match")
            return
        if status in _BAD:
            self._bad_streak += 1
            mode = str(result.get("match_mode") or "")
            score = _finite(result.get("score"))
            if mode.startswith("full_map") and score is not None and score < MAP_MISMATCH_SCORE:
                self._set(LOC_MAP_MISMATCH, f"full_map_best_{score:.2f}")
                return
            if self._bad_streak >= self._lost_after:
                self._set(LOC_LOST, status)
            else:
                self._set(LOC_DEGRADED, status)
            return
        # Unknown status string: record the attempt, keep the state.

    # -- outputs ----------------------------------------------------------
    def snapshot(self) -> dict:
        now = self._clock()
        out: dict = {
            "state": self.state,
            "since_s": round(max(0.0, now - self._since), 1),
            "reason": self.reason,
            "bad_cycles": self._bad_streak,
        }
        if self._last_attempt is not None:
            attempt = dict(self._last_attempt)
            at = attempt.pop("at", None)
            if at is not None:
                attempt["age_s"] = round(max(0.0, now - float(at)), 1)
            out["last_attempt"] = attempt
        action = self.suggested_action()
        if action:
            out["suggested_action"] = action
        return out

    def suggested_action(self) -> Optional[str]:
        if self.state == LOC_LOST:
            return (
                "set an initial pose near the robot, or drive 1-2 m into open "
                "space and stop so the watchdog can search again"
            )
        if self.state == LOC_MAP_MISMATCH:
            return (
                "the saved map does not match the scan anywhere; load the right "
                "map or remap this area"
            )
        if self.state == LOC_DEGRADED:
            return "pose fits loosely; the watchdog refines it at the next check"
        return None

    def describe(self) -> str:
        """One line for a goal refusal or a log."""
        snap = self.snapshot()
        parts = [f"{snap['state']} for {snap['since_s']:.0f} s"]
        attempt = snap.get("last_attempt") or {}
        status = attempt.get("status")
        score = _finite(attempt.get("score"))
        shift_m = _finite(attempt.get("shift_m"))
        shift_deg = _finite(attempt.get("shift_deg"))
        detail = []
        if status:
            detail.append(str(status))
        if score is not None:
            best = f"best match {score:.2f}"
            if shift_m is not None and shift_deg is not None:
                best += f" at {shift_m:.1f} m / {shift_deg:.0f} deg"
            detail.append(best)
        if detail:
            parts.append("(" + ", ".join(detail) + ")")
        action = self.suggested_action()
        text = " ".join(parts)
        if action:
            text += "; " + action
        return text

    def _set(self, state: str, reason: Optional[str]) -> None:
        if state != self.state:
            self.state = state
            self._since = self._clock()
        self.reason = reason


def describe_state(snapshot: Optional[Mapping]) -> Optional[str]:
    """Same one-liner from a published snapshot (nav side, no tracker)."""
    if not isinstance(snapshot, Mapping) or not snapshot.get("state"):
        return None
    state = str(snapshot.get("state"))
    since = _finite(snapshot.get("since_s"))
    text = state if since is None else f"{state} for {since:.0f} s"
    attempt = snapshot.get("last_attempt")
    if isinstance(attempt, Mapping):
        detail = []
        if attempt.get("status"):
            detail.append(str(attempt["status"]))
        score = _finite(attempt.get("score"))
        if score is not None:
            best = f"best match {score:.2f}"
            shift_m = _finite(attempt.get("shift_m"))
            shift_deg = _finite(attempt.get("shift_deg"))
            if shift_m is not None and shift_deg is not None:
                best += f" at {shift_m:.1f} m / {shift_deg:.0f} deg"
            detail.append(best)
        if detail:
            text += " (" + ", ".join(detail) + ")"
    action = snapshot.get("suggested_action")
    if action:
        text += "; " + str(action)
    return text

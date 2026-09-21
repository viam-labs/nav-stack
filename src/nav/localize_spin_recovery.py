"""In-place yaw steps to break localization ambiguity.

When the published map pose no longer explains the scan (or a large jump is
awaiting confirm), a short rotate → pause → rematch cycle often finds a better
peak than sitting still — especially for heading-wrong / corridor twins.

Matching is skipped while ``|yaw_rate|`` is high, so this deliberately uses
brief turns followed by a full stop before asking SLAM to re-check.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from ..geom import conversions as conv
from .simple_motion import spin_clearance_m


@dataclass(frozen=True)
class LocalizeSpinConfig:
    enabled: bool = True
    step_deg: float = 25.0
    vel_rad_s: float = 0.30
    pause_s: float = 0.85
    max_total_yaw_deg: float = 360.0
    cooldown_s: float = 45.0
    min_spin_clearance_m: float = 0.35


@dataclass
class SpinCommand:
    """Body-frame command for one control tick (ROS +x forward)."""

    vx: float = 0.0
    vtheta: float = 0.0
    phase: str = "idle"  # idle|turning|pausing|blocked|done|cooldown
    kick_check: bool = False
    yaw_turned_deg: float = 0.0
    steps: int = 0


def localization_needs_spin(check: Optional[Mapping[str, Any]]) -> bool:
    """True when a gentle spin is likely to help regain map pose."""
    if not isinstance(check, Mapping):
        return False
    status = str(check.get("status") or "")
    if status in ("awaiting_confirm", "nav_hold"):
        return True
    if status == "low_quality":
        prev = check.get("previous_score")
        try:
            prev_f = float(prev) if prev is not None else None
        except (TypeError, ValueError):
            prev_f = None
        # Published pose does not explain the scan.
        if prev_f is not None and prev_f < 0.45:
            return True
        if check.get("drifted") and check.get("large_jump"):
            return True
    return False


def localization_recovered(check: Optional[Mapping[str, Any]]) -> bool:
    """True when spin recovery should stop (pose trusted again)."""
    if not isinstance(check, Mapping):
        return False
    status = str(check.get("status") or "")
    if status in ("ok", "corrected", "previous_better", "soft_loc_resume", "idle"):
        return True
    if status == "awaiting_confirm":
        return False
    if bool(check.get("corrected")):
        return True
    if bool(check.get("good_match")) and not bool(check.get("drifted")):
        return True
    return False


class LocalizeSpinRecovery:
    """Turn → pause → rematch state machine (sync; safe for nav worker thread)."""

    def __init__(self, cfg: Optional[LocalizeSpinConfig] = None):
        self._cfg = cfg or LocalizeSpinConfig()
        self.reset()

    @property
    def cfg(self) -> LocalizeSpinConfig:
        return self._cfg

    def reset(self) -> None:
        self._phase = "idle"
        self._turn_dir = 1.0
        self._yaw_at_step_start: Optional[float] = None
        self._yaw_turned_rad = 0.0
        self._steps = 0
        self._phase_since = 0.0
        self._cooldown_until = 0.0
        self._active = False

    def active(self) -> bool:
        return self._active

    def maybe_start(self, check: Optional[Mapping[str, Any]], *, now: Optional[float] = None) -> bool:
        """Begin a recovery cycle if lost and not in cooldown."""
        now = time.monotonic() if now is None else float(now)
        if not self._cfg.enabled:
            return False
        if self._active:
            return True
        if now < self._cooldown_until:
            return False
        if not localization_needs_spin(check):
            return False
        self._active = True
        self._phase = "turning"
        self._turn_dir = 1.0
        self._yaw_at_step_start = None
        self._yaw_turned_rad = 0.0
        self._steps = 0
        self._phase_since = now
        return True

    def tick(
        self,
        *,
        check: Optional[Mapping[str, Any]],
        pose_theta: float,
        scan: Optional[conv.LaserScan2D],
        now: Optional[float] = None,
    ) -> SpinCommand:
        now = time.monotonic() if now is None else float(now)
        if not self._cfg.enabled:
            self.reset()
            return SpinCommand(phase="idle")

        if not self._active:
            if now < self._cooldown_until:
                return SpinCommand(phase="cooldown")
            return SpinCommand(phase="idle")

        if localization_recovered(check):
            return self._finish(now, phase="done")

        step_rad = math.radians(max(5.0, float(self._cfg.step_deg)))
        max_rad = math.radians(max(step_rad, float(self._cfg.max_total_yaw_deg)))
        vel = max(0.08, float(self._cfg.vel_rad_s))
        pause_s = max(0.2, float(self._cfg.pause_s))
        min_clear = max(0.0, float(self._cfg.min_spin_clearance_m))

        if self._yaw_turned_rad >= max_rad - 1e-3:
            return self._finish(now, phase="done")

        if self._phase == "turning":
            if scan is not None and min_clear > 0.0:
                clear = spin_clearance_m(scan)
                if clear < min_clear:
                    # Try the other direction once; otherwise wait.
                    if self._steps == 0 and self._turn_dir > 0:
                        self._turn_dir = -1.0
                        self._yaw_at_step_start = None
                        self._phase_since = now
                    else:
                        self._phase = "pausing"
                        self._phase_since = now
                        return SpinCommand(
                            phase="blocked",
                            yaw_turned_deg=math.degrees(self._yaw_turned_rad),
                            steps=self._steps,
                        )

            if self._yaw_at_step_start is None:
                self._yaw_at_step_start = float(pose_theta)
                self._phase_since = now

            turned = abs(
                conv.normalize_angle(float(pose_theta) - float(self._yaw_at_step_start))
            )
            # Also bound by time so a stuck pose still advances the cycle.
            timed_out = (now - self._phase_since) >= (step_rad / vel) + 0.75
            if turned >= step_rad - 1e-3 or timed_out:
                self._yaw_turned_rad += max(turned, step_rad if timed_out else turned)
                self._steps += 1
                self._phase = "pausing"
                self._phase_since = now
                self._yaw_at_step_start = None
                return SpinCommand(
                    phase="pausing",
                    yaw_turned_deg=math.degrees(self._yaw_turned_rad),
                    steps=self._steps,
                )
            return SpinCommand(
                vtheta=self._turn_dir * vel,
                phase="turning",
                yaw_turned_deg=math.degrees(self._yaw_turned_rad + turned),
                steps=self._steps,
            )

        if self._phase == "pausing":
            if now - self._phase_since < pause_s:
                return SpinCommand(
                    phase="pausing",
                    yaw_turned_deg=math.degrees(self._yaw_turned_rad),
                    steps=self._steps,
                )
            # End of pause: stop is already commanded; kick a rematch once.
            self._phase = "turning"
            self._yaw_at_step_start = None
            self._phase_since = now
            return SpinCommand(
                phase="pausing",
                kick_check=True,
                yaw_turned_deg=math.degrees(self._yaw_turned_rad),
                steps=self._steps,
            )

        return self._finish(now, phase="done")

    def _finish(self, now: float, *, phase: str) -> SpinCommand:
        turned = math.degrees(self._yaw_turned_rad)
        steps = self._steps
        self._active = False
        self._phase = "idle"
        self._cooldown_until = now + max(0.0, float(self._cfg.cooldown_s))
        self._yaw_at_step_start = None
        return SpinCommand(
            phase=phase,
            yaw_turned_deg=turned,
            steps=steps,
        )

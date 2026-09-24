"""Confirm large automatic pose jumps before applying them.

Tick-level builtin scan matching already requires agreeing frames for big
corrections. Async paths (periodic relocalize, mapping revisit, seed /
startup localize) historically applied a single match — including weak
full-map peaks that yank the robot metres away. This gate is the shared
policy for those paths; manual ``relocalize`` / ``apply=true`` bypasses it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..geom import conversions as conv


@dataclass(frozen=True)
class JumpDecision:
    should_apply: bool
    large_jump: bool
    confirm_count: int
    confirm_needed: int
    status: str  # apply_small | apply_confirmed | awaiting_confirm | forced
    shift_m: float
    shift_deg: float


class PoseJumpGate:
    """Require N agreeing candidates before applying a large map jump."""

    def __init__(
        self,
        *,
        confirm_count: int = 2,
        agree_m: float = 0.4,
        agree_deg: float = 15.0,
        large_m: float = 0.75,
        large_deg: float = 25.0,
    ):
        self.confirm_needed = max(1, int(confirm_count))
        self.agree_m = max(0.0, float(agree_m))
        self.agree_deg = max(0.0, float(agree_deg))
        self.large_m = max(0.0, float(large_m))
        self.large_deg = max(0.0, float(large_deg))
        self._pending: Optional[conv.Pose2D] = None
        self._count = 0

    def clear(self) -> None:
        self._pending = None
        self._count = 0

    def snapshot(self) -> dict:
        pending = None
        if self._pending is not None:
            pending = {
                "x": self._pending.x,
                "y": self._pending.y,
                "theta": self._pending.theta,
            }
        return {
            "confirm_count": self._count,
            "confirm_needed": self.confirm_needed,
            "pending": pending,
        }

    def evaluate(
        self,
        current: conv.Pose2D,
        candidate: conv.Pose2D,
        *,
        force: bool = False,
    ) -> JumpDecision:
        shift_m = math.hypot(candidate.x - current.x, candidate.y - current.y)
        shift_deg = abs(
            math.degrees(conv.normalize_angle(candidate.theta - current.theta))
        )
        large = shift_m >= self.large_m or shift_deg >= self.large_deg

        if force:
            self.clear()
            return JumpDecision(
                should_apply=True,
                large_jump=large,
                confirm_count=self.confirm_needed,
                confirm_needed=self.confirm_needed,
                status="forced",
                shift_m=shift_m,
                shift_deg=shift_deg,
            )

        if not large:
            self.clear()
            return JumpDecision(
                should_apply=True,
                large_jump=False,
                confirm_count=0,
                confirm_needed=self.confirm_needed,
                status="apply_small",
                shift_m=shift_m,
                shift_deg=shift_deg,
            )

        if self.confirm_needed <= 1:
            self.clear()
            return JumpDecision(
                should_apply=True,
                large_jump=True,
                confirm_count=1,
                confirm_needed=1,
                status="apply_confirmed",
                shift_m=shift_m,
                shift_deg=shift_deg,
            )

        if self._pending is not None and self._poses_agree(candidate, self._pending):
            self._count += 1
            self._pending = candidate
        else:
            self._pending = candidate
            self._count = 1

        if self._count >= self.confirm_needed:
            self.clear()
            return JumpDecision(
                should_apply=True,
                large_jump=True,
                confirm_count=self.confirm_needed,
                confirm_needed=self.confirm_needed,
                status="apply_confirmed",
                shift_m=shift_m,
                shift_deg=shift_deg,
            )

        return JumpDecision(
            should_apply=False,
            large_jump=True,
            confirm_count=self._count,
            confirm_needed=self.confirm_needed,
            status="awaiting_confirm",
            shift_m=shift_m,
            shift_deg=shift_deg,
        )

    def _poses_agree(self, a: conv.Pose2D, b: conv.Pose2D) -> bool:
        if math.hypot(a.x - b.x, a.y - b.y) > self.agree_m:
            return False
        dyaw = abs(math.degrees(conv.normalize_angle(a.theta - b.theta)))
        return dyaw <= self.agree_deg


def should_hold_drive_for_pose_jump(check: Optional[object]) -> bool:
    """True when nav must stop for a localization jump / uncertainty hold.

    Holds for:
    - ``awaiting_confirm`` — large jump gated until N agreeing matches
    - ``nav_hold`` — during nav, large false peak and/or soft loc where the
      published pose no longer explains the scan (do not keep driving)

    Does **not** hold for ``soft_loc_resume`` (soft quality, no large jump,
    hold timer expired — continue on the published pose) or ``refused_large``
    (mid-nav refine rejected a hallway-sized yank — keep the published pose).
    """
    if not isinstance(check, dict):
        return False
    return str(check.get("status") or "") in ("awaiting_confirm", "nav_hold")


def candidate_beats_previous(
    *,
    previous_score: float,
    candidate_score: float,
    shift_m: float,
    shift_deg: float,
    previous_ray_mae_m: Optional[float] = None,
    candidate_ray_mae_m: Optional[float] = None,
) -> bool:
    """True if the jumped pose is a clear win over staying put.

    Large jumps need a stronger score margin (same idea as tick ``refine_pose``)
    so a weak secondary peak cannot yank the robot metres away while the scan
    still fits the previous pose.
    """
    if not math.isfinite(candidate_score):
        return False
    if not math.isfinite(previous_score):
        return True
    dyaw_rad = math.radians(abs(float(shift_deg)))
    # Mild bar: base + distance + yaw. A 7 m false peak must beat previous by ~0.8.
    need = 0.06 + 0.10 * max(0.0, float(shift_m)) + 0.08 * (
        dyaw_rad / (math.pi / 4.0)
    )
    if candidate_score < previous_score + need:
        return False
    # Prefer previous when its ray alignment is clearly better.
    if (
        previous_ray_mae_m is not None
        and candidate_ray_mae_m is not None
        and math.isfinite(previous_ray_mae_m)
        and math.isfinite(candidate_ray_mae_m)
        and previous_ray_mae_m + 0.15 < candidate_ray_mae_m
    ):
        return False
    return True

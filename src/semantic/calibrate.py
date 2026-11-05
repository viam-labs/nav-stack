"""Camera mount + time-offset calibration (stub for Phase 1).

Collects landmark bearing residuals while lidar tracking is strong and solves
for mount yaw/x/y and a constant capture-time offset. Wired in a later commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MountGuess:
    x: float
    y: float
    z: float
    theta: float
    time_offset_s: float = 0.0


@dataclass
class CalibSample:
    """One observation used by the least-squares mount solve."""

    bearing_meas_rad: float
    bearing_pred_rad: float
    range_m: float
    captured_at: float


def solve_mount(
    samples: List[CalibSample],
    *,
    prior: Optional[MountGuess] = None,
) -> Tuple[MountGuess, dict]:
    """Placeholder: returns prior (or zeros) until the solver lands."""
    del samples
    guess = prior or MountGuess(0.0, 0.0, 0.0, 0.0, 0.0)
    return guess, {"status": "stub", "n": 0}

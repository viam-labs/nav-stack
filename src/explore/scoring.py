"""Frontier cost and failed-goal blacklist for exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from .grid import Frontier

# Metres of path a frontier's size is worth, per metre of frontier, capped so
# one huge opening far away does not outrank everything near the robot.
GAIN_PER_M = 0.3
GAIN_CAP_M = 6.0
# Metres of path one radian of heading change is worth. Fewer and gentler
# turns keep scan matching healthy and avoid back-and-forth zig-zags, and a
# rectangular body turning round beside furniture is how it gets wedged:
# turning round costs as much as about 3 m of driving.
TURN_M_PER_RAD = 1.0


@dataclass
class Blacklist:
    radius_m: float = 1.0
    ttl_s: float = 600.0
    _entries: List[Tuple[float, float, float]] = field(default_factory=list)

    def add(self, x: float, y: float, now: float) -> None:
        self._entries.append((x, y, now + self.ttl_s))

    def blocked(self, x: float, y: float, now: float) -> bool:
        self._entries = [e for e in self._entries if e[2] > now]
        return any(
            math.hypot(ex - x, ey - y) <= self.radius_m for ex, ey, _ in self._entries
        )


def cost(f: Frontier, x: float, y: float, theta: float) -> float:
    """Lower is better: path length, plus turning, minus frontier size."""
    turn = 0.0
    if math.hypot(f.goal_x - x, f.goal_y - y) > 1e-6:
        bearing = math.atan2(f.goal_y - y, f.goal_x - x)
        turn = abs(math.atan2(math.sin(bearing - theta), math.cos(bearing - theta)))
    return f.path_m + TURN_M_PER_RAD * turn - GAIN_PER_M * min(f.size_m, GAIN_CAP_M)

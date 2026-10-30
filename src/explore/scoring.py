"""Goal scoring and failed-goal blacklist for exploration."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .grid import Candidate


@dataclass
class Blacklist:
    radius_m: float = 1.0
    ttl_s: float = 600.0
    _entries: List[Tuple[float, float, float]] = field(default_factory=list)

    def add(self, x: float, y: float, now: float) -> None:
        self._entries.append((x, y, now + self.ttl_s))

    def blocked(self, x: float, y: float, now: float) -> bool:
        self._entries = [e for e in self._entries if e[2] > now]
        return any(math.hypot(ex - x, ey - y) <= self.radius_m for ex, ey, _ in self._entries)


def score(gain_cells: int, path_len_m: float, min_path_m: float = 0.5) -> float:
    return gain_cells / max(path_len_m, min_path_m)


@dataclass
class Scored:
    candidate: Candidate
    gain: int
    path_len_m: float
    score: float


def rank(scored: Sequence[Scored], min_score: float) -> List[Scored]:
    return sorted((s for s in scored if s.score >= min_score), key=lambda s: -s.score)

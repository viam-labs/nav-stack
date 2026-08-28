"""Shared types for builtin occupancy SLAM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ros import conversions as conv


@dataclass
class LogOddsGrid:
    """Expandable 2D log-odds occupancy grid (row-major, origin at min corner)."""

    log_odds: np.ndarray  # float32, shape (H, W)
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.log_odds.shape[0])

    @property
    def width(self) -> int:
        return int(self.log_odds.shape[1])

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        col = int(np.floor((x_m - self.origin_x) / self.resolution))
        row = int(np.floor((y_m - self.origin_y) / self.resolution))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return float(x), float(y)


@dataclass
class SlamState:
    pose: conv.Pose2D
    map_to_odom: conv.Pose2D
    mode: str
    generation: int = 0

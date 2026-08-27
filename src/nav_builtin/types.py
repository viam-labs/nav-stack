"""Shared types for the ROS-free builtin navigator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ros import conversions as conv

# Re-export for callers that stay inside nav_builtin.
Pose2D = conv.Pose2D


@dataclass(frozen=True)
class OccupancyGrid:
    """Nav2-style occupancy grid (row-major, height x width).

    Cells: -1 unknown, 0 free, 1..100 occupied (100 = lethal).
    """

    grid: np.ndarray  # int16
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        col = int((x_m - self.origin_x) / self.resolution)
        row = int((y_m - self.origin_y) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


@dataclass(frozen=True)
class Path2D:
    """World-frame polyline (meters). ``poses`` may include yaw on the last point."""

    points: Tuple[Tuple[float, float], ...]
    goal_theta: float = 0.0

    @property
    def empty(self) -> bool:
        return len(self.points) < 2

    def as_plan_points(self, max_points: int = 400) -> List[dict]:
        pts = list(self.points)
        if len(pts) > max_points and max_points >= 2:
            # Uniform downsample keeping endpoints.
            idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = [pts[i] for i in idx]
        out: List[dict] = []
        for i, (x, y) in enumerate(pts):
            theta = self.goal_theta if i == len(pts) - 1 else 0.0
            out.append({"x": float(x), "y": float(y), "theta": float(theta)})
        return out


@dataclass
class NavStatus:
    state: str = "idle"  # idle | active | succeeded | failed | canceled
    active: bool = False
    goal: Optional[dict] = None
    pose: Optional[dict] = None
    error_msg: str = ""
    path: Optional[List[dict]] = None
    length_m: float = 0.0
    motion: str = "builtin"
    # Live follower diagnostics (obstacle / cmd / bearing) while active.
    progress: Optional[dict] = None

    def to_dict(self) -> dict:
        out = {
            "state": self.state,
            "active": self.active,
            "goal": dict(self.goal) if self.goal else None,
            "pose": dict(self.pose) if self.pose else None,
            "motion": self.motion,
            "error_msg": self.error_msg,
            "length_m": self.length_m,
        }
        if self.path is not None:
            out["path"] = list(self.path)
        if self.progress is not None:
            out["progress"] = dict(self.progress)
            # Convenience mirrors for describe_motion / get_status consumers.
            for key in (
                "obstacle",
                "forward_clearance_m",
                "cmd_vx_mps",
                "cmd_vtheta_rad_s",
                "bearing_error_rad",
                "distance_remaining_m",
            ):
                if key in self.progress and key not in out:
                    out[key] = self.progress[key]
        return out


@dataclass
class PlanResult:
    feasible: bool
    path: Path2D = field(default_factory=lambda: Path2D(points=()))
    error_code: int = 0
    error_msg: str = ""
    planning_time_s: float = 0.0

    def to_preview_dict(
        self,
        *,
        goal: Sequence[float],
        start: Optional[conv.Pose2D],
        planner_id: str = "LazyThetaStar",
        max_points: int = 400,
    ) -> dict:
        points = self.path.as_plan_points(max_points=max_points) if self.feasible else []
        length = 0.0
        for i in range(1, len(points)):
            length += float(
                np.hypot(
                    points[i]["x"] - points[i - 1]["x"],
                    points[i]["y"] - points[i - 1]["y"],
                )
            )
        return {
            "feasible": self.feasible and len(points) >= 2,
            "error_code": int(self.error_code),
            "error_msg": self.error_msg,
            "planner_id": planner_id,
            "planning_time_s": round(float(self.planning_time_s), 4),
            "length_m": round(length, 3),
            "path": points,
            "goal": {
                "x": float(goal[0]),
                "y": float(goal[1]),
                "theta": float(goal[2]) if len(goal) > 2 else 0.0,
            },
            "start": (
                {"x": float(start.x), "y": float(start.y), "theta": float(start.theta)}
                if start is not None
                else None
            ),
            "point_count": len(points),
        }

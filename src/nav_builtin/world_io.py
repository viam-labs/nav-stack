"""World I/O protocol for the builtin navigator (map / pose / scan / drive)."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..ros import conversions as conv


@runtime_checkable
class WorldIO(Protocol):
    """Sync facade over whatever supplies map, pose, scan, and base velocity.

    Implementations must be safe to call from a background nav worker thread.
    """

    def get_map(self) -> Optional[dict]:
        """Bridge-style map dict: grid, resolution, origin_x, origin_y."""
        ...

    def get_pose(self) -> Optional[conv.Pose2D]:
        ...

    def get_scan(self, max_age_s: float = 2.0) -> Optional[conv.LaserScan2D]:
        ...

    def set_velocity(self, vx: float, vy: float, vtheta: float) -> None:
        """Body-frame cmd (m/s, rad/s), ROS convention (+x forward)."""
        ...

    def stop(self) -> None:
        ...

    def set_viz_plan(
        self,
        path_xy: tuple,
        goal: Optional[tuple] = None,
    ) -> None:
        """Optional nav-camera overlay; default no-op."""
        return None

    def set_viz_costmap(self, costmap: dict) -> None:
        """Optional inflated costmap for nav-camera; default no-op."""
        return None

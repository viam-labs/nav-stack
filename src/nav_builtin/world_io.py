"""World I/O protocol for the builtin navigator (map / pose / scan / drive)."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..geom import conversions as conv


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

    def get_scan(
        self, max_age_s: float = 2.0, *, include_obstacles_only: bool = True
    ) -> Optional[conv.LaserScan2D]:
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

    def set_viz_local_costmap(self, costmap: dict) -> None:
        """Optional rolling local costmap for operator UIs; default no-op."""
        return None

    def get_localization_hold(self) -> Optional[dict]:
        """If non-None, nav must stop translating (large pose jump / loc hold)."""
        return None

    def check_localization(
        self,
        *,
        allow_during_navigation: bool = True,
        full_map_escalation: str = "still_bad",
        apply: Optional[bool] = None,
    ) -> Optional[dict]:
        """Optional SLAM local refine. Implementations may return None."""
        return None

    def set_above_cart(self, enabled: bool, cart_height_m: Optional[float] = None) -> None:
        """Remember depth returns that leave the vertical view. Default no-op."""
        return None

    def get_above_cart_frames(self) -> list:
        """Overhead clouds ``(stamp, xyz_base, sensor, capture_pose)``."""
        return []

    def refine_stuck_pose(self) -> Optional[dict]:
        """One small local scan match when planning cannot leave the current cell.

        Implementations return a dict with ``corrected`` when the pose moved.
        Default is a no-op so a real obstacle still fails the goal.
        """
        return None

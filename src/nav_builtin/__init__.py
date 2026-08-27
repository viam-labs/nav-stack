"""ROS-free builtin navigation stack (costmap + A* + path follower).

Used as the default ``nav_backend`` for MoveOnMap / navigate_* / plan_to_*.
Nav2 remains available behind ``nav_backend: "nav2"``.
"""
from .bridge_io import BridgeWorldIO
from .navigator import BuiltinNavigator
from .types import NavStatus, OccupancyGrid, Path2D, PlanResult, Pose2D
from .world_io import WorldIO

__all__ = [
    "BridgeWorldIO",
    "BuiltinNavigator",
    "NavStatus",
    "OccupancyGrid",
    "Path2D",
    "PlanResult",
    "Pose2D",
    "WorldIO",
]

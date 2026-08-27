"""In-process viz snapshot for builtin nav (nav-camera / get_costmap)."""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np

from ..ros import conversions as conv


class NavVizStore:
    """Thread-safe stand-in for BridgeNode.viz_snapshot() without ROS."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._map: Optional[dict] = None
        self._costmap: Optional[dict] = None
        self._global_plan: tuple = ()
        self._plan_history: list = []
        self._local_plan: tuple = ()
        self._footprint: tuple = ()
        self._goal: Optional[Tuple[float, float, float]] = None
        self._pose: Optional[Tuple[float, float, float]] = None
        self._history_len: int = 8

    def set_history_len(self, n: int) -> None:
        self._history_len = max(1, int(n))

    def enable_viz(self, history_len: int = 8) -> None:
        """BridgeNode-compatible hook used by nav-camera / get_costmap."""
        self.set_history_len(history_len)

    def viz_snapshot(self) -> Dict:
        return self.snapshot()

    def get_map(self) -> Optional[dict]:
        with self._lock:
            return self._map

    def set_map(self, map_data: Optional[dict]) -> None:
        with self._lock:
            self._map = map_data

    def set_costmap(self, costmap: Optional[dict]) -> None:
        with self._lock:
            if costmap is None:
                self._costmap = None
                return
            self._costmap = {
                "grid": np.asarray(costmap["grid"]),
                "resolution": float(costmap["resolution"]),
                "origin_x": float(costmap["origin_x"]),
                "origin_y": float(costmap["origin_y"]),
            }

    def set_plan(
        self,
        path_xy: tuple,
        goal: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        with self._lock:
            prev = self._global_plan
            if prev and prev != path_xy:
                self._plan_history.append(prev)
                if len(self._plan_history) > self._history_len:
                    self._plan_history = self._plan_history[-self._history_len :]
            self._global_plan = tuple(path_xy)
            if goal is not None:
                self._goal = (
                    float(goal[0]),
                    float(goal[1]),
                    float(goal[2]),
                )

    def set_pose(self, pose: Optional[conv.Pose2D]) -> None:
        with self._lock:
            if pose is None:
                self._pose = None
            else:
                self._pose = (float(pose.x), float(pose.y), float(pose.theta))

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "costmap": (
                    {
                        "grid": self._costmap["grid"],
                        "resolution": self._costmap["resolution"],
                        "origin_x": self._costmap["origin_x"],
                        "origin_y": self._costmap["origin_y"],
                    }
                    if self._costmap is not None
                    else None
                ),
                "map": self._map,
                "global_plan": self._global_plan,
                "plan_history": list(self._plan_history),
                "local_plan": self._local_plan,
                "footprint": self._footprint,
                "goal": self._goal,
                "pose": self._pose,
            }

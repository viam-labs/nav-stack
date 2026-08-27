"""Bridge-backed WorldIO: map/pose/scan from BridgeNode, drive via IOProvider."""
from __future__ import annotations

import threading
from typing import Optional

from ..ros import conversions as conv
from .world_io import WorldIO


class BridgeWorldIO:
    """Adapt an in-process BridgeNode (+ its asyncio IOProvider) to WorldIO.

    ``node`` may be swapped after SLAM restart; we always read through the
    callable ``get_node`` so callers keep a stable WorldIO instance.
    """

    def __init__(self, get_node, *, drive_timeout_s: float = 2.0):
        self._get_node = get_node
        self._drive_timeout_s = drive_timeout_s
        self._viz_lock = threading.Lock()

    def _node(self):
        node = self._get_node()
        if node is None:
            raise RuntimeError("ROS bridge not started")
        return node

    def get_map(self) -> Optional[dict]:
        node = self._get_node()
        if node is None:
            return None
        return node.get_map()

    def get_pose(self) -> Optional[conv.Pose2D]:
        node = self._get_node()
        if node is None:
            return None
        return node.get_pose_in_map()

    def get_scan(self, max_age_s: float = 2.0) -> Optional[conv.LaserScan2D]:
        node = self._get_node()
        if node is None:
            return None
        return node.get_base_scan(max_age_s)

    def set_velocity(self, vx: float, vy: float, vtheta: float) -> None:
        node = self._node()
        # Prefer the same IOProvider path Nav2 cmd_vel uses so convention /
        # history recording stay consistent.
        node._run(  # noqa: SLF001 - intentional bridge seam
            node._io.drive_base(vx, vy, vtheta, record_source="builtin"),
            timeout=self._drive_timeout_s,
        )

    def stop(self) -> None:
        node = self._get_node()
        if node is None:
            return
        try:
            node._run(node._io.stop_base(), timeout=self._drive_timeout_s)  # noqa: SLF001
        except Exception:  # noqa: BLE001 - stop is best-effort on teardown
            pass

    def set_viz_plan(
        self,
        path_xy: tuple,
        goal: Optional[tuple] = None,
    ) -> None:
        node = self._get_node()
        if node is None:
            return
        with getattr(node, "_viz_lock", self._viz_lock):
            node._viz_global_plan = tuple(path_xy)  # noqa: SLF001
            if goal is not None:
                node._viz_goal = (float(goal[0]), float(goal[1]), float(goal[2]))  # noqa: SLF001


# Explicit Protocol satisfaction for type checkers.
def _check_protocol() -> None:
    _: WorldIO = BridgeWorldIO(lambda: None)  # type: ignore[assignment]

"""Duck-typed RosManager stand-in for ``slam_backend: builtin``."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import SLAM_BACKEND_BUILTIN
from ..ros import conversions as conv
from .engine import BuiltinSlamEngine


class BuiltinSlamHost:
    """SLAM surface used by ``RosSlam`` when there is no slam_toolbox/ROS.

    Implements the methods ``slam.py`` calls on ``runtime.manager`` /
    ``manager.node``. ``node`` is self (same object) so
    ``mgr.node.get_map()`` works.
    """

    def __init__(self, engine: BuiltinSlamEngine):
        self._engine = engine
        self.node = self
        self._map_updates_enabled = True
        self._visible_generation = 0
        self._still_keyframe_hook: Optional[
            Callable[[conv.LaserScan2D, List, conv.Pose2D], None]
        ] = None

    @property
    def engine(self) -> BuiltinSlamEngine:
        return self._engine

    # -- RosManager-like -----------------------------------------------------
    def start(self, io_provider=None, loop=None) -> None:
        del io_provider, loop
        self._engine.start()

    def shutdown(self) -> None:
        self._engine.stop()

    def start_slam(self, map_stem: Optional[Path], mode: str) -> None:
        map_dir = Path(map_stem).parent if map_stem is not None else None
        self._engine.configure_mode(mode, map_dir)

    def stop_slam(self) -> None:
        pass

    def slam_running(self) -> bool:
        return bool(self._engine.diagnostics().get("running"))

    def reset_slam_map(self) -> bool:
        self._engine.reset_map()
        return True

    def save_map(self, map_stem: Path) -> None:
        self._engine.save_map(Path(map_stem).parent)

    def optimize_pose_graph(self, map_stem: Path) -> Dict:
        del map_stem
        return {
            "status": "unsupported",
            "slam_backend": SLAM_BACKEND_BUILTIN,
            "detail": "builtin SLAM has no pose-graph optimizer",
        }

    def set_initial_pose(self, pose: conv.Pose2D) -> None:
        self._engine.set_pose(pose)

    def relocalize(
        self,
        pose: conv.Pose2D,
        *,
        position_variance_m2: float = 4.0,
        yaw_variance_rad2: float = 0.0,
    ) -> None:
        del position_variance_m2, yaw_variance_rad2
        # Ignore tiny nudges the continuous matcher already owns. Periodic
        # relocalize can report a 0.2–0.3 m drift fix — the old 0.35 m gate
        # dropped those while manual global_localize (apply_map_pose_correction)
        # still worked, which looked like "auto refine finds the right spot".
        current = self._engine.get_pose()
        dist = math.hypot(pose.x - current.x, pose.y - current.y)
        dyaw = abs(conv.normalize_angle(pose.theta - current.theta))
        if dist <= 0.15 and dyaw <= math.radians(8.0):
            return
        self._engine.set_pose(pose)

    def get_pose_in_map(self) -> Optional[conv.Pose2D]:
        return self._engine.get_pose()

    def apply_map_pose_correction(self, pose: conv.Pose2D) -> Dict:
        return self._engine.apply_map_pose_correction(pose)

    def set_still_keyframe_hook(self, hook) -> None:
        self._still_keyframe_hook = hook
        self._engine.set_keyframe_hook(self._forward_still_keyframe)

    def _forward_still_keyframe(
        self,
        scan: conv.LaserScan2D,
        band_points: List,
        pose: conv.Pose2D,
    ) -> None:
        hook = self._still_keyframe_hook
        if hook is None:
            return
        hook(scan, band_points, pose)

    def slam_bridge_status(self) -> Dict:
        vx, vy, vtheta = self._engine.get_odom_twist()
        return {"odom_velocity": {"vx": vx, "vy": vy, "vtheta": vtheta}}

    def nav_status(self) -> Dict:
        return {"active": False, "number_of_recoveries": 0, "state": "idle"}

    def slam_diagnostics(self) -> Dict:
        d = self._engine.diagnostics()
        d["slam_toolbox_lifecycle"] = "n/a"
        d["bridge"] = {"ok": True, "slam_backend": SLAM_BACKEND_BUILTIN}
        return d

    def get_base_scan(self, max_age_s: float = 1.0):
        return self._engine._sensors.get_scan(max_age_s)  # noqa: SLF001

    def record_cmd_vel(
        self, vx: float, vy: float, vtheta: float, *, source: str = "nav"
    ) -> None:
        # No ROS bridge cmd_vel history on the builtin path.
        del vx, vy, vtheta, source

    # -- BridgeNode-like (via self.node = self) ------------------------------
    def get_map(self) -> Optional[dict]:
        if not self._map_updates_enabled:
            return None
        return self._engine.get_map()

    def set_map_updates_enabled(self, enabled: bool) -> None:
        self._map_updates_enabled = bool(enabled)

    def flush_map_subscription(self) -> int:
        self._visible_generation = int(
            self._engine.diagnostics().get("generation", 0)
        )
        return self._visible_generation

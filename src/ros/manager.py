"""ROS2 process + node manager.

Owns the lifecycle of:

* the in-process rclpy bridge node (spun in a background thread), and
* the slam_toolbox and Nav2 child processes.

The SLAM model creates/starts the manager and drives mapping/localization; the
navigation model reuses the same manager (shared via the SLAM service) to launch
Nav2 and send goals. Keeping a single manager avoids two rclpy contexts fighting
over DDS.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import MODE_MAPPING, NavConfig, SlamConfig
from . import conversions as conv


def _yaml_dump(data: Dict) -> str:
    """Tiny YAML emitter for the nested ROS params we generate (avoids a PyYAML dep
    at import time; PyYAML is used if present for robustness)."""
    try:
        import yaml  # noqa: WPS433 - optional dependency

        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    except Exception:  # pragma: no cover
        # Minimal fallback for flat-ish structures.
        import json

        return json.dumps(data, indent=2)


class RosManager:
    def __init__(self, slam_cfg: SlamConfig, logger=None):
        self._slam_cfg = slam_cfg
        self._logger = logger
        self._node = None  # BridgeNode
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._slam_proc: Optional[subprocess.Popen] = None
        self._nav2_procs: List[subprocess.Popen] = []
        self._started = False
        self._scratch = Path(slam_cfg.maps_dir).expanduser() / ".runtime"
        self._scratch.mkdir(parents=True, exist_ok=True)

    # -- lifecycle -----------------------------------------------------------
    def start(self, io, loop: asyncio.AbstractEventLoop, nav_cfg: Optional[NavConfig] = None) -> None:
        if self._started:
            return
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        from .bridge import BridgeNode

        if not rclpy.ok():
            rclpy.init()
        self._loop = loop
        self._node = BridgeNode(self._slam_cfg, io, loop, nav_cfg=nav_cfg)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        self._started = True
        self._log("ROS bridge started")

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001
            self._log(f"executor stopped: {exc}")

    def shutdown(self) -> None:
        self.stop_nav2()
        self.stop_slam()
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self._started = False

    @property
    def node(self):
        return self._node

    def _log(self, msg: str) -> None:
        if self._logger:
            self._logger.info(msg)

    def _popen(self, args: List[str]) -> subprocess.Popen:
        self._log("launch: " + " ".join(args))
        return subprocess.Popen(args, env=os.environ.copy())

    # -- slam_toolbox --------------------------------------------------------
    def start_slam(self, map_stem: Path, mode: str) -> None:
        """Launch slam_toolbox in mapping or localization mode.

        ``map_stem`` is the path stem of the serialized pose-graph (``<stem>.posegraph``
        / ``<stem>.data``); used to continue mapping or to localize on a saved map.
        """
        self.stop_slam()
        params = self._slam_params(map_stem, mode)
        params_file = self._scratch / "slam_params.yaml"
        params_file.write_text(_yaml_dump(params))
        node_exe = (
            "async_slam_toolbox_node"
            if mode == MODE_MAPPING
            else "localization_slam_toolbox_node"
        )
        self._slam_proc = self._popen(
            ["ros2", "run", "slam_toolbox", node_exe,
             "--ros-args", "--params-file", str(params_file)]
        )

    def _slam_params(self, map_stem: Path, mode: str) -> Dict:
        node = "mapping" if mode == MODE_MAPPING else "localization"
        stb = self._slam_cfg.slam_toolbox
        params = {
            "slam_toolbox": {
                "ros__parameters": {
                    "mode": node,
                    "odom_frame": self._slam_cfg.frames.odom,
                    "map_frame": self._slam_cfg.frames.map,
                    "base_frame": self._slam_cfg.frames.base_link,
                    **stb.to_ros_dict(),
                    **dict(self._slam_cfg.slam_params),
                }
            }
        }
        rp = params["slam_toolbox"]["ros__parameters"]
        if map_stem and Path(str(map_stem) + ".posegraph").exists():
            rp["map_file_name"] = str(map_stem)
            # Start localizing/continuing from the saved graph's stored pose.
            rp["map_start_at_dock"] = True
        return params

    def save_map(self, map_stem: Path) -> None:
        """Serialize the current slam_toolbox pose-graph to ``<map_stem>``."""
        # slam_toolbox SerializePoseGraph service.
        payload = "{filename: '%s'}" % str(map_stem)
        subprocess.run(
            ["ros2", "service", "call", "/slam_toolbox/serialize_map",
             "slam_toolbox/srv/SerializePoseGraph", payload],
            env=os.environ.copy(), check=False, timeout=30,
        )
        # Also export an occupancy grid (pgm/yaml) for inspection/export.
        subprocess.run(
            ["ros2", "run", "nav2_map_server", "map_saver_cli",
             "-f", str(map_stem), "--ros-args", "-p", "save_map_timeout:=20.0"],
            env=os.environ.copy(), check=False, timeout=40,
        )

    def stop_slam(self) -> None:
        self._terminate(self._slam_proc)
        self._slam_proc = None

    def slam_running(self) -> bool:
        return self._slam_proc is not None and self._slam_proc.poll() is None

    # -- Nav2 ----------------------------------------------------------------
    def start_nav2(self, nav_cfg: NavConfig, params_path: Path) -> None:
        self.stop_nav2()
        # Core Nav2 (planner, controller, costmaps, BT, behaviors). slam_toolbox
        # supplies /map and map->odom, so no map_server/AMCL here.
        self._nav2_procs.append(
            self._popen(
                ["ros2", "launch", "nav2_bringup", "navigation_launch.py",
                 f"params_file:={params_path}", "use_sim_time:=false"]
            )
        )
        # Costmap filter info servers for keepout + speed zones. The mask
        # OccupancyGrids themselves are published by the bridge; these servers
        # publish the matching CostmapFilterInfo. They are lifecycle nodes, managed
        # by a dedicated lifecycle_manager with autostart.
        for which, node_name in (
            ("keepout", "costmap_filter_info_server_keepout"),
            ("speed", "costmap_filter_info_server_speed"),
        ):
            self._nav2_procs.append(
                self._popen(
                    ["ros2", "run", "nav2_map_server", "costmap_filter_info_server",
                     "--ros-args", "-r", f"__node:={node_name}",
                     "--params-file", str(params_path)]
                )
            )
        lm_params = self._scratch / "filter_lifecycle.yaml"
        lm_params.write_text(
            _yaml_dump(
                {
                    "filter_lifecycle_manager": {
                        "ros__parameters": {
                            "autostart": True,
                            "node_names": [
                                "costmap_filter_info_server_keepout",
                                "costmap_filter_info_server_speed",
                            ],
                        }
                    }
                }
            )
        )
        self._nav2_procs.append(
            self._popen(
                ["ros2", "run", "nav2_lifecycle_manager", "lifecycle_manager",
                 "--ros-args", "-r", "__node:=filter_lifecycle_manager",
                 "--params-file", str(lm_params)]
            )
        )

    def stop_nav2(self) -> None:
        for proc in self._nav2_procs:
            self._terminate(proc)
        self._nav2_procs = []

    def nav2_running(self) -> bool:
        return any(p.poll() is None for p in self._nav2_procs)

    # -- navigation delegation ----------------------------------------------
    def navigate(self, x: float, y: float, theta: float) -> None:
        self._require_node().send_nav_goal(x, y, theta)

    def cancel(self) -> None:
        if self._node is not None:
            self._node.cancel_nav()

    def nav_status(self) -> Dict:
        if self._node is None:
            return {"state": "idle", "active": False}
        return self._node.nav_status()

    # -- zone masks ----------------------------------------------------------
    def publish_zone_masks(
        self,
        keepout_mask: np.ndarray,
        speed_mask: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        node = self._require_node()
        node.publish_mask("keepout", keepout_mask, resolution, origin_x, origin_y)
        node.publish_mask("speed", speed_mask, resolution, origin_x, origin_y)

    def set_initial_pose(self, pose: conv.Pose2D) -> None:
        self._require_node().set_initial_odom(pose)

    def set_nav_config(self, nav_cfg: NavConfig) -> None:
        self._require_node().set_nav_config(nav_cfg)

    # -- helpers -------------------------------------------------------------
    def _require_node(self):
        if self._node is None:
            raise RuntimeError("ROS bridge not started")
        return self._node

    @staticmethod
    def _terminate(proc: Optional[subprocess.Popen]) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            time.sleep(0.2)

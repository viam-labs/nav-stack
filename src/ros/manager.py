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
import math
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import MODE_MAPPING, NavConfig, SlamConfig
from . import conversions as conv

SLAM_LIFECYCLE_NODE = "/slam_toolbox"
_REQUIRED_NAV_NODES = ("controller_server", "bt_navigator")


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
        self._nav_cfg: Optional[NavConfig] = None
        self._nav_params_path: Optional[Path] = None
        self._nav_action_ok_until = 0.0
        self._nav2_ensure_lock = threading.Lock()
        self._nav2_ensure_thread: Optional[threading.Thread] = None
        self._started = False
        self._scratch = Path(slam_cfg.maps_dir).expanduser() / ".runtime"
        self._scratch.mkdir(parents=True, exist_ok=True)

    # -- lifecycle -----------------------------------------------------------
    def start(self, io, loop: asyncio.AbstractEventLoop, nav_cfg: Optional[NavConfig] = None) -> None:
        if self._started:
            return
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        from .bridge import BridgeNode

        if not rclpy.ok():
            rclpy.init()
        self._loop = loop
        self._node = BridgeNode(self._slam_cfg, io, loop, nav_cfg=nav_cfg)
        # Run callbacks concurrently so slow scan reads do not starve odom/tf updates.
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        self._started = True
        self._log("ROS bridge started")

    def _spin(self) -> None:
        # A single misbehaving callback must not silently kill every timer and
        # subscription (frozen TF/odom/scans -> Nav2 "Robot pose is not
        # available"); log and resume spinning.
        while self._started or self._executor is not None:
            try:
                self._executor.spin()
                return
            except Exception as exc:  # noqa: BLE001
                self._log(f"executor crashed (resuming): {exc!r}")
                time.sleep(0.2)

    def shutdown(self) -> None:
        self.stop_nav2()
        self.stop_slam()
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        self._node = None
        self._executor = None
        self._spin_thread = None
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

    def _ros_env(self) -> dict:
        env = os.environ.copy()
        env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        env.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        env.setdefault("RCUTILS_LOGGING_USE_STDOUT", "1")
        distro = env.get("ROS_DISTRO", "jazzy")
        ros_bin = f"/opt/ros/{distro}/bin"
        path = env.get("PATH", "")
        if ros_bin not in path.split(os.pathsep):
            env["PATH"] = ros_bin + os.pathsep + path
        return env

    def _ros2_cmd(self) -> str:
        distro = os.environ.get("ROS_DISTRO", "jazzy")
        return shutil.which("ros2") or f"/opt/ros/{distro}/bin/ros2"

    def _ros_setup(self) -> str:
        return os.environ.get("ROS_ENV", f"/opt/ros/{os.environ.get('ROS_DISTRO', 'jazzy')}/setup.bash")

    def _popen(self, args: List[str]) -> subprocess.Popen:
        ros2 = self._ros2_cmd()
        if args and args[0] == "ros2":
            args = [ros2, *args[1:]]
        self._log("launch: " + " ".join(args))
        log_path = self._scratch / "nav2_launch.log"
        log_fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - long-lived child log
        log_fh.write(f"\n--- launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_fh.write(" ".join(args) + "\n")
        log_fh.flush()
        return subprocess.Popen(
            args,
            env=self._ros_env(),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _run_ros(
        self, args: List[str], *, timeout: float = 5.0
    ) -> subprocess.CompletedProcess:
        ros2 = self._ros2_cmd()
        if args and args[0] == "ros2":
            cmd = [ros2, *args[1:]]
        else:
            cmd = list(args)
        setup = self._ros_setup()
        shell_cmd = f"source {shlex.quote(setup)} && " + " ".join(shlex.quote(a) for a in cmd)
        try:
            return subprocess.run(
                ["bash", "-lc", shell_cmd],
                env=self._ros_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            self._log(f"ros2 command timed out after {timeout:.0f}s: {' '.join(cmd)}")
            return subprocess.CompletedProcess(
                cmd,
                returncode=-1,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "timeout") if isinstance(exc.stderr, str) else "timeout",
            )

    def _nav_action_visible(self) -> bool:
        proc = self._run_ros(["ros2", "action", "list"])
        return proc.returncode == 0 and "navigate_to_pose" in (proc.stdout or "")

    def _nav_action_server_visible(self) -> bool:
        proc = self._run_ros(["ros2", "action", "info", "/navigate_to_pose"])
        if proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                if "Action servers:" in line:
                    try:
                        count = int(line.split(":", 1)[1].strip())
                        return count > 0
                    except ValueError:
                        break
            # Older ROS tools may not print server counts; treat successful info
            # output as server visibility.
            return True
        # Fallback for environments where `action info` is unavailable.
        return self._nav_action_visible()

    def _missing_required_nav_nodes(self) -> List[str]:
        proc = self._run_ros(["ros2", "node", "list"])
        if proc.returncode != 0:
            return list(_REQUIRED_NAV_NODES)
        nodes = proc.stdout or ""
        return [name for name in _REQUIRED_NAV_NODES if name not in nodes]

    def _required_nav_nodes_present(self) -> bool:
        return len(self._missing_required_nav_nodes()) == 0

    def _required_nav_nodes_active(self) -> bool:
        """True when core Nav2 lifecycle nodes are in the ``active`` state.

        An inactive bt_navigator still advertises /navigate_to_pose but rejects
        every goal, so action visibility alone is not sufficient readiness.
        """
        for name in _REQUIRED_NAV_NODES:
            if self._lifecycle_get_state("/" + name) != "active":
                return False
        return True

    def nav_action_ready(self) -> bool:
        now = time.monotonic()
        if now < self._nav_action_ok_until and self.nav2_running():
            if self._required_nav_nodes_present():
                return True
            self._nav_action_ok_until = 0.0
        if not self.nav2_running():
            self._nav_action_ok_until = 0.0
        ok = (
            self._nav_action_server_visible()
            and self._required_nav_nodes_present()
            and self._required_nav_nodes_active()
        )
        if ok:
            self._nav_action_ok_until = now + 30.0
        return ok

    def wait_for_nav_action(self, timeout: float = 90.0) -> bool:
        return self._wait_for_nav_action(timeout=timeout)

    def _wait_for_nav_action(self, timeout: float = 90.0) -> bool:
        """Wait until Nav2 exposes ``/navigate_to_pose``."""
        deadline = time.monotonic() + timeout
        last_detail = ""
        while time.monotonic() < deadline:
            if self._nav2_procs and not any(p.poll() is None for p in self._nav2_procs):
                self._log("Nav2 launch process exited before action server appeared")
                return False
            action_ok = self._nav_action_server_visible()
            missing = self._missing_required_nav_nodes()
            nodes_ok = len(missing) == 0
            active_ok = nodes_ok and self._required_nav_nodes_active()
            if action_ok and nodes_ok and active_ok:
                return True
            last_detail = (
                f"ros2={self._ros2_cmd()} action_ok={action_ok} "
                f"nodes_ok={nodes_ok} active_ok={active_ok} missing_nodes={missing}"
            )
            time.sleep(0.5)
        self._log(f"timed out waiting for Nav2 action server ({timeout:.0f}s): {last_detail}")
        return False

    def _wait_for_ros_node(self, node_name: str, timeout: float = 30.0) -> bool:
        """Wait until ``node_name`` appears in ``ros2 node list``."""
        bare = node_name.lstrip("/")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._slam_proc is not None and self._slam_proc.poll() is not None:
                self._log(f"{node_name} process exited before the node registered")
                return False
            proc = self._run_ros(["ros2", "node", "list"])
            if proc.returncode == 0 and bare in (proc.stdout or ""):
                return True
            time.sleep(0.25)
        self._log(f"timed out waiting for ROS node {node_name}")
        return False

    def _wait_for_ros_node_gone(self, node_name: str, timeout: float = 10.0) -> bool:
        """Wait until ``node_name`` disappears from ``ros2 node list``."""
        bare = node_name.lstrip("/")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc = self._run_ros(["ros2", "node", "list"])
            if proc.returncode == 0 and bare not in (proc.stdout or ""):
                return True
            time.sleep(0.25)
        self._log(f"timed out waiting for ROS node {node_name} to disappear")
        return False

    def _lifecycle_get_state(self, node_name: str) -> Optional[str]:
        proc = self._run_ros(["ros2", "lifecycle", "get", node_name])
        if proc.returncode != 0:
            return None
        text = (proc.stdout or "").strip().lower()
        # Match the primary state word only. Substring checks mis-read transitional
        # states (e.g. "active" inside "deactivating") and break restart cycles.
        for state in ("unconfigured", "inactive", "active", "finalized"):
            if text.startswith(state) or f"{state} [" in text:
                return state
        return None

    def _lifecycle_set(self, node_name: str, transition: str) -> bool:
        proc = self._run_ros(["ros2", "lifecycle", "set", node_name, transition])
        if proc.returncode == 0:
            return True
        detail = (proc.stderr or proc.stdout or "").strip()
        self._log(f"slam lifecycle {transition} failed: {detail}")
        return False

    def _activate_slam_lifecycle(self, timeout: float = 15.0) -> None:
        """Configure + activate slam_toolbox when it is a lifecycle node (Jazzy+).

        Older distros expose a plain node that starts active; in that case the
        lifecycle CLI fails and we leave the node as-is.
        """
        node = SLAM_LIFECYCLE_NODE
        if not self._wait_for_ros_node(node, timeout=timeout):
            raise RuntimeError(f"{node} did not register with ROS")

        deadline = time.monotonic() + timeout
        last_detail = ""
        while time.monotonic() < deadline:
            if self._slam_proc is not None and self._slam_proc.poll() is not None:
                raise RuntimeError(f"{node} process exited during lifecycle activation")
            state = self._lifecycle_get_state(node)
            if state is None:
                self._log(
                    f"{node} is not lifecycle-managed; skipping configure/activate"
                )
                return
            if state == "finalized":
                time.sleep(0.25)
                continue
            if state == "unconfigured":
                if not self._lifecycle_set(node, "configure"):
                    last_detail = "configure failed"
                    time.sleep(0.5)
                    continue
                time.sleep(0.5)
                continue
            if state == "inactive":
                if self._lifecycle_set(node, "activate"):
                    time.sleep(0.25)
                    continue
                last_detail = "activate failed"
                time.sleep(0.5)
                continue
            if state == "active":
                self._log("slam_toolbox lifecycle activated")
                return
            time.sleep(0.25)
        raise RuntimeError(
            f"slam_toolbox did not reach active state within {timeout}s ({last_detail})"
        )

    # -- slam_toolbox --------------------------------------------------------
    def start_slam(self, map_stem: Path, mode: str) -> None:
        """Launch slam_toolbox in mapping or localization mode.

        ``map_stem`` is the path stem of the serialized pose-graph (``<stem>.posegraph``
        / ``<stem>.data``); used to continue mapping or to localize on a saved map.
        """
        self.stop_slam()
        last_error: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                self._launch_slam(map_stem, mode)
                self._activate_slam_lifecycle()
                return
            except RuntimeError as exc:
                last_error = exc
                self._log(f"slam_toolbox start attempt {attempt} failed: {exc}")
                self._terminate(self._slam_proc)
                self._slam_proc = None
                self._wait_for_ros_node_gone(SLAM_LIFECYCLE_NODE, timeout=8.0)
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"failed to start slam_toolbox after 2 attempts: {last_error}")

    def _launch_slam(self, map_stem: Path, mode: str) -> None:
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
             "--ros-args", "-r", "__node:=slam_toolbox",
             "--params-file", str(params_file)]
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
        # slam_toolbox stamps map->odom as last-scan-time + transform_timeout.
        # Slow MiR lidar reads can gap scans by ~10s; future-stamp the transform
        # so Nav2's controller does not reject it as stale between scans.
        rp.setdefault("transform_timeout", 5.0)
        rp.setdefault("tf_buffer_duration", 60.0)
        rp.setdefault("transform_publish_period", 0.02)
        if map_stem and Path(str(map_stem) + ".posegraph").exists():
            rp["map_file_name"] = str(map_stem)
            # Keep localization startup robust for saved posegraphs. Allow user
            # overrides in slam_params to disable this if needed.
            rp.setdefault("map_start_at_dock", True)
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
        if self.slam_running():
            state = self._lifecycle_get_state(SLAM_LIFECYCLE_NODE)
            if state == "active":
                self._lifecycle_set(SLAM_LIFECYCLE_NODE, "deactivate")
                time.sleep(0.3)
        self._terminate(self._slam_proc)
        self._slam_proc = None
        self._wait_for_ros_node_gone(SLAM_LIFECYCLE_NODE, timeout=8.0)
        time.sleep(0.25)

    def reset_slam_map(self) -> bool:
        """Clear slam_toolbox's in-memory map (publishes a fresh empty /map)."""
        proc = subprocess.run(
            [
                "ros2",
                "service",
                "call",
                "/slam_toolbox/reset",
                "slam_toolbox/srv/Reset",
                "{pause_new_measurements: false}",
            ],
            env=os.environ.copy(),
            check=False,
            timeout=10,
        )
        return proc.returncode == 0

    def slam_running(self) -> bool:
        return self._slam_proc is not None and self._slam_proc.poll() is None

    # -- Nav2 ----------------------------------------------------------------
    def start_nav2(self, nav_cfg: NavConfig, params_path: Path) -> None:
        self._nav_cfg = nav_cfg
        self._nav_params_path = params_path
        self.stop_nav2()
        # Core Nav2 (planner, controller, costmaps, BT, behaviors). slam_toolbox
        # supplies /map and map->odom, so no map_server/AMCL here.
        self._nav2_procs.append(
            self._popen(
                ["ros2", "launch", "nav2_bringup", "navigation_launch.py",
                 f"params_file:={params_path}", "use_sim_time:=false",
                 "autostart:=false", "use_collision_monitor:=False",
                 "use_composition:=False"]
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
                            # Bond heartbeats miss on a loaded Pi and the manager
                            # then deactivates every node; disable them.
                            "bond_timeout": 0.0,
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
        nav_lm_params = self._scratch / "nav_lifecycle.yaml"
        nav_lm_params.write_text(
            _yaml_dump(
                {
                    "navigation_lifecycle_manager_override": {
                        "ros__parameters": {
                            "autostart": True,
                            # Bond heartbeats miss on a loaded Pi and the manager
                            # then deactivates every Nav2 node mid-run ("CRITICAL
                            # FAILURE: SERVER ... IS DOWN"); disable them.
                            "bond_timeout": 0.0,
                            "node_names": [
                                "controller_server",
                                "smoother_server",
                                "planner_server",
                                "behavior_server",
                                "bt_navigator",
                                "waypoint_follower",
                                "velocity_smoother",
                                "route_server",
                                "docking_server",
                            ],
                        }
                    }
                }
            )
        )
        self._nav2_procs.append(
            self._popen(
                [
                    "ros2",
                    "run",
                    "nav2_lifecycle_manager",
                    "lifecycle_manager",
                    "--ros-args",
                    "-r",
                    "__node:=navigation_lifecycle_manager_override",
                    "--params-file",
                    str(nav_lm_params),
                ]
            )
        )
        if self._wait_for_nav_action():
            if self._node is not None:
                self._node.reset_nav_action_client()
        else:
            self._log("Nav2 started but /navigate_to_pose is not ready yet")

    def stop_nav2(self) -> None:
        self._nav_action_ok_until = 0.0
        if self._node is not None:
            try:
                self._node.cancel_nav()
            except Exception:
                pass
            try:
                self._node.reset_nav_action_client()
            except Exception:
                pass
        for proc in self._nav2_procs:
            self._terminate(proc)
        self._nav2_procs = []
        # Orphaned ros2 launch / Nav2 nodes can survive parent termination after
        # crashes or abrupt service restarts; clean up common leftovers. Patterns
        # use the slash form because that is how the executables appear in
        # process command lines (e.g. /opt/ros/jazzy/lib/nav2_controller/controller_server).
        for pattern in (
            "nav2_bringup/navigation_launch.py",
            "nav2_lifecycle_manager/lifecycle_manager",
            "nav2_map_server/costmap_filter_info_server",
            "nav2_collision_monitor/collision_monitor",
            "nav2_controller/controller_server",
            "nav2_bt_navigator/bt_navigator",
            "nav2_planner/planner_server",
            "nav2_behaviors/behavior_server",
            "nav2_smoother/smoother_server",
            "nav2_velocity_smoother/velocity_smoother",
            "nav2_waypoint_follower/waypoint_follower",
            "nav2_route/route_server",
            "opennav_docking/opennav_docking",
        ):
            subprocess.run(
                ["pkill", "-f", pattern],
                env=self._ros_env(),
                check=False,
            )
        time.sleep(0.5)

    def nav2_running(self) -> bool:
        return any(p.poll() is None for p in self._nav2_procs)

    def _nav2_log_errors(self, max_lines: int = 40) -> str:
        """Extract crash/error lines from the Nav2 launch log.

        The raw tail is usually dominated by lifecycle-manager wait spam, hiding
        the actual reason server nodes died.
        """
        log_path = self._scratch / "nav2_launch.log"
        if not log_path.exists():
            return ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        markers = (
            "[ERROR]",
            "[FATAL]",
            "process has died",
            "Traceback",
            "Caught exception",
            "what():",
            "exited with code",
            "Failed to parse",
            "InvalidParameter",
        )
        interesting = [ln for ln in lines if any(m in ln for m in markers)]
        return "\n".join(interesting[-max_lines:])

    def nav2_diagnostics(self) -> Dict:
        action_proc = self._run_ros(["ros2", "action", "list"])
        action_info_proc = self._run_ros(["ros2", "action", "info", "/navigate_to_pose"])
        node_proc = self._run_ros(["ros2", "node", "list"])
        lifecycle_proc = self._run_ros(["ros2", "lifecycle", "get", "/bt_navigator"])
        controller_proc = self._run_ros(["ros2", "lifecycle", "get", "/controller_server"])
        log_path = self._scratch / "nav2_launch.log"
        log_tail = ""
        if log_path.exists():
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            except OSError:
                log_tail = ""
        odom_tf_age = None
        if self._node is not None:
            try:
                odom_tf_age = self._node.odom_tf_age_s()
            except Exception:  # noqa: BLE001 - diagnostics must not raise
                pass
        return {
            "nav2_processes_running": self.nav2_running(),
            "nav2_startup_in_progress": self.nav2_startup_in_progress(),
            # Age of the bridge's last odom/TF publish; more than a few seconds
            # means the bridge executor is stalled or dead.
            "odom_tf_age_s": odom_tf_age,
            "nav_action_ready": self.nav_action_ready(),
            "actions": (action_proc.stdout or "").strip(),
            "actions_rc": action_proc.returncode,
            "actions_stderr": (action_proc.stderr or "").strip(),
            "action_info": (action_info_proc.stdout or "").strip(),
            "action_info_rc": action_info_proc.returncode,
            "action_info_stderr": (action_info_proc.stderr or "").strip(),
            "nodes": (node_proc.stdout or "").strip(),
            "core_nodes_present": self._required_nav_nodes_present(),
            "missing_core_nodes": self._missing_required_nav_nodes(),
            "controller_server_lifecycle": (
                controller_proc.stdout or controller_proc.stderr or ""
            ).strip(),
            "bt_navigator_lifecycle": (lifecycle_proc.stdout or lifecycle_proc.stderr or "").strip(),
            "nav2_log_errors": self._nav2_log_errors(),
            "nav2_log_tail": log_tail,
            "nav2_log_path": str(log_path),
        }

    def ensure_nav2(self, nav_cfg: NavConfig, params_path: Path) -> None:
        """Start or restart Nav2 until ``/navigate_to_pose`` is available."""
        if self.nav_action_ready():
            return

        last_detail: Optional[Dict] = None
        for attempt in range(1, 4):
            if self.nav2_running():
                self._log(
                    f"Nav2 unhealthy (attempt {attempt}/3); waiting briefly before restart"
                )
                if self.wait_for_nav_action(timeout=15.0):
                    return
                self._log("Nav2 still unhealthy; restarting Nav2")
            else:
                self._log(f"Nav2 not running (attempt {attempt}/3); launching Nav2")

            self.start_nav2(nav_cfg, params_path)
            if self.wait_for_nav_action(timeout=60.0):
                return
            last_detail = self.nav2_diagnostics()
            self._log(f"Nav2 startup attempt {attempt}/3 failed: {last_detail}")
            self.stop_nav2()
            time.sleep(1.0 * attempt)

        detail = last_detail or self.nav2_diagnostics()
        raise RuntimeError(
            "Nav2 failed to reach healthy state after 3 attempts "
            f"(missing: {detail.get('missing_core_nodes')}). "
            f"See {detail.get('nav2_log_path')} and viam-server logs."
        )

    def nav2_startup_in_progress(self) -> bool:
        thread = self._nav2_ensure_thread
        return thread is not None and thread.is_alive()

    def ensure_nav2_async(self, nav_cfg: NavConfig, params_path: Path) -> None:
        """Run ``ensure_nav2`` in a background thread.

        Reconfigure must return within viam-server's deadline; Nav2 bringup with
        retries can take minutes on a Pi, so it cannot run inline.
        """
        self._nav_cfg = nav_cfg
        self._nav_params_path = params_path
        with self._nav2_ensure_lock:
            if self.nav2_startup_in_progress():
                self._log("Nav2 startup already in progress; skipping duplicate request")
                return

            def _run() -> None:
                try:
                    self.ensure_nav2(nav_cfg, params_path)
                    self._log("background Nav2 startup finished")
                except Exception as exc:  # noqa: BLE001 - surfaced via logs + get_status
                    self._log(f"background Nav2 startup failed: {exc}")

            self._nav2_ensure_thread = threading.Thread(
                target=_run, name="nav2-ensure", daemon=True
            )
            self._nav2_ensure_thread.start()

    # -- navigation delegation ----------------------------------------------
    def navigate(self, x: float, y: float, theta: float) -> None:
        node = self._require_node()
        if self.nav2_startup_in_progress():
            raise RuntimeError(
                "Nav2 is still starting up in the background; retry in a few seconds "
                "(check progress with the get_status command)"
            )
        if self._nav_cfg is not None and self._nav_params_path is not None and not self.nav_action_ready():
            self._log("Nav2 not healthy during navigate; ensuring Nav2 before sending goal")
            self.ensure_nav2(self._nav_cfg, self._nav_params_path)
        try:
            node.send_nav_goal(x, y, theta)
            return
        except RuntimeError as exc:
            if "Nav2 action server not available" not in str(exc):
                raise
        if self._nav_cfg is None or self._nav_params_path is None:
            raise RuntimeError(
                "Nav2 action server not available and Nav2 has no saved startup config"
            )
        self._log("Nav2 action unavailable during navigate; ensuring Nav2 and retrying")
        self.ensure_nav2(self._nav_cfg, self._nav_params_path)
        node.send_nav_goal(x, y, theta)

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
        self._clear_localization_buffer()
        self._require_node().set_initial_pose(pose)

    def relocalize(
        self,
        pose: conv.Pose2D,
        *,
        position_variance_m2: float = 4.0,
        yaw_variance_rad2: float = (math.pi / 4) ** 2,
    ) -> None:
        """Trigger scan-to-map matching from an approximate map pose."""
        self._clear_localization_buffer()
        self._require_node().set_initial_pose(
            pose,
            position_variance_m2=position_variance_m2,
            yaw_variance_rad2=yaw_variance_rad2,
        )

    def get_pose_in_map(self) -> Optional[conv.Pose2D]:
        return self._require_node().get_pose_in_map()

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
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            time.sleep(0.2)

    def _clear_localization_buffer(self) -> None:
        """Best-effort clear of stale localization history before reseeding."""
        self._run_ros(
            [
                "ros2",
                "service",
                "call",
                "/slam_toolbox/clear_localization_buffer",
                "std_srvs/srv/Empty",
                "{}",
            ],
            timeout=2.0,
        )


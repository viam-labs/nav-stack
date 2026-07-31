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
import hashlib
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
from .dds_env import apply_dds_isolation, dds_status

SLAM_LIFECYCLE_NODE = "/slam_toolbox"
_REQUIRED_NAV_NODES = ("controller_server", "bt_navigator")
# Bundled launch that omits collision_monitor / docking / route (see launch/).
_NAV2_LAUNCH = (
    Path(__file__).resolve().parent.parent.parent / "launch" / "navigation_launch.py"
)


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
    def __init__(self, slam_cfg: SlamConfig, logger=None, external_slam=None):
        self._slam_cfg = slam_cfg
        self._logger = logger
        # When set (external-SLAM navigation), the bridge publishes /map +
        # map->odom from this Viam SLAM service instead of running slam_toolbox.
        self._external_slam = external_slam
        self._node = None  # BridgeNode
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._slam_proc: Optional[subprocess.Popen] = None
        self._nav2_procs: List[subprocess.Popen] = []
        self._nav_cfg: Optional[NavConfig] = None
        self._nav_params_path: Optional[Path] = None
        self._nav_action_ok_until = 0.0
        self._nav2_params_sig = ""
        self._nav2_ensure_lock = threading.Lock()
        self._nav2_ensure_thread: Optional[threading.Thread] = None
        self._started = False
        self._scratch = Path(slam_cfg.maps_dir).expanduser() / ".runtime"
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._last_slam_params: Dict = {}
        # Ensure DDS isolation before any child processes inherit the env.
        # Prefer the module-root persist file; scratch is a fallback location.
        apply_dds_isolation(self._scratch / "ros_domain_id")

    # -- lifecycle -----------------------------------------------------------
    def start(self, io, loop: asyncio.AbstractEventLoop, nav_cfg: Optional[NavConfig] = None) -> None:
        if self._started:
            return
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        from .bridge import BridgeNode

        # rclpy.init reads discovery/domain from the environment once.
        apply_dds_isolation(self._scratch / "ros_domain_id")
        if not rclpy.ok():
            rclpy.init()
        self._loop = loop
        self._node = BridgeNode(
            self._slam_cfg, io, loop, nav_cfg=nav_cfg, external_slam=self._external_slam
        )
        # The bridge has 4 callback sources that block on Viam IO (scan, odom,
        # drive, watchdog/misc — each capped at 1 in-flight by its mutually
        # exclusive callback group). With only 4 threads they can all be blocked
        # at once, starving the TF listener subscription: the local TF buffer
        # then goes stale and map->base_link lookups fail even while
        # slam_toolbox publishes normally. Keep threads > max blocked callbacks.
        self._executor = MultiThreadedExecutor(num_threads=8)
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
        """Tear down the in-process ROS stack.

        Every step is individually guarded: reconfigure calls this before
        building a fresh manager, and a partially-failed teardown leaves the
        old bridge's DDS participant alive — the ROS graph then shows two
        /viam_nav_stack_bridge nodes fighting over the odom TF.
        """
        self._started = False
        self.stop_nav2()
        self.stop_slam()
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except Exception as exc:  # noqa: BLE001
                self._log(f"executor shutdown failed: {exc!r}")
        if self._spin_thread is not None and self._spin_thread.is_alive():
            # Callbacks can block on Viam sensor reads for up to
            # sensor_read_timeout_s; give the spin thread a real chance to
            # drain before destroying the node under it.
            self._spin_thread.join(timeout=8.0)
            if self._spin_thread.is_alive():
                self._log("ROS spin thread still alive; forcing context shutdown")
        if self._node is not None:
            if self._executor is not None:
                try:
                    self._executor.remove_node(self._node)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._node.destroy_node()
            except Exception as exc:  # noqa: BLE001
                self._log(f"bridge node destroy failed: {exc!r}")
        self._node = None
        self._executor = None
        self._spin_thread = None
        # Shutting down the rclpy context tears down the DDS participant even
        # if destroy_node failed above — this is the backstop against zombie
        # bridge nodes surviving a reconfigure.
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    @property
    def node(self):
        return self._node

    def _log(self, msg: str) -> None:
        if self._logger:
            self._logger.info(msg)

    def _ros_env(self) -> dict:
        apply_dds_isolation(self._scratch / "ros_domain_id")
        env = os.environ.copy()
        env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        env.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        env.setdefault("ROS_LOCALHOST_ONLY", "1")
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
        if not self.nav2_running():
            self._nav_action_ok_until = 0.0
            return False
        # Recently verified fully healthy: run only the cheapest check (one
        # `ros2 node list`) instead of the ~7 CLI subprocesses (seconds each on
        # a Pi). A vanished core node is a real crash and resets the grace
        # window; action/lifecycle CLI flake alone must not block goals.
        if now < self._nav_action_ok_until:
            if self._required_nav_nodes_present():
                return True
            self._nav_action_ok_until = 0.0
        action_ok = self._nav_action_server_visible()
        nodes_ok = self._required_nav_nodes_present()
        if not (action_ok and nodes_ok):
            self._nav_action_ok_until = 0.0
            return False
        if self._required_nav_nodes_active():
            self._nav_action_ok_until = now + 60.0
            return True
        # Lifecycle CLI is slow on Pi; keep accepting goals briefly if we were
        # recently healthy so a transient lifecycle query does not block nav.
        if now < self._nav_action_ok_until:
            return True
        return False

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

    def _lifecycle_set(
        self, node_name: str, transition: str, *, timeout: float = 5.0
    ) -> bool:
        proc = self._run_ros(
            ["ros2", "lifecycle", "set", node_name, transition], timeout=timeout
        )
        if proc.returncode == 0:
            return True
        detail = (proc.stderr or proc.stdout or "").strip() or f"rc={proc.returncode}"
        self._last_lifecycle_error = f"{transition}: {detail}"
        self._log(f"slam lifecycle {transition} failed: {detail}")
        return False

    def _activate_slam_lifecycle(self, timeout: float = 90.0) -> None:
        """Configure + activate slam_toolbox when it is a lifecycle node (Jazzy+).

        Older distros expose a plain node that starts active; in that case the
        lifecycle CLI fails and we leave the node as-is.

        Localization activate loads the serialized pose-graph; on a Pi that often
        exceeds the old 5s ``ros2 lifecycle set`` timeout, so we give activate a
        long CLI budget and keep polling for ``active`` even if the CLI times out
        while the transition is still running.
        """
        node = SLAM_LIFECYCLE_NODE
        self._last_lifecycle_error = ""
        # Registration wait is separate from the configure/activate budget.
        if not self._wait_for_ros_node(node, timeout=min(timeout, 30.0)):
            raise RuntimeError(f"{node} did not register with ROS")

        deadline = time.monotonic() + timeout
        last_detail = ""
        activate_attempts = 0
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
                if not self._lifecycle_set(node, "configure", timeout=30.0):
                    last_detail = self._last_lifecycle_error or "configure failed"
                    time.sleep(0.5)
                    continue
                time.sleep(0.5)
                continue
            if state == "inactive":
                # Only issue activate a few times; repeated calls while a slow
                # posegraph load is in flight can wedge the lifecycle FSM.
                if activate_attempts < 3:
                    remaining = max(deadline - time.monotonic(), 5.0)
                    ok = self._lifecycle_set(
                        node, "activate", timeout=min(60.0, remaining)
                    )
                    activate_attempts += 1
                    if not ok:
                        last_detail = self._last_lifecycle_error or "activate failed"
                # Poll: map load may still finish after a CLI timeout.
                time.sleep(0.5)
                continue
            if state == "active":
                self._log("slam_toolbox lifecycle activated")
                return
            time.sleep(0.25)
        raise RuntimeError(
            f"slam_toolbox did not reach active state within {timeout}s "
            f"({last_detail or 'unknown'}); check posegraph load / slam_toolbox logs"
        )

    # -- slam_toolbox --------------------------------------------------------
    def stop_slam(self) -> None:
        if self.slam_running():
            state = self._lifecycle_get_state(SLAM_LIFECYCLE_NODE)
            if state == "active":
                self._lifecycle_set(SLAM_LIFECYCLE_NODE, "deactivate")
                time.sleep(0.3)
        self._terminate(self._slam_proc)
        self._slam_proc = None
        # slam_toolbox is launched with start_new_session=True, so it survives
        # if this module process dies without a clean shutdown (crash or hard
        # kill mid-reconfigure). An orphaned instance keeps publishing a
        # competing map->odom TF — the pose then flickers between two SLAM
        # solutions and walls imprint at multiple angles. Reap any leftovers
        # before considering the stop complete.
        self._reap_slam_toolbox_processes(force=False)
        if not self._wait_for_slam_toolbox_processes_gone(timeout=4.0):
            self._log("slam_toolbox process still alive after SIGTERM — sending SIGKILL")
            self._reap_slam_toolbox_processes(force=True)
            self._wait_for_slam_toolbox_processes_gone(timeout=8.0)
        # DDS can keep zombie /slam_toolbox names on the graph for a while —
        # do not block forever waiting for the name to disappear.
        self._wait_for_ros_node_gone(SLAM_LIFECYCLE_NODE, timeout=2.0)
        time.sleep(0.25)

    def _reap_slam_toolbox_processes(self, *, force: bool = False) -> None:
        sig = ["-9"] if force else []
        for pattern in (
            "slam_toolbox/async_slam_toolbox_node",
            "slam_toolbox/localization_slam_toolbox_node",
            "async_slam_toolbox_node",
            "localization_slam_toolbox_node",
        ):
            subprocess.run(
                ["pkill", *sig, "-f", pattern],
                env=self._ros_env(),
                check=False,
            )

    def _slam_toolbox_binary_count(self) -> int:
        """Count live slam_toolbox *binaries* (ignore DDS phantoms / ros2 CLI)."""
        proc = subprocess.run(
            [
                "pgrep",
                "-f",
                r"/lib/slam_toolbox/(async|localization)_slam_toolbox_node",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode not in (0, 1):
            return 0
        return len([ln for ln in (proc.stdout or "").splitlines() if ln.strip()])

    def _wait_for_slam_toolbox_processes_gone(self, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._slam_toolbox_binary_count() == 0:
                return True
            time.sleep(0.25)
        self._log(
            f"timed out waiting for slam_toolbox processes to exit "
            f"(still {self._slam_toolbox_binary_count()})"
        )
        return False

    def start_slam(self, map_stem: Path, mode: str) -> None:
        """Launch slam_toolbox in mapping or localization mode.

        ``map_stem`` is the path stem of the serialized pose-graph (``<stem>.posegraph``
        / ``<stem>.data``); used to continue mapping or to localize on a saved map.
        """
        stem = Path(map_stem)
        posegraph = Path(str(stem) + ".posegraph")
        data_file = Path(str(stem) + ".data")
        if mode != MODE_MAPPING:
            missing = [str(p) for p in (posegraph, data_file) if not p.exists()]
            if missing:
                raise RuntimeError(
                    "cannot localize: missing slam_toolbox serialize file(s): "
                    + ", ".join(missing)
                )
        self.stop_slam()
        # Extra pass: a previous module instance can leave a session orphan that
        # stop_slam's pkill raced; never launch while a sibling process exists.
        self._reap_slam_toolbox_processes(force=True)
        self._wait_for_slam_toolbox_processes_gone(timeout=8.0)
        last_error: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                if self._slam_toolbox_binary_count() > 0:
                    self._reap_slam_toolbox_processes(force=True)
                    self._wait_for_slam_toolbox_processes_gone(timeout=4.0)
                self._launch_slam(stem, mode)
                # Localization activate loads the pose-graph; give a Pi enough time.
                activate_timeout = 90.0 if mode != MODE_MAPPING else 45.0
                self._activate_slam_lifecycle(timeout=activate_timeout)
                self._apply_slam_tf_params()
                # Trust processes, not the ROS graph: FastDDS often keeps stale
                # /slam_toolbox names after pkill (looks like "3 nodes" with ps empty).
                n = self._slam_toolbox_binary_count()
                if n > 1:
                    raise RuntimeError(
                        f"{n} slam_toolbox binaries still running after start — "
                        "orphaned instance fighting over map->odom. "
                        "Kill leftovers: pkill -9 -f async_slam_toolbox_node"
                    )
                if n == 0:
                    raise RuntimeError(
                        "slam_toolbox activated but no async/localization binary "
                        "is running (process died immediately)"
                    )
                return
            except RuntimeError as exc:
                last_error = exc
                self._log(f"slam_toolbox start attempt {attempt} failed: {exc}")
                self._terminate(self._slam_proc)
                self._slam_proc = None
                self._reap_slam_toolbox_processes(force=True)
                self._wait_for_slam_toolbox_processes_gone(timeout=8.0)
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"failed to start slam_toolbox after 2 attempts: {last_error}")

    def _count_ros_nodes_named(self, node_name: str) -> int:
        """How many graph entries match ``node_name`` (may include DDS phantoms)."""
        names: List[str] = []
        if self._node is not None:
            try:
                names = [
                    ("/" + name) if ns in ("", "/") else (ns.rstrip("/") + "/" + name)
                    for name, ns in self._node.get_node_names_and_namespaces()
                ]
            except Exception:  # noqa: BLE001
                names = []
        if not names:
            proc = self._run_ros(["ros2", "node", "list"])
            names = [n.strip() for n in (proc.stdout or "").splitlines() if n.strip()]
        target = node_name if node_name.startswith("/") else f"/{node_name}"
        return sum(1 for n in names if n == target or n.endswith(target))

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
        # slam_toolbox stamps map->odom at scan_time + transform_timeout. Any
        # positive value makes map->odom newer than odom->base_link (same scan
        # stamp) and Nav2 TF lookups fail with extrapolation errors.
        rp.setdefault("transform_timeout", 0.0)
        rp.setdefault("tf_buffer_duration", 60.0)
        rp.setdefault("transform_publish_period", 0.02)
        if self._slam_cfg.heading_only_odom:
            rp.setdefault("use_odometry", False)
            rp.setdefault("minimum_travel_distance", 0.0)
            rp.setdefault("minimum_travel_heading", 0.0)
        elif getattr(self._slam_cfg, "map_when_still", False) and any(
            lidar.scan_source == "point_cloud" for lidar in self._slam_cfg.lidars
        ):
            # Stop-and-go Livox: bridge gates /scan. slam_toolbox always uses
            # odom→base as the match prior — widen the real correlative search
            # (default coarse angle window is only ~±20°).
            rp.setdefault("minimum_time_interval", 0.0)
            rp.setdefault("correlation_search_space_dimension", 1.0)
            rp.setdefault("link_scan_maximum_distance", 3.0)
            rp.setdefault("link_match_minimum_response_fine", 0.25)
            # ±~30° around gyro prior — NOT ±π (false room-orientation peaks).
            rp["coarse_search_angle_offset"] = float(
                self._slam_cfg.slam_params.get("coarse_search_angle_offset", 0.52)
            )
            rp["coarse_angle_resolution"] = float(
                self._slam_cfg.slam_params.get("coarse_angle_resolution", 0.0349)
            )
            rp["use_response_expansion"] = bool(
                self._slam_cfg.slam_params.get("use_response_expansion", True)
            )
            # Near-stock loop closure: wide/early search false-closes corridors.
            rp.setdefault("loop_match_minimum_chain_size", 10)
            rp.setdefault("loop_search_maximum_distance", 5.0)
            rp.setdefault("loop_search_space_dimension", 8.0)
            rp.setdefault("loop_match_minimum_response_coarse", 0.35)
            rp.setdefault("loop_match_minimum_response_fine", 0.45)
            rp.setdefault("do_loop_closing", True)
            rp.setdefault("angle_variance_penalty", 1.0)
            # Always override: continuous-Livox travel gates drop stop-and-go scans.
            rp["minimum_travel_distance"] = 0.0
            rp["minimum_travel_heading"] = 0.0
        elif any(
            lidar.scan_source == "point_cloud" for lidar in self._slam_cfg.lidars
        ):
            rp.setdefault("minimum_time_interval", 0.3)
            rp.setdefault("correlation_search_space_dimension", 0.6)
            rp.setdefault("link_scan_maximum_distance", 2.5)
            rp.setdefault("minimum_travel_distance", 0.15)
            rp.setdefault("minimum_travel_heading", 0.12)
        self._last_slam_params = rp
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
            env=self._ros_env(), check=False, timeout=30,
        )
        # Also export an occupancy grid (pgm/yaml) for inspection/export.
        subprocess.run(
            ["ros2", "run", "nav2_map_server", "map_saver_cli",
             "-f", str(map_stem), "--ros-args", "-p", "save_map_timeout:=20.0"],
            env=self._ros_env(), check=False, timeout=40,
        )

    def optimize_pose_graph(self, map_stem: Path) -> Dict:
        """Force a slam_toolbox pose-graph optimization while mapping.

        Stock slam_toolbox has no bare ``CorrectPoses`` service — SPA only runs
        on loop closure or when a serialized graph is loaded
        (``loadSerializedPoseGraph`` ends with ``solver_->Compute()``). So we
        serialize the live graph and immediately deserialize it back with the
        current map pose as the continue-mapping seed.
        """
        if not self.slam_running():
            raise RuntimeError("slam_toolbox is not running")
        stem = str(map_stem)
        pose = None
        try:
            pose = self.get_pose_in_map()
        except Exception:  # noqa: BLE001 - fall back to first-node seed
            pose = None

        # Pause scan processing for the swap (service is a toggle).
        paused = self._toggle_slam_pause()
        try:
            ser = self._run_ros(
                [
                    "ros2",
                    "service",
                    "call",
                    "/slam_toolbox/serialize_map",
                    "slam_toolbox/srv/SerializePoseGraph",
                    "{filename: '%s'}" % stem,
                ],
                timeout=60.0,
            )
            if pose is not None:
                # DeserializePoseGraph.START_AT_GIVEN_POSE = 2
                match_type = 2
                match_name = "START_AT_GIVEN_POSE"
                des_payload = (
                    "{filename: '%s', match_type: 2, "
                    "initial_pose: {x: %.6f, y: %.6f, theta: %.6f}}"
                    % (stem, float(pose.x), float(pose.y), float(pose.theta))
                )
            else:
                # DeserializePoseGraph.START_AT_FIRST_NODE = 1
                match_type = 1
                match_name = "START_AT_FIRST_NODE"
                des_payload = (
                    "{filename: '%s', match_type: 1, "
                    "initial_pose: {x: 0.0, y: 0.0, theta: 0.0}}" % stem
                )
            des = self._run_ros(
                [
                    "ros2",
                    "service",
                    "call",
                    "/slam_toolbox/deserialize_map",
                    "slam_toolbox/srv/DeserializePoseGraph",
                    des_payload,
                ],
                timeout=120.0,
            )
        finally:
            if paused:
                self._toggle_slam_pause()

        ok = (ser.returncode == 0) and (des.returncode == 0)
        return {
            "status": "optimized" if ok else "optimize_failed",
            "ok": ok,
            "match_type": match_type,
            "match_type_name": match_name,
            "map_stem": stem,
            "seed_pose": (
                {"x": pose.x, "y": pose.y, "theta": pose.theta}
                if pose is not None
                else None
            ),
            "serialize_rc": ser.returncode,
            "deserialize_rc": des.returncode,
            "serialize_out": ((ser.stdout or "") + (ser.stderr or ""))[-500:],
            "deserialize_out": ((des.stdout or "") + (des.stderr or ""))[-500:],
        }

    def _toggle_slam_pause(self) -> bool:
        """Toggle ``/slam_toolbox/pause_new_measurements``. Returns True if call ok."""
        proc = self._run_ros(
            [
                "ros2",
                "service",
                "call",
                "/slam_toolbox/pause_new_measurements",
                "slam_toolbox/srv/Pause",
                "{}",
            ],
            timeout=10.0,
        )
        return proc.returncode == 0

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
            env=self._ros_env(),
            check=False,
            timeout=10,
        )
        return proc.returncode == 0

    def slam_running(self) -> bool:
        return self._slam_proc is not None and self._slam_proc.poll() is None

    # -- Nav2 ----------------------------------------------------------------
    def _wait_for_map_tf_before_nav2(self, timeout: float = 30.0) -> None:
        """Delay Nav2 launch until localization publishes map->base_link.

        Launching earlier makes global_costmap activation wait on a transform
        stamped at its own start time; if TF appears later, that fixed stamp
        ages out of the buffer and bringup spins for minutes before aborting.
        """
        node = self._node
        if node is None:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if node._lookup_pose_in_map() is not None:
                    return
            except Exception:  # noqa: BLE001 - lookup is best-effort
                pass
            time.sleep(0.5)
        self._log(
            "map->base_link TF not available before Nav2 launch; launching anyway "
            "(costmaps may fail to activate until localization is running)"
        )

    @staticmethod
    def _params_file_sig(params_path: Path) -> str:
        try:
            return hashlib.sha256(Path(params_path).read_bytes()).hexdigest()
        except OSError:
            return ""

    def _rotate_nav2_log(self) -> None:
        """Start each Nav2 launch with a fresh log.

        The log is opened in append mode by every child process; without
        rotation, get_status diagnostics surface error lines from launches that
        happened hours or days ago, which is badly misleading.
        """
        log_path = self._scratch / "nav2_launch.log"
        if log_path.exists():
            try:
                log_path.replace(log_path.with_name("nav2_launch.log.prev"))
            except OSError:
                pass

    def start_nav2(self, nav_cfg: NavConfig, params_path: Path) -> None:
        self._nav_cfg = nav_cfg
        self._nav_params_path = params_path
        self._nav2_params_sig = self._params_file_sig(params_path)
        self.stop_nav2()
        self._rotate_nav2_log()
        self._wait_for_map_tf_before_nav2()
        # Use our bundled launch (no collision_monitor / docking / route). Stock
        # nav2_bringup always starts those; killing them races activation and
        # leaves nodes present with lifecycle timeouts / zero action servers.
        if not _NAV2_LAUNCH.is_file():
            raise FileNotFoundError(f"Nav2 launch file missing: {_NAV2_LAUNCH}")
        self._nav2_procs.append(
            self._popen(
                [
                    "ros2",
                    "launch",
                    str(_NAV2_LAUNCH),
                    f"params_file:={params_path}",
                    "use_sim_time:=false",
                    "autostart:=true",
                ]
            )
        )
        self._apply_slam_tf_params()
        # Keepout/speed filter servers are optional and must not block core
        # activation (historically the filter LM waited forever on get_state).
        if not self._wait_for_required_nav_nodes(timeout=60.0):
            self._log(
                "core Nav2 nodes still missing after launch wait; "
                f"missing={self._missing_required_nav_nodes()}"
            )
        if self._wait_for_nav_action(timeout=90.0):
            if self._node is not None:
                self._node.reset_nav_action_client()
        else:
            self._log(
                "lifecycle manager did not activate Nav2; trying manual configure/activate"
            )
            if self._activate_core_nav_nodes_manually() and self._wait_for_nav_action(
                timeout=30.0
            ):
                if self._node is not None:
                    self._node.reset_nav_action_client()
            else:
                self._log("Nav2 started but /navigate_to_pose is not ready yet")
        self._start_costmap_filter_stack(params_path)

    def _activate_core_nav_nodes_manually(self) -> bool:
        """Best-effort configure+activate when the lifecycle manager stalls."""
        core = (
            "controller_server",
            "planner_server",
            "smoother_server",
            "behavior_server",
            "bt_navigator",
            "velocity_smoother",
            "waypoint_follower",
        )
        node_list = self._run_ros(["ros2", "node", "list"])
        present = set()
        if node_list.returncode == 0:
            present = {
                line.strip().lstrip("/")
                for line in (node_list.stdout or "").splitlines()
                if line.strip()
            }
        ok_any = False
        for bare in core:
            if bare not in present:
                continue
            name = "/" + bare
            state = self._lifecycle_get_state(name)
            if state == "active":
                ok_any = True
                continue
            if state in (None, "unconfigured", "finalized"):
                if not self._lifecycle_set(name, "configure", timeout=30.0):
                    self._log(f"manual configure failed for {name}")
                    continue
            state = self._lifecycle_get_state(name)
            if state == "inactive":
                if not self._lifecycle_set(name, "activate", timeout=60.0):
                    self._log(f"manual activate failed for {name}")
                    continue
            if self._lifecycle_get_state(name) == "active":
                ok_any = True
        return ok_any

    def _start_costmap_filter_stack(self, params_path: Path) -> None:
        """Start keepout/speed filter info servers if they come up quickly.

        These are optional (zones). Never block Nav2 activation on them.
        """
        for which, node_name in (
            ("keepout", "costmap_filter_info_server_keepout"),
            ("speed", "costmap_filter_info_server_speed"),
        ):
            self._nav2_procs.append(
                self._popen(
                    [
                        "ros2",
                        "run",
                        "nav2_map_server",
                        "costmap_filter_info_server",
                        "--ros-args",
                        "-r",
                        f"__node:={node_name}",
                        "--params-file",
                        str(params_path),
                    ]
                )
            )
        # Only start the filter LM if both servers register; otherwise it spam-
        # waits on get_state forever and adds DDS/CPU noise during bringup.
        deadline = time.monotonic() + 15.0
        have_both = False
        while time.monotonic() < deadline:
            proc = self._run_ros(["ros2", "node", "list"])
            nodes = proc.stdout or ""
            if (
                "costmap_filter_info_server_keepout" in nodes
                and "costmap_filter_info_server_speed" in nodes
            ):
                have_both = True
                break
            time.sleep(0.5)
        if not have_both:
            self._log(
                "costmap filter info servers did not register; skipping filter "
                "lifecycle manager (keepout/speed zones inactive until next restart)"
            )
            return
        lm_params = self._scratch / "filter_lifecycle.yaml"
        lm_params.write_text(
            _yaml_dump(
                {
                    "filter_lifecycle_manager": {
                        "ros__parameters": {
                            "autostart": True,
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
                [
                    "ros2",
                    "run",
                    "nav2_lifecycle_manager",
                    "lifecycle_manager",
                    "--ros-args",
                    "-r",
                    "__node:=filter_lifecycle_manager",
                    "--params-file",
                    str(lm_params),
                ]
            )
        )

    def _apply_slam_tf_params(self) -> None:
        """Push slam_toolbox TF timing params onto a running node (no restart needed)."""
        if not self.slam_running():
            return
        self._run_ros(
            [
                "ros2",
                "param",
                "set",
                "/slam_toolbox",
                "transform_timeout",
                "0.0",
            ],
            timeout=5.0,
        )

    def _wait_for_required_nav_nodes(self, timeout: float = 45.0) -> bool:
        """Block until core Nav2 nodes appear in the graph (not yet activated)."""
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() < deadline:
            if self._required_nav_nodes_present():
                return True
            time.sleep(1.0)
        missing = self._missing_required_nav_nodes()
        self._log(
            f"Nav2 core nodes not all present after launch wait "
            f"(missing: {missing})"
        )
        return False

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
            "launch/navigation_launch.py",
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
        # Jazzy bringup always spawns these; we pkill them. Their death lines
        # dominate get_status and hide real configure/activate failures.
        suppress_noise = (
            "lifecycle_manager_navigation",
            "nav2_collision_monitor/collision_monitor",
            "context cannot be slept with because it's invalid",
        )
        interesting = [
            ln
            for ln in lines
            if any(m in ln for m in markers)
            and not any(n in ln for n in suppress_noise)
        ]
        return "\n".join(interesting[-max_lines:])

    def nav2_diagnostics(self) -> Dict:
        action_proc = self._run_ros(["ros2", "action", "list"])
        action_info_proc = self._run_ros(["ros2", "action", "info", "/navigate_to_pose"])
        node_proc = self._run_ros(["ros2", "node", "list"])
        lifecycle_proc = self._run_ros(["ros2", "lifecycle", "get", "/bt_navigator"])
        controller_proc = self._run_ros(["ros2", "lifecycle", "get", "/controller_server"])
        # Read back the frequency the running controller actually loaded, so a
        # stale Nav2 running old params is visible directly in get_status.
        freq_proc = self._run_ros(
            ["ros2", "param", "get", "/controller_server", "controller_frequency"]
        )
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
            "controller_frequency_loaded": (
                freq_proc.stdout or freq_proc.stderr or ""
            ).strip(),
            "bt_navigator_lifecycle": (lifecycle_proc.stdout or lifecycle_proc.stderr or "").strip(),
            "nav2_log_errors": self._nav2_log_errors(),
            "nav2_log_tail": log_tail,
            "nav2_log_path": str(log_path),
        }

    def ensure_nav2(self, nav_cfg: NavConfig, params_path: Path) -> None:
        """Start or restart Nav2 until ``/navigate_to_pose`` is available."""
        if self.nav_action_ready():
            # Nav2 params are only read at process start. If the generated
            # params changed (retuning, module update), a healthy Nav2 is still
            # running with stale settings and must be restarted to apply them.
            if self._nav2_params_sig and self._params_file_sig(params_path) == self._nav2_params_sig:
                return
            self._log("Nav2 params changed since launch; restarting Nav2 to apply them")
            self.stop_nav2()

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
        self._apply_slam_tf_params()
        # Cancel any stuck recovery / prior goal so a second navigate_to_location
        # cannot appear to do nothing while BT is still busy.
        try:
            node.cancel_nav()
        except Exception:  # noqa: BLE001
            pass
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

    def compute_path(
        self,
        x: float,
        y: float,
        theta: float = 0.0,
        *,
        planner_id: str = "GridBased",
        start: Optional[conv.Pose2D] = None,
        timeout_s: float = 20.0,
        max_points: int = 400,
    ) -> Dict:
        """Plan a Nav2 path to ``(x, y, theta)`` without moving the base."""
        node = self._require_node()
        if self.nav2_startup_in_progress():
            raise RuntimeError(
                "Nav2 is still starting up in the background; retry in a few seconds "
                "(check progress with the get_status command)"
            )
        self._apply_slam_tf_params()
        if not self.nav_action_ready():
            # Do not block plan on a multi-minute ensure_nav2 retry loop — the UI
            # looks hung. Ask the caller to restart_nav2 / wait for readiness.
            raise RuntimeError(
                "Nav2 is not ready (no active /navigate_to_pose). "
                "Check get_status: nav_action_ready should be true and "
                "bt_navigator_lifecycle should be active. Try restart_nav2."
            )
        try:
            return node.compute_path_to_pose(
                x,
                y,
                theta,
                planner_id=planner_id,
                start=start,
                timeout_s=timeout_s,
                max_points=max_points,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "not available" not in msg and "unavailable" not in msg:
                raise
        if self._nav_cfg is None or self._nav_params_path is None:
            raise RuntimeError(
                "compute_path_to_pose not available and Nav2 has no saved startup config"
            )
        self._log("compute_path unavailable; ensuring Nav2 and retrying")
        self.ensure_nav2(self._nav_cfg, self._nav_params_path)
        return node.compute_path_to_pose(
            x,
            y,
            theta,
            planner_id=planner_id,
            start=start,
            timeout_s=timeout_s,
            max_points=max_points,
        )

    def last_preview_plan(self) -> Optional[Dict]:
        node = self._node
        if node is None:
            return None
        return node.last_preview_plan()

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

    def apply_map_pose_correction(self, pose: conv.Pose2D) -> Dict:
        """Mapping-mode revisit fix: shift odom TF so the slam prior hits ``pose``."""
        return self._require_node().apply_map_pose_correction(pose)

    def set_still_keyframe_hook(self, hook) -> None:
        """Forward still-publish keyframe callback to the bridge (or clear)."""
        node = self._node
        if node is not None:
            node.set_still_keyframe_hook(hook)

    def get_base_scan(self, max_age_s: float = 1.0) -> Optional[conv.LaserScan2D]:
        node = self._node
        if node is None:
            return None
        return node.get_base_scan(max_age_s)

    def slam_diagnostics(self) -> Dict:
        """Runtime health snapshot for SLAM (bridge + slam_toolbox process)."""
        lifecycle_proc = self._run_ros(["ros2", "lifecycle", "get", SLAM_LIFECYCLE_NODE])
        node_proc = self._run_ros(["ros2", "node", "list"])
        bridge: Dict = {}
        if self._node is not None:
            try:
                bridge = self._node.slam_bridge_status()
            except Exception:  # noqa: BLE001 - diagnostics must not raise
                pass
        lifecycle_text = (lifecycle_proc.stdout or lifecycle_proc.stderr or "").strip()
        # Prefer the live DDS graph over `ros2 node list`: the CLI answers from
        # a discovery-daemon cache that can serve long-dead nodes (phantom
        # duplicates) — our own participant sees the graph as it is right now.
        node_names: List[str] = []
        if self._node is not None:
            try:
                node_names = [
                    ("/" + name) if ns in ("", "/") else (ns.rstrip("/") + "/" + name)
                    for name, ns in self._node.get_node_names_and_namespaces()
                ]
            except Exception:  # noqa: BLE001
                node_names = []
        if not node_names:
            node_names = [
                n.strip() for n in (node_proc.stdout or "").splitlines() if n.strip()
            ]
        duplicates = sorted(
            {n for n in node_names if node_names.count(n) > 1 and "transform_listener" not in n}
        )
        binary_count = self._slam_toolbox_binary_count()
        map_publishers = None
        if self._node is not None:
            try:
                map_publishers = self._node.map_publisher_count()
            except Exception:  # noqa: BLE001
                map_publishers = None
        return {
            # DDS often lists stale /slam_toolbox names after pkill — trust
            # slam_toolbox_binary_count (and `ps`) over duplicate_ros_nodes.
            "duplicate_ros_nodes": duplicates,
            "slam_toolbox_binary_count": binary_count,
            "slam_toolbox_running": self.slam_running(),
            "slam_toolbox_lifecycle": lifecycle_text,
            "slam_toolbox_active": self._lifecycle_get_state(SLAM_LIFECYCLE_NODE)
            == "active",
            "slam_toolbox_params": {
                k: getattr(self, "_last_slam_params", {}).get(k)
                for k in (
                    "use_odometry",
                    "use_tf_scan_transformation",
                    "minimum_travel_distance",
                    "minimum_travel_heading",
                    "correlation_search_space_dimension",
                    "coarse_search_angle_offset",
                    "coarse_angle_resolution",
                    "link_match_minimum_response_fine",
                    "minimum_time_interval",
                )
                if getattr(self, "_last_slam_params", {}).get(k) is not None
            },
            "ros_nodes": (node_proc.stdout or "").strip(),
            "map_publisher_count": map_publishers,
            **dds_status(),
            **bridge,
        }

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


"""rclpy bridge node.

Bridges Viam components <-> ROS2 so that slam_toolbox and Nav2 (running as separate
processes) can see the robot's sensors and drive its base:

* reads each Viam lidar -> publishes ``/scan_<i>`` (in that lidar's frame) and a
  single merged ``/scan`` (in base_link) for slam_toolbox
* publishes wheel/IMU odometry as ``/odom`` + the ``odom -> base_link`` TF
* publishes static ``base_link -> laser_<i>`` TFs from each lidar's mount
* subscribes ``/cmd_vel`` and drives the Viam base (only while navigation is active),
  with a watchdog that stops the base if commands go stale
* publishes keepout / speed costmap-filter masks as latched ``OccupancyGrid``s
* exposes a ``NavigateToPose`` action client for the navigation model

The node runs in a background thread; Viam component calls (which are async) are
marshalled onto the module's asyncio loop via ``run_coroutine_threadsafe``.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.action import ActionClient

from geometry_msgs.msg import Quaternion, Twist, TransformStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tf2_ros import (
    TransformBroadcaster,
    StaticTransformBroadcaster,
    Buffer,
    TransformListener,
)

try:
    from nav2_msgs.action import NavigateToPose
except Exception:  # pragma: no cover - nav2 may be absent in mapping-only installs
    NavigateToPose = None

from ..config import NavConfig, SlamConfig
from . import conversions as conv

_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

_NAV_ACTION_NAME = "/navigate_to_pose"
_NAV_ACTION_CLIENT_NAME = "navigate_to_pose"


def _quat_msg(yaw: float) -> Quaternion:
    x, y, z, w = conv.yaw_to_quaternion(yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


class IOProvider:
    """Adapter the bridge uses to talk to Viam components.

    The navigation/SLAM models supply concrete async callables; the bridge stays
    free of any Viam SDK imports.
    """

    def __init__(self, read_lidar_points, read_odometry, drive_base, stop_base):
        self.read_lidar_points = read_lidar_points  # async (name) -> (N,3) np.ndarray
        self.read_odometry = read_odometry  # async () -> conv.OdomReading
        self.drive_base = drive_base  # async (vx, vy, vtheta) -> None
        self.stop_base = stop_base  # async () -> None


class BridgeNode(Node):
    def __init__(
        self,
        slam_cfg: SlamConfig,
        io: IOProvider,
        loop: asyncio.AbstractEventLoop,
        nav_cfg: Optional[NavConfig] = None,
        node_name: str = "viam_nav_stack_bridge",
    ):
        super().__init__(node_name)
        self._slam_cfg = slam_cfg
        self._nav_cfg = nav_cfg
        self._io = io
        self._loop = loop

        self._frames = slam_cfg.frames
        self._nav_active = False
        self._last_cmd_time = 0.0
        self._last_odom_stamp = None
        self._cmd_timeout = nav_cfg.cmd_vel_timeout if nav_cfg else 0.5

        # Odom pose: prefer the movement sensor's absolute /odom pose when available.
        self._odom = conv.Pose2D(0.0, 0.0, 0.0)
        self._last_twist = (0.0, 0.0, 0.0)
        self._last_odom_time = time.monotonic()
        self._last_odom_pub_wall = time.monotonic()
        self._odom_integrate_warned = False
        self._empty_scan_warned = False

        # Publishers -------------------------------------------------------
        self._scan_pubs: List = []
        for i, _ in enumerate(slam_cfg.lidars):
            self._scan_pubs.append(self.create_publisher(LaserScan, f"scan_{i}", 10))
        self._merged_scan_pub = self.create_publisher(LaserScan, "scan", 10)
        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "initialpose", 10
        )
        self._keepout_pub = self.create_publisher(
            OccupancyGrid, "keepout_filter_mask", _LATCHED_QOS
        )
        self._speed_pub = self.create_publisher(
            OccupancyGrid, "speed_filter_mask", _LATCHED_QOS
        )

        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_lidar_tfs()

        # TF listener for map-frame pose queries. spin_thread=True gives the
        # /tf subscription its own executor thread so blocked timer callbacks
        # (slow MiR rosbridge reads) can never starve it and stale the buffer.
        self._tf_buffer = Buffer()
        try:
            self._tf_listener = TransformListener(
                self._tf_buffer, self, spin_thread=True
            )
        except TypeError:  # pragma: no cover - older tf2_ros without the kwarg
            self._tf_listener = TransformListener(self._tf_buffer, self)

        # Latest occupancy map published by slam_toolbox.
        self._latest_map: Optional[Dict] = None
        self._map_updates_enabled = True
        # Last successful map->base pose; used when TF lookup transiently fails.
        self._last_pose_in_map: Optional[conv.Pose2D] = None

        # Callback groups --------------------------------------------------
        self._cb_scan = None
        self._cb_odom = None
        self._cb_misc = None
        try:
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

            self._cb_scan = MutuallyExclusiveCallbackGroup()
            self._cb_odom = MutuallyExclusiveCallbackGroup()
            self._cb_misc = MutuallyExclusiveCallbackGroup()
        except Exception:
            # Older/minimal test environments may not expose callback_groups.
            pass

        # Subscriptions ----------------------------------------------------
        sub_kwargs = (
            {"callback_group": self._cb_misc}
            if self._cb_misc is not None
            else {}
        )
        # Drive only from the velocity smoother output. Subscribing to cmd_vel_nav
        # (raw MPPI) as well made latest-wins pick unsmoothed commands and caused
        # overshoot on momentum-heavy bases like the MiR.
        self.create_subscription(
            Twist, "cmd_vel_smoothed", self._guarded(self._on_cmd_vel), 10, **sub_kwargs
        )
        self._map_sub = self.create_subscription(
            OccupancyGrid, "map", self._guarded(self._on_map), _LATCHED_QOS, **sub_kwargs
        )
        self._map_generation = 0

        # Timers -----------------------------------------------------------
        scan_timer_kwargs = (
            {"callback_group": self._cb_scan}
            if self._cb_scan is not None
            else {}
        )
        odom_timer_kwargs = (
            {"callback_group": self._cb_odom}
            if self._cb_odom is not None
            else {}
        )
        misc_timer_kwargs = (
            {"callback_group": self._cb_misc}
            if self._cb_misc is not None
            else {}
        )
        self.create_timer(
            1.0 / max(slam_cfg.scan_rate_hz, 1.0),
            self._guarded(self._on_scan_timer),
            **scan_timer_kwargs,
        )
        self.create_timer(
            1.0 / max(slam_cfg.odom_rate_hz, 1.0),
            self._guarded(self._on_odom_timer),
            **odom_timer_kwargs,
        )
        self.create_timer(0.1, self._guarded(self._on_watchdog), **misc_timer_kwargs)

        # Dedicated group so a slow base call can never back up cmd_vel
        # subscription callbacks (latest-wins dispatch, see _on_drive_timer).
        drive_timer_kwargs: Dict = {}
        try:
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

            drive_timer_kwargs = {"callback_group": MutuallyExclusiveCallbackGroup()}
        except Exception:
            pass
        self._pending_cmd_vel: Optional[tuple] = None
        self._cmd_vel_lock = threading.Lock()
        self.create_timer(0.05, self._guarded(self._on_drive_timer), **drive_timer_kwargs)

        # Action client (created lazily once Nav2 is running) ----------------
        self._nav_action: Optional[ActionClient] = None
        self._goal_handle = None
        self._nav_cli_proc: Optional[subprocess.Popen] = None
        self._nav_goal_lock = threading.Lock()
        self._pending_nav_goal: Optional[tuple] = None
        self._executor_lock = threading.Lock()
        self._executor_queue: List = []
        self._last_feedback: Dict = {}
        self._last_result_status: Optional[str] = None

    # -- helpers -------------------------------------------------------------
    def _guarded(self, fn):
        """Wrap a timer/subscription callback so exceptions cannot kill the executor.

        An uncaught exception in any callback aborts ``executor.spin()``, which
        silently freezes every timer and subscription (TF/odom/scans stop and
        Nav2 loses the robot pose).
        """

        def wrapped(*args):
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001 - keep the executor alive
                try:
                    self.get_logger().error(
                        f"callback {getattr(fn, '__name__', fn)!r} crashed: {exc!r}"
                    )
                except Exception:  # noqa: BLE001 - logging must never re-raise
                    pass

        return wrapped

    def _run(self, coro, timeout: float = 2.0):
        """Run an async Viam call on the module loop from this ROS thread."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _submit_on_executor(self, fn, timeout: float = 10.0):
        """Run ``fn`` on the rclpy executor thread (safe for publishers/actions)."""
        done = threading.Event()
        outcome: Dict[str, object] = {}

        def wrapped() -> None:
            try:
                outcome["result"] = fn()
            except Exception as exc:  # noqa: BLE001 - propagate to caller thread
                outcome["exc"] = exc
            finally:
                done.set()

        with self._executor_lock:
            self._executor_queue.append(wrapped)
        if not done.wait(timeout=timeout):
            raise RuntimeError("ROS executor dispatch timed out")
        if "exc" in outcome:
            raise outcome["exc"]  # type: ignore[misc]
        return outcome.get("result")

    def _flush_executor_queue(self) -> None:
        with self._executor_lock:
            queue = self._executor_queue[:]
            self._executor_queue.clear()
        for fn in queue:
            fn()

    def _bounded_odom_dt(self, dt: float) -> float:
        """Bound odom integration dt to avoid large stale-velocity jumps."""
        if dt <= 0.0:
            return 0.0
        hz = max(float(self._slam_cfg.odom_rate_hz), 1.0)
        # Keep a generous cap so slower rosbridge cycles still integrate, while
        # preventing single very-late samples from creating huge pose jumps.
        stale_dt = max(8.0 / hz, 0.5)
        if dt > stale_dt:
            self.get_logger().warn(
                f"odom integration dt {dt:.3f}s exceeds stale threshold "
                f"{stale_dt:.3f}s; clamping integration step"
            )
            return stale_dt
        return dt

    def _now_msg(self) -> Header:
        return Header(stamp=self.get_clock().now().to_msg(), frame_id=self._frames.base_link)

    # -- static TF -----------------------------------------------------------
    def _publish_static_lidar_tfs(self) -> None:
        transforms = []
        for i, lidar in enumerate(self._slam_cfg.lidars):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self._frames.base_link
            t.child_frame_id = f"laser_{i}"
            t.transform.translation.x = lidar.x
            t.transform.translation.y = lidar.y
            t.transform.translation.z = lidar.z
            t.transform.rotation = _quat_msg(lidar.theta)
            transforms.append(t)
        if transforms:
            self._static_tf.sendTransform(transforms)

    # -- scans ---------------------------------------------------------------
    def _on_scan_timer(self) -> None:
        merged_points = []
        per_lidar_scans = []
        lidar_timeout = max(float(self._slam_cfg.sensor_read_timeout_s), 1.0)
        for i, lidar in enumerate(self._slam_cfg.lidars):
            try:
                lidar_data = self._run(
                    self._io.read_lidar_points(lidar.name), timeout=lidar_timeout
                )
            except Exception as exc:  # noqa: BLE001 - keep bridge alive on sensor hiccups
                # repr() because timeout errors often have an empty str().
                self.get_logger().warn(f"lidar {lidar.name} read failed: {exc!r}")
                continue

            sensor_scan = None
            base_pts_arr = np.empty((0, 3))
            if isinstance(lidar_data, conv.LidarPoints):
                sensor_scan = lidar_data.sensor_scan
                base_pts_arr = np.asarray(lidar_data.base_link, dtype=float)
            else:
                base_pts_arr = np.asarray(lidar_data, dtype=float)

            if sensor_scan is None or not conv.scan_has_returns(sensor_scan):
                sensor_pts = (
                    np.asarray(lidar_data.sensor, dtype=float)
                    if isinstance(lidar_data, conv.LidarPoints)
                    else base_pts_arr
                )
                if sensor_pts.size == 0 and base_pts_arr.size == 0:
                    continue
                sensor_scan = conv.pointcloud_to_scan(
                    sensor_pts,
                    z_min=lidar.z_min,
                    z_max=lidar.z_max,
                    num_bins=self._slam_cfg.scan_bins,
                    range_min=lidar.min_range,
                    range_max=lidar.max_range,
                )
                if not conv.scan_has_returns(sensor_scan):
                    continue

            per_lidar_scans.append((i, sensor_scan))

            if base_pts_arr.size:
                merged_points.append(base_pts_arr[:, :2])

        if not per_lidar_scans:
            if not self._empty_scan_warned:
                self.get_logger().error(
                    "no lidar data available; /scan not published — check MiR lidar "
                    "rosbridge (scan_mode, mir_rosbridge_timeout_s) and "
                    "slam sensor_read_timeout_s"
                )
                self._empty_scan_warned = True
            return
        self._empty_scan_warned = False

        # Fresh stamp + synchronized odom TF. Lidar reads block the module asyncio
        # loop (MiR rosbridge), so _last_odom_stamp can lag seconds behind the TF
        # cache while scans still carry that stale time — Nav2 then drops them.
        stamp = self.get_clock().now().to_msg()
        vx, vy, vtheta = self._last_twist
        self._publish_odom_snapshot(stamp, vx, vy, vtheta)
        for i, scan in per_lidar_scans:
            self._scan_pubs[i].publish(self._to_ros_scan(scan, f"laser_{i}", stamp))

        stacked = np.vstack(merged_points) if merged_points else np.empty((0, 2))
        ref = self._slam_cfg.lidars[0]
        merged = conv.points_to_scan(
            stacked,
            num_bins=self._slam_cfg.scan_bins,
            range_min=ref.min_range,
            range_max=ref.max_range,
        )
        if not conv.scan_has_returns(merged):
            if not self._empty_scan_warned:
                self.get_logger().error(
                    "lidar points received but /scan has no valid returns"
                )
                self._empty_scan_warned = True
            return
        self._empty_scan_warned = False
        self._merged_scan_pub.publish(
            self._to_ros_scan(merged, self._frames.base_link, stamp)
        )

    def _to_ros_scan(self, scan: conv.LaserScan2D, frame_id: str, stamp) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.angle_min = float(scan.angle_min)
        msg.angle_increment = float(scan.angle_increment)
        msg.angle_max = float(scan.angle_min + scan.angle_increment * len(scan.ranges))
        msg.range_min = float(scan.range_min)
        msg.range_max = float(scan.range_max if math.isfinite(scan.range_max) else 100.0)
        ranges = np.asarray(scan.ranges, dtype=float)
        msg.ranges = [float(r) for r in ranges]
        return msg

    # -- odometry ------------------------------------------------------------
    def _on_odom_timer(self) -> None:
        try:
            sample = self._run(self._io.read_odometry())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"odometry read failed: {exc!r}")
            return
        now = time.monotonic()
        raw_dt = now - self._last_odom_time
        self._last_odom_time = now
        if sample.pose is not None:
            self._odom = sample.pose
        elif sample.heading_rad is not None:
            dt = self._bounded_odom_dt(raw_dt)
            if dt <= 0:
                self._odom = conv.Pose2D(
                    self._odom.x, self._odom.y, sample.heading_rad
                )
            else:
                th = sample.heading_rad
                vx, vy, _vtheta = sample.vx, sample.vy, sample.vtheta
                dx = (vx * math.cos(th) - vy * math.sin(th)) * dt
                dy = (vx * math.sin(th) + vy * math.cos(th)) * dt
                self._odom = conv.Pose2D(
                    self._odom.x + dx,
                    self._odom.y + dy,
                    sample.heading_rad,
                )
        else:
            dt = self._bounded_odom_dt(raw_dt)
            if not self._odom_integrate_warned:
                self.get_logger().warn(
                    "movement sensor did not provide odom pose or heading in "
                    "get_readings(); dead-reckoning from velocity — orientation "
                    "drift likely"
                )
                self._odom_integrate_warned = True
            if dt <= 0:
                return
            th = self._odom.theta
            vx, vy, vtheta = sample.vx, sample.vy, sample.vtheta
            dx = (vx * math.cos(th) - vy * math.sin(th)) * dt
            dy = (vx * math.sin(th) + vy * math.cos(th)) * dt
            self._odom = conv.Pose2D(
                self._odom.x + dx, self._odom.y + dy, th + vtheta * dt
            )

        vx, vy, vtheta = sample.vx, sample.vy, sample.vtheta
        stamp = self.get_clock().now().to_msg()
        self._publish_odom_snapshot(stamp, vx, vy, vtheta)

    def _publish_odom_snapshot(
        self, stamp, vx: float, vy: float, vtheta: float
    ) -> None:
        """Publish cached odom pose + TF at ``stamp`` (also used to sync with scans)."""
        # Snapshot the reference once: the scan timer and odom timer run on
        # different executor threads, and Pose2D is reassigned atomically.
        pose = self._odom
        self._last_odom_stamp = stamp
        self._last_odom_pub_wall = time.monotonic()
        self._last_twist = (vx, vy, vtheta)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._frames.odom
        odom.child_frame_id = self._frames.base_link
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.orientation = _quat_msg(pose.theta)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = vtheta
        self._odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._frames.odom
        t.child_frame_id = self._frames.base_link
        t.transform.translation.x = pose.x
        t.transform.translation.y = pose.y
        t.transform.rotation = _quat_msg(pose.theta)
        self._tf_broadcaster.sendTransform(t)

    # -- cmd_vel + watchdog --------------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        """Stash the newest velocity command; never block in the callback.

        Nav2's velocity_smoother publishes ramped velocities on cmd_vel_smoothed.
        A Viam->base call can take 100-300 ms. Driving the base inline made
        callbacks queue up, so the robot executed velocity commands that were
        seconds stale. Latest-wins dispatch happens on _on_drive_timer.
        """
        self._last_cmd_time = time.time()
        if not self._nav_active:
            return
        vx = msg.linear.x
        vy = msg.linear.y if self._is_omni() else 0.0
        vtheta = msg.angular.z
        with self._cmd_vel_lock:
            self._pending_cmd_vel = (vx, vy, vtheta)

    def _on_drive_timer(self) -> None:
        """Send only the freshest cmd_vel to the base (drops superseded ones)."""
        with self._cmd_vel_lock:
            pending = self._pending_cmd_vel
            self._pending_cmd_vel = None
        if pending is None or not self._nav_active:
            return
        vx, vy, vtheta = pending
        # Snap near-zero commands so the MiR does not creep past the goal.
        if abs(vx) < 0.03 and abs(vy) < 0.03 and abs(vtheta) < 0.05:
            vx, vy, vtheta = 0.0, 0.0, 0.0
        try:
            self._run(self._io.drive_base(vx, vy, vtheta))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"drive_base failed: {exc!r}")

    def _on_watchdog(self) -> None:
        self._flush_executor_queue()
        self._flush_pending_nav_goal()
        self._keep_odom_tf_alive()
        if not self._nav_active:
            return
        if time.time() - self._last_cmd_time > self._cmd_timeout:
            try:
                self._run(self._io.stop_base())
            except Exception:  # noqa: BLE001
                pass

    def odom_tf_age_s(self) -> float:
        """Seconds since the bridge last published odom + TF (liveness signal)."""
        return round(time.monotonic() - self._last_odom_pub_wall, 2)

    def _keep_odom_tf_alive(self) -> None:
        """Republish last-known odom pose when odometry reads stall.

        Nav2 costmaps refuse to (re)activate without a recent base_link->odom
        transform; a stretch of failed movement-sensor reads must not take the
        whole TF chain down with it. Velocity is zeroed so stale twists cannot
        leak into the velocity smoother.
        """
        gap = max(3.0 / max(float(self._slam_cfg.odom_rate_hz), 1.0), 1.0)
        if time.monotonic() - self._last_odom_pub_wall < gap:
            return
        self._publish_odom_snapshot(self.get_clock().now().to_msg(), 0.0, 0.0, 0.0)

    def _is_omni(self) -> bool:
        return bool(self._nav_cfg and self._nav_cfg.kinematics == "omni")

    def set_nav_config(self, nav_cfg: NavConfig) -> None:
        self._nav_cfg = nav_cfg
        self._cmd_timeout = nav_cfg.cmd_vel_timeout

    def set_nav_active(self, active: bool) -> None:
        self._nav_active = active
        if active:
            # Avoid the watchdog calling stop_base before Nav2 publishes the first cmd_vel.
            self._last_cmd_time = time.time()
        else:
            with self._cmd_vel_lock:
                self._pending_cmd_vel = None
            try:
                self._run(self._io.stop_base())
            except Exception:  # noqa: BLE001
                pass

    # -- costmap filter masks ------------------------------------------------
    def publish_mask(
        self,
        which: str,
        mask: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self._frames.map
        grid.info.resolution = float(resolution)
        grid.info.height, grid.info.width = mask.shape
        grid.info.origin.position.x = float(origin_x)
        grid.info.origin.position.y = float(origin_y)
        grid.info.origin.orientation.w = 1.0
        grid.data = [int(v) for v in np.asarray(mask, dtype=np.int8).flatten()]
        if which == "keepout":
            self._keepout_pub.publish(grid)
        elif which == "speed":
            self._speed_pub.publish(grid)
        else:
            raise ValueError(f"unknown mask {which!r}")

    # -- navigation action ---------------------------------------------------
    def reset_nav_action_client(self) -> None:
        """Drop and recreate the action client (call after Nav2 starts)."""
        if self._nav_action is not None:
            try:
                self._nav_action.destroy()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._nav_action = None

    def _ensure_nav_action_client(self) -> None:
        if NavigateToPose is None:
            return
        if self._nav_action is None:
            self._nav_action = ActionClient(self, NavigateToPose, _NAV_ACTION_CLIENT_NAME)

    def _wait_for_map_tf(self, timeout_s: float = 8.0) -> bool:
        """Block until map->base_link is available (post-localize / set_initial_pose).

        Uses a raw TF lookup, not get_pose_in_map(): that helper falls back to a
        cached pose on lookup failure, which would defeat this readiness check.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._lookup_pose_in_map() is not None:
                return True
            time.sleep(0.1)
        return False

    def _goal_pose_stamp(self):
        """Use latest-available TF; avoids extrapolation when slam lags scan stamps."""
        try:
            from builtin_interfaces.msg import Time as TimeMsg

            return TimeMsg(sec=0, nanosec=0)
        except ImportError:
            return self.get_clock().now().to_msg()

    def send_nav_goal(self, x: float, y: float, theta: float) -> bool:
        """Send a map-frame goal to Nav2."""
        if NavigateToPose is None:
            raise RuntimeError(f"Nav2 action {_NAV_ACTION_NAME!r} unavailable")
        # Shorter wait once localization has succeeded before: we would proceed
        # anyway, so do not add 8s of latency to every goal on a stale buffer.
        tf_wait_s = 8.0 if self._last_pose_in_map is None else 3.0
        if not self._wait_for_map_tf(timeout_s=tf_wait_s):
            # Only hard-fail when localization has never succeeded. A stale
            # bridge-local TF buffer must not block goals: bt_navigator has its
            # own buffer and is the real authority, and will abort visibly if
            # the map pose is truly unavailable.
            if self._last_pose_in_map is None:
                raise RuntimeError(
                    "map->base_link transform not available; run set_initial_pose "
                    "or global_localize and wait for localization before navigating"
                )
            self.get_logger().warn(
                "map->base_link lookup timed out but localization previously "
                "succeeded; sending goal anyway (Nav2 aborts if pose is unavailable)"
            )
        self._ensure_nav_action_client()
        if self._wait_for_rclpy_action_server():
            self._cancel_inflight_nav()
            return self._publish_nav_goal(x, y, theta)
        # rclpy action clients can become stale after Nav2 restarts; recreate once.
        self.reset_nav_action_client()
        self._ensure_nav_action_client()
        if self._wait_for_rclpy_action_server():
            self._cancel_inflight_nav()
            return self._publish_nav_goal(x, y, theta)
        # Last-resort fallback to CLI send if discovery sees the action.
        if self._cli_nav_action_visible(timeout=2.0):
            self._cancel_inflight_nav()
            return self._send_nav_goal_via_cli(x, y, theta)
        self._log_nav_action_diagnostics()
        raise RuntimeError("Nav2 action server not available")

    def _cancel_inflight_nav(self) -> None:
        if self._nav_cli_proc is not None and self._nav_cli_proc.poll() is None:
            self._nav_cli_proc.terminate()
            self._nav_cli_proc = None
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:  # noqa: BLE001 - best-effort cancel before new goal
                pass
            self._goal_handle = None

    def _flush_pending_nav_goal(self) -> None:
        with self._nav_goal_lock:
            pending = self._pending_nav_goal
            if pending is None:
                return
            x, y, theta, done, outcome = pending
            self._pending_nav_goal = None
        try:
            self._cancel_inflight_nav()
            self._ensure_nav_action_client()
            outcome["ok"] = self._publish_nav_goal(x, y, theta)
        except Exception as exc:  # noqa: BLE001 - propagate to caller thread
            outcome["exc"] = exc
        finally:
            done.set()

    def _dispatch_rclpy_nav_goal(self, x: float, y: float, theta: float) -> bool:
        # Force dispatch onto executor and wait briefly so we fail fast instead
        # of reporting "navigating" for a goal that never left this process.
        res = self._submit_on_executor(
            lambda: self._publish_nav_goal(x, y, theta), timeout=3.0
        )
        return bool(res)

    def _wait_for_rclpy_action_server(self) -> bool:
        if self._nav_action is None:
            return False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._nav_action.server_is_ready():
                return True
            time.sleep(0.2)
        return False

    def _ros_env(self) -> dict:
        env = os.environ.copy()
        env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        env.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        distro = env.get("ROS_DISTRO", "jazzy")
        ros_bin = f"/opt/ros/{distro}/bin"
        path = env.get("PATH", "")
        if ros_bin not in path.split(os.pathsep):
            env["PATH"] = ros_bin + os.pathsep + path
        return env

    def _ros2_cmd(self) -> str:
        distro = os.environ.get("ROS_DISTRO", "jazzy")
        return shutil.which("ros2") or f"/opt/ros/{distro}/bin/ros2"

    def _cli_nav_action_visible(self, timeout: float = 5.0) -> bool:
        info_proc = subprocess.run(
            [self._ros2_cmd(), "action", "info", _NAV_ACTION_NAME],
            env=self._ros_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if info_proc.returncode == 0:
            match = re.search(r"Action servers:\s*(\d+)", info_proc.stdout or "")
            if match:
                return int(match.group(1)) > 0
            return True
        proc = subprocess.run(
            [self._ros2_cmd(), "action", "list"],
            env=self._ros_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return proc.returncode == 0 and "navigate_to_pose" in (proc.stdout or "")

    def _log_nav_action_diagnostics(self) -> None:
        try:
            proc = subprocess.run(
                [self._ros2_cmd(), "action", "list"],
                env=self._ros_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
        except subprocess.TimeoutExpired:
            cli_actions = "timeout"
            proc = subprocess.CompletedProcess([], returncode=-1)
        else:
            cli_actions = (proc.stdout or "").strip() if proc.returncode == 0 else "unavailable"
        self.get_logger().error(
            f"Nav2 action {_NAV_ACTION_NAME!r} not ready "
            f"(ros2={self._ros2_cmd()}, rc={proc.returncode}, "
            f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}, "
            f"RMW={os.environ.get('RMW_IMPLEMENTATION', 'default')}, "
            f"cli_actions={cli_actions!r}, cli_stderr={(proc.stderr or '').strip()!r})"
        )

    def _send_nav_goal_via_cli(self, x: float, y: float, theta: float) -> bool:
        q = _quat_msg(theta)
        goal_yaml = (
            "{pose: {header: {frame_id: '"
            + self._frames.map
            + "'}, pose: {position: {x: "
            + str(float(x))
            + ", y: "
            + str(float(y))
            + ", z: 0.0}, orientation: {x: "
            + str(q.x)
            + ", y: "
            + str(q.y)
            + ", z: "
            + str(q.z)
            + ", w: "
            + str(q.w)
            + "}}}}}"
        )
        if self._nav_cli_proc is not None and self._nav_cli_proc.poll() is None:
            self._nav_cli_proc.terminate()
        self._nav_cli_proc = subprocess.Popen(
            [
                self._ros2_cmd(),
                "action",
                "send_goal",
                _NAV_ACTION_NAME,
                "nav2_msgs/action/NavigateToPose",
                goal_yaml,
            ],
            env=self._ros_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._last_result_status = "active"
        self.set_nav_active(True)
        threading.Thread(
            target=self._watch_cli_nav_proc,
            args=(self._nav_cli_proc,),
            daemon=True,
        ).start()
        return True

    def _watch_cli_nav_proc(self, proc: subprocess.Popen) -> None:
        try:
            output = proc.communicate(timeout=20.0)[0] or ""
        except subprocess.TimeoutExpired:
            proc.terminate()
            self.get_logger().warn(
                "nav CLI goal send timed out waiting for action server/result"
            )
            self._last_result_status = "failed"
            self.set_nav_active(False)
            return
        except Exception as exc:  # noqa: BLE001 - keep bridge alive on CLI errors
            self.get_logger().warn(f"nav CLI goal failed: {exc}")
            self._last_result_status = "failed"
            self.set_nav_active(False)
            return
        lowered = output.lower()
        if proc.returncode == 0 and "succeeded" in lowered:
            self._last_result_status = "succeeded"
        elif "canceled" in lowered:
            self._last_result_status = "canceled"
        elif "aborted" in lowered:
            self._last_result_status = "aborted"
        elif proc.returncode == 0:
            self._last_result_status = "succeeded"
        else:
            self.get_logger().warn(f"nav CLI goal exited {proc.returncode}: {output.strip()}")
            self._last_result_status = "failed"
        self.set_nav_active(False)

    def _publish_nav_goal(self, x: float, y: float, theta: float) -> bool:
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self._frames.map
        pose.header.stamp = self._goal_pose_stamp()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation = _quat_msg(theta)
        goal.pose = pose

        self._last_result_status = "active"
        self.set_nav_active(True)
        send_future = self._nav_action.send_goal_async(
            goal, feedback_callback=self._on_nav_feedback
        )
        send_future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001 - keep executor alive on action errors
            self.get_logger().warn(f"nav goal send failed: {exc}")
            self._goal_handle = None
            self._last_result_status = "failed"
            self.set_nav_active(False)
            return
        self._goal_handle = goal_handle
        if goal_handle is None or not goal_handle.accepted:
            self._last_result_status = "rejected"
            self.set_nav_active(False)
            return
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_feedback(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        self._last_feedback = {
            "distance_remaining": float(getattr(fb, "distance_remaining", 0.0)),
            "navigation_time_sec": float(
                getattr(fb, "navigation_time", _ZeroDuration()).sec
            ),
            "number_of_recoveries": int(getattr(fb, "number_of_recoveries", 0)),
        }

    def _on_nav_result(self, future) -> None:
        try:
            result = future.result()
            status = result.status
        except Exception as exc:  # noqa: BLE001 - keep executor alive on action errors
            self.get_logger().warn(f"nav goal result failed: {exc}")
            self._last_result_status = "failed"
            self.set_nav_active(False)
            return
        # action_msgs/GoalStatus: 4 = SUCCEEDED, 5 = CANCELED, 6 = ABORTED
        self._last_result_status = {4: "succeeded", 5: "canceled", 6: "aborted"}.get(
            status, "unknown"
        )
        self.set_nav_active(False)

    def cancel_nav(self) -> None:
        with self._nav_goal_lock:
            self._pending_nav_goal = None
        if self._nav_cli_proc is not None and self._nav_cli_proc.poll() is None:
            self._nav_cli_proc.terminate()
            self._nav_cli_proc = None
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self.set_nav_active(False)
        self._last_result_status = "canceled"

    def nav_status(self) -> Dict:
        return {
            "state": self._last_result_status or "idle",
            "active": self._nav_active,
            **self._last_feedback,
        }

    def set_initial_odom(self, pose: conv.Pose2D) -> None:
        """Reset wheel odometry (``odom -> base_link``). Not a map-frame pose."""
        self._odom = pose

    def set_initial_pose(
        self,
        pose: conv.Pose2D,
        *,
        position_variance_m2: float = 0.25,
        yaw_variance_rad2: float = 0.06853891945200942,
    ) -> None:
        """Seed slam_toolbox with a map-frame pose via ``/initialpose``."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frames.map
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        msg.pose.pose.orientation = _quat_msg(pose.theta)
        # AMCL/slam_toolbox-compatible diagonal covariance (x, y, yaw).
        msg.pose.covariance[0] = position_variance_m2
        msg.pose.covariance[7] = position_variance_m2
        msg.pose.covariance[35] = yaw_variance_rad2
        self._initialpose_pub.publish(msg)
        self._last_pose_in_map = conv.Pose2D(pose.x, pose.y, pose.theta)

    # -- map + pose queries --------------------------------------------------
    def _on_map(self, msg: OccupancyGrid) -> None:
        if not self._map_updates_enabled:
            return
        width = msg.info.width
        height = msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(height, width)
        self._latest_map = {
            "grid": grid,
            "resolution": msg.info.resolution,
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
            "generation": self._map_generation,
        }

    def map_generation(self) -> int:
        return self._map_generation

    def flush_map_subscription(self) -> int:
        """Drop latched /map history and ignore grids from before this call."""
        sub_kwargs = (
            {"callback_group": self._cb_misc}
            if self._cb_misc is not None
            else {}
        )
        if self._map_sub is not None:
            self.destroy_subscription(self._map_sub)
        self._map_generation += 1
        self._latest_map = None
        self._map_sub = self.create_subscription(
            OccupancyGrid, "map", self._on_map, _LATCHED_QOS, **sub_kwargs
        )
        return self._map_generation

    def get_map(self) -> Optional[Dict]:
        return self._latest_map

    def clear_map(self) -> None:
        """Drop cached occupancy grid so clients stop showing a deleted map."""
        self._latest_map = None
        self._last_pose_in_map = None

    def set_map_updates_enabled(self, enabled: bool) -> None:
        """Ignore or accept incoming ``/map`` messages (used during SLAM reset)."""
        self._map_updates_enabled = enabled
        if not enabled:
            self._latest_map = None

    def _lookup_pose_in_map(self) -> Optional[conv.Pose2D]:
        """Raw map->base_link TF lookup; None when unavailable (no cache fallback)."""
        try:
            lookup_kwargs: Dict = {}
            try:
                from rclpy.duration import Duration

                lookup_kwargs["timeout"] = Duration(seconds=0.2)
            except ImportError:
                pass
            tf = self._tf_buffer.lookup_transform(
                self._frames.map,
                self._frames.base_link,
                rclpy.time.Time(),
                **lookup_kwargs,
            )
        except Exception:  # noqa: BLE001 - transform may not be available yet
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return conv.Pose2D(t.x, t.y, conv.quaternion_to_yaw(q.x, q.y, q.z, q.w))

    def get_pose_in_map(self) -> Optional[conv.Pose2D]:
        pose = self._lookup_pose_in_map()
        if pose is None:
            return self._last_pose_in_map
        self._last_pose_in_map = pose
        return pose


class _ZeroDuration:
    sec = 0

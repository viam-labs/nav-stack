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
from collections import deque
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

from ..config import (
    IMU_ODOM_ACCEL_ONLY,
    IMU_ODOM_COAST,
    IMU_ODOM_NONE,
    LIDAR_SCAN_POINT_CLOUD,
    NavConfig,
    SlamConfig,
)
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
        # Always-integrated copy used only by map_when_still hop / yaw-step gating
        # so we can freeze the published TF prior without breaking the still-gate.
        self._gate_odom = conv.Pose2D(0.0, 0.0, 0.0)
        # Recent (stamp_ns, pose) samples so scans stamped in the past (lidar
        # read latency) can be paired with the pose the robot actually had then.
        self._odom_history: deque = deque(maxlen=150)
        self._last_twist = (0.0, 0.0, 0.0)
        self._last_odom_time = time.monotonic()
        self._last_odom_pub_wall = time.monotonic()
        self._odom_integrate_warned = False
        self._imu_vx = 0.0
        self._imu_vy = 0.0
        self._imu_still_ticks = 0
        # True when the latest movement-sensor sample had explicit linear
        # velocity (wheel odom / mir-base). Lidar odometry is skipped then.
        self._has_wheel_twist = False
        self._last_imu_ax: Optional[float] = None
        self._lidar_odom_status: Dict = {}
        self._prev_scan_for_odom: Optional[conv.LaserScan2D] = None
        self._prev_scan_odom_theta = 0.0
        self._prev_scan_odom_wall = time.monotonic()
        self._scan_accumulation_s = float(slam_cfg.scan_accumulation_s)
        self._imu_odom_mode = (
            IMU_ODOM_NONE
            if slam_cfg.heading_only_odom
            else str(slam_cfg.imu_odom_mode)
        )
        self._heading_only_odom = self._imu_odom_mode == IMU_ODOM_NONE
        self._lidar_odom_enabled = bool(slam_cfg.lidar_odom_enabled)
        self._lidar_odom_range_flow_only = bool(slam_cfg.lidar_odom_range_flow_only)
        self._map_when_still = bool(getattr(slam_cfg, "map_when_still", False))
        self._map_when_still_dwell_s = float(
            getattr(slam_cfg, "map_when_still_dwell_s", 1.0)
        )
        self._map_when_still_lin = float(
            getattr(slam_cfg, "map_when_still_linear_speed_m_s", 0.02)
        )
        self._map_when_still_yaw = float(
            getattr(slam_cfg, "map_when_still_yaw_rate_rad_s", 0.04)
        )
        self._map_when_still_yaw_step = math.radians(
            float(getattr(slam_cfg, "map_when_still_yaw_step_deg", 0.0))
        )
        self._map_when_still_max_drift_m = float(
            getattr(slam_cfg, "map_when_still_max_drift_m", 0.03)
        )
        self._map_when_still_max_drift_deg = float(
            getattr(slam_cfg, "map_when_still_max_drift_deg", 1.5)
        )
        self._wall_yaw_correction = bool(
            getattr(slam_cfg, "wall_yaw_correction", False)
        )
        self._wall_yaw_min_length_m = float(
            getattr(slam_cfg, "wall_yaw_min_length_m", 2.0)
        )
        self._wall_yaw_max_step_deg = float(
            getattr(slam_cfg, "wall_yaw_max_step_deg", 2.0)
        )
        self._wall_yaw_blend = float(getattr(slam_cfg, "wall_yaw_blend", 0.5))
        self._wall_yaw_status: Dict = {}
        self._still_since: Optional[float] = None
        self._dwell_pose0: Optional[tuple] = None
        self._scan_published_this_stop = False
        self._last_still_scan_yaw: Optional[float] = None
        self._last_still_scan_pose: Optional[tuple] = None
        self._last_still_scan_wall: Optional[float] = None
        self._map_when_still_status = "disabled"
        self._last_odom_ok_wall = time.monotonic()
        self._odom_fail_streak = 0
        self._last_odom_error: Optional[str] = None
        # Continuous Livox mapping: publish /scan in laser_i with mount TF.
        # Stop-and-go: publish in base_link so lidar mount.theta affects walls
        # vs the App arrow in one transform (laser↔mount round-trip made
        # mount.theta "do nothing" for arrow-vs-wall alignment diagnostics).
        self._point_cloud_lidars = any(
            lidar.scan_source == LIDAR_SCAN_POINT_CLOUD for lidar in slam_cfg.lidars
        )
        self._use_lidar_frame_scans = (
            self._point_cloud_lidars and not self._map_when_still
        )
        self._pc_accum: Dict[int, deque] = {}
        # IMU forward-accel bias (tilt / vibration); learned whenever accel is
        # flat, since real ax ~ 0 both when parked and cruising at constant speed.
        self._imu_ax_bias = 0.0
        self._imu_ax_window: deque = deque(maxlen=12)
        # Recent range-flow velocity samples (wall_time, m/s); the median over
        # ~1s rejects the single-frame noise of non-repetitive Livox scans.
        self._flow_history: deque = deque()
        self._last_flow_wall = 0.0
        self._last_flow_vx = 0.0
        if self._heading_only_odom:
            self.get_logger().info(
                "heading-only odom (imu_odom_mode=none): IMU yaw only in /odom"
            )
        elif self._imu_odom_mode == IMU_ODOM_ACCEL_ONLY:
            self.get_logger().info(
                "accel-only IMU translation odom for non-repetitive lidar mapping"
            )
        if self._scan_accumulation_s > 0.0:
            self.get_logger().info(
                f"point-cloud scan accumulation window {self._scan_accumulation_s:.2f}s"
            )
        if self._map_when_still:
            step_deg = math.degrees(self._map_when_still_yaw_step)
            step_note = (
                "yaw-step scans disabled"
                if self._map_when_still_yaw_step <= 0
                else f"extra scan every {step_deg:.0f}° while pivoting"
            )
            self.get_logger().info(
                "map_when_still: /scan in base_link after "
                f"{self._map_when_still_dwell_s:.2f}s stopped (once per stop; {step_note})"
            )
        self._empty_scan_warned = False
        self._stale_scan_warned = False
        # Latest merged base_link-frame scan, cached for reactive obstacle
        # avoidance in simple (non-Nav2) go_to_* motion.
        self._latest_scan: Optional[conv.LaserScan2D] = None
        self._latest_scan_wall = 0.0

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
        self._last_map_wall = 0.0
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

    def _integrate_imu_body_velocity(
        self, sample: conv.OdomReading, dt: float
    ) -> tuple[float, float]:
        """Prefer explicit twist; otherwise dead-reckon body velocity from IMU accel.

        At constant speed horizontal accel is near zero, so we *coast* (keep
        velocity) instead of zeroing it. Only apply a soft ZUPT when nearly
        stopped for several ticks — otherwise mapping freezes while driving.

        When accel clearly opposes the current velocity (hard brake / reverse),
        decay velocity quickly so back-and-forth drives don't imprint the map.
        """
        if self._imu_odom_mode == IMU_ODOM_NONE and not self._has_wheel_twist:
            self._imu_vx = 0.0
            self._imu_vy = 0.0
            return 0.0, 0.0
        if abs(sample.vx) > 1e-6 or abs(sample.vy) > 1e-6:
            self._imu_vx = float(sample.vx)
            self._imu_vy = float(sample.vy)
            self._imu_still_ticks = 0
            self._has_wheel_twist = True
            return self._imu_vx, self._imu_vy
        self._has_wheel_twist = False
        if sample.ax is None or sample.ay is None:
            return float(sample.vx), float(sample.vy)

        ax = float(sample.ax)
        ay = float(sample.ay)
        still = abs(ax) < 0.35 and abs(ay) < 0.35 and abs(sample.vtheta) < 0.08

        if self._imu_odom_mode == IMU_ODOM_ACCEL_ONLY:
            self._imu_ax_window.append(ax)
            window = np.asarray(self._imu_ax_window, dtype=float)
            # Flat accel (low variance) means no real forward accel — true both
            # parked and cruising. Instantaneous thresholds don't work here:
            # this IMU shows a large, drifting bias (|ax| up to ~0.7 at rest).
            accel_flat = (
                window.size >= 6
                and float(np.std(window)) < 0.12
                and abs(sample.vtheta) < 0.05
            )
            if accel_flat:
                self._imu_still_ticks += 1
                self._imu_ax_bias += 0.05 * (ax - self._imu_ax_bias)
                flow_age = time.monotonic() - self._last_flow_wall
                if flow_age < 1.0 and abs(self._last_flow_vx) > 0.1:
                    # Lidar range flow says we're moving (constant speed looks
                    # "flat" to the IMU); keep coasting on the flow velocity.
                    pass
                elif flow_age < 1.0:
                    # Flow matched and reported ~zero motion: confident ZUPT.
                    self._imu_vx = 0.0
                else:
                    # Flow has no info (nothing matchable ahead, e.g. driving
                    # down a bare corridor). Decay gently instead of freezing
                    # mapping mid-drive.
                    self._imu_vx *= 0.9
                    if abs(self._imu_vx) < 0.05:
                        self._imu_vx = 0.0
            else:
                self._imu_still_ticks = 0
                ax_c = ax - self._imu_ax_bias
                if self._imu_vx * ax_c < -0.35 and abs(ax_c) > 0.4:
                    self._imu_vx *= 0.5
                    if abs(self._imu_vx) < 0.06:
                        self._imu_vx = 0.0
                elif abs(ax_c) > 0.25:
                    self._imu_vx += ax_c * dt
                else:
                    self._imu_vx *= 0.98
            self._imu_vy = 0.0
            max_v = 1.0
            if abs(self._imu_vx) > max_v:
                self._imu_vx = math.copysign(max_v, self._imu_vx)
            return self._imu_vx, self._imu_vy

        if still:
            self._imu_still_ticks += 1
        else:
            self._imu_still_ticks = 0
            # Forward accel only for non-holonomic / skid-steer carts. Lateral
            # accel from a tilted IMU or residual gravity bias curves the map.
            # Require a clear opposing accel so noise doesn't kill coasting.
            opposing = self._imu_vx * ax < -0.4 and abs(ax) > 0.5
            if opposing:
                self._imu_vx *= 0.5
                if abs(self._imu_vx) < 0.08:
                    self._imu_vx = 0.0
            # At speed, small accel noise skews scale; coast unless braking hard
            # or clearly accelerating from rest.
            elif abs(self._imu_vx) > 0.2 and abs(ax) < 0.6:
                pass
            else:
                self._imu_vx += ax * dt

        # Soft ZUPT only after ~0.5s of near-zero accel (not every coasting tick).
        # Do NOT decay just because lidar scan-matching failed — that freezes
        # mapping whenever Livox matches are sparse (the previous regression).
        if self._imu_still_ticks >= 8:
            self._imu_vx *= 0.85
            if abs(self._imu_vx) < 0.05:
                self._imu_vx = 0.0

        # Skid-steer / differential: no body-frame lateral velocity.
        self._imu_vy = 0.0
        max_v = 3.0
        if abs(self._imu_vx) > max_v:
            self._imu_vx = math.copysign(max_v, self._imu_vx)
        return self._imu_vx, self._imu_vy

    def _apply_lidar_odometry(self, scan: conv.LaserScan2D) -> None:
        """Update body velocity from scan-to-scan matching (IMU+lidar path).

        Only sets ``_imu_vx``; pose is integrated by the odom timer so we never
        double-apply the same motion. Skipped when wheel odometry already
        provides linear velocity (MiR / wheeled-odometry carts unchanged).
        """
        if not self._lidar_odom_enabled:
            self._lidar_odom_status = {
                "accepted": False,
                "reason": "disabled",
                "method": "none",
            }
            return
        now = time.monotonic()
        if self._has_wheel_twist:
            self._prev_scan_for_odom = scan
            self._prev_scan_odom_theta = self._odom.theta
            self._prev_scan_odom_wall = now
            return
        if self._prev_scan_for_odom is None:
            self._prev_scan_for_odom = scan
            self._prev_scan_odom_theta = self._odom.theta
            self._prev_scan_odom_wall = now
            return

        prev_theta = self._prev_scan_odom_theta
        dt = max(now - self._prev_scan_odom_wall, 1e-3)
        dtheta = conv.normalize_angle(self._odom.theta - prev_theta)
        # Ignore large heading jumps between scans — those corrupt translation.
        if abs(dtheta) > 0.35:  # ~20 deg
            self._prev_scan_for_odom = scan
            self._prev_scan_odom_theta = self._odom.theta
            self._prev_scan_odom_wall = now
            return
        if self._lidar_odom_range_flow_only:
            motion = conv.estimate_forward_range_flow(
                self._prev_scan_for_odom,
                scan,
                dtheta=dtheta,
                min_beams=20,
                max_median_deviation_m=0.15,
                min_abs_dx=0.0,
            )
            self._prev_scan_for_odom = scan
            self._prev_scan_odom_theta = self._odom.theta
            self._prev_scan_odom_wall = now
            if motion is None:
                self._lidar_odom_status = {
                    "accepted": False,
                    "reason": "no_match",
                    "method": "range_flow",
                    "age_s": round(dt, 3),
                }
                return
            # Median over ~1s of samples rejects the single-frame noise of
            # non-repetitive Livox scans; a near-zero median is a valid
            # lidar-confirmed "not moving" (natural ZUPT while parked).
            meas_vx = max(-1.5, min(1.5, motion.dx / dt))
            self._flow_history.append((now, meas_vx))
            while self._flow_history and now - self._flow_history[0][0] > 1.0:
                self._flow_history.popleft()
            flow_vx = float(np.median([v for _, v in self._flow_history]))
            alpha = 0.35
            self._imu_vx = (1.0 - alpha) * self._imu_vx + alpha * flow_vx
            if abs(self._imu_vx) < 0.04:
                self._imu_vx = 0.0
            self._imu_vy = 0.0
            self._last_flow_wall = now
            self._last_flow_vx = flow_vx
            self._lidar_odom_status = {
                "accepted": True,
                "method": "range_flow",
                "meas_vx": round(meas_vx, 4),
                "flow_median_vx": round(flow_vx, 4),
                "imu_vx": round(self._imu_vx, 4),
                "residual_m": round(motion.residual, 4),
                "history_n": len(self._flow_history),
                "dt_s": round(dt, 3),
                "age_s": round(dt, 3),
            }
            return

        motion, debug = conv.estimate_scan_motion_with_debug(
            self._prev_scan_for_odom,
            scan,
            dtheta=dtheta,
            allow_lateral=False,
        )
        self._prev_scan_for_odom = scan
        self._prev_scan_odom_theta = self._odom.theta
        self._prev_scan_odom_wall = now
        if motion is None:
            self._lidar_odom_status = {
                "accepted": False,
                "reason": debug.get("reject", "no_match"),
                "method": debug.get("method", "none"),
                "best_dx": round(float(debug.get("best_dx", 0.0)), 4),
                "best_residual_m": round(float(debug.get("best_residual_m", 0.0)), 4),
                "best_fraction": round(float(debug.get("best_fraction", 0.0)), 3),
                "zero_residual_m": round(float(debug.get("zero_residual_m", 0.0)), 4),
                "age_s": round(dt, 3),
            }
            return
        dx, _dy, _ = motion
        dist = abs(dx)
        # Cap to ~1.5 m/s at the scan rate; reject larger jumps as bad matches.
        max_step = max(0.15, 1.5 * dt)
        if dist < 0.01 or dist > max_step:
            self._lidar_odom_status = {
                "accepted": False,
                "reason": "distance_gate",
                "dx": round(dx, 4),
                "dt_s": round(dt, 3),
                "age_s": round(dt, 3),
            }
            return

        # Reject sign flips unless acceleration confirms the new direction —
        # sparse Livox scans often alias forward/backward motion.
        meas_vx_raw = dx / dt
        hint_ax = self._last_imu_ax
        reversed_motion = (
            abs(self._imu_vx) > 0.15 and self._imu_vx * meas_vx_raw < 0.0
        )
        accel_confirms = (
            hint_ax is not None
            and abs(hint_ax) > 0.5
            and hint_ax * meas_vx_raw > 0.0
        )
        if reversed_motion and not accel_confirms:
            self._lidar_odom_status = {
                "accepted": False,
                "reason": "sign_conflict",
                "dx": round(dx, 4),
                "imu_vx": round(self._imu_vx, 4),
                "residual_m": round(motion.residual, 4),
                "improvement_m": round(motion.improvement, 4),
                "age_s": round(dt, 3),
            }
            return

        # Blend into forward body velocity; undershoot slightly so slam_toolbox
        # scan matching can correct over-aggressive odom priors.
        meas_vx = 0.85 * meas_vx_raw
        if motion.residual < 0.06 and motion.improvement > 0.06:
            alpha = 0.45
        elif motion.residual < 0.08 and motion.improvement > 0.04:
            alpha = 0.30
        elif motion.method == "range_flow":
            alpha = 0.25
        else:
            alpha = 0.20
        if reversed_motion and accel_confirms:
            alpha = min(0.55, alpha + 0.15)
        self._imu_vx = (1.0 - alpha) * self._imu_vx + alpha * meas_vx
        self._imu_vy = 0.0
        self._imu_still_ticks = 0
        self._lidar_odom_status = {
            "accepted": True,
            "dx": round(dx, 4),
            "meas_vx": round(meas_vx, 4),
            "imu_vx": round(self._imu_vx, 4),
            "residual_m": round(motion.residual, 4),
            "improvement_m": round(motion.improvement, 4),
            "match_fraction": round(motion.match_fraction, 3),
            "method": motion.method,
            "dt_s": round(dt, 3),
            "age_s": round(dt, 3),
        }

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

    def _accumulate_point_cloud(
        self,
        lidar_idx: int,
        points: np.ndarray,
        pose: Optional[conv.Pose2D] = None,
    ) -> np.ndarray:
        """Time-merge recent point clouds into the current base_link frame."""
        now = time.monotonic()
        buf = self._pc_accum.setdefault(lidar_idx, deque(maxlen=12))
        if pose is None:
            pose = conv.Pose2D(self._odom.x, self._odom.y, self._odom.theta)
        points = np.asarray(points, dtype=float)
        # While not fully settled in stop-and-go mode, never accumulate: Livox
        # frames plus slow spin/creep smear walls into the pause scan (ghosts).
        # Thresholds match the still-gate, not the looser spin-only check below.
        if self._map_when_still and not self._motion_is_still():
            buf.clear()
            buf.append((now, points, pose))
            return points
        # While spinning (continuous mapping path), pose-compensated merging
        # still smears: each Livox frame is itself an ~0.1 s sweep.
        if abs(self._last_twist[2]) > 0.15:
            buf.clear()
            buf.append((now, points, pose))
            return points
        # Stop-and-go: only densify during an active dwell so driving frames
        # never leak into the published pause scan.
        if self._map_when_still and self._still_since is None:
            buf.clear()
            buf.append((now, points, pose))
            return points
        buf.append((now, points, pose))
        cutoff = now - self._scan_accumulation_s
        while buf and buf[0][0] < cutoff:
            buf.popleft()
        frames = [(pts, p) for t, pts, p in buf if t >= cutoff]
        # Full pose compensation: odom now carries translation (accel + range
        # flow), so old frames are re-expressed in the current body frame
        # instead of smearing walls while driving (rotation-only regression).
        return conv.merge_accumulated_point_clouds(frames, pose)

    def _motion_is_still(self) -> bool:
        vx, vy, vtheta = self._last_twist
        return (
            math.hypot(vx, vy) < self._map_when_still_lin
            and abs(vtheta) < self._map_when_still_yaw
        )

    def _gate_pose_tuple(self) -> tuple:
        return (
            float(self._gate_odom.x),
            float(self._gate_odom.y),
            float(self._gate_odom.theta),
        )

    def _dwell_pose_drifted(self) -> bool:
        """True if pose moved during the current still dwell (creep / bias)."""
        if self._dwell_pose0 is None:
            return False
        x, y, yaw = self._gate_pose_tuple()
        x0, y0, yaw0 = self._dwell_pose0
        dist = math.hypot(x - x0, y - y0)
        dyaw = abs(conv.normalize_angle(yaw - yaw0))
        return dist > self._map_when_still_max_drift_m or dyaw > math.radians(
            self._map_when_still_max_drift_deg
        )

    def _reset_still_dwell(self, status: str) -> None:
        self._still_since = None
        self._dwell_pose0 = None
        self._map_when_still_status = status
        for buf in self._pc_accum.values():
            buf.clear()

    def _still_gate_ready(self) -> bool:
        """True when a still-scan may be captured — does not commit publish state.

        Commit happens only after the lidar read finishes and motion is still
        confirmed, so a slow Livox read cannot publish while the cart rolls.
        """
        if not self._map_when_still:
            self._map_when_still_status = "disabled"
            return True
        if not self._point_cloud_lidars:
            self._map_when_still_status = "disabled_non_point_cloud"
            return True

        now = time.monotonic()
        odom_dead_s = now - self._last_odom_ok_wall
        if odom_dead_s > 1.0:
            # Without odom we cannot prove stillness — do not publish.
            self._map_when_still_status = f"odom_unavailable:{odom_dead_s:.0f}s"
            return False

        if not self._motion_is_still():
            vx, vy, vtheta = self._last_twist
            if math.hypot(vx, vy) >= self._map_when_still_lin:
                self._scan_published_this_stop = False
                self._reset_still_dwell("moving")
            else:
                self._scan_published_this_stop = False
                self._reset_still_dwell("pivoting")
            return False

        if self._still_since is None:
            self._still_since = now
            self._dwell_pose0 = self._gate_pose_tuple()

        if self._dwell_pose_drifted():
            self._scan_published_this_stop = False
            self._reset_still_dwell("dwell_drift")
            return False

        dwell = now - self._still_since
        if dwell < self._map_when_still_dwell_s:
            self._map_when_still_status = f"dwelling:{dwell:.2f}s"
            return False

        x, y, yaw = self._gate_pose_tuple()
        hop_m = 0.2
        if self._last_still_scan_pose is None:
            dist, dyaw = float("inf"), float("inf")
        else:
            lx, ly, lt = self._last_still_scan_pose
            dist = math.hypot(x - lx, y - ly)
            dyaw = abs(conv.normalize_angle(yaw - lt))

        if (
            self._scan_published_this_stop
            and dist < hop_m
            and (
                self._map_when_still_yaw_step <= 0
                or dyaw < self._map_when_still_yaw_step
            )
        ):
            self._map_when_still_status = "published_this_stop"
            return False

        if (
            self._map_when_still_yaw_step > 0
            and math.isfinite(dyaw)
            and dyaw >= self._map_when_still_yaw_step
            and dist < hop_m
        ):
            self._map_when_still_status = "ready_yaw_step"
        else:
            self._map_when_still_status = "ready"
        return True

    def _still_gate_commit(self) -> None:
        """Record that a still-scan was published for this stop."""
        now = time.monotonic()
        x, y, yaw = self._gate_pose_tuple()
        self._scan_published_this_stop = True
        self._last_still_scan_yaw = yaw
        self._last_still_scan_pose = (x, y, yaw)
        self._last_still_scan_wall = now
        self._map_when_still_status = "publishing"

    def _should_publish_map_when_still(self) -> bool:
        """Compatibility wrapper: readiness only (commit is separate)."""
        return self._still_gate_ready()

    # -- scans ---------------------------------------------------------------
    def _on_scan_timer(self) -> None:
        # Advance the still/dwell tracker before the lidar read so this tick's
        # frame can join the pause-accumulation window. Commit (and matching)
        # still happens only after the read finishes and motion is rechecked.
        if self._map_when_still and self._point_cloud_lidars:
            self._still_gate_ready()

        # Capture the stamp BEFORE reading: rosbridge lidar reads take up to
        # seconds, and the returned scan reflects the world at capture time.
        # Stamping stale geometry with a post-read now() shifts obstacles away
        # from a moving robot and raytrace-clears their true cells (the robot
        # then drives into obstacles the lidar plainly sees).
        read_start = self.get_clock().now()
        per_lidar_scans = []
        scan_age_s: Optional[float] = None
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
                if lidar_data.age_s is not None:
                    scan_age_s = (
                        lidar_data.age_s
                        if scan_age_s is None
                        else max(scan_age_s, lidar_data.age_s)
                    )
            else:
                base_pts_arr = np.asarray(lidar_data, dtype=float)

            if sensor_scan is None or not conv.scan_has_returns(sensor_scan):
                # Prefer base_link points when present (typical for get_point_cloud
                # lidars); z_min/z_max then act as a height band above the floor.
                if base_pts_arr.size:
                    scan_pts = base_pts_arr
                else:
                    scan_pts = (
                        np.asarray(lidar_data.sensor, dtype=float)
                        if isinstance(lidar_data, conv.LidarPoints)
                        else base_pts_arr
                    )
                if scan_pts.size == 0:
                    continue
                if (
                    self._scan_accumulation_s > 0.0
                    and lidar.scan_source == LIDAR_SCAN_POINT_CLOUD
                ):
                    # Tag the frame with the pose at capture time (read start),
                    # not after the read returned — misaligned poses smear the
                    # accumulated cloud while driving.
                    scan_pts = self._accumulate_point_cloud(
                        i, scan_pts, pose=self._odom_pose_at(read_start.to_msg())
                    )
                if (
                    self._use_lidar_frame_scans
                    and lidar.scan_source == LIDAR_SCAN_POINT_CLOUD
                ):
                    sensor_scan = conv.base_link_cloud_to_lidar_scan(
                        scan_pts,
                        x=lidar.x,
                        y=lidar.y,
                        z=lidar.z,
                        theta=lidar.theta,
                        z_min=lidar.z_min,
                        z_max=lidar.z_max,
                        points_in_base_link=lidar.points_in_base_link,
                        num_bins=self._slam_cfg.scan_bins,
                        range_min=lidar.min_range,
                        range_max=lidar.max_range,
                    )
                else:
                    sensor_scan = conv.pointcloud_to_scan(
                        scan_pts,
                        z_min=lidar.z_min,
                        z_max=lidar.z_max,
                        num_bins=self._slam_cfg.scan_bins,
                        range_min=lidar.min_range,
                        range_max=lidar.max_range,
                    )
                if not conv.scan_has_returns(sensor_scan):
                    continue

            per_lidar_scans.append((i, sensor_scan))

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

        # Safety net: mir-base's SLAM path already refuses to serve stale scans,
        # but if a genuinely old scan slips through, don't hand SLAM/Nav2 a
        # misregistered scan — skip this cycle instead.
        if scan_age_s is not None:
            max_age = float(getattr(self._slam_cfg, "scan_max_age_s", 2.0))
            if max_age > 0.0 and scan_age_s > max_age:
                if not self._stale_scan_warned:
                    self.get_logger().warn(
                        f"lidar scan cache age {scan_age_s:.2f}s exceeds "
                        f"scan_max_age_s={max_age:.2f}s; skipping /scan publish "
                        "(check mir-base scan_cache_max_age_s / rosbridge health)"
                    )
                    self._stale_scan_warned = True
                return
            self._stale_scan_warned = False

        if not self._still_gate_ready():
            # Accumulate during dwell for a dense pause scan; publish is gated.
            return

        if self._map_when_still:
            # Lidar read can take hundreds of ms — refuse to match if we moved.
            if not self._motion_is_still() or self._dwell_pose_drifted():
                self._scan_published_this_stop = False
                self._reset_still_dwell("moved_during_capture")
                return
            self._still_gate_commit()
            # Hard ZUPT at each accepted still-scan so IMU coast doesn't invent
            # translation before the next hop.
            self._imu_vx = 0.0
            self._imu_vy = 0.0
            self._imu_still_ticks = 0

        ref = self._slam_cfg.lidars[0]
        merged_frame = self._frames.base_link
        if self._use_lidar_frame_scans and len(per_lidar_scans) == 1:
            merged = per_lidar_scans[0][1]
            merged_frame = f"laser_{per_lidar_scans[0][0]}"
        else:
            merged = conv.merge_scans(
                [scan for _, scan in per_lidar_scans],
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

        # Soft-correct gyro yaw from a long side wall before publishing TF so
        # slam_toolbox's match prior (and this scan stamp) see the debiased yaw.
        if self._wall_yaw_correction:
            self._apply_wall_yaw_correction(merged)

        # Stamp at capture time (read_start - age_s) so obstacles/scan-match
        # register where the robot actually was when the scan was captured, not
        # where it is after the (cached) read returns.
        stamp = self._bounded_scan_stamp(read_start, age_s=scan_age_s or 0.0)
        # Also publish odom->base_link at exactly this stamp: freshly started
        # slam/Nav2 nodes have empty TF buffers and drop past-stamped scans
        # ("earlier than all the data in the transform cache") — a sample at
        # the scan time makes the lookup succeed by construction.
        self._publish_scan_time_tf(stamp)
        for i, scan in per_lidar_scans:
            self._scan_pubs[i].publish(self._to_ros_scan(scan, f"laser_{i}", stamp))

        self._latest_scan = merged
        self._latest_scan_wall = time.monotonic()
        self._apply_lidar_odometry(merged)
        self._merged_scan_pub.publish(
            self._to_ros_scan(merged, merged_frame, stamp)
        )

    def _apply_wall_yaw_correction(self, scan: conv.LaserScan2D) -> None:
        """Debias ``_odom.theta`` from a dominant side wall in ``scan``."""
        obs = conv.extract_dominant_wall(
            scan, min_length_m=self._wall_yaw_min_length_m
        )
        if obs is None:
            self._wall_yaw_status = {"accepted": False, "reason": "no_wall"}
            return
        delta = conv.wall_yaw_correction_delta(
            obs.wall_yaw_body,
            max_step_rad=math.radians(self._wall_yaw_max_step_deg),
            blend=self._wall_yaw_blend,
        )
        if abs(delta) < 1e-6:
            self._wall_yaw_status = {
                "accepted": True,
                "applied": False,
                "side": obs.side,
                "wall_yaw_deg": round(math.degrees(obs.wall_yaw_body), 2),
                "wall_yaw_correction_deg": 0.0,
                "wall_yaw_inliers": obs.inliers,
                "wall_length_m": round(obs.length_m, 2),
            }
            return
        new_theta = conv.normalize_angle(self._odom.theta + delta)
        self._odom = conv.Pose2D(self._odom.x, self._odom.y, new_theta)
        self._gate_odom = conv.Pose2D(
            self._gate_odom.x, self._gate_odom.y, new_theta
        )
        self._wall_yaw_status = {
            "accepted": True,
            "applied": True,
            "side": obs.side,
            "wall_yaw_deg": round(math.degrees(obs.wall_yaw_body), 2),
            "wall_yaw_correction_deg": round(math.degrees(delta), 2),
            "wall_yaw_inliers": obs.inliers,
            "wall_length_m": round(obs.length_m, 2),
        }

    def _odom_pose_at(self, stamp) -> conv.Pose2D:
        """Nearest recorded odom pose to ``stamp`` (falls back to current)."""
        try:
            target_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        except (TypeError, AttributeError):
            return self._odom
        best_pose = self._odom
        best_delta = None
        for sample_ns, pose in self._odom_history:
            delta = abs(sample_ns - target_ns)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_pose = pose
        return best_pose

    def _publish_scan_time_tf(self, stamp) -> None:
        pose = self._odom_pose_at(stamp)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._frames.odom
        t.child_frame_id = self._frames.base_link
        t.transform.translation.x = pose.x
        t.transform.translation.y = pose.y
        t.transform.rotation = _quat_msg(pose.theta)
        self._tf_broadcaster.sendTransform(t)

    def _bounded_scan_stamp(self, read_start, age_s: float = 0.0):
        """Capture-time stamp (read_start - age_s), clamped to stay in TF buffers.

        ``age_s`` is the cache age mir-base reports for the scan; subtracting it
        places the stamp at when the scan was actually captured. TF history is
        ~10s in Nav2 costmaps; a scan stamped older than that would be dropped
        entirely, which is worse than a bounded position error, so we clamp.
        """
        max_lag_ns = 8_000_000_000
        try:
            capture_ns = read_start.nanoseconds - int(max(0.0, age_s) * 1e9)
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - capture_ns > max_lag_ns:
                return rclpy.time.Time(nanoseconds=now_ns - max_lag_ns).to_msg()
            return rclpy.time.Time(nanoseconds=capture_ns).to_msg()
        except Exception:  # noqa: BLE001 - clamping is best-effort
            return read_start.to_msg()

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
            self._odom_fail_streak += 1
            self._last_odom_error = repr(exc)
            if self._odom_fail_streak in (1, 5, 30) or self._odom_fail_streak % 50 == 0:
                self.get_logger().warn(
                    f"odometry read failed ({self._odom_fail_streak}x): {exc!r}"
                )
            return
        self._odom_fail_streak = 0
        self._last_odom_error = None
        self._last_odom_ok_wall = time.monotonic()
        now = time.monotonic()
        raw_dt = now - self._last_odom_time
        self._last_odom_time = now
        if sample.pose is not None:
            self._odom = sample.pose
            self._gate_odom = sample.pose
            self._imu_vx = 0.0
            self._imu_vy = 0.0
            self._has_wheel_twist = True
            stamp = self.get_clock().now().to_msg()
            self._publish_odom_snapshot(
                stamp, float(sample.vx), float(sample.vy), sample.vtheta
            )
            return

        dt = self._bounded_odom_dt(raw_dt)
        self._last_imu_ax = sample.ax
        if dt > 0:
            vx, vy = self._integrate_imu_body_velocity(sample, dt)
        else:
            vx, vy = self._imu_vx, self._imu_vy
        vtheta = sample.vtheta

        # Absolute AHRS/magnetometer yaw from a bare IMU is often biased vs the
        # lidar frame (looks like a fixed ~45° arrow offset) and jumps indoors.
        # For IMU dead-reckoning, integrate gyro Z only — never snap to orientation.
        imu_dead_reckon = (
            sample.pose is None
            and not self._has_wheel_twist
            and (sample.ax is not None or sample.ay is not None)
        )
        if sample.heading_rad is not None and not imu_dead_reckon:
            if dt <= 0:
                self._odom = conv.Pose2D(
                    self._odom.x, self._odom.y, sample.heading_rad
                )
            else:
                # Mid-point heading reduces banana curves when yaw changes
                # during the integration step (common with IMU heading).
                th = conv.normalize_angle(
                    self._odom.theta
                    + 0.5 * conv.normalize_angle(sample.heading_rad - self._odom.theta)
                )
                dx = (vx * math.cos(th) - vy * math.sin(th)) * dt
                dy = (vx * math.sin(th) + vy * math.cos(th)) * dt
                self._odom = conv.Pose2D(
                    self._odom.x + dx,
                    self._odom.y + dy,
                    sample.heading_rad,
                )
            self._gate_odom = self._odom
        else:
            has_imu_motion = sample.ax is not None and sample.ay is not None
            if (
                not self._odom_integrate_warned
                and not has_imu_motion
                and abs(sample.vx) < 1e-6
                and abs(sample.vy) < 1e-6
                and abs(sample.vtheta) < 1e-6
            ):
                self.get_logger().warn(
                    "movement sensor did not provide odom pose or heading in "
                    "get_readings(); dead-reckoning from velocity — orientation "
                    "drift likely"
                )
                self._odom_integrate_warned = True
            if dt > 0:
                # gate_odom: hop / yaw-step detection (may integrate IMU XY).
                gate_th = self._gate_odom.theta
                gate_vx, gate_vy = vx, vy
                if self._map_when_still and abs(vtheta) >= self._map_when_still_yaw:
                    gate_vx, gate_vy = 0.0, 0.0
                gdx = (gate_vx * math.cos(gate_th) - gate_vy * math.sin(gate_th)) * dt
                gdy = (gate_vx * math.sin(gate_th) + gate_vy * math.cos(gate_th)) * dt
                self._gate_odom = conv.Pose2D(
                    self._gate_odom.x + gdx,
                    self._gate_odom.y + gdy,
                    gate_th + vtheta * dt,
                )

                th = self._odom.theta
                pub_vx, pub_vy = vx, vy
                # slam_toolbox uses odom→base TF as the scan-match prior.
                # Freeze XY only while pivoting (accel noise); allow translation
                # while hopping so the matcher isn’t stuck at the origin.
                if self._map_when_still and abs(vtheta) >= self._map_when_still_yaw:
                    pub_vx, pub_vy = 0.0, 0.0
                    self._imu_vx = 0.0
                    self._imu_vy = 0.0
                dx = (pub_vx * math.cos(th) - pub_vy * math.sin(th)) * dt
                dy = (pub_vx * math.sin(th) + pub_vy * math.cos(th)) * dt
                self._odom = conv.Pose2D(
                    self._odom.x + dx, self._odom.y + dy, th + vtheta * dt
                )
                # Keep gate in sync for hop/yaw-step detection when not using
                # a separate IMU-dead-reckon path… already updated above.
            elif self._map_when_still and abs(vtheta) >= self._map_when_still_yaw:
                vx, vy = 0.0, 0.0
                self._imu_vx = 0.0
                self._imu_vy = 0.0

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
        try:
            self._odom_history.append(
                (int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec), pose)
            )
        except (TypeError, AttributeError):  # non-Time stamps in tests
            pass
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

    def apply_map_pose_correction(self, map_pose: conv.Pose2D) -> Dict:
        """Shift the published odom pose so the slam prior lands on ``map_pose``.

        Mapping-mode revisit correction: slam_toolbox has no /initialpose while
        mapping — its scan-match prior is map->odom ∘ (our odom->base TF). By
        moving odom->base so that composition equals the revisit match, the next
        pause scan is matched (and graph-linked) at the true location instead of
        extending a drifted duplicate corridor.
        """
        map_to_odom = self._lookup_map_to_odom()
        if map_to_odom is None:
            return {"applied": False, "reason": "map_to_odom_unavailable"}
        old = self._odom
        new_odom = conv.map_pose_to_odom_pose(map_pose, map_to_odom)
        self._odom = new_odom
        self._gate_odom = new_odom
        # ZUPT so IMU coast doesn't drag the corrected pose before the next hop.
        self._imu_vx = 0.0
        self._imu_vy = 0.0
        # Publish TF at the corrected pose immediately so the next scan's
        # stamp lookup sees it (rather than waiting one odom tick).
        self._publish_odom_snapshot(self.get_clock().now().to_msg(), 0.0, 0.0, 0.0)
        return {
            "applied": True,
            "old_odom": self._pose_dict(old),
            "new_odom": self._pose_dict(new_odom),
            "shift_m": round(math.hypot(new_odom.x - old.x, new_odom.y - old.y), 3),
            "shift_deg": round(
                abs(math.degrees(conv.normalize_angle(new_odom.theta - old.theta))),
                2,
            ),
        }

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
        self._last_map_wall = time.monotonic()

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
        self._last_map_wall = 0.0
        self._map_sub = self.create_subscription(
            OccupancyGrid, "map", self._on_map, _LATCHED_QOS, **sub_kwargs
        )
        return self._map_generation

    def get_map(self) -> Optional[Dict]:
        return self._latest_map

    def clear_map(self) -> None:
        """Drop cached occupancy grid so clients stop showing a deleted map."""
        self._latest_map = None
        self._last_map_wall = 0.0
        self._last_pose_in_map = None

    def set_map_updates_enabled(self, enabled: bool) -> None:
        """Ignore or accept incoming ``/map`` messages (used during SLAM reset)."""
        self._map_updates_enabled = enabled
        if not enabled:
            self._latest_map = None
            self._last_map_wall = 0.0

    @staticmethod
    def _pose_dict(pose: conv.Pose2D) -> Dict[str, float]:
        return {"x": pose.x, "y": pose.y, "theta": pose.theta}

    def slam_bridge_status(self) -> Dict:
        """Bridge-side liveness signals for SLAM debugging (no sensor I/O)."""
        scan_age_s = None
        scan_valid_returns = 0
        if self._latest_scan is not None:
            scan_age_s = round(time.monotonic() - self._latest_scan_wall, 2)
            scan_valid_returns = sum(
                1
                for r in self._latest_scan.ranges
                if math.isfinite(r)
                and r >= self._latest_scan.range_min
                and (
                    not math.isfinite(self._latest_scan.range_max)
                    or r <= self._latest_scan.range_max
                )
            )
        scan_fresh_limit = max(
            3.0 / max(float(self._slam_cfg.scan_rate_hz), 1.0), 2.0
        )

        map_age_s = None
        map_size = None
        map_cells = None
        if self._latest_map is not None:
            if self._last_map_wall > 0.0:
                map_age_s = round(time.monotonic() - self._last_map_wall, 2)
            grid = self._latest_map["grid"]
            map_size = {
                "width": int(grid.shape[1]),
                "height": int(grid.shape[0]),
                "resolution": float(self._latest_map["resolution"]),
            }
            map_cells = {
                "unknown": int(np.sum(grid < 0)),
                "free": int(np.sum(grid == 0)),
                "occupied": int(np.sum(grid > 0)),
            }

        tf_pose = self._lookup_pose_in_map()
        map_to_odom = self._lookup_map_to_odom()
        cached_pose = self._last_pose_in_map
        pose = tf_pose if tf_pose is not None else cached_pose
        vx, vy, vtheta = self._last_twist

        return {
            "scan_age_s": scan_age_s,
            "scan_valid_returns": scan_valid_returns,
            "scan_publishing": scan_age_s is not None and scan_age_s < scan_fresh_limit,
            "odom_tf_age_s": self.odom_tf_age_s(),
            "odom_pose": self._pose_dict(self._odom),
            "gate_odom_pose": self._pose_dict(self._gate_odom),
            "odom_velocity": {"vx": vx, "vy": vy, "vtheta": vtheta},
            "odom_integrate_warned": self._odom_integrate_warned,
            "has_wheel_twist": self._has_wheel_twist,
            "imu_odom_vx": round(self._imu_vx, 4),
            "imu_ax_bias": round(self._imu_ax_bias, 4),
            "flow_vx": round(self._last_flow_vx, 4),
            "heading_only_odom": self._heading_only_odom,
            "imu_odom_mode": self._imu_odom_mode,
            "scan_accumulation_s": self._scan_accumulation_s,
            "scan_accum_frames": sum(len(buf) for buf in self._pc_accum.values()),
            "lidar_odom": self._lidar_odom_status or None,
            "use_lidar_frame_scans": self._use_lidar_frame_scans,
            "lidar_odom_range_flow_only": self._lidar_odom_range_flow_only,
            "map_when_still": self._map_when_still,
            "map_when_still_status": self._map_when_still_status,
            "wall_yaw_correction": self._wall_yaw_correction,
            "wall_yaw": self._wall_yaw_status or None,
            "odometry_ok": self._last_odom_error is None
            and (time.monotonic() - self._last_odom_ok_wall) < 1.0,
            "odometry_error": self._last_odom_error,
            "odometry_fail_streak": self._odom_fail_streak,
            "empty_scan_warned": self._empty_scan_warned,
            "stale_scan_warned": self._stale_scan_warned,
            "map_received": self._latest_map is not None,
            "map_generation": self._map_generation,
            "map_age_s": map_age_s,
            "map_size": map_size,
            "map_cells": map_cells,
            "map_tf_available": tf_pose is not None,
            "map_to_odom": (
                self._pose_dict(map_to_odom) if map_to_odom is not None else None
            ),
            "pose_in_map": self._pose_dict(pose) if pose is not None else None,
            "pose_from_cache": tf_pose is None and cached_pose is not None,
        }

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

    def _lookup_map_to_odom(self) -> Optional[conv.Pose2D]:
        """map->odom TF from slam_toolbox (translation here drives pose_in_map)."""
        try:
            lookup_kwargs: Dict = {}
            try:
                from rclpy.duration import Duration

                lookup_kwargs["timeout"] = Duration(seconds=0.2)
            except ImportError:
                pass
            tf = self._tf_buffer.lookup_transform(
                self._frames.map,
                self._frames.odom,
                rclpy.time.Time(),
                **lookup_kwargs,
            )
        except Exception:  # noqa: BLE001
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

    def get_base_scan(self, max_age_s: float = 1.0) -> Optional[conv.LaserScan2D]:
        """Latest merged base_link-frame scan, or None if too stale/absent.

        Forward (+x) is at angle 0. Used by simple go_to_* obstacle avoidance;
        a stale scan is treated as "no data" so we never dodge phantom returns.
        """
        scan = self._latest_scan
        if scan is None:
            return None
        if time.monotonic() - self._latest_scan_wall > max_age_s:
            return None
        return scan


class _ZeroDuration:
    sec = 0

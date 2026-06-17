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
import time
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.action import ActionClient

from geometry_msgs.msg import Quaternion, Twist, TransformStamped, PoseStamped
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


def _quat_msg(yaw: float) -> Quaternion:
    x, y, z, w = conv.yaw_to_quaternion(yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


class IOProvider:
    """Adapter the bridge uses to talk to Viam components.

    The navigation/SLAM models supply concrete async callables; the bridge stays
    free of any Viam SDK imports.
    """

    def __init__(self, read_lidar_points, read_twist, drive_base, stop_base):
        self.read_lidar_points = read_lidar_points  # async (name) -> (N,3) np.ndarray
        self.read_twist = read_twist  # async () -> (vx, vy, vtheta)
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
        self._cmd_timeout = nav_cfg.cmd_vel_timeout if nav_cfg else 0.5

        # Dead-reckoned odom pose (corrected by slam_toolbox's map->odom).
        self._odom = conv.Pose2D(0.0, 0.0, 0.0)
        self._last_odom_time = time.time()

        # Publishers -------------------------------------------------------
        self._scan_pubs: List = []
        for i, _ in enumerate(slam_cfg.lidars):
            self._scan_pubs.append(self.create_publisher(LaserScan, f"scan_{i}", 10))
        self._merged_scan_pub = self.create_publisher(LaserScan, "scan", 10)
        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._keepout_pub = self.create_publisher(
            OccupancyGrid, "keepout_filter_mask", _LATCHED_QOS
        )
        self._speed_pub = self.create_publisher(
            OccupancyGrid, "speed_filter_mask", _LATCHED_QOS
        )

        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_lidar_tfs()

        # TF listener for map-frame pose queries.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Latest occupancy map published by slam_toolbox.
        self._latest_map: Optional[Dict] = None

        # Subscriptions ----------------------------------------------------
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(OccupancyGrid, "map", self._on_map, _LATCHED_QOS)

        # Timers -----------------------------------------------------------
        self.create_timer(1.0 / max(slam_cfg.scan_rate_hz, 1.0), self._on_scan_timer)
        self.create_timer(1.0 / max(slam_cfg.odom_rate_hz, 1.0), self._on_odom_timer)
        self.create_timer(0.1, self._on_watchdog)

        # Action client ----------------------------------------------------
        self._nav_action: Optional[ActionClient] = None
        self._goal_handle = None
        self._last_feedback: Dict = {}
        self._last_result_status: Optional[str] = None
        if NavigateToPose is not None:
            self._nav_action = ActionClient(self, NavigateToPose, "navigate_to_pose")

    # -- helpers -------------------------------------------------------------
    def _run(self, coro, timeout: float = 2.0):
        """Run an async Viam call on the module loop from this ROS thread."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

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
        stamp = self.get_clock().now().to_msg()
        for i, lidar in enumerate(self._slam_cfg.lidars):
            try:
                points = self._run(self._io.read_lidar_points(lidar.name))
            except Exception as exc:  # noqa: BLE001 - keep bridge alive on sensor hiccups
                self.get_logger().warn(f"lidar {lidar.name} read failed: {exc}")
                continue
            if points is None:
                continue
            points = np.asarray(points, dtype=float)

            # Per-lidar scan in its own frame for Nav2's obstacle layer.
            scan = conv.pointcloud_to_scan(
                points,
                z_min=lidar.z_min,
                z_max=lidar.z_max,
                num_bins=self._slam_cfg.scan_bins,
                range_min=lidar.min_range,
                range_max=lidar.max_range,
            )
            self._scan_pubs[i].publish(self._to_ros_scan(scan, f"laser_{i}", stamp))

            # Accumulate base_link-frame points for the merged SLAM scan.
            mount = conv.Pose2D(lidar.x, lidar.y, lidar.theta)
            base_pts = conv.pointcloud_to_scan(
                points, z_min=lidar.z_min, z_max=lidar.z_max, sensor_pose=mount,
                num_bins=self._slam_cfg.scan_bins, range_min=lidar.min_range,
                range_max=lidar.max_range,
            ).to_points()
            if base_pts.size:
                merged_points.append(base_pts)

        stacked = np.vstack(merged_points) if merged_points else np.empty((0, 2))
        merged = conv.points_to_scan(stacked, num_bins=self._slam_cfg.scan_bins)
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
            vx, vy, vtheta = self._run(self._io.read_twist())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"odometry read failed: {exc}")
            return
        now = time.time()
        dt = now - self._last_odom_time
        self._last_odom_time = now
        if dt <= 0 or dt > 1.0:
            return
        # Integrate body-frame twist into the odom-frame pose (dead reckoning).
        th = self._odom.theta
        dx = (vx * math.cos(th) - vy * math.sin(th)) * dt
        dy = (vx * math.sin(th) + vy * math.cos(th)) * dt
        self._odom = conv.Pose2D(self._odom.x + dx, self._odom.y + dy, th + vtheta * dt)

        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._frames.odom
        odom.child_frame_id = self._frames.base_link
        odom.pose.pose.position.x = self._odom.x
        odom.pose.pose.position.y = self._odom.y
        odom.pose.pose.orientation = _quat_msg(self._odom.theta)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = vtheta
        self._odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._frames.odom
        t.child_frame_id = self._frames.base_link
        t.transform.translation.x = self._odom.x
        t.transform.translation.y = self._odom.y
        t.transform.rotation = _quat_msg(self._odom.theta)
        self._tf_broadcaster.sendTransform(t)

    # -- cmd_vel + watchdog --------------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        self._last_cmd_time = time.time()
        if not self._nav_active:
            return
        vx = msg.linear.x
        vy = msg.linear.y if self._is_omni() else 0.0
        vtheta = msg.angular.z
        try:
            self._run(self._io.drive_base(vx, vy, vtheta))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"drive_base failed: {exc}")

    def _on_watchdog(self) -> None:
        if not self._nav_active:
            return
        if time.time() - self._last_cmd_time > self._cmd_timeout:
            try:
                self._run(self._io.stop_base())
            except Exception:  # noqa: BLE001
                pass

    def _is_omni(self) -> bool:
        return bool(self._nav_cfg and self._nav_cfg.kinematics == "omni")

    def set_nav_config(self, nav_cfg: NavConfig) -> None:
        self._nav_cfg = nav_cfg
        self._cmd_timeout = nav_cfg.cmd_vel_timeout

    def set_nav_active(self, active: bool) -> None:
        self._nav_active = active
        if not active:
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
    def send_nav_goal(self, x: float, y: float, theta: float) -> bool:
        if self._nav_action is None:
            raise RuntimeError("Nav2 action 'navigate_to_pose' unavailable")
        if not self._nav_action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Nav2 action server not available")
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self._frames.map
        pose.header.stamp = self.get_clock().now().to_msg()
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
        self._odom = pose

    # -- map + pose queries --------------------------------------------------
    def _on_map(self, msg: OccupancyGrid) -> None:
        width = msg.info.width
        height = msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(height, width)
        self._latest_map = {
            "grid": grid,
            "resolution": msg.info.resolution,
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
        }

    def get_map(self) -> Optional[Dict]:
        return self._latest_map

    def get_pose_in_map(self) -> Optional[conv.Pose2D]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._frames.map, self._frames.base_link, rclpy.time.Time()
            )
        except Exception:  # noqa: BLE001 - transform may not be available yet
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return conv.Pose2D(t.x, t.y, conv.quaternion_to_yaw(q.x, q.y, q.z, q.w))


class _ZeroDuration:
    sec = 0

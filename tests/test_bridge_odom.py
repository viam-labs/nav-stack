import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy")


class _FakeNode:
    pass


sys.modules.setdefault("rclpy", MagicMock())
sys.modules.setdefault("rclpy.node", MagicMock(Node=_FakeNode))
sys.modules.setdefault("rclpy.qos", MagicMock())
sys.modules.setdefault("rclpy.action", MagicMock())
sys.modules.setdefault("rclpy.time", MagicMock())
sys.modules.setdefault("rclpy.duration", MagicMock(Duration=MagicMock()))
for _mod in (
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "tf2_ros",
    "nav2_msgs",
    "nav2_msgs.action",
):
    sys.modules.setdefault(_mod, MagicMock())

from src.ros import conversions as conv
from src.ros.bridge import BridgeNode


def _odom_bridge_stub(*, sample: conv.OdomReading):
    bridge = SimpleNamespace(
        _io=SimpleNamespace(read_odometry=MagicMock(return_value=sample)),
        _slam_cfg=SimpleNamespace(odom_rate_hz=15.0),
        _odom=conv.Pose2D(0.0, 0.0, 0.0),
        _last_odom_time=0.0,
        _odom_integrate_warned=False,
        _frames=SimpleNamespace(odom="odom", base_link="base_link"),
        get_logger=MagicMock(return_value=MagicMock()),
        get_clock=MagicMock(
            return_value=MagicMock(now=MagicMock(return_value=MagicMock(to_msg=MagicMock())))
        ),
        _odom_pub=MagicMock(),
        _tf_broadcaster=MagicMock(),
        _run=lambda coro: coro,
    )
    bridge._bounded_odom_dt = lambda dt: BridgeNode._bounded_odom_dt(bridge, dt)
    bridge._publish_odom_snapshot = lambda stamp, vx, vy, vtheta: BridgeNode._publish_odom_snapshot(
        bridge, stamp, vx, vy, vtheta
    )
    bridge._last_twist = (0.0, 0.0, 0.0)
    return bridge


def test_scan_timer_stamps_scans_at_read_start(monkeypatch):
    import numpy as np

    stamp = MagicMock(name="stamp")
    clock = MagicMock(now=MagicMock(return_value=MagicMock(to_msg=MagicMock(return_value=stamp))))
    scan = conv.LaserScan2D(
        ranges=np.array([1.0, 2.0]),
        angle_min=-1.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )
    lidar_pts = conv.LidarPoints(
        sensor=np.array([[1.0, 0.0, 0.0]]),
        base_link=np.array([[1.0, 0.0, 0.0]]),
        sensor_scan=scan,
    )
    bridge = SimpleNamespace(
        _io=SimpleNamespace(read_lidar_points=MagicMock()),
        _slam_cfg=SimpleNamespace(
            lidars=[MagicMock(name="l0", z_min=-0.5, z_max=0.5, min_range=0.1, max_range=10.0)],
            scan_bins=360,
            sensor_read_timeout_s=1.0,
        ),
        _run=lambda coro, timeout=None: lidar_pts,
        _empty_scan_warned=False,
        _last_twist=(0.2, 0.0, 0.1),
        _frames=SimpleNamespace(base_link="base_link"),
        get_clock=MagicMock(return_value=clock),
        _publish_odom_snapshot=MagicMock(),
        _scan_pubs=[MagicMock()],
        _merged_scan_pub=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )
    bridge._to_ros_scan = lambda s, frame, st: SimpleNamespace(
        header=SimpleNamespace(stamp=st, frame_id=frame)
    )
    bridge._bounded_scan_stamp = lambda read_start, age_s=0.0: read_start.to_msg()
    bridge._publish_scan_time_tf = MagicMock()
    monkeypatch.setattr("src.ros.bridge.conv.points_to_scan", lambda *a, **k: scan)

    BridgeNode._on_scan_timer(bridge)

    # Scans must carry the read-start stamp (stale geometry stamped "now" would
    # shift obstacles ahead of a moving robot and raytrace-clear their cells).
    published = bridge._merged_scan_pub.publish.call_args[0][0]
    assert published.header.stamp is stamp
    # A TF sample must be published at the scan stamp so fresh slam/Nav2 TF
    # buffers never drop the (past-stamped) scan.
    bridge._publish_scan_time_tf.assert_called_once_with(stamp)
    bridge._publish_odom_snapshot.assert_not_called()


def _scan_timer_bridge(lidar_pts, scan, *, scan_max_age_s=2.0):
    stamp = MagicMock(name="stamp")
    clock = MagicMock(
        now=MagicMock(return_value=MagicMock(to_msg=MagicMock(return_value=stamp)))
    )
    bridge = SimpleNamespace(
        _io=SimpleNamespace(read_lidar_points=MagicMock()),
        _slam_cfg=SimpleNamespace(
            lidars=[
                MagicMock(name="l0", z_min=-0.5, z_max=0.5, min_range=0.1, max_range=10.0)
            ],
            scan_bins=360,
            sensor_read_timeout_s=1.0,
            scan_max_age_s=scan_max_age_s,
        ),
        _run=lambda coro, timeout=None: lidar_pts,
        _empty_scan_warned=False,
        _stale_scan_warned=False,
        _last_twist=(0.2, 0.0, 0.1),
        _frames=SimpleNamespace(base_link="base_link"),
        get_clock=MagicMock(return_value=clock),
        _publish_odom_snapshot=MagicMock(),
        _scan_pubs=[MagicMock()],
        _merged_scan_pub=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )
    bridge._to_ros_scan = lambda s, frame, st: SimpleNamespace(
        header=SimpleNamespace(stamp=st, frame_id=frame)
    )
    bridge._publish_scan_time_tf = MagicMock()
    return bridge, stamp


def test_scan_timer_skips_stale_scan(monkeypatch):
    import numpy as np

    scan = conv.LaserScan2D(
        ranges=np.array([1.0, 2.0]),
        angle_min=-1.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )
    lidar_pts = conv.LidarPoints(
        sensor=np.array([[1.0, 0.0, 0.0]]),
        base_link=np.array([[1.0, 0.0, 0.0]]),
        sensor_scan=scan,
        age_s=5.0,
    )
    bridge, _ = _scan_timer_bridge(lidar_pts, scan, scan_max_age_s=2.0)
    bridge._bounded_scan_stamp = MagicMock()
    monkeypatch.setattr("src.ros.bridge.conv.points_to_scan", lambda *a, **k: scan)

    BridgeNode._on_scan_timer(bridge)

    # A stale scan must not be published to SLAM/Nav2.
    bridge._merged_scan_pub.publish.assert_not_called()
    bridge._scan_pubs[0].publish.assert_not_called()
    bridge._bounded_scan_stamp.assert_not_called()
    assert bridge._stale_scan_warned is True


def test_scan_timer_stamps_at_capture_time_using_age(monkeypatch):
    import numpy as np

    scan = conv.LaserScan2D(
        ranges=np.array([1.0, 2.0]),
        angle_min=-1.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )
    lidar_pts = conv.LidarPoints(
        sensor=np.array([[1.0, 0.0, 0.0]]),
        base_link=np.array([[1.0, 0.0, 0.0]]),
        sensor_scan=scan,
        age_s=0.3,
    )
    bridge, stamp = _scan_timer_bridge(lidar_pts, scan, scan_max_age_s=2.0)
    seen = {}

    def _record_stamp(read_start, age_s=0.0):
        seen["age_s"] = age_s
        return read_start.to_msg()

    bridge._bounded_scan_stamp = _record_stamp
    monkeypatch.setattr("src.ros.bridge.conv.points_to_scan", lambda *a, **k: scan)

    BridgeNode._on_scan_timer(bridge)

    # The reported cache age is forwarded so the scan is stamped at capture time.
    assert seen["age_s"] == 0.3
    bridge._merged_scan_pub.publish.assert_called_once()


def test_odom_pose_at_picks_nearest_history_sample():
    from collections import deque

    p1 = conv.Pose2D(1.0, 0.0, 0.0)
    p2 = conv.Pose2D(2.0, 0.0, 0.0)
    bridge = SimpleNamespace(
        _odom=conv.Pose2D(9.0, 9.0, 0.0),
        _odom_history=deque([(1_000_000_000, p1), (2_000_000_000, p2)]),
    )
    stamp = SimpleNamespace(sec=1, nanosec=100_000_000)  # 1.1s -> nearest p1
    assert BridgeNode._odom_pose_at(bridge, stamp) is p1
    stamp = SimpleNamespace(sec=1, nanosec=900_000_000)  # 1.9s -> nearest p2
    assert BridgeNode._odom_pose_at(bridge, stamp) is p2


def test_odom_pose_at_falls_back_to_current_pose():
    from collections import deque

    current = conv.Pose2D(5.0, 5.0, 0.0)
    bridge = SimpleNamespace(_odom=current, _odom_history=deque())
    stamp = SimpleNamespace(sec=1, nanosec=0)
    assert BridgeNode._odom_pose_at(bridge, stamp) is current


def test_publish_odom_snapshot_updates_twist_and_publishes():
    bridge = SimpleNamespace(
        _odom=conv.Pose2D(1.0, 2.0, 0.25),
        _frames=SimpleNamespace(odom="odom", base_link="base_link"),
        _last_odom_stamp=None,
        _last_twist=(0.0, 0.0, 0.0),
        _odom_pub=MagicMock(),
        _tf_broadcaster=MagicMock(),
    )

    stamp = MagicMock()
    BridgeNode._publish_odom_snapshot(bridge, stamp, 0.5, 0.0, 0.1)

    assert bridge._last_odom_stamp is stamp
    assert bridge._last_twist == (0.5, 0.0, 0.1)
    bridge._odom_pub.publish.assert_called_once()
    bridge._tf_broadcaster.sendTransform.assert_called_once()


def test_bridge_uses_odom_pose_when_available(monkeypatch):
    sample = conv.OdomReading(0.1, 0.0, 0.0, pose=conv.Pose2D(3.0, 4.0, 0.5))
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._odom.x == 3.0
    assert bridge._odom.y == 4.0
    assert bridge._odom.theta == 0.5


def test_bridge_pose_mode_ignores_stale_dt(monkeypatch):
    sample = conv.OdomReading(0.0, 0.0, 0.0, pose=conv.Pose2D(1.0, 2.0, 0.3))
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 5.0)
    bridge._last_odom_time = 0.0

    BridgeNode._on_odom_timer(bridge)

    assert bridge._odom.x == 1.0
    assert bridge._odom.y == 2.0
    assert bridge._odom.theta == 0.3
    bridge.get_logger.return_value.warn.assert_not_called()


def test_bridge_integrates_with_odom_heading_only(monkeypatch):
    sample = conv.OdomReading(1.0, 0.0, 0.0, heading_rad=0.0)
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert math.isclose(bridge._odom.x, 0.1, abs_tol=1e-6)
    assert bridge._odom.y == 0.0
    assert bridge._odom.theta == 0.0


def test_bridge_does_not_snap_to_fused_yaw_deg(monkeypatch):
    """Without odom_yaw_deg, orientation integrates from vtheta instead."""
    sample = conv.parse_odom_from_readings(
        {
            "yaw_deg": 90.0,
            "linear_velocity_mps": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity_dps": {"x": 0.0, "y": 0.0, "z": 10.0},
        }
    )
    assert sample.heading_rad is None
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert math.isclose(bridge._odom.theta, math.radians(10.0) * 0.1, abs_tol=1e-6)


def test_bridge_clamps_integration_for_stale_dt(monkeypatch):
    sample = conv.OdomReading(1.0, 0.0, 0.0, heading_rad=0.0)
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 3.0)
    bridge._last_odom_time = 0.0

    BridgeNode._on_odom_timer(bridge)

    # stale_dt = max(8/15, 0.5) = 0.533...
    assert math.isclose(bridge._odom.x, 8.0 / 15.0, abs_tol=1e-6)
    assert math.isclose(bridge._odom.theta, 0.0, abs_tol=1e-6)


def test_map_updates_can_be_disabled():
    bridge = SimpleNamespace(
        _map_updates_enabled=True,
        _latest_map={"grid": "old"},
    )
    BridgeNode.set_map_updates_enabled(bridge, False)
    assert bridge._map_updates_enabled is False
    assert bridge._latest_map is None

    msg = SimpleNamespace(
        info=SimpleNamespace(
            width=1,
            height=1,
            resolution=0.05,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        ),
        data=[0],
    )
    BridgeNode._on_map(bridge, msg)
    assert bridge._latest_map is None


def test_on_map_tags_generation():
    bridge = SimpleNamespace(
        _map_updates_enabled=True,
        _map_generation=3,
        _latest_map=None,
    )
    msg = SimpleNamespace(
        info=SimpleNamespace(
            width=1,
            height=1,
            resolution=0.05,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        ),
        data=[0],
    )
    BridgeNode._on_map(bridge, msg)
    assert bridge._latest_map["generation"] == 3


def test_set_initial_pose_publishes_to_initialpose():
    published = []

    class _Pub:
        def publish(self, msg):
            published.append(msg)

    bridge = SimpleNamespace(
        _frames=SimpleNamespace(map="map"),
        _initialpose_pub=_Pub(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
    )

    BridgeNode.set_initial_pose(bridge, conv.Pose2D(1.0, 2.0, 0.5))

    assert len(published) == 1
    msg = published[0]
    assert msg.header.frame_id == "map"
    assert math.isclose(msg.pose.pose.position.x, 1.0)
    assert math.isclose(msg.pose.pose.position.y, 2.0)
    assert math.isclose(bridge._last_pose_in_map.x, 1.0)
    assert math.isclose(bridge._last_pose_in_map.y, 2.0)


def test_set_initial_pose_does_not_rewrite_odom_pose():
    published = []

    class _Pub:
        def publish(self, msg):
            published.append(msg)

    bridge = SimpleNamespace(
        _odom=conv.Pose2D(9.0, 8.0, 0.2),
        _frames=SimpleNamespace(map="map"),
        _initialpose_pub=_Pub(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
    )

    BridgeNode.set_initial_pose(bridge, conv.Pose2D(1.0, 2.0, 0.5))

    assert math.isclose(bridge._odom.x, 9.0)
    assert math.isclose(bridge._odom.y, 8.0)
    assert len(published) == 1


def test_get_pose_in_map_returns_cached_pose_on_lookup_failure():
    translation = SimpleNamespace(x=1.2, y=-0.4)
    rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    tf = SimpleNamespace(transform=SimpleNamespace(translation=translation, rotation=rotation))
    bridge = SimpleNamespace(
        _frames=SimpleNamespace(map="map", base_link="base_link"),
        _tf_buffer=SimpleNamespace(lookup_transform=MagicMock(return_value=tf)),
        _last_pose_in_map=None,
    )
    bridge._lookup_pose_in_map = lambda: BridgeNode._lookup_pose_in_map(bridge)

    pose = BridgeNode.get_pose_in_map(bridge)
    assert pose is not None
    assert math.isclose(pose.x, 1.2)
    assert math.isclose(pose.y, -0.4)

    bridge._tf_buffer.lookup_transform = MagicMock(side_effect=RuntimeError("tf unavailable"))
    cached = BridgeNode.get_pose_in_map(bridge)
    assert cached is not None
    assert math.isclose(cached.x, 1.2)
    assert math.isclose(cached.y, -0.4)


def test_keep_odom_tf_alive_republishes_after_stall(monkeypatch):
    bridge = SimpleNamespace(
        _slam_cfg=SimpleNamespace(odom_rate_hz=5.0),
        _last_odom_pub_wall=0.0,
        _publish_odom_snapshot=MagicMock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
    )
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 10.0)

    BridgeNode._keep_odom_tf_alive(bridge)

    bridge._publish_odom_snapshot.assert_called_once_with("stamp", 0.0, 0.0, 0.0)


def test_keep_odom_tf_alive_noop_when_recent(monkeypatch):
    bridge = SimpleNamespace(
        _slam_cfg=SimpleNamespace(odom_rate_hz=5.0),
        _last_odom_pub_wall=9.9,
        _publish_odom_snapshot=MagicMock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp")),
    )
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 10.0)

    BridgeNode._keep_odom_tf_alive(bridge)

    bridge._publish_odom_snapshot.assert_not_called()

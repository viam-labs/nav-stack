import math
import sys
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
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

_STILL_METHODS = (
    "_motion_is_still",
    "_gate_pose_tuple",
    "_dwell_pose_drifted",
    "_reset_still_dwell",
    "_still_gate_ready",
    "_still_gate_commit",
)


def _bind_still(bridge):
    for name in _STILL_METHODS:
        setattr(bridge, name, types.MethodType(getattr(BridgeNode, name), bridge))
    return bridge


def _odom_bridge_stub(*, sample: conv.OdomReading):
    bridge = SimpleNamespace(
        _io=SimpleNamespace(read_odometry=MagicMock(return_value=sample)),
        _slam_cfg=SimpleNamespace(odom_rate_hz=15.0, scan_rate_hz=10.0),
        _odom=conv.Pose2D(0.0, 0.0, 0.0),
        _gate_odom=conv.Pose2D(0.0, 0.0, 0.0),
        _last_odom_time=0.0,
        _odom_integrate_warned=False,
        _imu_vx=0.0,
        _imu_vy=0.0,
        _imu_still_ticks=0,
        _has_wheel_twist=False,
        _heading_only_odom=False,
        _imu_odom_mode="coast",
        _map_when_still=False,
        _map_when_still_yaw=0.08,
        _lidar_odom_enabled=True,
        _lidar_odom_range_flow_only=False,
        _use_lidar_frame_scans=False,
        _imu_ax_bias=0.0,
        _imu_ax_window=deque(maxlen=12),
        _flow_history=deque(),
        _last_flow_wall=0.0,
        _last_flow_vx=0.0,
        _scan_accumulation_s=0.0,
        _pc_accum={},
        _last_imu_ax=None,
        _lidar_odom_status={},
        _prev_scan_for_odom=None,
        _prev_scan_odom_theta=0.0,
        _prev_scan_odom_wall=0.0,
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
    bridge._integrate_imu_body_velocity = (
        lambda sample, dt: BridgeNode._integrate_imu_body_velocity(bridge, sample, dt)
    )
    bridge._apply_lidar_odometry = lambda scan: BridgeNode._apply_lidar_odometry(
        bridge, scan
    )
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
        _has_wheel_twist=True,
        _use_lidar_frame_scans=False,
        _scan_accumulation_s=0.0,
        _map_when_still=False,
        _wall_yaw_correction=False,
        _wall_yaw_status={},
        _apply_lidar_odometry=MagicMock(),
        _lidar_read_times=deque(maxlen=64),
        _scan_pub_times=deque(maxlen=64),
    )
    bridge._to_ros_scan = lambda s, frame, st: SimpleNamespace(
        header=SimpleNamespace(stamp=st, frame_id=frame)
    )
    bridge._bounded_scan_stamp = lambda read_start, age_s=0.0: read_start.to_msg()
    bridge._publish_scan_time_tf = MagicMock()
    bridge._still_gate_ready = lambda: True
    monkeypatch.setattr("src.ros.bridge.conv.merge_scans", lambda *a, **k: scan)

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
        _has_wheel_twist=True,
        _use_lidar_frame_scans=False,
        _scan_accumulation_s=0.0,
        _map_when_still=False,
        _wall_yaw_correction=False,
        _wall_yaw_status={},
        _apply_lidar_odometry=MagicMock(),
        _lidar_read_times=deque(maxlen=64),
        _scan_pub_times=deque(maxlen=64),
    )
    bridge._to_ros_scan = lambda s, frame, st: SimpleNamespace(
        header=SimpleNamespace(stamp=st, frame_id=frame)
    )
    bridge._publish_scan_time_tf = MagicMock()
    bridge._still_gate_ready = lambda: True
    return bridge, stamp


def test_scan_timer_merges_mir_base_link_points_not_native_sensor_ranges():
    # Native ranges say the return is straight ahead in the scanner frame, while
    # mir-base's transformed points place it to the robot's left in base_link.
    # The merged /scan is stamped base_link and therefore must use the latter.
    native_ranges = np.full(360, np.inf)
    native_ranges[180] = 1.0
    native_scan = conv.LaserScan2D(
        ranges=native_ranges,
        angle_min=-math.pi,
        angle_increment=2.0 * math.pi / 360,
        range_min=0.1,
        range_max=10.0,
    )
    lidar_pts = conv.LidarPoints(
        sensor=np.array([[1.0, 0.0, 0.0]]),
        base_link=np.array([[0.0, 2.0, 0.0]]),
        sensor_scan=native_scan,
    )
    bridge, _ = _scan_timer_bridge(lidar_pts, native_scan)
    bridge._bounded_scan_stamp = lambda read_start, age_s=0.0: read_start.to_msg()

    BridgeNode._on_scan_timer(bridge)

    merged = bridge._apply_lidar_odometry.call_args.args[0]
    assert conv.nearest_return_bearing_deg(merged) == pytest.approx(90.0, abs=1.0)
    published = bridge._merged_scan_pub.publish.call_args.args[0]
    assert published.header.frame_id == "base_link"


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
    monkeypatch.setattr("src.ros.bridge.conv.merge_scans", lambda *a, **k: scan)

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
    monkeypatch.setattr("src.ros.bridge.conv.merge_scans", lambda *a, **k: scan)

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
        _last_twist=(0.0, 0.0, 0.0),
        _odom_pub=MagicMock(),
        _tf_broadcaster=MagicMock(),
    )

    stamp = MagicMock()
    BridgeNode._publish_odom_snapshot(bridge, stamp, 0.5, 0.0, 0.1)

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


def test_bridge_uses_imu_orientation_yaw(monkeypatch):
    sample = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.75, "pitch": 0.0, "roll": 0.0},
        }
    )
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._odom.theta == pytest.approx(0.75)


def test_bridge_imu_accel_path_ignores_ahrs_orientation_yaw(monkeypatch):
    """Wit AHRS yaw must not snap /odom — it biases lidar and fans ghost walls."""
    sample = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": math.degrees(1.0)},
            "orientation": {"yaw": 0.75, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.8},
        }
    )
    assert sample.heading_rad == pytest.approx(0.75)
    assert sample.ax is not None
    bridge = _odom_bridge_stub(sample=sample)
    bridge._odom = conv.Pose2D(0.0, 0.0, 0.0)
    bridge._map_when_still = False
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    # Gyro-integrated Δθ≈0.1 rad — NOT snapped to AHRS 0.75.
    assert bridge._odom.theta == pytest.approx(0.1, abs=1e-6)


def test_bridge_map_when_still_freezes_xy_while_pivoting(monkeypatch):
    sample = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": math.degrees(1.0)},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.5, "y": 0.0, "z": 9.8},
        }
    )
    bridge = _odom_bridge_stub(sample=sample)
    bridge._map_when_still = True
    bridge._map_when_still_yaw = 0.08
    bridge._imu_vx = 0.4
    bridge._imu_vy = 0.1
    bridge._odom = conv.Pose2D(1.0, 2.0, 0.0)
    bridge._gate_odom = conv.Pose2D(1.0, 2.0, 0.0)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    # Published TF XY frozen while pivoting (accel noise).
    assert bridge._odom.x == pytest.approx(1.0)
    assert bridge._odom.y == pytest.approx(2.0)
    assert bridge._odom.theta == pytest.approx(0.1, abs=1e-6)
    assert bridge._gate_odom.theta == pytest.approx(0.1, abs=1e-6)
    assert bridge._imu_vx == 0.0
    assert bridge._imu_vy == 0.0


def test_bridge_uses_imu_angular_velocity(monkeypatch):
    sample = conv.parse_odom_from_readings(
        {
            # Viam angular_velocity is deg/s: 57.3 deg/s ≈ 1 rad/s
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": math.degrees(1.0)},
        }
    )
    assert sample.heading_rad is None
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert math.isclose(bridge._odom.theta, 0.1, abs_tol=1e-6)


def test_bridge_mir_odom_pose_bypasses_imu_integration(monkeypatch):
    sample = conv.parse_odom_from_readings(
        {
            "odom_position_x_m": 4.0,
            "odom_position_y_m": 5.0,
            "odom_yaw_deg": 180.0,
            "linear_velocity_mps": {"x": 0.3, "y": 0.0, "z": 0.0},
            "angular_velocity_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 2.0, "y": 0.0, "z": 9.8},
        }
    )
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._odom.x == 4.0
    assert bridge._odom.y == 5.0
    assert math.isclose(bridge._odom.theta, math.pi)
    assert bridge._last_twist == (0.3, 0.0, 0.0)


def test_bridge_mir_twist_integration_unchanged(monkeypatch):
    sample = conv.parse_odom_from_readings(
        {
            "source": "odom+/odom+/status",
            "position_x_m": 10.0,
            "position_y_m": 20.0,
            "yaw_deg": 45.0,
            "linear_velocity_mps": {"x": 1.0, "y": 0.0, "z": 0.0},
            "angular_velocity_dps": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    )
    bridge = _odom_bridge_stub(sample=sample)
    bridge._odom = conv.Pose2D(0.0, 0.0, 0.0)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert math.isclose(bridge._odom.x, 0.1, abs_tol=1e-6)
    assert bridge._odom.y == 0.0
    assert bridge._odom.theta == 0.0


def test_estimate_scan_motion_forward():
    # True planar front wall: range = wall_x / cos(angle). Drive +0.2 m.
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    prev_r = 5.0 / np.cos(angles)
    curr_r = 4.8 / np.cos(angles)
    prev = conv.LaserScan2D(prev_r, angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(curr_r, angle_min, angle_inc, 0.1, 30.0)

    motion = conv.estimate_scan_motion(prev, curr, dtheta=0.0, step_m=0.05)
    assert motion is not None
    dx, dy, dtheta = motion
    assert dx == pytest.approx(0.2, abs=0.06)
    assert abs(dy) <= 0.06
    assert dtheta == 0.0
    assert motion.residual < 0.08
    assert motion.match_fraction > 0.4


def test_bridge_imu_coasts_when_accel_near_zero(monkeypatch):
    # First tick: accelerate forward. Second: near-zero accel should keep velocity.
    accel = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 1.0, "y": 0.0, "z": 9.80665},
        }
    )
    coast = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.05, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=accel)
    clock = {"t": 1.0}

    def _mono():
        return clock["t"]

    monkeypatch.setattr("src.ros.bridge.time.monotonic", _mono)
    bridge._last_odom_time = 0.9
    BridgeNode._on_odom_timer(bridge)
    x_after_accel = bridge._odom.x
    assert x_after_accel > 0.0
    assert bridge._imu_vx > 0.0

    bridge._io.read_odometry = MagicMock(return_value=coast)
    clock["t"] = 1.1
    BridgeNode._on_odom_timer(bridge)
    assert bridge._odom.x > x_after_accel
    assert bridge._imu_vx > 0.05


def test_bridge_accel_only_odom_integrates_on_strong_accel(monkeypatch):
    accel = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 1.2, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=accel)
    bridge._imu_odom_mode = "accel_only"
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._imu_vx > 0.1
    assert bridge._odom.x > 0.0


def test_bridge_accel_only_learns_bias_and_stays_parked(monkeypatch):
    # A constant nonzero ax (tilted IMU) with flat variance must NOT integrate
    # into velocity — this is the parked-cart phantom-motion imprint bug.
    biased = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.45, "y": -0.6, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=biased)
    bridge._imu_odom_mode = "accel_only"
    bridge._last_flow_vx = 0.0
    clock = {"t": 1.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: clock["t"])
    bridge._last_odom_time = 0.95

    for _ in range(10):
        clock["t"] += 0.05
        # Parked robot: lidar range flow keeps reporting ~zero motion.
        bridge._last_flow_wall = clock["t"]
        BridgeNode._on_odom_timer(bridge)

    assert bridge._imu_vx == 0.0
    assert abs(bridge._odom.x) < 0.05
    assert bridge._imu_ax_bias > 0.05


def test_bridge_accel_only_coasts_gently_without_flow_info(monkeypatch):
    # Flat accel + NO flow info (featureless corridor ahead) must decay gently,
    # not freeze — the flow-confirmed hard ZUPT only applies when flow matched.
    coast = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.02, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=coast)
    bridge._imu_odom_mode = "accel_only"
    bridge._imu_vx = 0.5
    bridge._last_flow_wall = 0.0  # stale — no flow matches for a while
    clock = {"t": 100.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: clock["t"])
    bridge._last_odom_time = 99.95

    for _ in range(8):
        clock["t"] += 0.05
        BridgeNode._on_odom_timer(bridge)

    # Window fills at 6 samples; only ~3 gentle decays applied.
    assert bridge._imu_vx > 0.3


def test_bridge_flow_median_zupts_parked_velocity(monkeypatch):
    # Identical scans (parked) must pull a stale nonzero velocity back to zero
    # via the flow median — lidar-confirmed ZUPT.
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    scan = conv.LaserScan2D(5.0 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    bridge = _odom_bridge_stub(sample=conv.OdomReading(0.0, 0.0, 0.0, heading_rad=0.0))
    bridge._lidar_odom_range_flow_only = True
    bridge._imu_vx = 0.5
    clock = {"t": 0.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: clock["t"])

    for _ in range(6):
        clock["t"] += 0.1
        BridgeNode._apply_lidar_odometry(bridge, scan)

    assert bridge._imu_vx < 0.1
    assert bridge._lidar_odom_status.get("accepted") is True
    assert bridge._lidar_odom_status.get("method") == "range_flow"


def test_bridge_flow_median_tracks_forward_motion(monkeypatch):
    # Consistent forward range flow should build up positive velocity.
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    bridge = _odom_bridge_stub(sample=conv.OdomReading(0.0, 0.0, 0.0, heading_rad=0.0))
    bridge._lidar_odom_range_flow_only = True
    clock = {"t": 0.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: clock["t"])

    wall = 6.0
    for _ in range(6):
        scan = conv.LaserScan2D(
            wall / np.cos(angles), angle_min, angle_inc, 0.1, 30.0
        )
        BridgeNode._apply_lidar_odometry(bridge, scan)
        clock["t"] += 0.2
        wall -= 0.1  # 0.5 m/s toward the wall

    assert bridge._imu_vx > 0.2
    assert bridge._last_flow_vx > 0.2


def test_bridge_heading_only_odom_zeros_translation(monkeypatch):
    accel = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 1.0, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=accel)
    bridge._imu_odom_mode = "none"
    bridge._heading_only_odom = True
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._imu_vx == 0.0
    assert bridge._odom.x == 0.0
    assert bridge._odom.y == 0.0


def test_bridge_lidar_odometry_rejects_sign_conflict(monkeypatch):
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    prev = conv.LaserScan2D(5.0 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(4.8 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    bridge = _odom_bridge_stub(sample=conv.OdomReading(0.0, 0.0, 0.0, heading_rad=0.0))
    bridge._has_wheel_twist = False
    bridge._odom = conv.Pose2D(0.0, 0.0, 0.0)
    bridge._imu_vx = -0.6
    bridge._last_imu_ax = -0.8
    clock = {"t": 0.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: clock["t"])

    BridgeNode._apply_lidar_odometry(bridge, prev)
    clock["t"] = 0.2
    BridgeNode._apply_lidar_odometry(bridge, curr)

    assert bridge._imu_vx == pytest.approx(-0.6, abs=0.01)
    assert bridge._lidar_odom_status.get("accepted") is False


def test_bridge_imu_coasts_at_speed_on_small_accel(monkeypatch):
    coast = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.25, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=coast)
    bridge._imu_vx = 0.8
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._imu_vx == pytest.approx(0.8, abs=0.01)


def test_bridge_lidar_odometry_translates_without_wheel_twist(monkeypatch):
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    prev = conv.LaserScan2D(5.0 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(4.8 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    bridge = _odom_bridge_stub(sample=conv.OdomReading(0.0, 0.0, 0.0, heading_rad=0.0))
    bridge._has_wheel_twist = False
    bridge._odom = conv.Pose2D(0.0, 0.0, 0.0)
    clock = {"t": 0.0}

    def _mono():
        return clock["t"]

    monkeypatch.setattr("src.ros.bridge.time.monotonic", _mono)

    BridgeNode._apply_lidar_odometry(bridge, prev)
    clock["t"] = 0.2
    BridgeNode._apply_lidar_odometry(bridge, curr)

    # Lidar odom updates body velocity; pose is integrated by the odom timer.
    assert bridge._imu_vx > 0.15
    assert abs(bridge._imu_vy) < 0.5


def test_bridge_lidar_odometry_skipped_with_wheel_twist():
    n = 90
    angle_min = -math.pi / 3
    angle_inc = (2 * math.pi / 3) / (n - 1)
    angles = angle_min + np.arange(n) * angle_inc
    prev = conv.LaserScan2D(5.0 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    curr = conv.LaserScan2D(4.8 / np.cos(angles), angle_min, angle_inc, 0.1, 30.0)
    bridge = _odom_bridge_stub(sample=conv.OdomReading(0.5, 0.0, 0.0))
    bridge._has_wheel_twist = True
    bridge._odom = conv.Pose2D(0.0, 0.0, 0.0)
    bridge._imu_vx = 0.0

    BridgeNode._apply_lidar_odometry(bridge, prev)
    BridgeNode._apply_lidar_odometry(bridge, curr)

    assert bridge._imu_vx == 0.0
    assert bridge._odom.x == 0.0


def test_bridge_imu_braking_decays_forward_velocity(monkeypatch):
    # Coasting forward, then clear opposing accel (hard brake) should cut vx.
    coast = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.05, "y": 0.0, "z": 9.80665},
        }
    )
    brake = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": -1.5, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=coast)
    bridge._imu_vx = 1.0
    clock = {"t": 1.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: clock["t"])
    bridge._last_odom_time = 0.9

    bridge._io.read_odometry = MagicMock(return_value=brake)
    BridgeNode._on_odom_timer(bridge)

    assert bridge._imu_vx < 0.6


def test_bridge_imu_coasts_despite_no_lidar_odom(monkeypatch):
    # Regression: failing lidar matches must NOT zero coast velocity.
    coast = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 0.05, "y": 0.0, "z": 9.80665},
        }
    )
    bridge = _odom_bridge_stub(sample=coast)
    bridge._imu_vx = 0.8
    bridge._imu_still_ticks = 0
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._imu_vx > 0.7


def test_bridge_integrates_imu_acceleration_with_heading(monkeypatch):
    sample = conv.parse_odom_from_readings(
        {
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "linear_acceleration": {"x": 1.0, "y": 0.0, "z": 9.80665},
        }
    )
    assert sample.ax is not None
    bridge = _odom_bridge_stub(sample=sample)
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: 1.0)
    bridge._last_odom_time = 0.9

    BridgeNode._on_odom_timer(bridge)

    assert bridge._odom.x > 0.0


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
    bridge._lookup_tf_pose_2d = lambda child: BridgeNode._lookup_tf_pose_2d(bridge, child)
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


def _still_gate_ns(**kwargs):
    base = dict(
        _map_when_still=True,
        _map_when_still_dwell_s=0.5,
        _map_when_still_lin=0.05,
        _map_when_still_yaw=0.08,
        _map_when_still_yaw_step=0.0,
        _map_when_still_max_drift_m=0.03,
        _map_when_still_max_drift_deg=1.5,
        _map_when_still_status="",
        _point_cloud_lidars=True,
        _use_lidar_frame_scans=False,
        _still_since=None,
        _dwell_pose0=None,
        _scan_published_this_stop=False,
        _last_still_scan_pose=None,
        _last_odom_ok_wall=9999.0,
        _odom_fail_streak=0,
        _last_odom_error=None,
        _last_twist=(0.0, 0.0, 0.0),
        _odom=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        _gate_odom=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        _pc_accum={},
    )
    base.update(kwargs)
    return _bind_still(SimpleNamespace(**base))


def test_map_when_still_disabled_always_publishes():
    bridge = _bind_still(
        SimpleNamespace(
            _map_when_still=False,
            _map_when_still_status="",
            _point_cloud_lidars=True,
            _use_lidar_frame_scans=True,
        )
    )
    assert bridge._still_gate_ready() is True
    assert bridge._map_when_still_status == "disabled"


def test_map_when_still_ignores_non_point_cloud_mir_path():
    """MiR get_laser_scan path must keep continuous /scan even if flag is set."""
    bridge = _bind_still(
        SimpleNamespace(
            _map_when_still=True,
            _map_when_still_status="",
            _point_cloud_lidars=False,
            _use_lidar_frame_scans=False,
            _last_twist=(1.0, 0.0, 0.0),
        )
    )
    assert bridge._still_gate_ready() is True
    assert bridge._map_when_still_status == "disabled_non_point_cloud"


def test_map_when_still_one_scan_per_stop(monkeypatch):
    bridge = _still_gate_ns()
    t = {"now": 100.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])

    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status.startswith("dwelling")
    t["now"] = 100.6
    assert bridge._still_gate_ready() is True
    bridge._still_gate_commit()
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status == "published_this_stop"

    bridge._last_twist = (0.2, 0.0, 0.0)
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status == "moving"
    bridge._last_twist = (0.0, 0.0, 0.0)
    bridge._gate_odom.x = 1.0
    t["now"] = 101.0
    assert bridge._still_gate_ready() is False
    t["now"] = 101.7
    assert bridge._still_gate_ready() is True


def test_map_when_still_blocks_when_odom_dead(monkeypatch):
    """Without odom we cannot prove stillness — never publish."""
    bridge = _still_gate_ns(
        _scan_published_this_stop=True,
        _last_still_scan_pose=(0.0, 0.0, 0.0),
        _last_odom_ok_wall=50.0,
        _odom_fail_streak=10,
        _last_odom_error="EOF",
    )
    t = {"now": 52.0}  # 2s after last odom ok
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status.startswith("odom_unavailable")
    t["now"] = 60.0
    assert bridge._still_gate_ready() is False


def test_map_when_still_yaw_step_during_pivot(monkeypatch):
    """Optional yaw-step: pause mid-turn → extra /scan when step_deg > 0."""
    bridge = _still_gate_ns(_map_when_still_yaw_step=math.radians(30.0))
    t = {"now": 200.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])

    assert bridge._still_gate_ready() is False
    t["now"] = 200.6
    assert bridge._still_gate_ready() is True
    bridge._still_gate_commit()
    assert bridge._last_still_scan_pose[2] == pytest.approx(0.0)

    # In-place spin: no linear motion, yaw rate above still threshold.
    bridge._last_twist = (0.0, 0.0, 0.5)
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status == "pivoting"

    # Pause after ~45° — enough for a yaw-step scan.
    bridge._last_twist = (0.0, 0.0, 0.0)
    bridge._gate_odom.theta = math.radians(45.0)
    t["now"] = 201.0
    assert bridge._still_gate_ready() is False
    t["now"] = 201.6
    assert bridge._still_gate_ready() is True
    assert bridge._map_when_still_status == "ready_yaw_step"
    bridge._still_gate_commit()
    assert bridge._still_gate_ready() is False


def test_map_when_still_pose_hop_allows_rescan(monkeypatch):
    """If twist looks still but odom moved, still allow a new stop scan."""
    bridge = _still_gate_ns(
        _scan_published_this_stop=True,
        _last_still_scan_pose=(0.0, 0.0, 0.0),
        _odom=SimpleNamespace(x=0.5, y=0.0, theta=0.0),
        _gate_odom=SimpleNamespace(x=0.5, y=0.0, theta=0.0),
    )
    t = {"now": 300.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])
    assert bridge._still_gate_ready() is False
    t["now"] = 300.6
    assert bridge._still_gate_ready() is True
    bridge._still_gate_commit()
    assert bridge._map_when_still_status == "publishing"


def test_apply_map_pose_correction_shifts_odom_to_match():
    map_to_odom = conv.Pose2D(2.0, 1.0, math.radians(15.0))
    target_map_pose = conv.Pose2D(5.0, -3.0, math.radians(90.0))
    stamp = MagicMock(name="stamp")
    clock = MagicMock(
        now=MagicMock(return_value=MagicMock(to_msg=MagicMock(return_value=stamp)))
    )
    bridge = SimpleNamespace(
        _odom=conv.Pose2D(10.0, 10.0, 0.0),
        _gate_odom=conv.Pose2D(10.0, 10.0, 0.0),
        _imu_vx=0.3,
        _imu_vy=0.1,
        _lookup_map_to_odom=lambda: map_to_odom,
        _publish_odom_snapshot=MagicMock(),
        _pose_dict=BridgeNode._pose_dict,
        get_clock=MagicMock(return_value=clock),
    )
    out = BridgeNode.apply_map_pose_correction(bridge, target_map_pose)

    assert out["applied"] is True
    # The corrected odom pose must compose with map->odom back to the target.
    got = conv.compose_poses(map_to_odom, bridge._odom)
    assert got.x == pytest.approx(target_map_pose.x)
    assert got.y == pytest.approx(target_map_pose.y)
    assert got.theta == pytest.approx(target_map_pose.theta)
    assert bridge._gate_odom == bridge._odom
    # IMU coast must not drag the corrected pose.
    assert bridge._imu_vx == 0.0 and bridge._imu_vy == 0.0
    # TF published immediately at the corrected pose.
    bridge._publish_odom_snapshot.assert_called_once_with(stamp, 0.0, 0.0, 0.0)


def test_apply_map_pose_correction_requires_map_to_odom():
    bridge = SimpleNamespace(
        _odom=conv.Pose2D(0.0, 0.0, 0.0),
        _gate_odom=conv.Pose2D(0.0, 0.0, 0.0),
        _imu_vx=0.0,
        _imu_vy=0.0,
        _lookup_map_to_odom=lambda: None,
        _publish_odom_snapshot=MagicMock(),
        _pose_dict=BridgeNode._pose_dict,
        get_clock=MagicMock(),
    )
    out = BridgeNode.apply_map_pose_correction(bridge, conv.Pose2D(1.0, 1.0, 0.0))
    assert out["applied"] is False
    assert bridge._odom == conv.Pose2D(0.0, 0.0, 0.0)
    bridge._publish_odom_snapshot.assert_not_called()


def test_apply_wall_yaw_correction_updates_odom():
    bias = math.radians(6.0)
    xs = np.linspace(-2.0, 2.0, 80)
    ys = np.full(80, 1.5)
    c, s = math.cos(bias), math.sin(bias)
    pts = np.stack([c * xs - s * ys, s * xs + c * ys], axis=1)
    scan = conv.points_to_scan(pts, num_bins=720, range_min=0.05, range_max=25.0)

    bridge = SimpleNamespace(
        _wall_yaw_min_length_m=2.0,
        _wall_yaw_max_step_deg=2.0,
        _wall_yaw_blend=1.0,
        _wall_yaw_status={},
        _odom=conv.Pose2D(1.0, 2.0, 0.0),
        _gate_odom=conv.Pose2D(1.0, 2.0, 0.0),
    )
    BridgeNode._apply_wall_yaw_correction(bridge, scan)
    assert bridge._wall_yaw_status.get("accepted") is True
    assert bridge._wall_yaw_status.get("applied") is True
    # Cap 2° with blend 1.0 → theta ~= -2°.
    assert bridge._odom.theta == pytest.approx(-math.radians(2.0), abs=math.radians(0.5))
    assert bridge._odom.x == pytest.approx(1.0)
    assert bridge._gate_odom.theta == pytest.approx(bridge._odom.theta)


def test_measured_hz_from_recent_timestamps():
    from src.ros.bridge import measured_hz

    times = deque([100.0, 100.1, 100.2, 100.3, 100.4], maxlen=64)
    hz = measured_hz(times, now=100.4, window_s=2.0)
    assert hz == pytest.approx(10.0)
    assert measured_hz(deque([100.0]), now=100.1) is None
    # Samples outside the window are ignored.
    old = deque([90.0, 90.1, 100.0, 100.1], maxlen=64)
    assert measured_hz(old, now=100.1, window_s=2.0) == pytest.approx(10.0)

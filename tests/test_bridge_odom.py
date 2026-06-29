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
    return bridge


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


def test_get_pose_in_map_returns_cached_pose_on_lookup_failure():
    translation = SimpleNamespace(x=1.2, y=-0.4)
    rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    tf = SimpleNamespace(transform=SimpleNamespace(translation=translation, rotation=rotation))
    bridge = SimpleNamespace(
        _frames=SimpleNamespace(map="map", base_link="base_link"),
        _tf_buffer=SimpleNamespace(lookup_transform=MagicMock(return_value=tf)),
        _last_pose_in_map=None,
    )

    pose = BridgeNode.get_pose_in_map(bridge)
    assert pose is not None
    assert math.isclose(pose.x, 1.2)
    assert math.isclose(pose.y, -0.4)

    bridge._tf_buffer.lookup_transform = MagicMock(side_effect=RuntimeError("tf unavailable"))
    cached = BridgeNode.get_pose_in_map(bridge)
    assert cached is not None
    assert math.isclose(cached.x, 1.2)
    assert math.isclose(cached.y, -0.4)

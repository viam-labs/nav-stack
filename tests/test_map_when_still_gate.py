"""Strict still-gate: only match when truly stopped (post-capture commit)."""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


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


def _still_bridge(**kwargs):
    base = dict(
        _map_when_still=True,
        _map_when_still_dwell_s=1.0,
        _map_when_still_lin=0.02,
        _map_when_still_yaw=0.04,
        _map_when_still_yaw_step=0.0,
        _map_when_still_max_drift_m=0.03,
        _map_when_still_max_drift_deg=1.5,
        _map_when_still_status="",
        _point_cloud_lidars=True,
        _use_lidar_frame_scans=False,
        _still_since=None,
        _dwell_pose0=None,
        _scan_published_this_stop=False,
        _last_still_scan_yaw=None,
        _last_still_scan_pose=None,
        _last_still_scan_wall=None,
        _last_odom_ok_wall=9999.0,
        _odom_fail_streak=0,
        _last_odom_error=None,
        _last_twist=(0.0, 0.0, 0.0),
        _odom=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        _gate_odom=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        _pc_accum={},
        _imu_vx=0.0,
        _imu_vy=0.0,
        _imu_still_ticks=0,
    )
    base.update(kwargs)
    return _bind_still(SimpleNamespace(**base))


def test_ready_requires_full_dwell(monkeypatch):
    bridge = _still_bridge()
    t = {"now": 10.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status.startswith("dwelling")
    t["now"] = 10.5
    assert bridge._still_gate_ready() is False
    t["now"] = 11.1
    assert bridge._still_gate_ready() is True
    assert bridge._map_when_still_status == "ready"
    # Uncommitted: ready again until commit.
    assert bridge._still_gate_ready() is True


def test_commit_blocks_until_hop(monkeypatch):
    bridge = _still_bridge()
    t = {"now": 20.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])
    assert bridge._still_gate_ready() is False  # start dwell
    t["now"] = 21.1
    assert bridge._still_gate_ready() is True
    bridge._still_gate_commit()
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status == "published_this_stop"
    # Hop shifts gate pose; dwell aborts as drift then restarts at the new pose.
    bridge._gate_odom.x = 0.5
    t["now"] = 22.0
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status == "dwell_drift"
    t["now"] = 22.1
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status.startswith("dwelling")
    t["now"] = 23.2
    assert bridge._still_gate_ready() is True


def test_dwell_aborts_on_pose_drift(monkeypatch):
    bridge = _still_bridge()
    t = {"now": 30.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])
    assert bridge._still_gate_ready() is False
    bridge._gate_odom.theta = math.radians(3.0)
    t["now"] = 30.2
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status == "dwell_drift"


def test_odom_unavailable_blocks_publish(monkeypatch):
    bridge = _still_bridge(_last_odom_ok_wall=0.0)
    t = {"now": 5.0}
    monkeypatch.setattr("src.ros.bridge.time.monotonic", lambda: t["now"])
    assert bridge._still_gate_ready() is False
    assert bridge._map_when_still_status.startswith("odom_unavailable")

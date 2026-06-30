import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Minimal Node stand-in so BridgeNode remains a real class.
class _FakeNode:
    pass

# Stub ROS 2 deps so bridge can be imported without a ROS install.
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

from src.ros.bridge import BridgeNode


def _nav_bridge_stub():
    bridge = SimpleNamespace(
        _goal_handle=None,
        _last_result_status="active",
        _nav_active=True,
        get_logger=MagicMock(return_value=MagicMock()),
        set_nav_active=MagicMock(),
    )
    return bridge


def test_on_goal_response_handles_future_exception():
    bridge = _nav_bridge_stub()
    future = MagicMock()
    future.result.side_effect = RuntimeError("action server unavailable")

    BridgeNode._on_goal_response(bridge, future)

    assert bridge._last_result_status == "failed"
    assert bridge._goal_handle is None
    bridge.set_nav_active.assert_called_once_with(False)


def test_on_goal_response_handles_rejected_goal():
    bridge = _nav_bridge_stub()
    future = MagicMock()
    handle = MagicMock()
    handle.accepted = False
    future.result.return_value = handle

    BridgeNode._on_goal_response(bridge, future)

    assert bridge._last_result_status == "rejected"
    bridge.set_nav_active.assert_called_once_with(False)


def test_on_nav_result_handles_future_exception():
    bridge = _nav_bridge_stub()
    future = MagicMock()
    future.result.side_effect = RuntimeError("result unavailable")

    BridgeNode._on_nav_result(bridge, future)

    assert bridge._last_result_status == "failed"
    bridge.set_nav_active.assert_called_once_with(False)


def test_on_nav_result_maps_success_status():
    bridge = _nav_bridge_stub()
    future = MagicMock()
    future.result.return_value = MagicMock(status=4)

    BridgeNode._on_nav_result(bridge, future)

    assert bridge._last_result_status == "succeeded"
    bridge.set_nav_active.assert_called_once_with(False)


def test_flush_pending_nav_goal_runs_on_watchdog():
    bridge = _nav_bridge_stub()
    bridge._nav_goal_lock = __import__("threading").Lock()
    bridge._pending_nav_goal = None
    bridge._cancel_inflight_nav = MagicMock()
    bridge._ensure_nav_action_client = MagicMock()
    bridge._publish_nav_goal = MagicMock(return_value=True)
    done = __import__("threading").Event()
    outcome: dict = {}
    bridge._pending_nav_goal = (1.0, 2.0, 0.5, done, outcome)

    BridgeNode._flush_pending_nav_goal(bridge)

    assert bridge._pending_nav_goal is None
    assert outcome["ok"] is True
    assert done.is_set()
    bridge._publish_nav_goal.assert_called_once_with(1.0, 2.0, 0.5)

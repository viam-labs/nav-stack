import math
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
        _ensure_map_pose_ready=MagicMock(),
        _wait_for_map_tf=MagicMock(return_value=True),
        _lookup_pose_in_map=MagicMock(return_value=None),
        _last_pose_in_map=None,
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


def test_send_nav_goal_fails_without_map_tf():
    bridge = _nav_bridge_stub()
    bridge._ensure_map_pose_ready = MagicMock(
        side_effect=RuntimeError("map->base_link transform not available")
    )

    with pytest.raises(RuntimeError, match="map->base_link transform not available"):
        BridgeNode.send_nav_goal(bridge, 1.0, 2.0, 0.5)


def test_send_nav_goal_proceeds_on_stale_tf_when_previously_localized():
    from src.ros import conversions as conv

    bridge = _nav_bridge_stub()
    # Cached pose is enough: readiness helper returns without blocking.
    bridge._ensure_map_pose_ready = MagicMock()
    bridge._last_pose_in_map = conv.Pose2D(1.0, 2.0, 0.0)
    bridge._ensure_nav_action_client = MagicMock()
    bridge._wait_for_rclpy_action_server = MagicMock(return_value=True)
    bridge._cancel_inflight_nav = MagicMock()
    bridge._publish_nav_goal = MagicMock(return_value=True)

    ok = BridgeNode.send_nav_goal(bridge, 1.0, 2.0, 0.5)

    assert ok is True
    bridge._publish_nav_goal.assert_called_once_with(1.0, 2.0, 0.5)
    bridge._ensure_map_pose_ready.assert_called_once()


def test_ensure_map_pose_ready_uses_cache_without_waiting():
    from src.ros import conversions as conv

    bridge = SimpleNamespace(
        _last_pose_in_map=conv.Pose2D(1.0, 2.0, 0.0),
        _lookup_pose_in_map=MagicMock(return_value=None),
        _wait_for_map_tf=MagicMock(return_value=False),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._ensure_map_pose_ready(bridge, require_live=False)

    bridge._wait_for_map_tf.assert_not_called()


def test_ensure_map_pose_ready_fails_fast_when_never_localized():
    bridge = SimpleNamespace(
        _last_pose_in_map=None,
        _lookup_pose_in_map=MagicMock(return_value=None),
        _wait_for_map_tf=MagicMock(return_value=False),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    with pytest.raises(RuntimeError, match="localize before planning"):
        BridgeNode._ensure_map_pose_ready(bridge, fail_fast_s=0.5)

    bridge._wait_for_map_tf.assert_called_once_with(timeout_s=0.5)


def test_send_nav_goal_recreates_stale_action_client_then_publishes():
    bridge = _nav_bridge_stub()
    bridge._ensure_nav_action_client = MagicMock()
    bridge._wait_for_rclpy_action_server = MagicMock(side_effect=[False, True])
    bridge.reset_nav_action_client = MagicMock()
    bridge._cancel_inflight_nav = MagicMock()
    bridge._publish_nav_goal = MagicMock(return_value=True)
    bridge._cli_nav_action_visible = MagicMock(return_value=False)
    bridge._send_nav_goal_via_cli = MagicMock()
    bridge._log_nav_action_diagnostics = MagicMock()

    ok = BridgeNode.send_nav_goal(bridge, 1.0, 2.0, 0.5)

    assert ok is True
    bridge.reset_nav_action_client.assert_called_once()
    bridge._publish_nav_goal.assert_called_once_with(1.0, 2.0, 0.5)
    bridge._send_nav_goal_via_cli.assert_not_called()


def test_send_nav_goal_falls_back_to_cli_when_rclpy_not_ready():
    bridge = _nav_bridge_stub()
    bridge._ensure_nav_action_client = MagicMock()
    bridge._wait_for_rclpy_action_server = MagicMock(side_effect=[False, False])
    bridge.reset_nav_action_client = MagicMock()
    bridge._cancel_inflight_nav = MagicMock()
    bridge._publish_nav_goal = MagicMock(return_value=True)
    bridge._cli_nav_action_visible = MagicMock(return_value=True)
    bridge._send_nav_goal_via_cli = MagicMock(return_value=True)
    bridge._log_nav_action_diagnostics = MagicMock()

    ok = BridgeNode.send_nav_goal(bridge, 1.0, 2.0, 0.5)

    assert ok is True
    bridge._publish_nav_goal.assert_not_called()
    bridge._send_nav_goal_via_cli.assert_called_once_with(1.0, 2.0, 0.5)


def _twist(vx=0.0, vy=0.0, wz=0.0):
    return SimpleNamespace(
        linear=SimpleNamespace(x=vx, y=vy),
        angular=SimpleNamespace(z=wz),
    )


def test_on_cmd_vel_stashes_latest_without_blocking():
    lock = __import__("threading").Lock()
    bridge = SimpleNamespace(
        _nav_active=True,
        _last_cmd_time=0.0,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=None,
        _is_omni=lambda: False,
        _run=MagicMock(),
    )

    BridgeNode._on_cmd_vel(bridge, _twist(0.1, 0.0, 0.2))
    BridgeNode._on_cmd_vel(bridge, _twist(0.3, 0.0, 0.4))

    assert bridge._pending_cmd_vel == (0.3, 0.0, 0.4)
    bridge._run.assert_not_called()


def test_on_drive_timer_sends_latest_and_clears():
    lock = __import__("threading").Lock()
    io = SimpleNamespace(drive_base=MagicMock(return_value="coro"))
    bridge = SimpleNamespace(
        _nav_active=True,
        _nav_cfg=None,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=(0.3, 0.0, 0.4),
        _io=io,
        _run=MagicMock(),
        record_cmd_vel=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._on_drive_timer(bridge)

    bridge.record_cmd_vel.assert_called_once_with(0.3, 0.0, 0.4, source="nav2")
    io.drive_base.assert_called_once_with(0.3, 0.0, 0.4, record_source=None)
    bridge._run.assert_called_once_with("coro")
    assert bridge._pending_cmd_vel is None

    # Nothing pending -> no base call.
    bridge._run.reset_mock()
    BridgeNode._on_drive_timer(bridge)
    bridge._run.assert_not_called()

def test_on_drive_timer_snaps_near_zero_to_stop():
    lock = __import__("threading").Lock()
    io = SimpleNamespace(drive_base=MagicMock(return_value="coro"))
    bridge = SimpleNamespace(
        _nav_active=True,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=(0.01, 0.0, 0.02),
        _io=io,
        _run=MagicMock(),
        record_cmd_vel=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._on_drive_timer(bridge)

    # History keeps the pre-snap Nav2 command; base gets zeros.
    bridge.record_cmd_vel.assert_called_once_with(0.01, 0.0, 0.02, source="nav2")
    io.drive_base.assert_called_once_with(0.0, 0.0, 0.0, record_source=None)


def test_set_nav_active_false_clears_pending_cmd_vel():
    lock = __import__("threading").Lock()
    io = SimpleNamespace(stop_base=MagicMock(return_value="coro"))
    bridge = SimpleNamespace(
        _nav_active=True,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=(0.3, 0.0, 0.4),
        _io=io,
        _run=MagicMock(),
    )

    BridgeNode.set_nav_active(bridge, False)

    assert bridge._pending_cmd_vel is None
    bridge._run.assert_called_once_with("coro")


def test_record_cmd_vel_maps_viam_axes():
    from collections import deque

    bridge = SimpleNamespace(
        _nav_active=True,
        _slam_cfg=SimpleNamespace(base_velocity_convention="viam"),
        _last_cmd_vel_wall=0.0,
        _last_cmd_vel={},
        _cmd_vel_history=deque(maxlen=20),
    )
    BridgeNode.record_cmd_vel(bridge, 0.5, 0.0, 0.2, source="nav2")
    cmd = bridge._last_cmd_vel
    assert cmd["source"] == "nav2"
    assert cmd["ros_vx_mps"] == 0.5
    assert cmd["viam_linear_x_mm_s"] == 0.0
    assert cmd["viam_linear_y_mm_s"] == 500.0
    assert cmd["viam_angular_z_deg_s"] == pytest.approx(math.degrees(0.2), rel=1e-3)
    assert cmd["convention"] == "viam"


def test_cmd_vel_history_keeps_drive_cmds_after_stop():
    from collections import deque

    bridge = SimpleNamespace(
        _nav_active=True,
        _slam_cfg=SimpleNamespace(base_velocity_convention="viam"),
        _last_cmd_vel_wall=0.0,
        _last_cmd_vel={},
        _cmd_vel_history=deque(maxlen=20),
    )
    BridgeNode.record_cmd_vel(bridge, 0.5, 0.0, -1.5, source="nav2")
    BridgeNode.record_cmd_vel(bridge, 0.5, 0.0, -1.5, source="nav2")
    BridgeNode.record_cmd_vel(bridge, 0.4, 0.0, -1.2, source="nav2")
    BridgeNode.record_cmd_vel(bridge, 0.0, 0.0, 0.0, source="stop")

    hist = BridgeNode.cmd_vel_history(bridge)
    assert len(hist) == 3
    assert hist[0]["ros_vx_mps"] == 0.5
    assert hist[2]["source"] == "stop"
    assert BridgeNode.last_cmd_vel(bridge)["source"] == "stop"


def test_nav_status_includes_last_cmd_vel():
    bridge = SimpleNamespace(
        _last_result_status="active",
        _nav_active=True,
        _last_feedback={"distance_remaining": 1.2},
        _active_nav_goal={"x": 3.0, "y": -1.0, "theta": 0.5},
        last_cmd_vel=MagicMock(
            return_value={
                "source": "nav2",
                "viam_linear_y_mm_s": 300.0,
                "age_s": 0.1,
            }
        ),
        cmd_vel_history=MagicMock(return_value=[{"source": "nav2", "ros_vx_mps": 0.5}]),
        get_pose_in_map=MagicMock(return_value=SimpleNamespace(x=1.0, y=0.0, theta=0.0)),
        _pose_dict=BridgeNode._pose_dict,
    )
    status = BridgeNode.nav_status(bridge)
    assert status["active"] is True
    assert status["last_cmd_vel"]["viam_linear_y_mm_s"] == 300.0
    assert status["cmd_vel_history"][0]["ros_vx_mps"] == 0.5
    assert status["goal"] == {"x": 3.0, "y": -1.0, "theta": 0.5}
    assert status["pose"] == {"x": 1.0, "y": 0.0, "theta": 0.0}


def test_on_drive_timer_noop_when_nav_inactive():
    lock = __import__("threading").Lock()
    io = SimpleNamespace(drive_base=MagicMock())
    bridge = SimpleNamespace(
        _nav_active=False,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=(0.3, 0.0, 0.4),
        _io=io,
        _run=MagicMock(),
        record_cmd_vel=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._on_drive_timer(bridge)

    bridge._run.assert_not_called()


def test_on_drive_timer_does_not_apply_simple_stiction_floor_to_nav2():
    lock = __import__("threading").Lock()
    io = SimpleNamespace(drive_base=MagicMock(return_value="coro"))
    nav_cfg = SimpleNamespace(
        min_cmd_vel_x=0.2,
        min_cmd_vel_theta=0.4,
        max_vel_x=0.75,
        max_vel_theta=1.2,
    )
    bridge = SimpleNamespace(
        _nav_active=True,
        _nav_cfg=nav_cfg,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=(0.05, 0.0, 0.1),
        _io=io,
        _run=MagicMock(),
        record_cmd_vel=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._on_drive_timer(bridge)

    bridge._run.assert_called_once_with("coro")
    bridge.record_cmd_vel.assert_called_once_with(0.05, 0.0, 0.1, source="nav2")
    # These floors remain available to simple go_to_* but must not distort
    # Nav2's independently optimized linear/angular command or curvature.
    io.drive_base.assert_called_once_with(0.05, 0.0, 0.1, record_source=None)


def test_guarded_callback_swallows_exception_and_logs():
    logger = MagicMock()
    bridge = SimpleNamespace(
        get_logger=MagicMock(return_value=logger),
        _closing=False,
    )

    def boom():
        raise RuntimeError("kaput")

    wrapped = BridgeNode._guarded(bridge, boom)
    wrapped()  # must not raise

    logger.error.assert_called_once()
    assert "kaput" in logger.error.call_args.args[0]


def test_guarded_callback_passes_arguments():
    bridge = SimpleNamespace(
        get_logger=MagicMock(return_value=MagicMock()),
        _closing=False,
    )
    seen = []

    wrapped = BridgeNode._guarded(bridge, seen.append)
    wrapped("msg")

    assert seen == ["msg"]


def test_guarded_callback_skips_when_closing():
    bridge = SimpleNamespace(
        get_logger=MagicMock(return_value=MagicMock()),
        _closing=True,
    )
    called = []

    wrapped = BridgeNode._guarded(bridge, called.append)
    wrapped("msg")

    assert called == []

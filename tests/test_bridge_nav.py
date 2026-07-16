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
        _wait_for_map_tf=MagicMock(return_value=True),
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


def test_send_nav_goal_fails_without_map_tf():
    bridge = _nav_bridge_stub()
    bridge._wait_for_map_tf = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="map->base_link transform not available"):
        BridgeNode.send_nav_goal(bridge, 1.0, 2.0, 0.5)


def test_send_nav_goal_proceeds_on_stale_tf_when_previously_localized():
    from src.ros import conversions as conv

    bridge = _nav_bridge_stub()
    bridge._wait_for_map_tf = MagicMock(return_value=False)
    bridge._last_pose_in_map = conv.Pose2D(1.0, 2.0, 0.0)
    bridge._ensure_nav_action_client = MagicMock()
    bridge._wait_for_rclpy_action_server = MagicMock(return_value=True)
    bridge._cancel_inflight_nav = MagicMock()
    bridge._publish_nav_goal = MagicMock(return_value=True)

    ok = BridgeNode.send_nav_goal(bridge, 1.0, 2.0, 0.5)

    assert ok is True
    bridge._publish_nav_goal.assert_called_once_with(1.0, 2.0, 0.5)


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
        get_logger=MagicMock(return_value=MagicMock()),
    )
    bridge._apply_cmd_vel_floor = lambda vx, vy, vt: BridgeNode._apply_cmd_vel_floor(
        bridge, vx, vy, vt
    )

    BridgeNode._on_drive_timer(bridge)

    io.drive_base.assert_called_once_with(0.3, 0.0, 0.4)
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
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._on_drive_timer(bridge)

    io.drive_base.assert_called_once_with(0.0, 0.0, 0.0)


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
    bridge = SimpleNamespace(
        _nav_active=True,
        _slam_cfg=SimpleNamespace(base_velocity_convention="viam"),
        _last_cmd_vel_wall=0.0,
        _last_cmd_vel={},
    )
    BridgeNode.record_cmd_vel(bridge, 0.5, 0.0, 0.2, source="nav2")
    cmd = bridge._last_cmd_vel
    assert cmd["source"] == "nav2"
    assert cmd["ros_vx_mps"] == 0.5
    assert cmd["viam_linear_x_mm_s"] == 0.0
    assert cmd["viam_linear_y_mm_s"] == 500.0
    assert cmd["viam_angular_z_deg_s"] == pytest.approx(math.degrees(0.2), rel=1e-3)
    assert cmd["convention"] == "viam"


def test_nav_status_includes_last_cmd_vel():
    bridge = SimpleNamespace(
        _last_result_status="active",
        _nav_active=True,
        _last_feedback={"distance_remaining": 1.2},
        last_cmd_vel=MagicMock(
            return_value={
                "source": "nav2",
                "viam_linear_y_mm_s": 300.0,
                "age_s": 0.1,
            }
        ),
    )
    status = BridgeNode.nav_status(bridge)
    assert status["active"] is True
    assert status["last_cmd_vel"]["viam_linear_y_mm_s"] == 300.0


def test_on_drive_timer_noop_when_nav_inactive():
    lock = __import__("threading").Lock()
    io = SimpleNamespace(drive_base=MagicMock())
    bridge = SimpleNamespace(
        _nav_active=False,
        _cmd_vel_lock=lock,
        _pending_cmd_vel=(0.3, 0.0, 0.4),
        _io=io,
        _run=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    BridgeNode._on_drive_timer(bridge)

    bridge._run.assert_not_called()


def test_on_drive_timer_applies_stiction_floor():
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
        get_logger=MagicMock(return_value=MagicMock()),
    )
    bridge._apply_cmd_vel_floor = lambda vx, vy, vth: BridgeNode._apply_cmd_vel_floor(
        bridge, vx, vy, vth
    )

    BridgeNode._on_drive_timer(bridge)

    bridge._run.assert_called_once_with("coro")
    io.drive_base.assert_called_once_with(0.2, 0.0, 0.4)


def test_guarded_callback_swallows_exception_and_logs():
    logger = MagicMock()
    bridge = SimpleNamespace(get_logger=MagicMock(return_value=logger))

    def boom():
        raise RuntimeError("kaput")

    wrapped = BridgeNode._guarded(bridge, boom)
    wrapped()  # must not raise

    logger.error.assert_called_once()
    assert "kaput" in logger.error.call_args.args[0]


def test_guarded_callback_passes_arguments():
    bridge = SimpleNamespace(get_logger=MagicMock(return_value=MagicMock()))
    seen = []

    wrapped = BridgeNode._guarded(bridge, seen.append)
    wrapped("msg")

    assert seen == ["msg"]

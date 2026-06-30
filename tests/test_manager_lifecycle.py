from unittest.mock import MagicMock, patch

import pytest

from src.config import NavConfig, SlamConfig
from src.ros import conversions as conv
from src.ros.manager import RosManager, SLAM_LIFECYCLE_NODE


def _manager() -> RosManager:
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "maps_dir": "/tmp/maps"})
    return RosManager(cfg, logger=MagicMock())


def test_lifecycle_get_state_parses_primary_state():
    mgr = _manager()
    with patch.object(mgr, "_run_ros") as run:
        run.return_value = MagicMock(returncode=0, stdout="active [3]")
        assert mgr._lifecycle_get_state(SLAM_LIFECYCLE_NODE) == "active"
        run.return_value = MagicMock(returncode=0, stdout="inactive [2]")
        assert mgr._lifecycle_get_state(SLAM_LIFECYCLE_NODE) == "inactive"


def test_lifecycle_get_state_ignores_transitional_labels():
    mgr = _manager()
    with patch.object(mgr, "_run_ros") as run:
        run.return_value = MagicMock(returncode=0, stdout="deactivating [4]")
        assert mgr._lifecycle_get_state(SLAM_LIFECYCLE_NODE) is None
        run.return_value = MagicMock(returncode=0, stdout="activating [4]")
        assert mgr._lifecycle_get_state(SLAM_LIFECYCLE_NODE) is None


def test_activate_slam_lifecycle_configure_and_activate():
    mgr = _manager()
    mgr._slam_proc = MagicMock()
    mgr._slam_proc.poll.return_value = None

    with patch.object(mgr, "_wait_for_ros_node", return_value=True), patch.object(
        mgr,
        "_lifecycle_get_state",
        side_effect=["unconfigured", "inactive", "active"],
    ), patch.object(mgr, "_lifecycle_set", side_effect=[True, True]) as set_mock:
        mgr._activate_slam_lifecycle()

    set_mock.assert_any_call(SLAM_LIFECYCLE_NODE, "configure")
    set_mock.assert_any_call(SLAM_LIFECYCLE_NODE, "activate")


def test_activate_slam_lifecycle_skips_non_lifecycle_node():
    mgr = _manager()
    mgr._slam_proc = MagicMock()
    mgr._slam_proc.poll.return_value = None

    with patch.object(mgr, "_wait_for_ros_node", return_value=True), patch.object(
        mgr, "_lifecycle_get_state", return_value=None
    ), patch.object(mgr, "_lifecycle_set") as set_mock:
        mgr._activate_slam_lifecycle()

    set_mock.assert_not_called()


def test_activate_slam_lifecycle_raises_when_node_missing():
    mgr = _manager()
    with patch.object(mgr, "_wait_for_ros_node", return_value=False):
        with pytest.raises(RuntimeError, match="did not register"):
            mgr._activate_slam_lifecycle()


def test_set_initial_pose_delegates_to_bridge_node():
    mgr = _manager()
    mgr._node = MagicMock()

    with patch.object(mgr, "_run_ros") as run:
        mgr.set_initial_pose(conv.Pose2D(1.0, 2.0, 0.3))

    run.assert_called_once()
    mgr._node.set_initial_pose.assert_called_once()


def test_navigate_retries_after_ensuring_nav2_when_action_unavailable():
    mgr = _manager()
    mgr._node = MagicMock()
    mgr._node.send_nav_goal.side_effect = [
        RuntimeError("Nav2 action server not available"),
        True,
    ]
    mgr._nav_cfg = MagicMock()
    mgr._nav_params_path = MagicMock()

    with patch.object(mgr, "ensure_nav2") as ensure:
        mgr.navigate(1.0, 2.0, 0.5)

    ensure.assert_called_once_with(mgr._nav_cfg, mgr._nav_params_path)
    assert mgr._node.send_nav_goal.call_count == 2


def test_start_nav2_disables_collision_monitor_launch_arg(tmp_path):
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "maps_dir": str(tmp_path)})
    mgr = RosManager(cfg, logger=MagicMock())
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    params_path = tmp_path / "nav2_params.yaml"
    params_path.write_text("{}", encoding="utf-8")

    proc = MagicMock()
    proc.poll.return_value = None
    with patch.object(mgr, "stop_nav2"), patch.object(
        mgr, "_popen", return_value=proc
    ) as popen, patch.object(mgr, "_wait_for_nav_action", return_value=False), patch.object(
        mgr, "_run_ros", return_value=MagicMock(returncode=0, stdout="", stderr="")
    ):
        mgr.start_nav2(nav_cfg, params_path)

    launch_args = popen.call_args_list[0].args[0]
    assert "autostart:=false" in launch_args
    assert "use_collision_monitor:=False" in launch_args
    assert any(
        "__node:=navigation_lifecycle_manager_override" in call.args[0]
        for call in popen.call_args_list
    )

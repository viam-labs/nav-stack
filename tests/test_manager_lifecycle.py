from unittest.mock import MagicMock, patch

import pytest

from src.config import SlamConfig
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

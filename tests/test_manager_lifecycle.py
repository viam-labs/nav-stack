import threading
from unittest.mock import MagicMock, patch

import pytest

from src.config import NavConfig, SlamConfig
from src.ros import conversions as conv
from src.ros.manager import RosManager, SLAM_LIFECYCLE_NODE


def _manager() -> RosManager:
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "maps_dir": "/tmp/maps"})
    return RosManager(cfg, logger=MagicMock())


def test_optimize_pose_graph_serializes_then_deserializes_at_pose(tmp_path):
    mgr = _manager()
    mgr._slam_proc = MagicMock()
    mgr._slam_proc.poll.return_value = None
    stem = tmp_path / "map"
    calls = []

    def _run(args, timeout=5.0):
        calls.append(args)
        return MagicMock(returncode=0, stdout="ok", stderr="")

    with patch.object(mgr, "slam_running", return_value=True), patch.object(
        mgr, "get_pose_in_map", return_value=conv.Pose2D(1.5, -2.0, 0.25)
    ), patch.object(mgr, "_toggle_slam_pause", side_effect=[True, True]) as pause, patch.object(
        mgr, "_run_ros", side_effect=_run
    ):
        result = mgr.optimize_pose_graph(stem)

    assert result["ok"] is True
    assert result["status"] == "optimized"
    assert result["match_type"] == 2
    assert result["seed_pose"]["x"] == pytest.approx(1.5)
    assert pause.call_count == 2  # pause then unpause
    joined = [" ".join(c) for c in calls]
    assert any("serialize_map" in j for j in joined)
    assert any("deserialize_map" in j for j in joined)
    des = next(c for c in calls if any("deserialize_map" in a for a in c))
    payload = des[-1]
    assert "match_type: 2" in payload
    assert "1.500000" in payload


def test_optimize_pose_graph_falls_back_to_first_node_without_pose(tmp_path):
    mgr = _manager()
    with patch.object(mgr, "slam_running", return_value=True), patch.object(
        mgr, "get_pose_in_map", return_value=None
    ), patch.object(mgr, "_toggle_slam_pause", return_value=False), patch.object(
        mgr, "_run_ros", return_value=MagicMock(returncode=0, stdout="", stderr="")
    ) as run:
        result = mgr.optimize_pose_graph(tmp_path / "map")
    assert result["match_type"] == 1
    des = run.call_args_list[-1].args[0]
    assert "match_type: 1" in des[-1]


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

    assert set_mock.call_args_list[0].args[:2] == (SLAM_LIFECYCLE_NODE, "configure")
    assert set_mock.call_args_list[1].args[:2] == (SLAM_LIFECYCLE_NODE, "activate")


def test_activate_slam_lifecycle_polls_active_after_activate_cli_timeout():
    """Map load can outlive ros2 lifecycle set; keep polling for active."""
    mgr = _manager()
    mgr._slam_proc = MagicMock()
    mgr._slam_proc.poll.return_value = None
    activated = {"n": 0}

    def _set(node, transition, *, timeout=5.0):
        activated["n"] += 1
        return False  # CLI timed out, but transition may still finish

    def _state(_node):
        return "active" if activated["n"] else "inactive"

    with patch.object(mgr, "_wait_for_ros_node", return_value=True), patch.object(
        mgr, "_lifecycle_get_state", side_effect=_state
    ), patch.object(mgr, "_lifecycle_set", side_effect=_set) as set_mock, patch(
        "src.ros.manager.time.sleep"
    ):
        mgr._activate_slam_lifecycle(timeout=30.0)

    assert set_mock.call_count == 1
    assert set_mock.call_args.kwargs.get("timeout", 0) >= 20.0


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


def test_slam_params_use_small_transform_timeout(tmp_path):
    mgr = _manager()
    params = mgr._slam_params(tmp_path / "map", "localizing")
    assert params["slam_toolbox"]["ros__parameters"]["transform_timeout"] == 0.0


def test_set_initial_pose_delegates_to_bridge_node():
    mgr = _manager()
    mgr._node = MagicMock()

    with patch.object(mgr, "_run_ros") as run:
        mgr.set_initial_pose(conv.Pose2D(1.0, 2.0, 0.3))

    run.assert_called_once()
    mgr._node.set_initial_pose.assert_called_once()


def test_navigate_cancels_prior_goal():
    mgr = _manager()
    mgr._node = MagicMock()
    mgr._node.send_nav_goal.return_value = True
    with patch.object(mgr, "nav_action_ready", return_value=True):
        mgr.navigate(1.0, 2.0, 0.5)
    mgr._node.cancel_nav.assert_called_once()


def test_navigate_retries_after_ensuring_nav2_when_action_unavailable():
    mgr = _manager()
    mgr._node = MagicMock()
    mgr._node.send_nav_goal.side_effect = [
        RuntimeError("Nav2 action server not available"),
        True,
    ]
    mgr._nav_cfg = MagicMock()
    mgr._nav_params_path = MagicMock()

    with patch.object(mgr, "ensure_nav2") as ensure, patch.object(
        mgr, "nav_action_ready", return_value=True
    ):
        mgr.navigate(1.0, 2.0, 0.5)

    mgr._node.cancel_nav.assert_called_once()
    ensure.assert_called_once_with(mgr._nav_cfg, mgr._nav_params_path)
    assert mgr._node.send_nav_goal.call_count == 2


def test_compute_path_skips_cli_when_readiness_cached():
    mgr = _manager()
    mgr._node = MagicMock()
    mgr._node.compute_path_to_pose.return_value = {"feasible": True, "path": []}
    mgr._nav2_procs = [MagicMock()]
    mgr._nav2_procs[0].poll.return_value = None
    mgr._nav_action_ok_until = __import__("time").monotonic() + 60.0

    with patch.object(mgr, "_apply_slam_tf_params") as apply_tf, patch.object(
        mgr, "nav_action_ready"
    ) as ready, patch.object(mgr, "_run_ros") as run:
        out = mgr.compute_path(1.0, 2.0, 0.0)

    assert out["feasible"] is True
    apply_tf.assert_not_called()
    ready.assert_not_called()
    run.assert_not_called()
    mgr._node.compute_path_to_pose.assert_called_once()


def test_apply_slam_tf_params_sets_transform_timeout():
    mgr = _manager()
    mgr._slam_proc = MagicMock()
    mgr._slam_proc.poll.return_value = None
    with patch.object(mgr, "_run_ros") as run:
        mgr._apply_slam_tf_params()
    run.assert_called_once()
    assert run.call_args.args[0][-1] == "0.0"


def test_nav_action_ready_trusts_grace_window_without_cli():
    mgr = _manager()
    mgr._nav2_procs = [MagicMock()]
    mgr._nav2_procs[0].poll.return_value = None
    mgr._nav_action_ok_until = __import__("time").monotonic() + 60.0
    with patch.object(mgr, "_required_nav_nodes_present") as present, patch.object(
        mgr, "_nav_action_server_visible"
    ) as visible:
        assert mgr.nav_action_ready() is True
    present.assert_not_called()
    visible.assert_not_called()


def test_nav_action_ready_clears_cache_when_nav2_not_running():
    mgr = _manager()
    mgr._nav2_procs = []
    mgr._nav_action_ok_until = __import__("time").monotonic() + 60.0
    assert mgr.nav_action_ready() is False
    assert mgr._nav_action_ok_until == 0.0


def test_bundled_nav2_launch_file_exists():
    from src.ros.manager import _NAV2_LAUNCH

    assert _NAV2_LAUNCH.is_file()
    text = _NAV2_LAUNCH.read_text(encoding="utf-8")
    assert 'package="nav2_collision_monitor"' not in text
    assert 'name="docking_server"' not in text
    assert 'name="route_server"' not in text
    assert 'name="controller_server"' in text
    assert 'name="velocity_smoother"' in text
    assert 'name="lifecycle_manager_navigation"' in text


def test_start_nav2_rotates_previous_launch_log(tmp_path):
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "maps_dir": str(tmp_path)})
    mgr = RosManager(cfg, logger=MagicMock())
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b", "nav_backend": "nav2"})
    params_path = tmp_path / "nav2_params.yaml"
    params_path.write_text("{}", encoding="utf-8")
    log_path = tmp_path / ".runtime" / "nav2_launch.log"
    log_path.write_text("old errors", encoding="utf-8")

    proc = MagicMock()
    proc.poll.return_value = None
    with patch.object(mgr, "stop_nav2"), patch.object(
        mgr, "_popen", return_value=proc
    ), patch.object(mgr, "_wait_for_nav_action", return_value=True), patch.object(
        mgr, "_wait_for_required_nav_nodes", return_value=True
    ), patch.object(mgr, "_apply_slam_tf_params"), patch.object(
        mgr, "_start_costmap_filter_stack"
    ):
        mgr.start_nav2(nav_cfg, params_path)

    assert not log_path.exists()
    assert (tmp_path / ".runtime" / "nav2_launch.log.prev").read_text() == "old errors"


def test_wait_for_map_tf_before_nav2_returns_once_tf_available():
    mgr = _manager()
    node = MagicMock()
    node._lookup_pose_in_map.side_effect = [None, MagicMock()]
    mgr._node = node

    mgr._wait_for_map_tf_before_nav2(timeout=5.0)

    assert node._lookup_pose_in_map.call_count == 2


def test_wait_for_map_tf_before_nav2_noop_without_node():
    mgr = _manager()
    mgr._node = None
    mgr._wait_for_map_tf_before_nav2(timeout=0.1)


def test_start_nav2_uses_bundled_launch_with_autostart(tmp_path):
    from src.ros.manager import _NAV2_LAUNCH

    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "maps_dir": str(tmp_path)})
    mgr = RosManager(cfg, logger=MagicMock())
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b", "nav_backend": "nav2"})
    params_path = tmp_path / "nav2_params.yaml"
    params_path.write_text("{}", encoding="utf-8")

    proc = MagicMock()
    proc.poll.return_value = None
    with patch.object(mgr, "stop_nav2"), patch.object(
        mgr, "_popen", return_value=proc
    ) as popen, patch.object(mgr, "_wait_for_nav_action", return_value=False), patch.object(
        mgr, "_wait_for_required_nav_nodes", return_value=True
    ) as wait_nodes, patch.object(mgr, "_apply_slam_tf_params") as apply_tf, patch.object(
        mgr, "_activate_core_nav_nodes_manually", return_value=False
    ), patch.object(mgr, "_start_costmap_filter_stack"):
        mgr.start_nav2(nav_cfg, params_path)

    wait_nodes.assert_called_once()
    apply_tf.assert_called_once()

    launch_args = popen.call_args_list[0].args[0]
    assert launch_args[0:3] == ["ros2", "launch", str(_NAV2_LAUNCH)]
    assert "autostart:=true" in launch_args
    assert f"params_file:={params_path}" in launch_args
    # No stock-bringup kill/override dance.
    assert not any(
        "__node:=navigation_lifecycle_manager_override" in call.args[0]
        for call in popen.call_args_list
    )


def test_lifecycle_manager_params_disable_bond_timeout(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "maps_dir": str(tmp_path)})
    mgr = RosManager(cfg, logger=MagicMock())
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b", "nav_backend": "nav2"})
    params_path = tmp_path / "nav2_params.yaml"
    params_path.write_text("{}", encoding="utf-8")

    proc = MagicMock()
    proc.poll.return_value = None
    nodes = (
        "/controller_server\n/bt_navigator\n"
        "/costmap_filter_info_server_keepout\n"
        "/costmap_filter_info_server_speed\n"
    )
    with patch.object(mgr, "stop_nav2"), patch.object(
        mgr, "_popen", return_value=proc
    ), patch.object(mgr, "_wait_for_nav_action", return_value=True), patch.object(
        mgr, "_wait_for_required_nav_nodes", return_value=True
    ), patch.object(mgr, "_apply_slam_tf_params"), patch.object(
        mgr,
        "_run_ros",
        return_value=MagicMock(returncode=0, stdout=nodes, stderr=""),
    ), patch("time.sleep"):
        mgr.start_nav2(nav_cfg, params_path)

    scratch = tmp_path / ".runtime"
    filter_params = yaml.safe_load((scratch / "filter_lifecycle.yaml").read_text())
    assert (
        filter_params["filter_lifecycle_manager"]["ros__parameters"]["bond_timeout"]
        == 0.0
    )
    # Navigation LM lives in the bundled launch now (no override yaml).
    assert not (scratch / "nav_lifecycle.yaml").exists()


def test_ensure_nav2_returns_early_when_params_unchanged(tmp_path):
    mgr = _manager()
    params_path = tmp_path / "nav2_params.yaml"
    params_path.write_text("a: 1", encoding="utf-8")
    mgr._nav2_params_sig = mgr._params_file_sig(params_path)
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b", "nav_backend": "nav2"})

    with patch.object(mgr, "nav_action_ready", return_value=True), patch.object(
        mgr, "stop_nav2"
    ) as stop, patch.object(mgr, "start_nav2") as start:
        mgr.ensure_nav2(nav_cfg, params_path)

    stop.assert_not_called()
    start.assert_not_called()


def test_ensure_nav2_restarts_when_params_changed(tmp_path):
    mgr = _manager()
    params_path = tmp_path / "nav2_params.yaml"
    params_path.write_text("a: 1", encoding="utf-8")
    mgr._nav2_params_sig = "stale-signature-from-old-params"
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b", "nav_backend": "nav2"})

    with patch.object(mgr, "nav_action_ready", return_value=True), patch.object(
        mgr, "stop_nav2"
    ) as stop, patch.object(mgr, "nav2_running", return_value=False), patch.object(
        mgr, "start_nav2"
    ) as start, patch.object(mgr, "wait_for_nav_action", return_value=True):
        mgr.ensure_nav2(nav_cfg, params_path)

    stop.assert_called_once()
    start.assert_called_once_with(nav_cfg, params_path)


def test_ensure_nav2_retries_three_times_then_raises():
    mgr = _manager()
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b", "nav_backend": "nav2"})
    params_path = MagicMock()
    with patch.object(mgr, "nav_action_ready", return_value=False), patch.object(
        mgr, "nav2_running", return_value=False
    ), patch.object(mgr, "start_nav2") as start, patch.object(
        mgr, "wait_for_nav_action", return_value=False
    ), patch.object(mgr, "stop_nav2") as stop, patch.object(
        mgr, "nav2_diagnostics", return_value={"missing_core_nodes": ["controller_server"]}
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            mgr.ensure_nav2(nav_cfg, params_path)
    assert start.call_count == 3
    assert stop.call_count == 3


def test_nav_action_ready_requires_active_lifecycle_nodes():
    mgr = _manager()
    proc = MagicMock()
    proc.poll.return_value = None
    mgr._nav2_procs = [proc]

    with patch.object(mgr, "_nav_action_server_visible", return_value=True), \
        patch.object(mgr, "_required_nav_nodes_present", return_value=True), \
        patch.object(mgr, "_lifecycle_get_state", return_value="inactive"):
        assert mgr.nav_action_ready() is False

    with patch.object(mgr, "_nav_action_server_visible", return_value=True), \
        patch.object(mgr, "_required_nav_nodes_present", return_value=True), \
        patch.object(mgr, "_lifecycle_get_state", return_value="active"):
        assert mgr.nav_action_ready() is True


def test_ensure_nav2_async_runs_in_background_and_deduplicates():

    mgr = _manager()
    started = threading.Event()
    release = threading.Event()

    def _slow_ensure(cfg, path):
        started.set()
        release.wait(timeout=5.0)

    nav_cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "nav_backend": "nav2"}
    )
    params_path = MagicMock()
    with patch.object(mgr, "ensure_nav2", side_effect=_slow_ensure) as ensure:
        mgr.ensure_nav2_async(nav_cfg, params_path)
        assert started.wait(timeout=2.0)
        assert mgr.nav2_startup_in_progress() is True
        # Duplicate request while running is a no-op.
        mgr.ensure_nav2_async(nav_cfg, params_path)
        release.set()
        mgr._nav2_ensure_thread.join(timeout=2.0)

    assert ensure.call_count == 1
    assert mgr.nav2_startup_in_progress() is False
    assert mgr._nav_cfg is nav_cfg
    assert mgr._nav_params_path is params_path


def test_ensure_nav2_async_swallows_background_failure():
    mgr = _manager()
    nav_cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "nav_backend": "nav2"}
    )
    with patch.object(mgr, "ensure_nav2", side_effect=RuntimeError("boom")):
        mgr.ensure_nav2_async(nav_cfg, MagicMock())
        mgr._nav2_ensure_thread.join(timeout=2.0)
    assert mgr.nav2_startup_in_progress() is False


def test_ensure_nav2_async_skips_for_builtin_backend():
    mgr = _manager()
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    assert nav_cfg.uses_builtin_nav()
    with patch.object(mgr, "ensure_nav2") as ensure:
        mgr.ensure_nav2_async(nav_cfg, MagicMock())
    ensure.assert_not_called()
    assert mgr._builtin_nav is not None
    assert mgr.nav2_startup_in_progress() is False


def test_navigate_uses_builtin_when_configured():
    mgr = _manager()
    nav_cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    mgr.set_nav_config(nav_cfg)
    assert mgr._builtin_nav is not None
    with patch.object(mgr._builtin_nav, "navigate") as nav:
        mgr.navigate(1.0, 2.0, 0.3)
    nav.assert_called_once_with(1.0, 2.0, 0.3)


def test_navigate_rejects_while_nav2_startup_in_progress():
    mgr = _manager()
    mgr._node = MagicMock()
    mgr._builtin_nav = None  # force Nav2 path
    with patch.object(mgr, "nav2_startup_in_progress", return_value=True):
        with pytest.raises(RuntimeError, match="still starting up"):
            mgr.navigate(1.0, 2.0, 0.5)
    mgr._node.send_nav_goal.assert_not_called()

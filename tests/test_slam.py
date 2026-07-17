from pathlib import Path
import asyncio
import math
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import MODE_LOCALIZING, MODE_MAPPING, SlamConfig

# Stub ROS 2 Python deps so model tests run without a ROS install.
for _mod in (
    "rclpy",
    "rclpy.node",
    "rclpy.qos",
    "rclpy.action",
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

pytest.importorskip("viam")

from src.models.slam import RosSlam
from src.nav.maps import MapStore
from src.ros import conversions as conv


def test_get_status_includes_bridge_and_sensor_probe(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = SlamConfig.from_dict(
        {
            "base": "cartbase",
            "movement_sensor": "imu",
            "lidar": {
                "name": "livox-pc",
                "scan_source": "point_cloud",
                "mount": {"x": 0.5, "y": 0.0, "z": 1.1, "theta": 0.0},
            },
            "mode": "mapping",
            "active_map": "floor1",
        }
    )
    slam._manager = MagicMock()
    slam._manager.slam_diagnostics.return_value = {
        "slam_toolbox_running": True,
        "scan_publishing": True,
        "scan_valid_returns": 120,
        "odom_tf_age_s": 0.1,
    }
    slam._probe_sensors = AsyncMock(
        return_value={
            "lidars": [{"name": "livox-pc", "scan_valid_returns": 120}],
            "odometry": {"vx": 0.0, "vy": 0.0, "vtheta": 0.0, "has_pose": False},
        }
    )

    result = asyncio.run(slam.do_command({"command": "get_status"}))

    assert result["slam_toolbox_running"] is True
    assert result["active_map"] == "floor1"
    assert result["movement_sensor"] == "imu"
    assert result["sensor_probe"]["lidars"][0]["scan_valid_returns"] == 120
    slam._probe_sensors.assert_awaited_once()


def test_get_status_skips_sensor_probe_when_disabled(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    slam._map_store = store
    slam._cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "mode": "mapping", "active_map": "floor1"}
    )
    slam._manager = MagicMock()
    slam._manager.slam_diagnostics.return_value = {"slam_toolbox_running": True}
    slam._probe_sensors = AsyncMock()

    asyncio.run(slam.do_command({"command": "get_status", "probe_sensors": False}))

    slam._probe_sensors.assert_not_awaited()


def test_resolve_pose_by_location_requires_active_map(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    with pytest.raises(RuntimeError, match="no active map"):
        slam._resolve_pose({"location": "kitchen"})


def test_resolve_pose_by_location_uses_active_map(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store

    from src.nav.locations import LocationStore

    handle = store.active_handle()
    LocationStore(handle.locations_path).add("kitchen", 1.0, 2.0, 0.5)

    pose = slam._resolve_pose({"location": "kitchen"})
    assert pose.x == 1.0
    assert pose.y == 2.0
    assert pose.theta == 0.5


def test_resolve_pose_explicit_pose():
    slam = RosSlam("slam")
    pose = slam._resolve_pose({"pose": {"x": 3.0, "y": 4.0, "theta": 1.0}})
    assert pose.x == 3.0
    assert pose.y == 4.0
    assert pose.theta == 1.0


def test_delete_active_map_clears_live_slam(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, slam_toolbox=MagicMock(resolution=0.05))

    node = MagicMock()
    slam._manager = MagicMock(node=node)
    slam._reset_live_slam = MagicMock()

    result = asyncio.run(slam.do_command({"command": "delete_map", "map": "floor1"}))

    assert result == {
        "status": "deleted",
        "map": "floor1",
        "active_map": "floor1",
        "mode": MODE_MAPPING,
    }
    assert store.get_active_map_name() == "floor1"
    assert store.handle("floor1").exists()
    slam._reset_live_slam.assert_called_once_with(MODE_MAPPING)
    assert slam._cfg.active_map == "floor1"


def test_delete_inactive_map_does_not_restart_slam(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.create_map("floor2")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, slam_toolbox=MagicMock(resolution=0.05))

    node = MagicMock()
    slam._manager = MagicMock(node=node)
    slam._reset_live_slam = MagicMock()

    result = asyncio.run(slam.do_command({"command": "delete_map", "map": "floor2"}))

    assert result == {
        "status": "deleted",
        "map": "floor2",
        "active_map": "floor1",
        "mode": MODE_MAPPING,
    }
    assert store.get_active_map_name() == "floor1"
    slam._reset_live_slam.assert_not_called()


def test_delete_live_map_resets_when_configured_active_without_store_active(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("config-map")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, active_map="config-map", slam_toolbox=MagicMock(resolution=0.05))
    slam._manager = MagicMock()
    slam._reset_live_slam = MagicMock()

    result = asyncio.run(slam.do_command({"command": "delete_map", "map": "config-map"}))

    assert result == {
        "status": "deleted",
        "map": "config-map",
        "active_map": "config-map",
        "mode": MODE_MAPPING,
    }
    slam._reset_live_slam.assert_called_once_with(MODE_MAPPING)


def test_delete_configured_name_does_not_reset_other_active_map(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.create_map("config-map")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, active_map="config-map", slam_toolbox=MagicMock(resolution=0.05))
    slam._manager = MagicMock()
    slam._reset_live_slam = MagicMock()

    result = asyncio.run(slam.do_command({"command": "delete_map", "map": "config-map"}))

    assert result == {
        "status": "deleted",
        "map": "config-map",
        "active_map": "floor1",
        "mode": MODE_MAPPING,
    }
    slam._reset_live_slam.assert_not_called()


def test_clear_map_resets_live_slam(tmp_path: Path):
    slam = RosSlam("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._manager = MagicMock()
    slam._reset_live_slam = MagicMock()

    result = asyncio.run(slam.do_command({"command": "clear_map"}))

    assert result == {"status": "cleared", "map": "floor1", "mode": MODE_MAPPING}
    slam._reset_live_slam.assert_called_once_with(MODE_MAPPING)


def test_clear_map_requires_active_map(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._manager = MagicMock()
    with pytest.raises(ValueError, match="no active map"):
        asyncio.run(slam.do_command({"command": "clear_map"}))


def test_relocalize_uses_current_map_pose(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_LOCALIZING)
    mgr = MagicMock()
    mgr.get_pose_in_map.return_value = conv.Pose2D(1.0, 2.0, 0.5)
    slam._manager = mgr

    result = asyncio.run(slam.do_command({"command": "relocalize"}))

    assert result == {
        "status": "relocalizing",
        "seed_pose": {"x": 1.0, "y": 2.0, "theta": 0.5},
    }
    mgr.relocalize.assert_called_once()
    pose_arg = mgr.relocalize.call_args.args[0]
    assert pose_arg.x == 1.0
    assert pose_arg.y == 2.0
    assert pose_arg.theta == 0.5


def test_set_initial_pose_refine_runs_seeded_yaw_search(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_LOCALIZING)
    mgr = MagicMock()
    slam._manager = mgr

    refine_result = {"status": "localized", "score": 0.8}

    async def _fake_global_localize(command):
        _fake_global_localize.command = dict(command)
        return refine_result

    slam._global_localize = _fake_global_localize

    result = asyncio.run(
        slam.do_command(
            {
                "command": "set_initial_pose",
                "pose": {"x": 1.0, "y": 2.0, "theta": 0.5},
                "refine": True,
            }
        )
    )

    mgr.set_initial_pose.assert_called_once()
    assert result["status"] == "ok"
    assert result["refine"] == refine_result
    sent = _fake_global_localize.command
    assert sent["pose"] == {"x": 1.0, "y": 2.0, "theta": 0.5}
    assert sent["local_yaw_window_deg"] == 360.0
    assert sent["full_map"] is False
    assert sent["auto_full_map_fallback"] is False


def test_set_initial_pose_without_refine_skips_search(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_LOCALIZING)
    mgr = MagicMock()
    slam._manager = mgr
    slam._global_localize = AsyncMock()

    result = asyncio.run(
        slam.do_command(
            {"command": "set_initial_pose", "pose": {"x": 1.0, "y": 2.0, "theta": 0.5}}
        )
    )

    assert result == {"status": "ok"}
    slam._global_localize.assert_not_called()


def test_relocalize_use_mir_pose_from_movement_sensor(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_LOCALIZING)
    slam._manager = MagicMock()
    movement = MagicMock()

    async def _readings():
        return {
            "position_x_m": 3.0,
            "position_y_m": 4.0,
            "yaw_deg": 90.0,
        }

    movement.get_readings = _readings
    slam._movement_sensor = movement

    result = asyncio.run(
        slam.do_command({"command": "relocalize", "use_mir_pose": True})
    )

    assert result["seed_pose"]["x"] == 3.0
    assert result["seed_pose"]["y"] == 4.0
    assert result["seed_pose"]["theta"] == pytest.approx(math.pi / 2)
    slam._manager.relocalize.assert_called_once()


def test_relocalize_requires_localizing_mode(tmp_path: Path):
    slam = RosSlam("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_MAPPING)
    slam._manager = MagicMock()
    with pytest.raises(ValueError, match="localizing"):
        asyncio.run(slam.do_command({"command": "relocalize"}))


def test_schedule_startup_global_localize_skips_when_disabled():
    slam = RosSlam("slam")
    slam._cfg = MagicMock(mode=MODE_LOCALIZING, global_localize_on_start=False)
    loop = MagicMock()

    slam._schedule_startup_global_localize(loop)

    loop.create_task.assert_not_called()


def test_run_startup_global_localize_retries_then_succeeds():
    slam = RosSlam("slam")
    slam.do_command = AsyncMock(
        side_effect=[
            RuntimeError("slam not ready"),
            {
                "status": "matched",
                "score": 0.7,
                "ray_mae_m": 0.4,
                "pose": {"x": 1.0, "y": 2.0, "theta": 0.3},
            },
            {"status": "relocalizing"},
        ]
    )

    asyncio.run(
        slam._run_startup_global_localize(
            {"full_map": True},
            delay_s=0.0,
            max_attempts=2,
            retry_delay_s=0.0,
            run_post_apply_refine=False,
        )
    )

    assert slam.do_command.await_count == 3
    first_cmd = slam.do_command.await_args_list[0].args[0]
    second_cmd = slam.do_command.await_args_list[1].args[0]
    third_cmd = slam.do_command.await_args_list[2].args[0]
    assert first_cmd["command"] == "global_localize"
    assert first_cmd["apply"] is False
    assert first_cmd["full_map"] is True
    assert second_cmd["command"] == "global_localize"
    assert second_cmd["apply"] is False
    assert third_cmd["command"] == "relocalize"
    assert third_cmd["pose"]["x"] == pytest.approx(1.0)


def test_run_startup_global_localize_runs_refinement_pass():
    slam = RosSlam("slam")
    slam.do_command = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.52,
                "ray_mae_m": 0.9,
                "pose": {"x": 1.0, "y": 2.0, "theta": 0.1},
            },
            {
                "status": "matched",
                "score": 0.71,
                "ray_mae_m": 0.35,
                "pose": {"x": 3.0, "y": 4.0, "theta": 0.2},
            },
            {"status": "relocalizing"},
        ]
    )

    asyncio.run(
        slam._run_startup_global_localize(
            {"full_map": True},
            delay_s=0.0,
            max_attempts=1,
            retry_delay_s=0.0,
            run_refine_pass=True,
            refine_delay_s=0.0,
            refine_max_passes=1,
            target_score=0.95,
            target_ray_mae_m=0.2,
            refine_options={"local_yaw_window_deg": 90.0},
            run_post_apply_refine=False,
        )
    )

    assert slam.do_command.await_count == 3
    first_cmd = slam.do_command.await_args_list[0].args[0]
    second_cmd = slam.do_command.await_args_list[1].args[0]
    third_cmd = slam.do_command.await_args_list[2].args[0]
    assert first_cmd["full_map"] is True
    assert first_cmd["apply"] is False
    assert second_cmd["full_map"] is False
    assert second_cmd["local_yaw_window_deg"] == 90.0
    assert second_cmd["apply"] is False
    assert second_cmd["pose"]["x"] == pytest.approx(1.0)
    assert third_cmd["command"] == "relocalize"
    assert third_cmd["pose"]["x"] == pytest.approx(3.0)


def test_run_startup_global_localize_runs_post_apply_refine_when_weak():
    slam = RosSlam("slam")
    slam.do_command = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.58,
                "ray_mae_m": 0.82,
                "pose": {"x": 0.5, "y": 1.2, "theta": 0.1},
            },
            {"status": "relocalizing"},
            {
                "status": "localized",
                "score": 0.73,
                "ray_mae_m": 0.36,
                "pose": {"x": 0.7, "y": 1.0, "theta": 0.08},
            },
        ]
    )

    asyncio.run(
        slam._run_startup_global_localize(
            {"full_map": True},
            delay_s=0.0,
            max_attempts=1,
            retry_delay_s=0.0,
            run_refine_pass=False,
            run_post_apply_refine=True,
            post_apply_refine_delay_s=0.0,
            post_apply_refine_options={"map_source": "live"},
        )
    )

    assert slam.do_command.await_count == 3
    first_cmd = slam.do_command.await_args_list[0].args[0]
    second_cmd = slam.do_command.await_args_list[1].args[0]
    third_cmd = slam.do_command.await_args_list[2].args[0]
    assert first_cmd["command"] == "global_localize"
    assert first_cmd["apply"] is False
    assert second_cmd["command"] == "relocalize"
    assert third_cmd["command"] == "global_localize"
    assert third_cmd["apply"] is True
    assert third_cmd["map_source"] == "live"


def test_startup_localize_readiness_waits_for_scan_and_map():
    slam = RosSlam("slam")
    slam._manager = MagicMock()
    slam._manager.slam_running.return_value = True
    slam._read_merged_scan = AsyncMock(
        side_effect=[RuntimeError("no lidar returns"), MagicMock()]
    )
    slam._load_active_occupancy_map = MagicMock(return_value=(MagicMock(), "live"))

    ready = asyncio.run(
        slam._wait_for_startup_localize_ready(timeout_s=10.0, poll_interval_s=0.0)
    )

    assert ready is True
    assert slam._read_merged_scan.await_count == 2
    slam._load_active_occupancy_map.assert_called_once()


def test_startup_localize_readiness_times_out():
    slam = RosSlam("slam")
    slam._manager = MagicMock()
    slam._manager.slam_running.return_value = False

    ready = asyncio.run(
        slam._wait_for_startup_localize_ready(timeout_s=0.05, poll_interval_s=0.0)
    )

    assert ready is False


def test_run_startup_global_localize_skips_when_navigation_active():
    slam = RosSlam("slam")
    slam._manager = MagicMock()
    slam._manager.nav_status.return_value = {"active": True}
    slam.do_command = AsyncMock()

    asyncio.run(
        slam._run_startup_global_localize(
            {"full_map": True},
            delay_s=0.0,
            max_attempts=1,
            retry_delay_s=0.0,
        )
    )

    assert slam.do_command.await_count == 0


def test_get_point_cloud_map_hides_stale_generation():
    import numpy as np

    slam = RosSlam("slam")
    slam._visible_map_generation = 2
    grid = {
        "grid": np.ones((2, 2), dtype=np.int16) * 100,
        "resolution": 0.05,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "generation": 1,
    }
    slam._manager = MagicMock(node=MagicMock(get_map=MagicMock(return_value=grid)))

    chunks = asyncio.run(slam.get_point_cloud_map())
    assert b"POINTS 0" in chunks[0]


def test_get_point_cloud_map_shows_current_generation():
    import numpy as np

    slam = RosSlam("slam")
    slam._visible_map_generation = 2
    grid = {
        "grid": np.ones((2, 2), dtype=np.int16) * 100,
        "resolution": 0.05,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "generation": 2,
    }
    slam._manager = MagicMock(node=MagicMock(get_map=MagicMock(return_value=grid)))

    chunks = asyncio.run(slam.get_point_cloud_map())
    assert b"POINTS 4" in chunks[0]


def test_stop_base_zeros_velocity_without_full_stop():
    slam = RosSlam("slam")
    slam._cfg = MagicMock(
        base_velocity_convention="viam",
        sensor_read_timeout_s=1.0,
        lidars=[],
        movement_sensor_upside_down=False,
        movement_sensor_yaw_deg=0.0,
        heading_sensor_invert=False,
        heading_sensor_yaw_deg=0.0,
    )
    slam._base = AsyncMock()
    slam._movement_sensor = None
    slam._heading_sensor = None
    slam._cameras = {}
    slam._skip_get_laser_scan = set()
    slam._manager = MagicMock(node=None)

    io = slam._build_io()
    asyncio.run(io.stop_base())

    slam._base.stop.assert_not_called()
    slam._base.set_velocity.assert_awaited_once()
    kwargs = slam._base.set_velocity.await_args.kwargs
    assert kwargs["linear"].x == 0.0
    assert kwargs["linear"].y == 0.0
    assert kwargs["angular"].z == 0.0


def test_nav2_drive_base_sends_angular_z_to_viam_base():
    slam = RosSlam("slam")
    slam._cfg = MagicMock(
        base_velocity_convention="viam",
        sensor_read_timeout_s=1.0,
        lidars=[],
        movement_sensor_upside_down=False,
        movement_sensor_yaw_deg=0.0,
        heading_sensor_invert=False,
        heading_sensor_yaw_deg=0.0,
    )
    slam._base = AsyncMock()
    slam._movement_sensor = None
    slam._heading_sensor = None
    slam._cameras = {}
    slam._skip_get_laser_scan = set()
    node = MagicMock()
    slam._manager = MagicMock(node=node)

    io = slam._build_io()
    asyncio.run(io.drive_base(0.5, 0.0, -1.0))

    node.record_cmd_vel.assert_called_once_with(0.5, 0.0, -1.0, source="nav2")
    slam._base.set_velocity.assert_awaited_once()
    kwargs = slam._base.set_velocity.await_args.kwargs
    assert kwargs["linear"].x == 0.0
    assert kwargs["linear"].y == 500.0
    assert kwargs["angular"].z == pytest.approx(-57.2958, rel=1e-5)


# -- periodic relocalize (drift watchdog) -----------------------------------
def _relocalize_slam(**cfg_overrides):
    d = {
        "base": "b",
        "lidar": "f",
        "mode": "localizing",
        "periodic_relocalize": True,
    }
    d.update(cfg_overrides)
    slam = RosSlam("slam")
    slam._cfg = SlamConfig.from_dict(d)
    slam._manager = MagicMock()
    slam._manager.get_pose_in_map.return_value = conv.Pose2D(0.0, 0.0, 0.0)
    slam._manager.nav_status.return_value = {
        "active": False,
        "number_of_recoveries": 0,
    }
    slam._startup_global_localize_task = None
    slam._is_navigation_active = MagicMock(return_value=False)
    return slam


def test_schedule_periodic_relocalize_skips_when_disabled():
    slam = _relocalize_slam(periodic_relocalize=False)
    loop = MagicMock()
    slam._schedule_periodic_relocalize(loop)
    loop.create_task.assert_not_called()


def test_schedule_periodic_relocalize_skips_when_mapping():
    slam = RosSlam("slam")
    slam._cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "mode": "mapping", "periodic_relocalize": True}
    )
    loop = MagicMock()
    slam._schedule_periodic_relocalize(loop)
    loop.create_task.assert_not_called()


def test_schedule_periodic_relocalize_starts_when_enabled():
    slam = _relocalize_slam()
    slam._run_periodic_relocalize = MagicMock(return_value=None)  # avoid coroutine
    loop = MagicMock()
    slam._schedule_periodic_relocalize(loop)
    loop.create_task.assert_called_once()


def test_schedule_periodic_relocalize_starts_by_default_in_localizing():
    slam = RosSlam("slam")
    slam._cfg = SlamConfig.from_dict({"base": "b", "lidar": "f", "mode": "localizing"})
    slam._run_periodic_relocalize = MagicMock(return_value=None)
    loop = MagicMock()
    slam._schedule_periodic_relocalize(loop)
    loop.create_task.assert_called_once()


def test_periodic_relocalize_cycle_corrects_on_drift():
    slam = _relocalize_slam(
        periodic_relocalize_min_score=0.5,
        periodic_relocalize_min_shift_m=0.2,
    )
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.8,
            "ray_mae_m": 0.3,
            "pose": {"x": 1.0, "y": 0.0, "theta": 0.0},
        }
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "corrected"
    assert result["corrected"] is True
    assert result["shift_m"] == pytest.approx(1.0)
    relocalize_cmd = slam.do_command.await_args.args[0]
    assert relocalize_cmd["command"] == "relocalize"
    assert relocalize_cmd["pose"]["x"] == pytest.approx(1.0)


def test_periodic_relocalize_cycle_no_correction_when_close():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.9,
            "ray_mae_m": 0.2,
            "pose": {"x": 0.05, "y": 0.0, "theta": 0.0},
        }
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "ok"
    assert result["corrected"] is False
    slam.do_command.assert_not_awaited()


def test_periodic_relocalize_cycle_low_quality_no_correction():
    slam = _relocalize_slam(periodic_relocalize_min_score=0.5)
    slam._global_localize = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.2,
                "ray_mae_m": 1.5,
                "pose": {"x": 3.0, "y": 0.0, "theta": 0.0},
            },
            {
                "status": "matched",
                "score": 0.25,
                "ray_mae_m": 1.2,
                "pose": {"x": 3.0, "y": 0.0, "theta": 0.0},
            },
        ]
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "low_quality"
    assert slam._global_localize.await_count == 2
    slam.do_command.assert_not_awaited()


def test_periodic_relocalize_cycle_escalates_full_map_on_low_quality():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._global_localize = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.2,
                "ray_mae_m": 1.5,
                "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            },
            {
                "status": "matched",
                "score": 0.85,
                "ray_mae_m": 0.25,
                "pose": {"x": 2.0, "y": 0.0, "theta": 0.0},
            },
        ]
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "corrected"
    assert result["match_mode"] == "full_map_after_low_quality"
    assert slam._global_localize.await_count == 2
    full_cmd = slam._global_localize.await_args_list[1].args[0]
    assert full_cmd["full_map"] is True
    slam.do_command.assert_awaited_once()


def test_periodic_relocalize_cycle_full_map_when_nav_recoveries_high():
    slam = _relocalize_slam(periodic_relocalize_nav_recoveries_threshold=2)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._manager.nav_status.return_value = {
        "active": True,
        "number_of_recoveries": 11,
    }
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.9,
            "ray_mae_m": 0.2,
            "pose": {"x": 1.5, "y": 0.0, "theta": 0.0},
        }
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "corrected"
    assert result["match_mode"] == "full_map"
    first_cmd = slam._global_localize.await_args_list[0].args[0]
    assert first_cmd["full_map"] is True


def test_periodic_relocalize_cycle_recovery_applies_high_ray_mae():
    # Nav is failing so the watchdog forces a full-map match. The best match is
    # correct (manual global_localize applies it fine) but ray_mae is above even
    # the generous good_match gate. The score-only recovery path must still apply
    # it instead of logging low_quality forever.
    slam = _relocalize_slam(
        periodic_relocalize_min_score=0.5,
        periodic_relocalize_max_ray_mae_m=1.0,
        periodic_relocalize_recovery_min_score=0.45,
        periodic_relocalize_nav_recoveries_threshold=2,
    )
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._manager.nav_status.return_value = {
        "active": True,
        "number_of_recoveries": 23,
    }
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.61,
            "ray_mae_m": 1.30,
            "pose": {"x": 1.9, "y": 0.0, "theta": 1.38},
        }
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "corrected"
    assert result["corrected"] is True
    assert result["good_match"] is False
    assert result["recovery_apply"] is True
    assert result["match_mode"] == "full_map"
    slam.do_command.assert_awaited_once()


def test_periodic_relocalize_cycle_recovery_floor_blocks_garbage():
    # A full-map recovery match whose score is below the recovery floor is genuine
    # garbage and must not be applied.
    slam = _relocalize_slam(
        periodic_relocalize_recovery_min_score=0.45,
        periodic_relocalize_nav_recoveries_threshold=2,
    )
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._manager.nav_status.return_value = {
        "active": True,
        "number_of_recoveries": 23,
    }
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.3,
            "ray_mae_m": 1.4,
            "pose": {"x": 5.0, "y": 0.0, "theta": 0.0},
        }
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "low_quality"
    assert result["recovery_apply"] is False
    slam.do_command.assert_not_awaited()


def test_periodic_relocalize_cycle_skips_during_navigation():
    slam = _relocalize_slam(periodic_relocalize_during_navigation=False)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock()
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "skipped"
    assert result["reason"] == "navigation_active"
    slam._global_localize.assert_not_awaited()


def test_periodic_relocalize_cycle_skips_while_startup_running():
    slam = _relocalize_slam()
    pending = MagicMock()
    pending.done.return_value = False
    slam._startup_global_localize_task = pending
    slam._global_localize = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "skipped"
    assert result["reason"] == "startup_localize_running"
    slam._global_localize.assert_not_awaited()


def test_check_localization_apply_override_forces_correction():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=5.0)  # would not drift
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.3,  # low quality, but override forces apply
            "ray_mae_m": 1.2,
            "pose": {"x": 0.1, "y": 0.0, "theta": 0.0},
        }
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = asyncio.run(slam._periodic_relocalize_cycle(apply_override=True))

    assert result["corrected"] is True
    relocalize_cmd = slam.do_command.await_args.args[0]
    assert relocalize_cmd["command"] == "relocalize"

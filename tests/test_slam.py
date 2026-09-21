from pathlib import Path
import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import MODE_LOCALIZING, MODE_MAPPING, SlamConfig

pytest.importorskip("viam")

from src.models.slam import SlamService
from src.nav.maps import MapStore
from src.geom import conversions as conv


def test_get_status_includes_diagnostics_and_sensor_probe(tmp_path: Path):
    slam = SlamService("slam")
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
        "slam_backend": "builtin",
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

    assert result["slam_backend"] == "builtin"
    assert result["active_map"] == "floor1"
    assert result["movement_sensor"] == "imu"
    assert result["sensor_probe"]["lidars"][0]["scan_valid_returns"] == 120
    slam._probe_sensors.assert_awaited_once()


def test_get_status_skips_sensor_probe_when_disabled(tmp_path: Path):
    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    slam._map_store = store
    slam._cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "mode": "mapping", "active_map": "floor1"}
    )
    slam._manager = MagicMock()
    slam._manager.slam_diagnostics.return_value = {"slam_backend": "builtin"}
    slam._probe_sensors = AsyncMock()

    asyncio.run(slam.do_command({"command": "get_status", "probe_sensors": False}))

    slam._probe_sensors.assert_not_awaited()


def test_resolve_pose_by_location_requires_active_map(tmp_path: Path):
    slam = SlamService("slam")
    slam._map_store = MapStore(str(tmp_path))
    with pytest.raises(RuntimeError, match="no active map"):
        slam._resolve_pose({"location": "kitchen"})


def test_resolve_pose_by_location_uses_active_map(tmp_path: Path):
    slam = SlamService("slam")
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
    slam = SlamService("slam")
    pose = slam._resolve_pose({"pose": {"x": 3.0, "y": 4.0, "theta": 1.0}})
    assert pose.x == 3.0
    assert pose.y == 4.0
    assert pose.theta == 1.0


def test_delete_active_map_clears_live_slam(tmp_path: Path):
    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, map=MagicMock(resolution=0.05))

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
    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.create_map("floor2")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, map=MagicMock(resolution=0.05))

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
    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("config-map")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, active_map="config-map", map=MagicMock(resolution=0.05))
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
    slam = SlamService("slam")
    store = MapStore(str(tmp_path))
    store.create_map("floor1")
    store.create_map("config-map")
    store.set_active_map("floor1")
    slam._map_store = store
    slam._cfg = MagicMock(mode=MODE_MAPPING, active_map="config-map", map=MagicMock(resolution=0.05))
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
    slam = SlamService("slam")
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
    slam = SlamService("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._manager = MagicMock()
    with pytest.raises(ValueError, match="no active map"):
        asyncio.run(slam.do_command({"command": "clear_map"}))


def test_optimize_do_command_requires_mapping_mode(tmp_path: Path):
    slam = SlamService("slam")
    slam._cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "mode": MODE_LOCALIZING, "maps_dir": str(tmp_path)}
    )
    slam._map_store = MapStore(str(tmp_path))
    slam._manager = MagicMock()
    with pytest.raises(ValueError, match="mapping"):
        asyncio.run(slam.do_command({"command": "optimize"}))


def test_optimize_do_command_calls_manager(tmp_path: Path):
    slam = SlamService("slam")
    slam._cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "mode": MODE_MAPPING, "maps_dir": str(tmp_path)}
    )
    store = MapStore(str(tmp_path))
    store.create_map("floor")
    store.set_active_map("floor")
    slam._map_store = store
    slam._manager = MagicMock()
    slam._manager.optimize_pose_graph.return_value = {
        "status": "optimized",
        "ok": True,
        "match_type": 2,
    }
    result = asyncio.run(slam.do_command({"command": "optimize"}))
    assert result["status"] == "optimized"
    handle = store.active_handle()
    slam._manager.optimize_pose_graph.assert_called_once_with(handle.serialization_stem)


def test_relocalize_uses_current_map_pose(tmp_path: Path):
    slam = SlamService("slam")
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
    slam = SlamService("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_LOCALIZING)
    mgr = MagicMock()
    slam._manager = mgr
    slam._cancel_startup_global_localize_task = MagicMock()

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
    slam._cancel_startup_global_localize_task.assert_called()
    assert result["status"] == "ok"
    assert result["refine"] == refine_result
    sent = _fake_global_localize.command
    assert sent["pose"] == {"x": 1.0, "y": 2.0, "theta": 0.5}
    assert sent["local_yaw_window_deg"] == 360.0
    assert sent["search_radius_m"] == 1.0
    assert sent["full_map"] is False
    assert sent["auto_full_map_fallback"] is False
    assert sent["min_apply_score"] == 0.22
    assert sent["refuse_ambiguous"] is False
    assert sent["max_apply_ray_mae_m"] == 1.5


def test_set_initial_pose_without_refine_skips_search(tmp_path: Path):
    slam = SlamService("slam")
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
    slam = SlamService("slam")
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
    slam = SlamService("slam")
    slam._map_store = MapStore(str(tmp_path))
    slam._cfg = MagicMock(mode=MODE_MAPPING)
    slam._manager = MagicMock()
    with pytest.raises(ValueError, match="localizing"):
        asyncio.run(slam.do_command({"command": "relocalize"}))


def test_schedule_startup_global_localize_skips_when_disabled():
    slam = SlamService("slam")
    slam._cfg = MagicMock(mode=MODE_LOCALIZING, global_localize_on_start=False)
    loop = MagicMock()

    slam._schedule_startup_global_localize(loop)

    loop.create_task.assert_not_called()


def test_run_startup_global_localize_retries_then_succeeds():
    slam = SlamService("slam")
    slam.do_command = AsyncMock(
        side_effect=[
            RuntimeError("slam not ready"),
            {
                "status": "matched",
                "score": 0.7,
                "ray_mae_m": 0.4,
                "pose": {"x": 1.0, "y": 2.0, "theta": 0.3},
            },
            {
                "status": "matched",
                "score": 0.72,
                "ray_mae_m": 0.38,
                "pose": {"x": 1.05, "y": 2.02, "theta": 0.31},
            },
            {"status": "relocalizing"},
        ]
    )

    asyncio.run(
        slam._run_startup_global_localize(
            {"full_map": True},
            delay_s=0.0,
            max_attempts=3,
            retry_delay_s=0.0,
            run_post_apply_refine=False,
        )
    )

    assert slam.do_command.await_count == 4
    cmds = [c.args[0] for c in slam.do_command.await_args_list]
    assert cmds[0]["command"] == "global_localize"
    assert cmds[0]["apply"] is False
    assert cmds[1]["command"] == "global_localize"
    assert cmds[2]["command"] == "global_localize"
    assert cmds[3]["command"] == "relocalize"
    assert cmds[3]["pose"]["x"] == pytest.approx(1.05)


def test_run_startup_global_localize_runs_refinement_pass():
    from src.nav.pose_jump_gate import PoseJumpGate

    slam = SlamService("slam")
    slam._pose_jump_gate = PoseJumpGate(confirm_count=1)
    slam.do_command = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.50,
                "ray_mae_m": 0.50,
                "pose": {"x": 1.0, "y": 2.0, "theta": 0.1},
            },
            {
                "status": "matched",
                "score": 0.71,
                "ray_mae_m": 0.35,
                "pose": {"x": 1.1, "y": 2.05, "theta": 0.12},
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
    assert third_cmd["pose"]["x"] == pytest.approx(1.1)


def test_run_startup_global_localize_runs_post_apply_refine_when_weak():
    from src.nav.pose_jump_gate import PoseJumpGate

    slam = SlamService("slam")
    slam._pose_jump_gate = PoseJumpGate(confirm_count=1)
    slam.do_command = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.68,
                "ray_mae_m": 0.40,
                # Small jump applies immediately; post-apply refine still runs.
                "pose": {"x": 0.3, "y": 0.4, "theta": 0.1},
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
    slam = SlamService("slam")
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
    slam = SlamService("slam")
    slam._manager = MagicMock()
    slam._manager.slam_running.return_value = False

    ready = asyncio.run(
        slam._wait_for_startup_localize_ready(timeout_s=0.05, poll_interval_s=0.0)
    )

    assert ready is False


def test_run_startup_global_localize_skips_when_navigation_active():
    slam = SlamService("slam")
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

    slam = SlamService("slam")
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

    slam = SlamService("slam")
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
    slam = SlamService("slam")
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


def test_drive_base_sends_angular_z_to_viam_base():
    slam = SlamService("slam")
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

    node.record_cmd_vel.assert_called_once_with(0.5, 0.0, -1.0, source="builtin")
    slam._base.set_velocity.assert_awaited_once()
    kwargs = slam._base.set_velocity.await_args.kwargs
    assert kwargs["linear"].x == 0.0
    assert kwargs["linear"].y == 500.0
    assert kwargs["angular"].z == pytest.approx(-57.2958, rel=1e-5)


# -- periodic relocalize (drift watchdog) -----------------------------------
def _relocalize_slam(**cfg_overrides):
    from src.nav.pose_jump_gate import PoseJumpGate

    d = {
        "base": "b",
        "lidar": "f",
        "mode": "localizing",
        "periodic_relocalize": True,
        # Unit tests exercise mid-nav hold/correct paths; production default is off.
        "periodic_relocalize_during_navigation": True,
    }
    d.update(cfg_overrides)
    slam = SlamService("slam")
    slam._cfg = SlamConfig.from_dict(d)
    slam._pose_jump_gate = PoseJumpGate(
        confirm_count=slam._cfg.localize_jump_confirm_count,
        agree_m=slam._cfg.localize_jump_agree_m,
        agree_deg=slam._cfg.localize_jump_agree_deg,
        large_m=slam._cfg.localize_jump_large_m,
        large_deg=slam._cfg.localize_jump_large_deg,
    )
    slam._manager = MagicMock()
    slam._manager.get_pose_in_map.return_value = conv.Pose2D(0.0, 0.0, 0.0)
    slam._manager.nav_status.return_value = {
        "active": False,
        "number_of_recoveries": 0,
    }
    slam._manager.node = None
    slam._engine = None
    slam._soft_nav_hold_since = None
    slam._soft_nav_hold_released = False
    slam._startup_global_localize_task = None
    slam._is_navigation_active = MagicMock(return_value=False)
    return slam


def _run_relocalize_until_settled(slam, **cycle_kwargs):
    """Run up to confirm_count cycles so large jumps can clear the gate."""
    needed = max(1, int(slam._cfg.localize_jump_confirm_count))
    result = None
    for _ in range(needed):
        result = asyncio.run(slam._periodic_relocalize_cycle(**cycle_kwargs))
        if result.get("status") != "awaiting_confirm":
            return result
    return result


def test_schedule_periodic_relocalize_skips_when_disabled():
    slam = _relocalize_slam(periodic_relocalize=False)
    loop = MagicMock()
    slam._schedule_periodic_relocalize(loop)
    loop.create_task.assert_not_called()


def test_schedule_periodic_relocalize_skips_when_mapping():
    slam = SlamService("slam")
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
    slam = SlamService("slam")
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

    first = asyncio.run(slam._periodic_relocalize_cycle())
    assert first["status"] == "awaiting_confirm"
    assert first["corrected"] is False
    slam.do_command.assert_not_awaited()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "corrected"
    assert result["corrected"] is True
    assert result["shift_m"] == pytest.approx(1.0)
    relocalize_cmd = slam.do_command.await_args.args[0]
    assert relocalize_cmd["command"] == "relocalize"
    assert relocalize_cmd["pose"]["x"] == pytest.approx(1.0)


def test_periodic_relocalize_cycle_applies_small_jump_immediately():
    slam = _relocalize_slam(
        periodic_relocalize_min_shift_m=0.2,
        localize_jump_large_m=0.75,
    )
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.8,
            "ray_mae_m": 0.3,
            "pose": {"x": 0.4, "y": 0.0, "theta": 0.0},
        }
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = asyncio.run(slam._periodic_relocalize_cycle())
    assert result["status"] == "corrected"
    assert result["jump_status"] == "apply_small"
    slam.do_command.assert_awaited_once()


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


def test_periodic_relocalize_keeps_previous_when_scan_still_fits():
    """Large weak peak during nav must not yank pose if prior still matches."""
    slam = _relocalize_slam(periodic_relocalize_min_score=0.5)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.43,
            "ray_mae_m": 1.44,
            "pose": {"x": 7.7, "y": 0.0, "theta": 0.0},
            "prior_score": 0.55,
            "prior_ray_mae_m": 0.40,
        }
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "previous_better"
    assert result["prefer_previous"] is True
    assert result["shift_m"] == pytest.approx(7.7)
    slam.do_command.assert_not_awaited()
    from src.nav.pose_jump_gate import should_hold_drive_for_pose_jump

    assert not should_hold_drive_for_pose_jump(result)


def test_periodic_relocalize_holds_nav_on_large_uncertain_shift():
    """Lost during nav: large shift, weak match, prior also bad → hold base."""
    slam = _relocalize_slam(periodic_relocalize_min_score=0.5)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.2,
                "ray_mae_m": 1.5,
                "pose": {"x": 7.7, "y": 0.0, "theta": 0.0},
                "prior_score": 0.15,
                "prior_ray_mae_m": 1.6,
            },
            {
                "status": "matched",
                "score": 0.22,
                "ray_mae_m": 1.4,
                "pose": {"x": 7.7, "y": 0.0, "theta": 0.0},
                "prior_score": 0.15,
                "prior_ray_mae_m": 1.6,
            },
        ]
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "nav_hold"
    assert result["navigation_active"] is True
    assert result["large_jump"] is True
    slam.do_command.assert_not_awaited()
    from src.nav.pose_jump_gate import should_hold_drive_for_pose_jump

    assert should_hold_drive_for_pose_jump(result)


def test_periodic_relocalize_holds_nav_on_soft_loc_small_shift():
    """Soft loc with tiny shift (locked onto a bad pose) must still hold."""
    slam = _relocalize_slam(periodic_relocalize_min_score=0.5)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.41,
                "ray_mae_m": 1.00,
                "pose": {"x": 0.08, "y": 0.0, "theta": 0.0},
                "prior_score": 0.32,
                "prior_ray_mae_m": 1.01,
            },
            {
                "status": "matched",
                "score": 0.42,
                "ray_mae_m": 0.98,
                "pose": {"x": 0.08, "y": 0.0, "theta": 0.0},
                "prior_score": 0.32,
                "prior_ray_mae_m": 1.01,
            },
        ]
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "nav_hold"
    assert result.get("soft_loc") is True
    assert result.get("large_jump") is False
    assert result["shift_m"] == pytest.approx(0.08)
    slam.do_command.assert_not_awaited()
    from src.nav.pose_jump_gate import should_hold_drive_for_pose_jump

    assert should_hold_drive_for_pose_jump(result)


def test_periodic_relocalize_holds_borderline_score_during_nav():
    """score 0.495 (above recovery floor, below good_match) must hold mid-nav."""
    slam = _relocalize_slam(
        periodic_relocalize_min_score=0.5,
        periodic_relocalize_max_ray_mae_m=1.0,
        periodic_relocalize_recovery_min_score=0.45,
    )
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.495,
            "ray_mae_m": 0.90,
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "prior_score": 0.495,
            "prior_ray_mae_m": 0.90,
        }
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "nav_hold"
    assert result.get("soft_loc") is True
    assert result.get("previous_ok") is False
    slam.do_command.assert_not_awaited()


def test_periodic_relocalize_soft_hold_resumes_after_timeout():
    """No map jump: brief soft hold, then keep driving on the published pose."""
    import time

    from src.nav.pose_jump_gate import should_hold_drive_for_pose_jump

    slam = _relocalize_slam(
        periodic_relocalize_min_score=0.5,
        periodic_relocalize_soft_hold_max_s=20.0,
    )
    slam._is_navigation_active = MagicMock(return_value=True)
    match = {
        "status": "matched",
        "score": 0.489,
        "ray_mae_m": 0.63,
        "pose": {"x": 0.08, "y": 0.0, "theta": 0.035},
        "prior_score": 0.42,
        "prior_ray_mae_m": 0.70,
    }
    slam._global_localize = AsyncMock(return_value=match)
    slam.do_command = AsyncMock()

    first = asyncio.run(slam._periodic_relocalize_cycle())
    assert first["status"] == "nav_hold"
    assert first.get("soft_loc") is True
    assert should_hold_drive_for_pose_jump(first)

    slam._soft_nav_hold_since = time.monotonic() - 21.0
    second = asyncio.run(slam._periodic_relocalize_cycle())
    assert second["status"] == "soft_loc_resume"
    assert second.get("soft_loc") is True
    assert not should_hold_drive_for_pose_jump(second)
    slam.do_command.assert_not_awaited()

    # Subsequent soft cycles stay in resume (do not re-hold until quality recovers).
    third = asyncio.run(slam._periodic_relocalize_cycle())
    assert third["status"] == "soft_loc_resume"
    assert not should_hold_drive_for_pose_jump(third)


def test_periodic_relocalize_soft_candidate_keeps_driving_if_prior_ok():
    """Weak candidate with a still-plausible published pose → no hold."""
    slam = _relocalize_slam(periodic_relocalize_min_score=0.5)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock(
        side_effect=[
            {
                "status": "matched",
                "score": 0.2,
                "ray_mae_m": 1.5,
                "pose": {"x": 0.1, "y": 0.0, "theta": 0.0},
                "prior_score": 0.55,
                "prior_ray_mae_m": 0.35,
            },
            {
                "status": "matched",
                "score": 0.22,
                "ray_mae_m": 1.4,
                "pose": {"x": 0.1, "y": 0.0, "theta": 0.0},
                "prior_score": 0.55,
                "prior_ray_mae_m": 0.35,
            },
        ]
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "low_quality"
    from src.nav.pose_jump_gate import should_hold_drive_for_pose_jump

    assert not should_hold_drive_for_pose_jump(result)


def test_periodic_relocalize_cycle_escalates_full_map_on_low_quality():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)

    async def _localize(command):
        if command.get("full_map"):
            return {
                "status": "matched",
                "score": 0.85,
                "ray_mae_m": 0.25,
                "pose": {"x": 2.0, "y": 0.0, "theta": 0.0},
            }
        return {
            "status": "matched",
            "score": 0.2,
            "ray_mae_m": 1.5,
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        }

    slam._global_localize = AsyncMock(side_effect=_localize)
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = _run_relocalize_until_settled(slam)

    assert result["status"] == "corrected"
    assert result["match_mode"] == "full_map_after_low_quality"
    assert slam._global_localize.await_count >= 2
    full_cmds = [
        call.args[0]
        for call in slam._global_localize.await_args_list
        if call.args[0].get("full_map")
    ]
    assert full_cmds
    slam.do_command.assert_awaited_once()


def test_periodic_relocalize_still_bad_skips_full_map_when_prior_ok():
    """Route-leg policy: weak local peel must not trigger a full-map search
    when the published pose still explains the scan."""
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._tick_match_score = MagicMock(return_value=0.6)

    async def _localize(command):
        assert not command.get("full_map")
        return {
            "status": "matched",
            "score": 0.2,
            "ray_mae_m": 1.5,
            "prior_score": 0.55,
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        }

    slam._global_localize = AsyncMock(side_effect=_localize)
    result = asyncio.run(
        slam._periodic_relocalize_cycle(full_map_escalation="still_bad")
    )
    assert result["match_mode"] == "local"
    assert slam._global_localize.await_count == 1


def test_periodic_relocalize_still_bad_escalates_when_prior_bad():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._tick_match_score = MagicMock(return_value=-0.2)

    async def _localize(command):
        if command.get("full_map"):
            return {
                "status": "matched",
                "score": 0.85,
                "ray_mae_m": 0.25,
                "pose": {"x": 2.0, "y": 0.0, "theta": 0.0},
            }
        return {
            "status": "matched",
            "score": 0.2,
            "ray_mae_m": 1.5,
            "prior_score": -0.3,
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
        }

    slam._global_localize = AsyncMock(side_effect=_localize)
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})
    result = _run_relocalize_until_settled(
        slam, full_map_escalation="still_bad"
    )
    assert result["match_mode"] == "full_map_after_still_bad"
    assert slam._global_localize.await_count >= 2


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

    result = _run_relocalize_until_settled(slam)

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

    result = _run_relocalize_until_settled(slam)

    assert result["status"] == "corrected"
    assert result["corrected"] is True
    assert result["good_match"] is False
    assert result["recovery_apply"] is True
    assert result["match_mode"] == "full_map"
    slam.do_command.assert_awaited_once()


def test_periodic_relocalize_cycle_recovery_floor_blocks_garbage():
    # A full-map recovery match whose score is below the recovery floor is genuine
    # garbage and must not be applied. During nav a large weak shift also holds.
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
            "prior_score": 0.2,
            "prior_ray_mae_m": 1.5,
        }
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "nav_hold"
    assert result["recovery_apply"] is False
    slam.do_command.assert_not_awaited()
    from src.nav.pose_jump_gate import should_hold_drive_for_pose_jump

    assert should_hold_drive_for_pose_jump(result)


def test_periodic_relocalize_skips_while_spinning():
    slam = _relocalize_slam(periodic_relocalize_max_yaw_rate_rad_s=0.35)
    slam._manager.node = MagicMock()
    slam._manager.node.slam_bridge_status.return_value = {
        "odom_velocity": {"vx": 0.0, "vy": 0.0, "vtheta": 0.8}
    }
    slam._global_localize = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "skipped"
    assert result["reason"] == "spinning"
    slam._global_localize.assert_not_awaited()


def test_periodic_relocalize_skips_stale_scan():
    slam = _relocalize_slam(periodic_relocalize_max_scan_age_s=0.75)
    slam._manager.node = MagicMock()
    slam._manager.node.slam_bridge_status.return_value = {
        "odom_velocity": {"vx": 0.2, "vy": 0.0, "vtheta": 0.0}
    }
    slam._engine = MagicMock()
    slam._engine.diagnostics.return_value = {
        "last_scan_age_s": 1.5,
        "yaw_rate_deg_s": 0.0,
    }
    slam._global_localize = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "skipped"
    assert result["reason"] == "stale_scan"
    slam._global_localize.assert_not_awaited()


def test_periodic_relocalize_cycle_skips_during_navigation():
    slam = _relocalize_slam(periodic_relocalize_during_navigation=False)
    slam._is_navigation_active = MagicMock(return_value=True)
    slam._global_localize = AsyncMock()
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "skipped"
    assert result["reason"] == "navigation_active"
    slam._global_localize.assert_not_awaited()


def test_is_navigation_active_sees_registered_builtin_nav_host():
    """BuiltinSlamHost always reports idle; registered nav host is the source of truth."""
    from src.runtime import any_navigation_active, register_nav_host, unregister_nav_host

    unregister_nav_host("nav-test-active")
    assert any_navigation_active() is False

    host = MagicMock()
    host.nav_status.return_value = {"active": True, "state": "active"}
    register_nav_host("nav-test-active", host)
    try:
        slam = SlamService("slam-test-active")
        slam._manager = MagicMock()
        slam._manager.nav_status.return_value = {
            "active": False,
            "state": "idle",
        }  # BuiltinSlamHost shape
        cancel = MagicMock()
        slam._cancel_startup_global_localize_task = cancel
        assert slam._is_navigation_active() is True
        cancel.assert_called_once()
    finally:
        unregister_nav_host("nav-test-active")


def test_periodic_relocalize_cycle_skips_while_startup_running():
    slam = _relocalize_slam()
    pending = MagicMock()
    pending.done.return_value = False
    slam._startup_global_localize_task = pending
    slam._startup_localize_started_at = time.monotonic()
    slam._global_localize = AsyncMock()
    # No engine tick score → do not bypass yet.
    slam._engine = None

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "skipped"
    assert result["reason"] == "startup_localize_running"
    slam._global_localize.assert_not_awaited()


def test_periodic_relocalize_bypasses_startup_when_tick_match_bad():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    pending = MagicMock()
    pending.done.return_value = False
    slam._startup_global_localize_task = pending
    slam._startup_localize_started_at = time.monotonic()
    slam._engine = MagicMock()
    slam._engine.diagnostics.return_value = {"last_match_score": -0.3}
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.85,
            "ray_mae_m": 0.2,
            "pose": {"x": 1.5, "y": 0.0, "theta": 0.0},
            "ambiguous": False,
        }
    )
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = _run_relocalize_until_settled(slam)

    assert slam._startup_global_localize_task is None
    assert result["status"] == "corrected"
    assert result["match_mode"] == "full_map_still_recovery"
    first = slam._global_localize.await_args_list[0].args[0]
    assert first.get("full_map") is True


def test_periodic_relocalize_still_bad_goes_straight_to_full_map():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._engine = MagicMock()
    slam._engine.diagnostics.return_value = {"last_match_score": -0.25}

    async def _localize(command):
        assert command.get("full_map") is True
        return {
            "status": "matched",
            "score": 0.8,
            "ray_mae_m": 0.2,
            "pose": {"x": 2.0, "y": 0.0, "theta": 0.0},
            "ambiguous": False,
        }

    slam._global_localize = AsyncMock(side_effect=_localize)
    slam.do_command = AsyncMock(return_value={"status": "relocalizing"})

    result = _run_relocalize_until_settled(slam)

    assert result["match_mode"] == "full_map_still_recovery"
    assert result["status"] == "corrected"
    assert slam._global_localize.await_count >= 1


def test_periodic_relocalize_refuses_ambiguous_full_map():
    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._engine = MagicMock()
    slam._engine.diagnostics.return_value = {"last_match_score": -0.4}
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.55,
            "ray_mae_m": 0.5,
            "pose": {"x": 3.0, "y": 0.0, "theta": 0.0},
            "ambiguous": True,
            "second_best_score": 0.7,
        }
    )
    slam.do_command = AsyncMock()

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "ambiguous"
    assert result["corrected"] is False
    slam.do_command.assert_not_awaited()


def test_periodic_relocalize_refuses_ambiguous_local_fallback():
    """Local command that auto-fell-back to full-map must still refuse twins."""
    from src.geom.conversions import Pose2D

    slam = _relocalize_slam(periodic_relocalize_min_shift_m=0.2)
    slam._engine = MagicMock()
    # Above still-bad floor so the cycle starts as a local peek.
    slam._engine.diagnostics.return_value = {"last_match_score": 0.32}
    slam._global_localize = AsyncMock(
        return_value={
            "status": "matched",
            "score": 0.55,
            "ray_mae_m": 0.45,
            "pose": {"x": 30.0, "y": 0.0, "theta": 0.0},
            "ambiguous": True,
            "second_best_score": 0.52,
            "full_map": True,
            "fallback_used": True,
        }
    )
    slam.do_command = AsyncMock()
    # Pretend a prior confirm was already counting toward a yank.
    slam._pose_jump_gate.evaluate(Pose2D(0.0, 0.0, 0.0), Pose2D(30.0, 0.0, 0.0))
    assert slam._pose_jump_gate.snapshot()["confirm_count"] == 1

    result = asyncio.run(slam._periodic_relocalize_cycle())

    assert result["status"] == "ambiguous"
    assert result["match_mode"] == "full_map_via_local_fallback"
    assert result["corrected"] is False
    assert slam._pose_jump_gate.snapshot()["confirm_count"] == 0
    slam.do_command.assert_not_awaited()


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

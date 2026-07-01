from pathlib import Path
import asyncio
import math
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import MODE_LOCALIZING, MODE_MAPPING

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

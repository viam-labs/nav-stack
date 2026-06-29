from pathlib import Path
import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from src.config import MODE_MAPPING

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

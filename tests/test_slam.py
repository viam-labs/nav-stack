from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

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

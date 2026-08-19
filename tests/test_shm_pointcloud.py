from __future__ import annotations

import asyncio
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

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

from src.config import SlamConfig
from src.ros import pcshm
from src.ros.sensor_io import build_io_provider
from src.ros.shm_lidar import ShmPointCloudClient

_MIN_PCD = (
    b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
    b"COUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n0 0 1\n"
)


def test_auto_mode_skips_rplidar_status_docommand():
    name = "/viam-pc-rplidar-auto"
    writer = pcshm.open_writer(name)
    client = ShmPointCloudClient()
    cam = MagicMock()
    cam.do_command = AsyncMock(
        return_value={"scans": 42, "serial_path": "/dev/ttyUSB1", "shm_name": name}
    )
    cam.get_point_cloud = AsyncMock(return_value=(_MIN_PCD, "pointcloud/pcd"))
    logger = MagicMock()
    skip: set[str] = set()
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": [{"name": "lidar", "scan_source": "auto", "shm_name": name}],
            "sensor_read_timeout_s": 1.0,
        }
    )
    io = build_io_provider(
        base=MagicMock(),
        cameras={"lidar": cam},
        cfg=cfg,
        logger=logger,
        shm_lidar=client,
        skip_get_laser_scan=skip,
    )
    try:
        writer.write(_MIN_PCD, timestamp_ns=time.time_ns())
        pts = asyncio.run(io.read_lidar_points("lidar"))
        assert pts.sensor.shape[0] == 1
        assert "lidar" in skip
        cam.do_command.assert_awaited_once()
        logger.warning.assert_not_called()
    finally:
        client.close()
        writer.close()


def test_rplidar_get_laser_scan_raises_not_implemented():
    from src.models.rplidar_shm import RPLidarShm

    cam = RPLidarShm("lidar")
    with pytest.raises(NotImplementedError, match="get_laser_scan"):
        asyncio.run(cam.do_command({"command": "get_laser_scan"}))


def test_shm_pointcloud_get_point_cloud_waits_for_producer():
    from src.models.shm_pointcloud import ShmPointCloud

    source = MagicMock()

    async def slow_pcd(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return _MIN_PCD, "pointcloud/pcd"

    source.get_point_cloud = slow_pcd
    cam = ShmPointCloud("rep")
    cam._timeout_s = 1.0
    cam._produce_hz = 5.0
    cam._source = source
    cam._shm = pcshm.open_writer("/viam-pc-producer-test")
    cam._stop = threading.Event()
    cam._thread = threading.Thread(target=cam._produce_loop, daemon=True)
    cam._thread.start()
    try:
        raw, mime = asyncio.run(asyncio.wait_for(cam.get_point_cloud(timeout=1.0), 1.0))
        assert mime == "pointcloud/pcd"
        assert raw == _MIN_PCD
        assert cam._writes >= 1
    finally:
        cam.close_sync()


def test_shm_pointcloud_on_demand_mode_fetches_inline():
    from src.models.shm_pointcloud import ShmPointCloud

    source = MagicMock()
    source.get_point_cloud = AsyncMock(return_value=(_MIN_PCD, "pointcloud/pcd"))
    cam = ShmPointCloud("rep")
    cam._timeout_s = 1.0
    cam._produce_hz = 0.0
    cam._source = source
    cam._shm = pcshm.open_writer("/viam-pc-on-demand")
    try:
        raw, mime = asyncio.run(cam.get_point_cloud(timeout=1.0))
        assert raw == _MIN_PCD
        assert mime == "pointcloud/pcd"
        source.get_point_cloud.assert_awaited_once()
        assert cam._writes == 1
    finally:
        cam.close_sync()


def test_shm_pointcloud_reconfigure_reuses_stop_event():
    from src.models.shm_pointcloud import ShmPointCloud

    cam = ShmPointCloud("rep")
    stop = cam._stop
    cam._stop.set()
    cam.close_sync()
    cam._stop.clear()
    assert cam._stop is stop
    assert not cam._stop.is_set()


def test_close_sync_keeps_thread_and_shm_when_join_fails():
    from src.models.shm_pointcloud import ShmPointCloud

    cam = ShmPointCloud("rep")
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True
    fake_shm = MagicMock()
    cam._thread = fake_thread
    cam._shm = fake_shm
    cam.close_sync()
    assert cam._thread is fake_thread
    assert cam._shm is fake_shm
    fake_shm.close.assert_not_called()

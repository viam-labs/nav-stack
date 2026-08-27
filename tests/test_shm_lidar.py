from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub ROS 2 Python deps so IO tests run without a ROS install.
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


def _io(*, lidars, cameras, shm_lidar=None):
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": lidars,
            "sensor_read_timeout_s": 1.0,
        }
    )
    return build_io_provider(
        base=MagicMock(),
        cameras=cameras,
        cfg=cfg,
        logger=MagicMock(),
        shm_lidar=shm_lidar,
    )


def test_read_lidar_uses_shm_when_configured():
    name = "/viam-pc-navio"
    writer = pcshm.open_writer(name)
    client = ShmPointCloudClient()
    cam = MagicMock()
    cam.get_point_cloud = AsyncMock(side_effect=AssertionError("gRPC should not run"))
    try:
        writer.write(_MIN_PCD, timestamp_ns=time.time_ns())
        io = _io(
            lidars=[
                {
                    "name": "front",
                    "scan_source": "point_cloud",
                    "shm_name": name,
                    "shm_required": True,
                }
            ],
            cameras={"front": cam},
            shm_lidar=client,
        )
        pts = asyncio.run(io.read_lidar_points("front"))
        assert pts.sensor.shape[0] == 1
        cam.get_point_cloud.assert_not_called()
        stats = client.status()[name]
        assert stats["hits"] >= 1
        assert stats["grpc_fallbacks"] == 0
    finally:
        client.close()
        writer.close()


def test_read_lidar_falls_back_to_grpc_when_shm_empty():
    name = "/viam-pc-navio2"
    writer = pcshm.open_writer(name)
    client = ShmPointCloudClient()
    cam = MagicMock()
    cam.get_point_cloud = AsyncMock(return_value=(_MIN_PCD, "pointcloud/pcd"))
    try:
        io = _io(
            lidars=[
                {
                    "name": "front",
                    "scan_source": "point_cloud",
                    "shm_name": name,
                    "shm_required": False,
                }
            ],
            cameras={"front": cam},
            shm_lidar=client,
        )
        pts = asyncio.run(io.read_lidar_points("front"))
        assert pts.sensor.shape[0] == 1
        cam.get_point_cloud.assert_awaited()
        assert client.status()[name]["grpc_fallbacks"] == 1
    finally:
        client.close()
        writer.close()


def test_read_lidar_shm_required_raises_when_empty():
    name = "/viam-pc-navio3"
    writer = pcshm.open_writer(name)
    client = ShmPointCloudClient()
    cam = MagicMock()
    cam.get_point_cloud = AsyncMock()
    try:
        io = _io(
            lidars=[
                {
                    "name": "front",
                    "scan_source": "point_cloud",
                    "shm_name": name,
                    "shm_required": True,
                }
            ],
            cameras={"front": cam},
            shm_lidar=client,
        )
        with pytest.raises(RuntimeError, match="no complete frame"):
            asyncio.run(io.read_lidar_points("front"))
        cam.get_point_cloud.assert_not_called()
    finally:
        client.close()
        writer.close()


def test_client_remaps_after_writer_restart():
    name = "/viam-pc-restart"
    client = ShmPointCloudClient()
    w1 = pcshm.open_writer(name)
    stale_ns = 1_000_000_000
    try:
        w1.write(_MIN_PCD, timestamp_ns=stale_ns)
        assert client.try_read(name, max_age_s=2.0) is None
        assert client.status()[name]["stale_hits"] >= 1
        assert client.status()[name]["remaps"] >= 1
    finally:
        w1.close()

    w2 = pcshm.open_writer(name)
    try:
        w2.write(_MIN_PCD, timestamp_ns=time.time_ns())
        got = client.try_read(name, max_age_s=2.0)
        assert got is not None
        stats = client.status()[name]
        assert stats["hits"] >= 1
        assert stats["remaps"] >= 1
    finally:
        w2.close()
        client.close()


def test_client_remaps_on_no_frame_after_writer_restart():
    name = "/viam-pc-restart2"
    client = ShmPointCloudClient()
    w1 = pcshm.open_writer(name)
    try:
        w1.write(_MIN_PCD, timestamp_ns=time.time_ns())
        assert client.try_read(name) is not None
    finally:
        w1.close()

    w2 = pcshm.open_writer(name)
    try:
        w2.write(_MIN_PCD, timestamp_ns=time.time_ns())
        got = client.try_read(name)
        assert got is not None
        assert client.status()[name]["hits"] >= 2
    finally:
        w2.close()
        client.close()


def test_read_lidar_rejects_stale_shm_frame():
    name = "/viam-pc-navio-stale"
    writer = pcshm.open_writer(name)
    client = ShmPointCloudClient()
    cam = MagicMock()
    cam.get_point_cloud = AsyncMock()
    stale_ns = 1_000_000_000  # 1s epoch — always older than max_age
    try:
        writer.write(_MIN_PCD, timestamp_ns=stale_ns)
        cfg = SlamConfig.from_dict(
            {
                "base": "b",
                "lidars": [
                    {
                        "name": "front",
                        "scan_source": "point_cloud",
                        "shm_name": name,
                        "shm_required": True,
                    }
                ],
                "sensor_read_timeout_s": 1.0,
                "scan_max_age_s": 2.0,
            }
        )
        io = build_io_provider(
            base=MagicMock(),
            cameras={"front": cam},
            cfg=cfg,
            logger=MagicMock(),
            shm_lidar=client,
        )
        with pytest.raises(RuntimeError, match="frame too old"):
            asyncio.run(io.read_lidar_points("front"))
        stats = client.status()[name]
        assert stats["stale_hits"] >= 1
        assert stats["hits"] == 0
        cam.get_point_cloud.assert_not_called()
    finally:
        client.close()
        writer.close()


def test_read_lidar_stale_shm_does_not_fallback_when_not_required():
    """shm_required=false must still refuse gRPC fallback for stale frames."""
    name = "/viam-pc-navio-stale-opt"
    writer = pcshm.open_writer(name)
    client = ShmPointCloudClient()
    cam = MagicMock()
    cam.get_point_cloud = AsyncMock(return_value=(_MIN_PCD, "pointcloud/pcd"))
    stale_ns = 1_000_000_000
    try:
        writer.write(_MIN_PCD, timestamp_ns=stale_ns)
        cfg = SlamConfig.from_dict(
            {
                "base": "b",
                "lidars": [
                    {
                        "name": "front",
                        "scan_source": "point_cloud",
                        "shm_name": name,
                        "shm_required": False,
                    }
                ],
                "sensor_read_timeout_s": 1.0,
                "scan_max_age_s": 2.0,
            }
        )
        io = build_io_provider(
            base=MagicMock(),
            cameras={"front": cam},
            cfg=cfg,
            logger=MagicMock(),
            shm_lidar=client,
        )
        with pytest.raises(RuntimeError, match="frame too old"):
            asyncio.run(io.read_lidar_points("front"))
        cam.get_point_cloud.assert_not_called()
        assert client.status()[name]["grpc_fallbacks"] == 0
    finally:
        client.close()
        writer.close()

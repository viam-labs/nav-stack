"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server.

Builtin slam/nav and camera helpers always register and need no ROS.
``slam_backend: slam_toolbox`` and ``nav_backend: nav2`` require ``rclpy``
(set ``REQUIRE_ROS=1`` at setup on Ubuntu).
"""
from __future__ import annotations

import asyncio
import logging

from viam.module.module import Module

LOGGER = logging.getLogger(__name__)

# Isolate DDS before any model imports spin up rclpy / child ROS processes.
from .ros.availability import rclpy_available
from .ros.dds_env import apply_dds_isolation

apply_dds_isolation()

from .models.shm_pointcloud import ShmPointCloud
from .models.rplidar_shm import RPLidarShm
from .models.wit_imu import WitImu
from .models.navigation_external import RosNavigationExternal
from .models.navigation import RosNavigation
from .models.nav_camera import NavCamera
from .models.slam import RosSlam

if not rclpy_available():
    LOGGER.info(
        "rclpy not found — ROS-free defaults active "
        "(slam_backend=builtin, nav_backend=builtin). "
        "slam_toolbox / Nav2 need REQUIRE_ROS=1 at setup on Ubuntu."
    )


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(RosSlam.API, RosSlam.MODEL)
    module.add_model_from_registry(RosNavigation.API, RosNavigation.MODEL)
    module.add_model_from_registry(
        RosNavigationExternal.API, RosNavigationExternal.MODEL
    )
    module.add_model_from_registry(NavCamera.API, NavCamera.MODEL)
    module.add_model_from_registry(ShmPointCloud.API, ShmPointCloud.MODEL)
    module.add_model_from_registry(RPLidarShm.API, RPLidarShm.MODEL)
    module.add_model_from_registry(WitImu.API, WitImu.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

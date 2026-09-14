"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server.

ROS-free only: builtin SLAM + builtin navigation (no rclpy / Nav2 / slam_toolbox).
"""
from __future__ import annotations

import asyncio

from viam.module.module import Module

from .models.shm_pointcloud import ShmPointCloud
from .models.rplidar_shm import RPLidarShm
from .models.wit_imu import WitImu
from .models.navigation_external import RosNavigationExternal
from .models.navigation import RosNavigation
from .models.nav_camera import NavCamera
from .models.sim_base import SimBase
from .models.slam import RosSlam


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(RosSlam.API, RosSlam.MODEL)
    module.add_model_from_registry(RosNavigation.API, RosNavigation.MODEL)
    module.add_model_from_registry(
        RosNavigationExternal.API, RosNavigationExternal.MODEL
    )
    module.add_model_from_registry(NavCamera.API, NavCamera.MODEL)
    module.add_model_from_registry(SimBase.API, SimBase.MODEL)
    module.add_model_from_registry(ShmPointCloud.API, ShmPointCloud.MODEL)
    module.add_model_from_registry(RPLidarShm.API, RPLidarShm.MODEL)
    module.add_model_from_registry(WitImu.API, WitImu.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

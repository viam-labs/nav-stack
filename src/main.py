"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server.

Builtin SLAM + builtin navigation only.
"""
from __future__ import annotations

import asyncio

from viam.module.module import Module

from .models.shm_pointcloud import ShmPointCloud
from .models.rplidar_shm import RPLidarShm
from .models.wit_imu import WitImu
from .models.navigation_external import ExternalNavigationService
from .models.navigation import NavigationService
from .models.nav_camera import NavCamera
from .models.sim_base import SimBase
from .models.slam import SlamService


async def main() -> None:
    module = Module.from_args()
    from .runtime import set_module

    set_module(module)
    module.add_model_from_registry(SlamService.API, SlamService.MODEL)
    module.add_model_from_registry(NavigationService.API, NavigationService.MODEL)
    module.add_model_from_registry(
        ExternalNavigationService.API, ExternalNavigationService.MODEL
    )
    module.add_model_from_registry(NavCamera.API, NavCamera.MODEL)
    module.add_model_from_registry(SimBase.API, SimBase.MODEL)
    module.add_model_from_registry(ShmPointCloud.API, ShmPointCloud.MODEL)
    module.add_model_from_registry(RPLidarShm.API, RPLidarShm.MODEL)
    module.add_model_from_registry(WitImu.API, WitImu.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

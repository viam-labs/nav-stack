"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server.

Builtin SLAM + builtin navigation only.
"""
from __future__ import annotations

import asyncio
import sys

from viam.module.module import Module

from .models.shm_pointcloud import ShmPointCloud
from .models.rplidar_shm import RPLidarShm
from .models.wit_imu import WitImu
from .models.navigation_external import ExternalNavigationService
from .models.navigation import NavigationService
from .models.nav_camera import NavCamera
from .models.sim_base import SimBase
from .models.slam import SlamService


# SLAM ticks, nav, the lidar/IMU readers and the module event loop share one
# GIL. A pure-Python planner flood holds it for seconds; at the default 5 ms
# switch interval every event-loop step behind it waits ~5 ms, so SLAM's
# 80 ms odometry reads time out for the whole plan and turns made meanwhile
# are dropped. 0.5 ms keeps those reads ~15 ms with no planner slowdown.
_GIL_SWITCH_INTERVAL_S = 0.0005


async def main() -> None:
    sys.setswitchinterval(_GIL_SWITCH_INTERVAL_S)
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

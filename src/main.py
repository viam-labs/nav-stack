"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server.

ROS-backed models (navigation with ``nav_backend: nav2``, slam with
``slam_backend: slam_toolbox``) require ``rclpy``. Builtin slam/nav and camera
helpers always register.
"""
from __future__ import annotations

import asyncio
import logging

from viam.module.module import Module

LOGGER = logging.getLogger(__name__)


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


# Isolate DDS before any model imports spin up rclpy / child ROS processes.
from .ros.dds_env import apply_dds_isolation

apply_dds_isolation()

# Always-on (no ROS required at import time).
from .models.shm_pointcloud import ShmPointCloud
from .models.rplidar_shm import RPLidarShm
from .models.navigation_external import RosNavigationExternal
from .models.nav_camera import NavCamera
from .models.slam import RosSlam

_NAV2_MODEL = None
if _rclpy_available():
    from .models.navigation import RosNavigation

    _NAV2_MODEL = RosNavigation
else:
    LOGGER.warning(
        "rclpy not found — registering ROS-free models only "
        "(slam builtin, navigation-external, nav-camera, shm-pointcloud, rplidar). "
        "slam_backend=slam_toolbox and nav_backend=nav2 require a ROS 2 install."
    )


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(RosSlam.API, RosSlam.MODEL)
    if _NAV2_MODEL is not None:
        module.add_model_from_registry(_NAV2_MODEL.API, _NAV2_MODEL.MODEL)
    module.add_model_from_registry(
        RosNavigationExternal.API, RosNavigationExternal.MODEL
    )
    module.add_model_from_registry(NavCamera.API, NavCamera.MODEL)
    module.add_model_from_registry(ShmPointCloud.API, ShmPointCloud.MODEL)
    module.add_model_from_registry(RPLidarShm.API, RPLidarShm.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

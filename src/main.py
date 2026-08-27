"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server.

ROS-backed models (slam / navigation-nav2 / nav-camera against a bridge) require
``rclpy``. Camera helpers and ``navigation-external`` with ``nav_backend: builtin``
are ROS-free and always register.
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

# Always-on (no ROS required).
from .models.shm_pointcloud import ShmPointCloud
from .models.rplidar_shm import RPLidarShm

# Builtin nav + external SLAM: import without requiring rclpy at import time.
# navigation_external only imports RosManager lazily on nav_backend=nav2.
from .models.navigation_external import RosNavigationExternal
from .models.nav_camera import NavCamera

_ROS_MODELS = None
if _rclpy_available():
    from .models.slam import RosSlam
    from .models.navigation import RosNavigation

    _ROS_MODELS = (RosSlam, RosNavigation)
else:
    LOGGER.warning(
        "rclpy not found — registering ROS-free models only "
        "(navigation-external, nav-camera, shm-pointcloud, rplidar). "
        "nav-stack:slam and nav_backend=nav2 require a ROS 2 install."
    )


async def main() -> None:
    module = Module.from_args()
    if _ROS_MODELS is not None:
        for model_cls in _ROS_MODELS:
            module.add_model_from_registry(model_cls.API, model_cls.MODEL)
    module.add_model_from_registry(
        RosNavigationExternal.API, RosNavigationExternal.MODEL
    )
    module.add_model_from_registry(NavCamera.API, NavCamera.MODEL)
    module.add_model_from_registry(ShmPointCloud.API, ShmPointCloud.MODEL)
    module.add_model_from_registry(RPLidarShm.API, RPLidarShm.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

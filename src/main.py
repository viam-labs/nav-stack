"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server:

* ``viam-labs:nav-stack:slam``                 - SLAM service (mapping + localization)
* ``viam-labs:nav-stack:navigation``           - Motion (MoveOnMap) + Nav2/zones DoCommand
* ``viam-labs:nav-stack:navigation-external``  - Motion against any rdk:service:slam
* ``viam-labs:nav-stack:nav-camera``           - camera rendering the costmap + active plans
"""
import asyncio

from viam.module.module import Module

# Isolate DDS before any model imports spin up rclpy / child ROS processes.
from .ros.dds_env import apply_dds_isolation

apply_dds_isolation()

# Importing the model modules triggers their registration with the Viam registry.
from .models.slam import RosSlam
from .models.navigation import RosNavigation
from .models.navigation_external import RosNavigationExternal
from .models.nav_camera import NavCamera


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(RosSlam.API, RosSlam.MODEL)
    module.add_model_from_registry(RosNavigation.API, RosNavigation.MODEL)
    module.add_model_from_registry(
        RosNavigationExternal.API, RosNavigationExternal.MODEL
    )
    module.add_model_from_registry(NavCamera.API, NavCamera.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

"""Module entrypoint.

Registers the nav-stack resource models with the Viam module server:

* ``viam-labs:nav-stack:slam``                 - SLAM service (mapping + localization)
* ``viam-labs:nav-stack:navigation``           - rdk:service:navigation (built-in SLAM runtime)
* ``viam-labs:nav-stack:navigation-external``  - rdk:service:navigation against any rdk:service:slam
"""
import asyncio

from viam.module.module import Module

# Importing the model modules triggers their registration with the Viam registry.
from .models.slam import RosSlam
from .models.navigation import RosNavigation
from .models.navigation_external import RosNavigationExternal


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(RosSlam.API, RosSlam.MODEL)
    module.add_model_from_registry(RosNavigation.API, RosNavigation.MODEL)
    module.add_model_from_registry(
        RosNavigationExternal.API, RosNavigationExternal.MODEL
    )
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

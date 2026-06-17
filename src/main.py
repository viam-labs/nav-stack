"""Module entrypoint.

Registers the two nav-stack resource models with the Viam module server:

* ``viam-labs:nav-stack:slam``        - SLAM service (mapping + localization)
* ``viam-labs:nav-stack:navigation``  - generic service (Nav2 navigation + zones)
"""
import asyncio

from viam.module.module import Module

# Importing the model modules triggers their registration with the Viam registry.
from .models.slam import RosSlam
from .models.navigation import RosNavigation


async def main() -> None:
    module = Module.from_args()
    module.add_model_from_registry(RosSlam.API, RosSlam.MODEL)
    module.add_model_from_registry(RosNavigation.API, RosNavigation.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

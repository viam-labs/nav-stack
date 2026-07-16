"""External-SLAM navigation model: ``viam-labs:nav-stack:navigation-external``.

Like ``viam-labs:nav-stack:navigation`` (same generic DoCommand surface, shared
:class:`~.nav_core.NavServiceBase`), but instead of borrowing the built-in SLAM
model's in-process runtime it drives Nav2 from an **arbitrary Viam
``rdk:service:slam``** dependency.

It stands up its own ROS runtime:

* a :class:`~..ros.manager.RosManager` that runs the sensor bridge (lidars ->
  ``/scan``, movement sensor -> ``/odom`` + ``odom->base_link`` TF) but **not**
  slam_toolbox, and
* an :class:`~..ros.external_slam.ExternalSlamPublisher` (started by the bridge
  when given ``external_slam``) that republishes the Viam SLAM service's pose and
  occupancy grid as ``map->odom`` + ``/map``.

Odometry is read through the portable typed MovementSensor API
(:class:`~..ros.odom_source.TypedMovementSensorOdom`), so it works with any
movement sensor, not just those matching a specific ``get_readings`` schema.
"""
from __future__ import annotations

import asyncio
from typing import ClassVar, Mapping, Optional, Sequence, cast

from typing_extensions import Self

from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.movement_sensor import MovementSensor
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.slam import SLAM
from viam.utils import struct_to_dict

from ..config import ExternalNavConfig
from ..nav.maps import MapStore
from ..ros.manager import RosManager
from ..ros.odom_source import TypedMovementSensorOdom, TypedOdomConfig
from ..ros.sensor_io import build_io_provider
from ..runtime import SlamRuntime
from .nav_core import NavServiceBase

LOGGER = getLogger(__name__)


class RosNavigationExternal(NavServiceBase):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "nav-stack"), "navigation-external"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._manager: Optional[RosManager] = None
        self._runtime = None

    # -- registration --------------------------------------------------------
    @classmethod
    def new(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(
        cls, config: ServiceConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        cfg = ExternalNavConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    # -- runtime resolution --------------------------------------------------
    def _resolve_runtime(self):
        return self._runtime

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        ext = ExternalNavConfig.from_dict(struct_to_dict(config.attributes))
        bridge_cfg = ext.bridge
        self._cfg = ext.nav  # NavServiceBase drives Nav2 from the NavConfig

        # Resolve Viam dependencies.
        self._base = cast(Base, dependencies[Base.get_resource_name(ext.nav.base)])
        slam = cast(SLAM, dependencies[SLAM.get_resource_name(ext.slam_service)])
        cameras = {
            lidar.name: cast(
                Camera, dependencies[Camera.get_resource_name(lidar.name)]
            )
            for lidar in bridge_cfg.lidars
        }
        movement_sensor = (
            cast(
                MovementSensor,
                dependencies[MovementSensor.get_resource_name(bridge_cfg.movement_sensor)],
            )
            if bridge_cfg.movement_sensor
            else None
        )
        heading_sensor = (
            cast(
                MovementSensor,
                dependencies[MovementSensor.get_resource_name(bridge_cfg.heading_sensor)],
            )
            if bridge_cfg.heading_sensor
            else None
        )

        # Rebuild the ROS stack from scratch on every reconfigure.
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

        odom_reader = (
            TypedMovementSensorOdom(
                movement_sensor,
                TypedOdomConfig(
                    trust_pose=ext.trust_movement_sensor_pose,
                    snap_heading=ext.snap_heading,
                ),
                logger=LOGGER,
            )
            if movement_sensor is not None
            else None
        )
        io = build_io_provider(
            base=self._base,
            cameras=cameras,
            cfg=bridge_cfg,
            movement_sensor=movement_sensor,
            heading_sensor=heading_sensor,
            skip_get_laser_scan=set(),
            odom_reader=odom_reader,
            logger=LOGGER,
        )

        # external_slam -> the bridge starts the ExternalSlamPublisher (publishes
        # /map + map->odom from the SLAM service); slam_toolbox is never launched.
        self._manager = RosManager(bridge_cfg, logger=LOGGER, external_slam=slam)
        loop = asyncio.get_event_loop()
        self._manager.start(io, loop, nav_cfg=ext.nav)

        # Locations/zones live in a nav-stack-managed map store (the external
        # SLAM owns the occupancy grid, delivered live via get_grid).
        map_store = MapStore(bridge_cfg.maps_dir)
        active = bridge_cfg.active_map or map_store.get_active_map_name() or "default"
        map_store.get_or_create_map(active)
        map_store.set_active_map(active)

        node = self._manager.node
        loc_check = (
            node._external.localization_check
            if node is not None and node._external is not None
            else {"status": "unknown"}
        )
        self._runtime = SlamRuntime(self._manager, map_store, bridge_cfg, loc_check)

        self._manager.set_nav_config(ext.nav)
        params_path = self._write_nav2_params(ext.nav)
        # Nav2 bringup can take minutes on a Pi; start in the background so
        # reconfigure returns within viam-server's deadline.
        self._manager.ensure_nav2_async(ext.nav, params_path)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation-external '{self.name}' configured "
            f"({ext.nav.kinematics}); Nav2 starting in background against "
            f"SLAM service {ext.slam_service!r}"
        )

    async def close(self) -> None:
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._runtime = None


Registry.register_resource_creator(
    Generic.API,
    RosNavigationExternal.MODEL,
    ResourceCreatorRegistration(
        RosNavigationExternal.new, RosNavigationExternal.validate_config
    ),
)

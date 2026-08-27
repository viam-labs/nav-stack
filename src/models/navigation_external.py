"""External-SLAM navigation model: ``viam-labs:nav-stack:navigation-external``.

Like ``viam-labs:nav-stack:navigation`` (same ``rdk:service:motion`` + DoCommand
surface, shared :class:`~.nav_core.NavServiceBase`), but instead of borrowing
the built-in SLAM model's in-process runtime it drives navigation from an
**arbitrary Viam ``rdk:service:slam``** dependency.

Default ``nav_backend`` is ``builtin``. In that mode there is **no ROS**: map
and pose come from the SLAM service (``get_grid`` / ``GetPosition``), scans from
lidar cameras, and drive from ``Base.SetVelocity``.

With ``nav_backend: nav2`` it stands up its own ROS runtime:

* a :class:`~..ros.manager.RosManager` that runs the sensor bridge (lidars ->
  ``/scan``, movement sensor -> ``/odom`` + ``odom->base_link`` TF) but **not**
  slam_toolbox, and
* an :class:`~..ros.external_slam.ExternalSlamPublisher` (started by the bridge
  when given ``external_slam``) that republishes the Viam SLAM service's pose and
  occupancy grid as ``map->odom`` + ``/map``.

Odometry (Nav2 path only) is read through the portable typed MovementSensor API
(:class:`~..ros.odom_source.TypedMovementSensorOdom`).
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
from viam.services.motion import Motion
from viam.services.slam import SLAM
from viam.utils import struct_to_dict

from ..config import ExternalNavConfig
from ..nav.maps import MapStore
from ..nav_builtin import (
    BuiltinNavHost,
    NavVizStore,
    ViamWorldIO,
    make_builtin_navigator,
)
from ..ros.shm_lidar import ShmPointCloudClient
from ..runtime import (
    SlamRuntime,
    register_bridge,
    register_nav_viz,
    unregister_bridge,
    unregister_nav_viz,
)
from .nav_core import NavServiceBase

LOGGER = getLogger(__name__)


class RosNavigationExternal(NavServiceBase):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "nav-stack"), "navigation-external"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._manager = None
        self._runtime = None
        self._viz: Optional[NavVizStore] = None
        self._shm_lidar = ShmPointCloudClient(logger=LOGGER)

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

    def _teardown(self) -> None:
        if self._simple_nav_cancel is not None:
            self._simple_nav_cancel.set()
        unregister_bridge(self.name)
        unregister_nav_viz(self.name)
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._manager = None
        self._shm_lidar.close()
        self._shm_lidar = ShmPointCloudClient(logger=LOGGER)
        self._viz = None
        self._runtime = None

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        ext = ExternalNavConfig.from_dict(struct_to_dict(config.attributes))
        bridge_cfg = ext.bridge
        self._cfg = ext.nav  # NavServiceBase drives nav from the NavConfig

        self._base = cast(Base, dependencies[Base.get_resource_name(ext.nav.base)])
        slam = cast(SLAM, dependencies[SLAM.get_resource_name(ext.slam_service)])
        cameras = {
            lidar.name: cast(
                Camera, dependencies[Camera.get_resource_name(lidar.name)]
            )
            for lidar in bridge_cfg.lidars
        }

        self._teardown()

        map_store = MapStore(bridge_cfg.maps_dir)
        active = bridge_cfg.active_map or map_store.get_active_map_name() or "default"
        map_store.get_or_create_map(active)
        map_store.set_active_map(active)

        if ext.nav.uses_builtin_nav():
            self._configure_builtin(ext, slam, cameras, map_store)
            return

        self._configure_nav2(ext, slam, cameras, map_store, dependencies)

    def _configure_builtin(
        self,
        ext: ExternalNavConfig,
        slam,
        cameras: dict,
        map_store: MapStore,
    ) -> None:
        loop = asyncio.get_event_loop()
        viz = NavVizStore()
        self._viz = viz
        world = ViamWorldIO(
            slam=slam,
            base=self._base,
            loop=loop,
            cameras=cameras,
            lidars=ext.bridge.lidars,
            base_velocity_convention=ext.bridge.base_velocity_convention,
            viz=viz,
            shm_lidar=self._shm_lidar,
            scan_max_age_s=float(ext.bridge.scan_max_age_s or 2.0),
            logger=lambda m: LOGGER.info(m),
        )
        navigator = make_builtin_navigator(
            world, ext.nav, logger=lambda m: LOGGER.info(m)
        )
        host = BuiltinNavHost(navigator, world, viz, nav_cfg=ext.nav)
        self._manager = host
        self._runtime = SlamRuntime(
            host,
            map_store,
            ext.bridge,
            {"status": "viam"},
            cameras=cameras,
            shm_lidar=self._shm_lidar,
        )
        register_nav_viz(self.name, viz)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation-external '{self.name}' configured "
            f"({ext.nav.kinematics}, nav_backend=builtin, ROS-free) against "
            f"SLAM service {ext.slam_service!r}"
        )

    def _configure_nav2(
        self,
        ext: ExternalNavConfig,
        slam,
        cameras: dict,
        map_store: MapStore,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> None:
        from ..ros.manager import RosManager
        from ..ros.odom_source import TypedMovementSensorOdom, TypedOdomConfig
        from ..ros.sensor_io import build_io_provider

        bridge_cfg = ext.bridge
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
            shm_lidar=self._shm_lidar,
        )

        self._manager = RosManager(bridge_cfg, logger=LOGGER, external_slam=slam)
        loop = asyncio.get_event_loop()
        self._manager.start(io, loop, nav_cfg=ext.nav)
        node = self._manager.node
        if node is not None:
            node._io = build_io_provider(
                base=self._base,
                cameras=cameras,
                cfg=bridge_cfg,
                movement_sensor=movement_sensor,
                heading_sensor=heading_sensor,
                skip_get_laser_scan=set(),
                odom_reader=odom_reader,
                logger=LOGGER,
                record_cmd_vel=node.record_cmd_vel,
                shm_lidar=self._shm_lidar,
            )

        loc_check = (
            node._external.localization_check
            if node is not None and node._external is not None
            else {"status": "unknown"}
        )
        self._runtime = SlamRuntime(
            self._manager,
            map_store,
            bridge_cfg,
            loc_check,
            cameras=cameras,
            shm_lidar=self._shm_lidar,
        )
        register_bridge(
            self.name,
            lambda: self._manager.node if self._manager is not None else None,
        )

        self._manager.set_nav_config(ext.nav)
        params_path = self._write_nav2_params(ext.nav)
        self._manager.ensure_nav2_async(ext.nav, params_path)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation-external '{self.name}' configured "
            f"({ext.nav.kinematics}, nav_backend=nav2) against "
            f"SLAM service {ext.slam_service!r}; Nav2 starting in background"
        )

    async def close(self) -> None:
        await self._cancel_simple_nav()
        self._teardown()


Registry.register_resource_creator(
    Motion.API,
    RosNavigationExternal.MODEL,
    ResourceCreatorRegistration(
        RosNavigationExternal.new, RosNavigationExternal.validate_config
    ),
)

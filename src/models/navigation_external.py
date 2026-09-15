"""External-SLAM navigation model: ``viam-labs:nav-stack:navigation-external``.

Like ``viam-labs:nav-stack:navigation`` (same ``rdk:service:motion`` + DoCommand
surface, shared :class:`~.nav_core.NavServiceBase`), but instead of borrowing
the built-in SLAM model's in-process runtime it drives navigation from an
**arbitrary Viam ``rdk:service:slam``** dependency.

ROS-free: map and pose come from the SLAM service (``get_grid`` / ``GetPosition``),
scans from lidar cameras / shm, and drive from ``Base.SetVelocity``.
"""
from __future__ import annotations

import asyncio
from typing import ClassVar, Mapping, Optional, Sequence, cast

from typing_extensions import Self

from viam.components.base import Base
from viam.components.camera import Camera
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
from ..shm.lidar import ShmPointCloudClient
from ..runtime import (
    SlamRuntime,
    register_nav_host,
    register_nav_viz,
    unregister_nav_host,
    unregister_nav_viz,
)
from .nav_core import NavServiceBase

LOGGER = getLogger(__name__)


class ExternalNavigationService(NavServiceBase):
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
        unregister_nav_viz(self.name)
        unregister_nav_host(self.name)
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

        self._configure_builtin(ext, slam, cameras, map_store)

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
            obstacles_only_period_s=ext.nav.obstacles_only_period_s(),
            drive_timeout_s=float(getattr(ext.nav.builtin, "drive_timeout_s", 5.0)),
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
        register_nav_host(self.name, host)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation-external '{self.name}' configured "
            f"({ext.nav.kinematics}, nav_backend=builtin) against "
            f"SLAM service {ext.slam_service!r}"
        )

    async def close(self) -> None:
        await self._cancel_simple_nav()
        self._teardown()


Registry.register_resource_creator(
    Motion.API,
    ExternalNavigationService.MODEL,
    ResourceCreatorRegistration(
        ExternalNavigationService.new, ExternalNavigationService.validate_config
    ),
)

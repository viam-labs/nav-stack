"""External-SLAM navigation model: ``viam-labs:nav-stack:navigation-external``.

Like ``viam-labs:nav-stack:navigation`` (same standard ``rdk:service:navigation``
API via :class:`~.nav_api.NavApiMixin`, same Nav2 orchestration via
:class:`~.nav_core.NavCoreMixin`), but instead of borrowing the built-in SLAM
model's in-process runtime it drives Nav2 from an **arbitrary Viam
``rdk:service:slam``** dependency.

The external ROS runtime (own sensor bridge + ``ExternalSlamPublisher`` +
typed-odom reader + ``SlamRuntime``) is assembled by
:func:`~.external_runtime.build_external_runtime`, shared with the built-in
``navigation`` model's runtime resolution only at the ``NavCoreMixin`` seam.
"""
from __future__ import annotations

import asyncio
from typing import ClassVar, Mapping, Optional, Sequence

from typing_extensions import Self

from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.navigation import Navigation
from viam.utils import struct_to_dict

from ..config import ExternalNavConfig
from ..ros.manager import RosManager
from ..runtime import register_bridge, unregister_bridge
from .external_runtime import build_external_runtime
from .nav_api import NavApiMixin
from .nav_core import NavCoreMixin

LOGGER = getLogger(__name__)


class RosNavigationExternal(NavApiMixin, NavCoreMixin, Navigation):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam-labs", "nav-stack"), "navigation-external"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._manager: Optional[RosManager] = None
        self._runtime = None
        self._external_slam = None  # the Viam SLAM dep, for annotation reads

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

    def _resolve_runtime(self):
        return self._runtime

    async def _get_annotations(self) -> dict:
        """Read annotations from the external SLAM service (e.g. rtabmap's
        get_annotations). Empty if it has no annotation support."""
        if self._external_slam is None:
            return {}
        try:
            resp = await self._external_slam.do_command({"command": "get_annotations"})
        except Exception:  # noqa: BLE001 - SLAM may not implement get_annotations
            return {}
        return resp.get("annotations", {}) if isinstance(resp, dict) else {}

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        ext = ExternalNavConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = ext.nav  # NavCoreMixin drives Nav2 from the NavConfig
        self._reset_nav_state()

        # Rebuild the ROS stack from scratch on every reconfigure.
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

        self._base, self._manager, self._runtime, self._external_slam = (
            build_external_runtime(
                ext, dependencies, loop=asyncio.get_event_loop(), logger=LOGGER
            )
        )

        # Publish the live bridge node so a nav-camera can find it and render
        # this nav's costmap/plans.
        node = self._manager.node
        if node is not None:
            register_bridge(self.name, node)

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
        unregister_bridge(self.name)
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._runtime = None
        await super().close()  # -> NavApiMixin.close (stops waypoint driver) -> Navigation


Registry.register_resource_creator(
    Navigation.API,
    RosNavigationExternal.MODEL,
    ResourceCreatorRegistration(
        RosNavigationExternal.new, RosNavigationExternal.validate_config
    ),
)

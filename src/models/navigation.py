"""Navigation service model: ``viam-labs:nav-stack:navigation``.

A Viam ``rdk:service:motion`` for map-frame navigation. By default it uses the
in-module builtin navigator (``nav_backend: builtin``) with **ViamWorldIO**
(SLAM ``get_grid`` / ``GetPosition``, lidar cameras, base SetVelocity) — no
Nav2 and no ROS topics on the nav path. Set ``nav_backend: nav2`` to launch
ROS2 Nav2 against the SLAM service's shared ROS context instead.

The built-in SLAM service may still use ROS/slam_toolbox internally; this model
only borrows that shared runtime for map store / (optional) Nav2.

Exposes:

* Motion ``MoveOnMap`` / ``StopPlan`` / ``GetPlan`` / ``ListPlanStatuses`` /
  ``GetPose``
* DoCommand: locations CRUD, zones CRUD, ``navigate_*`` / ``go_to_*``, cancel,
  status, ``get_costmap``, and Nav2 ops (``restart_nav2``, …) when using Nav2
"""
from __future__ import annotations

import asyncio
from typing import ClassVar, Mapping, Optional, Sequence, cast

from typing_extensions import Self

from viam.components.base import Base
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.motion import Motion
from viam.services.slam import SLAM
from viam.utils import struct_to_dict

from ..config import NavConfig
from ..nav_builtin import NavVizStore, ViamWorldIO
from ..runtime import (
    get_slam,
    register_bridge,
    register_nav_viz,
    unregister_bridge,
    unregister_nav_viz,
)

# Re-exported for backwards-compatible imports (tests + any external callers
# still do ``from src.models.navigation import _sync_mppi_model_dt`` etc.).
from .nav_core import (  # noqa: F401
    NavServiceBase,
    _apply_diffdrive_controller,
    _apply_local_costmap_size,
    _apply_nav2_tuning,
    _apply_overrides,
    _apply_velocity_limits,
    _deep_merge,
    _find_template_section_paths,
    _mppi_profile_snapshot,
    _nav_status_to_plan_state,
    _normalize_nav2_user_params,
    _set_obstacle_sources,
    _sync_mppi_model_dt,
    _sync_smoother_reverse_to_mppi,
    _tune_nav2_bt_xml,
    _validate_nav2_params_structure,
    _write_nav2_bt_xml,
)

LOGGER = getLogger(__name__)


class RosNavigation(NavServiceBase):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "navigation")

    def __init__(self, name: str):
        super().__init__(name)
        self._viz: Optional[NavVizStore] = None
        self._slam_resource = None

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
        cfg = NavConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    # -- runtime resolution --------------------------------------------------
    def _resolve_runtime(self):
        cfg = self._require_cfg()
        return get_slam(cfg.slam_service)

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = NavConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = cfg
        self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])
        self._slam_resource = cast(
            SLAM, dependencies[SLAM.get_resource_name(cfg.slam_service)]
        )

        runtime = get_slam(cfg.slam_service)
        if runtime is None:
            raise RuntimeError(
                f"SLAM service {cfg.slam_service!r} not found; it must be configured "
                "and started before the navigation service"
            )
        if self._simple_nav_cancel is not None:
            self._simple_nav_cancel.set()

        unregister_nav_viz(self.name)
        unregister_bridge(self.name)
        self._viz = None

        if cfg.uses_builtin_nav():
            viz = NavVizStore()
            self._viz = viz
            loop = asyncio.get_event_loop()
            world = ViamWorldIO(
                slam=self._slam_resource,
                base=self._base,
                loop=loop,
                cameras=runtime.cameras,
                lidars=runtime.slam_cfg.lidars,
                base_velocity_convention=runtime.slam_cfg.base_velocity_convention,
                viz=viz,
                shm_lidar=runtime.shm_lidar,
                scan_max_age_s=float(
                    getattr(runtime.slam_cfg, "scan_max_age_s", 2.0) or 2.0
                ),
                logger=lambda m: LOGGER.info(m),
            )
            runtime.manager.set_builtin_world(world)
            register_nav_viz(self.name, viz)
            # Keep bridge registration so legacy callers still find the SLAM
            # bridge; nav-camera prefers nav viz via get_nav_view.
            slam_service = cfg.slam_service
            register_bridge(
                self.name,
                lambda: (
                    rt.manager.node
                    if (rt := get_slam(slam_service)) is not None
                    else None
                ),
            )
            runtime.manager.set_nav_config(cfg)
            self._refresh_zone_masks()
            LOGGER.info(
                f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}, "
                f"nav_backend=builtin, ViamWorldIO)"
            )
            return

        runtime.manager.set_builtin_world(None)
        slam_service = cfg.slam_service
        register_bridge(
            self.name,
            lambda: (
                rt.manager.node if (rt := get_slam(slam_service)) is not None else None
            ),
        )
        runtime.manager.set_nav_config(cfg)
        params_path = self._write_nav2_params(cfg)
        runtime.manager.ensure_nav2_async(cfg, params_path)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}, "
            f"nav_backend=nav2); Nav2 starting in background"
        )

    async def close(self) -> None:
        await self._cancel_simple_nav()
        unregister_bridge(self.name)
        unregister_nav_viz(self.name)
        self._viz = None


Registry.register_resource_creator(
    Motion.API,
    RosNavigation.MODEL,
    ResourceCreatorRegistration(RosNavigation.new, RosNavigation.validate_config),
)

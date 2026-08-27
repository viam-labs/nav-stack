"""Navigation service model: ``viam-labs:nav-stack:navigation``.

Default ``nav_backend: builtin`` drives the in-module navigator over Viam APIs
only (``ViamWorldIO`` + ``BuiltinNavHost``): SLAM ``get_grid`` / ``GetPosition``,
lidar shm/cameras, ``Base.SetVelocity``. No RosManager, no bridge registration,
no Nav2.

Set ``nav_backend: nav2`` to launch ROS2 Nav2 against the SLAM service's shared
ROS context (legacy).

Note: the companion ``viam-labs:nav-stack:slam`` model still uses ROS
(slam_toolbox) for mapping. For a fully ROS-free robot, use
``navigation-external`` against any non-ROS ``rdk:service:slam``.
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
from ..nav_builtin import (
    BuiltinNavHost,
    NavVizStore,
    ViamWorldIO,
    make_builtin_navigator,
)
from ..runtime import (
    SlamRuntime,
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
        # When builtin: a SlamRuntime whose ``manager`` is BuiltinNavHost (no ROS).
        self._builtin_runtime: Optional[SlamRuntime] = None

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
        if self._builtin_runtime is not None:
            return self._builtin_runtime
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

        slam_rt = get_slam(cfg.slam_service)
        if slam_rt is None:
            raise RuntimeError(
                f"SLAM service {cfg.slam_service!r} not found; it must be configured "
                "and started before the navigation service"
            )
        if self._simple_nav_cancel is not None:
            self._simple_nav_cancel.set()

        unregister_nav_viz(self.name)
        unregister_bridge(self.name)
        self._viz = None
        if self._builtin_runtime is not None:
            try:
                self._builtin_runtime.manager.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._builtin_runtime = None

        if cfg.uses_builtin_nav():
            # Fully ROS-free nav surface: never touch slam_rt.manager (RosManager).
            viz = NavVizStore()
            self._viz = viz
            loop = asyncio.get_event_loop()
            world = ViamWorldIO(
                slam=self._slam_resource,
                base=self._base,
                loop=loop,
                cameras=slam_rt.cameras,
                lidars=slam_rt.slam_cfg.lidars,
                base_velocity_convention=slam_rt.slam_cfg.base_velocity_convention,
                viz=viz,
                shm_lidar=slam_rt.shm_lidar,
                scan_max_age_s=float(
                    getattr(slam_rt.slam_cfg, "scan_max_age_s", 2.0) or 2.0
                ),
                logger=lambda m: LOGGER.info(m),
            )
            navigator = make_builtin_navigator(
                world, cfg, logger=lambda m: LOGGER.info(m)
            )
            host = BuiltinNavHost(navigator, world, viz, nav_cfg=cfg)
            self._builtin_runtime = SlamRuntime(
                host,
                slam_rt.map_store,
                slam_rt.slam_cfg,
                slam_rt.localization_check,
                cameras=slam_rt.cameras,
                shm_lidar=slam_rt.shm_lidar,
            )
            register_nav_viz(self.name, viz)
            self._refresh_zone_masks()
            LOGGER.info(
                f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}, "
                f"nav_backend=builtin, ROS-free ViamWorldIO)"
            )
            return

        # Legacy Nav2 path: share the SLAM RosManager / bridge.
        if hasattr(slam_rt.manager, "set_builtin_world"):
            slam_rt.manager.set_builtin_world(None)
        slam_service = cfg.slam_service
        register_bridge(
            self.name,
            lambda: (
                rt.manager.node if (rt := get_slam(slam_service)) is not None else None
            ),
        )
        slam_rt.manager.set_nav_config(cfg)
        params_path = self._write_nav2_params(cfg)
        slam_rt.manager.ensure_nav2_async(cfg, params_path)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}, "
            f"nav_backend=nav2); Nav2 starting in background"
        )

    async def close(self) -> None:
        await self._cancel_simple_nav()
        unregister_bridge(self.name)
        unregister_nav_viz(self.name)
        if self._builtin_runtime is not None:
            try:
                self._builtin_runtime.manager.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._builtin_runtime = None
        self._viz = None


Registry.register_resource_creator(
    Motion.API,
    RosNavigation.MODEL,
    ResourceCreatorRegistration(RosNavigation.new, RosNavigation.validate_config),
)

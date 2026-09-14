"""Navigation service model: ``viam-labs:nav-stack:navigation``.

Builtin MoveOnMap over Viam APIs only (``ViamWorldIO`` + ``BuiltinNavHost``):
SLAM ``get_grid`` / ``GetPosition``, lidar shm/cameras, ``Base.SetVelocity``.
No ROS / Nav2. Or use ``navigation-external`` against any ``rdk:service:slam``.
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
    get_slam_service,
    register_nav_host,
    register_nav_viz,
    unregister_nav_host,
    unregister_nav_viz,
)
from .nav_core import NavServiceBase, _nav_status_to_plan_state  # noqa: F401

LOGGER = getLogger(__name__)


def _sync_slam_pose_provider(slam_service_name: str):
    """Sync map pose from the live in-process SLAM service (no event-loop hop).

    The motion dependency is often a gRPC client stub even for same-module SLAM.
    Calling ``GetPosition`` every control tick then contends with ``SetVelocity``
    on the shared loop — commands are computed while the base never moves. Always
    re-resolve the registered service / runtime so SLAM reconfigure cannot leave
    a stale host.
    """

    def _get():
        svc = get_slam_service(slam_service_name)
        if svc is not None:
            fn = getattr(type(svc), "get_position_pose2d", None)
            if callable(fn):
                try:
                    return fn(svc)
                except Exception:  # noqa: BLE001
                    pass
        rt = get_slam(slam_service_name)
        if rt is None or rt.manager is None:
            return None
        manager = rt.manager
        getter = getattr(manager, "get_pose_in_map", None)
        if callable(getter):
            return getter()
        return None

    return _get


def _localization_hold_provider(slam_service_name: str):
    """Stop nav while SLAM is awaiting confirm on a large pose jump."""
    from ..nav.pose_jump_gate import should_hold_drive_for_pose_jump

    def _get():
        svc = get_slam_service(slam_service_name)
        if svc is not None:
            for attr in ("_last_relocalize_check", "_last_revisit_check"):
                check = getattr(svc, attr, None)
                if should_hold_drive_for_pose_jump(check):
                    return dict(check)
        rt = get_slam(slam_service_name)
        if rt is not None and should_hold_drive_for_pose_jump(rt.localization_check):
            return dict(rt.localization_check)
        return None

    return _get


def _in_process_map_provider(slam_service_name: str):
    """Sync occupancy dict from the current SLAM host (re-resolves each call)."""

    def _get():
        rt = get_slam(slam_service_name)
        if rt is None or rt.manager is None:
            return None
        manager = rt.manager
        getter = getattr(manager, "get_map", None)
        if callable(getter):
            return getter()
        return None

    return _get


class NavigationService(NavServiceBase):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "navigation")

    def __init__(self, name: str):
        super().__init__(name)
        self._viz: Optional[NavVizStore] = None
        self._slam_resource = None
        # SlamRuntime whose ``manager`` is BuiltinNavHost.
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
        unregister_nav_host(self.name)
        self._viz = None
        if self._builtin_runtime is not None:
            try:
                self._builtin_runtime.manager.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._builtin_runtime = None

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
            drive_timeout_s=float(getattr(cfg.builtin, "drive_timeout_s", 5.0)),
            pose_provider=_sync_slam_pose_provider(cfg.slam_service),
            map_provider=_in_process_map_provider(cfg.slam_service),
            localization_hold_provider=_localization_hold_provider(cfg.slam_service),
            scan_provider=(
                (lambda max_age_s, s=slam_rt.sim_sensors: s.get_scan(max_age_s))
                if slam_rt.sim_sensors is not None
                else None
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
            sim_sensors=slam_rt.sim_sensors,
        )
        register_nav_viz(self.name, viz)
        register_nav_host(self.name, host)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}, "
            f"nav_backend=builtin, ViamWorldIO)"
        )

    async def close(self) -> None:
        await self._cancel_simple_nav()
        unregister_nav_viz(self.name)
        unregister_nav_host(self.name)
        if self._builtin_runtime is not None:
            try:
                self._builtin_runtime.manager.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._builtin_runtime = None
        self._viz = None


Registry.register_resource_creator(
    Motion.API,
    NavigationService.MODEL,
    ResourceCreatorRegistration(NavigationService.new, NavigationService.validate_config),
)

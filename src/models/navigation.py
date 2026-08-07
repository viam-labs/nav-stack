"""Navigation service model: ``viam-labs:nav-stack:navigation``.

A Viam ``rdk:service:motion`` that wraps ROS2 Nav2. It launches Nav2 against the
SLAM service's shared ROS context and exposes:

* Motion ``MoveOnMap`` / ``StopPlan`` / ``GetPlan`` / ``ListPlanStatuses`` /
  ``GetPose`` (map-frame Nav2 navigation)
* DoCommand: locations CRUD, zones CRUD, ``navigate_*`` / ``go_to_*``, cancel,
  status, ``get_costmap``, and Nav2 ops (``restart_nav2``, …)

Physical obstacle avoidance is automatic via Nav2's costmaps (live ``/scan`` data).

This model borrows the built-in SLAM model's shared in-process ROS runtime
(looked up in the process-global registry by ``slam_service`` name). All of the
Motion + DoCommand + Nav2 orchestration lives in
:class:`~.nav_core.NavServiceBase`; this model only supplies the registry-backed
runtime resolution.
"""
from __future__ import annotations

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
from viam.utils import struct_to_dict

from ..config import NavConfig
from ..runtime import get_slam, register_bridge, unregister_bridge

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

        runtime = get_slam(cfg.slam_service)
        if runtime is None:
            raise RuntimeError(
                f"SLAM service {cfg.slam_service!r} not found; it must be configured "
                "and started before the navigation service"
            )
        runtime.manager.set_nav_config(cfg)
        # Publish the shared bridge node so a nav-camera can find it and render
        # this nav's costmap/plans.
        if runtime.manager.node is not None:
            register_bridge(self.name, runtime.manager.node)
        params_path = self._write_nav2_params(cfg)
        # Nav2 bringup (with retries) can take minutes on a Pi; run it in the
        # background so reconfigure returns within viam-server's deadline.
        runtime.manager.ensure_nav2_async(cfg, params_path)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}); "
            "Nav2 starting in background"
        )

    async def close(self) -> None:
        # The bridge node itself is owned by the SLAM service; only drop our
        # nav-camera registration pointer.
        unregister_bridge(self.name)


Registry.register_resource_creator(
    Motion.API,
    RosNavigation.MODEL,
    ResourceCreatorRegistration(RosNavigation.new, RosNavigation.validate_config),
)

"""Navigation model: ``viam-labs:nav-stack:navigation``.

Wraps ROS2 Nav2 against the **built-in** SLAM model's shared in-process ROS
runtime (looked up in the process-global registry by ``slam_service`` name) and
exposes Viam's standard ``rdk:service:navigation`` API — so a Navigation-API
client (e.g. a webapp) can drive it. The API surface is shared with
``navigation-external`` via :class:`~.nav_api.NavApiMixin`; the Nav2 orchestration
+ rich ``DoCommand`` surface (locations, zones, go/navigate, status) live in
:class:`~.nav_core.NavCoreMixin`. This model only supplies registry-backed runtime
resolution.
"""
from __future__ import annotations

from typing import ClassVar, Mapping, Sequence, cast

from typing_extensions import Self

from viam.components.base import Base
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.navigation import Navigation
from viam.utils import struct_to_dict

from ..config import NavConfig
from ..runtime import get_slam
from .nav_api import NavApiMixin

# Re-exported for backwards-compatible imports (tests + any external callers
# still do ``from src.models.navigation import _sync_mppi_model_dt`` etc.).
from .nav_core import (  # noqa: F401
    NavCoreMixin,
    _apply_local_costmap_size,
    _apply_nav2_tuning,
    _apply_overrides,
    _apply_velocity_limits,
    _deep_merge,
    _find_template_section_paths,
    _normalize_nav2_user_params,
    _set_obstacle_sources,
    _sync_mppi_model_dt,
    _validate_nav2_params_structure,
)

LOGGER = getLogger(__name__)


class RosNavigation(NavApiMixin, NavCoreMixin, Navigation):
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

    async def _get_annotations(self) -> dict:
        """Read annotations from the built-in SLAM model's per-map store."""
        runtime = self._resolve_runtime()
        handle = runtime.map_store.active_handle() if runtime is not None else None
        if handle is None:
            return {}
        from ..nav.annotations import AnnotationStore

        return AnnotationStore(handle.annotations_path).feature_collection()

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = NavConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = cfg
        self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])
        self._reset_nav_state()

        runtime = get_slam(cfg.slam_service)
        if runtime is None:
            raise RuntimeError(
                f"SLAM service {cfg.slam_service!r} not found; it must be configured "
                "and started before the navigation service"
            )
        runtime.manager.set_nav_config(cfg)
        params_path = self._write_nav2_params(cfg)
        # Nav2 bringup (with retries) can take minutes on a Pi; run it in the
        # background so reconfigure returns within viam-server's deadline.
        runtime.manager.ensure_nav2_async(cfg, params_path)
        self._refresh_zone_masks()
        LOGGER.info(
            f"nav-stack navigation '{self.name}' configured ({cfg.kinematics}); "
            "Nav2 starting in background"
        )


Registry.register_resource_creator(
    Navigation.API,
    RosNavigation.MODEL,
    ResourceCreatorRegistration(RosNavigation.new, RosNavigation.validate_config),
)

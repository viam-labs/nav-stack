"""Autonomous exploration drive: ``viam-labs:nav-stack:explore``.

A generic service that drives the robot around an unmapped space for a time
budget so the builtin SLAM maps it, then saves the map and returns home.
"""
from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional, Sequence, cast

from typing_extensions import Self

from viam.components.base import Base
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.navigation import Navigation
from viam.services.slam import SLAM
from viam.utils import struct_to_dict

from ..config import ExploreConfig

LOGGER = getLogger(__name__)


class ExploreService(Generic):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "explore")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[ExploreConfig] = None
        self._slam = None
        self._nav = None
        self._base = None

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
        cfg = ExploreConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = ExploreConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = cfg
        self._slam = cast(SLAM, dependencies[SLAM.get_resource_name(cfg.slam_service)])
        self._nav = cast(
            Navigation, dependencies[Navigation.get_resource_name(cfg.navigation_service)]
        )
        self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])

    async def do_command(
        self, command: Mapping[str, Any], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, Any]:
        cmd = command.get("command")
        if cmd == "status":
            return {"phase": "idle"}
        return {"error": "unknown command"}


Registry.register_resource_creator(
    Generic.API,
    ExploreService.MODEL,
    ResourceCreatorRegistration(ExploreService.new, ExploreService.validate_config),
)

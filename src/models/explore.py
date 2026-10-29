"""Autonomous exploration drive: ``viam-labs:nav-stack:explore``.

A generic service that drives the robot around an unmapped space for a time
budget so the builtin SLAM maps it, then saves the map and returns home.
"""
from __future__ import annotations

import asyncio
import time
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
from ..explore.clients import ViamNavClient, ViamSlamClient
from ..explore.run import ExploreRun, RunParams

LOGGER = getLogger(__name__)


class ExploreService(Generic):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "explore")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[ExploreConfig] = None
        self._slam = None
        self._nav = None
        self._base = None
        self._run: Optional[ExploreRun] = None
        self._run_task: Optional[asyncio.Task] = None

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
        if self._run is not None and self._run_active():
            self._run.request_stop()
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
            return {"phase": "idle"} if self._run is None else self._run.status()
        if cmd == "start":
            return await self._start(command)
        if cmd == "stop":
            if self._run is None or not self._run_active():
                return {"error": "no active run"}
            self._run.request_stop()
            return {"stopping": True}
        return {"error": "unknown command"}

    def _run_active(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    async def _start(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._run_active():
            return {"error": "a run is already active"}
        cfg = self._cfg
        if cfg is None or self._slam is None or self._nav is None or self._base is None:
            return {"error": "explore service not configured"}
        try:
            duration_s = float(command["duration_s"])
        except (KeyError, TypeError, ValueError):
            return {"error": "start requires duration_s"}
        if duration_s <= 0:
            return {"error": "duration_s must be positive"}
        nav = ViamNavClient(self._nav)
        slam = ViamSlamClient(self._slam)
        radius = (await nav.status()).robot_radius_m or 0.3
        params = RunParams(
            duration_s=duration_s,
            max_linear_mps=cfg.max_linear_mps,
            max_angular_dps=cfg.max_angular_dps,
            home_tolerance_m=cfg.home_tolerance_m,
            structure_fraction=cfg.structure_fraction,
            revisit_fraction=cfg.revisit_fraction,
            well_seen_m=cfg.well_seen_m,
            keyframe_max=cfg.keyframe_max,
            robot_radius_m=radius,
        )
        base = self._base

        async def stop_base() -> None:
            await base.stop()

        run = ExploreRun(
            nav, slam, params, clock=time.monotonic, sleep=asyncio.sleep, stop_base=stop_base
        )
        error = await run.preflight()
        if error is not None:
            return {"error": error}
        self._run = run
        self._run_task = asyncio.create_task(run.run())
        LOGGER.info("explore run started for %.0f s", duration_s)
        return {"started": True, "duration_s": duration_s}


Registry.register_resource_creator(
    Generic.API,
    ExploreService.MODEL,
    ResourceCreatorRegistration(ExploreService.new, ExploreService.validate_config),
)

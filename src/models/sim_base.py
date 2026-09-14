"""Simulated differential base: ``viam-labs:nav-stack:sim-base``.

``SetVelocity`` / ``Stop`` update the shared :class:`~..sim.world.SimWorld` so
teleop and builtin nav drive the raycast environment like a real Viam base.
"""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence

from typing_extensions import Self

from viam.components.base import Base, Vector3
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from ..config import (
    BASE_VELOCITY_CONVENTIONS,
    BASE_VELOCITY_MIR,
    BASE_VELOCITY_VIAM,
    BASE_VELOCITY_Y_FORWARD,
    viam_set_velocity_to_body_twist,
)
from ..geom import conversions as conv
from ..sim import (
    SimWorld,
    get_sim_world,
    load_sim_map,
    make_builtin_corridor,
    register_sim_world,
)

LOGGER = getLogger(__name__)


class SimBase(Base):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "sim-base")

    def __init__(self, name: str):
        super().__init__(name)
        self._world: Optional[SimWorld] = None
        self._convention = BASE_VELOCITY_VIAM
        self._world_name = "default"
        self._moving = False

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        base = cls(config.name)
        base.reconfigure(config, dependencies)
        return base

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        del config
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        del dependencies
        attrs = struct_to_dict(config.attributes)
        world_name = str(attrs.get("world_name", "default") or "default")
        convention = attrs.get("base_velocity_convention", BASE_VELOCITY_VIAM)
        if convention not in BASE_VELOCITY_CONVENTIONS:
            raise ValueError(
                f"base_velocity_convention must be one of "
                f"{sorted(BASE_VELOCITY_CONVENTIONS)}"
            )
        if convention == BASE_VELOCITY_MIR:
            convention = BASE_VELOCITY_VIAM
        self._convention = str(convention)
        self._world_name = world_name

        existing = get_sim_world(world_name)
        map_path = attrs.get("map_path")
        seed = conv.Pose2D(
            float(attrs.get("seed_x", 1.0)),
            float(attrs.get("seed_y", 1.0)),
            float(attrs.get("seed_theta", 0.0)),
        )
        scan_bins = int(attrs.get("scan_bins", 360))
        range_min = float(attrs.get("range_min", 0.05))
        range_max = float(attrs.get("range_max", 20.0))

        if existing is not None:
            self._world = existing
            existing.set_seed(seed)
            existing.scan_bins = scan_bins
            existing.range_min = range_min
            existing.range_max = range_max
            if map_path:
                LOGGER.info(
                    f"sim-base '{self.name}' reusing world {world_name!r}; "
                    "map_path ignored (already loaded)"
                )
        else:
            sim_map = (
                load_sim_map(str(map_path))
                if map_path
                else make_builtin_corridor()
            )
            self._world = SimWorld(
                sim_map,
                seed_pose=seed,
                scan_bins=scan_bins,
                range_min=range_min,
                range_max=range_max,
                name=world_name,
            )
            register_sim_world(world_name, self._world)
        LOGGER.info(
            f"nav-stack sim-base '{self.name}' world={world_name!r} "
            f"convention={self._convention}"
        )

    def _require_world(self) -> SimWorld:
        if self._world is None:
            raise RuntimeError(f"sim-base '{self.name}' not configured")
        return self._world

    async def set_velocity(
        self,
        linear: Vector3,
        angular: Vector3,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        vx, vy, vtheta = viam_set_velocity_to_body_twist(
            linear.x, linear.y, angular.z, self._convention
        )
        self._require_world().set_velocity_ros(vx, vy, vtheta)
        self._moving = abs(vx) > 1e-6 or abs(vy) > 1e-6 or abs(vtheta) > 1e-6

    async def stop(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        self._require_world().stop()
        self._moving = False

    async def is_moving(self) -> bool:
        return self._moving

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Base.Properties:
        del timeout, kwargs
        return Base.Properties(
            width_meters=0.4,
            turning_radius_meters=0.0,
            wheel_circumference_meters=0.3,
        )

    async def set_power(
        self,
        linear: Vector3,
        angular: Vector3,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Map normalized power roughly into SetVelocity (sim teleop helper)."""
        del extra, timeout, kwargs
        scale_mm = 500.0  # 0.5 m/s at power=1
        scale_deg = 57.3  # ~1 rad/s at power=1
        await self.set_velocity(
            Vector3(x=linear.x * scale_mm, y=linear.y * scale_mm, z=0.0),
            Vector3(x=0.0, y=0.0, z=angular.z * scale_deg),
        )

    async def move_straight(
        self,
        distance: int,
        velocity: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        dist_m = float(distance) / 1000.0
        vel_mps = float(velocity) / 1000.0
        if abs(vel_mps) < 1e-9 or abs(dist_m) < 1e-9:
            await self.stop()
            return
        duration = abs(dist_m / vel_mps)
        sign = 1.0 if dist_m * vel_mps >= 0 else -1.0
        forward_mm = abs(vel_mps) * 1000.0 * sign
        if self._convention in BASE_VELOCITY_Y_FORWARD:
            linear = Vector3(x=0.0, y=forward_mm, z=0.0)
        else:
            linear = Vector3(x=forward_mm, y=0.0, z=0.0)
        await self.set_velocity(linear, Vector3(x=0.0, y=0.0, z=0.0))
        await asyncio.sleep(duration)
        await self.stop()

    async def spin(
        self,
        angle: float,
        velocity: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        del extra, timeout, kwargs
        if abs(velocity) < 1e-9 or abs(angle) < 1e-9:
            await self.stop()
            return
        duration = abs(float(angle) / float(velocity))
        await self.set_velocity(
            Vector3(x=0.0, y=0.0, z=0.0),
            Vector3(x=0.0, y=0.0, z=float(velocity)),
        )
        await asyncio.sleep(duration)
        await self.stop()

    async def do_command(
        self,
        command: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, Any]:
        del timeout, kwargs
        cmd = str(command.get("command") or command.get("cmd") or "").strip().lower()
        world = self._require_world()
        if cmd in {"reset", "reset_pose"}:
            pose = conv.Pose2D(
                float(command.get("x", world._seed.x)),  # noqa: SLF001
                float(command.get("y", world._seed.y)),  # noqa: SLF001
                float(command.get("theta", world._seed.theta)),  # noqa: SLF001
            )
            world.reset(pose)
            self._moving = False
            return {"ok": True, "x": pose.x, "y": pose.y, "theta": pose.theta}
        if cmd in {"get_pose", "pose"}:
            pose = world.get_pose()
            return {"x": pose.x, "y": pose.y, "theta": pose.theta}
        raise Exception(f"unknown sim-base command: {cmd!r}")


Registry.register_resource_creator(
    Base.API,
    SimBase.MODEL,
    ResourceCreatorRegistration(SimBase.new, SimBase.validate_config),
)

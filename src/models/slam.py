"""SLAM service model: ``viam-labs:nav-stack:slam``.

Wraps slam_toolbox (via the ROS manager/bridge) to provide mapping and
localization for any Viam base, and exposes the standard Viam SLAM service API plus
map-management / mode / initial-pose commands through ``DoCommand``.
"""
from __future__ import annotations

import asyncio
import math
from typing import ClassVar, List, Mapping, Optional, Sequence, cast

import numpy as np
from typing_extensions import Self

from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.movement_sensor import MovementSensor
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.slam import SLAM, MappingMode, Pose
from viam.utils import ValueTypes, struct_to_dict

from ..config import MODE_LOCALIZING, MODE_MAPPING, SlamConfig
from ..nav.maps import MapStore
from ..ros import conversions as conv
from ..ros.bridge import IOProvider
from ..ros.manager import RosManager
from ..runtime import SlamRuntime, register_slam, unregister_slam

LOGGER = getLogger(__name__)


class RosSlam(SLAM):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "slam")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[SlamConfig] = None
        self._manager: Optional[RosManager] = None
        self._map_store: Optional[MapStore] = None
        self._base: Optional[Base] = None
        self._cameras: dict = {}
        self._movement_sensor: Optional[MovementSensor] = None

    # -- registration --------------------------------------------------------
    @classmethod
    def new(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        attrs = struct_to_dict(config.attributes)
        cfg = SlamConfig.from_dict(attrs)
        return cfg.required_dependencies()

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        cfg = SlamConfig.from_dict(attrs)
        self._cfg = cfg

        self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])
        self._cameras = {
            lidar.name: cast(
                Camera, dependencies[Camera.get_resource_name(lidar.name)]
            )
            for lidar in cfg.lidars
        }
        self._movement_sensor = (
            cast(
                MovementSensor,
                dependencies[MovementSensor.get_resource_name(cfg.movement_sensor)],
            )
            if cfg.movement_sensor
            else None
        )

        self._map_store = MapStore(cfg.maps_dir)
        active = cfg.active_map or self._map_store.get_active_map_name() or "default"
        self._map_store.get_or_create_map(active, resolution=cfg.slam_toolbox.resolution)
        self._map_store.set_active_map(active)

        # (Re)start the ROS stack.
        if self._manager is not None:
            self._manager.shutdown()
        self._manager = RosManager(cfg, logger=LOGGER)
        loop = asyncio.get_event_loop()
        self._manager.start(self._build_io(), loop)
        self._start_mode(cfg.mode)

        register_slam(
            self.name, SlamRuntime(self._manager, self._map_store, cfg)
        )
        LOGGER.info(f"nav-stack SLAM '{self.name}' configured in {cfg.mode} mode")

    # -- ROS IO --------------------------------------------------------------
    def _build_io(self) -> IOProvider:
        async def read_lidar_points(name: str):
            data = await self._cameras[name].get_point_cloud()
            raw = data[0] if isinstance(data, tuple) else data
            return conv.parse_pcd(raw)

        async def read_twist():
            if self._movement_sensor is None:
                return (0.0, 0.0, 0.0)
            lin = await self._movement_sensor.get_linear_velocity()
            ang = await self._movement_sensor.get_angular_velocity()
            return (float(lin.x), float(lin.y), math.radians(float(ang.z)))

        async def drive_base(vx: float, vy: float, vtheta: float):
            await self._base.set_velocity(
                linear=Vector3(x=vx * 1000.0, y=vy * 1000.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=math.degrees(vtheta)),
            )

        async def stop_base():
            await self._base.stop()

        return IOProvider(read_lidar_points, read_twist, drive_base, stop_base)

    def _start_mode(self, mode: str) -> None:
        assert self._manager and self._map_store and self._cfg
        handle = self._map_store.active_handle()
        stem = handle.serialization_stem if handle else None
        self._manager.start_slam(stem, mode)
        self._cfg.mode = mode

    # -- SLAM API ------------------------------------------------------------
    async def get_position(self, *, timeout: Optional[float] = None, **kwargs) -> Pose:
        node = self._manager.node if self._manager else None
        pose2d = node.get_pose_in_map() if node else None
        if pose2d is None:
            return Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
        x_mm, y_mm, theta_deg = conv.pose2d_to_viam_pose(pose2d)
        return Pose(x=x_mm, y=y_mm, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=theta_deg)

    async def get_point_cloud_map(
        self, return_edited_map: bool = False, *, timeout: Optional[float] = None, **kwargs
    ) -> List[bytes]:
        node = self._manager.node if self._manager else None
        grid = node.get_map() if node else None
        if not grid:
            return [conv.points_to_pcd(np.empty((0, 3)))]
        pcd = conv.occupancy_grid_to_pcd(
            grid["grid"], grid["resolution"], grid["origin_x"], grid["origin_y"]
        )
        return conv.chunk_bytes(pcd)

    async def get_internal_state(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> List[bytes]:
        handle = self._map_store.active_handle() if self._map_store else None
        if handle and handle.posegraph_path.exists():
            return conv.chunk_bytes(handle.posegraph_path.read_bytes())
        return [b""]

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> SLAM.Properties:
        mapping = self._cfg is not None and self._cfg.mode == MODE_MAPPING
        mode = (
            MappingMode.MAPPING_MODE_CREATE_NEW_MAP
            if mapping
            else MappingMode.MAPPING_MODE_LOCALIZE_ONLY
        )
        return SLAM.Properties(
            cloud_slam=False,
            mapping_mode=mode,
            internal_state_file_type=".posegraph",
            sensor_info=[],
        )

    # -- DoCommand -----------------------------------------------------------
    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        store = self._map_store
        mgr = self._manager
        if store is None or mgr is None:
            raise RuntimeError("SLAM service not configured")

        if cmd == "get_mode":
            return {"mode": self._cfg.mode if self._cfg else None}

        if cmd == "start_mapping":
            name = command.get("map")
            if name:
                store.get_or_create_map(str(name))
                store.set_active_map(str(name))
            self._start_mode(MODE_MAPPING)
            return {"status": "mapping", "map": store.get_active_map_name()}

        if cmd == "start_localizing":
            name = command.get("map")
            if name:
                store.set_active_map(str(name))
            handle = store.active_handle()
            if not handle or not handle.has_serialized_map():
                raise ValueError("active map has no saved data; map it first")
            self._start_mode(MODE_LOCALIZING)
            return {"status": "localizing", "map": store.get_active_map_name()}

        if cmd == "save_map":
            handle = store.active_handle()
            if not handle:
                raise ValueError("no active map")
            mgr.save_map(handle.serialization_stem)
            return {"status": "saved", "map": handle.name}

        if cmd == "set_initial_pose":
            pose = self._resolve_pose(command)
            mgr.set_initial_pose(pose)
            return {"status": "ok"}

        # -- map management --
        if cmd == "list_maps":
            return {"maps": store.list_maps()}
        if cmd == "get_active_map":
            return {"map": store.get_active_map_name()}
        if cmd == "set_active_map":
            store.set_active_map(str(command["map"]))
            return {"map": store.get_active_map_name()}
        if cmd == "rename_map":
            handle = store.rename_map(str(command["map"]), str(command["new_name"]))
            return {"map": handle.name}
        if cmd == "delete_map":
            store.delete_map(str(command["map"]))
            return {"status": "deleted"}

        raise ValueError(f"unknown command: {cmd!r}")

    def _resolve_pose(self, command: Mapping[str, ValueTypes]) -> conv.Pose2D:
        if "location" in command:
            from ..nav.locations import LocationStore

            handle = self._map_store.active_handle()
            if handle is None:
                raise RuntimeError(
                    "no active map; create or select one before using a location name"
                )
            loc = LocationStore(handle.locations_path).get(str(command["location"]))
            return conv.Pose2D(loc.x, loc.y, loc.theta)
        pose = command.get("pose", command)
        return conv.Pose2D(
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("theta", 0.0)),
        )

    async def close(self) -> None:
        unregister_slam(self.name)
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None


Registry.register_resource_creator(
    SLAM.API,
    RosSlam.MODEL,
    ResourceCreatorRegistration(RosSlam.new, RosSlam.validate_config),
)

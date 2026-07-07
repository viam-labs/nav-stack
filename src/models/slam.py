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

from ..config import (
    MODE_LOCALIZING,
    MODE_MAPPING,
    SlamConfig,
    ros_cmd_vel_to_viam_linear_mm_s,
)
from ..nav.global_localize import (
    global_localize_scan,
    load_occupancy_from_bridge_map,
    load_occupancy_from_map_dir,
)
from ..nav.maps import MapStore, validate_map_name
from ..ros import conversions as conv
from ..ros.bridge import IOProvider
from ..ros.manager import RosManager
from ..runtime import SlamRuntime, register_slam, unregister_slam

LOGGER = getLogger(__name__)

# Default /initialpose uncertainty for relocalize (~2 m, ~45 deg std dev).
RELOCALIZE_POSITION_VARIANCE_M2 = 4.0
RELOCALIZE_YAW_VARIANCE_RAD2 = (math.pi / 4) ** 2


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
        self._map_display_hold = False
        self._visible_map_generation = 0
        self._startup_global_localize_task: Optional[asyncio.Task] = None

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
        attrs = struct_to_dict(config.attributes)
        cfg = SlamConfig.from_dict(attrs)
        return cfg.required_dependencies(), []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self._cancel_startup_global_localize_task()
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
        self._schedule_startup_global_localize(loop)

        register_slam(
            self.name, SlamRuntime(self._manager, self._map_store, cfg)
        )
        LOGGER.info(f"nav-stack SLAM '{self.name}' configured in {cfg.mode} mode")

    def _cancel_startup_global_localize_task(self) -> None:
        task = self._startup_global_localize_task
        if task is not None and not task.done():
            task.cancel()
        self._startup_global_localize_task = None

    def _schedule_startup_global_localize(self, loop: asyncio.AbstractEventLoop) -> None:
        cfg = self._cfg
        if (
            cfg is None
            or cfg.mode != MODE_LOCALIZING
            or not cfg.global_localize_on_start
        ):
            return
        options = dict(cfg.global_localize_on_start_options)
        delay_s = max(0.0, float(cfg.global_localize_on_start_delay_s))
        refine_options = dict(cfg.global_localize_on_start_refine_options)
        post_apply_refine_options = dict(
            cfg.global_localize_on_start_post_apply_refine_options
        )
        self._startup_global_localize_task = loop.create_task(
            self._run_startup_global_localize(
                options,
                delay_s=delay_s,
                run_refine_pass=bool(cfg.global_localize_on_start_refine),
                refine_delay_s=max(0.0, float(cfg.global_localize_on_start_refine_delay_s)),
                refine_max_passes=max(
                    0, int(cfg.global_localize_on_start_refine_max_passes)
                ),
                target_score=float(cfg.global_localize_on_start_target_score),
                target_ray_mae_m=float(cfg.global_localize_on_start_target_ray_mae_m),
                refine_options=refine_options,
                run_post_apply_refine=bool(cfg.global_localize_on_start_post_apply_refine),
                post_apply_refine_delay_s=max(
                    0.0, float(cfg.global_localize_on_start_post_apply_refine_delay_s)
                ),
                post_apply_refine_options=post_apply_refine_options,
            )
        )

    def _is_navigation_active(self) -> bool:
        mgr = self._manager
        if mgr is None:
            return False
        try:
            return bool(mgr.nav_status().get("active", False))
        except Exception:  # noqa: BLE001 - nav stack may not be up yet
            return False

    @staticmethod
    def _startup_global_localize_quality(
        result: Mapping[str, ValueTypes],
    ) -> tuple[float, float, float]:
        score = float(result.get("score", float("-inf")))
        ray_mae_raw = result.get("ray_mae_m")
        ray_mae = float(ray_mae_raw) if ray_mae_raw is not None else float("inf")
        hit_rate = float(result.get("hit_rate", 0.0))
        return (score, -ray_mae, hit_rate)

    @staticmethod
    def _startup_global_localize_meets_target(
        result: Mapping[str, ValueTypes],
        *,
        target_score: float,
        target_ray_mae_m: float,
    ) -> bool:
        score = float(result.get("score", float("-inf")))
        ray_mae_raw = result.get("ray_mae_m")
        if ray_mae_raw is None:
            return False
        ray_mae = float(ray_mae_raw)
        return score >= target_score and ray_mae <= target_ray_mae_m

    async def _run_startup_global_localize(
        self,
        options: Mapping[str, ValueTypes],
        *,
        delay_s: float = 4.0,
        max_attempts: int = 3,
        retry_delay_s: float = 2.0,
        run_refine_pass: bool = False,
        refine_delay_s: float = 8.0,
        refine_max_passes: int = 3,
        target_score: float = 0.7,
        target_ray_mae_m: float = 0.4,
        refine_options: Optional[Mapping[str, ValueTypes]] = None,
        run_post_apply_refine: bool = True,
        post_apply_refine_delay_s: float = 3.0,
        post_apply_refine_options: Optional[Mapping[str, ValueTypes]] = None,
    ) -> None:
        if delay_s > 0.0:
            await asyncio.sleep(delay_s)
        if self._is_navigation_active():
            LOGGER.info("startup global_localize skipped: navigation already active")
            return
        command: dict = {
            "command": "global_localize",
            # Evaluate candidates first; apply only the best pose at the end.
            "apply": False,
            "auto_full_map_fallback": True,
        }
        command.update(dict(options))
        command["apply"] = False
        best_result: Optional[Mapping[str, ValueTypes]] = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.do_command(command)
                best_result = result
                LOGGER.info(
                    "startup global_localize matched on attempt %d: score=%s ray_mae=%s",
                    attempt,
                    result.get("score"),
                    result.get("ray_mae_m"),
                )
                if run_refine_pass:
                    passes = max(0, int(refine_max_passes))
                    for refine_pass in range(1, passes + 1):
                        if self._is_navigation_active():
                            LOGGER.info(
                                "startup global_localize refinement aborted: navigation active"
                            )
                            break
                        if self._startup_global_localize_meets_target(
                            best_result,
                            target_score=target_score,
                            target_ray_mae_m=target_ray_mae_m,
                        ):
                            break
                        if refine_delay_s > 0.0:
                            await asyncio.sleep(refine_delay_s)
                        refine_command = dict(command)
                        # Refine around the best candidate pose found so far.
                        refine_command["full_map"] = False
                        if refine_options:
                            refine_command.update(dict(refine_options))
                        best_pose = (
                            best_result.get("pose")
                            if best_result is not None
                            else None
                        )
                        if isinstance(best_pose, Mapping):
                            refine_command["pose"] = {
                                "x": float(best_pose.get("x", 0.0)),
                                "y": float(best_pose.get("y", 0.0)),
                                "theta": float(best_pose.get("theta", 0.0)),
                            }
                        refine_result = await self.do_command(refine_command)
                        if (
                            self._startup_global_localize_quality(refine_result)
                            > self._startup_global_localize_quality(best_result)
                        ):
                            best_result = refine_result
                        LOGGER.info(
                            "startup global_localize refinement pass %d/%d: "
                            "score=%s ray_mae=%s",
                            refine_pass,
                            passes,
                            refine_result.get("score"),
                            refine_result.get("ray_mae_m"),
                        )

                best_pose = best_result.get("pose") if best_result is not None else None
                if isinstance(best_pose, Mapping):
                    await self.do_command(
                        {
                            "command": "relocalize",
                            "pose": {
                                "x": float(best_pose.get("x", 0.0)),
                                "y": float(best_pose.get("y", 0.0)),
                                "theta": float(best_pose.get("theta", 0.0)),
                            },
                            "position_variance_m2": 0.25,
                            "yaw_variance_rad2": 0.06853891945200942,
                        }
                    )
                if run_post_apply_refine and best_result is not None:
                    if post_apply_refine_delay_s > 0.0:
                        await asyncio.sleep(post_apply_refine_delay_s)
                    if self._is_navigation_active():
                        LOGGER.info(
                            "startup global_localize post-apply skipped: navigation active"
                        )
                        return
                    post_command: dict = {
                        "command": "global_localize",
                        # Match known-good manual behavior exactly.
                        "apply": True,
                    }
                    if post_apply_refine_options:
                        post_command.update(dict(post_apply_refine_options))
                    post_command["apply"] = True
                    post_result = await self.do_command(post_command)
                    best_result = post_result
                    LOGGER.info(
                        "startup global_localize post-apply pass: score=%s ray_mae=%s",
                        post_result.get("score"),
                        post_result.get("ray_mae_m"),
                    )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - startup best-effort
                LOGGER.warning(
                    "startup global_localize attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(max(retry_delay_s, 0.0))

    # -- ROS IO --------------------------------------------------------------
    def _build_io(self) -> IOProvider:
        async def read_lidar_points(name: str) -> conv.LidarPoints:
            assert self._cfg is not None
            timeout = max(float(self._cfg.sensor_read_timeout_s), 1.0)
            cam = self._cameras[name]
            # MiR lidar: get_laser_scan is more reliable than the PCD round-trip.
            try:
                raw_payload = await cam.do_command({"command": "get_laser_scan"})
                payload = (
                    struct_to_dict(raw_payload)
                    if not isinstance(raw_payload, dict)
                    else raw_payload
                )
                mir_pts = conv.points_from_mir_laser_scan_payload(payload)
                if mir_pts.base_link.size > 0 or (
                    mir_pts.sensor_scan is not None
                    and conv.scan_has_returns(mir_pts.sensor_scan)
                ):
                    return mir_pts
                LOGGER.warning(
                    "lidar %s get_laser_scan returned no valid ranges", name
                )
            except Exception as exc:
                LOGGER.warning("lidar %s get_laser_scan failed: %s", name, exc)
            data = await cam.get_point_cloud(timeout=timeout)
            raw = data[0] if isinstance(data, tuple) else data
            pts = conv.parse_pcd(raw)
            return conv.ndarray_as_base_link_points(pts)

        async def read_odometry() -> conv.OdomReading:
            if self._movement_sensor is None:
                return conv.OdomReading(0.0, 0.0, 0.0)
            readings = await self._movement_sensor.get_readings()
            return conv.parse_odom_from_readings(readings)

        async def drive_base(vx: float, vy: float, vtheta: float):
            assert self._cfg is not None
            lx_mm, ly_mm = ros_cmd_vel_to_viam_linear_mm_s(
                vx,
                vy,
                self._cfg.base_velocity_convention,
            )
            await self._base.set_velocity(
                linear=Vector3(x=lx_mm, y=ly_mm, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=math.degrees(vtheta)),
            )

        async def stop_base():
            # MiR base.stop() also calls REST stop_immediately (PAUSE), which
            # drops Manualcontrol and kills the rosbridge /cmd_vel session — breaking
            # go_to_position and the next navigate_to_location. Nav2 only needs zeros.
            await self._base.set_velocity(
                linear=Vector3(x=0.0, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            )

        return IOProvider(read_lidar_points, read_odometry, drive_base, stop_base)

    def _start_mode(self, mode: str) -> None:
        assert self._manager and self._map_store and self._cfg
        handle = self._map_store.active_handle()
        stem = handle.serialization_stem if handle else None
        self._manager.start_slam(stem, mode)
        self._cfg.mode = mode

    def _reset_live_slam(self, mode: str) -> None:
        """Reset slam_toolbox in place when possible; full restart as fallback."""
        assert self._manager is not None
        mgr = self._manager
        if mode == MODE_MAPPING and mgr.slam_running() and mgr.reset_slam_map():
            if self._cfg is not None:
                self._cfg.mode = mode
            return
        self._start_mode(mode)

    def _begin_map_reset(self) -> None:
        """Blank the control-tab map immediately."""
        self._map_display_hold = True
        mgr = self._manager
        node = mgr.node if mgr else None
        if node is not None:
            node.set_map_updates_enabled(False)
            self._visible_map_generation = node.flush_map_subscription()

    def _end_map_reset(self) -> None:
        mgr = self._manager
        node = mgr.node if mgr else None
        if node is not None:
            self._visible_map_generation = node.flush_map_subscription()
            node.set_map_updates_enabled(True)
        self._map_display_hold = False

    def _map_grid_visible(self, grid: Optional[Dict]) -> bool:
        if not grid:
            return False
        return int(grid.get("generation", 0)) >= self._visible_map_generation

    def _map_is_live(self, name: str, store: MapStore) -> bool:
        """True when ``name`` is the map currently driving the SLAM session."""
        active = store.get_active_map_name()
        if active is not None:
            return active == name
        return self._cfg is not None and self._cfg.active_map == name

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
        if self._map_display_hold:
            return [conv.points_to_pcd(np.empty((0, 3)))]
        node = self._manager.node if self._manager else None
        grid = node.get_map() if node else None
        if not self._map_grid_visible(grid):
            return [conv.points_to_pcd(np.empty((0, 3)))]
        pcd = conv.occupancy_grid_to_pcd(
            grid["grid"], grid["resolution"], grid["origin_x"], grid["origin_y"]
        )
        return conv.chunk_bytes(pcd)

    async def get_internal_state(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> List[bytes]:
        if self._map_display_hold or not self._map_grid_visible(
            self._manager.node.get_map() if self._manager and self._manager.node else None
        ):
            return [b""]
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
            # Subprocess-heavy manager calls must stay off the module event
            # loop: the bridge marshals odom/lidar/cmd_vel onto it, so blocking
            # here stalls TF and scans.
            await asyncio.to_thread(self._start_mode, MODE_MAPPING)
            return {"status": "mapping", "map": store.get_active_map_name()}

        if cmd == "start_localizing":
            name = command.get("map")
            if name:
                store.set_active_map(str(name))
            handle = store.active_handle()
            if not handle or not handle.has_serialized_map():
                raise ValueError("active map has no saved data; map it first")
            await asyncio.to_thread(self._start_mode, MODE_LOCALIZING)
            result: dict[str, ValueTypes] = {
                "status": "localizing",
                "map": store.get_active_map_name(),
            }
            if command.get("use_mir_pose", False):
                relocalize_result = await self.do_command(
                    {"command": "relocalize", "use_mir_pose": True},
                    **kwargs,
                )
                result["relocalize"] = relocalize_result
            return result

        if cmd == "save_map":
            handle = store.active_handle()
            if not handle:
                raise ValueError("no active map")
            # save_map runs ros2 CLI subprocesses (30-40s timeouts); keep the
            # event loop free so mapping TF/scans continue while saving.
            await asyncio.to_thread(mgr.save_map, handle.serialization_stem)
            return {"status": "saved", "map": handle.name}

        if cmd == "set_initial_pose":
            pose = self._resolve_pose(command)
            await asyncio.to_thread(mgr.set_initial_pose, pose)
            return {"status": "ok"}

        if cmd in ("relocalize", "refine_localization"):
            if self._cfg is not None and self._cfg.mode != MODE_LOCALIZING:
                raise ValueError(
                    "relocalize requires SLAM mode localizing; call start_localizing first"
                )
            pose = await self._resolve_relocalize_seed(command)
            position_variance_m2 = float(
                command.get("position_variance_m2", RELOCALIZE_POSITION_VARIANCE_M2)
            )
            yaw_variance_rad2 = float(
                command.get("yaw_variance_rad2", RELOCALIZE_YAW_VARIANCE_RAD2)
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: mgr.relocalize(
                    pose,
                    position_variance_m2=position_variance_m2,
                    yaw_variance_rad2=yaw_variance_rad2,
                ),
            )
            return {
                "status": "relocalizing",
                "seed_pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
            }

        if cmd == "global_localize":
            return await self._global_localize(command)

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
        if cmd == "clear_map":
            handle = store.active_handle()
            if not handle:
                raise ValueError("no active map")
            self._begin_map_reset()
            try:
                handle.clear_serialized_data()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._reset_live_slam, MODE_MAPPING)
            finally:
                self._end_map_reset()
            return {"status": "cleared", "map": handle.name, "mode": MODE_MAPPING}
        if cmd == "delete_map":
            name = validate_map_name(
                str(command.get("map") or store.get_active_map_name() or "")
            )
            if not name:
                raise ValueError("no map specified and no active map")
            was_live = self._map_is_live(name, store)
            if was_live:
                self._begin_map_reset()
            try:
                store.delete_map(name)
                if was_live:
                    resolution = (
                        self._cfg.slam_toolbox.resolution if self._cfg else 0.05
                    )
                    store.get_or_create_map(name, resolution=resolution)
                    store.set_active_map(name)
                    if self._cfg is not None:
                        self._cfg.active_map = name
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, self._reset_live_slam, MODE_MAPPING
                    )
            finally:
                if was_live:
                    self._end_map_reset()
            return {
                "status": "deleted",
                "map": name,
                "active_map": store.get_active_map_name(),
                "mode": MODE_MAPPING if was_live else (self._cfg.mode if self._cfg else None),
            }

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

    async def _read_merged_scan(self) -> conv.LaserScan2D:
        assert self._cfg is not None
        io = self._build_io()
        scans: List[conv.LaserScan2D] = []
        for lidar in self._cfg.lidars:
            points = await io.read_lidar_points(lidar.name)
            mount = conv.Pose2D(lidar.x, lidar.y, lidar.theta)
            # Prefer base_link points: for MiR get_laser_scan this already merges
            # all available scanners into one common frame.
            if points.base_link.size:
                scan = conv.pointcloud_to_scan(
                    points.base_link,
                    # points.base_link are already in base_link coordinates.
                    sensor_pose=conv.Pose2D(0.0, 0.0, 0.0),
                    num_bins=self._cfg.scan_bins,
                )
                if conv.scan_has_returns(scan):
                    scans.append(scan)
                    continue
            if points.sensor_scan is not None and conv.scan_has_returns(points.sensor_scan):
                scan = points.sensor_scan
                if lidar.x or lidar.y or lidar.theta:
                    scan = conv.LaserScan2D(
                        ranges=scan.ranges,
                        angle_min=scan.angle_min,
                        angle_increment=scan.angle_increment,
                        range_min=scan.range_min,
                        range_max=scan.range_max,
                        sensor_pose=mount,
                    )
                scans.append(scan)
        if not scans:
            raise RuntimeError("no lidar returns available for global_localize")
        return conv.merge_scans(
            scans,
            num_bins=self._cfg.scan_bins,
            range_max=self._cfg.slam_toolbox.max_laser_range,
        )

    def _load_active_occupancy_map(self, source: str = "auto"):
        assert self._map_store is not None and self._manager is not None
        handle = self._map_store.active_handle()
        if handle is None:
            raise RuntimeError("no active map")

        source = (source or "auto").lower()
        if source not in {"auto", "live", "pgm"}:
            raise ValueError("map_source must be one of: auto, live, pgm")

        node = self._manager.node
        if source in {"auto", "live"} and node is not None:
            live = node.get_map()
            if live is not None:
                return load_occupancy_from_bridge_map(live), "live"
            if source == "live":
                raise RuntimeError("requested live map_source but /map is unavailable")

        occ_map = load_occupancy_from_map_dir(handle.root)
        if occ_map is not None:
            return occ_map, "pgm"
        if node is None:
            raise RuntimeError("ROS bridge not started")
        live = node.get_map()
        if live is None:
            raise RuntimeError(
                "no occupancy map available; save the map or wait for /map from slam_toolbox"
            )
        return load_occupancy_from_bridge_map(live), "live"

    async def _global_localize(
        self, command: Mapping[str, ValueTypes]
    ) -> Mapping[str, ValueTypes]:
        if self._cfg is not None and self._cfg.mode != MODE_LOCALIZING:
            raise ValueError(
                "global_localize requires SLAM mode localizing; call start_localizing first"
            )
        assert self._manager is not None
        mgr = self._manager
        scan = await self._read_merged_scan()
        occ_map, map_source = self._load_active_occupancy_map(
            str(command.get("map_source", "auto"))
        )

        hint: Optional[conv.Pose2D] = None
        if "location" in command or "pose" in command or "x" in command:
            hint = self._resolve_pose(command)
        elif not command.get("full_map", False):
            hint = mgr.get_pose_in_map()

        full_map = bool(command.get("full_map", hint is None))
        search_radius_m = float(command.get("search_radius_m", 8.0))
        apply_pose = command.get("apply", True) is not False
        auto_full_map_fallback = bool(command.get("auto_full_map_fallback", True))
        fallback_score_threshold = float(command.get("fallback_score_threshold", 0.42))
        fallback_hit_rate_threshold = float(command.get("fallback_hit_rate_threshold", 0.6))

        loop = asyncio.get_running_loop()

        def _run_match(full_map_override: bool):
            default_coarse_pos = 0.6 if full_map_override else 0.4
            default_coarse_yaw = 18.0 if full_map_override else 12.0
            default_local_yaw_window = 360.0 if full_map_override else 180.0
            return global_localize_scan(
                occ_map,
                scan,
                hint=hint,
                full_map=full_map_override,
                search_radius_m=search_radius_m,
                coarse_position_step_m=float(
                    command.get("coarse_position_step_m", default_coarse_pos)
                ),
                coarse_yaw_step_deg=float(
                    command.get("coarse_yaw_step_deg", default_coarse_yaw)
                ),
                local_yaw_window_deg=float(
                    command.get("local_yaw_window_deg", default_local_yaw_window)
                ),
                fine_position_step_m=float(command.get("fine_position_step_m", 0.08)),
                fine_yaw_step_deg=float(command.get("fine_yaw_step_deg", 2.0)),
                max_scan_points=int(command.get("max_scan_points", 240)),
                min_in_map_points=int(command.get("min_in_map_points", 40)),
                min_in_map_ratio=float(command.get("min_in_map_ratio", 0.35)),
                hit_radius_cells=int(command.get("hit_radius_cells", 2)),
                ray_refine_candidates=int(command.get("ray_refine_candidates", 24)),
                ray_refine_beams=int(command.get("ray_refine_beams", 64)),
                ray_step_m=float(command.get("ray_step_m", 0.08)),
                ray_weight=float(command.get("ray_weight", 0.35)),
            )

        result = await loop.run_in_executor(None, lambda: _run_match(full_map))
        fallback_used = False
        if (
            not full_map
            and auto_full_map_fallback
            and (
                result.score < fallback_score_threshold
                or result.hit_rate < fallback_hit_rate_threshold
            )
        ):
            full_result = await loop.run_in_executor(None, lambda: _run_match(True))
            if (
                full_result.score > result.score
                or full_result.hit_rate > result.hit_rate
            ):
                result = full_result
                fallback_used = True
        resolved_full_map = full_map or fallback_used

        if apply_pose:
            await loop.run_in_executor(
                None,
                lambda: mgr.relocalize(
                    result.pose,
                    position_variance_m2=0.25,
                    yaw_variance_rad2=0.06853891945200942,
                ),
            )

        return {
            "status": "localized" if apply_pose else "matched",
            "pose": {
                "x": result.pose.x,
                "y": result.pose.y,
                "theta": result.pose.theta,
            },
            "score": result.score,
            "candidates_evaluated": result.candidates_evaluated,
            "scan_points_used": result.scan_points_used,
            "in_map_points": result.in_map_points,
            "hit_rate": result.hit_rate,
            "ray_score": result.ray_score,
            "ray_mae_m": result.ray_mae_m,
            "map_source": map_source,
            "full_map": resolved_full_map,
            "fallback_used": fallback_used,
        }

    async def _resolve_relocalize_seed(
        self, command: Mapping[str, ValueTypes]
    ) -> conv.Pose2D:
        """Pick a map-frame seed pose for scan-to-map relocalization."""
        if "location" in command or "pose" in command or "x" in command:
            return self._resolve_pose(command)

        if command.get("use_mir_pose", False):
            if self._movement_sensor is None:
                raise RuntimeError(
                    "use_mir_pose requires movement_sensor on the SLAM service"
                )
            readings = await self._movement_sensor.get_readings()
            x = readings.get("position_x_m")
            y = readings.get("position_y_m")
            yaw_deg = readings.get("yaw_deg")
            if x is not None and y is not None and yaw_deg is not None:
                return conv.Pose2D(
                    float(x),
                    float(y),
                    math.radians(float(yaw_deg)),
                )
            raise RuntimeError(
                "MiR map pose unavailable in movement sensor readings "
                "(need position_x_m, position_y_m, yaw_deg)"
            )

        mgr = self._manager
        if mgr is None:
            raise RuntimeError("SLAM service not configured")
        pose = mgr.get_pose_in_map()
        if pose is not None:
            return pose

        raise RuntimeError(
            "no seed pose for relocalize; provide pose/location, set use_mir_pose, "
            "or call set_initial_pose with a rough estimate first"
        )

    async def close(self) -> None:
        self._cancel_startup_global_localize_task()
        unregister_slam(self.name)
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None


Registry.register_resource_creator(
    SLAM.API,
    RosSlam.MODEL,
    ResourceCreatorRegistration(RosSlam.new, RosSlam.validate_config),
)

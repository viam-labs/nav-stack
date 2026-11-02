"""SLAM service model: ``viam-labs:nav-stack:slam``.

Builtin occupancy SLAM on any Viam base. Exposes the standard Viam SLAM service
API plus map-management / mode / initial-pose commands through ``DoCommand``.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import replace
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, cast

import numpy as np
from typing_extensions import Self

from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.movement_sensor import MovementSensor
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.slam import SLAM, MappingMode, Pose
from viam.utils import ValueTypes, struct_to_dict

from ..config import (
    LIDAR_SCAN_GET_LASER_SCAN,
    MODE_LOCALIZING,
    MODE_MAPPING,
    SlamConfig,
    resolve_footprint,
)
from ..nav.global_localize import (
    GlobalLocalizeResult,
    load_occupancy_from_map_dict,
    load_occupancy_from_map_dir,
)
from ..nav.localize_worker import LocalizeWorker
from ..nav.loc_state import LocalizationStateTracker
from ..nav import pause_keyframes, slice_match
from ..nav.maps import MapStore, validate_map_name
from ..nav.pose_jump_gate import (
    JumpDecision,
    PoseJumpGate,
    candidate_beats_previous,
    stuck_refine_decision,
)
from ..geom import conversions as conv
from ..shm.lidar import ShmPointCloudClient
from ..runtime import (
    SlamRuntime,
    active_navigation_status,
    any_navigation_active,
    get_parent_robot,
    register_slam,
    register_slam_service,
    unregister_slam,
    unregister_slam_service,
)
from ..viam_frames import (
    apply_framesystem_to_slam_cfg,
    fetch_frame_system_config,
)
from ..slam_builtin import BuiltinSlamEngine, BuiltinSlamHost
from ..slam_builtin.localize_supervisor import LocalizeSupervisor
from ..slam_builtin.io_sensors import BuiltinSensors
from .do_command_util import build_alias_map, dispatch_do_command, require

LOGGER = getLogger(__name__)


# Default /initialpose uncertainty for relocalize (~2 m, ~45 deg std dev).
RELOCALIZE_POSITION_VARIANCE_M2 = 4.0
RELOCALIZE_YAW_VARIANCE_RAD2 = (math.pi / 4) ** 2


# A goal can fail for no progress in ~8 s. Notice a stop well before that,
# but ignore a single zero sample between commands.
_NAV_STOPPED_BEFORE_CHECK_S = 2.0
_NAV_STOPPED_LIN_MPS = 0.05
_NAV_STOPPED_YAW_RAD_S = 0.08

_SLAM_DO_ALIASES = build_alias_map({
    "optimize": ("optimize_graph",),
    "relocalize": ("refine_localization",),
    "clear_obstacles": ("erase_obstacles",),
    "add_obstacles": ("mark_obstacles",),
})


class SlamService(SLAM):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "slam")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[SlamConfig] = None
        # BuiltinSlamHost.
        self._manager = None
        self._engine: Optional[BuiltinSlamEngine] = None
        self._map_store: Optional[MapStore] = None
        self._base: Optional[Base] = None
        self._cameras: dict = {}
        self._movement_sensor: Optional[MovementSensor] = None
        self._heading_sensor: Optional[MovementSensor] = None
        self._map_display_hold = False
        self._visible_map_generation = 0
        self._startup_global_localize_task: Optional[asyncio.Task] = None
        self._startup_localize_started_at: Optional[float] = None
        self._startup_localize_local = False
        self._periodic_relocalize_task: Optional[asyncio.Task] = None
        self._nav_stopped_since: Optional[float] = None
        self._mapping_revisit_task: Optional[asyncio.Task] = None
        self._last_relocalize_check: dict = {"status": "idle"}
        self._last_revisit_check: dict = {"status": "idle"}
        # Folds those checks into one state the operator can read.
        self._loc_state = LocalizationStateTracker()
        # Soft-loc nav_hold started_at / released: resume driving on the published
        # pose after a short hold when there is no large jump to confirm.
        self._soft_nav_hold_since: Optional[float] = None
        self._soft_nav_hold_released: bool = False
        self._slice_library: Optional[slice_match.SliceLibrary] = None
        self._pause_keyframes: Optional[pause_keyframes.PauseKeyframeStore] = None
        self._keyframe_lock = threading.Lock()
        # Lidars that don't implement get_laser_scan (auto mode); skip re-probing.
        self._skip_get_laser_scan: set[str] = set()
        self._shm_lidar = ShmPointCloudClient(logger=LOGGER)
        self._pose_jump_gate = PoseJumpGate()
        # Scan-to-map matcher subprocess (keeps the heavy search off this GIL).
        self._localizer = LocalizeWorker(enabled=True, logger=LOGGER)
        self._localize = LocalizeSupervisor(self)
        self._framesystem_task: Optional[asyncio.Task] = None
        self._framesystem_gen = 0
        self._framesystem_raw_attrs: Optional[Mapping] = None
        self._builtin_sensors: Optional[BuiltinSensors] = None

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
        self._cancel_periodic_relocalize_task()
        self._cancel_mapping_revisit_task()
        self._cancel_framesystem_task()
        self._skip_get_laser_scan = set()
        attrs = struct_to_dict(config.attributes)
        cfg = SlamConfig.from_dict(attrs)
        # Framesystem mounts are resolved async on the module loop after
        # reconfigure returns — never via a sync cross-thread parent RPC.
        self._cfg = cfg
        self._framesystem_raw_attrs = attrs
        if self._localizer.enabled != bool(cfg.localize_subprocess):
            self._localizer.close()
            self._localizer = LocalizeWorker(
                enabled=bool(cfg.localize_subprocess), logger=LOGGER
            )
        self._pose_jump_gate = PoseJumpGate(
            confirm_count=cfg.localize_jump_confirm_count,
            agree_m=cfg.localize_jump_agree_m,
            agree_deg=cfg.localize_jump_agree_deg,
            large_m=cfg.localize_jump_large_m,
            large_deg=cfg.localize_jump_large_deg,
        )

        self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])
        if cfg.uses_sim():
            self._cameras = {}
            self._movement_sensor = None
            self._heading_sensor = None
        else:
            self._cameras = {
                lidar.name: cast(
                    Camera, dependencies[Camera.get_resource_name(lidar.name)]
                )
                for lidar in cfg.lidars
            }
            self._movement_sensor = (
                cast(
                    MovementSensor,
                    dependencies[
                        MovementSensor.get_resource_name(cfg.movement_sensor)
                    ],
                )
                if cfg.movement_sensor
                else None
            )
            self._heading_sensor = (
                cast(
                    MovementSensor,
                    dependencies[
                        MovementSensor.get_resource_name(cfg.heading_sensor)
                    ],
                )
                if cfg.heading_sensor
                else None
            )

        self._map_store = MapStore(cfg.maps_dir)
        active = cfg.active_map or self._map_store.get_active_map_name() or "default"
        self._map_store.get_or_create_map(active, resolution=cfg.map.resolution)
        self._map_store.set_active_map(active)

        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._engine = None

        loop = asyncio.get_event_loop()
        sim_sensors = None
        self._builtin_sensors = None
        if cfg.uses_sim():
            from ..sim import SimSensors, ensure_sim_world_from_slam_cfg

            world = ensure_sim_world_from_slam_cfg(cfg)
            sensors = SimSensors(world)
            sim_sensors = sensors
        else:
            odom_reader = self._make_typed_odom_reader()
            sensors = BuiltinSensors(
                cfg=cfg,
                cameras=self._cameras,
                movement_sensor=self._movement_sensor,
                heading_sensor=self._heading_sensor,
                shm_lidar=self._shm_lidar,
                loop=loop,
                logger=LOGGER.info,
                skip_get_laser_scan=self._skip_get_laser_scan,
                scan_max_age_s=float(cfg.scan_max_age_s or 2.0),
                odom_reader=odom_reader,
            )
            self._builtin_sensors = sensors
            if self._lidars_need_framesystem(attrs) and get_parent_robot() is not None:
                # Hold scans until FS mounts land — identity mounts poison maps.
                sensors._mounts_ready = False  # noqa: SLF001
        self._engine = BuiltinSlamEngine(
            cfg,
            sensors,
            self._map_store,
            logger=LOGGER.info,
            rate_hz=cfg.tick_rate_hz(),
        )
        if cfg.uses_sim():
            # SimWorld is ground truth: SLAM pose tracks the body each tick.
            # Do not teleport the body when localize jumps the estimate —
            # that caused OOB freezes and big visible jumps.
            seed = conv.Pose2D(
                cfg.sim.seed_x, cfg.sim.seed_y, cfg.sim.seed_theta
            )
            self._engine.set_pose(seed)
        self._manager = BuiltinSlamHost(self._engine)
        self._manager.start()
        self._start_mode(cfg.mode)
        self._wire_still_keyframe_hook()
        backend = "builtin+sim" if cfg.uses_sim() else "builtin"

        self._schedule_startup_global_localize(loop)
        self._schedule_periodic_relocalize(loop)
        self._schedule_mapping_revisit(loop)
        self._schedule_framesystem_mounts(loop)

        register_slam(
            self.name,
            SlamRuntime(
                self._manager,
                self._map_store,
                cfg,
                self._last_relocalize_check,
                cameras=self._cameras,
                shm_lidar=self._shm_lidar,
                sim_sensors=sim_sensors,
            ),
        )
        register_slam_service(self.name, self)
        LOGGER.info(
            f"nav-stack SLAM '{self.name}' configured in {cfg.mode} mode "
            f"(slam_backend={backend})"
        )

    def _cancel_startup_global_localize_task(self) -> None:
        task = self._startup_global_localize_task
        if task is not None and not task.done():
            task.cancel()
        self._startup_global_localize_task = None
        self._startup_localize_started_at = None

    def _cancel_periodic_relocalize_task(self) -> None:
        task = self._periodic_relocalize_task
        if task is not None and not task.done():
            task.cancel()
        self._periodic_relocalize_task = None

    def _cancel_mapping_revisit_task(self) -> None:
        task = self._mapping_revisit_task
        if task is not None and not task.done():
            task.cancel()
        self._mapping_revisit_task = None

    def _lidars_need_framesystem(self, raw_attrs: Mapping) -> bool:
        """True when at least one lidar will take its mount from the framesystem."""
        for entry in list(raw_attrs.get("lidars") or []):
            if isinstance(entry, str):
                return True
            if not isinstance(entry, Mapping):
                continue
            if entry.get("points_in_base_link"):
                continue
            if "mount" in entry or any(
                k in entry for k in ("x", "y", "z", "theta", "pitch", "roll")
            ):
                continue
            return True
        return False

    def _imu_needs_framesystem(self, raw_attrs: Mapping) -> bool:
        """True when an IMU yaw should be taken from the framesystem."""
        cfg = self._cfg
        if cfg is None:
            return False
        if cfg.movement_sensor and "movement_sensor_yaw_deg" not in raw_attrs:
            return True
        if cfg.heading_sensor and "heading_sensor_yaw_deg" not in raw_attrs:
            return True
        return False

    def _footprint_needs_framesystem(self, raw_attrs: Mapping) -> bool:
        """True when the self-return crop is on and config leaves a side unset."""
        cfg = self._cfg
        if cfg is None or not cfg.crop_inside_footprint:
            return False
        return resolve_footprint(raw_attrs).source != "config"

    def _set_mounts_ready(self, ready: bool) -> None:
        sensors = self._builtin_sensors
        if sensors is not None:
            sensors._mounts_ready = bool(ready)  # noqa: SLF001

    def _cancel_framesystem_task(self) -> None:
        task = self._framesystem_task
        if task is not None and not task.done():
            task.cancel()
        self._framesystem_task = None
        self._framesystem_gen += 1

    def _schedule_framesystem_mounts(self, loop: asyncio.AbstractEventLoop) -> None:
        cfg = self._cfg
        if cfg is None or cfg.uses_sim():
            self._set_mounts_ready(True)
            return
        if get_parent_robot() is None:
            self._set_mounts_ready(True)
            return
        raw = self._framesystem_raw_attrs or {}
        if not (
            self._lidars_need_framesystem(raw)
            or self._imu_needs_framesystem(raw)
            or self._footprint_needs_framesystem(raw)
        ):
            self._set_mounts_ready(True)
            return
        self._framesystem_gen += 1
        gen = self._framesystem_gen
        self._framesystem_task = loop.create_task(
            self._apply_framesystem_mounts(gen, raw)
        )

    async def _apply_framesystem_mounts(
        self, gen: int, raw_attrs: Mapping
    ) -> None:
        if gen != self._framesystem_gen:
            return
        robot = get_parent_robot()
        cfg = self._cfg
        if robot is None or cfg is None:
            self._set_mounts_ready(True)
            return
        try:
            fs = await fetch_frame_system_config(robot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep JSON mounts on failure
            LOGGER.warning(
                "framesystem mount resolve failed; using config mounts: %s", exc
            )
            self._set_mounts_ready(True)
            return
        if gen != self._framesystem_gen or self._cfg is not cfg:
            return
        try:
            apply_framesystem_to_slam_cfg(
                cfg, fs, raw_attrs=raw_attrs, logger=LOGGER
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "framesystem mount apply failed; using config mounts: %s", exc
            )
        finally:
            if gen == self._framesystem_gen:
                self._set_mounts_ready(True)

    def _reschedule_mode_watchdogs(self) -> None:
        """Restart mode-gated background tasks after a runtime mode switch.

        ``start_mapping`` / ``start_localizing`` change ``cfg.mode`` outside
        reconfigure; without this, e.g. periodic relocalize / startup
        global_localize would never start after switching to localizing.
        """
        loop = asyncio.get_event_loop()
        self._cancel_periodic_relocalize_task()
        self._cancel_mapping_revisit_task()
        self._cancel_startup_global_localize_task()
        self._schedule_periodic_relocalize(loop)
        self._schedule_mapping_revisit(loop)
        self._schedule_startup_global_localize(loop)

    def _schedule_mapping_revisit(self, loop: asyncio.AbstractEventLoop) -> None:
        cfg = self._cfg
        if (
            cfg is None
            or cfg.mode != MODE_MAPPING
            or not cfg.mapping_revisit_check
        ):
            return
        self._mapping_revisit_task = loop.create_task(
            self._run_mapping_revisit(
                interval_s=max(5.0, float(cfg.mapping_revisit_interval_s)),
            )
        )

    async def _run_mapping_revisit(self, *, interval_s: float) -> None:
        """Mapping-time revisit watchdog (anti duplicate-corridor).

        Loop-closure only fires when the drifted return pose is inside its
        loop search radius; after a long excursion on IMU-only odom it often is
        not, so a revisited corridor gets mapped as a second copy. This task
        periodically matches the live scan against the live map — near the
        current pose first, wider only when needed — and on a strong disagreeing
        match shifts the odom TF so the next scans link back to the original
        geometry.
        """
        while True:
            await asyncio.sleep(interval_s)
            try:
                await self._mapping_revisit_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - watchdog must survive hiccups
                LOGGER.warning("mapping revisit cycle failed: %s", exc)
                self._publish_revisit_check({"status": "error", "error": str(exc)})

    def _publish_revisit_check(self, result: Mapping[str, ValueTypes]) -> dict:
        return self._localize._publish_revisit_check(result)

    async def _flip_current_map_yaw(self) -> Mapping[str, ValueTypes]:
        return await self._localize._flip_current_map_yaw()

    def _reset_slice_library(self) -> None:
        self._slice_library = None
        with self._keyframe_lock:
            if self._pause_keyframes is not None:
                self._pause_keyframes.clear()
            self._pause_keyframes = None

    def _wire_still_keyframe_hook(self) -> None:
        mgr = self._manager
        cfg = self._cfg
        if mgr is None or cfg is None or not cfg.mapping_revisit_keyframes:
            if mgr is not None:
                mgr.set_still_keyframe_hook(None)
            return
        mgr.set_still_keyframe_hook(self._on_still_keyframe)

    def _ensure_keyframe_store(
        self, cfg: SlamConfig
    ) -> pause_keyframes.PauseKeyframeStore:
        with self._keyframe_lock:
            if self._pause_keyframes is None:
                self._pause_keyframes = pause_keyframes.PauseKeyframeStore(
                    min_spacing_m=cfg.mapping_revisit_keyframe_min_spacing_m,
                    min_spacing_deg=cfg.mapping_revisit_keyframe_min_spacing_deg,
                    max_keyframes=cfg.mapping_revisit_keyframe_max,
                    match_tol_m=cfg.mapping_revisit_keyframe_match_tol_m,
                )
            return self._pause_keyframes

    def _on_still_keyframe(
        self,
        scan: conv.LaserScan2D,
        band_points: List[np.ndarray],
        pose: conv.Pose2D,
    ) -> None:
        """Callback: record pause 2D+slice keyframe."""
        cfg = self._cfg
        if cfg is None or not cfg.mapping_revisit_keyframes:
            return
        store = self._ensure_keyframe_store(cfg)
        with self._keyframe_lock:
            added = store.add(pose, scan, band_points)
        if not added:
            return
        library = self._get_slice_library(cfg)
        if library is not None and any(np.asarray(p).size for p in band_points):
            with self._keyframe_lock:
                library.record(band_points, pose)
        LOGGER.info(
            "pause keyframe recorded at (%.2f, %.2f, %.1f deg); store=%d",
            pose.x,
            pose.y,
            math.degrees(pose.theta),
            len(store),
        )

    def _get_slice_library(
        self, cfg: SlamConfig
    ) -> Optional[slice_match.SliceLibrary]:
        """Lazily build the per-session multi-height slice library."""
        if not cfg.mapping_revisit_slice_verify:
            return None
        # Height slices need 3D points; ``auto`` lidars may still deliver them,
        # and if they turn out 2D-only the verify simply reports no data.
        if all(
            lidar.scan_source == LIDAR_SCAN_GET_LASER_SCAN
            for lidar in cfg.slam_lidars()
        ):
            return None
        if self._slice_library is None:
            try:
                bands = slice_match.SliceBand.parse_bands(
                    cfg.mapping_revisit_slice_bands
                )
            except (ValueError, TypeError, IndexError) as exc:
                LOGGER.warning("invalid mapping_revisit_slice_bands: %s", exc)
                return None
            if not bands:
                return None
            self._slice_library = slice_match.SliceLibrary(
                bands,
                resolution_m=cfg.mapping_revisit_slice_resolution_m,
                min_hit_rate=cfg.mapping_revisit_slice_min_hit_rate,
            )
        return self._slice_library

    def _revisit_match_quality(
        self, result, cfg: SlamConfig
    ) -> tuple[bool, float, Optional[float]]:
        return self._localize._revisit_match_quality(result, cfg)

    async def _mapping_revisit_cycle(
        self,
        *,
        apply_override: Optional[bool] = None,
        yaw_flip: bool = False,
    ) -> Mapping[str, ValueTypes]:
        return await self._localize._mapping_revisit_cycle(
            apply_override=apply_override, yaw_flip=yaw_flip
        )

    async def _stuck_pose_refine(self) -> Mapping[str, ValueTypes]:
        return await self._localize._stuck_pose_refine()

    def _schedule_periodic_relocalize(self, loop: asyncio.AbstractEventLoop) -> None:
        cfg = self._cfg
        if (
            cfg is None
            or cfg.mode != MODE_LOCALIZING
            or not cfg.periodic_relocalize
        ):
            return
        self._periodic_relocalize_task = loop.create_task(
            self._run_periodic_relocalize(
                interval_s=max(1.0, float(cfg.periodic_relocalize_interval_s)),
            )
        )

    async def _run_periodic_relocalize(self, *, interval_s: float) -> None:
        """Background drift watchdog: periodically re-localize when pose drifts.

        Per-scan pose tracking has no automatic global correction;
        on long runs (or after CPU-starved navigation) the map->odom estimate can
        slide with no recovery. Uses a cheap local match each cycle, escalating
        to full-map global_localize (like a manual command) when the local match
        is untrusted or Nav2 is struggling with recoveries.

        While a goal is active the match stays off, unless the base has been
        sitting still. A stall can fail the goal in about 8 s, so that case is
        polled quickly and then uses the parked check.
        """
        while True:
            cfg = self._cfg
            nav = cfg is not None and self._is_navigation_active()
            during = bool(cfg and cfg.periodic_relocalize_during_navigation)
            if nav and not during:
                if not self._base_is_stopped():
                    self._nav_stopped_since = None
                    await asyncio.sleep(1.0)
                    continue
                now = time.monotonic()
                if self._nav_stopped_since is None:
                    self._nav_stopped_since = now
                if now - float(self._nav_stopped_since) < _NAV_STOPPED_BEFORE_CHECK_S:
                    await asyncio.sleep(0.5)
                    continue
                self._nav_stopped_since = None
                await self._relocalize_cycle_guarded()
                await asyncio.sleep(interval_s)
                continue
            self._nav_stopped_since = None
            sleep_s = interval_s
            if nav and cfg is not None:
                sleep_s = max(1.0, float(cfg.periodic_relocalize_nav_interval_s))
            await asyncio.sleep(sleep_s)
            await self._relocalize_cycle_guarded()

    async def _relocalize_cycle_guarded(self) -> None:
        await self._localize._relocalize_cycle_guarded()
    def _base_is_stopped(self) -> bool:
        """True when odom, or the active goal's command, says the base is still.

        Unknown motion is not stopped: a missing sample must not start a
        localize while the cart might be driving.
        """
        lin: Optional[float] = None
        yaw: Optional[float] = None
        mgr = self._manager
        try:
            node = getattr(mgr, "node", None) if mgr is not None else None
            status_fn = getattr(node, "slam_bridge_status", None) if node else None
            if callable(status_fn):
                vel = (status_fn() or {}).get("odom_velocity") or {}
                if vel:
                    lin = math.hypot(
                        float(vel.get("vx", 0.0)), float(vel.get("vy", 0.0))
                    )
                    yaw = abs(float(vel.get("vtheta", 0.0)))
        except Exception:  # noqa: BLE001
            lin, yaw = None, None
        if lin is None and yaw is None:
            status = active_navigation_status()
            if isinstance(status, dict):
                try:
                    lin = math.hypot(
                        float(status.get("cmd_vx_mps", 0.0)),
                        float(status.get("cmd_vy_mps", 0.0)),
                    )
                    yaw = abs(float(status.get("cmd_vtheta_rad_s", 0.0)))
                except (TypeError, ValueError):
                    return False
            else:
                return False
        if lin is None:
            lin = 0.0
        if yaw is None:
            yaw = 0.0
        return lin < _NAV_STOPPED_LIN_MPS and yaw < _NAV_STOPPED_YAW_RAD_S

    def _pose_from_mapping(self, pose: Mapping) -> conv.Pose2D:
        return conv.Pose2D(
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("theta", 0.0)),
        )

    def _confirm_pose_jump(
        self,
        current: Optional[conv.Pose2D],
        candidate: conv.Pose2D,
        *,
        force: bool = False,
    ) -> JumpDecision:
        """Shared gate for automatic large map jumps (manual / force bypasses)."""
        if current is None:
            current = conv.Pose2D(0.0, 0.0, 0.0)
        return self._pose_jump_gate.evaluate(current, candidate, force=force)

    def _should_refuse_mid_nav_jump(
        self,
        *,
        nav_active: bool,
        allow_during_navigation: bool,
        apply_override: Optional[bool],
        shift_m: float,
        shift_deg: float,
        large_jump: bool,
        cfg: SlamConfig,
        stopped_recovery: bool = False,
        good_match: bool = False,
        score: float = float("-inf"),
    ) -> bool:
        """True when a mid-drive pose yank must stay ``refused_large``.

        Loc-refine may probe a ≤1 m match then re-match with ``apply: true``
        and see a hallway-scale twin. Never force-apply past the mid-nav cap
        while a goal is active — that teleport is what drives into walls.

        ``stopped_recovery`` is the one exception: base already halted for bad
        loc, so a *trusted* larger match may apply once and let nav continue.
        """
        mid_nav = bool(nav_active or allow_during_navigation)
        if not mid_nav:
            return False
        if math.isinf(shift_m) or math.isinf(shift_deg):
            return True
        if stopped_recovery:
            trusted = bool(good_match)
            if not trusted and apply_override is True and math.isfinite(float(score)):
                trusted = float(score) >= float(cfg.localize_stopped_recovery_min_score)
            if trusted:
                max_m = max(0.5, float(cfg.localize_stopped_recovery_max_m))
                max_deg = max(10.0, float(cfg.localize_stopped_recovery_max_deg))
                return float(shift_m) > max_m or float(shift_deg) > max_deg
        max_m = max(0.2, float(cfg.localize_mid_nav_apply_max_m))
        max_deg = max(5.0, float(cfg.localize_mid_nav_apply_max_deg))
        if apply_override is True:
            return float(shift_m) > max_m or float(shift_deg) > max_deg
        # Probe / auto mid-nav refine: refuse hallway-sized jumps (no sticky hold).
        return bool(allow_during_navigation and large_jump)

    def _jump_decision_dict(self, decision: JumpDecision) -> dict:
        return {
            "jump_status": decision.status,
            "large_jump": decision.large_jump,
            "confirm_count": decision.confirm_count,
            "confirm_needed": decision.confirm_needed,
            "jump_shift_m": round(decision.shift_m, 3),
            "jump_shift_deg": round(decision.shift_deg, 2),
            **self._pose_jump_gate.snapshot(),
        }

    def _publish_relocalize_check(self, result: Mapping[str, ValueTypes]) -> dict:
        return self._localize._publish_relocalize_check(result)

    def _mark_pose_seeded(self, reason: str) -> None:
        """A pose was placed by hand or by an applied search: tracking again."""
        self._loc_state.seeded(reason)
        self._last_relocalize_check["state"] = self._loc_state.snapshot()

    def _localization_match_quality(
        self, match: Mapping[str, ValueTypes], cfg: SlamConfig
    ) -> tuple[bool, float, Optional[float]]:
        score = float(match.get("score", float("-inf")))
        ray_mae_raw = match.get("ray_mae_m")
        ray_mae = float(ray_mae_raw) if ray_mae_raw is not None else None
        good = score >= cfg.periodic_relocalize_min_score and (
            ray_mae is None or ray_mae <= cfg.periodic_relocalize_max_ray_mae_m
        )
        return good, score, ray_mae

    def _parked_refine_worth_applying(
        self,
        cfg: SlamConfig,
        match: Mapping[str, ValueTypes],
        shift_m: float,
        shift_deg: float,
    ) -> bool:
        """True when a parked 1 m match is one the UI auto-refine would keep.

        The periodic good-match bar (score 0.5, ray error 1.0 m) drops the
        hallway slip the button applies. This uses the button's floors and
        stays inside the 1 m window, and it will not take an ambiguous peak
        or a heading change past ``periodic_relocalize_parked_max_deg``.
        """
        try:
            score = float(match.get("score", float("-inf")))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score) or score < float(cfg.periodic_relocalize_parked_min_score):
            return False
        ray_raw = match.get("ray_mae_m")
        if ray_raw is not None:
            try:
                ray = float(ray_raw)
            except (TypeError, ValueError):
                return False
            if math.isfinite(ray) and ray > float(cfg.periodic_relocalize_parked_max_ray_mae_m):
                return False
        if math.isinf(shift_m) or shift_m > float(cfg.periodic_relocalize_parked_radius_m):
            return False
        if math.isinf(shift_deg) or shift_deg > float(cfg.periodic_relocalize_parked_max_deg):
            return False
        if shift_m < float(cfg.periodic_relocalize_min_shift_m) and (
            shift_deg < float(cfg.periodic_relocalize_min_shift_deg)
        ):
            return False
        if bool(match.get("ambiguous")):
            return False
        prior_raw = match.get("prior_score")
        if prior_raw is not None:
            try:
                prior = float(prior_raw)
            except (TypeError, ValueError):
                prior = None
            if prior is not None and math.isfinite(prior) and score < prior + 0.08:
                return False
        return True

    def _stopped_local_beats_published(
        self,
        cfg: SlamConfig,
        match: Mapping[str, ValueTypes],
        shift_m: float,
        shift_deg: float,
        match_command: Mapping[str, ValueTypes],
        *,
        driving: bool,
        match_mode: str,
    ) -> bool:
        """True when a stopped wider-local peak should replace the published pose.

        The 1 m parked window has its own apply path. This is the ordinary
        local search around a pose (the 3 m watchdog window). A coin-flip
        with the next wall stays refused. The peak has to clear the recovery
        score floor and beat the pose that is published now.
        """
        if driving or not str(match_mode).startswith("local"):
            return False
        if bool(match.get("ambiguous")):
            return False
        try:
            score = float(match.get("score", float("-inf")))
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(score)
            or score < float(cfg.periodic_relocalize_recovery_min_score)
        ):
            return False
        ray_mae: Optional[float] = None
        ray_raw = match.get("ray_mae_m")
        if ray_raw is not None:
            try:
                ray_mae = float(ray_raw)
            except (TypeError, ValueError):
                return False
            if (
                math.isfinite(ray_mae)
                and ray_mae > float(cfg.periodic_relocalize_parked_max_ray_mae_m)
            ):
                return False
        if math.isinf(shift_m) or math.isinf(shift_deg):
            return False
        if shift_m < float(cfg.periodic_relocalize_min_shift_m) and (
            shift_deg < float(cfg.periodic_relocalize_min_shift_deg)
        ):
            return False
        try:
            radius = float(match_command.get("search_radius_m") or 0.0)
        except (TypeError, ValueError):
            radius = 0.0
        if radius <= 0.0:
            try:
                radius = float(
                    dict(cfg.periodic_relocalize_options).get("search_radius_m") or 3.0
                )
            except (TypeError, ValueError):
                radius = 3.0
        if shift_m > radius:
            return False
        prior_raw = match.get("prior_score")
        if prior_raw is None:
            return False
        try:
            prior = float(prior_raw)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(prior):
            return False
        prior_ray_raw = match.get("prior_ray_mae_m")
        prior_ray: Optional[float] = None
        if prior_ray_raw is not None:
            try:
                prior_ray = float(prior_ray_raw)
            except (TypeError, ValueError):
                prior_ray = None
        return candidate_beats_previous(
            previous_score=prior,
            candidate_score=score,
            shift_m=float(shift_m),
            shift_deg=float(shift_deg),
            previous_ray_mae_m=prior_ray,
            candidate_ray_mae_m=ray_mae,
        )

    def _periodic_relocalize_preflight(
        self, cfg: SlamConfig
    ) -> Optional[Mapping[str, ValueTypes]]:
        return self._localize._periodic_relocalize_preflight(cfg)

    @staticmethod
    def _pose_shift_from_current(
        current: Optional[conv.Pose2D], matched_pose: Optional[Mapping]
    ) -> tuple[float, float]:
        if current is None or not isinstance(matched_pose, Mapping):
            return float("inf"), float("inf")
        shift_m = math.hypot(
            float(matched_pose.get("x", 0.0)) - current.x,
            float(matched_pose.get("y", 0.0)) - current.y,
        )
        shift_deg = abs(
            math.degrees(
                conv.normalize_angle(
                    float(matched_pose.get("theta", 0.0)) - current.theta
                )
            )
        )
        return shift_m, shift_deg

    def _tick_match_score(self) -> Optional[float]:
        """Latest continuous scan-match score for the published pose, if any."""
        engine = self._engine
        if engine is None:
            return None
        try:
            diag = engine.diagnostics()
        except Exception:  # noqa: BLE001
            return None
        raw = diag.get("last_match_score")
        if raw is None:
            return None
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def _published_pose_is_bad(self, cfg: SlamConfig) -> tuple[bool, Optional[float]]:
        """True when the live tick match says the published pose is wrong."""
        score = self._tick_match_score()
        if score is None:
            return False, None
        return score <= float(cfg.periodic_relocalize_still_bad_score), score

    def _maybe_bypass_startup_localize(
        self, cfg: SlamConfig
    ) -> Optional[Mapping[str, ValueTypes]]:
        return self._localize._maybe_bypass_startup_localize(cfg)

    async def _periodic_relocalize_cycle(
        self,
        *,
        apply_override: Optional[bool] = None,
        full_map_escalation: str = "low_quality",
        allow_during_navigation: bool = False,
        stopped_recovery: bool = False,
    ) -> Mapping[str, ValueTypes]:
        return await self._localize._periodic_relocalize_cycle(
            apply_override=apply_override,
            full_map_escalation=full_map_escalation,
            allow_during_navigation=allow_during_navigation,
            stopped_recovery=stopped_recovery,
        )

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
        # Last-pose restore already seeds near truth — prefer a local refine
        # instead of a full-map search that can yank a good seed.
        restored = False
        mgr = self._manager
        if mgr is not None and bool(
            getattr(cfg, "persist_pose_local_refine_on_start", True)
        ):
            probe = getattr(mgr, "pose_restored_from_disk", None)
            if callable(probe):
                try:
                    restored = bool(probe())
                except Exception:  # noqa: BLE001
                    restored = False
            if not restored:
                try:
                    restored = bool(
                        (mgr.slam_diagnostics() or {}).get(
                            "pose_restored_from_disk", False
                        )
                    )
                except Exception:  # noqa: BLE001
                    restored = False
        if restored:
            options = dict(refine_options)
            options.setdefault("full_map", False)
            options.setdefault("map_source", "live")
            LOGGER.info(
                "startup localize using local refine around restored last pose "
                "(full_map disabled)"
            )
        LOGGER.info(
            "scheduling startup global_localize (full_map=%s delay=%.1fs)",
            options.get("full_map", True),
            delay_s,
        )
        self._startup_localize_local = not bool(options.get("full_map", True))
        self._startup_localize_started_at = time.monotonic()
        self._startup_global_localize_task = loop.create_task(
            self._run_startup_global_localize(
                options,
                delay_s=delay_s,
                readiness_timeout_s=max(
                    0.0, float(cfg.global_localize_on_start_readiness_timeout_s)
                ),
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

    def abort_background_localize(self) -> None:
        """Cancel startup global_localize so navigation can drive the base.

        Full-map matching runs in an executor, but readiness polls / scan reads
        still contend on the shared module loop with ``Base.SetVelocity``.
        """
        self._cancel_startup_global_localize_task()

    def _is_navigation_active(self) -> bool:
        """True when a navigation goal is in flight.

        Builtin SLAM's manager (``BuiltinSlamHost``) always reports nav idle —
        the real active flag lives on the registered ``BuiltinNavHost``. Without
        that lookup, startup/periodic ``global_localize`` keeps running during
        MoveOnMap and starves ``Base.SetVelocity`` (``last_drive.issued: false``,
        ``Viam IO timed out``) — mapping mode looked fine because those localize
        tasks never start.
        """
        active = any_navigation_active()
        if not active:
            mgr = self._manager
            if mgr is not None:
                try:
                    active = bool(mgr.nav_status().get("active", False))
                except Exception:  # noqa: BLE001 - nav stack may not be up yet
                    active = False
        if active:
            # Free the event loop for SetVelocity; refine passes check this too.
            self._cancel_startup_global_localize_task()
        return active

    @staticmethod
    def _startup_global_localize_quality(
        result: Mapping[str, ValueTypes],
    ) -> tuple[float, float, float]:
        return LocalizeSupervisor._startup_global_localize_quality(result)

    @staticmethod
    def _startup_global_localize_meets_target(
        result: Mapping[str, ValueTypes],
        *,
        target_score: float,
        target_ray_mae_m: float,
    ) -> bool:
        return LocalizeSupervisor._startup_global_localize_meets_target(
            result,
            target_score=target_score,
            target_ray_mae_m=target_ray_mae_m,
        )

    async def _wait_for_startup_localize_ready(
        self, *, timeout_s: float = 90.0, poll_interval_s: float = 2.0
    ) -> bool:
        return await self._localize._wait_for_startup_localize_ready(
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
    async def _startup_apply_parked_refine(self) -> bool:
        """Apply one parked auto-refine while startup is still the local search.

        Returns True when the pose was moved. A match the button would refuse
        returns False so the wider startup search can still run.
        """
        cfg = self._cfg
        mgr = self._manager
        if cfg is None or mgr is None:
            return False
        needed = max(1, int(cfg.localize_jump_confirm_count))
        for _ in range(needed):
            if self._is_navigation_active():
                return False
            result = await self.do_command(
                {
                    "command": "global_localize",
                    "apply": False,
                    "full_map": False,
                    "auto_full_map_fallback": False,
                    "map_source": "live",
                    "search_radius_m": float(cfg.periodic_relocalize_parked_radius_m),
                    "local_yaw_window_deg": float(
                        cfg.periodic_relocalize_parked_yaw_window_deg
                    ),
                }
            )
            if not isinstance(result, Mapping):
                return False
            pose_map = result.get("pose")
            if not isinstance(pose_map, Mapping):
                return False
            current = mgr.get_pose_in_map()
            shift_m, shift_deg = self._pose_shift_from_current(current, pose_map)
            if not self._parked_refine_worth_applying(
                cfg, result, shift_m, shift_deg
            ):
                LOGGER.info(
                    "startup parked refine: not an auto-refine match "
                    "(score=%s ray_mae=%s shift=%.2f m / %.1f deg)",
                    result.get("score"),
                    result.get("ray_mae_m"),
                    0.0 if math.isinf(shift_m) else shift_m,
                    0.0 if math.isinf(shift_deg) else shift_deg,
                )
                return False
            candidate = self._pose_from_mapping(pose_map)
            decision = self._confirm_pose_jump(current, candidate, force=False)
            if not decision.should_apply:
                LOGGER.info(
                    "startup parked refine: %.2f m / %.1f deg awaiting "
                    "confirm %d/%d",
                    decision.shift_m,
                    decision.shift_deg,
                    decision.confirm_count,
                    decision.confirm_needed,
                )
                continue
            await self.do_command(
                {
                    "command": "relocalize",
                    "pose": {
                        "x": float(pose_map.get("x", 0.0)),
                        "y": float(pose_map.get("y", 0.0)),
                        "theta": float(pose_map.get("theta", 0.0)),
                    },
                    "position_variance_m2": 0.25,
                    "yaw_variance_rad2": 0.06853891945200942,
                }
            )
            LOGGER.info(
                "startup parked refine applied (%.2f, %.2f, %.2f) "
                "shift=%.2f m score=%s ray_mae=%s",
                candidate.x,
                candidate.y,
                candidate.theta,
                decision.shift_m,
                result.get("score"),
                result.get("ray_mae_m"),
            )
            return True
        return False

    async def _run_startup_global_localize(
        self,
        options: Mapping[str, ValueTypes],
        *,
        delay_s: float = 4.0,
        max_attempts: int = 5,
        retry_delay_s: float = 5.0,
        readiness_timeout_s: float = 90.0,
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
        await self._localize._run_startup_global_localize(
            options,
            delay_s=delay_s,
            max_attempts=max_attempts,
            retry_delay_s=retry_delay_s,
            readiness_timeout_s=readiness_timeout_s,
            run_refine_pass=run_refine_pass,
            refine_delay_s=refine_delay_s,
            refine_max_passes=refine_max_passes,
            target_score=target_score,
            target_ray_mae_m=target_ray_mae_m,
            refine_options=refine_options,
            run_post_apply_refine=run_post_apply_refine,
            post_apply_refine_delay_s=post_apply_refine_delay_s,
            post_apply_refine_options=post_apply_refine_options,
        )
    def _make_typed_odom_reader(self):
        """Portable GetLinearVelocity / GetAngularVelocity reader for wheel odom."""
        if self._movement_sensor is None:
            return None
        from ..odom.source import TypedMovementSensorOdom, TypedOdomConfig

        assert self._cfg is not None
        return TypedMovementSensorOdom(
            self._movement_sensor,
            TypedOdomConfig(
                velocity_convention=self._cfg.base_velocity_convention,
            ),
            logger=LOGGER,
        )

    def _build_io(self):
        from ..sensors.viam_io import build_io_provider

        assert self._cfg is not None
        # Prefer typed MovementSensor getters (GetLinearVelocity / AngularVelocity)
        # so velocity-only wheeled odometry (no Position / sparse get_readings)
        # still feeds /odom. Same reader as navigation-external.
        node = self._manager.node if self._manager else None
        return build_io_provider(
            base=self._base,
            cameras=self._cameras,
            cfg=self._cfg,
            movement_sensor=self._movement_sensor,
            heading_sensor=self._heading_sensor,
            skip_get_laser_scan=self._skip_get_laser_scan,
            odom_reader=self._make_typed_odom_reader(),
            logger=LOGGER,
            record_cmd_vel=getattr(node, "record_cmd_vel", None),
            shm_lidar=self._shm_lidar,
        )

    async def _probe_sensors(self) -> dict:
        """One-shot lidar + odom read for get_status (does not affect /scan)."""
        assert self._cfg is not None
        if self._engine is not None:
            return await self._probe_sensors_builtin()
        io = self._build_io()
        lidars: list[dict] = []
        for lidar in self._cfg.lidars:
            entry: dict = {
                "name": lidar.name,
                "scan_source": lidar.scan_source,
                "shm_name": lidar.shm_name,
                "obstacles_only": bool(lidar.obstacles_only),
            }
            try:
                data = await io.read_lidar_points(lidar.name)
                sensor_pts = np.asarray(data.sensor, dtype=float)
                base_pts = np.asarray(data.base_link, dtype=float)
                entry["sensor_points"] = int(sensor_pts.shape[0])
                entry["base_link_points"] = int(base_pts.shape[0])
                if data.age_s is not None:
                    entry["age_s"] = float(data.age_s)
                scan_pts = base_pts if base_pts.size else sensor_pts
                if scan_pts.size:
                    # Height-band stats help catch mount/z-filter mistakes
                    # (e.g. all points discarded because the cloud is still in
                    # the sensor frame, or the mast tilt pulls the floor in).
                    if scan_pts.shape[1] >= 3:
                        z = scan_pts[:, 2]
                        in_band = int(
                            np.count_nonzero((z >= lidar.z_min) & (z <= lidar.z_max))
                        )
                        entry["z_band_points"] = in_band
                        entry["z_min_m"] = float(np.min(z))
                        entry["z_max_m"] = float(np.max(z))
                    scan = conv.pointcloud_to_scan(
                        scan_pts,
                        z_min=lidar.z_min,
                        z_max=lidar.z_max,
                        num_bins=self._cfg.scan_bins,
                        range_min=lidar.min_range,
                        range_max=lidar.max_range,
                    )
                    entry["scan_valid_returns"] = sum(
                        1
                        for r in scan.ranges
                        if math.isfinite(r)
                        and r >= scan.range_min
                        and (not math.isfinite(scan.range_max) or r <= scan.range_max)
                    )
                    # Nearest return within ±15° of robot forward — standing in
                    # front of the cart should drop this even when the occupancy
                    # map stays frozen (scans are only added after travel).
                    forward = conv.forward_sector_min_range(
                        scan, half_width_rad=math.radians(15.0)
                    )
                    if forward is not None:
                        entry["forward_min_range_m"] = round(forward, 3)
                    bearing = conv.nearest_return_bearing_deg(scan)
                    if bearing is not None:
                        entry["nearest_return_bearing_deg"] = round(bearing, 1)
                        # Suggest the mount.theta that would put this wall on +X.
                        entry["suggested_mount_theta_deg"] = round(-bearing, 1)
                else:
                    entry["scan_valid_returns"] = 0
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                entry["error"] = repr(exc)
            lidars.append(entry)

        odom_probe: dict = {}
        try:
            sample = await io.read_odometry()
            odom_probe = {
                "vx": sample.vx,
                "vy": sample.vy,
                "vtheta": sample.vtheta,
                "has_pose": sample.pose is not None,
                "has_heading": sample.heading_rad is not None,
                "has_twist": (
                    abs(sample.vx) > 1e-9
                    or abs(sample.vy) > 1e-9
                    or abs(sample.vtheta) > 1e-9
                ),
                "has_acceleration": sample.ax is not None and sample.ay is not None,
            }
            if sample.ax is not None and sample.ay is not None:
                odom_probe["ax"] = sample.ax
                odom_probe["ay"] = sample.ay
            if sample.pose is not None:
                odom_probe["pose"] = {
                    "x": sample.pose.x,
                    "y": sample.pose.y,
                    "theta": sample.pose.theta,
                }
            if sample.heading_rad is not None:
                odom_probe["heading_rad"] = sample.heading_rad
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            odom_probe["error"] = repr(exc)

        return {"lidars": lidars, "odometry": odom_probe}

    async def _probe_sensors_builtin(self) -> dict:
        """Probe via BuiltinSensors (no ROS bridge / IOProvider)."""
        assert self._engine is not None and self._cfg is not None
        sensors = self._engine._sensors  # noqa: SLF001
        lidars: list[dict] = []
        for lidar in self._cfg.lidars:
            entry: dict = {
                "name": lidar.name,
                "scan_source": lidar.scan_source,
                "shm_name": lidar.shm_name,
                "obstacles_only": bool(lidar.obstacles_only),
            }
            try:
                # SimSensors has get_scan only (no Viam Camera / _read_lidar_scan_sync).
                read_sync = getattr(sensors, "_read_lidar_scan_sync", None)
                if read_sync is not None:
                    scan = await asyncio.to_thread(
                        read_sync, lidar, max_age_s=2.0
                    )
                else:
                    scan = await asyncio.to_thread(sensors.get_scan, 2.0)
                if scan is None:
                    entry["scan_valid_returns"] = 0
                else:
                    entry["scan_valid_returns"] = sum(
                        1
                        for r in scan.ranges
                        if math.isfinite(r)
                        and r >= scan.range_min
                        and (
                            not math.isfinite(scan.range_max) or r <= scan.range_max
                        )
                    )
                    forward = conv.forward_sector_min_range(
                        scan, half_width_rad=math.radians(15.0)
                    )
                    if forward is not None:
                        entry["forward_min_range_m"] = round(forward, 3)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = repr(exc)
            lidars.append(entry)

        odom_probe: dict = {}
        try:
            sample = await asyncio.to_thread(sensors.get_odom)
            if sample is None:
                odom_probe["error"] = "no_odometry"
            else:
                odom_probe = {
                    "vx": sample.vx,
                    "vy": sample.vy,
                    "vtheta": sample.vtheta,
                    "has_pose": sample.pose is not None,
                    "has_heading": sample.heading_rad is not None,
                    "has_twist": (
                        abs(sample.vx) > 1e-9
                        or abs(sample.vy) > 1e-9
                        or abs(sample.vtheta) > 1e-9
                    ),
                }
                if sample.heading_rad is not None:
                    odom_probe["heading_deg"] = round(
                        math.degrees(sample.heading_rad), 2
                    )
                heading_dbg = getattr(sensors, "heading_debug", None)
                if callable(heading_dbg):
                    hd = heading_dbg()
                    if hd:
                        odom_probe["heading"] = hd
                        if hd.get("source"):
                            odom_probe["heading_source"] = hd.get("source")
                debug = sensors.odom_debug()
                if debug:
                    odom_probe["raw"] = debug
        except Exception as exc:  # noqa: BLE001
            odom_probe["error"] = repr(exc)
        return {"lidars": lidars, "odometry": odom_probe}

    def _start_mode(self, mode: str) -> None:
        assert self._manager and self._map_store and self._cfg
        handle = self._map_store.active_handle()
        stem = handle.serialization_stem if handle else None
        self._manager.start_slam(stem, mode)
        self._cfg.mode = mode
        # Slice reference geometry is pose-graph-relative; a (re)started SLAM
        # session invalidates it.
        self._reset_slice_library()
        # Spawn the matcher subprocess now so the first relocalize / revisit
        # check does not pay the import cost.
        self._localizer.warm()

    def _reset_live_slam(self, mode: str) -> None:
        """Reset SLAM in place when possible; full restart as fallback."""
        assert self._manager is not None
        mgr = self._manager
        if mode == MODE_MAPPING and mgr.slam_running() and mgr.reset_slam_map():
            if self._cfg is not None:
                self._cfg.mode = mode
            # The in-place reset wiped the pose graph, so pose-graph-relative
            # slice/keyframe references are stale too.
            self._reset_slice_library()
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
    def get_position_pose2d(self) -> Optional[conv.Pose2D]:
        """Sync map-frame pose in meters / radians (same source as ``GetPosition``).

        Builtin nav calls this from its control-loop thread so pose feedback does
        not hop through the module event loop via the async SLAM API. Returns
        ``None`` when the pose is unknown (unlike ``GetPosition``, which must
        return an origin placeholder for the Viam API).
        """
        mgr = self._manager
        if mgr is None:
            return None
        getter = getattr(mgr, "get_pose_in_map", None)
        if callable(getter):
            pose2d = getter()
            if pose2d is not None:
                return pose2d
        node = getattr(mgr, "node", None)
        if node is not None and hasattr(node, "get_pose_in_map"):
            return node.get_pose_in_map()
        return None

    async def get_position(self, *, timeout: Optional[float] = None, **kwargs) -> Pose:
        pose2d = self.get_position_pose2d()
        if pose2d is None:
            # Viam SLAM API requires a Pose; origin means "unknown" to some
            # clients. Builtin nav uses ``get_position_pose2d`` (None) instead.
            return Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
        offset = float(
            getattr(self._cfg, "map_pose_yaw_offset_deg", 0.0) if self._cfg else 0.0
        )
        x_mm, y_mm, z_mm, o_x, o_y, o_z, theta_deg = conv.pose2d_to_viam_slam_pose(
            pose2d, yaw_offset_deg=offset
        )
        return Pose(
            x=x_mm, y=y_mm, z=z_mm, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta_deg
        )

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

    def slam_mode(self) -> Optional[str]:
        """Current SLAM mode without a DoCommand round-trip."""
        cfg = self._cfg
        if cfg is None:
            return None
        return str(cfg.mode)

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> SLAM.Properties:
        mapping = self._cfg is not None and self._cfg.mode == MODE_MAPPING
        mode = (
            MappingMode.MAPPING_MODE_CREATE_NEW_MAP
            if mapping
            else MappingMode.MAPPING_MODE_LOCALIZE_ONLY
        )
        builtin = self._cfg is not None and self._cfg.uses_builtin_slam()
        return SLAM.Properties(
            cloud_slam=False,
            mapping_mode=mode,
            internal_state_file_type="" if builtin else ".posegraph",
            sensor_info=[],
        )

    # -- DoCommand -----------------------------------------------------------
    def _do_command_handlers(self):
        return {
            "get_mode": self._cmd_get_mode,
            "get_grid": self._cmd_get_grid,
            "get_status": self._cmd_get_status,
            "start_mapping": self._cmd_start_mapping,
            "start_localizing": self._cmd_start_localizing,
            "save_map": self._cmd_save_map,
            "optimize": self._cmd_optimize,
            "set_initial_pose": self._cmd_set_initial_pose,
            "relocalize": self._cmd_relocalize,
            "global_localize": self._cmd_global_localize,
            "check_localization": self._cmd_check_localization,
            "get_localization_check": self._cmd_get_localization_check,
            "stuck_pose_refine": self._cmd_stuck_pose_refine,
            "revisit_check": self._cmd_revisit_check,
            "get_revisit_check": self._cmd_get_revisit_check,
            "list_maps": self._cmd_list_maps,
            "get_active_map": self._cmd_get_active_map,
            "set_active_map": self._cmd_set_active_map,
            "rename_map": self._cmd_rename_map,
            "clear_map": self._cmd_clear_map,
            "clear_obstacles": self._cmd_clear_obstacles,
            "add_obstacles": self._cmd_add_obstacles,
            "delete_map": self._cmd_delete_map,
        }

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        if self._map_store is None or self._manager is None:
            raise RuntimeError("SLAM service not configured")
        return await dispatch_do_command(
            self._do_command_handlers(),
            command,
            aliases=_SLAM_DO_ALIASES,
        )

    def _cmd_get_mode(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        return {"mode": self._cfg.mode if self._cfg else None}


    async def _cmd_get_grid(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        # Occupancy for external consumers (builtin nav ViamWorldIO, etc.).
        def _grid():
            node = mgr.node
            if node is None:
                return None
            return node.get_map()

        map_data = await asyncio.to_thread(_grid)
        if map_data is None or map_data.get("grid") is None:
            return {"available": False, "reason": "no_map"}
        from ..nav_builtin.viam_io import map_dict_to_get_grid

        return map_dict_to_get_grid(map_data)


    async def _cmd_get_status(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        probe_sensors = require(command, "probe_sensors", bool, default=True)

        def _status():
            return mgr.slam_diagnostics()

        status = await asyncio.to_thread(_status)
        status["mode"] = self._cfg.mode if self._cfg else None
        status["slam_backend"] = (
            self._cfg.slam_backend if self._cfg else None
        )
        status["localize_worker"] = dict(self._localizer.stats)
        status["active_map"] = store.get_active_map_name()
        if self._cfg is not None:
            status["base"] = self._cfg.base
            status["movement_sensor"] = self._cfg.movement_sensor
            status["movement_sensor_yaw_deg"] = self._cfg.movement_sensor_yaw_deg
            status["map_when_still"] = self._cfg.map_when_still
            status["heading_sensor"] = self._cfg.heading_sensor
            status["lidars"] = [
                {
                    "name": lidar.name,
                    "scan_source": lidar.scan_source,
                    "mount": {
                        "x": lidar.x,
                        "y": lidar.y,
                        "z": lidar.z,
                        "theta": lidar.theta,
                        "pitch": lidar.pitch,
                        "roll": lidar.roll,
                    },
                    "z_min": lidar.z_min,
                    "z_max": lidar.z_max,
                    "min_range": lidar.min_range,
                    "max_range": lidar.max_range,
                    "shm_name": lidar.shm_name,
                    "shm_required": lidar.shm_required,
                    "obstacles_only": bool(lidar.obstacles_only),
                }
                for lidar in self._cfg.lidars
            ]
        status["localization_check"] = dict(self._last_relocalize_check)
        status["localization_state"] = self._loc_state.snapshot()
        status["revisit_check"] = dict(self._last_revisit_check)
        if self._pause_keyframes is not None:
            status["pause_keyframes"] = self._pause_keyframes.status()
        shm_status = self._shm_lidar.status()
        if shm_status:
            status["lidar_shm"] = shm_status
        if probe_sensors and self._cfg is not None:
            status["sensor_probe"] = await self._probe_sensors()
        return status


    async def _cmd_start_mapping(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        name = command.get("map")
        if name:
            store.get_or_create_map(str(name))
            store.set_active_map(str(name))
        # Subprocess-heavy manager calls must stay off the module event
        # loop: the bridge marshals odom/lidar/cmd_vel onto it, so blocking
        # here stalls TF and scans.
        await asyncio.to_thread(self._start_mode, MODE_MAPPING)
        self._reschedule_mode_watchdogs()
        return {"status": "mapping", "map": store.get_active_map_name()}


    async def _cmd_start_localizing(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        name = command.get("map")
        if name:
            store.set_active_map(str(name))
        handle = store.active_handle()
        need_map = (
            handle.has_any_map()
            if self._cfg is not None and self._cfg.uses_builtin_slam()
            else (handle.has_serialized_map() if handle else False)
        )
        if not handle or not need_map:
            raise ValueError("active map has no saved data; map it first")
        await asyncio.to_thread(self._start_mode, MODE_LOCALIZING)
        self._reschedule_mode_watchdogs()
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


    async def _cmd_save_map(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        handle = store.active_handle()
        if not handle:
            raise ValueError("no active map")
        # save_map runs ros2 CLI subprocesses (30-40s timeouts); keep the
        # event loop free so mapping TF/scans continue while saving.
        await asyncio.to_thread(mgr.save_map, handle.serialization_stem)
        return {"status": "saved", "map": handle.name}


    async def _cmd_optimize(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        # Force SPA: there is no bare CorrectPoses service, so we
        # serialize + deserialize the live graph (loader calls Compute()).
        if self._cfg is not None and self._cfg.mode != MODE_MAPPING:
            raise ValueError(
                "optimize requires SLAM mode mapping; call start_mapping first"
            )
        handle = store.active_handle()
        if not handle:
            raise ValueError("no active map")
        return await asyncio.to_thread(
            mgr.optimize_pose_graph, handle.serialization_stem
        )


    async def _cmd_set_initial_pose(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        pose = self._resolve_pose(command)
        await asyncio.to_thread(mgr.set_initial_pose, pose)
        # Manual seed wins over any in-flight large-jump confirm toward a
        # corridor twin — otherwise the next watchdog tick can finish the
        # confirm and yank the pose back across the map.
        self._pose_jump_gate.clear()
        self._mark_pose_seeded("set_initial_pose")
        # Also cancel boot full-map localize: it can still finish after the
        # user has already placed the robot and overwrite a good seed.
        self._cancel_startup_global_localize_task()
        # ``refine: true`` runs a seeded scan match around the given XY with
        # a full yaw sweep — the matcher itself only searches ~±30° of
        # heading, so a seed with roughly-right XY but wrong theta can never
        # self-correct without this.
        if command.get("refine"):
            refine_cmd: dict = {
                "command": "global_localize",
                "pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
                # Keep XY close to the manual seed. A 3 m window was sliding
                # ~2 m along repetitive corridor geometry to a slightly
                # different local max; 1 m is enough to fix heading/offset.
                "search_radius_m": float(command.get("search_radius_m", 1.0)),
                "local_yaw_window_deg": float(
                    command.get("local_yaw_window_deg", 360.0)
                ),
                "full_map": False,
                "auto_full_map_fallback": False,
                # Manual seed refine must be willing to accept a weak-but-
                # better peak (corridor twins often top out ~0.3). Absolute
                # floors still apply via beat-prior logic in _global_localize;
                # also drop ambiguous refuse so a near-tie local peak can land.
                "min_apply_score": float(command.get("min_apply_score", 0.22)),
                "max_apply_ray_mae_m": float(
                    command.get("max_apply_ray_mae_m", 1.5)
                ),
                "refuse_ambiguous": bool(command.get("refuse_ambiguous", False)),
            }
            # Forward optional denser-search knobs from the caller.
            for key in (
                "coarse_position_step_m",
                "coarse_yaw_step_deg",
                "fine_position_step_m",
                "fine_yaw_step_deg",
                "ray_weight",
                "ray_refine_candidates",
                "ray_refine_beams",
            ):
                if key in command:
                    refine_cmd[key] = command[key]
            result = await self._global_localize(refine_cmd)
            return {"status": "ok", "refine": result}
        return {"status": "ok"}


    async def _cmd_relocalize(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        cmd = str(command.get("command"))
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
        self._mark_pose_seeded(cmd)
        return {
            "status": "relocalizing",
            "seed_pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
        }


    async def _cmd_global_localize(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        return await self._global_localize(command)


    async def _cmd_check_localization(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        # Run one drift-watchdog cycle on demand. ``apply`` forces or
        # suppresses the correction; omit it to use the drift thresholds.
        # ``full_map_escalation``: ``low_quality`` (default) or ``still_bad``
        # (local first; full-map only when the published pose still looks wrong).
        # ``stopped_recovery``: nav already halted for bad loc — wider search
        # and a larger trusted apply so the goal can continue.
        apply_override = command.get("apply")
        escalation = command.get("full_map_escalation", "low_quality")
        return await self._periodic_relocalize_cycle(
            apply_override=(
                None if apply_override is None else bool(apply_override)
            ),
            full_map_escalation=str(escalation or "low_quality"),
            allow_during_navigation=bool(
                command.get("allow_during_navigation")
            ),
            stopped_recovery=bool(command.get("stopped_recovery")),
        )


    def _cmd_get_localization_check(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        return dict(self._last_relocalize_check)


    async def _cmd_stuck_pose_refine(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        # One ~1 m local match while navigation cannot leave its cell.
        # Mapping and localizing both apply a small better pose. No full-map search.
        return await self._stuck_pose_refine()


    async def _cmd_revisit_check(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        # Mapping-mode revisit check on demand. ``apply`` forces the odom
        # correction (or set false for a dry run); omit for gated behavior.
        # ``yaw_flip`` takes the opposite heading of the auto corridor
        # disambiguation (same XY). ``flip_yaw_only`` reverses the current
        # map heading in place without re-matching — recovery when XY is
        # already right but facing is 180° wrong.
        if bool(command.get("flip_yaw_only")):
            return await self._flip_current_map_yaw()
        apply_override = command.get("apply")
        return await self._mapping_revisit_cycle(
            apply_override=(
                None if apply_override is None else bool(apply_override)
            ),
            yaw_flip=bool(command.get("yaw_flip")),
        )


    def _cmd_get_revisit_check(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        return dict(self._last_revisit_check)

        # -- map management --


    def _cmd_list_maps(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        return {"maps": store.list_maps()}


    def _cmd_get_active_map(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        return {"map": store.get_active_map_name()}


    def _cmd_set_active_map(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        assert store is not None
        store.set_active_map(require(command, "map", str))
        return {"map": store.get_active_map_name()}


    def _cmd_rename_map(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        assert store is not None
        handle = store.rename_map(
            require(command, "map", str),
            require(command, "new_name", str),
        )
        return {"map": handle.name}


    async def _cmd_clear_map(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
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


    async def _cmd_clear_obstacles(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        return await self._cmd_paint_obstacles(command, adding=False)

    async def _cmd_add_obstacles(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        return await self._cmd_paint_obstacles(command, adding=True)

    async def _cmd_paint_obstacles(
        self, command: Mapping[str, ValueTypes], *, adding: bool
    ) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
        # Paint free or occupied space into the live occupancy grid (builtin SLAM).
        x = command.get("x")
        y = command.get("y")
        if x is None or y is None:
            pose = command.get("pose")
            if isinstance(pose, Mapping):
                x = pose.get("x")
                y = pose.get("y")
        if x is None or y is None:
            raise ValueError(
                f"{('add_obstacles' if adding else 'clear_obstacles')} requires x,y (or pose.x/pose.y)"
            )
        radius = command.get("radius_m", command.get("radius", 0.35))
        try:
            x_m = float(x)
            y_m = float(y)
            radius_m = float(radius)
        except (TypeError, ValueError) as exc:
            raise ValueError("x, y, radius_m must be numbers") from exc
        if radius_m <= 0:
            raise ValueError("radius_m must be > 0")

        def _paint():
            method = "add_obstacles" if adding else "clear_obstacles"
            paint_fn = getattr(mgr, method, None)
            if paint_fn is None and getattr(mgr, "node", None) is not None:
                paint_fn = getattr(mgr.node, method, None)
            if paint_fn is None:
                raise RuntimeError(
                    f"{method} requires slam_backend=builtin "
                    "(no live occupancy editor on this SLAM host)"
                )
            return paint_fn(x_m, y_m, radius_m)

        result = await asyncio.to_thread(_paint)
        save = command.get("save", False)
        out = dict(result) if isinstance(result, Mapping) else {"result": result}
        out["status"] = "ok"
        out["map"] = store.get_active_map_name()
        out["saved"] = False
        if save:
            handle = store.active_handle()
            if not handle:
                raise ValueError("no active map")
            await asyncio.to_thread(mgr.save_map, handle.serialization_stem)
            out["saved"] = True
        return out


    async def _cmd_delete_map(self, command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        store = self._map_store
        mgr = self._manager
        assert store is not None and mgr is not None
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
                    self._cfg.map.resolution if self._cfg else 0.05
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
        scan, _ = await self._read_merged_scan_and_bands(None)
        return scan

    async def _read_merged_scan_and_bands(
        self, bands: Optional[List[slice_match.SliceBand]]
    ) -> tuple[conv.LaserScan2D, List[np.ndarray]]:
        """Merged 2D scan plus per-band base_link XY points from one lidar read.

        Bands come only from ``get_point_cloud``-style lidars (z is height in
        base_link); pure 2D scans contribute to the merged scan but not bands.
        """
        assert self._cfg is not None
        # Sim has no Viam lidar Camera; placeholder name ``sim-lidar`` is not in
        # ``self._cameras``. Use in-process SimSensors raycast instead.
        if self._cfg.uses_sim() and self._engine is not None:
            sensors = self._engine._sensors  # noqa: SLF001
            scan = await asyncio.to_thread(sensors.get_scan, 2.0)
            if scan is None or not conv.scan_has_returns(scan):
                raise RuntimeError("no lidar returns available for global_localize")
            empty_bands: List[np.ndarray] = (
                [np.empty((0, 2)) for _ in bands] if bands else []
            )
            return scan, empty_bands
        io = self._build_io()
        scans: List[conv.LaserScan2D] = []
        band_points: List[np.ndarray] = (
            [np.empty((0, 2)) for _ in bands] if bands else []
        )
        for lidar in self._cfg.lidars:
            if lidar.obstacles_only:
                continue
            points = await io.read_lidar_points(lidar.name)
            mount = conv.Pose2D(lidar.x, lidar.y, lidar.theta)
            # Prefer base_link points: for MiR get_laser_scan this already merges
            # all available scanners into one common frame.
            if points.base_link.size:
                scan = conv.pointcloud_to_scan(
                    points.base_link,
                    # points.base_link are already in base_link coordinates.
                    # Use the lidar's configured height band — the default band
                    # would let (tilted-mast) floor returns pollute the match.
                    z_min=lidar.z_min,
                    z_max=lidar.z_max,
                    sensor_pose=conv.Pose2D(0.0, 0.0, 0.0),
                    num_bins=self._cfg.scan_bins,
                )
                if conv.scan_has_returns(scan):
                    scans.append(scan)
                    if bands:
                        sliced = slice_match.slice_points_by_bands(
                            np.asarray(points.base_link, dtype=float), bands
                        )
                        band_points = [
                            np.vstack([acc, cur]) if cur.size else acc
                            for acc, cur in zip(band_points, sliced)
                        ]
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
        merged = conv.merge_scans(
            scans,
            num_bins=self._cfg.scan_bins,
            range_max=self._cfg.map.max_laser_range,
        )
        return merged, band_points

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
                return load_occupancy_from_map_dict(live), "live"
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
                "no occupancy map available; save the map or wait for mapping to produce a grid"
            )
        return load_occupancy_from_map_dict(live), "live"

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
        auto_full_map_fallback = bool(command.get("auto_full_map_fallback", False))
        fallback_score_threshold = float(command.get("fallback_score_threshold", 0.42))
        fallback_hit_rate_threshold = float(command.get("fallback_hit_rate_threshold", 0.6))

        loop = asyncio.get_running_loop()

        async def _run_match(full_map_override: bool):
            default_coarse_pos = 0.35 if full_map_override else 0.4
            default_coarse_yaw = 10.0 if full_map_override else 12.0
            default_local_yaw_window = 360.0 if full_map_override else 180.0
            default_ray_weight = 0.55 if full_map_override else 0.35
            default_ray_candidates = 48 if full_map_override else 24
            return await self._localizer.run(
                "global_localize_scan",
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
                ray_refine_candidates=int(
                    command.get("ray_refine_candidates", default_ray_candidates)
                ),
                ray_refine_beams=int(command.get("ray_refine_beams", 64)),
                ray_step_m=float(command.get("ray_step_m", 0.08)),
                ray_weight=float(command.get("ray_weight", default_ray_weight)),
                ambiguity_margin=float(command.get("ambiguity_margin", 0.06)),
            )

        result = await _run_match(full_map)
        fallback_used = False
        if (
            not full_map
            and auto_full_map_fallback
            and (
                result.score < fallback_score_threshold
                or result.hit_rate < fallback_hit_rate_threshold
            )
        ):
            full_result = await _run_match(True)
            if (
                full_result.score > result.score
                or full_result.hit_rate > result.hit_rate
            ):
                result = full_result
                fallback_used = True
        resolved_full_map = full_map or fallback_used

        prior_score = None
        prior_ray_mae_m = None
        if hint is not None:
            prior_score, prior_ray_mae_m = await self._localizer.run(
                "score_pose_with_rays",
                occ_map,
                scan,
                hint,
                max_scan_points=int(command.get("max_scan_points", 240)),
                min_in_map_points=int(command.get("min_in_map_points", 40)),
                min_in_map_ratio=float(command.get("min_in_map_ratio", 0.35)),
                hit_radius_cells=int(command.get("hit_radius_cells", 2)),
                ray_refine_beams=int(command.get("ray_refine_beams", 64)),
                ray_step_m=float(command.get("ray_step_m", 0.08)),
            )

        # Refuse to auto-apply a weak or ambiguous full-map peak — that was
        # placing the robot in a "random" corridor that then locked continuous
        # matching. Local refine around a good hint still applies freely.
        min_apply_score = float(
            command.get(
                "min_apply_score",
                0.55 if resolved_full_map else 0.35,
            )
        )
        max_apply_ray_mae_m = float(
            command.get(
                "max_apply_ray_mae_m",
                0.55 if resolved_full_map else 1.0,
            )
        )
        refuse_ambiguous = bool(command.get("refuse_ambiguous", resolved_full_map))
        apply_blocked_reason = None
        if apply_pose:
            if float(result.score) < min_apply_score:
                apply_blocked_reason = "low_score"
            elif (
                math.isfinite(float(result.ray_mae_m))
                and float(result.ray_mae_m) > max_apply_ray_mae_m
            ):
                apply_blocked_reason = "high_ray_mae"
            elif refuse_ambiguous and bool(getattr(result, "ambiguous", False)):
                apply_blocked_reason = "ambiguous"

        # Seeded local refine (UI auto-refine / set_initial_pose): if the match
        # clearly beats the seed prior, apply even when below the absolute score
        # floor. Otherwise a nearly-right XY with wrong heading stays stuck —
        # best local peak is often ~0.30 while min_apply_score defaults to 0.35.
        if (
            apply_pose
            and apply_blocked_reason == "low_score"
            and not resolved_full_map
            and prior_score is not None
            and math.isfinite(float(prior_score))
            and float(result.score) >= float(prior_score) + 0.08
        ):
            apply_blocked_reason = None

        did_apply = False
        if apply_pose and apply_blocked_reason is None:
            await loop.run_in_executor(
                None,
                lambda: mgr.relocalize(
                    result.pose,
                    position_variance_m2=0.25,
                    yaw_variance_rad2=0.06853891945200942,
                ),
            )
            did_apply = True
            self._mark_pose_seeded("global_localize")
        elif apply_pose and apply_blocked_reason is not None:
            LOGGER.warning(
                "global_localize: not applying (%s) score=%.3f ray_mae=%s "
                "ambiguous=%s second_best=%s full_map=%s",
                apply_blocked_reason,
                float(result.score),
                result.ray_mae_m,
                getattr(result, "ambiguous", False),
                getattr(result, "second_best_score", None),
                resolved_full_map,
            )

        out: dict = {
            "status": (
                "localized"
                if did_apply
                else ("matched_rejected" if apply_blocked_reason else "matched")
            ),
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
            "coverage": round(float(getattr(result, "coverage", 0.0)), 3),
            "ray_score": result.ray_score,
            "ray_mae_m": result.ray_mae_m,
            "second_best_score": getattr(result, "second_best_score", None),
            "ambiguous": bool(getattr(result, "ambiguous", False)),
            "map_source": map_source,
            "full_map": resolved_full_map,
            "fallback_used": fallback_used,
            "applied": did_apply,
        }
        if apply_blocked_reason is not None:
            out["apply_blocked_reason"] = apply_blocked_reason
        if prior_score is not None:
            out["prior_score"] = prior_score
            out["prior_ray_mae_m"] = prior_ray_mae_m
        return out

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
        self._cancel_periodic_relocalize_task()
        self._cancel_mapping_revisit_task()
        self._cancel_framesystem_task()
        unregister_slam(self.name)
        unregister_slam_service(self.name)
        self._shm_lidar.close()
        self._localizer.close()
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None


Registry.register_resource_creator(
    SLAM.API,
    SlamService.MODEL,
    ResourceCreatorRegistration(SlamService.new, SlamService.validate_config),
)

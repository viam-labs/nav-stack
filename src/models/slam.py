"""SLAM service model: ``viam-labs:nav-stack:slam``.

Wraps slam_toolbox (via the ROS manager/bridge) to provide mapping and
localization for any Viam base, and exposes the standard Viam SLAM service API plus
map-management / mode / initial-pose commands through ``DoCommand``.
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
    MODE_LOCALIZING,
    MODE_MAPPING,
    SlamConfig,
)
from ..nav.global_localize import (
    GlobalLocalizeResult,
    choose_yaw_or_flip,
    global_localize_scan,
    load_occupancy_from_bridge_map,
    load_occupancy_from_map_dir,
)
from ..nav import pause_keyframes, slice_match
from ..nav.maps import MapStore, validate_map_name
from ..ros import conversions as conv
from ..ros.bridge import IOProvider
from ..ros.manager import RosManager
from ..ros.sensor_io import build_io_provider
from ..runtime import SlamRuntime, register_slam, unregister_slam

LOGGER = getLogger(__name__)


# Default /initialpose uncertainty for relocalize (~2 m, ~45 deg std dev).
RELOCALIZE_POSITION_VARIANCE_M2 = 4.0
RELOCALIZE_YAW_VARIANCE_RAD2 = (math.pi / 4) ** 2


def _normalize_angle(rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(rad), math.cos(rad))


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
        self._heading_sensor: Optional[MovementSensor] = None
        self._map_display_hold = False
        self._visible_map_generation = 0
        self._startup_global_localize_task: Optional[asyncio.Task] = None
        self._periodic_relocalize_task: Optional[asyncio.Task] = None
        self._mapping_revisit_task: Optional[asyncio.Task] = None
        self._last_relocalize_check: dict = {"status": "idle"}
        self._last_revisit_check: dict = {"status": "idle"}
        self._slice_library: Optional[slice_match.SliceLibrary] = None
        self._pause_keyframes: Optional[pause_keyframes.PauseKeyframeStore] = None
        self._keyframe_lock = threading.Lock()
        # Lidars that don't implement get_laser_scan (auto mode); skip re-probing.
        self._skip_get_laser_scan: set[str] = set()

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
        self._skip_get_laser_scan = set()
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
        self._heading_sensor = (
            cast(
                MovementSensor,
                dependencies[MovementSensor.get_resource_name(cfg.heading_sensor)],
            )
            if cfg.heading_sensor
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
        self._wire_still_keyframe_hook()
        self._schedule_startup_global_localize(loop)
        self._schedule_periodic_relocalize(loop)
        self._schedule_mapping_revisit(loop)

        register_slam(
            self.name,
            SlamRuntime(self._manager, self._map_store, cfg, self._last_relocalize_check),
        )
        LOGGER.info(f"nav-stack SLAM '{self.name}' configured in {cfg.mode} mode")

    def _cancel_startup_global_localize_task(self) -> None:
        task = self._startup_global_localize_task
        if task is not None and not task.done():
            task.cancel()
        self._startup_global_localize_task = None

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

        slam_toolbox only loop-closes when the drifted return pose is inside its
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
        self._last_revisit_check.clear()
        self._last_revisit_check.update(result)
        return dict(self._last_revisit_check)

    async def _flip_current_map_yaw(self) -> Mapping[str, ValueTypes]:
        """Reverse map-frame heading in place (corridor 180° recovery)."""
        mgr = self._manager
        if mgr is None:
            return self._publish_revisit_check({"status": "unconfigured"})
        current = mgr.get_pose_in_map()
        if current is None:
            return self._publish_revisit_check(
                {"status": "skipped", "reason": "no_pose_in_map"}
            )
        flipped = conv.Pose2D(
            current.x, current.y, _normalize_angle(current.theta + math.pi)
        )
        applied = await asyncio.to_thread(mgr.apply_map_pose_correction, flipped)
        out: dict = {
            "status": "corrected" if applied.get("applied") else "correction_failed",
            "flip_yaw_only": True,
            "yaw_flipped": True,
            "pose": {"x": flipped.x, "y": flipped.y, "theta": flipped.theta},
            "correction": applied,
            "corrected": bool(applied.get("applied")),
        }
        if out["corrected"]:
            LOGGER.info(
                "mapping revisit: flipped map yaw in place (%.1f -> %.1f deg)",
                math.degrees(current.theta),
                math.degrees(flipped.theta),
            )
        return self._publish_revisit_check(out)

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
        """Bridge callback: record pause 2D+slice keyframe (ROS thread)."""
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
            lidar.scan_source == LIDAR_SCAN_GET_LASER_SCAN for lidar in cfg.lidars
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
        score = float(result.score)
        ray_mae = float(result.ray_mae_m) if math.isfinite(result.ray_mae_m) else None
        good = score >= cfg.mapping_revisit_min_score and (
            ray_mae is None or ray_mae <= cfg.mapping_revisit_max_ray_mae_m
        )
        return good, score, ray_mae

    async def _mapping_revisit_cycle(
        self,
        *,
        apply_override: Optional[bool] = None,
        yaw_flip: bool = False,
    ) -> Mapping[str, ValueTypes]:
        """One revisit check: tiered scan-to-live-map match, odom shift on drift.

        Tier 1 searches ``mapping_revisit_search_radius_m`` around the current
        pose; tier 2 widens to ``mapping_revisit_wide_radius_m``; tier 3 (full
        map) runs only when enabled and must clear a stricter score gate —
        self-similar offices produce convincing wrong corridors at map scale.

        After the best 2D match, ``choose_yaw_or_flip`` breaks corridor 180°
        ambiguity. Pass ``yaw_flip=True`` to take the opposite heading of that
        auto choice (for when XY is right but facing is wrong).
        """
        cfg = self._cfg
        mgr = self._manager
        if cfg is None or mgr is None:
            return self._publish_revisit_check({"status": "unconfigured"})
        if cfg.mode != MODE_MAPPING:
            return self._publish_revisit_check(
                {"status": "skipped", "reason": "not_mapping"}
            )
        node = mgr.node
        if node is None:
            return self._publish_revisit_check(
                {"status": "skipped", "reason": "bridge_not_started"}
            )

        # Only correct while parked: the odom shift must not land mid-hop, and
        # the fresh lidar read has to match where the robot actually is.
        try:
            bridge_status = node.slam_bridge_status()
        except Exception:  # noqa: BLE001
            bridge_status = {}
        vel = bridge_status.get("odom_velocity") or {}
        moving = (
            math.hypot(float(vel.get("vx", 0.0)), float(vel.get("vy", 0.0)))
            >= cfg.map_when_still_linear_speed_m_s
            or abs(float(vel.get("vtheta", 0.0)))
            >= cfg.map_when_still_yaw_rate_rad_s
        )
        if moving and apply_override is not True:
            return self._publish_revisit_check(
                {"status": "skipped", "reason": "moving"}
            )

        current = mgr.get_pose_in_map()
        if current is None:
            return self._publish_revisit_check(
                {"status": "skipped", "reason": "no_pose_in_map"}
            )

        try:
            occ_map, _ = self._load_active_occupancy_map("live")
        except Exception as exc:  # noqa: BLE001 - live map not ready yet
            return self._publish_revisit_check(
                {"status": "skipped", "reason": f"no_live_map: {exc}"}
            )
        library = self._get_slice_library(cfg)
        scan, band_points = await self._read_merged_scan_and_bands(
            library.bands if library is not None else None
        )

        loop = asyncio.get_running_loop()

        def _match(radius_m: float, yaw_window_deg: float, full_map: bool):
            return global_localize_scan(
                occ_map,
                scan,
                hint=None if full_map else current,
                full_map=full_map,
                search_radius_m=radius_m,
                local_yaw_window_deg=yaw_window_deg,
                coarse_position_step_m=0.6 if full_map else 0.4,
                coarse_yaw_step_deg=18.0 if full_map else 12.0,
            )

        tiers_tried = []
        match_mode = "local"
        result = await loop.run_in_executor(
            None,
            lambda: _match(cfg.mapping_revisit_search_radius_m, 90.0, False),
        )
        good, score, ray_mae = self._revisit_match_quality(result, cfg)
        tiers_tried.append({"tier": "local", "score": round(score, 3)})

        if not good:
            match_mode = "wide"
            wide = await loop.run_in_executor(
                None,
                lambda: _match(cfg.mapping_revisit_wide_radius_m, 180.0, False),
            )
            wide_good, wide_score, wide_ray_mae = self._revisit_match_quality(
                wide, cfg
            )
            tiers_tried.append({"tier": "wide", "score": round(wide_score, 3)})
            if wide_good or wide_score > score:
                result, good, score, ray_mae = wide, wide_good, wide_score, wide_ray_mae

        if not good and cfg.mapping_revisit_full_map_fallback:
            match_mode = "full_map"
            full = await loop.run_in_executor(
                None, lambda: _match(0.0, 360.0, True)
            )
            _, full_score, full_ray_mae = self._revisit_match_quality(full, cfg)
            tiers_tried.append({"tier": "full_map", "score": round(full_score, 3)})
            # Full map needs the stricter gate regardless of ray MAE outcome.
            if full_score >= cfg.mapping_revisit_full_map_min_score and (
                full_ray_mae is None
                or full_ray_mae <= cfg.mapping_revisit_max_ray_mae_m
            ):
                result, good, score, ray_mae = full, True, full_score, full_ray_mae

        kf_info = None
        # Pause keyframes: when occupancy match is weak, try stored stop views
        # (2D + height slices) — covers returning at a different pause angle.
        if (
            not good
            and cfg.mapping_revisit_keyframes
            and self._pause_keyframes is not None
            and len(self._pause_keyframes) > 0
        ):
            store = self._pause_keyframes

            def _kf_match(radius: Optional[float]):
                with self._keyframe_lock:
                    return store.match(
                        scan,
                        band_points,
                        hint=current,
                        search_radius_m=radius,
                    )

            kf = await asyncio.to_thread(
                _kf_match, cfg.mapping_revisit_wide_radius_m
            )
            scope = "near"
            if kf is None or kf.score < cfg.mapping_revisit_keyframe_min_score:
                kf_all = await asyncio.to_thread(_kf_match, None)
                scope = "all"
                if kf_all is not None and (
                    kf is None or kf_all.score > kf.score
                ):
                    kf = kf_all
            if kf is not None:
                tiers_tried.append(
                    {
                        "tier": f"keyframe_{scope}",
                        "score": round(kf.score, 3),
                        "keyframes": kf.keyframes_considered,
                    }
                )
                kf_info = {
                    "score": round(kf.score, 3),
                    "primary_hit_rate": round(kf.primary_hit_rate, 3),
                    "slice_hit_rate": (
                        None
                        if kf.slice_hit_rate is None
                        else round(kf.slice_hit_rate, 3)
                    ),
                    "keyframe_index": kf.keyframe_index,
                    "scope": scope,
                }
                if kf.score >= cfg.mapping_revisit_keyframe_min_score:
                    # Ray-align against the live map at the keyframe pose so
                    # the usual MAE gate still applies.
                    yaw_probe = await asyncio.to_thread(
                        choose_yaw_or_flip,
                        occ_map,
                        scan,
                        kf.pose,
                        reference_theta=current.theta,
                    )
                    result = GlobalLocalizeResult(
                        pose=yaw_probe.pose,
                        score=kf.score,
                        candidates_evaluated=kf.keyframes_considered,
                        scan_points_used=0,
                        in_map_points=0,
                        hit_rate=kf.primary_hit_rate,
                        ray_score=0.0,
                        ray_mae_m=yaw_probe.ray_mae_m,
                    )
                    match_mode = "keyframe"
                    good = (
                        kf.score >= cfg.mapping_revisit_keyframe_min_score
                        and (
                            not math.isfinite(yaw_probe.ray_mae_m)
                            or yaw_probe.ray_mae_m
                            <= cfg.mapping_revisit_max_ray_mae_m
                        )
                    )
                    score = float(kf.score)
                    ray_mae = (
                        float(yaw_probe.ray_mae_m)
                        if math.isfinite(yaw_probe.ray_mae_m)
                        else None
                    )

        # Corridor 180° ambiguity: same XY often scores similarly both ways.
        # Re-score pose vs pose+π and prefer IMU-nearer heading on a near-tie.
        yaw_choice = await asyncio.to_thread(
            choose_yaw_or_flip,
            occ_map,
            scan,
            result.pose,
            reference_theta=current.theta,
        )
        if yaw_flip:
            chosen_pose = yaw_choice.alt_pose
            chosen_score = yaw_choice.alt_score
            chosen_mae = yaw_choice.alt_ray_mae_m
            did_flip = not yaw_choice.flipped
        else:
            chosen_pose = yaw_choice.pose
            chosen_score = yaw_choice.score
            chosen_mae = yaw_choice.ray_mae_m
            did_flip = yaw_choice.flipped
        if match_mode == "keyframe":
            # Keep keyframe hit-rate as the score (occupancy score scale differs).
            result = replace(
                result,
                pose=chosen_pose,
                ray_mae_m=chosen_mae,
            )
            ray_mae = (
                float(chosen_mae) if math.isfinite(chosen_mae) else None
            )
            good = score >= cfg.mapping_revisit_keyframe_min_score and (
                ray_mae is None or ray_mae <= cfg.mapping_revisit_max_ray_mae_m
            )
        else:
            result = replace(
                result,
                pose=chosen_pose,
                score=chosen_score,
                ray_mae_m=chosen_mae,
            )
            good, score, ray_mae = self._revisit_match_quality(result, cfg)

        shift_m = math.hypot(result.pose.x - current.x, result.pose.y - current.y)
        shift_deg = abs(
            math.degrees(_normalize_angle(result.pose.theta - current.theta))
        )
        drifted = (
            shift_m >= cfg.mapping_revisit_min_shift_m
            or shift_deg >= cfg.mapping_revisit_min_shift_deg
        )
        sane = shift_m <= cfg.mapping_revisit_max_shift_m

        out: dict = {
            "status": "ok",
            "match_mode": match_mode,
            "tiers": tiers_tried,
            "score": round(score, 3),
            "ray_mae_m": None if ray_mae is None else round(ray_mae, 3),
            "shift_m": round(shift_m, 3),
            "shift_deg": round(shift_deg, 2),
            "good_match": good,
            "drifted": drifted,
            "pose": {
                "x": result.pose.x,
                "y": result.pose.y,
                "theta": result.pose.theta,
            },
            "yaw_flipped": did_flip,
            "yaw_alt": {
                "theta": yaw_choice.alt_pose.theta,
                "score": round(yaw_choice.alt_score, 3),
                "ray_mae_m": round(yaw_choice.alt_ray_mae_m, 3),
            },
            "corrected": False,
        }
        if kf_info is not None:
            out["keyframe_match"] = kf_info
        if self._pause_keyframes is not None:
            out["keyframes"] = self._pause_keyframes.status()

        should_apply = apply_override is True or (
            apply_override is None and good and drifted and sane
        )
        if not sane and apply_override is not True:
            out["status"] = "rejected_shift"
            LOGGER.warning(
                "mapping revisit: match %.1f m away exceeds "
                "mapping_revisit_max_shift_m=%.1f; not correcting (score=%.2f)",
                shift_m,
                cfg.mapping_revisit_max_shift_m,
                score,
            )
            return self._publish_revisit_check(out)
        if not good and apply_override is not True:
            out["status"] = "low_quality"
            return self._publish_revisit_check(out)

        # Multi-height-slice verification: the occupancy map only encodes the
        # primary z-band, which is self-similar across desk clutter. Before
        # shifting odom, require the proposed pose to also agree with any
        # height bands we have reference geometry for (recorded from earlier
        # trusted pause scans). An explicit apply_override skips the veto.
        if should_apply and drifted and library is not None:
            verdict = await asyncio.to_thread(
                library.verify, band_points, result.pose
            )
            out["slice_verify"] = verdict
            if verdict.get("pass") is False and apply_override is not True:
                out["status"] = "slice_verify_failed"
                LOGGER.warning(
                    "mapping revisit: 2D match at (%.2f, %.2f) rejected — "
                    "%d/%d height slices disagree with stored geometry",
                    result.pose.x,
                    result.pose.y,
                    verdict.get("bands_failed", 0),
                    verdict.get("bands_checked", 0),
                )
                return self._publish_revisit_check(out)

        if should_apply:
            applied = await asyncio.to_thread(
                mgr.apply_map_pose_correction, result.pose
            )
            out["correction"] = applied
            if applied.get("applied"):
                out["status"] = "corrected"
                out["corrected"] = True
                LOGGER.info(
                    "mapping revisit: odom shifted to rejoin map via %s "
                    "(shift=%.2f m, %.1f deg, score=%.2f, ray_mae=%s)",
                    match_mode,
                    shift_m,
                    shift_deg,
                    score,
                    ray_mae,
                )
            else:
                out["status"] = "correction_failed"

        # Grow the slice library from confirmed in-place / corrected poses as a
        # backup to the still-publish keyframe path (no-ops when already dense).
        if library is not None and good:
            record_pose: Optional[conv.Pose2D] = None
            if out["status"] == "corrected":
                record_pose = result.pose
            elif out["status"] == "ok" and not drifted:
                record_pose = current
            if record_pose is not None and any(
                np.asarray(pts).size for pts in band_points
            ):
                await asyncio.to_thread(library.record, band_points, record_pose)
                out["slice_scans_recorded"] = library.scans_recorded
        return self._publish_revisit_check(out)

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

        slam_toolbox tracks pose per-scan but has no automatic global correction;
        on long runs (or after CPU-starved navigation) the map->odom estimate can
        slide with no recovery. Uses a cheap local match each cycle, escalating
        to full-map global_localize (like a manual command) when the local match
        is untrusted or Nav2 is struggling with recoveries.
        """
        while True:
            cfg = self._cfg
            sleep_s = interval_s
            if cfg is not None and self._is_navigation_active():
                sleep_s = max(1.0, float(cfg.periodic_relocalize_nav_interval_s))
            await asyncio.sleep(sleep_s)
            try:
                await self._periodic_relocalize_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - watchdog must survive hiccups
                LOGGER.warning("periodic relocalize cycle failed: %s", exc)
                self._publish_relocalize_check({"status": "error", "error": str(exc)})

    def _publish_relocalize_check(self, result: Mapping[str, ValueTypes]) -> dict:
        self._last_relocalize_check.clear()
        self._last_relocalize_check.update(result)
        return dict(self._last_relocalize_check)

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
                _normalize_angle(
                    float(matched_pose.get("theta", 0.0)) - current.theta
                )
            )
        )
        return shift_m, shift_deg

    async def _periodic_relocalize_cycle(
        self, *, apply_override: Optional[bool] = None
    ) -> Mapping[str, ValueTypes]:
        """Run one drift check; correct pose when a trusted match has moved.

        Escalates to full-map ``global_localize`` when the local match is weak
        (the usual failure mode when manual global_localize is what fixes drift)
        or when Nav2 reports multiple recoveries on the active goal.
        """
        cfg = self._cfg
        mgr = self._manager
        if cfg is None or mgr is None:
            return self._publish_relocalize_check({"status": "unconfigured"})
        if cfg.mode != MODE_LOCALIZING:
            return self._publish_relocalize_check(
                {"status": "skipped", "reason": "not_localizing"}
            )
        startup = self._startup_global_localize_task
        if startup is not None and not startup.done():
            return self._publish_relocalize_check(
                {"status": "skipped", "reason": "startup_localize_running"}
            )

        nav_active = self._is_navigation_active()
        if (
            apply_override is not True
            and nav_active
            and not cfg.periodic_relocalize_during_navigation
        ):
            return self._publish_relocalize_check(
                {"status": "skipped", "reason": "navigation_active"}
            )

        nav_recoveries = 0
        if nav_active:
            try:
                nav_recoveries = int(mgr.nav_status().get("number_of_recoveries", 0))
            except Exception:  # noqa: BLE001
                nav_recoveries = 0

        force_full_map = nav_recoveries >= cfg.periodic_relocalize_nav_recoveries_threshold
        current = mgr.get_pose_in_map()

        base_command: dict = {"command": "global_localize"}
        base_command.update(dict(cfg.periodic_relocalize_options))
        base_command["apply"] = False

        match_mode = "full_map" if force_full_map else "local"
        match_command = dict(base_command)
        if force_full_map:
            match_command["full_map"] = True
        match = await self._global_localize(match_command)

        good_match, score, ray_mae = self._localization_match_quality(match, cfg)
        matched_pose = match.get("pose")
        shift_m, shift_deg = self._pose_shift_from_current(current, matched_pose)

        if (
            not good_match
            and not force_full_map
            and cfg.periodic_relocalize_full_map_on_low_quality
            and apply_override is not True
        ):
            full_command = dict(base_command)
            full_command["full_map"] = True
            full_command["auto_full_map_fallback"] = False
            full_match = await self._global_localize(full_command)
            full_good, full_score, full_ray_mae = self._localization_match_quality(
                full_match, cfg
            )
            if full_good or full_score > score:
                match = full_match
                good_match = full_good
                score = full_score
                ray_mae = full_ray_mae
                matched_pose = full_match.get("pose")
                shift_m, shift_deg = self._pose_shift_from_current(current, matched_pose)
                match_mode = "full_map_after_low_quality"

        drifted = (
            shift_m >= cfg.periodic_relocalize_min_shift_m
            or shift_deg >= cfg.periodic_relocalize_min_shift_deg
        )
        full_map_recovery = match_mode.startswith("full_map")
        # In a recovery situation the robot is (or looks) lost, so mirror a manual
        # global_localize: trust the best full-map match once its score clears the
        # recovery floor, ignoring the stricter ray_mae gate that good_match uses.
        # This is the case that previously left the robot stuck for minutes -- the
        # full-map match was correct (and manual apply fixed it) but ray_mae was
        # above periodic_relocalize_max_ray_mae_m so good_match was False.
        recovery_apply = (
            full_map_recovery
            and score >= cfg.periodic_relocalize_recovery_min_score
        )
        should_apply = apply_override is True or (
            apply_override is None
            and (
                recovery_apply
                or (good_match and drifted)
            )
        )

        result: dict = {
            "status": "ok",
            "match_mode": match_mode,
            "score": score,
            "ray_mae_m": ray_mae,
            "shift_m": None if math.isinf(shift_m) else round(shift_m, 3),
            "shift_deg": None if math.isinf(shift_deg) else round(shift_deg, 2),
            "good_match": good_match,
            "drifted": drifted,
            "recovery_apply": recovery_apply,
            "navigation_active": nav_active,
            "nav_recoveries": nav_recoveries,
            "corrected": False,
        }

        if not should_apply and apply_override is not True and not good_match:
            result["status"] = "low_quality"
            LOGGER.warning(
                "periodic relocalize: no trusted match after %s "
                "(score=%.2f ray_mae=%s recovery_floor=%.2f nav_recoveries=%d); "
                "not correcting",
                match_mode,
                score,
                ray_mae,
                cfg.periodic_relocalize_recovery_min_score,
                nav_recoveries,
            )
            return self._publish_relocalize_check(result)

        if should_apply and isinstance(matched_pose, Mapping):
            await self.do_command(
                {
                    "command": "relocalize",
                    "pose": {
                        "x": float(matched_pose.get("x", 0.0)),
                        "y": float(matched_pose.get("y", 0.0)),
                        "theta": float(matched_pose.get("theta", 0.0)),
                    },
                    "position_variance_m2": 0.25,
                    "yaw_variance_rad2": 0.06853891945200942,
                }
            )
            result["status"] = "corrected"
            result["corrected"] = True
            LOGGER.info(
                "periodic relocalize: corrected via %s (shift=%.2f m, %.1f deg, "
                "score=%.2f, ray_mae=%s, recovery=%s, nav_recoveries=%d)",
                match_mode,
                0.0 if math.isinf(shift_m) else shift_m,
                0.0 if math.isinf(shift_deg) else shift_deg,
                score,
                ray_mae,
                recovery_apply and not good_match,
                nav_recoveries,
            )
        else:
            LOGGER.debug(
                "periodic relocalize: pose ok (%s shift=%.2f m score=%.2f)",
                match_mode,
                0.0 if math.isinf(shift_m) else shift_m,
                score,
            )

        return self._publish_relocalize_check(result)

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

    async def _wait_for_startup_localize_ready(
        self, *, timeout_s: float = 90.0, poll_interval_s: float = 2.0
    ) -> bool:
        """Hold the startup auto-localize until scan matching can actually work.

        A fixed post-reconfigure delay fires while MiR rosbridge sessions are
        still connecting and slam_toolbox is still loading the posegraph; the
        retry attempts then burn out and the robot stays mislocalized until a
        manual global_localize. Ready = slam process up, a merged scan with
        returns readable, and an occupancy map loadable.
        """
        mgr = self._manager
        if mgr is None:
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if mgr.slam_running():
                    await self._read_merged_scan()  # raises when no returns yet
                    self._load_active_occupancy_map("auto")
                    return True
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - not ready yet; keep polling
                pass
            await asyncio.sleep(poll_interval_s)
        if timeout_s > 0.0:
            LOGGER.warning(
                "startup global_localize readiness wait timed out after %.0fs; "
                "attempting anyway",
                timeout_s,
            )
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
        if self._is_navigation_active():
            LOGGER.info("startup global_localize skipped: navigation already active")
            return
        await self._wait_for_startup_localize_ready(timeout_s=readiness_timeout_s)
        # Extra settle time after readiness so slam_toolbox has processed a few
        # scans against the loaded posegraph before we sample one for matching.
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
        assert self._cfg is not None
        # Built-in path: read odometry from the movement sensor's get_readings()
        # (odom_reader=None). The external-SLAM model reuses this same builder
        # with a typed MovementSensor reader injected.
        return build_io_provider(
            base=self._base,
            cameras=self._cameras,
            cfg=self._cfg,
            movement_sensor=self._movement_sensor,
            heading_sensor=self._heading_sensor,
            skip_get_laser_scan=self._skip_get_laser_scan,
            odom_reader=None,
            logger=LOGGER,
        )

    async def _probe_sensors(self) -> dict:
        """One-shot lidar + odom read for get_status (does not affect /scan)."""
        assert self._cfg is not None
        io = self._build_io()
        lidars: list[dict] = []
        for lidar in self._cfg.lidars:
            entry: dict = {
                "name": lidar.name,
                "scan_source": lidar.scan_source,
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
                    # map stays frozen (slam_toolbox only adds scans after travel).
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

    def _start_mode(self, mode: str) -> None:
        assert self._manager and self._map_store and self._cfg
        handle = self._map_store.active_handle()
        stem = handle.serialization_stem if handle else None
        self._manager.start_slam(stem, mode)
        self._cfg.mode = mode
        # Slice reference geometry is pose-graph-relative; a (re)started SLAM
        # session invalidates it.
        self._reset_slice_library()

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

        if cmd == "get_status":
            probe_sensors = command.get("probe_sensors", True)
            if probe_sensors is not None:
                probe_sensors = bool(probe_sensors)

            def _status():
                return mgr.slam_diagnostics()

            status = await asyncio.to_thread(_status)
            status["mode"] = self._cfg.mode if self._cfg else None
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
                    }
                    for lidar in self._cfg.lidars
                ]
            status["localization_check"] = dict(self._last_relocalize_check)
            status["revisit_check"] = dict(self._last_revisit_check)
            if self._pause_keyframes is not None:
                status["pause_keyframes"] = self._pause_keyframes.status()
            if probe_sensors and self._cfg is not None:
                status["sensor_probe"] = await self._probe_sensors()
            return status

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

        if cmd == "check_localization":
            # Run one drift-watchdog cycle on demand. ``apply`` forces or
            # suppresses the correction; omit it to use the drift thresholds.
            apply_override = command.get("apply")
            return await self._periodic_relocalize_cycle(
                apply_override=(
                    None if apply_override is None else bool(apply_override)
                )
            )

        if cmd == "get_localization_check":
            return dict(self._last_relocalize_check)

        if cmd == "revisit_check":
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

        if cmd == "get_revisit_check":
            return dict(self._last_revisit_check)

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
        io = self._build_io()
        scans: List[conv.LaserScan2D] = []
        band_points: List[np.ndarray] = (
            [np.empty((0, 2)) for _ in bands] if bands else []
        )
        for lidar in self._cfg.lidars:
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
            range_max=self._cfg.slam_toolbox.max_laser_range,
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
        self._cancel_periodic_relocalize_task()
        self._cancel_mapping_revisit_task()
        unregister_slam(self.name)
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None


Registry.register_resource_creator(
    SLAM.API,
    RosSlam.MODEL,
    ResourceCreatorRegistration(RosSlam.new, RosSlam.validate_config),
)

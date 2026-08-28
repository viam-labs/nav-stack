"""Sync lidar + odom reads for the builtin SLAM engine (no ROS)."""
from __future__ import annotations

import asyncio
import math
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from viam.utils import struct_to_dict

from ..config import (
    LIDAR_SCAN_GET_LASER_SCAN,
    LIDAR_SCAN_POINT_CLOUD,
    LidarConfig,
    SlamConfig,
)
from ..ros import conversions as conv
from ..ros import imushm
from ..ros import pcshm


def _get_laser_scan_not_implemented(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    msg = str(exc).lower()
    return (
        "not implemented" in msg
        or "does not support get_laser_scan" in msg
        or "docommand not implemented" in msg
        or "did not return get_laser_scan" in msg
    )


def _shm_error_is_stale(detail: object) -> bool:
    return "frame too old" in str(detail).lower()


class BuiltinSensors:
    """Thread-safe-ish sensor facade used from the SLAM background loop."""

    def __init__(
        self,
        *,
        cfg: SlamConfig,
        cameras: Mapping[str, object],
        movement_sensor,
        heading_sensor,
        shm_lidar,
        loop: asyncio.AbstractEventLoop,
        logger=None,
        skip_get_laser_scan: Optional[set] = None,
        scan_max_age_s: float = 2.0,
    ):
        self._cfg = cfg
        self._cameras = dict(cameras or {})
        self._movement = movement_sensor
        self._heading = heading_sensor
        self._shm = shm_lidar
        self._loop = loop
        self._logger = logger
        self._skip_get_laser_scan = (
            skip_get_laser_scan if skip_get_laser_scan is not None else set()
        )
        self._scan_max_age_s = float(scan_max_age_s)
        self._scan_cache: Optional[conv.LaserScan2D] = None
        self._scan_cache_at = 0.0
        self._imu_shm: Optional[imushm.Reader] = None
        if cfg.imu_shm_name:
            try:
                self._imu_shm = imushm.Reader(
                    cfg.imu_shm_name,
                    region_size=int(cfg.imu_shm_region_size),
                )
                self._log(f"builtin SLAM reading IMU shm {cfg.imu_shm_name}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"IMU shm open failed ({cfg.imu_shm_name}): {exc}")
                self._imu_shm = None

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger(msg)
            except Exception:  # noqa: BLE001
                pass

    def _run(self, coro, timeout: float = 2.0):
        if self._loop.is_closed():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("event loop is closed")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            fut.cancel()
            raise

    def get_scan(self, max_age_s: float = 2.0) -> Optional[conv.LaserScan2D]:
        now = time.monotonic()
        if (
            self._scan_cache is not None
            and now - self._scan_cache_at <= max_age_s
        ):
            return self._scan_cache
        lidars: Sequence[LidarConfig] = self._cfg.lidars
        if not lidars:
            return self._scan_cache
        scans = []
        for lidar in lidars:
            scan = self._read_lidar_scan_sync(lidar, max_age_s=max_age_s)
            if scan is not None:
                scans.append(scan)
        if not scans:
            return self._scan_cache
        merged = (
            scans[0]
            if len(scans) == 1
            else conv.merge_scans(scans, self._cfg.scan_bins)
        )
        self._scan_cache = merged
        self._scan_cache_at = now
        return merged

    def _pcd_to_scan(self, raw: bytes, lidar: LidarConfig) -> conv.LaserScan2D:
        pts = conv.parse_pcd(raw)
        if not lidar.points_in_base_link:
            pts = conv.transform_lidar_mount_to_base_link(
                pts,
                x=lidar.x,
                y=lidar.y,
                z=lidar.z,
                theta=lidar.theta,
                pitch=lidar.pitch,
                roll=lidar.roll,
            )
        pts = conv.filter_points_by_z(pts, lidar.z_min, lidar.z_max)
        return conv.points_to_scan(
            pts,
            angle_min=-math.pi,
            angle_max=math.pi,
            num_bins=self._cfg.scan_bins,
            range_min=lidar.min_range,
            range_max=lidar.max_range,
        )

    def _try_shm_scan(
        self, lidar: LidarConfig, *, max_age_s: float
    ) -> Optional[conv.LaserScan2D]:
        if not lidar.shm_name or self._shm is None:
            return None
        age_limit = self._scan_max_age_s if self._scan_max_age_s > 0 else max_age_s
        got = self._shm.try_read(
            lidar.shm_name,
            lidar.shm_region_size,
            max_age_s=age_limit if age_limit > 0 else None,
        )
        if got is None:
            stats = self._shm.status().get(
                pcshm.normalize_name(lidar.shm_name), {}
            )
            detail = stats.get("last_error") or "no complete frame"
            if lidar.shm_required or _shm_error_is_stale(detail):
                self._log(
                    f"lidar {lidar.name} shm {lidar.shm_name!r} unavailable: {detail}"
                )
                return None
            self._shm.note_fallback(lidar.shm_name)
            return None
        raw, _age = got
        return self._pcd_to_scan(raw, lidar)

    def _read_lidar_scan_sync(
        self, lidar: LidarConfig, *, max_age_s: float
    ) -> Optional[conv.LaserScan2D]:
        shm_scan = self._try_shm_scan(lidar, max_age_s=max_age_s)
        if shm_scan is not None:
            return shm_scan
        if lidar.shm_name and lidar.shm_required:
            return None
        cam = self._cameras.get(lidar.name)
        if cam is None:
            return None
        try:
            return self._run(
                self._read_lidar_scan_grpc(cam, lidar),
                timeout=1.0,
            )
        except Exception:  # noqa: BLE001
            return None

    async def _read_lidar_scan_grpc(
        self, cam, lidar: LidarConfig
    ) -> Optional[conv.LaserScan2D]:
        timeout = 1.0
        scan_source = lidar.scan_source
        name = lidar.name

        async def _from_point_cloud() -> Optional[conv.LaserScan2D]:
            data = await cam.get_point_cloud(timeout=timeout)
            raw = data[0] if isinstance(data, tuple) else data
            return self._pcd_to_scan(raw, lidar)

        async def _from_get_laser_scan() -> Optional[conv.LaserScan2D]:
            raw_payload = await cam.do_command({"command": "get_laser_scan"})
            payload = (
                struct_to_dict(raw_payload)
                if not isinstance(raw_payload, dict)
                else raw_payload
            )
            mir_pts = conv.points_from_mir_laser_scan_payload(payload)
            if mir_pts.sensor_scan is not None and conv.scan_has_returns(
                mir_pts.sensor_scan
            ):
                return mir_pts.sensor_scan
            if mir_pts.base_link.size > 0:
                return conv.points_to_scan(
                    mir_pts.base_link,
                    angle_min=-math.pi,
                    angle_max=math.pi,
                    num_bins=self._cfg.scan_bins,
                    range_min=lidar.min_range,
                    range_max=lidar.max_range,
                )
            return None

        if scan_source == LIDAR_SCAN_POINT_CLOUD or name in self._skip_get_laser_scan:
            try:
                return await _from_point_cloud()
            except Exception:  # noqa: BLE001
                return None

        if scan_source == LIDAR_SCAN_GET_LASER_SCAN:
            try:
                return await _from_get_laser_scan()
            except Exception:  # noqa: BLE001
                return None

        # auto
        try:
            return await _from_get_laser_scan()
        except Exception as exc:  # noqa: BLE001
            if _get_laser_scan_not_implemented(exc):
                self._skip_get_laser_scan.add(name)
            else:
                self._log(f"lidar {name} get_laser_scan failed: {exc}")
        try:
            return await _from_point_cloud()
        except Exception:  # noqa: BLE001
            return None

    def get_odom(self) -> Optional[conv.OdomReading]:
        sample = self._odom_from_imu_shm()
        if sample is not None:
            return sample
        if self._movement is None:
            return None
        try:
            return self._run(self._read_odom(), timeout=1.0)
        except Exception:  # noqa: BLE001
            return None

    def _odom_from_imu_shm(self) -> Optional[conv.OdomReading]:
        reader = self._imu_shm
        if reader is None:
            return None
        try:
            frame = reader.read_latest(max_age_s=float(self._cfg.imu_shm_max_age_s))
        except Exception:  # noqa: BLE001
            return None
        if frame is None:
            return None
        cfg = self._cfg
        sample = conv.parse_odom_from_readings(
            {
                "angular_velocity": {"x": frame.gx, "y": frame.gy, "z": frame.gz},
                "linear_acceleration": {"x": frame.ax, "y": frame.ay, "z": frame.az},
                "orientation": {
                    "roll": frame.roll,
                    "pitch": frame.pitch,
                    "yaw": frame.yaw,
                },
            }
        )
        if cfg.movement_sensor_upside_down:
            sample = conv.apply_sensor_upside_down(sample)
        if cfg.movement_sensor_yaw_deg:
            sample = conv.apply_sensor_mount_yaw(
                sample, math.radians(cfg.movement_sensor_yaw_deg)
            )
        return sample

    async def _read_odom(self) -> conv.OdomReading:
        cfg = self._cfg
        sample = conv.parse_odom_from_readings(await self._movement.get_readings())
        if cfg.movement_sensor_upside_down:
            sample = conv.apply_sensor_upside_down(sample)
        if cfg.movement_sensor_yaw_deg:
            sample = conv.apply_sensor_mount_yaw(
                sample, math.radians(cfg.movement_sensor_yaw_deg)
            )
        if self._heading is not None:
            heading_readings = await self._heading.get_readings()
            heading = conv.parse_heading_sensor_readings(heading_readings)
            if heading is not None:
                if cfg.heading_sensor_invert:
                    heading = conv.normalize_angle(-heading)
                if cfg.heading_sensor_yaw_deg:
                    heading = conv.normalize_angle(
                        heading - math.radians(cfg.heading_sensor_yaw_deg)
                    )
                sample = conv.merge_odom_heading(sample, heading)
        return sample

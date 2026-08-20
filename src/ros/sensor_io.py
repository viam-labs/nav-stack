"""Shared builder for the bridge's :class:`~.bridge.IOProvider`.

Both navigation-capable models feed the ROS bridge the same four Viam-backed
callables — lidar point reads, odometry reads, drive, and stop. The only
difference is *how odometry is read*:

* built-in SLAM parses the movement sensor's ``get_readings()`` dict, and
* the external-SLAM path uses the portable typed MovementSensor API
  (:class:`~.odom_source.TypedMovementSensorOdom`).

So this builder takes an optional ``odom_reader``: when provided it supplies the
raw sample; otherwise the ``get_readings`` parser is used. Either way the same
mount-yaw / upside-down / heading-sensor corrections are applied, so odometry
behaves identically regardless of the read path.
"""
from __future__ import annotations

import math
from typing import Optional

from viam.proto.common import Vector3
from viam.utils import struct_to_dict

from ..config import (
    LIDAR_SCAN_GET_LASER_SCAN,
    LIDAR_SCAN_POINT_CLOUD,
    ros_cmd_vel_to_viam_linear_mm_s,
)
from . import conversions as conv
from .bridge import IOProvider
from . import pcshm
from .shm_lidar import ShmPointCloudClient


def get_laser_scan_not_implemented(exc: BaseException) -> bool:
    # Prefer the exception type: RPLidarShm raises NotImplementedError with a
    # message that historically said "does not support" (no "not implemented").
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
    """True when shm miss is a rejected old frame (must not gRPC-fallback)."""
    return "frame too old" in str(detail).lower()


def is_mir_laser_scan_payload(payload: Mapping) -> bool:
    """True when ``payload`` looks like mir-base ``get_laser_scan`` output."""
    scans = payload.get("scans")
    if scans is None:
        return "message" in payload or "topic" in payload
    return isinstance(scans, list)


def build_io_provider(
    *,
    base,
    cameras,
    cfg,
    movement_sensor=None,
    heading_sensor=None,
    skip_get_laser_scan: Optional[set] = None,
    odom_reader=None,
    logger,
    record_cmd_vel=None,
    shm_lidar=None,
) -> IOProvider:
    """Build an ``IOProvider`` from resolved Viam components + a SlamConfig.

    ``skip_get_laser_scan`` is a mutable set used to remember lidars that don't
    implement ``get_laser_scan`` (so we stop probing them). ``odom_reader``, when
    given, is any object with an ``async read() -> OdomReading`` — the typed
    MovementSensor reader for the external path. ``record_cmd_vel``, when given,
    is called as ``record_cmd_vel(vx, vy, vtheta, source=...)`` before each
    base drive/stop so ``get_status`` can show the last SetVelocity mapping.
    ``shm_lidar`` is a shared :class:`~.shm_lidar.ShmPointCloudClient` used when
    a lidar has ``shm_name`` set.
    """
    if skip_get_laser_scan is None:
        skip_get_laser_scan = set()

    async def read_lidar_points(name: str) -> conv.LidarPoints:
        timeout = max(float(cfg.sensor_read_timeout_s), 1.0)
        cam = cameras[name]
        lidar_cfg = next((lidar for lidar in cfg.lidars if lidar.name == name), None)
        scan_source = lidar_cfg.scan_source if lidar_cfg is not None else "auto"

        async def _points_from_pcd(
            raw: bytes, *, age_s: Optional[float] = None
        ) -> conv.LidarPoints:
            pts = conv.parse_pcd(raw)
            if lidar_cfg is not None and not lidar_cfg.points_in_base_link:
                base_pts = conv.transform_lidar_mount_to_base_link(
                    pts,
                    x=lidar_cfg.x,
                    y=lidar_cfg.y,
                    z=lidar_cfg.z,
                    theta=lidar_cfg.theta,
                    pitch=lidar_cfg.pitch,
                    roll=lidar_cfg.roll,
                )
            else:
                base_pts = pts
            return conv.LidarPoints(sensor=pts, base_link=base_pts, age_s=age_s)

        async def _read_point_cloud() -> conv.LidarPoints:
            shm_name = lidar_cfg.shm_name if lidar_cfg is not None else None
            if shm_name:
                client = shm_lidar
                if client is None:
                    raise RuntimeError(
                        f"lidar {name} has shm_name={shm_name!r} but no shm client"
                    )
                max_age = float(cfg.scan_max_age_s) if cfg.scan_max_age_s > 0.0 else None
                got = client.try_read(
                    shm_name,
                    lidar_cfg.shm_region_size,
                    max_age_s=max_age,
                )
                if got is not None:
                    raw, age_s = got
                    return await _points_from_pcd(raw, age_s=age_s)
                stats = client.status().get(pcshm.normalize_name(shm_name), {})
                detail = stats.get("last_error") or "no complete frame"
                # Stale frames must not fall back to get_point_cloud: that path
                # returns the same cached PCD with age_s unset, so the bridge
                # scan_max_age_s gate never fires and a dead writer freezes SLAM.
                if lidar_cfg.shm_required or _shm_error_is_stale(detail):
                    raise RuntimeError(
                        f"lidar {name} shm {shm_name!r} unavailable: {detail}"
                    )
                client.note_fallback(shm_name)
            data = await cam.get_point_cloud(timeout=timeout)
            raw = data[0] if isinstance(data, tuple) else data
            return await _points_from_pcd(raw)

        if scan_source == LIDAR_SCAN_POINT_CLOUD or name in skip_get_laser_scan:
            return await _read_point_cloud()

        if scan_source == LIDAR_SCAN_GET_LASER_SCAN:
            raw_payload = await cam.do_command({"command": "get_laser_scan"})
            payload = (
                struct_to_dict(raw_payload)
                if not isinstance(raw_payload, dict)
                else raw_payload
            )
            if not is_mir_laser_scan_payload(payload):
                raise NotImplementedError(
                    f"lidar {name} do_command did not return get_laser_scan data"
                )
            mir_pts = conv.points_from_mir_laser_scan_payload(payload)
            if mir_pts.base_link.size > 0 or (
                mir_pts.sensor_scan is not None
                and conv.scan_has_returns(mir_pts.sensor_scan)
            ):
                return mir_pts
            raise RuntimeError(f"lidar {name} get_laser_scan returned no valid ranges")

        # auto: prefer mir-base get_laser_scan, fall back to point cloud.
        try:
            raw_payload = await cam.do_command({"command": "get_laser_scan"})
            payload = (
                struct_to_dict(raw_payload)
                if not isinstance(raw_payload, dict)
                else raw_payload
            )
            if not is_mir_laser_scan_payload(payload):
                raise NotImplementedError(
                    f"lidar {name} do_command did not return get_laser_scan data"
                )
            mir_pts = conv.points_from_mir_laser_scan_payload(payload)
            if mir_pts.base_link.size > 0 or (
                mir_pts.sensor_scan is not None
                and conv.scan_has_returns(mir_pts.sensor_scan)
            ):
                return mir_pts
            logger.warning(
                "lidar %s get_laser_scan returned no valid ranges; using point cloud",
                name,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to point cloud
            if get_laser_scan_not_implemented(exc):
                if name not in skip_get_laser_scan:
                    skip_get_laser_scan.add(name)
                    logger.info(
                        "lidar %s has no get_laser_scan; using get_point_cloud", name
                    )
            else:
                logger.warning(
                    "lidar %s get_laser_scan failed: %s; using point cloud", name, exc
                )
        return await _read_point_cloud()

    async def read_odometry() -> conv.OdomReading:
        if odom_reader is not None:
            sample = await odom_reader.read()
        elif movement_sensor is not None:
            sample = conv.parse_odom_from_readings(await movement_sensor.get_readings())
        else:
            return conv.OdomReading(0.0, 0.0, 0.0)
        if cfg.movement_sensor_upside_down:
            sample = conv.apply_sensor_upside_down(sample)
        if cfg.movement_sensor_yaw_deg:
            sample = conv.apply_sensor_mount_yaw(
                sample, math.radians(cfg.movement_sensor_yaw_deg)
            )
        if heading_sensor is not None:
            heading_readings = await heading_sensor.get_readings()
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

    async def drive_base(
        vx: float,
        vy: float,
        vtheta: float,
        *,
        record_source: Optional[str] = "nav2",
    ):
        # ``record_source=None`` skips history (caller already recorded the
        # pre-snap Nav2 command — see BridgeNode._on_drive_timer).
        if record_cmd_vel is not None and record_source is not None:
            record_cmd_vel(vx, vy, vtheta, source=record_source)
        lx_mm, ly_mm = ros_cmd_vel_to_viam_linear_mm_s(
            vx, vy, cfg.base_velocity_convention
        )
        await base.set_velocity(
            linear=Vector3(x=lx_mm, y=ly_mm, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=math.degrees(vtheta)),
        )

    async def stop_base():
        # MiR base.stop() also calls REST stop_immediately (PAUSE), which drops
        # Manualcontrol and kills the rosbridge /cmd_vel session. Nav2 only needs
        # zeros.
        if record_cmd_vel is not None:
            record_cmd_vel(0.0, 0.0, 0.0, source="stop")
        await base.set_velocity(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )

    return IOProvider(read_lidar_points, read_odometry, drive_base, stop_base)

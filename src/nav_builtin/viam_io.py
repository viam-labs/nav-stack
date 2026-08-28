"""Viam-backed WorldIO: SLAM + lidars + base with no ROS/rclpy."""
from __future__ import annotations

import asyncio
import base64
import math
import struct
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from viam.proto.common import Vector3
from viam.utils import struct_to_dict

from ..config import (
    LIDAR_SCAN_GET_LASER_SCAN,
    LIDAR_SCAN_POINT_CLOUD,
    LidarConfig,
    ros_cmd_vel_to_viam_linear_mm_s,
)
from ..ros import conversions as conv
from ..ros.external_slam import parse_get_grid, slam_pose_to_pose2d
from ..ros import pcshm
from .viz_store import NavVizStore
from .world_io import WorldIO


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


def get_grid_response_to_map(resp: Mapping) -> Optional[dict]:
    """Convert a ``get_grid`` DoCommand response to a bridge-style map dict."""
    parsed = parse_get_grid(dict(resp))
    if parsed is None:
        return None
    rows, cols, cell_size, x_min, y_min, cells = parsed
    grid = np.asarray(cells, dtype=np.int16).reshape(rows, cols)
    return {
        "grid": grid,
        "resolution": float(cell_size),
        "origin_x": float(x_min),
        "origin_y": float(y_min),
    }


def bridge_map_to_get_grid(map_data: dict) -> dict:
    """Encode a bridge-style map dict as a ``get_grid`` DoCommand payload."""
    grid = np.asarray(map_data["grid"], dtype=np.int16)
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    flat = np.clip(grid.reshape(-1), -1, 100).astype(np.int8)
    raw = struct.pack(f"{flat.size}b", *flat.tolist())
    return {
        "rows": rows,
        "cols": cols,
        "cellSize": float(map_data["resolution"]),
        "xMin": float(map_data["origin_x"]),
        "yMin": float(map_data["origin_y"]),
        "data": base64.b64encode(raw).decode("ascii"),
    }


class ViamWorldIO:
    """WorldIO over Viam SLAM / Camera / Base resources (no ROS topics)."""

    def __init__(
        self,
        *,
        slam,
        base,
        loop: asyncio.AbstractEventLoop,
        cameras: Optional[Mapping[str, object]] = None,
        lidars: Optional[Sequence[LidarConfig]] = None,
        base_velocity_convention: str = "viam",
        viz: Optional[NavVizStore] = None,
        shm_lidar=None,
        scan_max_age_s: float = 2.0,
        drive_timeout_s: float = 2.0,
        map_cache_s: float = 1.0,
        scan_bins: int = 360,
        logger=None,
    ):
        self._slam = slam
        self._base = base
        self._loop = loop
        self._cameras = dict(cameras or {})
        self._lidars = list(lidars or [])
        self._convention = base_velocity_convention
        self._viz = viz
        self._shm_lidar = shm_lidar
        self._scan_max_age_s = float(scan_max_age_s)
        self._drive_timeout_s = drive_timeout_s
        self._map_cache_s = map_cache_s
        self._scan_bins = scan_bins
        self._logger = logger
        self._skip_get_laser_scan: set[str] = set()
        self._map_cache: Optional[dict] = None
        self._map_cache_at = 0.0
        self._scan_cache: Optional[conv.LaserScan2D] = None
        self._scan_cache_at = 0.0

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
        except TimeoutError as exc:
            fut.cancel()
            # concurrent.futures.TimeoutError has empty str(); make it actionable.
            raise TimeoutError(
                f"Viam IO timed out after {timeout:.1f}s"
            ) from exc

    def get_map(self) -> Optional[dict]:
        now = time.monotonic()
        if (
            self._map_cache is not None
            and now - self._map_cache_at < self._map_cache_s
        ):
            return self._map_cache
        try:
            resp = self._run(
                self._slam.do_command({"command": "get_grid"}),
                timeout=5.0,
            )
        except Exception:  # noqa: BLE001
            return self._map_cache
        if not isinstance(resp, Mapping):
            return self._map_cache
        payload: Mapping = resp
        if "rows" not in resp and isinstance(resp.get("grid"), Mapping):
            payload = resp["grid"]  # type: ignore[assignment]
        parsed = get_grid_response_to_map(payload)
        if parsed is None and "rows" in resp:
            parsed = get_grid_response_to_map(resp)
        if parsed is not None:
            self._map_cache = parsed
            self._map_cache_at = now
            if self._viz is not None:
                self._viz.set_map(parsed)
        return parsed

    def get_pose(self) -> Optional[conv.Pose2D]:
        try:
            pose = self._run(self._slam.get_position(), timeout=2.0)
        except Exception:  # noqa: BLE001
            return None
        if pose is None:
            return None
        try:
            p2 = slam_pose_to_pose2d(pose)
        except Exception:  # noqa: BLE001
            return None
        if self._viz is not None:
            self._viz.set_pose(p2)
        return p2

    def get_scan(self, max_age_s: float = 2.0) -> Optional[conv.LaserScan2D]:
        now = time.monotonic()
        if (
            self._scan_cache is not None
            and now - self._scan_cache_at <= max_age_s
        ):
            return self._scan_cache
        if not self._lidars:
            return self._scan_cache
        scans = []
        for lidar in self._lidars:
            scan = self._read_lidar_scan_sync(lidar, max_age_s=max_age_s)
            if scan is not None:
                scans.append(scan)
        if not scans:
            return self._scan_cache
        merged = (
            scans[0]
            if len(scans) == 1
            else conv.merge_scans(scans, self._scan_bins)
        )
        self._scan_cache = merged
        self._scan_cache_at = now
        return merged

    def _pcd_to_scan(
        self, raw: bytes, lidar: LidarConfig
    ) -> conv.LaserScan2D:
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
            num_bins=self._scan_bins,
            range_min=lidar.min_range,
            range_max=lidar.max_range,
        )

    def _try_shm_scan(
        self, lidar: LidarConfig, *, max_age_s: float
    ) -> Optional[conv.LaserScan2D]:
        """Sync shm read — no event-loop hop (control-loop hot path)."""
        if not lidar.shm_name or self._shm_lidar is None:
            return None
        age_limit = self._scan_max_age_s if self._scan_max_age_s > 0 else max_age_s
        got = self._shm_lidar.try_read(
            lidar.shm_name,
            lidar.shm_region_size,
            max_age_s=age_limit if age_limit > 0 else None,
        )
        if got is None:
            stats = self._shm_lidar.status().get(
                pcshm.normalize_name(lidar.shm_name), {}
            )
            detail = stats.get("last_error") or "no complete frame"
            if lidar.shm_required or _shm_error_is_stale(detail):
                self._log(
                    f"lidar {lidar.name} shm {lidar.shm_name!r} unavailable: {detail}"
                )
                return None
            self._shm_lidar.note_fallback(lidar.shm_name)
            return None
        raw, _age = got
        return self._pcd_to_scan(raw, lidar)

    def _read_lidar_scan_sync(
        self, lidar: LidarConfig, *, max_age_s: float
    ) -> Optional[conv.LaserScan2D]:
        # Prefer POSIX shm (memcpy) so the 10 Hz control loop never blocks on
        # gRPC GetPointCloud — that lag was causing no_scan spin / circles.
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
                    num_bins=self._scan_bins,
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

        try:
            scan = await _from_get_laser_scan()
            if scan is not None:
                return scan
        except Exception as exc:  # noqa: BLE001
            if _get_laser_scan_not_implemented(exc):
                self._skip_get_laser_scan.add(name)
                self._log(f"lidar {name} has no get_laser_scan; using point cloud")
            else:
                self._log(f"lidar {name} get_laser_scan failed: {exc}")
        try:
            return await _from_point_cloud()
        except Exception:  # noqa: BLE001
            return None

    def set_velocity(self, vx: float, vy: float, vtheta: float) -> None:
        vx, vy, vtheta = _sanitize_base_cmd(vx, vy, vtheta)
        lx_mm, ly_mm = ros_cmd_vel_to_viam_linear_mm_s(vx, vy, self._convention)
        try:
            self._run(
                self._base.set_velocity(
                    linear=Vector3(x=lx_mm, y=ly_mm, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=math.degrees(vtheta)),
                ),
                timeout=self._drive_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            # Wheeled bases reject non-zero but tiny wheel RPM ("nearly 0").
            # Snap to a clean stop or pure spin and retry once.
            if not _is_near_zero_rpm_error(exc):
                raise
            if abs(vtheta) >= 0.15:
                self._run(
                    self._base.set_velocity(
                        linear=Vector3(x=0.0, y=0.0, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=math.degrees(vtheta)),
                    ),
                    timeout=self._drive_timeout_s,
                )
            else:
                self.stop()

    def stop(self) -> None:
        try:
            self._run(
                self._base.set_velocity(
                    linear=Vector3(x=0.0, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=0.0),
                ),
                timeout=self._drive_timeout_s,
            )
        except Exception:  # noqa: BLE001
            pass

    def set_viz_plan(
        self,
        path_xy: tuple,
        goal: Optional[tuple] = None,
    ) -> None:
        if self._viz is None:
            return
        self._viz.set_plan(path_xy, goal)

    def set_viz_costmap(self, costmap: dict) -> None:
        if self._viz is None:
            return
        self._viz.set_costmap(costmap)


def _sanitize_base_cmd(
    vx: float, vy: float, vtheta: float
) -> tuple[float, float, float]:
    """Snap sub-deadband speeds to zero so Viam wheeled bases don't reject RPM.

    Diff-drive with tiny ``vx`` + large ``vtheta`` also drives one wheel through
    ~0 RPM — prefer pure spin in that case.
    """
    lin_eps = 0.05  # m/s
    ang_eps = 0.08  # rad/s
    if abs(vx) < lin_eps:
        vx = 0.0
    if abs(vy) < lin_eps:
        vy = 0.0
    if abs(vtheta) < ang_eps:
        vtheta = 0.0
    if vx != 0.0 and abs(vx) < 0.12 and abs(vtheta) > 0.25:
        vx = 0.0
    return vx, vy, vtheta


def _is_near_zero_rpm_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "nearly 0" in msg or "rpm that is nearly" in msg


def _check_protocol() -> None:
    _: WorldIO
    del _

"""Viam-backed WorldIO: SLAM + lidars + base."""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import math
import struct
import time
from typing import Callable, Mapping, Optional, Sequence, Set

import numpy as np
from viam.proto.common import Vector3
from viam.utils import struct_to_dict

from ..config import (
    LIDAR_SCAN_GET_LASER_SCAN,
    LIDAR_SCAN_POINT_CLOUD,
    LidarConfig,
    body_twist_to_viam_set_velocity,
)
from ..geom import conversions as conv
from ..slam_client import parse_get_grid, slam_pose_to_pose2d
from ..shm import pcshm
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


def map_dict_to_get_grid(map_data: dict) -> dict:
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
    """WorldIO over Viam SLAM / Camera / Base resources."""

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
        obstacles_only_period_s: float = 0.20,
        drive_timeout_s: float = 5.0,
        map_cache_s: float = 1.0,
        scan_bins: int = 360,
        logger=None,
        pose_provider: Optional[Callable[[], Optional[conv.Pose2D]]] = None,
        map_provider: Optional[Callable[[], Optional[dict]]] = None,
        scan_provider: Optional[Callable[[float], Optional[conv.LaserScan2D]]] = None,
        localization_hold_provider: Optional[Callable[[], Optional[dict]]] = None,
    ):
        self._slam = slam
        self._base = base
        self._base_name = str(
            getattr(base, "name", None) or getattr(base, "_name", None) or base
        )
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
        # Prefer in-process SLAM engine reads over gRPC GetPosition/get_grid.
        # Same-module dependency RPCs share the module event loop with the nav
        # worker's run_coroutine_threadsafe waits and can return the SLAM API's
        # origin placeholder (0,0,0) or stall — which freezes bearing_error.
        self._pose_provider = pose_provider
        self._map_provider = map_provider
        self._scan_provider = scan_provider
        self._localization_hold_provider = localization_hold_provider
        self._skip_get_laser_scan: set[str] = set()
        self._map_cache: Optional[dict] = None
        self._map_cache_at = 0.0
        self._scan_cache: Optional[conv.LaserScan2D] = None
        self._scan_cache_at = 0.0
        self._scan_cache_pose: Optional[conv.Pose2D] = None
        # Obstacles-only depth cams: never await GetPointCloud on the nav tick.
        # A fire-and-forget refresh fills this cache; the control loop only reads it
        # so RealSense PCD cannot starve SetVelocity on the shared module loop.
        self._per_lidar_scan: dict[str, tuple[conv.LaserScan2D, float]] = {}
        # Depth is async + slow; keep it fresh enough that motion compensation works.
        # Configurable via NavConfig.obstacles_only_rate_hz (default 5 Hz).
        self._obstacles_only_period_s = max(0.0, float(obstacles_only_period_s))
        # Beyond this pose shift, cached depth is dropped (avoids phantom obstacles).
        self._obstacles_max_shift_m = 0.30
        self._obstacles_max_shift_rad = math.radians(20.0)
        self._obstacles_refresh_inflight: Set[str] = set()
        self._last_drive: Optional[dict] = None
        self._pose_source: str = "none"
        # In-flight Base.SetVelocity future. Never cancel mid-RPC: control ticks
        # are faster than a busy module loop, and cancelling made every command
        # land as ``superseded`` while the robot sat still → stall timeout.
        # Latest cmd is coalesced into ``_drive_followup`` and sent after the
        # current RPC finishes. Also never cancel just because the waiter timed
        # out (soft-loc / global_localize holding the loop).
        self._pending_drive_fut: Optional[concurrent.futures.Future] = None
        self._drive_followup: Optional[tuple] = None  # (make_coro, intent)
        # Last twist we accepted (scheduled or skipped-as-duplicate). Used to
        # skip redundant SetVelocity when the control tick is faster than the
        # base RPC — identical cmds need not hit the module loop.
        self._last_desired_twist: Optional[tuple[float, float, float]] = None
        self._drive_calls = 0
        self._drive_skipped = 0
        self._drive_coalesced = 0
        self._drive_rtt_last_s: Optional[float] = None
        self._drive_rtt_ema_s: Optional[float] = None
        # How long the control thread waits for SetVelocity to be scheduled /
        # ack'd. Soft-loc cycles often occupy the loop for seconds; waiting the
        # full drive_timeout_s made every tick look like a hard IO failure even
        # though a later manual SetVelocity worked fine.
        self._drive_ack_timeout_s = min(0.35, max(0.05, float(drive_timeout_s)))

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

    def _schedule_drive(self, make_coro, intent: dict, *, wait_s: float) -> None:
        """Queue Base.SetVelocity on the module loop without starving on wait.

        Soft-loc / global_localize often occupy the shared loop for longer than
        a control tick. Blocking then cancelling the future made every hold look
        like a base failure, and cancelling on the *next* tick made every drive
        ``superseded`` before the base ever moved.
        """
        if self._loop.is_closed():
            raise RuntimeError("event loop is closed")

        intent["pending"] = True
        intent.pop("error", None)
        intent["issued"] = False

        fut = self._pending_drive_fut
        if fut is not None and not fut.done():
            # Keep the in-flight RPC; send this cmd after it finishes.
            self._drive_followup = (make_coro, intent)
            intent["coalesced"] = True
            self._drive_coalesced += 1
            self._last_drive = dict(intent)
            return

        self._drive_followup = None
        self._drive_calls += 1
        self._last_drive = dict(intent)

        async def _issue():
            current_make = make_coro
            current_intent = intent
            while True:
                t0 = time.monotonic()
                try:
                    await current_make()
                    rtt = time.monotonic() - t0
                    self._drive_rtt_last_s = rtt
                    if self._drive_rtt_ema_s is None:
                        self._drive_rtt_ema_s = rtt
                    else:
                        self._drive_rtt_ema_s = 0.8 * self._drive_rtt_ema_s + 0.2 * rtt
                    current_intent["issued"] = True
                    current_intent["error"] = None
                    current_intent["rtt_s"] = round(rtt, 4)
                    current_intent.pop("pending", None)
                    current_intent.pop("coalesced", None)
                except asyncio.CancelledError:
                    current_intent["issued"] = False
                    current_intent["error"] = "cancelled"
                    current_intent.pop("pending", None)
                    self._last_drive = dict(current_intent)
                    raise
                except Exception as exc:  # noqa: BLE001
                    current_intent["issued"] = False
                    current_intent["error"] = str(exc).strip() or type(exc).__name__
                    current_intent.pop("pending", None)
                    self._last_drive = dict(current_intent)
                    # Still drain follow-up (e.g. stop after a failed spin).
                else:
                    self._last_drive = dict(current_intent)

                follow = self._drive_followup
                self._drive_followup = None
                if follow is None:
                    return
                current_make, current_intent = follow
                current_intent["pending"] = True
                current_intent.pop("coalesced", None)
                self._drive_calls += 1
                self._last_drive = dict(current_intent)

        fut = asyncio.run_coroutine_threadsafe(_issue(), self._loop)
        self._pending_drive_fut = fut
        try:
            fut.result(timeout=max(0.0, float(wait_s)))
        except concurrent.futures.TimeoutError as exc:
            # Leave the RPC running — do not cancel. Caller sees pending.
            raise TimeoutError(
                f"Viam IO timed out after {wait_s:.1f}s"
            ) from exc
        finally:
            if self._pending_drive_fut is fut and fut.done():
                self._pending_drive_fut = None
        # Surface the intent that finished (may be a follow-up).
        finished = self._last_drive or intent
        if finished.get("error"):
            raise RuntimeError(finished["error"])

    def last_drive(self) -> Optional[dict]:
        """Most recent SetVelocity mapping (body rad/s → Viam mm/s + deg/s)."""
        return dict(self._last_drive) if self._last_drive else None

    def drive_stats(self) -> dict:
        """SetVelocity health for higher control-rate soak (RTT / skip / coalesce)."""
        return {
            "calls": self._drive_calls,
            "skipped": self._drive_skipped,
            "coalesced": self._drive_coalesced,
            "rtt_last_s": (
                None
                if self._drive_rtt_last_s is None
                else round(self._drive_rtt_last_s, 4)
            ),
            "rtt_ema_s": (
                None
                if self._drive_rtt_ema_s is None
                else round(self._drive_rtt_ema_s, 4)
            ),
            "pending": bool(
                self._pending_drive_fut is not None
                and not self._pending_drive_fut.done()
            ),
        }

    def pose_source(self) -> str:
        """How the last ``get_pose`` was obtained (``in_process`` / ``get_position`` / …)."""
        return self._pose_source

    def get_map(self) -> Optional[dict]:
        now = time.monotonic()
        if (
            self._map_cache is not None
            and now - self._map_cache_at < self._map_cache_s
        ):
            return self._map_cache
        if self._map_provider is not None:
            try:
                parsed = self._map_provider()
            except Exception:  # noqa: BLE001
                parsed = None
            if parsed is not None and parsed.get("grid") is not None:
                self._map_cache = parsed
                self._map_cache_at = now
                if self._viz is not None:
                    self._viz.set_map(parsed)
                return parsed
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
        """Map-frame pose in meters / radians from the configured SLAM service.

        Prefer **sync** sources so the control loop does not schedule
        ``GetPosition`` on the shared module event loop every tick (that
        starves ``Base.SetVelocity``). Order:

        1. Sync ``slam.get_position_pose2d()`` when the dependency is local SlamService.
        2. ``pose_provider`` (in-process registered SLAM service / engine).
        3. Async ``GetPosition`` only as a last resort (remote SLAM).
        """
        # Only call sync helpers declared on the SLAM class (local SlamService).
        # Instance MagicMock / gRPC stubs must not invent get_position_pose2d.
        if callable(getattr(type(self._slam), "get_position_pose2d", None)):
            try:
                p2 = self._slam.get_position_pose2d()
            except Exception as exc:  # noqa: BLE001
                self._log(f"get_position_pose2d failed: {exc}")
                p2 = None
            if p2 is not None:
                self._pose_source = "get_position_sync"
                if self._viz is not None:
                    self._viz.set_pose(p2)
                return p2

        if self._pose_provider is not None:
            try:
                p2 = self._pose_provider()
            except Exception as exc:  # noqa: BLE001
                self._log(f"pose_provider failed: {exc}")
                p2 = None
            if p2 is not None:
                self._pose_source = "in_process"
                if self._viz is not None:
                    self._viz.set_pose(p2)
                return p2

        via_api = self._pose_from_get_position()
        if via_api is not None:
            self._pose_source = "get_position"
            if self._viz is not None:
                self._viz.set_pose(via_api)
            return via_api

        self._pose_source = "none"
        return None

    def _pose_from_get_position(self) -> Optional[conv.Pose2D]:
        try:
            pose = self._run(self._slam.get_position(), timeout=2.0)
        except Exception:  # noqa: BLE001
            return None
        if pose is None:
            return None
        try:
            return slam_pose_to_pose2d(pose)
        except Exception:  # noqa: BLE001
            return None

    def get_scan(
        self, max_age_s: float = 2.0, *, include_obstacles_only: bool = True
    ) -> Optional[conv.LaserScan2D]:
        now = time.monotonic()
        pose = self.get_pose()
        # Merged cache includes depth; only reuse when the caller wants that.
        if (
            include_obstacles_only
            and self._scan_cache is not None
            and now - self._scan_cache_at <= max_age_s
            and self._scan_cache_pose is not None
            and pose is not None
        ):
            dtheta = abs(
                conv.normalize_angle(pose.theta - self._scan_cache_pose.theta)
            )
            dist = math.hypot(
                pose.x - self._scan_cache_pose.x, pose.y - self._scan_cache_pose.y
            )
            if dtheta <= math.radians(12.0) and dist <= 0.12:
                return self._scan_cache
        if self._scan_provider is not None:
            try:
                provided = self._scan_provider(max_age_s)
            except Exception:  # noqa: BLE001
                provided = None
            if provided is not None:
                if pose is not None and provided.capture_pose is None:
                    provided = conv.LaserScan2D(
                        ranges=provided.ranges,
                        angle_min=provided.angle_min,
                        angle_increment=provided.angle_increment,
                        range_min=provided.range_min,
                        range_max=provided.range_max,
                        sensor_pose=provided.sensor_pose,
                        capture_pose=pose,
                    )
                if include_obstacles_only:
                    self._scan_cache = provided
                    self._scan_cache_at = now
                    self._scan_cache_pose = pose
                return provided
        if not self._lidars:
            return self._scan_cache if include_obstacles_only else None
        scans = []
        for lidar in self._lidars:
            # Depth (obstacles_only) is for reactive slowing — not the rolling
            # local costmap / DWA. Including it there caused phantom blobs and
            # left/right chatter after the depth camera was added.
            if lidar.obstacles_only and not include_obstacles_only:
                continue
            scan = self._read_lidar_scan_sync(lidar, max_age_s=max_age_s)
            if scan is None:
                continue
            if pose is not None and lidar.obstacles_only:
                scan = self._align_obstacles_scan_to_pose(scan, pose)
                if scan is None:
                    continue
            scans.append(scan)
        if not scans:
            return self._scan_cache if include_obstacles_only else None
        merged = (
            scans[0]
            if len(scans) == 1
            else conv.merge_scans(scans, self._scan_bins)
        )
        if pose is not None:
            merged = conv.LaserScan2D(
                ranges=merged.ranges,
                angle_min=merged.angle_min,
                angle_increment=merged.angle_increment,
                range_min=merged.range_min,
                range_max=merged.range_max,
                sensor_pose=merged.sensor_pose,
                capture_pose=pose,
            )
        if include_obstacles_only:
            self._scan_cache = merged
            self._scan_cache_at = now
            self._scan_cache_pose = pose
        return merged

    def _map_pose_now(self) -> Optional[conv.Pose2D]:
        if self._pose_provider is not None:
            try:
                return self._pose_provider()
            except Exception:  # noqa: BLE001
                pass
        try:
            return self.get_pose()
        except Exception:  # noqa: BLE001
            return None

    def _stamp_capture_pose(
        self, scan: conv.LaserScan2D, pose: Optional[conv.Pose2D]
    ) -> conv.LaserScan2D:
        if pose is None:
            return scan
        return conv.LaserScan2D(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            sensor_pose=scan.sensor_pose,
            capture_pose=pose,
        )

    def _align_obstacles_scan_to_pose(
        self, scan: conv.LaserScan2D, current: conv.Pose2D
    ) -> Optional[conv.LaserScan2D]:
        """Move a cached depth scan into the live base_link, or drop if too stale.

        Without this, async depth (~0.4–1 s old) is painted as if seen from the
        *current* pose — walls smear into free space and the local planner weaves.

        Important: never restamp ``capture_pose`` to ``current`` without warping
        the ranges. A prior "small motion" shortcut did that, then ``get_scan``
        stamped the merge as current — body-frame points stayed frozen while the
        pose advanced, so phantoms accumulated into a black local-costmap blob.
        """
        cap = scan.capture_pose
        if cap is None:
            # Unknown capture frame: safer to drop than invent obstacles.
            return None
        dtheta = abs(conv.normalize_angle(current.theta - cap.theta))
        dist = math.hypot(current.x - cap.x, current.y - cap.y)
        if dist > self._obstacles_max_shift_m or dtheta > self._obstacles_max_shift_rad:
            return None
        if dist < 1e-4 and dtheta < 1e-5:
            return self._stamp_capture_pose(scan, current)
        pts = scan.to_points()
        if pts.size == 0:
            return self._stamp_capture_pose(scan, current)
        pts3 = np.column_stack([pts, np.zeros(len(pts))])
        aligned = conv.transform_points_between_poses(pts3, cap, current)[:, :2]
        n_bins = int(len(scan.ranges)) if len(scan.ranges) else self._scan_bins
        rebuilt = conv.points_to_scan(
            aligned,
            angle_min=-math.pi,
            angle_max=math.pi,
            num_bins=max(n_bins, 8),
            range_min=float(scan.range_min),
            range_max=float(scan.range_max),
        )
        return self._stamp_capture_pose(rebuilt, current)

    def _pcd_to_scan(
        self, raw: bytes, lidar: LidarConfig
    ) -> conv.LaserScan2D:
        pts = conv.parse_pcd(raw)
        # Depth cams are dense; crop by range/height then downsample so the
        # gRPC path (no shm) still spends its point budget on near obstacles.
        max_pts = 8000 if lidar.obstacles_only else 0
        pts = conv.prepare_lidar_point_cloud(
            pts,
            cloud_frame=lidar.cloud_frame,
            points_in_base_link=lidar.points_in_base_link,
            x=lidar.x,
            y=lidar.y,
            z=lidar.z,
            theta=lidar.theta,
            pitch=lidar.pitch,
            roll=lidar.roll,
            z_min=lidar.z_min,
            z_max=lidar.z_max,
            max_points=max_pts,
            range_min=float(lidar.min_range) if lidar.obstacles_only else 0.0,
            range_max=float(lidar.max_range) if lidar.obstacles_only else 0.0,
        )
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

    def _kick_obstacles_only_refresh(self, lidar: LidarConfig) -> None:
        """Schedule a non-blocking depth refresh on the module loop.

        Nav never waits for this. While ``get_point_cloud`` is awaiting network,
        ``SetVelocity`` can still run on the same loop. PCD parse runs in an
        executor so CPU work does not freeze drive commands.
        """
        name = lidar.name
        if name in self._obstacles_refresh_inflight:
            return
        cached = self._per_lidar_scan.get(name)
        now = time.monotonic()
        if (
            cached is not None
            and now - cached[1] < self._obstacles_only_period_s
        ):
            return
        if self._loop.is_closed():
            return
        cam = self._cameras.get(name)
        if cam is None and not lidar.shm_name:
            return
        self._obstacles_refresh_inflight.add(name)

        async def _job() -> None:
            try:
                shm_scan = self._try_shm_scan(
                    lidar, max_age_s=max(2.0, self._obstacles_only_period_s * 2)
                )
                if shm_scan is not None:
                    self._per_lidar_scan[name] = (
                        self._stamp_capture_pose(shm_scan, self._map_pose_now()),
                        time.monotonic(),
                    )
                    return
                if cam is None:
                    return
                data = await cam.get_point_cloud(timeout=2.0)
                raw = data[0] if isinstance(data, tuple) else data
                loop = asyncio.get_running_loop()
                scan = await loop.run_in_executor(
                    None, lambda: self._pcd_to_scan(raw, lidar)
                )
                if scan is not None:
                    self._per_lidar_scan[name] = (
                        self._stamp_capture_pose(scan, self._map_pose_now()),
                        time.monotonic(),
                    )
            except Exception as exc:  # noqa: BLE001
                self._log(f"obstacles_only lidar {name} refresh failed: {exc}")
            finally:
                self._obstacles_refresh_inflight.discard(name)

        try:
            asyncio.run_coroutine_threadsafe(_job(), self._loop)
        except Exception:  # noqa: BLE001
            self._obstacles_refresh_inflight.discard(name)

    def _read_lidar_scan_sync(
        self, lidar: LidarConfig, *, max_age_s: float
    ) -> Optional[conv.LaserScan2D]:
        if lidar.obstacles_only:
            # Cache-only on the hot path — never block the nav worker / event loop
            # on RealSense GetPointCloud (that was starving SetVelocity → stall).
            self._kick_obstacles_only_refresh(lidar)
            cached = self._per_lidar_scan.get(lidar.name)
            return cached[0] if cached is not None else None

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
            if mir_pts.base_link.size > 0:
                return conv.points_to_scan(
                    mir_pts.base_link,
                    angle_min=-math.pi,
                    angle_max=math.pi,
                    num_bins=self._scan_bins,
                    range_min=lidar.min_range,
                    range_max=lidar.max_range,
                )
            if mir_pts.sensor_scan is not None and conv.scan_has_returns(
                mir_pts.sensor_scan
            ):
                return mir_pts.sensor_scan
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
        lx_mm, ly_mm, ang_deg_s = body_twist_to_viam_set_velocity(
            vx, vy, vtheta, self._convention
        )
        intent = {
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "body_vtheta_rad_s": vtheta,
            "viam_linear_x_mm_s": lx_mm,
            "viam_linear_y_mm_s": ly_mm,
            "viam_angular_z_deg_s": ang_deg_s,
            "base": self._base_name,
            "issued": False,
            "error": None,
        }
        desired = (vx, vy, vtheta)
        if desired == self._last_desired_twist:
            # Control tick faster than meaningful cmd changes — don't pile
            # identical SetVelocity RPCs onto the module loop.
            self._drive_skipped += 1
            intent["issued"] = True
            intent["skipped"] = True
            self._last_drive = dict(intent)
            return
        self._last_desired_twist = desired

        def _make_coro():
            async def _issue_with_retry():
                try:
                    await self._base.set_velocity(
                        linear=Vector3(x=lx_mm, y=ly_mm, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=ang_deg_s),
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    if not _is_near_zero_rpm_error(exc):
                        raise
                if vx > 0.0 and abs(vtheta) >= 0.08:
                    vx_retry = max(vx, 0.06 + 0.32 * abs(vtheta))
                    lx_mm_r, ly_mm_r, _ = body_twist_to_viam_set_velocity(
                        vx_retry, vy, vtheta, self._convention
                    )
                    intent["retry"] = {"kind": "widen_arc", "body_vx_mps": vx_retry}
                    intent["viam_linear_x_mm_s"] = lx_mm_r
                    intent["viam_linear_y_mm_s"] = ly_mm_r
                    await self._base.set_velocity(
                        linear=Vector3(x=lx_mm_r, y=ly_mm_r, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=ang_deg_s),
                    )
                elif vx == 0.0 and abs(vtheta) >= 0.15:
                    intent["retry"] = {"kind": "spin"}
                    await self._base.set_velocity(
                        linear=Vector3(x=0.0, y=0.0, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=ang_deg_s),
                    )
                else:
                    intent["retry"] = {"kind": "stop"}
                    await self._base.set_velocity(
                        linear=Vector3(x=0.0, y=0.0, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=0.0),
                    )

            return _issue_with_retry()

        try:
            self._schedule_drive(
                _make_coro,
                intent,
                wait_s=self._drive_ack_timeout_s,
            )
        except TimeoutError:
            if intent.get("pending") and not intent.get("error"):
                # Command is still queued on the module loop; do not fail the tick.
                self._log(
                    f"SetVelocity ack slow on base {self._base_name!r} "
                    f"(loop busy; leaving command pending)"
                )
                return
            raise
        except Exception as exc:  # noqa: BLE001
            # Allow a retry of the same twist after a hard failure.
            self._last_desired_twist = None
            self._log(
                f"SetVelocity failed on base {self._base_name!r}: "
                f"{str(exc).strip() or type(exc).__name__}"
            )
            raise

    def stop(self) -> None:
        """Zero the base without blocking the control thread on a busy loop."""
        # Shares skip-duplicate + coalesce paths with set_velocity so loc_hold /
        # succeed / cancel do not re-issue SetVelocity(0) every tick.
        self.set_velocity(0.0, 0.0, 0.0)
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

    def set_viz_local_costmap(self, costmap: dict) -> None:
        if self._viz is None:
            return
        self._viz.set_local_costmap(costmap)

    def get_localization_hold(self) -> Optional[dict]:
        provider = self._localization_hold_provider
        if provider is None:
            return None
        try:
            hold = provider()
        except Exception:  # noqa: BLE001 - never block drive on status read
            return None
        return hold if isinstance(hold, dict) else None


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

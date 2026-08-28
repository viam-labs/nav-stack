"""Background occupancy SLAM loop: predict → match → update."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import MODE_LOCALIZING, MODE_MAPPING, SlamConfig
from ..nav.global_localize import OccupancyMap
from ..nav.maps import MapStore
from ..ros import conversions as conv
from . import occupancy as occ
from . import persistence
from . import scan_match
from .io_sensors import BuiltinSensors
from .types import LogOddsGrid


class BuiltinSlamEngine:
    """In-process 2D occupancy mapping + localization."""

    def __init__(
        self,
        cfg: SlamConfig,
        sensors: BuiltinSensors,
        map_store: MapStore,
        *,
        logger=None,
        rate_hz: float = 10.0,
        match_period_s: float = 0.3,
    ):
        self._cfg = cfg
        self._sensors = sensors
        self._map_store = map_store
        self._logger = logger
        self._period = 1.0 / max(rate_hz, 1.0)
        # Scan matching + map inserts are CPU-heavy; running them at the full
        # predict rate starved the module event loop (nav "Viam IO timed out").
        self._match_period_s = max(0.0, float(match_period_s))

        self._lock = threading.RLock()
        self._grid: LogOddsGrid = occ.empty_grid(
            resolution=cfg.slam_toolbox.resolution
        )
        self._pose = conv.Pose2D(0.0, 0.0, 0.0)
        self._last_odom_pose: Optional[conv.Pose2D] = None
        self._last_odom_time: Optional[float] = None
        self._last_odom_heading: Optional[float] = None
        self._mode = cfg.mode
        self._generation = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_match_score = float("nan")
        self._last_prior_score = float("nan")
        self._last_scan_age_s = float("nan")
        self._ticks = 0
        self._updates = 0
        self._match_accepts = 0
        self._match_rejects = 0
        self._last_match_at = 0.0
        # Mapping inserts only after movement (slam_toolbox-style) or timeout.
        self._last_insert_pose: Optional[conv.Pose2D] = None
        self._last_insert_at = 0.0
        # Anti-oscillation: require two agreeing frames before applying a jump.
        self._pending_match: Optional[conv.Pose2D] = None
        self._pending_count = 0
        # Cached ROS-style grid for scan matching (invalidated on map edits).
        self._occ_cache: Optional[OccupancyMap] = None
        self._occ_cache_generation = -1
        self._occ_cache_known = 0.0

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger(msg)
            except Exception:  # noqa: BLE001
                pass

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="builtin-slam",
            daemon=True,
        )
        self._thread.start()
        self._log(f"builtin SLAM engine started ({self._mode})")

    def stop(self) -> None:
        self._running = False
        thr = self._thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=2.0)
        self._thread = None

    def configure_mode(self, mode: str, map_dir: Optional[Path] = None) -> None:
        with self._lock:
            self._mode = mode
            if map_dir is not None:
                loaded = persistence.load_log_odds(map_dir)
                if loaded is not None:
                    self._grid = loaded
                    self._invalidate_occ_cache()
                    self._generation += 1
                    self._log(f"loaded occupancy map from {map_dir}")
                elif mode == MODE_LOCALIZING:
                    raise RuntimeError(
                        f"cannot localize: no map.yaml/map.pgm under {map_dir}"
                    )
                elif mode == MODE_MAPPING:
                    # Fresh mapping session.
                    self._grid = occ.empty_grid(
                        resolution=self._cfg.slam_toolbox.resolution
                    )
                    self._pose = conv.Pose2D(0.0, 0.0, 0.0)
                    self._last_odom_pose = None
                    self._last_odom_heading = None
                    self._invalidate_occ_cache()
                    self._generation += 1

    def reset_map(self) -> None:
        with self._lock:
            self._grid = occ.empty_grid(
                resolution=self._cfg.slam_toolbox.resolution
            )
            self._pose = conv.Pose2D(0.0, 0.0, 0.0)
            self._last_odom_pose = None
            self._last_odom_heading = None
            self._last_insert_pose = None
            self._invalidate_occ_cache()
            self._generation += 1

    def set_pose(self, pose: conv.Pose2D) -> None:
        with self._lock:
            self._pose = pose
            self._pending_match = None
            self._pending_count = 0

    def save_map(self, map_dir: Path) -> None:
        with self._lock:
            persistence.save_occupancy(map_dir, self._grid)

    def _invalidate_occ_cache(self) -> None:
        self._occ_cache = None
        self._occ_cache_generation = -1

    def _occupancy_for_match(self) -> OccupancyMap:
        if (
            self._occ_cache is not None
            and self._occ_cache_generation == self._generation
        ):
            return self._occ_cache
        int16 = occ.to_occupancy_int16(self._grid)
        self._occ_cache = scan_match.occupancy_map_from_int16(
            int16,
            resolution=self._grid.resolution,
            origin_x=self._grid.origin_x,
            origin_y=self._grid.origin_y,
        )
        self._occ_cache_generation = self._generation
        self._occ_cache_known = float(np.mean(int16 >= 0))
        return self._occ_cache

    # -- queries -------------------------------------------------------------
    def get_pose(self) -> conv.Pose2D:
        with self._lock:
            return self._pose

    def get_map(self) -> dict:
        with self._lock:
            grid = occ.to_occupancy_int16(self._grid)
            return {
                "grid": grid,
                "resolution": self._grid.resolution,
                "origin_x": self._grid.origin_x,
                "origin_y": self._grid.origin_y,
                "generation": self._generation,
            }

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "slam_backend": "builtin",
                "mode": self._mode,
                "running": self._running,
                "ticks": self._ticks,
                "map_updates": self._updates,
                "match_accepts": self._match_accepts,
                "match_rejects": self._match_rejects,
                "pose": {
                    "x": self._pose.x,
                    "y": self._pose.y,
                    "theta": self._pose.theta,
                },
                "grid": {
                    "width": self._grid.width,
                    "height": self._grid.height,
                    "resolution": self._grid.resolution,
                    "origin_x": self._grid.origin_x,
                    "origin_y": self._grid.origin_y,
                },
                "last_match_score": self._last_match_score,
                "last_prior_score": self._last_prior_score,
                "last_scan_age_s": self._last_scan_age_s,
                "generation": self._generation,
            }

    # -- loop ----------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                self._log(f"builtin SLAM tick failed: {exc}")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._period - elapsed))

    def _tick(self) -> None:
        # Matching/mapping against a stale scan biases the pose toward where
        # the robot was when the scan was taken; keep the age tight even if
        # the nav-facing scan_max_age_s is generous.
        scan_age = min(float(self._cfg.scan_max_age_s or 2.0), 0.75)
        scan = self._sensors.get_scan(max_age_s=scan_age)
        odom = self._sensors.get_odom()
        now = time.monotonic()

        with self._lock:
            self._ticks += 1
            predicted = self._predict(odom, now)
            self._pose = predicted
            if scan is None:
                return
            occ_map = self._occupancy_for_match()
            known = self._occ_cache_known
            run_match = (
                known > 0.02
                and now - self._last_match_at >= self._match_period_s
            )

        # Heavy work happens OUTSIDE the lock: matching while holding it
        # blocked every get_pose/get_map (nav planning + pose reads stalled).
        if run_match:
            matched, score, prior_score = scan_match.refine_pose(
                occ_map, scan, predicted
            )
            with self._lock:
                self._last_match_at = now
                self._last_match_score = score
                self._last_prior_score = prior_score
                applied = self._apply_match(self._pose, matched, odom)
                if matched is not None and applied:
                    self._match_accepts += 1
                else:
                    self._match_rejects += 1

        if self._mode == MODE_MAPPING:
            self._maybe_insert_scan(scan, now)

    def _maybe_insert_scan(self, scan, now: float) -> None:
        """Ray-cast into the grid only after movement or a timeout.

        Bresenham inserts are Python loops; doing them at 10 Hz regardless of
        motion burned CPU for no map benefit (same pose = same rays).
        """
        with self._lock:
            pose = self._pose
            last = self._last_insert_pose
            if last is not None:
                moved = math.hypot(pose.x - last.x, pose.y - last.y)
                turned = abs(conv.normalize_angle(pose.theta - last.theta))
                if (
                    moved < 0.03
                    and turned < math.radians(1.5)
                    and now - self._last_insert_at < 1.0
                ):
                    return
            self._grid = occ.insert_scan(
                self._grid,
                pose.x,
                pose.y,
                pose.theta,
                np.asarray(scan.ranges, dtype=float),
                float(scan.angle_min),
                float(scan.angle_increment),
                range_min=float(scan.range_min),
                range_max=float(scan.range_max)
                if math.isfinite(scan.range_max)
                else 30.0,
            )
            self._last_insert_pose = pose
            self._last_insert_at = now
            self._updates += 1
            if self._updates % 20 == 0:
                self._generation += 1
                self._invalidate_occ_cache()

    def _poses_agree(self, a: conv.Pose2D, b: conv.Pose2D) -> bool:
        return (
            math.hypot(a.x - b.x, a.y - b.y) <= 0.08
            and abs(conv.normalize_angle(a.theta - b.theta)) <= math.radians(5.0)
        )

    def _blend_pose(
        self,
        prior: conv.Pose2D,
        target: conv.Pose2D,
        *,
        alpha: float,
        max_xy: float,
        max_yaw: float,
    ) -> conv.Pose2D:
        """Move partially toward ``target`` and clamp the step size."""
        alpha = min(max(alpha, 0.0), 1.0)
        x = prior.x + alpha * (target.x - prior.x)
        y = prior.y + alpha * (target.y - prior.y)
        th = conv.normalize_angle(
            prior.theta + alpha * conv.normalize_angle(target.theta - prior.theta)
        )
        dx = x - prior.x
        dy = y - prior.y
        dist = math.hypot(dx, dy)
        if dist > max_xy and dist > 1e-9:
            scale = max_xy / dist
            x = prior.x + dx * scale
            y = prior.y + dy * scale
        dth = conv.normalize_angle(th - prior.theta)
        if abs(dth) > max_yaw:
            th = conv.normalize_angle(prior.theta + math.copysign(max_yaw, dth))
        return conv.Pose2D(x, y, th)

    # A correction this small is drift tracking, not a peak jump — apply it
    # immediately. Requiring confirmation + tiny clamps for these let odom
    # drift outrun the correction rate (pose "arrived" before the robot).
    _SMALL_CORRECTION_M = 0.15
    _SMALL_CORRECTION_RAD = math.radians(8.0)

    def _apply_match(
        self,
        predicted: conv.Pose2D,
        matched: Optional[conv.Pose2D],
        odom: Optional[conv.OdomReading],
    ) -> bool:
        """Apply a blended match. Returns True if pose was corrected."""
        del odom  # prediction is delta-based; no map->odom re-anchor needed
        if matched is None:
            self._pending_match = None
            self._pending_count = 0
            self._pose = predicted
            return False

        dist = math.hypot(matched.x - predicted.x, matched.y - predicted.y)
        dyaw = abs(conv.normalize_angle(matched.theta - predicted.theta))
        small = dist <= self._SMALL_CORRECTION_M and dyaw <= self._SMALL_CORRECTION_RAD

        if small:
            # Continuous drift correction: no confirmation needed.
            self._pending_match = None
            self._pending_count = 0
            self._pose = self._blend_pose(
                predicted,
                matched,
                alpha=0.6,
                max_xy=self._SMALL_CORRECTION_M,
                max_yaw=self._SMALL_CORRECTION_RAD,
            )
            return True

        # Large jump (competing peak / recovery): require two agreeing frames.
        if self._pending_match is not None and self._poses_agree(
            matched, self._pending_match
        ):
            self._pending_count += 1
            self._pending_match = matched
        else:
            self._pending_match = matched
            self._pending_count = 1

        if self._pending_count < 2:
            self._pose = predicted
            return False

        self._pose = self._blend_pose(
            predicted,
            matched,
            alpha=0.6,
            max_xy=0.25,
            max_yaw=math.radians(12.0),
        )
        # Keep the streak so a persistent target keeps applying every match.
        self._pending_count = 2
        return True

    def _predict(
        self, odom: Optional[conv.OdomReading], now: float
    ) -> conv.Pose2D:
        if odom is None:
            return self._pose

        if odom.pose is not None:
            # Absolute odom pose: apply the *relative* motion since the last
            # sample onto the current map pose. Composing map_to_odom with the
            # absolute pose let IMU/magnetometer heading snaps (merged into
            # odom pose theta) teleport the map pose every tick — the source
            # of correct↔wrong pose flapping.
            prev = self._last_odom_pose
            self._last_odom_pose = odom.pose
            self._last_odom_time = now
            self._last_odom_heading = odom.pose.theta
            if prev is None:
                return self._pose
            delta = conv.compose_poses(conv.invert_pose(prev), odom.pose)
            # Gate implausible per-tick jumps (heading snap, odom reset).
            if (
                math.hypot(delta.x, delta.y) > 0.5
                or abs(delta.theta) > math.radians(40.0)
            ):
                self._log(
                    "builtin SLAM: rejected odom jump "
                    f"dx={delta.x:.2f} dy={delta.y:.2f} "
                    f"dth={math.degrees(delta.theta):.0f}deg"
                )
                return self._pose
            return conv.compose_poses(self._pose, delta)

        # Twist integration in the *map* frame. Never snap map yaw to an
        # absolute IMU/magnetometer heading — that frame is not map-aligned
        # and was yanking localization after every tick.
        dt = 0.0
        if self._last_odom_time is not None:
            dt = max(0.0, min(now - self._last_odom_time, 0.5))
        self._last_odom_time = now
        if dt <= 0.0:
            if odom.heading_rad is not None and self._last_odom_heading is None:
                self._last_odom_heading = odom.heading_rad
            return self._pose

        dth = odom.vtheta * dt
        if odom.heading_rad is not None:
            if self._last_odom_heading is not None:
                heading_delta = conv.normalize_angle(
                    odom.heading_rad - self._last_odom_heading
                )
                # Reject magnetometer jumps; fall back to gyro integration.
                if abs(heading_delta) <= math.radians(40.0):
                    dth = heading_delta
            self._last_odom_heading = odom.heading_rad

        c = math.cos(self._pose.theta)
        s = math.sin(self._pose.theta)
        dx = (c * odom.vx - s * odom.vy) * dt
        dy = (s * odom.vx + c * odom.vy) * dt
        return conv.Pose2D(
            self._pose.x + dx,
            self._pose.y + dy,
            conv.normalize_angle(self._pose.theta + dth),
        )

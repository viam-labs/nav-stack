"""Short-term memory of depth-camera obstacles outside the camera's view.

A forward depth camera sees the low parts of things the 2D lidar plane
misses (the bottom of a curved folding-table leg, feet, bumpers). The merged
scan only holds what each sensor sees *now*, so once such a part slides past
the camera's narrow field of view it vanishes while the body is right next to
it and the guard is left with the lidar's reading of the upper leg — ~8 cm too
far (live: three grazes on one table leg).

Like a Nav2 obstacle layer, remembered points are cleared only by the sensor
that marked them: when a newer frame looks at the spot and no longer sees
anything there. They are also dropped when the body covers them (a real
obstacle there would be a collision), beyond ``keep_radius_m``, or after
``ttl_s``.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from src.geom.conversions import Pose2D


class DepthObstacleMemory:
    def __init__(
        self,
        *,
        length_m: float,
        width_m: float,
        cell_m: float = 0.03,
        keep_radius_m: float = 1.2,
        ttl_s: float = 20.0,
        edge_margin_rad: float = math.radians(4.0),
        range_margin_m: float = 0.05,
    ):
        self._hl = length_m / 2.0
        self._hw = width_m / 2.0
        self._cell = float(cell_m)
        self._keep_r = float(keep_radius_m)
        self._ttl = float(ttl_s)
        self._edge = float(edge_margin_rad)
        self._rmargin = float(range_margin_m)
        self._cells: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._seen: Dict[str, float] = {}

    def clear(self) -> None:
        self._cells.clear()
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._cells)

    def update(self, frames: Iterable, pose: Pose2D, now: float) -> None:
        """Fold in new ``(stamp, scan, sensor_cfg)`` frames, then evict.

        ``scan`` is base-frame at ``scan.capture_pose``; ``sensor_cfg`` needs
        ``name, x, y, theta, fov_deg, min_range, max_range``.
        """
        for stamp, scan, sensor in frames:
            cap = getattr(scan, "capture_pose", None)
            if cap is None or self._seen.get(sensor.name) == stamp:
                continue
            self._seen[sensor.name] = stamp
            self._clear_visible(cap, sensor)
            pts = scan.to_points()
            if pts.size:
                c, s = math.cos(cap.theta), math.sin(cap.theta)
                wx = cap.x + c * pts[:, 0] - s * pts[:, 1]
                wy = cap.y + s * pts[:, 0] + c * pts[:, 1]
                near = (wx - pose.x) ** 2 + (wy - pose.y) ** 2 <= self._keep_r**2
                for x, y in zip(wx[near], wy[near]):
                    key = (int(math.floor(x / self._cell)), int(math.floor(y / self._cell)))
                    self._cells[key] = (float(x), float(y), now)
        self._evict(pose, now)

    def points(self) -> np.ndarray:
        if not self._cells:
            return np.empty((0, 2))
        return np.array([(x, y) for x, y, _ in self._cells.values()], dtype=float)

    def _clear_visible(self, cap: Pose2D, sensor) -> None:
        if not self._cells:
            return
        c, s = math.cos(cap.theta), math.sin(cap.theta)
        sx = cap.x + c * sensor.x - s * sensor.y
        sy = cap.y + s * sensor.x + c * sensor.y
        yaw = cap.theta + float(getattr(sensor, "theta", 0.0))
        half = math.radians(float(sensor.fov_deg)) / 2.0 - self._edge
        rmin = float(sensor.min_range) + self._rmargin
        rmax = float(sensor.max_range) - self._rmargin
        if half <= 0.0 or rmax <= rmin:
            return
        keys = list(self._cells.keys())
        xy = np.array([self._cells[k][:2] for k in keys])
        dx, dy = xy[:, 0] - sx, xy[:, 1] - sy
        r = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - yaw
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
        visible = (np.abs(bearing) <= half) & (r >= rmin) & (r <= rmax)
        for k, v in zip(keys, visible):
            if v:
                del self._cells[k]

    def _evict(self, pose: Pose2D, now: float) -> None:
        if not self._cells:
            return
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        drop = []
        for k, (x, y, t) in self._cells.items():
            dx, dy = x - pose.x, y - pose.y
            if now - t > self._ttl or dx * dx + dy * dy > self._keep_r**2:
                drop.append(k)
                continue
            if abs(c * dx + s * dy) <= self._hl and abs(-s * dx + c * dy) <= self._hw:
                drop.append(k)
        for k in drop:
            del self._cells[k]


def depth_frames(world) -> list:
    """``world.get_depth_frames()`` if the world exposes it, else ``[]``."""
    fn = getattr(world, "get_depth_frames", None)
    if fn is None:
        return []
    try:
        return list(fn() or [])
    except Exception:  # noqa: BLE001 - memory is best-effort
        return []

"""Short-term memory of depth returns that would pass above the cart.

A forward depth camera sees a low beam, sign, or table edge from a few metres
away. As the cart closes in, that return rises out of the vertical field of
view while the camera still points at the same ground spot, so a look that
"sees nothing" is the obstacle leaving the image, not the obstacle leaving
the hallway. Remembered points stay until a newer frame could still see that
height and no longer does, or they age out / fall behind the keep radius.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from src.geom.conversions import Pose2D


class AboveCartMemory:
    def __init__(
        self,
        *,
        cart_height_m: Optional[float] = None,
        keep_radius_m: float = 2.5,
        ttl_s: float = 20.0,
        cell_m: float = 0.05,
        edge_margin_rad: float = math.radians(4.0),
        range_margin_m: float = 0.05,
    ):
        # None: keep every return the camera already clipped to its z band.
        self._cart_h = float(cart_height_m) if cart_height_m else None
        self._keep_r = float(keep_radius_m)
        self._ttl = float(ttl_s)
        self._cell = float(cell_m)
        self._edge = float(edge_margin_rad)
        self._rmargin = float(range_margin_m)
        # cell -> (x, y, z, stamp)
        self._cells: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        self._seen: Dict[str, float] = {}

    def clear(self) -> None:
        self._cells.clear()
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._cells)

    def update(self, frames: Iterable, pose: Pose2D, now: float) -> None:
        """Fold in ``(stamp, xyz_base, sensor, capture_pose)`` frames, then evict.

        ``xyz_base`` is base-frame at ``capture_pose`` (XY in the robot, Z
        height above the floor). With ``cart_height_m`` set, only points above
        the cart are kept. Unset, every return in the frame is kept — the
        camera already limited that cloud to ``z_max``.
        """
        for stamp, xyz, sensor, cap in frames:
            if cap is None or self._seen.get(sensor.name) == stamp:
                continue
            self._seen[sensor.name] = stamp
            world = self._to_world(cap, xyz)
            self._clear_still_visible(cap, sensor, world)
            if world.size:
                above = (
                    np.ones(len(world), dtype=bool)
                    if self._cart_h is None
                    else world[:, 2] > self._cart_h
                )
                near = (world[:, 0] - pose.x) ** 2 + (world[:, 1] - pose.y) ** 2
                keep = above & (near <= self._keep_r**2)
                for x, y, z in world[keep]:
                    key = (
                        int(math.floor(float(x) / self._cell)),
                        int(math.floor(float(y) / self._cell)),
                    )
                    self._cells[key] = (float(x), float(y), float(z), now)
        self._evict(pose, now)

    def points(self) -> np.ndarray:
        if not self._cells:
            return np.empty((0, 2))
        return np.array([(x, y) for x, y, _, _ in self._cells.values()], dtype=float)

    def _to_world(self, cap: Pose2D, xyz) -> np.ndarray:
        pts = np.asarray(xyz, dtype=float)
        if pts.size == 0:
            return np.empty((0, 3))
        pts = pts.reshape(-1, pts.shape[-1])
        if pts.shape[1] < 3:
            return np.empty((0, 3))
        c, s = math.cos(cap.theta), math.sin(cap.theta)
        wx = cap.x + c * pts[:, 0] - s * pts[:, 1]
        wy = cap.y + s * pts[:, 0] + c * pts[:, 1]
        return np.column_stack([wx, wy, pts[:, 2]])

    def _camera_pose(self, cap: Pose2D, sensor) -> Tuple[float, float, float, float]:
        c, s = math.cos(cap.theta), math.sin(cap.theta)
        sx = cap.x + c * float(sensor.x) - s * float(sensor.y)
        sy = cap.y + s * float(sensor.x) + c * float(sensor.y)
        sz = float(getattr(sensor, "z", 0.0))
        yaw = cap.theta + float(getattr(sensor, "theta", 0.0))
        return sx, sy, sz, yaw

    def _can_see(self, cap: Pose2D, sensor, x: float, y: float, z: float) -> bool:
        """True when this frame's vertical and horizontal view still cover the point."""
        sx, sy, sz, yaw = self._camera_pose(cap, sensor)
        dx, dy = x - sx, y - sy
        horiz = math.hypot(dx, dy)
        half_h = math.radians(float(sensor.fov_deg)) / 2.0 - self._edge
        half_v = math.radians(float(getattr(sensor, "vfov_deg", 58.0))) / 2.0 - self._edge
        rmin = float(sensor.min_range) + self._rmargin
        rmax = float(sensor.max_range) - self._rmargin
        if half_h <= 0.0 or half_v <= 0.0 or rmax <= rmin or horiz < 1e-3:
            return False
        bearing = math.atan2(dy, dx) - yaw
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
        elev = math.atan2(z - sz, horiz)
        return (
            abs(bearing) <= half_h
            and abs(elev) <= half_v
            and rmin <= horiz <= rmax
        )

    def _clear_still_visible(self, cap: Pose2D, sensor, world: np.ndarray) -> None:
        if not self._cells:
            return
        fresh = set()
        if world.size:
            for x, y, z in world:
                if self._cart_h is not None and float(z) <= self._cart_h:
                    continue
                fresh.add(
                    (
                        int(math.floor(float(x) / self._cell)),
                        int(math.floor(float(y) / self._cell)),
                    )
                )
        drop = []
        for key, (x, y, z, _) in self._cells.items():
            if key in fresh:
                continue
            if self._can_see(cap, sensor, x, y, z):
                drop.append(key)
        for key in drop:
            del self._cells[key]

    def _evict(self, pose: Pose2D, now: float) -> None:
        if not self._cells:
            return
        drop = []
        for key, (x, y, _, t) in self._cells.items():
            dx, dy = x - pose.x, y - pose.y
            if now - t > self._ttl or dx * dx + dy * dy > self._keep_r**2:
                drop.append(key)
        for key in drop:
            del self._cells[key]


def above_cart_frames(world) -> list:
    """``world.get_above_cart_frames()`` if the world exposes it, else ``[]``."""
    fn = getattr(world, "get_above_cart_frames", None)
    if fn is None:
        return []
    try:
        return list(fn() or [])
    except Exception:  # noqa: BLE001 - memory is best-effort
        return []

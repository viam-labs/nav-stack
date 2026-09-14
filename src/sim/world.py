"""Ground-truth 2D world for builtin simulation (raycast lidar + kinematics)."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..geom import conversions as conv

# Occupancy conventions match nav-stack / ROS: -1 unknown, 0 free, 100 occupied.
_OCCUPIED_MIN = 50


@dataclass
class SimMap:
    """Axis-aligned occupancy grid in the map frame."""

    grid: np.ndarray  # (rows, cols) int16
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return int(self.grid[row, col]) >= _OCCUPIED_MIN


def make_builtin_corridor(
    *,
    resolution: float = 0.05,
    width_m: float = 8.0,
    height_m: float = 6.0,
    corridor_m: float = 1.6,
) -> SimMap:
    """Simple rectangular room with an internal wall forming an L-corridor."""
    cols = max(8, int(round(width_m / resolution)))
    rows = max(8, int(round(height_m / resolution)))
    grid = np.zeros((rows, cols), dtype=np.int16)
    # Outer walls.
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    # Interior wall: vertical barrier with a gap at the top, plus a horizontal stub.
    wall_col = cols // 2
    gap = max(2, int(round(corridor_m / resolution)))
    grid[gap:-1, wall_col] = 100
    stub_row = rows // 2
    grid[stub_row, wall_col : wall_col + cols // 4] = 100
    return SimMap(
        grid=grid,
        resolution=float(resolution),
        origin_x=0.0,
        origin_y=0.0,
    )


def load_sim_map(path: str | Path) -> SimMap:
    """Load a sim map from ``.npy`` (+ optional ``.json`` meta) or generate builtin."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sim map not found: {p}")
    if p.suffix == ".npy":
        grid = np.load(p)
        meta_path = p.with_suffix(".json")
        resolution = 0.05
        origin_x = 0.0
        origin_y = 0.0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            resolution = float(meta.get("resolution", resolution))
            origin_x = float(meta.get("origin_x", origin_x))
            origin_y = float(meta.get("origin_y", origin_y))
        return SimMap(
            grid=np.asarray(grid, dtype=np.int16),
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
        )
    raise ValueError(f"unsupported sim map format: {p.suffix} (use .npy)")


class SimWorld:
    """Shared kinematic + raycast world for ``sim-base`` and ``SimSensors``."""

    def __init__(
        self,
        sim_map: SimMap,
        *,
        seed_pose: Optional[conv.Pose2D] = None,
        scan_bins: int = 360,
        range_min: float = 0.05,
        range_max: float = 20.0,
        name: str = "default",
    ):
        self.name = name
        self.map = sim_map
        self.scan_bins = int(scan_bins)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self._lock = threading.RLock()
        seed = seed_pose or conv.Pose2D(1.0, 1.0, 0.0)
        self._seed = conv.Pose2D(seed.x, seed.y, seed.theta)
        self._pose = conv.Pose2D(seed.x, seed.y, seed.theta)
        self._vx = 0.0  # ROS body m/s
        self._vy = 0.0
        self._vtheta = 0.0  # rad/s
        self._last_integrate_at = time.monotonic()
        self._odom_pose = conv.Pose2D(seed.x, seed.y, seed.theta)

    def reset(self, pose: Optional[conv.Pose2D] = None) -> None:
        with self._lock:
            target = pose or self._seed
            self._pose = conv.Pose2D(target.x, target.y, target.theta)
            self._odom_pose = conv.Pose2D(target.x, target.y, target.theta)
            self._vx = self._vy = self._vtheta = 0.0
            self._last_integrate_at = time.monotonic()

    def set_seed(self, pose: conv.Pose2D) -> None:
        with self._lock:
            self._seed = conv.Pose2D(pose.x, pose.y, pose.theta)

    def teleport(self, pose: conv.Pose2D) -> None:
        """Snap body + odom to ``pose`` (e.g. SLAM localize) without clearing cmd."""
        with self._lock:
            self._pose = conv.Pose2D(pose.x, pose.y, pose.theta)
            self._odom_pose = conv.Pose2D(pose.x, pose.y, pose.theta)
            self._last_integrate_at = time.monotonic()

    def set_velocity_ros(self, vx: float, vy: float, vtheta: float) -> None:
        """Command body-frame twist in ROS convention (x forward, yaw CCW)."""
        with self._lock:
            self._integrate_unlocked(time.monotonic())
            self._vx = float(vx)
            self._vy = float(vy)
            self._vtheta = float(vtheta)

    def stop(self) -> None:
        self.set_velocity_ros(0.0, 0.0, 0.0)

    def get_pose(self) -> conv.Pose2D:
        with self._lock:
            self._integrate_unlocked(time.monotonic())
            return conv.Pose2D(self._pose.x, self._pose.y, self._pose.theta)

    def get_odom(self) -> conv.OdomReading:
        with self._lock:
            self._integrate_unlocked(time.monotonic())
            return conv.OdomReading(
                vx=self._vx,
                vy=self._vy,
                vtheta=self._vtheta,
                pose=conv.Pose2D(
                    self._odom_pose.x, self._odom_pose.y, self._odom_pose.theta
                ),
                heading_rad=self._odom_pose.theta,
            )

    def get_scan(self, *, capture_pose: bool = True) -> conv.LaserScan2D:
        with self._lock:
            self._integrate_unlocked(time.monotonic())
            pose = conv.Pose2D(self._pose.x, self._pose.y, self._pose.theta)
            ranges = self._raycast_unlocked(pose)
        return conv.LaserScan2D(
            ranges=ranges,
            angle_min=-math.pi,
            angle_increment=(2.0 * math.pi) / max(1, self.scan_bins),
            range_min=self.range_min,
            range_max=self.range_max,
            sensor_pose=conv.Pose2D(0.0, 0.0, 0.0),
            capture_pose=pose if capture_pose else None,
        )

    def _cell_blocked(self, x: float, y: float) -> bool:
        row, col = self.map.world_to_cell(x, y)
        return self.map.occupied(row, col)

    def _integrate_unlocked(self, now: float) -> None:
        dt = max(0.0, min(0.2, now - self._last_integrate_at))
        self._last_integrate_at = now
        if dt <= 0.0:
            return
        c, s = math.cos(self._pose.theta), math.sin(self._pose.theta)
        dx = (self._vx * c - self._vy * s) * dt
        dy = (self._vx * s + self._vy * c) * dt
        dtheta = self._vtheta * dt
        new_theta = conv.normalize_angle(self._pose.theta + dtheta)
        # Collision: refuse XY that enters occupied / out-of-bounds cells.
        # Allow yaw in place so the robot can turn away from a wall.
        trial_x = self._pose.x + dx
        trial_y = self._pose.y + dy
        if self._cell_blocked(trial_x, trial_y):
            # Axis slide: try X-only then Y-only.
            if not self._cell_blocked(trial_x, self._pose.y):
                trial_y = self._pose.y
            elif not self._cell_blocked(self._pose.x, trial_y):
                trial_x = self._pose.x
            else:
                trial_x, trial_y = self._pose.x, self._pose.y
                # Kill linear cmd so avoidance / teleop do not keep pushing.
                self._vx = 0.0
                self._vy = 0.0
        applied_dx = trial_x - self._pose.x
        applied_dy = trial_y - self._pose.y
        self._pose = conv.Pose2D(trial_x, trial_y, new_theta)
        self._odom_pose = conv.Pose2D(
            self._odom_pose.x + applied_dx,
            self._odom_pose.y + applied_dy,
            conv.normalize_angle(self._odom_pose.theta + dtheta),
        )

    def _raycast_unlocked(self, pose: conv.Pose2D) -> np.ndarray:
        n = max(1, self.scan_bins)
        ranges = np.full(n, np.inf, dtype=np.float64)
        step = max(self.map.resolution * 0.5, 0.01)
        angle_inc = (2.0 * math.pi) / n
        for i in range(n):
            ang = pose.theta + (-math.pi + i * angle_inc)
            dx = math.cos(ang) * step
            dy = math.sin(ang) * step
            x, y = pose.x, pose.y
            dist = 0.0
            hit = False
            while dist < self.range_max:
                x += dx
                y += dy
                dist += step
                row, col = self.map.world_to_cell(x, y)
                if self.map.occupied(row, col):
                    hit = True
                    break
            if hit and dist >= self.range_min:
                ranges[i] = dist
            elif not hit:
                ranges[i] = np.inf
        return ranges

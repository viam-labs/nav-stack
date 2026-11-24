"""Fakes for exploration tests: SLAM that reveals a sim world, nav that drives lines."""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.explore.clients import NavStatus, PaintHold, Pose, SlamStatus
from src.explore.grid import OccGrid
from src.geom import conversions as conv
from src.sim.world import SimWorld


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += max(0.0, float(s))


class FakeSlam:
    def __init__(self, world: SimWorld, *, n_rays: int = 360, max_range_m: float = 8.0):
        self.world = world
        self.base_map = np.array(world.map.grid, copy=True)
        self.known = np.full(self.base_map.shape, -1, dtype=np.int8)
        self.n_rays = n_rays
        self.max_range_m = max_range_m
        self.mode = "mapping"
        self.running = True
        self.saved = 0
        self.optimized = 0
        self.keyframes = 0
        self._last_kf: Optional[conv.Pose2D] = None
        # Tests replace this to say when scans stop fitting the map.
        self.hold = lambda: PaintHold()
        # Tests replace this to say SLAM has a jump awaiting its second look.
        self.jump = lambda: None

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        m = self.world.map
        return (
            int(math.floor((y - m.origin_y) / m.resolution + 1e-9)),
            int(math.floor((x - m.origin_x) / m.resolution + 1e-9)),
        )

    def reveal(self, pose: conv.Pose2D) -> None:
        """Mark every cell a 360-ray scan from ``pose`` passes (free) or hits."""
        m = self.world.map
        step = m.resolution / 2.0
        dist = np.arange(1, int(self.max_range_m / step) + 1) * step
        ang = pose.theta + 2.0 * math.pi * np.arange(self.n_rays) / self.n_rays
        rows = np.floor(
            (pose.y + np.outer(np.sin(ang), dist) - m.origin_y) / m.resolution + 1e-9
        )
        cols = np.floor(
            (pose.x + np.outer(np.cos(ang), dist) - m.origin_x) / m.resolution + 1e-9
        )
        rows, cols = rows.astype(int), cols.astype(int)
        h, w = self.base_map.shape
        oob = (rows < 0) | (rows >= h) | (cols < 0) | (cols >= w)
        rows, cols = np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)
        stop = oob | (self.base_map[rows, cols] >= 50)
        first = np.where(stop.any(axis=1), stop.argmax(axis=1), dist.size)
        j = np.arange(dist.size)[None, :]
        before = j < first[:, None]
        self.known[rows[before], cols[before]] = 0
        hit = (j == first[:, None]) & ~oob
        self.known[rows[hit], cols[hit]] = 100
        moved = self._last_kf is None or (
            math.hypot(pose.x - self._last_kf.x, pose.y - self._last_kf.y) >= 0.5
            or abs(conv.normalize_angle(pose.theta - self._last_kf.theta))
            >= math.radians(20)
        )
        if moved:
            self.keyframes += 1
            self._last_kf = conv.Pose2D(pose.x, pose.y, pose.theta)

    async def occupancy(self) -> OccGrid:
        m = self.world.map
        return OccGrid(
            cells=self.known.copy(),
            resolution=float(m.resolution),
            x_min=float(m.origin_x),
            y_min=float(m.origin_y),
        )

    async def status(self) -> SlamStatus:
        return SlamStatus(
            mode=self.mode, running=self.running, keyframes=self.keyframes
        )

    async def paint_hold(self) -> PaintHold:
        return self.hold()

    async def pending_jump(self):
        return self.jump()

    async def optimize(self) -> None:
        self.optimized += 1

    async def save_map(self) -> None:
        self.saved += 1


class FakeNav:
    def __init__(
        self,
        world: SimWorld,
        slam: FakeSlam,
        clock: FakeClock,
        *,
        speed_mps: float = 0.4,
        max_angular_rad_s: float = 0.5,
    ):
        self.world = world
        self.slam = slam
        self.clock = clock
        self.speed_mps = speed_mps
        self.max_angular_rad_s = max_angular_rad_s
        self.lose_pose = False
        # Navigation's reason for not trusting a pose it does have.
        self.untrusted = ""
        self.block_all = False
        self.cancels = 0
        self.legs = 0
        self._active = False
        self._state = "idle"
        self._goal: Optional[Tuple[float, float, float]] = None
        self._path: List[Tuple[float, float]] = []
        self._last_t = 0.0
        self._error = ""
        self._bfs_key: Optional[Tuple[int, int]] = None
        self._bfs_parent: Optional[np.ndarray] = None
        self._bfs_dist: Optional[np.ndarray] = None

    def _line_blocked(
        self, grid: np.ndarray, a: Tuple[float, float], b: Tuple[float, float]
    ) -> bool:
        m = self.world.map
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (m.resolution / 2.0)))
        for i in range(n + 1):
            t = i / n
            r, c = self.slam._cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not m.in_bounds(r, c) or grid[r, c] >= 50:
                return True
        return False

    def _bfs(self, start: Tuple[int, int]) -> None:
        if self._bfs_key == start:
            return
        grid = self.slam.base_map
        rows, cols = grid.shape
        parent = np.full((rows, cols), -1, dtype=np.int64)
        dist = np.full((rows, cols), -1, dtype=np.int64)
        r0, c0 = start
        if not (0 <= r0 < rows and 0 <= c0 < cols) or grid[r0, c0] >= 50:
            self._bfs_key, self._bfs_parent, self._bfs_dist = start, parent, dist
            return
        dist[r0, c0] = 0
        queue = deque([(r0, c0)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and dist[nr, nc] < 0
                    and grid[nr, nc] < 50
                ):
                    dist[nr, nc] = dist[r, c] + 1
                    parent[nr, nc] = r * cols + c
                    queue.append((nr, nc))
        self._bfs_key, self._bfs_parent, self._bfs_dist = start, parent, dist

    def _plan(self, x: float, y: float) -> Optional[List[Tuple[float, float]]]:
        p = self.world.get_pose()
        start = self.slam._cell(p.x, p.y)
        goal = self.slam._cell(x, y)
        self._bfs(start)
        rows, cols = self._bfs_dist.shape
        if (
            not (0 <= goal[0] < rows and 0 <= goal[1] < cols)
            or self._bfs_dist[goal] < 0
        ):
            return None
        m = self.world.map
        cells = []
        cur = goal
        while cur != start:
            cells.append(cur)
            lin = int(self._bfs_parent[cur])
            cur = (lin // cols, lin % cols)
        cells.reverse()
        path = [
            (
                m.origin_x + (c + 0.5) * m.resolution,
                m.origin_y + (r + 0.5) * m.resolution,
            )
            for r, c in cells
        ]
        if path:
            path[-1] = (x, y)
        else:
            path = [(x, y)]
        return path

    async def plan_to_point(self, x: float, y: float) -> Tuple[bool, float]:
        path = self._plan(x, y)
        if path is None:
            return False, 0.0
        p = self.world.get_pose()
        pts = [(p.x, p.y)] + path
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        return True, round(float(length), 6)

    def _start(self, x: float, y: float, theta: float) -> None:
        self.legs += 1
        self.last_theta = theta
        self._goal = (x, y, theta)
        self._last_t = self.clock.now()
        self._error = ""
        path = None if self.block_all else self._plan(x, y)
        if path is None:
            self._active, self._state, self._error = False, "failed", "no path"
            return
        self._path = path
        self._active, self._state = True, "active"
        self.slam.reveal(self.world.get_pose())

    async def navigate_to_point(self, x: float, y: float, theta: float) -> None:
        self._start(x, y, theta)

    async def cancel(self) -> None:
        self.cancels += 1
        if self._active:
            self._active, self._state = False, "canceled"

    def _advance(self) -> None:
        now = self.clock.now()
        dt, self._last_t = max(0.0, now - self._last_t), now
        if not self._active or self._goal is None or dt <= 0.0:
            return
        p = self.world.get_pose()
        x, y, heading = p.x, p.y, p.theta
        budget = self.speed_mps * dt
        while budget > 1e-9 and self._path:
            tx, ty = self._path[0]
            dist = math.hypot(tx - x, ty - y)
            if dist <= budget:
                nx, ny = tx, ty
                budget -= dist
                self._path.pop(0)
            else:
                nx, ny = x + (tx - x) / dist * budget, y + (ty - y) / dist * budget
                budget = 0.0
            if dist > 1e-9:
                heading = math.atan2(ty - y, tx - x)
            if self._line_blocked(self.world.map.grid, (x, y), (nx, ny)):
                self.world.teleport(conv.Pose2D(x, y, heading))
                self._active, self._state, self._error = False, "failed", "blocked"
                return
            x, y = nx, ny
        self.world.teleport(conv.Pose2D(x, y, heading))
        self.slam.reveal(conv.Pose2D(x, y, heading))
        gx, gy, gtheta = self._goal
        if not self._path and math.hypot(gx - x, gy - y) < 0.05:
            self.world.teleport(conv.Pose2D(x, y, gtheta))
            self._active, self._state = False, "succeeded"

    async def status(self) -> NavStatus:
        if not self._active:
            self.slam.reveal(self.world.get_pose())
        self._advance()
        p = self.world.get_pose()
        return NavStatus(
            active=self._active,
            state=self._state,
            pose=None if self.lose_pose else Pose(p.x, p.y, p.theta),
            localized=not self.lose_pose and not self.untrusted,
            max_angular_rad_s=self.max_angular_rad_s,
            robot_radius_m=0.3,
            error=self._error,
            not_localized_reason=self.untrusted,
        )


def known_fraction(slam: FakeSlam, world: SimWorld) -> float:
    truth_free = slam.base_map == 0
    if not truth_free.any():
        return 0.0
    return float(
        np.count_nonzero(truth_free & (slam.known == 0)) / np.count_nonzero(truth_free)
    )

"""Adaptive Monte Carlo localization, as the benchmark's AMCL baseline runs.

Written from the published algorithms (Thrun, Burgard and Fox, Probabilistic
Robotics: the odometry motion model and the likelihood-field sensor model;
Fox, NeurIPS 2001: KLD-sampling), with the widely used defaults for its
parameters, not from any implementation's code. It is here to check the
replay harness against the benchmark paper's AMCL numbers, and to try it
under the stack's localization watchdog.

- Motion: the odometry change is split into a turn, a straight move and a
  turn, and each particle samples its own noisy version (``alpha1``..``4``).
- Filter updates run only after the robot has moved ``update_min_d`` or
  turned ``update_min_a``; between them the estimate follows odometry.
- Weight: each of ``max_beams`` beams scores ``z_hit * exp(-d^2 / 2 sigma^2)
  + z_rand / range_max`` (``d`` the distance from its endpoint to the nearest
  wall, capped at ``max_dist``); the cubes of those scores are summed.
- Resampling: KLD-sampling chooses how many particles to keep, between
  ``min_particles`` and ``max_particles``.
- Estimate: the weighted mean of the particle cluster with the most weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.geom import conversions as conv
from src.nav.localize.global_localize import OccupancyMap


@dataclass
class AmclParams:
    # The benchmark paper's particle range and beam count; the rest are the
    # commonly shipped defaults.
    min_particles: int = 1000
    max_particles: int = 10000
    max_beams: int = 100
    z_hit: float = 0.5
    z_rand: float = 0.5
    sigma_hit: float = 0.2
    max_dist: float = 2.0
    alpha1: float = 0.2  # rotation noise from rotation
    alpha2: float = 0.2  # rotation noise from translation
    alpha3: float = 0.2  # translation noise from translation
    alpha4: float = 0.2  # translation noise from rotation
    update_min_d: float = 0.25
    update_min_a: float = 0.2
    kld_err: float = 0.05
    kld_z: float = 0.99
    bin_xy: float = 0.5
    bin_th: float = math.radians(10.0)


def distance_field(occ: OccupancyMap, max_dist: float) -> np.ndarray:
    """Distance in meters from each cell to the nearest occupied cell, capped."""
    walls = np.asarray(occ.grid) > 50
    res = float(occ.resolution)
    # Past 1 m the hit term is below 1e-5 at the usual 0.2 m sigma: every
    # cell that far out scores as if at ``max_dist``.
    reach = int(math.ceil(min(max_dist, 1.0) / res))
    h, w = walls.shape
    best = np.full(walls.shape, float(max_dist), dtype=np.float32)
    # Exact within ``reach`` cells: every offset at most that far away.
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            d = math.hypot(dx, dy) * res
            if d >= max_dist:
                continue
            src = walls[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
            dst = best[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)]
            np.minimum(dst, np.where(src, d, max_dist), out=dst)
    return best


class Amcl:
    def __init__(
        self,
        occ: OccupancyMap,
        start: conv.Pose2D,
        start_std=(0.2, 0.2, math.radians(10.0)),
        *,
        params: Optional[AmclParams] = None,
        seed: int = 0,
        field: Optional[np.ndarray] = None,
    ) -> None:
        self.p = params or AmclParams()
        self.rng = np.random.default_rng(seed)
        self.occ = occ
        self.res = float(occ.resolution)
        dist = field if field is not None else distance_field(occ, self.p.max_dist)
        self.dist = dist
        self._init(start, start_std, self.p.max_particles)
        self._odom_at_update: Optional[np.ndarray] = None
        self._odom_now: Optional[np.ndarray] = None
        self._estimate = start
        self.updates = 0

    def _init(self, pose: conv.Pose2D, std, n: int) -> None:
        self.x = pose.x + self.rng.normal(0.0, std[0], n)
        self.y = pose.y + self.rng.normal(0.0, std[1], n)
        self.th = pose.theta + self.rng.normal(0.0, std[2], n)
        self.w = np.full(n, 1.0 / n)

    def scatter(self, n: Optional[int] = None) -> None:
        """Spread particles over all free space in the map, any heading (no idea where)."""
        n = int(n or self.p.max_particles)
        grid = np.asarray(self.occ.grid)
        rows, cols = np.nonzero((grid >= 0) & (grid < 50))
        pick = self.rng.integers(0, rows.size, n)
        self.x = self.occ.origin_x + (cols[pick] + self.rng.random(n)) * self.res
        self.y = self.occ.origin_y + (rows[pick] + self.rng.random(n)) * self.res
        self.th = self.rng.uniform(-math.pi, math.pi, n)
        self.w = np.full(n, 1.0 / n)

    def converged(self) -> Optional[tuple]:
        """``(pose, share, spread_m, spread_deg)`` of the heaviest cluster, or None."""
        bins = self._bins(self.x, self.y, self.th)
        keys, inverse = np.unique(bins, return_inverse=True)
        if keys.size > 400:
            return None  # still spread over the map
        pose = self._cluster_mean()
        near = (np.hypot(self.x - pose.x, self.y - pose.y) < 1.0) & (
            np.abs(
                np.arctan2(np.sin(self.th - pose.theta), np.cos(self.th - pose.theta))
            )
            < math.radians(30.0)
        )
        share = float(self.w[near].sum() / self.w.sum())
        if not near.any():
            return None
        spread = float(math.hypot(self.x[near].std(), self.y[near].std()))
        dth = np.arctan2(
            np.sin(self.th[near] - pose.theta), np.cos(self.th[near] - pose.theta)
        )
        return pose, share, spread, float(np.degrees(dth.std()))

    def reset(self, pose: conv.Pose2D, std=(0.2, 0.2, math.radians(10.0))) -> None:
        """Start again around ``pose`` (what an initial-pose message does)."""
        self._init(pose, std, self.p.max_particles)
        self._estimate = pose
        if self._odom_now is not None:
            self._odom_at_update = self._odom_now.copy()

    # -- odometry ----------------------------------------------------------------
    def odometry(self, odom: conv.Pose2D) -> bool:
        """Record the latest odometry; True when the filter should update."""
        cur = np.array([odom.x, odom.y, odom.theta])
        self._odom_now = cur
        if self._odom_at_update is None:
            self._odom_at_update = cur
            return False
        d = cur - self._odom_at_update
        moved = math.hypot(d[0], d[1])
        turned = abs(conv.normalize_angle(d[2]))
        return moved >= self.p.update_min_d or turned >= self.p.update_min_a

    def _move(self) -> None:
        a, b = self._odom_at_update, self._odom_now
        dx, dy = b[0] - a[0], b[1] - a[1]
        trans = math.hypot(dx, dy)
        rot1 = conv.normalize_angle(math.atan2(dy, dx) - a[2]) if trans > 0.01 else 0.0
        rot2 = conv.normalize_angle(b[2] - a[2] - rot1)
        # Driving backwards: measure the turns from the reversed direction.
        r1 = min(abs(rot1), abs(conv.normalize_angle(rot1 - math.pi)))
        r2 = min(abs(rot2), abs(conv.normalize_angle(rot2 - math.pi)))
        p, n = self.p, self.x.size
        rot1_hat = rot1 - self.rng.normal(
            0.0, math.sqrt(p.alpha1 * r1**2 + p.alpha2 * trans**2), n
        )
        trans_hat = trans - self.rng.normal(
            0.0, math.sqrt(p.alpha3 * trans**2 + p.alpha4 * (r1**2 + r2**2)), n
        )
        rot2_hat = rot2 - self.rng.normal(
            0.0, math.sqrt(p.alpha1 * r2**2 + p.alpha2 * trans**2), n
        )
        self.x = self.x + trans_hat * np.cos(self.th + rot1_hat)
        self.y = self.y + trans_hat * np.sin(self.th + rot1_hat)
        self.th = np.arctan2(
            np.sin(self.th + rot1_hat + rot2_hat), np.cos(self.th + rot1_hat + rot2_hat)
        )

    # -- sensor model --------------------------------------------------------------
    def _distances(self, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        cols = np.floor((wx - self.occ.origin_x) / self.res).astype(np.int64)
        rows = np.floor((wy - self.occ.origin_y) / self.res).astype(np.int64)
        h, w = self.dist.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.full(wx.shape, self.p.max_dist, dtype=np.float32)
        out[inside] = self.dist[rows[inside], cols[inside]]
        return out

    def _weigh(self, scan: conv.LaserScan2D) -> None:
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = float(scan.angle_min) + np.arange(ranges.size) * float(
            scan.angle_increment
        )
        range_max = float(scan.range_max) if math.isfinite(scan.range_max) else 30.0
        ok = (
            np.isfinite(ranges)
            & (ranges >= float(scan.range_min))
            & (ranges < range_max)
        )
        r, a = ranges[ok], angles[ok]
        if r.size == 0:
            return
        if r.size > self.p.max_beams:
            pick = np.linspace(0, r.size - 1, self.p.max_beams).astype(np.int64)
            r, a = r[pick], a[pick]
        bx, by = r * np.cos(a), r * np.sin(a)
        c, s = np.cos(self.th)[:, None], np.sin(self.th)[:, None]
        wx = self.x[:, None] + c * bx[None, :] - s * by[None, :]
        wy = self.y[:, None] + s * bx[None, :] + c * by[None, :]
        d = self._distances(wx, wy)
        pz = (
            self.p.z_hit * np.exp(-(d**2) / (2.0 * self.p.sigma_hit**2))
            + self.p.z_rand / range_max
        )
        self.w = self.w * (1.0 + (pz**3).sum(axis=1))
        total = float(self.w.sum())
        self.w = (
            self.w / total if total > 0 else np.full(self.w.size, 1.0 / self.w.size)
        )

    # -- resampling ------------------------------------------------------------------
    def _bins(self, x, y, th) -> np.ndarray:
        bx = np.floor(x / self.p.bin_xy).astype(np.int64)
        by = np.floor(y / self.p.bin_xy).astype(np.int64)
        bt = np.floor((th + math.pi) / self.p.bin_th).astype(np.int64)
        return (bx * 1_000_003 + by) * 1_000 + bt

    def _kld_count(self, k: int) -> int:
        if k <= 1:
            return self.p.min_particles
        a = 2.0 / (9.0 * (k - 1))
        z = {0.99: 2.326, 0.95: 1.645}.get(self.p.kld_z, 2.326)
        n = (k - 1) / (2.0 * self.p.kld_err) * (1.0 - a + math.sqrt(a) * z) ** 3
        return int(min(self.p.max_particles, max(self.p.min_particles, math.ceil(n))))

    def _resample(self) -> None:
        cum = np.cumsum(self.w)
        cum[-1] = 1.0
        # KLD-sampling: draw until the number of occupied bins says enough.
        draw = self.rng.random(self.p.max_particles)
        picks = np.minimum(np.searchsorted(cum, draw, side="right"), self.w.size - 1)
        bins = self._bins(self.x[picks], self.y[picks], self.th[picks])
        _, first_seen = np.unique(bins, return_index=True)
        seen_by = np.zeros(picks.size, dtype=np.int64)
        seen_by[np.sort(first_seen)] = 1
        k_after = np.cumsum(seen_by)
        need = np.array([self._kld_count(int(k)) for k in k_after])
        enough = np.nonzero(np.arange(1, picks.size + 1) >= need)[0]
        n = int(enough[0] + 1) if enough.size else picks.size
        n = max(n, self.p.min_particles)
        picks = picks[:n]
        self.x, self.y, self.th = self.x[picks], self.y[picks], self.th[picks]
        self.w = np.full(n, 1.0 / n)

    # -- estimate ----------------------------------------------------------------------
    def _cluster_mean(self) -> conv.Pose2D:
        bins = self._bins(self.x, self.y, self.th)
        keys, inverse = np.unique(bins, return_inverse=True)
        # Group neighbouring occupied bins into clusters (flood fill over bins).
        bx = keys // 1_000 // 1_000_003
        by = (keys // 1_000) % 1_000_003
        bt = keys % 1_000
        index = {
            (int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(zip(bx, by, bt))
        }
        label = -np.ones(keys.size, dtype=np.int64)
        nbins = int(round(2 * math.pi / self.p.bin_th))
        cur = 0
        for i in range(keys.size):
            if label[i] >= 0:
                continue
            stack = [i]
            label[i] = cur
            while stack:
                j = stack.pop()
                a, b, c = int(bx[j]), int(by[j]), int(bt[j])
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            k = index.get((a + da, b + db, (c + dc) % nbins))
                            if k is not None and label[k] < 0:
                                label[k] = cur
                                stack.append(k)
            cur += 1
        part = label[inverse]
        weight = np.bincount(part, weights=self.w, minlength=cur)
        best = int(np.argmax(weight))
        m = part == best
        ww = self.w[m] / self.w[m].sum()
        mx, my = float((self.x[m] * ww).sum()), float((self.y[m] * ww).sum())
        mt = math.atan2(
            float((np.sin(self.th[m]) * ww).sum()),
            float((np.cos(self.th[m]) * ww).sum()),
        )
        return conv.Pose2D(mx, my, mt)

    # -- one step --------------------------------------------------------------------------
    def update(self, scan: conv.LaserScan2D) -> None:
        """Move by the odometry since the last update, weigh, resample, re-estimate."""
        self._move()
        self._weigh(scan)
        self._estimate = self._cluster_mean()
        self._resample()
        self._odom_at_update = self._odom_now.copy()
        self.updates += 1

    def estimate(self) -> conv.Pose2D:
        """The last filter estimate, carried forward by odometry since then."""
        if self._odom_at_update is None or self._odom_now is None:
            return self._estimate
        a, b = self._odom_at_update, self._odom_now
        delta = conv.compose_poses(
            conv.invert_pose(conv.Pose2D(*map(float, a))), conv.Pose2D(*map(float, b))
        )
        return conv.compose_poses(self._estimate, delta)

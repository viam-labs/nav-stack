"""A Monte Carlo localizer with expansion resetting, after Ueda's emcl2.

Written from the published algorithm and the behaviour of the emcl2 package
(R. Ueda; expansion resetting from Ueda et al., IROS 2004), not from its
code. It is here to be scored against the stack's scan-matching tracker on
the same recordings, alone and with that tracker's scan matcher refining its
estimate.

- Map: a likelihood field. Each occupied cell scores 1, falling off linearly
  with Chebyshev distance to 0 at ``likelihood_max_dist``.
- Weight: a particle's weight is multiplied by the sum of the field at its
  beam endpoints. A return in free space or unknown space adds nothing, so
  people and moved furniture cost reward but never veto.
- Reset: a sample of particles is checked for beams that run through a wall
  and on into open space ("penetrating"). A particle is wrong when a run of
  consecutive penetrating beams spans ``range_threshold`` radians. When fewer
  than ``alpha_threshold`` of the sample are right, every particle is
  scattered within ``expansion_radius_*`` of where it was, and reweighted.
- Estimate: the mean of the particles after resampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.geom import conversions as conv
from src.nav.localize.global_localize import OccupancyMap


@dataclass
class EmclParams:
    # As emcl2's launch file ships them (its node defaults differ: 0
    # particles, range_threshold 0.1).
    particles: int = 500
    likelihood_max_dist: float = 0.2
    alpha_threshold: float = 0.5
    expansion_radius_position: float = 0.1
    expansion_radius_orientation: float = 0.2
    extraction_rate: float = 0.1
    range_threshold: float = 0.3
    scan_increment: int = 1
    odom_fw_dev_per_fw: float = 0.19
    odom_fw_dev_per_rot: float = 0.0001
    odom_rot_dev_per_fw: float = 0.13
    odom_rot_dev_per_rot: float = 0.2


def likelihood_field(occ: OccupancyMap, max_dist: float) -> np.ndarray:
    """1 on occupied cells, falling linearly with Chebyshev distance to 0."""
    grid = np.asarray(occ.grid)
    walls = grid > 50
    reach = int(math.ceil(float(max_dist) / float(occ.resolution)))
    field = walls.astype(np.float32)
    seen = walls.copy()
    for k in range(1, reach + 1):
        grown = seen.copy()
        grown[1:, :] |= seen[:-1, :]
        grown[:-1, :] |= seen[1:, :]
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
        new = grown & ~seen
        field[new] = 1.0 - k / reach
        seen = grown
    peak = float(field.max())
    return field / peak if peak > 0 else field


class Emcl:
    """Particles in the map frame, moved by odometry and weighed by scans."""

    def __init__(
        self,
        occ: OccupancyMap,
        start: conv.Pose2D,
        start_std=(0.2, 0.2, math.radians(10.0)),
        *,
        params: Optional[EmclParams] = None,
        seed: int = 0,
    ) -> None:
        self.p = params or EmclParams()
        self.rng = np.random.default_rng(seed)
        self.occ = occ
        self.field = likelihood_field(occ, self.p.likelihood_max_dist)
        self.res = float(occ.resolution)
        n = int(self.p.particles)
        self.x = start.x + self.rng.normal(0.0, start_std[0], n)
        self.y = start.y + self.rng.normal(0.0, start_std[1], n)
        self.th = start.theta + self.rng.normal(0.0, start_std[2], n)
        self.w = np.full(n, 1.0 / n)
        self._prev_odom: Optional[np.ndarray] = None
        self._check_offset = 0
        self.resets = 0
        self.alpha = 1.0

    # -- motion ---------------------------------------------------------------
    def move(self, odom: conv.Pose2D) -> None:
        """Apply the odometry change since the last call, with emcl2's noise."""
        cur = np.array([odom.x, odom.y, odom.theta])
        if self._prev_odom is None:
            self._prev_odom = cur
            return
        dx, dy = cur[0] - self._prev_odom[0], cur[1] - self._prev_odom[1]
        dth = conv.normalize_angle(cur[2] - self._prev_odom[2])
        if abs(dx) < 1e-3 and abs(dy) < 1e-3 and abs(dth) < 1e-3:
            return  # too small: let it add up
        length = math.hypot(dx, dy)
        direction = math.atan2(dy, dx) - self._prev_odom[2]
        p = self.p
        fw_dev = math.sqrt(
            length * p.odom_fw_dev_per_fw**2 + abs(dth) * p.odom_fw_dev_per_rot**2
        )
        rot_dev = math.sqrt(
            length * p.odom_rot_dev_per_fw**2 + abs(dth) * p.odom_rot_dev_per_rot**2
        )
        n = self.x.size
        fw = length + self.rng.normal(0.0, fw_dev, n)
        rot = self.rng.normal(0.0, rot_dev, n)
        heading = direction + rot + self.th
        self.x = self.x + fw * np.cos(heading)
        self.y = self.y + fw * np.sin(heading)
        self.th = np.arctan2(np.sin(self.th + dth + rot), np.cos(self.th + dth + rot))
        self._prev_odom = cur

    # -- map lookups ------------------------------------------------------------
    def _lookup(self, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        cols = np.floor((wx - self.occ.origin_x) / self.res).astype(np.int64)
        rows = np.floor((wy - self.occ.origin_y) / self.res).astype(np.int64)
        h, w = self.field.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.zeros(wx.shape, dtype=np.float32)
        out[inside] = self.field[rows[inside], cols[inside]]
        return out

    @staticmethod
    def _beams(scan: conv.LaserScan2D, increment: int):
        ranges = np.asarray(scan.ranges, dtype=float)
        idx = np.arange(ranges.size)[:: max(1, int(increment))]
        r = ranges[idx]
        ok = (
            np.isfinite(r) & (r >= float(scan.range_min)) & (r <= float(scan.range_max))
        )
        angles = float(scan.angle_min) + idx * float(scan.angle_increment)
        return r[ok], angles[ok]

    def _likelihoods(self, r: np.ndarray, a: np.ndarray) -> np.ndarray:
        bx, by = r * np.cos(a), r * np.sin(a)  # endpoints in the body frame
        c, s = np.cos(self.th)[:, None], np.sin(self.th)[:, None]
        wx = self.x[:, None] + c * bx[None, :] - s * by[None, :]
        wy = self.y[:, None] + s * bx[None, :] + c * by[None, :]
        return self._lookup(wx, wy).sum(axis=1)

    def _penetrating_runs(
        self, idx: np.ndarray, r: np.ndarray, a: np.ndarray, inc: float
    ):
        """True for each particle in ``idx`` with a run of beams through a wall."""
        steps = np.floor((r - 1e-9) / self.res).astype(
            np.int64
        )  # samples at res, 2res, .. < r
        beam = np.repeat(np.arange(r.size), steps)
        t = (
            np.arange(beam.size) - np.repeat(np.cumsum(steps) - steps, steps) + 1
        ) * self.res
        px, py = t * np.cos(a[beam]), t * np.sin(a[beam])
        starts = np.cumsum(steps) - steps
        has = steps > 0
        wrong = np.zeros(idx.size, dtype=bool)
        need = int(math.ceil(self.p.range_threshold / max(inc, 1e-9) - 1e-9))
        for k, i in enumerate(idx):
            c, s = math.cos(self.th[i]), math.sin(self.th[i])
            val = self._lookup(self.x[i] + c * px - s * py, self.y[i] + s * px + c * py)
            hit = (val > 0.99).astype(np.int64)
            seen = np.cumsum(hit)
            before = np.zeros(r.size, dtype=np.int64)
            before[has] = seen[starts[has]] - hit[starts[has]]
            hits_so_far = seen - np.repeat(before, steps)
            through = (val == 0.0) & (hits_so_far > 0)
            pen = np.zeros(r.size, dtype=bool)
            if beam.size:
                pen[has] = np.logical_or.reduceat(through, starts[has])
            # Longest run of consecutive penetrating beams.
            run = best = 0
            for flag in pen:
                run = run + 1 if flag else 0
                best = max(best, run)
            wrong[k] = best >= need
        return wrong

    # -- sensor update ---------------------------------------------------------
    def weigh(self, scan: conv.LaserScan2D) -> None:
        r, a = self._beams(scan, self.p.scan_increment)
        if r.size == 0:
            return
        self.w = self.w * self._likelihoods(r, a)
        n = self.x.size
        skip = max(1, int(n * self.p.extraction_rate))
        idx = np.arange(self._check_offset % skip, n, skip)
        self._check_offset += 1
        inc = float(scan.angle_increment) * max(1, int(self.p.scan_increment))
        wrong = self._penetrating_runs(idx, r, a, abs(inc))
        self.alpha = float(1.0 - wrong.mean()) if idx.size else 1.0
        if self.alpha < self.p.alpha_threshold:
            self._expand()
            self.w = self.w * self._likelihoods(r, a)
            self.resets += 1
        total = float(self.w.sum())
        if total > 1e-6:
            self.w = self.w / total
            self._resample()
        else:
            self.w = np.full(n, 1.0 / n)

    def _expand(self) -> None:
        n = self.x.size
        length = 2.0 * (self.rng.random(n) - 0.5) * self.p.expansion_radius_position
        direction = self.rng.uniform(-math.pi, math.pi, n)
        self.x = self.x + length * np.cos(direction)
        self.y = self.y + length * np.sin(direction)
        self.th = (
            self.th
            + 2.0 * (self.rng.random(n) - 0.5) * self.p.expansion_radius_orientation
        )
        self.w = np.full(n, 1.0 / n)

    def _resample(self) -> None:
        n = self.x.size
        cum = np.cumsum(self.w)
        cum[-1] = 1.0
        picks = np.searchsorted(
            cum, self.rng.random() / n + np.arange(n) / n, side="right"
        )
        picks = np.minimum(picks, n - 1)
        self.x, self.y, self.th = self.x[picks], self.y[picks], self.th[picks]
        self.w = np.full(n, 1.0 / n)

    # -- estimate ----------------------------------------------------------------
    def estimate(self) -> conv.Pose2D:
        """Mean of the particles; heading averaged on whichever side of ±pi is tighter."""
        th = np.arctan2(np.sin(self.th), np.cos(self.th))
        flipped = np.arctan2(np.sin(th + math.pi), np.cos(th + math.pi))
        if np.var(flipped) < np.var(th):
            mean_th = conv.normalize_angle(float(flipped.mean()) - math.pi)
        else:
            mean_th = float(th.mean())
        return conv.Pose2D(float(self.x.mean()), float(self.y.mean()), mean_th)

    def spread(self) -> float:
        """Standard deviation of the particles' positions, in meters."""
        return float(math.hypot(self.x.std(), self.y.std()))

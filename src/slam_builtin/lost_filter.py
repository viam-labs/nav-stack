"""A particle filter over the whole map, for a robot that has no pose or the wrong one.

The scan matcher tracks a pose it is given; it cannot find the robot from
nothing, and one scan is not enough to tell look-alike rooms apart. This
filter can, given some driving:

- **finding** (no pose yet): particles over every free cell follow odometry
  and are weighed against each scan (likelihood field). Once one cluster
  holds nearly all the weight and has stayed tight over
  ``CONFIRM_TRAVEL_M`` of driving, its pose is handed to the tracker.
- **confirming**: a hand-over is provisional for ``PROVISIONAL_M`` of
  driving. If the scans stop fitting the walls at the tracker's pose for
  ``PROVISIONAL_BAD_S`` in that time, it is withdrawn and the search starts
  again.
- **watching**: once confirmed, the scans' fit at the tracker's pose is
  checked every ``FIT_EVERY_S``. After ``BAD_FIT_S`` without a fit, the
  filter searches the whole map again in the background while the tracker
  carries on (**checking**). If the search settles where the tracker is,
  the room has changed and nothing moves. If it settles elsewhere, both
  places are scored against the last ``HISTORY_S`` of scans, each carried
  back along odometry, and the tracker is moved only when the new place
  explains them clearly better.

An optional ``camera`` judge can hold back a hand-over or a move to a place
the camera does not recognize; the filter then starts over with most of its
particles around the place the camera does recognize.

The particle filter is adaptive Monte Carlo localization written from the
published algorithms (Thrun, Burgard and Fox, Probabilistic Robotics: the
odometry motion model and the likelihood-field sensor model; Fox, NeurIPS
2001: KLD-sampling).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv
from ..nav.localize.global_localize import OccupancyMap
from . import scan_match

# ``camera(pose, now)`` -> (why it rejects ``pose``, or None; poses of the
# place it recognizes, possibly empty).
CameraJudge = Callable[[conv.Pose2D, float], Tuple[Optional[str], List[conv.Pose2D]]]


@dataclass
class ParticleParams:
    min_particles: int = 1000
    max_particles: int = 20000
    max_beams: int = 100
    # Mostly the hit term: a robot that is lost needs walls, not tolerance.
    z_hit: float = 0.95
    z_rand: float = 0.05
    sigma_hit: float = 0.2
    max_dist: float = 2.0
    alpha1: float = 0.2  # rotation noise from rotation
    alpha2: float = 0.2  # rotation noise from translation
    alpha3: float = 0.2  # translation noise from translation
    alpha4: float = 0.2  # translation noise from rotation
    update_min_d: float = 0.1
    update_min_a: float = 0.1
    kld_err: float = 0.05
    kld_z: float = 0.99
    bin_xy: float = 0.5
    bin_th: float = math.radians(10.0)
    # More occupied bins than this: still spread over the map, no estimate.
    max_cluster_bins: int = 400


def distance_field(
    occ: OccupancyMap, max_dist: float, *, exact_m: float = 0.6
) -> np.ndarray:
    """Distance in meters from each cell to the nearest occupied cell, capped.

    Exact within ``exact_m`` of a wall; farther cells read ``max_dist``. At
    the 0.2 m sigma, a beam ending 0.6 m from a wall scores about 1% of a hit,
    and the filter sums cubes of those scores, so past that every cell
    weighs the same. The cost grows with the square of ``exact_m``.
    """
    walls = np.asarray(occ.grid) > 50
    res = float(occ.resolution)
    reach = int(math.ceil(min(max_dist, exact_m) / res))
    h, w = walls.shape
    best = np.full(walls.shape, float(max_dist), dtype=np.float32)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            d = math.hypot(dx, dy) * res
            if d >= max_dist or d > exact_m + 1e-9:
                continue
            src = walls[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
            dst = best[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)]
            np.minimum(dst, np.where(src, d, max_dist), out=dst)
    return best


class ParticleFilter:
    """Adaptive Monte Carlo localization in an occupancy map."""

    def __init__(
        self,
        occ: OccupancyMap,
        *,
        params: Optional[ParticleParams] = None,
        seed: int = 0,
        field: Optional[np.ndarray] = None,
    ) -> None:
        self.p = params or ParticleParams()
        self.rng = np.random.default_rng(seed)
        self.occ = occ
        self.res = float(occ.resolution)
        self.dist = field if field is not None else distance_field(occ, self.p.max_dist)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.th = np.zeros(0)
        self.w = np.zeros(0)
        self._odom_at_update: Optional[np.ndarray] = None
        self._odom_now: Optional[np.ndarray] = None
        self._estimate: Optional[conv.Pose2D] = None
        self.updates = 0

    @property
    def particles(self) -> int:
        return int(self.x.size)

    def scatter(self, n: Optional[int] = None) -> None:
        """Spread particles over all free space in the map, any heading."""
        n = int(n or self.p.max_particles)
        grid = np.asarray(self.occ.grid)
        rows, cols = np.nonzero((grid >= 0) & (grid < 50))
        if rows.size == 0:
            return
        pick = self.rng.integers(0, rows.size, n)
        self.x = self.occ.origin_x + (cols[pick] + self.rng.random(n)) * self.res
        self.y = self.occ.origin_y + (rows[pick] + self.rng.random(n)) * self.res
        self.th = self.rng.uniform(-math.pi, math.pi, n)
        self.w = np.full(n, 1.0 / n)
        self._estimate = None

    def seed(
        self,
        poses: Sequence[conv.Pose2D],
        *,
        share: float = 0.7,
        std=(0.5, 0.5, math.radians(25.0)),
    ) -> None:
        """Start over with ``share`` of the particles around ``poses``, the rest over the map.

        The particles around a hint still have to earn their weight from the
        scans, and the ones over the map keep the filter from betting
        everything on the hint.
        """
        n = int(self.p.max_particles)
        k = min(n, int(round(n * float(share)))) if poses else 0
        self.scatter(n)
        if k == 0 or self.x.size < k:
            return
        pick = self.rng.integers(0, len(poses), k)
        px = np.array([p.x for p in poses])[pick]
        py = np.array([p.y for p in poses])[pick]
        pth = np.array([p.theta for p in poses])[pick] + self.rng.normal(0.0, std[2], k)
        self.x[:k] = px + self.rng.normal(0.0, std[0], k)
        self.y[:k] = py + self.rng.normal(0.0, std[1], k)
        self.th[:k] = np.arctan2(np.sin(pth), np.cos(pth))

    # -- odometry ------------------------------------------------------------
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

    # -- sensor model ----------------------------------------------------------
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

    # -- resampling --------------------------------------------------------------
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

    # -- estimate ------------------------------------------------------------------
    def _cluster_mean(self, keys: np.ndarray, inverse: np.ndarray) -> conv.Pose2D:
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

    def _occupied_bins(self, limit: int) -> Optional[tuple]:
        """``(keys, inverse)`` of the occupied bins, or None when there are more than ``limit``."""
        bins = self._bins(self.x, self.y, self.th)
        keys, inverse = np.unique(bins, return_inverse=True)
        if keys.size > limit:
            return None
        return keys, inverse

    def converged(self) -> Optional[tuple]:
        """``(pose, share, spread_m, spread_deg)`` of the heaviest cluster, or None."""
        if self.x.size == 0:
            return None
        gathered = self._occupied_bins(self.p.max_cluster_bins)
        if gathered is None:
            return None  # still spread over the map
        pose = self._cluster_mean(*gathered)
        near = (np.hypot(self.x - pose.x, self.y - pose.y) < 1.0) & (
            np.abs(
                np.arctan2(np.sin(self.th - pose.theta), np.cos(self.th - pose.theta))
            )
            < math.radians(30.0)
        )
        if not near.any():
            return None
        share = float(self.w[near].sum() / self.w.sum())
        spread = float(math.hypot(self.x[near].std(), self.y[near].std()))
        dth = np.arctan2(
            np.sin(self.th[near] - pose.theta), np.cos(self.th[near] - pose.theta)
        )
        return pose, share, spread, float(np.degrees(dth.std()))

    def update(self, scan: conv.LaserScan2D) -> None:
        """Move by the odometry since the last update, weigh, estimate, resample."""
        if self.x.size == 0 or self._odom_now is None:
            return
        self._move()
        self._weigh(scan)
        # Spread over the map, the cluster search costs the most and says
        # nothing: there is no estimate until the particles start to gather.
        gathered = self._occupied_bins(4 * self.p.max_cluster_bins)
        self._estimate = self._cluster_mean(*gathered) if gathered is not None else None
        self._resample()
        self._odom_at_update = self._odom_now.copy()
        self.updates += 1

    def estimate(self) -> Optional[conv.Pose2D]:
        """The last estimate, carried forward by odometry since then."""
        if self._estimate is None:
            return None
        if self._odom_at_update is None or self._odom_now is None:
            return self._estimate
        a, b = self._odom_at_update, self._odom_now
        delta = conv.compose_poses(
            conv.invert_pose(conv.Pose2D(*map(float, a))), conv.Pose2D(*map(float, b))
        )
        return conv.compose_poses(self._estimate, delta)


@dataclass
class Handover:
    """A pose for the tracker, and why."""

    pose: conv.Pose2D
    why: str


FINDING = "finding"
CONFIRMING = "confirming"
WATCHING = "watching"
CHECKING = "checking"


class LostFilter:
    """Finds the robot with a particle filter and hands its pose to the tracker (module doc)."""

    # The cloud stays about 0.2 m and 8 degrees wide once it has found the
    # robot (the odometry noise keeps it so); the tracker takes it from there.
    SHARE = 0.9
    SPREAD_M = 0.3
    SPREAD_DEG = 10.0
    CONFIRM_UPDATES = 3
    # Rooms that look alike fool one scan; driving out of them does not.
    CONFIRM_TRAVEL_M = 1.5
    # A hand-over is withdrawn when the scans stop fitting this soon after it.
    PROVISIONAL_M = 2.0
    PROVISIONAL_BAD_S = 3.0
    # Scans that do not sit on the walls at the tracker's pose for this long
    # start a background check.
    BAD_FIT = 0.25
    BAD_FIT_S = 8.0
    FIT_EVERY_S = 1.0
    # A check that has not settled in this long is dropped; after a check,
    # wait this long before the next.
    CHECK_MAX_S = 90.0
    CHECK_COOLDOWN_S = 30.0
    # The searched place replaces the tracker's only when it explains the
    # recent scans this much better, on average.
    SWITCH_MARGIN = 0.1
    SAME_PLACE_M = 1.0
    SAME_PLACE_DEG = 20.0
    HISTORY_S = 10.0
    HISTORY_EVERY_S = 0.5
    LOG_LINES = 50

    def __init__(
        self,
        occ: OccupancyMap,
        *,
        has_pose: bool,
        camera: Optional[CameraJudge] = None,
        params: Optional[ParticleParams] = None,
        seed: int = 0,
        field: Optional[np.ndarray] = None,
        clock_origin: float = 0.0,
    ) -> None:
        self.occ = occ
        self.pf = ParticleFilter(occ, params=params, seed=seed, field=field)
        self.camera = camera
        self.phase = WATCHING if has_pose else FINDING
        if not has_pose:
            self.pf.scatter()
        self._agree = 0
        self._agree_from: Optional[np.ndarray] = None
        self._bad_since: Optional[float] = None
        self._last_fit_at = -math.inf
        self._last_fit: Optional[float] = None
        self._check_started = 0.0
        self._quiet_until = -math.inf
        self._confirm_from: Optional[np.ndarray] = None
        self._history: Deque[tuple] = deque()  # (now, odom (3,), scan)
        self._last_history_at = -math.inf
        self._clock_origin = float(clock_origin)
        self.handovers = 0
        self.withdrawn = 0
        self.log: Deque[str] = deque(maxlen=self.LOG_LINES)
        self.log_count = 0  # lines ever logged, for a reader following ``log``
        self.map_id: Optional[int] = None  # the owner's tag for the map it runs on

    # -- helpers ---------------------------------------------------------------
    def _note(self, now: float, text: str) -> None:
        self.log.append(f"lost filter: {text} ({now - self._clock_origin:.1f} s)")
        self.log_count += 1

    def _fit(self, scan: conv.LaserScan2D, pose: conv.Pose2D) -> float:
        fit, judged = scan_match.wall_agreement(self.occ, scan, pose)
        return float(fit) if judged and math.isfinite(fit) else -1.0

    def _remember(self, now: float, odom: np.ndarray, scan: conv.LaserScan2D) -> None:
        if now - self._last_history_at < self.HISTORY_EVERY_S:
            return
        self._last_history_at = now
        self._history.append((now, odom.copy(), scan))
        while self._history and now - self._history[0][0] > self.HISTORY_S:
            self._history.popleft()

    def _recent_fit(self, pose: conv.Pose2D, odom_now: np.ndarray) -> float:
        """Mean wall fit of the recent scans, each from ``pose`` carried back along odometry."""
        now = conv.Pose2D(*map(float, odom_now))
        fits = []
        for _t, odom_then, scan in self._history:
            back = conv.compose_poses(
                conv.invert_pose(now), conv.Pose2D(*map(float, odom_then))
            )
            fits.append(self._fit(scan, conv.compose_poses(pose, back)))
        return float(np.mean(fits)) if fits else float("nan")

    def _fit_is_bad(
        self, now: float, scan: conv.LaserScan2D, tracker: conv.Pose2D
    ) -> Optional[bool]:
        """Whether the scan misses the walls at the tracker's pose (None: not checked now)."""
        if now - self._last_fit_at < self.FIT_EVERY_S:
            return None
        self._last_fit_at = now
        self._last_fit = self._fit(scan, tracker)
        return self._last_fit < self.BAD_FIT

    def _settled(
        self, scan: conv.LaserScan2D, odom: np.ndarray, moved: bool
    ) -> Optional[conv.Pose2D]:
        """Run the filter one step; the pose once it has settled, else None."""
        if not moved:
            return None
        self.pf.update(scan)
        found = self.pf.converged()
        if found is None:
            self._agree, self._agree_from = 0, None
            return None
        _pose, share, spread, spread_deg = found
        tight = (
            share >= self.SHARE
            and spread <= self.SPREAD_M
            and spread_deg <= self.SPREAD_DEG
        )
        if not tight:
            self._agree, self._agree_from = 0, None
            return None
        self._agree += 1
        if self._agree_from is None:
            self._agree_from = odom[:2].copy()
        travelled = float(np.hypot(*(odom[:2] - self._agree_from)))
        if self._agree < self.CONFIRM_UPDATES or travelled < self.CONFIRM_TRAVEL_M:
            return None
        return self.pf.estimate()  # None if the update itself saw no cluster

    def _camera_holds(self, now: float, pose: conv.Pose2D, what: str) -> bool:
        if self.camera is None:
            return False
        why, places = self.camera(pose, now)
        if why is None:
            return False
        # Search again around the place the camera recognizes as well as
        # everywhere; then confirm over more driving before the next try.
        self.pf.seed(places)
        self._agree, self._agree_from = 0, None
        where = ""
        if places:
            cx = float(np.mean([p.x for p in places]))
            cy = float(np.mean([p.y for p in places]))
            where = f"; searching again around ({cx:.2f}, {cy:.2f})"
        self._note(
            now, f"{what} to ({pose.x:.2f}, {pose.y:.2f}) held back: {why}{where}"
        )
        return True

    def _hand_over(
        self, now: float, odom: np.ndarray, pose: conv.Pose2D, why: str
    ) -> Handover:
        self.handovers += 1
        self.phase = CONFIRMING
        self._confirm_from = odom[:2].copy()
        self._bad_since = None
        self._last_fit_at = -math.inf
        self._note(
            now,
            f"{why} ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.theta):.0f} deg)",
        )
        return Handover(pose=pose, why=why)

    # -- one step ----------------------------------------------------------------
    def step(
        self,
        now: float,
        odom: conv.Pose2D,
        scan: Optional[conv.LaserScan2D],
        tracker: conv.Pose2D,
    ) -> Optional[Handover]:
        """Advance on the newest odometry and scan; a pose for the tracker, if any."""
        o = np.array([odom.x, odom.y, odom.theta], dtype=float)
        moved = self.pf.odometry(odom)
        if scan is None:
            return None
        self._remember(now, o, scan)

        if self.phase == WATCHING:
            self._watch(now, scan, tracker)
            return None
        if self.phase == CONFIRMING:
            self._confirm(now, o, scan, tracker)
            return None

        found = self._settled(scan, o, moved)
        if self.phase == FINDING:
            if found is not None and not self._camera_holds(now, found, "hand-over"):
                return self._hand_over(now, o, found, "found the robot at")
            return None

        # Checking: the tracker carries on until the search says otherwise.
        if found is None:
            if now - self._check_started > self.CHECK_MAX_S:
                self.phase = WATCHING
                self._quiet_until = now + self.CHECK_COOLDOWN_S
                self._note(now, "background check did not settle; tracker kept")
            return None
        apart = math.hypot(found.x - tracker.x, found.y - tracker.y)
        turned = abs(math.degrees(conv.normalize_angle(found.theta - tracker.theta)))
        if apart < self.SAME_PLACE_M and turned < self.SAME_PLACE_DEG:
            self.phase = WATCHING
            self._quiet_until = now + self.CHECK_COOLDOWN_S
            self._note(
                now, "background check agrees with the tracker; the room has changed"
            )
            return None
        new_fit = self._recent_fit(found, o)
        old_fit = self._recent_fit(tracker, o)
        if new_fit >= old_fit + self.SWITCH_MARGIN:
            if self._camera_holds(now, found, "move"):
                return None  # keep checking; the search may settle elsewhere
            return self._hand_over(
                now,
                o,
                found,
                f"moved the tracker {apart:.1f} m (fit {old_fit:.2f} -> {new_fit:.2f}) to",
            )
        self.phase = WATCHING
        self._quiet_until = now + self.CHECK_COOLDOWN_S
        self._note(
            now,
            f"background check found ({found.x:.2f}, {found.y:.2f}), {apart:.1f} m away, "
            f"but it fits no better ({new_fit:.2f} vs {old_fit:.2f}); tracker kept",
        )
        return None

    def _watch(self, now: float, scan: conv.LaserScan2D, tracker: conv.Pose2D) -> None:
        if now < self._quiet_until:
            return
        bad = self._fit_is_bad(now, scan, tracker)
        if bad is None:
            return
        if not bad:
            self._bad_since = None
            return
        if self._bad_since is None:
            self._bad_since = now
        if now - self._bad_since < self.BAD_FIT_S:
            return
        self.pf.scatter()
        self.phase = CHECKING
        self._check_started = now
        self._agree, self._agree_from, self._bad_since = 0, None, None
        self._note(now, "scans stopped fitting the map; searching it in the background")

    def _confirm(
        self, now: float, odom: np.ndarray, scan: conv.LaserScan2D, tracker: conv.Pose2D
    ) -> None:
        bad = self._fit_is_bad(now, scan, tracker)
        if bad:
            if self._bad_since is None:
                self._bad_since = now
            if now - self._bad_since >= self.PROVISIONAL_BAD_S:
                self.withdrawn += 1
                self.phase = FINDING
                self.pf.scatter()
                self._agree, self._agree_from, self._bad_since = 0, None, None
                self._note(
                    now,
                    "scans stopped fitting right after the hand-over; searching again",
                )
            return
        if bad is False:
            self._bad_since = None
        travelled = float(np.hypot(*(odom[:2] - self._confirm_from)))
        if travelled >= self.PROVISIONAL_M and self._bad_since is None:
            self.phase = WATCHING
            self._note(now, f"hand-over confirmed over {travelled:.1f} m")

    def status(self) -> dict:
        return {
            "phase": self.phase,
            "particles": self.pf.particles,
            "updates": self.pf.updates,
            "handovers": self.handovers,
            "withdrawn": self.withdrawn,
            "last_fit": None if self._last_fit is None else round(self._last_fit, 3),
            "log": list(self.log)[-5:],
        }

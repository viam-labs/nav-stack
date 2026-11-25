"""Pose tracking in a known map, scored the way LILocBench scores localizers.

The benchmark paper (Trekel et al., IROS 2025, section IV-C) starts each
localizer near the true first pose of a run, in the dataset's occupancy map
(drawn from the mapping run with ground-truth poses), lets it track the whole
run, and compares every pose with ground truth in the map frame, with no
alignment: RMSE and maximum, and the share of poses within 5 and 10 cm and
within 2 and 4 degrees. Each method ran 10 times.

Here the SLAM engine runs in localizing mode on that map, on the recording's
clock, as ``replay_mapping`` runs it in mapping mode. A replay is
deterministic, so one run from the true first pose says how well it tracks.
Ten more start from poses drawn from the spread the paper samples its
particles from (0.2 m, 0.2 m, 10 degrees). That is a harder start than the
paper's: there the particles surround the true pose, here the one pose the
engine starts from is off by that much.

Each pose is compared with ground truth at the time of the newest data it
was made from (the odometry sample), as a localizer stamps the pose it
publishes and as the benchmark scores it. The same pose compared at the
moment it is read is also reported (``at_read_*``): that adds up to one
odometry period (70 ms here) of lag while driving.

With ``service=True`` the SLAM service runs around the engine as it does on
a robot: its startup localize (a robot restarting near its last saved pose)
and its periodic relocalize watchdog, on the recording's clock. Its scan
matching runs in-process and its background work inline, so the replay stays
deterministic. Without it, only the engine is exercised, which is the
like-for-like comparison with the paper's localizers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.config import MODE_LOCALIZING
from src.geom import conversions as conv
from src.nav.localize.global_localize import OccupancyMap
from src.nav.stores.maps import MapStore
from src.slam_builtin import occupancy as occ
from src.slam_builtin import persistence
from src.slam_builtin.engine import BuiltinSlamEngine

from .lilocbench import LilocSequence
from .replay import ReplayClock, ReplaySensors, _slam_config

# The spread the paper draws its starting particles from.
START_STD = (0.2, 0.2, math.radians(10.0))


@dataclass
class TrackingRun:
    start: conv.Pose2D
    stamps: np.ndarray  # when the data each pose was made from was taken
    est: np.ndarray  # (N, 3) published pose, map frame
    truth: np.ndarray  # (N, 3) ground truth at ``stamps``, map frame
    truth_at_read: np.ndarray  # (N, 3) ground truth when the pose was read
    wall_s: float
    # With the service: each pose it set, as (stamp, before, after).
    service_moves: List[tuple] = field(default_factory=list)
    service_log: List[str] = field(default_factory=list)

    @staticmethod
    def _errors(est: np.ndarray, truth: np.ndarray):
        pos = np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])
        dyaw = est[:, 2] - truth[:, 2]
        yaw = np.abs(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw))))
        return pos, yaw

    def errors(self):
        return self._errors(self.est, self.truth)

    def summary(self) -> Dict[str, float]:
        """The paper's Table III columns, in its units (cm, degrees, percent)."""
        pos, yaw = self.errors()
        late, late_yaw = self._errors(self.est, self.truth_at_read)
        return {
            "at_read_rmse_cm": float(100.0 * np.sqrt(np.mean(late**2))),
            "at_read_max_cm": float(100.0 * late.max()),
            "at_read_heading_rmse_deg": float(np.sqrt(np.mean(late_yaw**2))),
            "service_moves": len(self.service_moves),
            "samples": int(pos.size),
            "rmse_cm": float(100.0 * np.sqrt(np.mean(pos**2))),
            "max_cm": float(100.0 * pos.max()),
            "under_5cm_pct": float(100.0 * np.mean(pos < 0.05)),
            "under_10cm_pct": float(100.0 * np.mean(pos < 0.10)),
            "heading_rmse_deg": float(np.sqrt(np.mean(yaw**2))),
            "heading_max_deg": float(yaw.max()),
            "under_2deg_pct": float(100.0 * np.mean(yaw < 2.0)),
            "under_4deg_pct": float(100.0 * np.mean(yaw < 4.0)),
            "wall_s": round(self.wall_s, 1),
        }


def _span(seq: LilocSequence) -> tuple:
    """First and last stamp with both a scan and ground truth."""
    scan_stamps = seq.scan_stamps()
    t0 = max(float(scan_stamps[0]), float(seq.gt_stamps[0]))
    t1 = min(float(scan_stamps[-1]), float(seq.gt_stamps[-1]))
    # Ground truth interpolates between two poses, so not at its very first.
    while seq.gt_pose(t0) is None and t0 < t1:
        t0 += 0.01
    return t0, t1


def write_map(ref: OccupancyMap, folder: Path) -> None:
    """Save the reference map where the engine loads a map from."""
    grid = occ.from_occupancy_int16(
        np.asarray(ref.grid, dtype=np.int16),
        resolution=float(ref.resolution),
        origin_x=float(ref.origin_x),
        origin_y=float(ref.origin_y),
    )
    persistence.save_occupancy(Path(folder), grid)


class _InlineExecutor(concurrent.futures.ThreadPoolExecutor):
    """Runs each job at once on the caller's thread.

    The service hands its scan matching to a thread. Run there, it would
    finish at a different point of the recording on every replay.
    """

    def submit(self, fn, /, *args, **kwargs):
        done: concurrent.futures.Future = concurrent.futures.Future()
        try:
            done.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - handed to the awaiting task
            done.set_exception(exc)
        return done


class _Recorder:
    """Stands in for a module logger and keeps what was said."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def _keep(self, msg, *args, **_kw) -> None:
        try:
            self.lines.append(str(msg) % args if args else str(msg))
        except (TypeError, ValueError):
            self.lines.append(str(msg))

    info = warning = debug = error = exception = _keep


def _settle(loop: asyncio.AbstractEventLoop) -> None:
    """Run the service's tasks until all are waiting on the recording's clock."""
    while True:
        loop.run_until_complete(asyncio.sleep(0))
        if not loop._ready:  # noqa: SLF001
            return


class _Service:
    """The SLAM service around a replayed engine, ticking on the replay clock."""

    def __init__(self, engine, cfg, store, sensors) -> None:
        from src.models import slam as slam_module
        from src.nav.localize.localize_worker import LocalizeWorker
        from src.slam_builtin import localize_supervisor
        from src.slam_builtin.host import BuiltinSlamHost

        self.log = _Recorder()
        self._patched = [
            (module, module.LOGGER) for module in (slam_module, localize_supervisor)
        ]
        for module, _old in self._patched:
            module.LOGGER = self.log
        svc = slam_module.SlamService("slam")
        svc._cfg = cfg  # noqa: SLF001
        svc._map_store = store  # noqa: SLF001
        svc._engine = engine  # noqa: SLF001
        host = BuiltinSlamHost(engine)
        # The replay ticks the engine itself; its threads never start.
        host.slam_running = lambda: True
        svc._manager = host  # noqa: SLF001
        svc._localizer = LocalizeWorker(enabled=False)  # noqa: SLF001
        engine.set_match_outcome_hook(svc._on_tick_match_outcome)  # noqa: SLF001

        async def read_scan(bands):
            scan = sensors.get_scan(2.0)
            if scan is None or not conv.scan_has_returns(scan):
                raise RuntimeError("no lidar returns available for global_localize")
            return scan, ([np.empty((0, 2)) for _ in bands] if bands else [])

        svc._read_merged_scan_and_bands = read_scan  # noqa: SLF001
        self.svc = svc
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(_InlineExecutor(max_workers=1))

    def start(self) -> None:
        """Load the map and restored pose, then start the robot's checks."""
        svc = self.svc
        svc._start_mode(MODE_LOCALIZING)  # noqa: SLF001
        svc._schedule_startup_global_localize(self.loop)  # noqa: SLF001
        svc._schedule_periodic_relocalize(self.loop)  # noqa: SLF001
        _settle(self.loop)

    def step(self) -> None:
        _settle(self.loop)

    def close(self) -> None:
        svc = self.svc
        svc._cancel_periodic_relocalize_task()  # noqa: SLF001
        svc._cancel_startup_global_localize_task()  # noqa: SLF001
        try:
            _settle(self.loop)
        finally:
            self.loop.close()
            for module, old in self._patched:
                module.LOGGER = old


class _LostFinder:
    """A particle filter over the whole map, for when the tracker has no pose or the wrong one.

    With no pose, it starts with particles spread over every free cell,
    follows odometry and scans as AMCL does, and once one cluster holds
    nearly all the weight and has stayed tight over ``CONFIRM_TRAVEL_M`` of
    driving, hands that pose to the engine (as a relocalize would).

    After that it watches. When the scans have not sat on the walls at the
    tracker's pose for ``BAD_FIT_S``, it searches the whole map again in the
    background while the tracker carries on. If the search settles where
    the tracker already is, the tracker was right and the room has changed;
    nothing moves. If it settles elsewhere, both poses are scored against the
    last ``HISTORY_S`` of scans, each carried back along odometry, and the
    tracker is moved only when the new place fits clearly better.
    """

    # The cloud stays about 0.2 m and 8 degrees wide once it has found the
    # robot (the odometry noise keeps it so); the tracker takes it from there.
    SHARE = 0.9
    SPREAD_M = 0.3
    SPREAD_DEG = 10.0
    CONFIRM_UPDATES = 3
    # Rooms that look alike fool one scan; driving out of them does not.
    CONFIRM_TRAVEL_M = 1.5
    # Scans that do not sit on the walls at the tracker's pose for this long
    # start a background check.
    BAD_FIT = 0.25
    BAD_FIT_S = 8.0
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

    def __init__(
        self, ref: OccupancyMap, seq: LilocSequence, engine, *, has_pose: bool = False
    ) -> None:
        from .amcl import Amcl, AmclParams, distance_field

        params = AmclParams(
            min_particles=1000,
            max_particles=20000,
            z_hit=0.95,
            z_rand=0.05,
            update_min_d=0.1,
            update_min_a=0.1,
        )
        key = id(ref)
        if key not in _AMCL_FIELDS:
            _AMCL_FIELDS[key] = distance_field(ref, params.max_dist)
        self.pf = Amcl(
            ref, conv.Pose2D(0.0, 0.0, 0.0), params=params, field=_AMCL_FIELDS[key]
        )
        self.seq, self.engine, self.ref = seq, engine, ref
        self.odom_t, self.odom_pose, _twist = seq.odometry()
        # "finding" (no pose yet), "watching", or "checking" (background search).
        self.phase = "watching" if has_pose else "finding"
        if not has_pose:
            self.pf.scatter()
        self.agree = 0
        self.agree_from: Optional[np.ndarray] = None
        self.bad_since: Optional[float] = None
        self.last_fit_at = -1.0
        self.check_started = 0.0
        self.quiet_until = -1.0
        self.history: List[tuple] = []  # (stamp, odom pose (3,), scan)
        self.last_history_at = -1.0
        self.log: List[str] = []

    def _at(self, stamp: float) -> float:
        return stamp - float(self.seq.gt_stamps[0])

    def _odom(self, stamp: float) -> Optional[np.ndarray]:
        i = int(np.searchsorted(self.odom_t, stamp, side="right")) - 1
        return None if i < 0 else np.array(self.odom_pose[i], dtype=float)

    def _remember(self, stamp: float, odom: np.ndarray) -> None:
        if stamp - self.last_history_at < self.HISTORY_EVERY_S:
            return
        scan, _taken = self.seq.scan_before(stamp)
        if scan is None:
            return
        self.last_history_at = stamp
        self.history.append((stamp, odom.copy(), scan))
        while self.history and stamp - self.history[0][0] > self.HISTORY_S:
            self.history.pop(0)

    def _recent_fit(self, pose: conv.Pose2D, odom_now: np.ndarray) -> float:
        """Mean wall fit of the recent scans, each from ``pose`` carried back along odometry."""
        from src.slam_builtin import scan_match

        now = conv.Pose2D(*map(float, odom_now))
        fits = []
        for _stamp, odom_then, scan in self.history:
            back = conv.compose_poses(
                conv.invert_pose(now), conv.Pose2D(*map(float, odom_then))
            )
            fit, judged = scan_match.wall_agreement(
                self.ref, scan, conv.compose_poses(pose, back)
            )
            fits.append(fit if judged and math.isfinite(fit) else -1.0)
        return float(np.mean(fits)) if fits else float("nan")

    def _watch(self, stamp: float) -> None:
        from src.slam_builtin import scan_match

        if stamp - self.last_fit_at < 1.0 or stamp < self.quiet_until:
            return
        self.last_fit_at = stamp
        scan, _taken = self.seq.scan_before(stamp)
        if scan is None:
            return
        fit, judged = scan_match.wall_agreement(self.ref, scan, self.engine.get_pose())
        if judged and math.isfinite(fit) and fit >= self.BAD_FIT:
            self.bad_since = None
            return
        if self.bad_since is None:
            self.bad_since = stamp
        if stamp - self.bad_since < self.BAD_FIT_S:
            return
        self.pf.scatter()
        self.phase = "checking"
        self.check_started = stamp
        self.agree, self.agree_from, self.bad_since = 0, None, None
        self.log.append(
            f"lost finder: scans stopped fitting; checking in the background "
            f"{self._at(stamp):.1f} s into the run"
        )

    def _settled(self, stamp: float, odom: np.ndarray) -> Optional[conv.Pose2D]:
        """Run the filter one step; the pose once it has settled, else None."""
        if not self.pf.odometry(conv.Pose2D(*map(float, odom))):
            return None
        scan, _taken = self.seq.scan_before(stamp)
        if scan is None:
            return None
        self.pf.update(scan)
        found = self.pf.converged()
        if found is None:
            self.agree, self.agree_from = 0, None
            return None
        _pose, share, spread, spread_deg = found
        tight = (
            share >= self.SHARE
            and spread <= self.SPREAD_M
            and spread_deg <= self.SPREAD_DEG
        )
        if not tight:
            self.agree, self.agree_from = 0, None
            return None
        self.agree += 1
        if self.agree_from is None:
            self.agree_from = odom[:2].copy()
        travelled = float(np.hypot(*(odom[:2] - self.agree_from)))
        if self.agree < self.CONFIRM_UPDATES or travelled < self.CONFIRM_TRAVEL_M:
            return None
        return self.pf.estimate()

    def _hand_over(self, pose: conv.Pose2D, stamp: float, why: str) -> None:
        self.engine.set_pose(pose)
        truth = self.seq.gt_pose(stamp)
        off = (
            f", {math.hypot(pose.x - truth.x, pose.y - truth.y):.2f} m from truth"
            if truth is not None
            else ""
        )
        self.log.append(
            f"lost finder: {why} ({pose.x:.2f}, {pose.y:.2f}) "
            f"{self._at(stamp):.1f} s into the run{off}"
        )

    def step(self, stamp: float, clock) -> None:
        del clock
        odom = self._odom(stamp)
        if odom is None:
            return
        self._remember(stamp, odom)
        if self.phase == "watching":
            self._watch(stamp)
            return
        found = self._settled(stamp, odom)
        if self.phase == "finding":
            if found is not None:
                self._hand_over(found, stamp, "handed over")
                self.phase = "watching"
            return
        # Checking: the tracker carries on until the search says otherwise.
        if found is None:
            if stamp - self.check_started > self.CHECK_MAX_S:
                self.phase = "watching"
                self.quiet_until = stamp + self.CHECK_COOLDOWN_S
                self.log.append(
                    f"lost finder: check did not settle; tracker kept "
                    f"{self._at(stamp):.1f} s into the run"
                )
            return
        tracker = self.engine.get_pose()
        apart = math.hypot(found.x - tracker.x, found.y - tracker.y)
        turned = abs(math.degrees(conv.normalize_angle(found.theta - tracker.theta)))
        self.phase = "watching"
        self.quiet_until = stamp + self.CHECK_COOLDOWN_S
        if apart < self.SAME_PLACE_M and turned < self.SAME_PLACE_DEG:
            self.log.append(
                f"lost finder: check agrees with the tracker "
                f"{self._at(stamp):.1f} s into the run"
            )
            return
        new_fit = self._recent_fit(found, odom)
        old_fit = self._recent_fit(tracker, odom)
        if new_fit >= old_fit + self.SWITCH_MARGIN:
            self._hand_over(
                found,
                stamp,
                f"moved the tracker {apart:.1f} m "
                f"(fit {old_fit:.2f} -> {new_fit:.2f}) to",
            )
        else:
            self.log.append(
                f"lost finder: check found ({found.x:.2f}, {found.y:.2f}), "
                f"{apart:.1f} m away, but it fits no better "
                f"({new_fit:.2f} vs {old_fit:.2f}); tracker kept "
                f"{self._at(stamp):.1f} s into the run"
            )


def track(
    seq: LilocSequence,
    ref: OccupancyMap,
    start: Optional[conv.Pose2D] = None,
    *,
    tick_hz: float = 15.0,
    max_seconds: Optional[float] = None,
    config: Optional[dict] = None,
    service: bool = False,
    start_offset_s: float = 0.0,
    global_start: bool = False,
    lost_filter: bool = False,
) -> TrackingRun:
    """Localize through ``seq`` in ``ref`` from ``start`` (default: the true first pose).

    With ``service`` the robot's SLAM service runs around the engine, and
    ``start`` is the pose it restarts from (its saved last pose).
    """
    t_wall = time.perf_counter()
    t0, t1 = _span(seq)
    if start_offset_s:
        t0 = max(t0, float(seq.gt_stamps[0]) + float(start_offset_s))
        while seq.gt_pose(t0) is None and t0 < t1:
            t0 += 0.01
    if max_seconds is not None:
        t1 = min(t1, t0 + float(max_seconds))
    first = seq.gt_pose(t0)
    if first is None:
        raise ValueError(f"{seq.name}: no ground truth at the first scan")
    start = first if start is None else start
    clock = ReplayClock(t0)
    rows = []
    moves: List[tuple] = []
    driver: Optional[_Service] = None
    with tempfile.TemporaryDirectory(prefix="nav-track-") as tmp, clock:
        cfg = _slam_config(
            Path(tmp), vision=False, extra={"mode": MODE_LOCALIZING, **(config or {})}
        )
        store = MapStore(tmp)
        handle = store.get_or_create_map("reference")
        store.set_active_map("reference")
        write_map(ref, handle.root)
        sensors = ReplaySensors(seq, clock)
        engine = BuiltinSlamEngine(
            cfg,
            sensors,
            store,
            logger=lambda _msg: None,
            rate_hz=tick_hz,  # type: ignore[arg-type]
        )
        if service:
            # A robot restarting: its last pose is saved with the map, unless
            # it has to find itself from nothing.
            if not global_start:
                persistence.save_last_pose(handle.root, start)
            driver = _Service(engine, cfg, store, sensors)
            real_set_pose = engine.set_pose

            def set_pose(pose: conv.Pose2D) -> None:
                before = engine.get_pose()
                real_set_pose(pose)
                moves.append((clock.stamp(clock.now), before, pose))

            engine.set_pose = set_pose
            driver.start()
        else:
            engine.configure_mode(MODE_LOCALIZING, handle.root)
            if not global_start:
                engine.set_pose(start)
        period = 1.0 / float(tick_hz)
        stamp = t0
        finder = (
            _LostFinder(ref, seq, engine, has_pose=not global_start)
            if lost_filter
            else None
        )
        while stamp <= t1:
            clock.now = clock.of(stamp)
            engine._tick()  # noqa: SLF001
            if finder is not None:
                finder.step(stamp, clock)
            if driver is not None:
                driver.step()
            taken = sensors.odom_taken
            truth = seq.gt_pose(taken) if math.isfinite(taken) else None
            now = seq.gt_pose(stamp)
            if truth is not None and now is not None:
                pose = engine.get_pose()
                rows.append(
                    (
                        taken,
                        (pose.x, pose.y, pose.theta),
                        (truth.x, truth.y, truth.theta),
                        (now.x, now.y, now.theta),
                    )
                )
            stamp += period
        if driver is not None:
            driver.close()
    return TrackingRun(
        start=start,
        stamps=np.array([r[0] for r in rows]),
        est=np.array([r[1] for r in rows]),
        truth=np.array([r[2] for r in rows]),
        truth_at_read=np.array([r[3] for r in rows]),
        wall_s=time.perf_counter() - t_wall,
        service_moves=moves,
        service_log=(list(driver.log.lines) if driver is not None else [])
        + (finder.log if finder is not None else []),
    )


def perturbed_starts(
    first: conv.Pose2D, count: int, *, seed: int = 0
) -> List[conv.Pose2D]:
    """``count`` starting poses around ``first``, drawn with ``START_STD``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(count)):
        dx, dy, dth = rng.normal(0.0, START_STD)
        out.append(
            conv.Pose2D(
                first.x + dx, first.y + dy, conv.normalize_angle(first.theta + dth)
            )
        )
    return out


def _mean_std(rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = [k for k in rows[0] if k != "samples"]
    return {
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "std": {k: float(np.std([r[k] for r in rows])) for k in keys},
    }


def evaluate_tracking(
    seq: LilocSequence,
    ref: OccupancyMap,
    *,
    starts: int = 10,
    seed: int = 0,
    max_seconds: Optional[float] = None,
    service: bool = False,
    localizer: Optional[str] = None,
    lost_filter: bool = False,
    log=print,
) -> dict:
    """One run from the true first pose, then ``starts`` from drawn poses.

    ``localizer``: ``engine``, ``service`` (same as ``service=True``) or
    ``emcl2`` (the particle filter in ``emcl``, seeded per start).
    """
    who = localizer or ("service" if service else "engine")

    def run_from(start, i):
        if who == "emcl2":
            return track_emcl(seq, ref, start, seed=i, max_seconds=max_seconds)
        return track(
            seq,
            ref,
            start,
            max_seconds=max_seconds,
            service=who == "service",
            lost_filter=lost_filter,
        )

    first = seq.gt_pose(_span(seq)[0])
    exact = run_from(first, 0)
    lidars = list(getattr(seq, "lidars", ()))
    out: dict = {"lidars": lidars, "true_start": exact.summary()}
    log(
        f"  {seq.name} [{'+'.join(lidars)}, {who}]: true start "
        f"{out['true_start']['rmse_cm']:.2f} cm rmse in {exact.wall_s:.0f} s"
    )
    if starts > 0:
        runs = []
        for i, start in enumerate(perturbed_starts(first, starts, seed=seed)):
            run = run_from(start, i + 1)
            summary = run.summary()
            summary["start_offset_m"] = float(
                math.hypot(start.x - first.x, start.y - first.y)
            )
            summary["start_offset_deg"] = float(
                abs(math.degrees(conv.normalize_angle(start.theta - first.theta)))
            )
            runs.append(summary)
            log(
                f"  {seq.name} [{'+'.join(lidars)}, {who}]: start {i + 1}/{starts} "
                f"({summary['start_offset_m']:.2f} m, {summary['start_offset_deg']:.0f} deg off) "
                f"{summary['rmse_cm']:.2f} cm rmse"
            )
        out["drawn_starts"] = {"runs": runs, **_mean_std(runs)}
    return out


def track_emcl(
    seq: LilocSequence,
    ref: OccupancyMap,
    start: Optional[conv.Pose2D] = None,
    *,
    refine: bool = False,
    seed: int = 0,
    tick_hz: float = 15.0,
    max_seconds: Optional[float] = None,
    params=None,
) -> TrackingRun:
    """Localize with the expansion-resetting particle filter (``emcl.Emcl``).

    Particles start spread around ``start`` as the paper spreads them around
    the true pose. With ``refine`` the stack's scan matcher refines the
    filter's estimate against the map after every scan, and that refined
    pose is what is published (the filter itself is not told).
    """
    from src.slam_builtin import scan_match

    from .emcl import Emcl

    t_wall = time.perf_counter()
    t0, t1 = _span(seq)
    if max_seconds is not None:
        t1 = min(t1, t0 + float(max_seconds))
    first = seq.gt_pose(t0)
    start = first if start is None else start
    odom_t, odom_pose, _twist = seq.odometry()
    pf = Emcl(ref, start, START_STD, params=params, seed=seed)
    lookup = scan_match.occupied_lookup(ref) if refine else None
    correction = conv.Pose2D(0.0, 0.0, 0.0)
    last_taken = float("nan")
    rows = []
    period = 1.0 / float(tick_hz)
    stamp = t0
    while stamp <= t1:
        i = int(np.searchsorted(odom_t, stamp, side="right")) - 1
        if i >= 0:
            x, y, yaw = odom_pose[i]
            pf.move(conv.Pose2D(float(x), float(y), float(yaw)))
            scan, taken = seq.scan_before(stamp)
            if scan is not None and taken != last_taken:
                last_taken = taken
                pf.weigh(scan)
                if refine:
                    est = pf.estimate()
                    found = scan_match.refine_pose(
                        ref, scan, est, occupied_lookup=lookup
                    )
                    correction = (
                        conv.compose_poses(found[0], conv.invert_pose(est))
                        if found[0] is not None
                        else conv.Pose2D(0.0, 0.0, 0.0)
                    )
            pose = conv.compose_poses(correction, pf.estimate())
            taken_odom = float(odom_t[i])
            truth = seq.gt_pose(taken_odom)
            now = seq.gt_pose(stamp)
            if truth is not None and now is not None:
                rows.append(
                    (
                        taken_odom,
                        (pose.x, pose.y, pose.theta),
                        (truth.x, truth.y, truth.theta),
                        (now.x, now.y, now.theta),
                    )
                )
        stamp += period
    return TrackingRun(
        start=start,
        stamps=np.array([r[0] for r in rows]),
        est=np.array([r[1] for r in rows]),
        truth=np.array([r[2] for r in rows]),
        truth_at_read=np.array([r[3] for r in rows]),
        wall_s=time.perf_counter() - t_wall,
        service_log=[f"expansion resets: {pf.resets}"],
    )


_AMCL_FIELDS: Dict[int, np.ndarray] = {}


def track_amcl(
    seq: LilocSequence,
    ref: OccupancyMap,
    start: Optional[conv.Pose2D] = None,
    *,
    seed: int = 0,
    tick_hz: float = 15.0,
    max_seconds: Optional[float] = None,
    params=None,
) -> TrackingRun:
    """Localize with adaptive Monte Carlo localization (``amcl.Amcl``).

    Particles start spread around ``start`` as the paper spreads them around
    the true pose. The published pose is the filter's last estimate carried
    forward by odometry, as AMCL publishes it.
    """
    from .amcl import Amcl, AmclParams, distance_field

    t_wall = time.perf_counter()
    t0, t1 = _span(seq)
    if max_seconds is not None:
        t1 = min(t1, t0 + float(max_seconds))
    first = seq.gt_pose(t0)
    start = first if start is None else start
    p = params or AmclParams()
    key = id(ref)
    if key not in _AMCL_FIELDS:
        _AMCL_FIELDS[key] = distance_field(ref, p.max_dist)
    pf = Amcl(ref, start, START_STD, params=p, seed=seed, field=_AMCL_FIELDS[key])
    odom_t, odom_pose, _twist = seq.odometry()
    rows = []
    period = 1.0 / float(tick_hz)
    stamp = t0
    while stamp <= t1:
        i = int(np.searchsorted(odom_t, stamp, side="right")) - 1
        if i >= 0:
            x, y, yaw = odom_pose[i]
            if pf.odometry(conv.Pose2D(float(x), float(y), float(yaw))):
                scan, _taken = seq.scan_before(stamp)
                if scan is not None:
                    pf.update(scan)
            pose = pf.estimate()
            taken_odom = float(odom_t[i])
            truth = seq.gt_pose(taken_odom)
            now = seq.gt_pose(stamp)
            if truth is not None and now is not None:
                rows.append(
                    (
                        taken_odom,
                        (pose.x, pose.y, pose.theta),
                        (truth.x, truth.y, truth.theta),
                        (now.x, now.y, now.theta),
                    )
                )
        stamp += period
    return TrackingRun(
        start=start,
        stamps=np.array([r[0] for r in rows]),
        est=np.array([r[1] for r in rows]),
        truth=np.array([r[2] for r in rows]),
        truth_at_read=np.array([r[3] for r in rows]),
        wall_s=time.perf_counter() - t_wall,
        service_log=[f"filter updates: {pf.updates}"],
    )

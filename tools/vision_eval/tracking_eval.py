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

Only the engine is exercised. The localization watchdog and periodic
relocalize of the SLAM service are not part of the replay.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
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


def track(
    seq: LilocSequence,
    ref: OccupancyMap,
    start: Optional[conv.Pose2D] = None,
    *,
    tick_hz: float = 15.0,
    max_seconds: Optional[float] = None,
    config: Optional[dict] = None,
) -> TrackingRun:
    """Localize through ``seq`` in ``ref`` from ``start`` (default: the true first pose)."""
    t_wall = time.perf_counter()
    t0, t1 = _span(seq)
    if max_seconds is not None:
        t1 = min(t1, t0 + float(max_seconds))
    first = seq.gt_pose(t0)
    if first is None:
        raise ValueError(f"{seq.name}: no ground truth at the first scan")
    start = first if start is None else start
    clock = ReplayClock(t0)
    rows = []
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
        engine.configure_mode(MODE_LOCALIZING, handle.root)
        engine.set_pose(start)
        period = 1.0 / float(tick_hz)
        stamp = t0
        while stamp <= t1:
            clock.now = clock.of(stamp)
            engine._tick()  # noqa: SLF001
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
    return TrackingRun(
        start=start,
        stamps=np.array([r[0] for r in rows]),
        est=np.array([r[1] for r in rows]),
        truth=np.array([r[2] for r in rows]),
        truth_at_read=np.array([r[3] for r in rows]),
        wall_s=time.perf_counter() - t_wall,
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
    log=print,
) -> dict:
    """One run from the true first pose, then ``starts`` from drawn poses."""
    first = seq.gt_pose(_span(seq)[0])
    exact = track(seq, ref, first, max_seconds=max_seconds)
    lidars = list(getattr(seq, "lidars", ()))
    out: dict = {"lidars": lidars, "true_start": exact.summary()}
    log(
        f"  {seq.name} [{'+'.join(lidars)}]: true start "
        f"{out['true_start']['rmse_cm']:.2f} cm rmse in {exact.wall_s:.0f} s"
    )
    if starts > 0:
        runs = []
        for i, start in enumerate(perturbed_starts(first, starts, seed=seed)):
            run = track(seq, ref, start, max_seconds=max_seconds)
            summary = run.summary()
            summary["start_offset_m"] = float(
                math.hypot(start.x - first.x, start.y - first.y)
            )
            summary["start_offset_deg"] = float(
                abs(math.degrees(conv.normalize_angle(start.theta - first.theta)))
            )
            runs.append(summary)
            log(
                f"  {seq.name} [{'+'.join(lidars)}]: start {i + 1}/{starts} "
                f"({summary['start_offset_m']:.2f} m, {summary['start_offset_deg']:.0f} deg off) "
                f"{summary['rmse_cm']:.2f} cm rmse"
            )
        out["drawn_starts"] = {"runs": runs, **_mean_std(runs)}
    return out

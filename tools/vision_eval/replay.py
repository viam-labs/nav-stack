"""Replay a recorded run through the SLAM engine and score its trajectory.

The kidnapped-robot tests judge one frame at a time. Drift, mapping and loop
closure only exist inside a running session, so this feeds the engine the
recorded odometry and scans in order, on a clock that follows the recording,
and compares where the engine says the robot is with ground truth.

The engine is stepped the way its own threads would step it: one ``_tick``
per scan period, and a loop-closure pass whenever the engine asks for one or
a second has gone by. Nothing runs in the background, so a replay is
repeatable.
"""

from __future__ import annotations

import math
import pickle
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.config import MODE_MAPPING, SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.semantic.types import Detection, PatchToken
from src.slam_builtin.engine import BuiltinSlamEngine

from .features import FrameFeatures
from .lilocbench import LilocSequence

# The loop-closure worker wakes at least this often.
LOOP_WORKER_PERIOD_S = 1.0
_CLOCK_START = 1000.0


class ReplayClock:
    """Stands in for ``time.monotonic`` while a replay runs."""

    def __init__(self, first_stamp: float) -> None:
        self._offset = _CLOCK_START - float(first_stamp)
        self.now = _CLOCK_START
        self._real = time.monotonic

    def of(self, stamp: float) -> float:
        return float(stamp) + self._offset

    def stamp(self, mono: float) -> float:
        return float(mono) - self._offset

    def __enter__(self) -> "ReplayClock":
        time.monotonic = lambda: self.now  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        time.monotonic = self._real  # type: ignore[assignment]


class ReplaySensors:
    """What the engine reads: the latest scan and odometry at the replay time."""

    def __init__(self, seq: LilocSequence, clock: ReplayClock) -> None:
        self._seq, self._clock = seq, clock
        self._odom_t, self._odom_pose, self._odom_twist = seq.odometry()
        self._scan: Optional[conv.LaserScan2D] = None
        self._scan_taken = float("nan")

    def get_scan(self, max_age_s: float = 2.0, *, fresh: bool = False):
        del fresh
        stamp = self._clock.stamp(self._clock.now)
        scan, taken = self._seq.scan_before(stamp)
        if scan is None or stamp - taken > float(max_age_s):
            return None
        if taken != self._scan_taken:
            # A new object per scan: the engine tells scans apart by identity.
            self._scan, self._scan_taken = scan, taken
        return self._scan

    def scan_captured_at(self) -> float:
        return self._clock.of(self._scan_taken)

    def get_odom(self) -> Optional[conv.OdomReading]:
        stamp = self._clock.stamp(self._clock.now)
        i = int(np.searchsorted(self._odom_t, stamp, side="right")) - 1
        if i < 0:
            return None
        x, y, yaw = self._odom_pose[i]
        vx, vy, wz = self._odom_twist[i]
        return conv.OdomReading(
            float(vx),
            float(vy),
            float(wz),
            pose=conv.Pose2D(float(x), float(y), float(yaw)),
        )


def align_se2(
    est_xy: np.ndarray, truth_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rotation, translation and yaw that best lay ``est_xy`` on ``truth_xy``."""
    mu_e, mu_t = est_xy.mean(axis=0), truth_xy.mean(axis=0)
    h = (est_xy - mu_e).T @ (truth_xy - mu_t)
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    return rot, mu_t - rot @ mu_e, math.atan2(rot[1, 0], rot[0, 0])


@dataclass
class Trajectory:
    """Poses over time with the ground truth for the same instants."""

    stamps: np.ndarray
    est: np.ndarray  # (N, 3) x, y, yaw in the estimator's own frame
    truth: np.ndarray  # (N, 3) in the ground-truth frame

    def errors(
        self, *, anchor_s: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Position (m) and heading (deg) error per sample after aligning frames.

        With ``anchor_s`` only the first that many seconds set the alignment,
        so the error that follows is drift from a known start. Without it the
        whole path is fitted (the usual absolute trajectory error).
        """
        use = np.ones(len(self.stamps), dtype=bool)
        if anchor_s is not None:
            use = self.stamps - self.stamps[0] <= float(anchor_s)
        rot, shift, yaw = align_se2(self.est[use, :2], self.truth[use, :2])
        moved = self.est[:, :2] @ rot.T + shift
        pos = np.hypot(moved[:, 0] - self.truth[:, 0], moved[:, 1] - self.truth[:, 1])
        dyaw = self.est[:, 2] + yaw - self.truth[:, 2]
        return pos, np.abs(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw))))

    def position_summary(self) -> Dict[str, float]:
        """Position error only (for keyframes, whose stamps lag their heading)."""
        fitted, _yaw = self.errors()
        return {
            "samples": int(len(self.stamps)),
            "ate_rmse_m": float(np.sqrt(np.mean(fitted**2))),
            "ate_max_m": float(fitted.max()),
        }

    def summary(self) -> Dict[str, float]:
        fitted, fitted_yaw = self.errors()
        drift, _ = self.errors(anchor_s=20.0)
        return {
            "samples": int(len(self.stamps)),
            "ate_rmse_m": float(np.sqrt(np.mean(fitted**2))),
            "ate_max_m": float(fitted.max()),
            "heading_rmse_deg": float(np.sqrt(np.mean(fitted_yaw**2))),
            "drift_median_m": float(np.median(drift)),
            "drift_max_m": float(drift.max()),
            "drift_end_m": float(drift[-1]),
        }


def odometry_trajectory(seq: LilocSequence) -> Trajectory:
    """Wheel odometry alone, against ground truth."""
    stamps, poses, _twists = seq.odometry()
    rows = [(t, p, seq.gt_pose(t)) for t, p in zip(stamps[::3], poses[::3])]
    rows = [(t, p, g) for t, p, g in rows if g is not None]
    return Trajectory(
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([[r[2].x, r[2].y, r[2].theta] for r in rows]),
    )


@dataclass
class ReplayResult:
    live: Trajectory  # the pose the engine published at each tick
    # Pose-graph keyframes as they stand at the end, each against the ground
    # truth of the moment it was added. A keyframe drawn from a scan held
    # during a spin is added after the robot has turned on, so only its
    # position is comparable.
    final: Trajectory
    loop: dict
    vision: Optional[dict]
    travel_m: float
    wall_s: float
    ticks: int
    logs: List[str] = field(default_factory=list)
    # The occupancy map the engine ended with (``engine.get_map()``).
    map: Optional[dict] = None
    # How many frames showed a person, and how many of those reached the
    # engine while the mask was still fresh.
    vision_trace: Optional[dict] = None

    def save_map_png(self, path: Path) -> Path:
        """Write the final map as an image: black walls, white free, grey unknown."""
        from PIL import Image

        grid = np.asarray(self.map["grid"])
        img = np.full(grid.shape, 128, dtype=np.uint8)
        img[(grid >= 0) & (grid < 50)] = 255
        img[grid >= 65] = 0
        rows, cols = np.nonzero(grid >= 0)
        if rows.size:
            img = img[
                max(0, rows.min() - 5) : rows.max() + 6,
                max(0, cols.min() - 5) : cols.max() + 6,
            ]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.flipud(img)).save(path)
        return path

    def summary(self) -> dict:
        out = {
            "live": self.live.summary(),
            "final_map": self.final.position_summary()
            if len(self.final.stamps) >= 3
            else None,
            "loop_closures": int(self.loop.get("closures", 0)),
            "loop_attempts": int(self.loop.get("attempts", 0)),
            "travel_m": round(self.travel_m, 1),
            "wall_s": round(self.wall_s, 1),
        }
        if self.vision is not None:
            out["recovery_keyframes"] = int(
                self.vision.get("recovery", {}).get("keyframes", 0)
            )
            out["vision_trace"] = self.vision_trace
        return out


def _slam_config(
    maps_dir: Path, *, vision: bool, extra: Optional[dict] = None
) -> SlamConfig:
    cfg: dict = {
        "base": "base",
        "maps_dir": str(maps_dir),
        "mode": MODE_MAPPING,
        "lidars": [{"name": "front"}],
    }
    if vision:
        cfg["lidars"].append(
            {
                "name": "camera-front",
                "obstacles_only": True,
                "cloud_frame": "camera_optical",
                "scan_source": "point_cloud",
            }
        )
        cfg.update({"vision_enabled": True, "vision_cameras": ["camera-front"]})
    cfg.update(extra or {})
    return SlamConfig.from_dict(cfg)


class CachedWorker:
    """Runs the vision worker on dataset frames without repeating inference.

    DINOv2 tokens come from the feature cache. Detector boxes are computed on
    first use and kept on disk, so later replays of the same run cost nothing.
    """

    def __init__(
        self, worker, seq: LilocSequence, feats: FrameFeatures, cache_dir: Path
    ) -> None:
        self._worker, self._seq, self._feats = worker, seq, feats
        self._path = (
            Path(cache_dir).expanduser()
            / seq.name
            / f"camera_{seq.camera}"
            / "detections.pkl"
        )
        self._boxes: Dict[int, List[Detection]] = {}
        if self._path.is_file():
            self._boxes = pickle.loads(self._path.read_bytes())
        self._dirty = False
        self._real_yolo = worker._run_yolo  # noqa: SLF001
        self.frame = 0
        worker._dino_tokens = self._dino_tokens  # noqa: SLF001
        worker._run_yolo = self._run_yolo  # noqa: SLF001

    def _dino_tokens(self, _canvas, *, scale, pad_x, pad_y):
        del scale, pad_x, pad_y
        desc = self._feats.tokens(self.frame)
        tokens = [
            PatchToken(
                row=int(r), col=int(c), u=float(u), v=float(v), descriptor=desc[i]
            )
            for i, ((r, c), (u, v)) in enumerate(zip(self._feats.grid, self._feats.uv))
        ]
        return tokens, desc.mean(axis=0)

    def _run_yolo(self, color):
        if self.frame not in self._boxes:
            self._boxes[self.frame] = self._real_yolo(color)
            self._dirty = True
        return list(self._boxes[self.frame])

    def save(self) -> None:
        if self._dirty:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_bytes(pickle.dumps(self._boxes))
            self._dirty = False


def replay_mapping(
    seq: LilocSequence,
    *,
    feats: Optional[FrameFeatures] = None,
    cache_dir: Optional[Path] = None,
    vision: bool = False,
    vision_latency_s: float = 0.0,
    config: Optional[dict] = None,
    tick_hz: float = 15.0,
    max_seconds: Optional[float] = None,
    log: Callable[[str], None] = print,
) -> ReplayResult:
    """Map the run from scratch and compare the engine's poses with ground truth.

    ``vision`` runs the vision supervisor on the camera frames alongside
    (person masking, recovery keyframes). ``config`` overrides SLAM
    attributes.

    ``vision_latency_s`` is how long after a frame is taken its result reaches
    the engine. A replay has no inference time of its own, and the person mask
    only applies to scans within 0.2 s of the frame, so zero flatters it.
    """
    if vision and (feats is None or cache_dir is None):
        raise ValueError("vision replay needs the frame feature cache")
    t_wall = time.perf_counter()
    scan_stamps = seq.scan_stamps()
    t0 = max(float(scan_stamps[0]), float(seq.gt_stamps[0]))
    t1 = min(float(scan_stamps[-1]), float(seq.gt_stamps[-1]))
    if max_seconds is not None:
        t1 = min(t1, t0 + float(max_seconds))
    clock = ReplayClock(t0)
    logs: List[str] = []
    live: List[Tuple[float, conv.Pose2D, conv.Pose2D]] = []
    # Keyframes are told apart by their first scan's sequence number: the
    # engine drops keyframes when it heals a torn patch of map, so an index
    # does not keep meaning the same keyframe.
    node_stamp: Dict[int, float] = {}
    trace: dict = {"frames_with_people": 0, "frames_masked_in_time": 0}

    with tempfile.TemporaryDirectory(prefix="nav-replay-") as tmp, clock:
        cfg = _slam_config(Path(tmp), vision=vision, extra=config)
        store = MapStore(tmp)
        handle = store.get_or_create_map("replay")
        store.set_active_map("replay")
        sensors = ReplaySensors(seq, clock)
        engine = BuiltinSlamEngine(
            cfg,
            sensors,
            store,
            logger=logs.append,
            rate_hz=tick_hz,  # type: ignore[arg-type]
        )
        engine.configure_mode(MODE_MAPPING, handle.root)

        supervisor = None
        cached = None
        if vision:
            from src.semantic.supervisor import VisionSupervisor

            supervisor = VisionSupervisor(
                cfg=cfg,
                cameras={},
                get_pose=engine.get_pose,
                get_pose_at=engine.get_pose_at,
                get_anchor=engine.vision_anchor,
                note_paint_discs=engine.note_paint_discs,
                logger=logs.append,
            )
            supervisor._ensure_worker_loaded()  # noqa: SLF001
            if supervisor._worker.cfg.synthetic:  # noqa: SLF001
                raise RuntimeError(
                    "vision models did not load; replay would not be the real pipeline"
                )
            cached = CachedWorker(supervisor._worker, seq, feats, cache_dir)  # noqa: SLF001
            engine.set_anchor_motion_hook(supervisor.follow_anchor_motion)

            # Tally what the person mask did on each frame.
            real_discs = supervisor._note_paint_discs  # noqa: SLF001

            def counted_discs(discs, captured_at):
                if discs:
                    trace["frames_with_people"] += 1
                    if (
                        time.monotonic() - float(captured_at)
                        <= engine._PAINT_DISC_MAX_AGE_S
                    ):  # noqa: SLF001
                        trace["frames_masked_in_time"] += 1
                real_discs(discs, captured_at)

            supervisor._note_paint_discs = counted_discs  # noqa: SLF001

        period = 1.0 / float(tick_hz)
        next_loop = clock.of(t0) + LOOP_WORKER_PERIOD_S
        frame_i = 0
        ticks = 0
        stamp = t0
        while stamp <= t1:
            clock.now = clock.of(stamp)
            engine._tick()  # noqa: SLF001
            ticks += 1
            for node in reversed(engine._graph.nodes):  # noqa: SLF001
                key = node.scans[0].seq if node.scans else None
                if key is None or key in node_stamp:
                    break
                node_stamp[key] = stamp
            # The loop-closure worker: woken by the engine, or once a second.
            if engine._loop_event.is_set() or clock.now >= next_loop:  # noqa: SLF001
                engine._loop_event.clear()  # noqa: SLF001
                next_loop = clock.now + LOOP_WORKER_PERIOD_S
                engine.refresh_map_health()
                engine.run_loop_closure()
            if supervisor is not None:
                while (
                    frame_i < len(seq)
                    and seq.stamps[frame_i] + vision_latency_s <= stamp
                ):
                    if seq.stamps[frame_i] >= t0:
                        frame = seq.frame(frame_i)
                        frame.stamp = frame.captured_at = clock.of(seq.stamps[frame_i])
                        cached.frame = frame_i
                        supervisor.ingest_frame(frame, mapping=True)
                    frame_i += 1
            truth = seq.gt_pose(stamp)
            if truth is not None:
                live.append((stamp, engine.get_pose(), truth))
            if ticks % 3000 == 0:
                log(f"  {seq.name}: replayed {stamp - t0:.0f} of {t1 - t0:.0f} s")
            stamp += period

        with engine._lock:  # noqa: SLF001
            nodes = [
                (node_stamp.get(n.scans[0].seq) if n.scans else None, n.pose)
                for n in engine._graph.nodes  # noqa: SLF001
            ]
            travel = (
                float(engine._graph.nodes[-1].travel_m) if engine._graph.nodes else 0.0
            )  # noqa: SLF001
            loop_stats = dict(engine._loop_stats)  # noqa: SLF001
        final_map = engine.get_map()
        vision_status = supervisor.status() if supervisor is not None else None
        if cached is not None:
            cached.save()
        if supervisor is not None:
            supervisor.stop()

    def pack(rows) -> Trajectory:
        return Trajectory(
            np.array([r[0] for r in rows]),
            np.array([[r[1].x, r[1].y, r[1].theta] for r in rows]),
            np.array([[r[2].x, r[2].y, r[2].theta] for r in rows]),
        )

    final_rows = []
    for stamp_i, pose in nodes:
        truth = seq.gt_pose(stamp_i) if stamp_i is not None else None
        if truth is not None:
            final_rows.append((stamp_i, pose, truth))
    return ReplayResult(
        live=pack(live),
        final=pack(final_rows)
        if final_rows
        else Trajectory(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))),
        loop=loop_stats,
        vision=vision_status,
        travel_m=travel,
        wall_s=time.perf_counter() - t_wall,
        ticks=ticks,
        logs=logs,
        map=final_map,
        vision_trace=trace if vision else None,
    )

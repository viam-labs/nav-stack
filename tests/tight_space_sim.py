"""Closed-loop harness: real ``NavSupervisor`` vs a rectangular robot world.

The world is ground truth: exact oriented-rectangle body collision against an
occupancy grid (plus obstacles the nav map does not know about), raycast lidar
from the body centre, and a fake clock that advances physics whenever the
supervisor sleeps. Scenarios run in simulated time, so a 60 s traverse costs a
few seconds of CPU.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.config import NavConfig
from src.geom import conversions as conv
from src.nav_builtin import supervisor as sup_mod
from src.nav_builtin.supervisor import NavSupervisor

RES = 0.05
LENGTH = 0.72
WIDTH = 0.59


class FakeClock:
    def __init__(self, world: "RectWorld"):
        self.t = 1000.0
        self.world = world
        self.limit_s: Optional[float] = None
        self.on_limit = None

    def monotonic(self) -> float:
        return self.t

    def time(self) -> float:
        return self.t

    def perf_counter(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        self.world.advance(dt)
        self.t += dt
        if self.limit_s is not None and self.t - 1000.0 > self.limit_s and self.on_limit:
            cb, self.on_limit = self.on_limit, None
            cb()


def _body_samples(length: float, width: float, step: float) -> np.ndarray:
    xs = np.arange(-length / 2.0, length / 2.0 + 1e-9, step)
    ys = np.arange(-width / 2.0, width / 2.0 + 1e-9, step)
    xs = np.unique(np.concatenate([xs, [length / 2.0]]))
    ys = np.unique(np.concatenate([ys, [width / 2.0]]))
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


@dataclass
class RunResult:
    state: str
    error: str
    sim_s: float
    final_pose: conv.Pose2D
    contacts: int
    first_contact: Optional[Tuple[float, float, float]]
    path_lengths: List[float] = field(default_factory=list)
    obstacle_states: List[str] = field(default_factory=list)
    reversing_s: float = 0.0
    trace: List[Tuple[float, float, float, float, float]] = field(default_factory=list)

    def length_flips(self, ratio: float = 1.4) -> int:
        """Count short↔long alternations of the committed global path."""
        lens = [l for l in self.path_lengths if l > 0]
        if len(lens) < 3:
            return 0
        flips = 0
        direction = 0
        base = lens[0]
        for l in lens[1:]:
            if l > base * ratio:
                d = 1
            elif l < base / ratio:
                d = -1
            else:
                continue
            if direction != 0 and d != direction:
                flips += 1
            direction = d
            base = l
        return flips


class RectWorld:
    """Ground-truth world implementing ``WorldIO`` for the supervisor."""

    def __init__(
        self,
        nav_grid: np.ndarray,
        truth_grid: np.ndarray,
        pose: conv.Pose2D,
        *,
        length: float = LENGTH,
        width: float = WIDTH,
        scan_bins: int = 360,
        range_max: float = 8.0,
        range_noise_m: float = 0.0,
        pose_noise_m: float = 0.0,
        pose_noise_rad: float = 0.0,
        seed: int = 0,
        lidar_offset: Tuple[float, float] = (0.0, 0.0),
        lidar_min_range_m: float = 0.0,
        body_only_grid: Optional[np.ndarray] = None,
        scan_latency_s: float = 0.0,
        yaw_gain: float = 1.0,
        rot_center_fwd_m: float = 0.0,
    ):
        self.rng = np.random.default_rng(seed)
        # Real-robot error sources: scans stamped "now" but taken earlier, and
        # a tracked base that under/over-rotates and pivots off-centre.
        self.scan_latency_s = float(scan_latency_s)
        self.yaw_gain = float(yaw_gain)
        self.rot_center_fwd_m = float(rot_center_fwd_m)
        self._pose_hist: List[Tuple[float, conv.Pose2D]] = []
        # Lidar mount in the body frame (fwd, left) and the driver's min range
        # (returns closer than this to the *lidar* are dropped).
        self.lidar_offset = (float(lidar_offset[0]), float(lidar_offset[1]))
        self.lidar_min_range_m = float(lidar_min_range_m)
        # Obstacle parts the body hits but the lidar plane never sees (a table
        # leg that angles in below lidar height).
        self.body_only = (
            body_only_grid >= 50
            if body_only_grid is not None
            else np.zeros_like(truth_grid, dtype=bool)
        )
        self.range_noise_m = float(range_noise_m)
        self.pose_noise_m = float(pose_noise_m)
        self.pose_noise_rad = float(pose_noise_rad)
        self._pose_err = (0.0, 0.0, 0.0)
        self.nav_grid = nav_grid
        self._static_truth = truth_grid >= 50
        self.truth = self._static_truth
        self.elapsed_s = 0.0
        # (x0, y0, x1, y1, t_on, t_off): a person stepping in and out.
        self.timed: List[Tuple[float, float, float, float, float, float]] = []
        self._timed_active: Tuple[bool, ...] = ()
        self.pose = pose
        self.samples = _body_samples(length, width, RES / 2.0)
        self.scan_bins = scan_bins
        self.range_max = range_max
        self.cmd = (0.0, 0.0)
        self.contacts = 0
        self.first_contact: Optional[Tuple[float, float, float]] = None
        self._in_contact = False
        self._scan: Optional[conv.LaserScan2D] = None
        self._scan_age = 1e9
        self.reversing_s = 0.0
        self.trace: List[Tuple[float, float, float, float, float]] = []
        self.clock: Optional[FakeClock] = None
        assert not self.collides(pose), "start pose overlaps an obstacle"

    # --- geometry -----------------------------------------------------
    def collides(self, p: conv.Pose2D) -> bool:
        c, s = math.cos(p.theta), math.sin(p.theta)
        wx = p.x + self.samples[:, 0] * c - self.samples[:, 1] * s
        wy = p.y + self.samples[:, 0] * s + self.samples[:, 1] * c
        cols = np.floor(wx / RES).astype(int)
        rows = np.floor(wy / RES).astype(int)
        h, w = self.truth.shape
        if (rows < 0).any() or (cols < 0).any() or (rows >= h).any() or (cols >= w).any():
            return True
        return bool(self.truth[rows, cols].any() or self.body_only[rows, cols].any())

    def _update_timed(self) -> None:
        active = tuple(t0 <= self.elapsed_s < t1 for *_, t0, t1 in self.timed)
        if active == self._timed_active:
            return
        self._timed_active = active
        truth = self._static_truth.copy()
        for (x0, y0, x1, y1, _, _), on in zip(self.timed, active):
            if on:
                c0, c1 = int(math.floor(x0 / RES)), int(math.ceil(x1 / RES))
                r0, r1 = int(math.floor(y0 / RES)), int(math.ceil(y1 / RES))
                truth[r0:r1, c0:c1] = True
        self.truth = truth
        self._scan_age = 1e9

    def advance(self, dt: float) -> None:
        self.elapsed_s += dt
        self._update_timed()
        sub = 0.01
        t = 0.0
        vx, vth = self.cmd
        while t < dt - 1e-12:
            h = min(sub, dt - t)
            t += h
            if vx < -1e-4:
                self.reversing_s += h
            if abs(vx) < 1e-9 and abs(vth) < 1e-9:
                continue
            w = vth * self.yaw_gain
            th = self.pose.theta + w * h
            mid = self.pose.theta + 0.5 * w * h
            # Pivot about a point ``rot_center_fwd_m`` ahead of the centre:
            # the centre then moves sideways at -w * offset.
            lat = -w * self.rot_center_fwd_m
            nx = self.pose.x + (vx * math.cos(mid) - lat * math.sin(mid)) * h
            ny = self.pose.y + (vx * math.sin(mid) + lat * math.cos(mid)) * h
            trial = conv.Pose2D(nx, ny, conv.normalize_angle(th))
            if self.collides(trial):
                if not self._in_contact:
                    self.contacts += 1
                    if self.first_contact is None:
                        self.first_contact = (self.pose.x, self.pose.y, self.pose.theta)
                self._in_contact = True
                continue
            self._in_contact = False
            self.pose = trial
        self._scan_age += dt
        if self.scan_latency_s > 0.0:
            self._pose_hist.append((self.elapsed_s, self.pose))
            cutoff = self.elapsed_s - 2.0 * self.scan_latency_s - 0.1
            while self._pose_hist and self._pose_hist[0][0] < cutoff:
                self._pose_hist.pop(0)

    def _lidar_pose(self) -> conv.Pose2D:
        """True pose the scan is taken from (``scan_latency_s`` in the past)."""
        if self.scan_latency_s <= 0.0 or not self._pose_hist:
            return self.pose
        t = self.elapsed_s - self.scan_latency_s
        for ts, p in reversed(self._pose_hist):
            if ts <= t:
                return p
        return self._pose_hist[0][1]

    def _raycast(self) -> np.ndarray:
        if (
            self.lidar_offset != (0.0, 0.0)
            or self.lidar_min_range_m > 0.0
            or self.scan_latency_s > 0.0
        ):
            return self._raycast_mounted()
        n = self.scan_bins
        ang = self.pose.theta - math.pi + np.arange(n) * (2.0 * math.pi / n)
        r = np.arange(0.02, self.range_max, RES / 2.0)
        xs = self.pose.x + np.cos(ang)[:, None] * r[None, :]
        ys = self.pose.y + np.sin(ang)[:, None] * r[None, :]
        cols = np.floor(xs / RES).astype(int)
        rows = np.floor(ys / RES).astype(int)
        h, w = self.truth.shape
        inb = (rows >= 0) & (cols >= 0) & (rows < h) & (cols < w)
        hit = np.zeros_like(inb)
        hit[inb] = self.truth[rows[inb], cols[inb]]
        hit |= ~inb
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        out = np.full(n, np.inf)
        out[any_hit] = r[first[any_hit]]
        if self.range_noise_m > 0.0:
            out[any_hit] += self.rng.normal(0.0, self.range_noise_m, int(any_hit.sum()))
        return out

    def _raycast_mounted(self) -> np.ndarray:
        """Rays from the real mount, min-range filtered, re-binned about the base
        centre (what ``prepare_lidar_point_cloud`` + ``points_to_scan`` give)."""
        n = 720
        P = self._lidar_pose()
        c, s = math.cos(P.theta), math.sin(P.theta)
        f, l = self.lidar_offset
        lx = P.x + c * f - s * l
        ly = P.y + s * f + c * l
        ang = P.theta - math.pi + np.arange(n) * (2.0 * math.pi / n)
        r = np.arange(0.02, self.range_max, RES / 4.0)
        xs = lx + np.cos(ang)[:, None] * r[None, :]
        ys = ly + np.sin(ang)[:, None] * r[None, :]
        cols = np.floor(xs / RES).astype(int)
        rows = np.floor(ys / RES).astype(int)
        h, w = self.truth.shape
        inb = (rows >= 0) & (cols >= 0) & (rows < h) & (cols < w)
        hit = np.zeros_like(inb)
        hit[inb] = self.truth[rows[inb], cols[inb]]
        hit |= ~inb
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        rng = r[first]
        if self.range_noise_m > 0.0:
            rng = rng + self.rng.normal(0.0, self.range_noise_m, n)
        keep = any_hit & (rng >= self.lidar_min_range_m)
        wx = lx + np.cos(ang[keep]) * rng[keep]
        wy = ly + np.sin(ang[keep]) * rng[keep]
        dx, dy = wx - P.x, wy - P.y
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        scan = conv.points_to_scan(
            np.stack([bx, by], axis=1),
            angle_min=-math.pi,
            angle_max=math.pi,
            num_bins=self.scan_bins,
            range_min=0.0,
            range_max=self.range_max,
        )
        return np.asarray(scan.ranges, dtype=float)

    def _estimated_pose(self) -> conv.Pose2D:
        """True pose + slowly wandering localisation error (SLAM-like)."""
        if self.pose_noise_m <= 0.0 and self.pose_noise_rad <= 0.0:
            return self.pose
        ex, ey, et = self._pose_err
        a = 0.97
        ex = a * ex + self.rng.normal(0.0, self.pose_noise_m * 0.25)
        ey = a * ey + self.rng.normal(0.0, self.pose_noise_m * 0.25)
        et = a * et + self.rng.normal(0.0, self.pose_noise_rad * 0.25)
        self._pose_err = (ex, ey, et)
        return conv.Pose2D(self.pose.x + ex, self.pose.y + ey, conv.normalize_angle(self.pose.theta + et))

    # --- WorldIO ------------------------------------------------------
    def get_map(self) -> dict:
        return {
            "grid": self.nav_grid,
            "resolution": RES,
            "origin_x": 0.0,
            "origin_y": 0.0,
            "generation": 1,
        }

    def get_pose(self) -> conv.Pose2D:
        p = self._estimated_pose()
        return conv.Pose2D(p.x, p.y, p.theta)

    def get_scan(self, max_age_s: float = 2.0, *, include_obstacles_only: bool = True):
        if self._scan is None or self._scan_age >= 0.1:
            self._scan = conv.LaserScan2D(
                self._raycast(),
                angle_min=-math.pi,
                angle_increment=2.0 * math.pi / self.scan_bins,
                range_min=0.05,
                range_max=self.range_max,
                capture_pose=self.get_pose(),
            )
            self._scan_age = 0.0
        return self._scan

    def set_velocity(self, vx: float, vy: float, vtheta: float) -> None:
        from src.nav_builtin.viam_io import _sanitize_base_cmd

        vx, vy, vtheta = _sanitize_base_cmd(vx, vy, vtheta)
        self.cmd = (float(vx), float(vtheta))
        if self.clock is not None:
            self.trace.append((self.clock.t, self.pose.x, self.pose.y, float(vx), float(vtheta)))

    def stop(self) -> None:
        self.cmd = (0.0, 0.0)

    def set_viz_plan(self, path_xy, goal=None) -> None:
        return None

    def set_viz_costmap(self, costmap) -> None:
        return None

    def set_viz_local_costmap(self, costmap) -> None:
        return None

    def get_localization_hold(self):
        return None

    def check_localization(self, **kwargs):
        return None


def blank(w_m: float, h_m: float) -> np.ndarray:
    g = np.zeros((int(round(h_m / RES)), int(round(w_m / RES))), dtype=np.int16)
    g[0, :] = g[-1, :] = 100
    g[:, 0] = g[:, -1] = 100
    return g


def box(g: np.ndarray, x0: float, y0: float, x1: float, y1: float, v: int = 100) -> None:
    c0, c1 = int(math.floor(x0 / RES)), int(math.ceil(x1 / RES))
    r0, r1 = int(math.floor(y0 / RES)), int(math.ceil(y1 / RES))
    g[r0:r1, c0:c1] = v


def run(
    nav_grid: np.ndarray,
    start: conv.Pose2D,
    goal: conv.Pose2D,
    *,
    extra_obstacles: Sequence[Tuple[float, float, float, float]] = (),
    timed_obstacles: Sequence[Tuple[float, float, float, float, float, float]] = (),
    clearance_m: float = 0.03,
    max_sim_s: float = 120.0,
    monkeypatch=None,
    nav_overrides: Optional[dict] = None,
    noise: bool = False,
    seed: int = 0,
    lidar_offset: Tuple[float, float] = (0.0, 0.0),
    lidar_min_range_m: float = 0.0,
    body_only_obstacles: Sequence[Tuple[float, float, float, float]] = (),
    scan_latency_s: float = 0.0,
    yaw_gain: float = 1.0,
    rot_center_fwd_m: float = 0.0,
) -> RunResult:
    truth = nav_grid.copy()
    for ob in extra_obstacles:
        box(truth, *ob)
    body_only = np.zeros_like(nav_grid)
    for ob in body_only_obstacles:
        box(body_only, *ob)
    world = RectWorld(
        nav_grid,
        truth,
        start,
        range_noise_m=0.01 if noise else 0.0,
        pose_noise_m=0.02 if noise else 0.0,
        pose_noise_rad=math.radians(1.0) if noise else 0.0,
        seed=seed,
        lidar_offset=lidar_offset,
        lidar_min_range_m=lidar_min_range_m,
        body_only_grid=body_only,
        scan_latency_s=scan_latency_s,
        yaw_gain=yaw_gain,
        rot_center_fwd_m=rot_center_fwd_m,
    )
    world.timed = list(timed_obstacles)
    clock = FakeClock(world)
    world.clock = clock
    fake_time = types.SimpleNamespace(
        monotonic=clock.monotonic,
        time=clock.time,
        perf_counter=clock.perf_counter,
        sleep=clock.sleep,
    )
    assert monkeypatch is not None
    monkeypatch.setattr(sup_mod, "time", fake_time)
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "base",
            "kinematics": "differential",
            "control_rate_hz": 20,
            "footprint_length_m": LENGTH,
            "footprint_width_m": WIDTH,
            "clearance_m": clearance_m,
            "max_vel_x": 0.4,
            "max_vel_theta": 1.0,
            **(nav_overrides or {}),
        }
    )
    sup = NavSupervisor(world, cfg, nav_loc_refine_on_disagree=False)
    lengths: List[float] = []
    states: List[str] = []
    orig_sleep = clock.sleep

    def sampling_sleep(dt: float) -> None:
        st = sup.status()
        if st.length_m and (not lengths or abs(lengths[-1] - st.length_m) > 1e-6):
            lengths.append(float(st.length_m))
        obs = (st.progress or {}).get("obstacle")
        if obs and (not states or states[-1] != obs):
            states.append(str(obs))
        orig_sleep(dt)

    clock.sleep = sampling_sleep  # type: ignore[method-assign]
    fake_time.sleep = sampling_sleep
    clock.limit_s = max_sim_s
    clock.on_limit = sup.request_cancel
    sup.run_goal(goal)
    st = sup.status()
    return RunResult(
        state=st.state,
        error=st.error_msg or "",
        sim_s=clock.t - 1000.0,
        final_pose=world.get_pose(),
        contacts=world.contacts,
        first_contact=world.first_contact,
        path_lengths=lengths,
        obstacle_states=states,
        reversing_s=world.reversing_s,
        trace=world.trace,
    )

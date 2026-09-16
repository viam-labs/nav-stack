"""Closed-loop path-following regression tests.

Drives a unicycle (diff-drive) model with the real controller stack —
``compute_path_command`` → base sanitizer → one control tick of latency —
under SLAM-like pose noise. These catch the failure modes that unit tests on
a single tick cannot: crosstrack blow-up at corners, and vθ chatter / spin
toggling that shows up on the robot as "crazy arcs".
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from src.nav.simple_motion import ObstacleConfig, SimpleMotionConfig
from src.nav_builtin.controller import (
    FollowerConfig,
    compute_path_command,
    limit_twist_rate,
    update_speed_estimate,
)
from src.nav_builtin.path_utils import signed_crosstrack_m
from src.nav_builtin.types import Path2D, Pose2D
from src.nav_builtin.viam_io import _sanitize_base_cmd
from src.geom import conversions as conv


def _densify(pts: List[Tuple[float, float]], step: float = 0.1) -> Tuple[Tuple[float, float], ...]:
    out: List[Tuple[float, float]] = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return tuple(out)


def _robot_cfg(*, obstacle: Optional[ObstacleConfig] = None) -> FollowerConfig:
    """Skid-steer with a 0.45 m footprint, like the field robot."""
    cfg = FollowerConfig()
    cfg.motion = SimpleMotionConfig(
        xy_tolerance_m=0.25,
        yaw_tolerance_rad=math.radians(10.0),
        max_linear_mps=0.4,
        max_angular_rad_s=1.0,
        min_linear_mps=0.12,
        min_angular_rad_s=0.15,
    )
    cfg.obstacle = obstacle
    return cfg


def _uniform_scan(front_range_m: float, num_bins: int = 360) -> conv.LaserScan2D:
    """Scan with a wall at ``front_range_m`` everywhere (forces steady 'slow')."""
    ranges = np.full(num_bins, float(front_range_m))
    return conv.LaserScan2D(ranges, -math.pi, 2 * math.pi / num_bins, range_min=0.05)


@dataclass(frozen=True)
class BaseModel:
    """Real skid-steer / Viam base effects the ideal unicycle ignores."""

    half_track_m: float = 0.28  # (vx=0.12, ω=0.25) is the empirical sanitizer edge
    wheel_min_mps: float = 0.05  # inner wheel below this → "nearly 0 RPM" rejection
    ang_deadband: float = 0.12  # no yaw response below this while translating
    slip: float = 0.85  # actual ω / commanded ω while translating
    tau_v_s: float = 0.4  # first-order actuator lags
    tau_w_s: float = 0.25
    pose_lag_ticks: int = 2  # SLAM pose latency in control ticks


IDEAL = None
HARSH = BaseModel()


@dataclass
class RunLog:
    crosstrack: List[float]
    omega: List[float]
    vx: List[float]
    final: Pose2D
    reached: bool
    ticks: int
    rejections: int = 0
    max_heading_err: float = 0.0

    def sign_flips(self, deadband: float = 0.1, window: int = 5) -> int:
        """Count *sustained* vθ sign reversals (chatter / S-curve metric).

        Uses a ``window``-tick moving average so a single 100 ms blip from
        pose noise does not count; alternating arcs lasting ≥ half a second do.
        """
        if not self.omega:
            return 0
        kernel = np.ones(window) / window
        smooth = np.convolve(np.asarray(self.omega), kernel, mode="valid")
        flips = 0
        last = 0.0
        for w in smooth:
            if abs(w) < deadband:
                continue
            if last != 0.0 and (w > 0) != (last > 0):
                flips += 1
            last = w
        return flips

    def spin_toggles(self) -> int:
        """Count transitions between translating and pure-spin while moving."""
        toggles = 0
        prev_spin: Optional[bool] = None
        for v, w in zip(self.vx, self.omega):
            if v == 0.0 and w == 0.0:
                continue
            spin = v == 0.0
            if prev_spin is not None and spin != prev_spin:
                toggles += 1
            prev_spin = spin
        return toggles


def _run(
    path: Path2D,
    start: Pose2D,
    *,
    cfg: FollowerConfig,
    scan: Optional[conv.LaserScan2D] = None,
    dt: float = 0.1,
    latency_ticks: int = 1,
    noise_xy_m: float = 0.015,
    noise_yaw_rad: float = math.radians(1.5),
    max_ticks: int = 1500,
    stop_dist_m: float = 0.5,
    seed: int = 0,
    base: Optional[BaseModel] = IDEAL,
    slew_limit: bool = False,
) -> RunLog:
    rng = random.Random(seed)
    true = start
    goal = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
    queue: deque = deque([(0.0, 0.0)] * latency_ticks)
    pose_lag = base.pose_lag_ticks if base is not None else 0
    pose_hist: deque = deque([start] * (pose_lag + 1), maxlen=pose_lag + 1)
    rotate_active = False
    last_vx = 0.0
    prev_cmd = None
    v_act = w_act = 0.0
    log = RunLog([], [], [], true, False, 0)

    for tick in range(max_ticks):
        lagged = pose_hist[0]
        meas = Pose2D(
            lagged.x + rng.gauss(0.0, noise_xy_m),
            lagged.y + rng.gauss(0.0, noise_xy_m),
            conv.normalize_angle(lagged.theta + rng.gauss(0.0, noise_yaw_rad)),
        )
        cmd, progress = compute_path_command(
            meas,
            path,
            cfg=cfg,
            scan=scan,
            speed_mps=last_vx,
            rotate_active=rotate_active,
            prev_cmd=prev_cmd,
        )
        rotate_active = bool(progress.get("rotate_to_heading"))
        if slew_limit:
            # Mirrors the supervisor's final gate before SetVelocity.
            cmd = limit_twist_rate(cmd, prev_cmd, cfg=cfg, dt_s=dt)
        last_vx = update_speed_estimate(last_vx, cmd.vx)  # mirrors supervisor
        prev_cmd = cmd
        vx, _vy, w = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
        if base is not None and vx > 0.0 and vx - abs(w) * base.half_track_m < base.wheel_min_mps:
            # Base rejects "nearly 0 RPM"; ViamWorldIO retries with a wider arc.
            log.rejections += 1
            vx = max(vx, 0.06 + 0.32 * abs(w)) if abs(w) >= 0.08 else 0.0
            if vx == 0.0:
                w = 0.0
        queue.append((vx, w))
        vx_b, w_b = queue.popleft()
        if base is None:
            vx_a, w_a = vx_b, w_b
        else:
            v_act += (vx_b - v_act) * min(1.0, dt / base.tau_v_s)
            if vx_b == 0.0:
                w_tgt = w_b
            elif abs(w_b) < base.ang_deadband:
                w_tgt = 0.0
            else:
                w_tgt = w_b * base.slip
            w_act += (w_tgt - w_act) * min(1.0, dt / base.tau_w_s)
            vx_a, w_a = v_act, w_act
        # Unicycle integration (midpoint heading).
        th_mid = true.theta + 0.5 * w_a * dt
        true = Pose2D(
            true.x + vx_a * math.cos(th_mid) * dt,
            true.y + vx_a * math.sin(th_mid) * dt,
            conv.normalize_angle(true.theta + w_a * dt),
        )
        pose_hist.append(true)
        ct, path_yaw = signed_crosstrack_m(true, path)
        log.crosstrack.append(ct)
        log.omega.append(w_a)
        log.vx.append(vx_a)
        if vx_a > 0.05:
            herr = abs(conv.normalize_angle(true.theta - path_yaw))
            log.max_heading_err = max(log.max_heading_err, herr)
        log.ticks = tick + 1
        if math.hypot(goal.x - true.x, goal.y - true.y) <= stop_dist_m:
            log.reached = True
            break
    log.final = true
    return log


def _straight() -> Path2D:
    return Path2D(points=_densify([(0.0, 0.0), (5.0, 0.0)]), goal_theta=0.0)


def _l_corner() -> Path2D:
    return Path2D(points=_densify([(0.0, 0.0), (3.0, 0.0), (3.0, 3.0)]), goal_theta=math.pi / 2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_straight_path_no_chatter(seed: int):
    """On a straight path with SLAM noise, stay on the line and don't zig-zag."""
    log = _run(_straight(), Pose2D(0.0, 0.0, 0.0), cfg=_robot_cfg(), seed=seed)
    assert log.reached, f"did not reach goal region in {log.ticks} ticks"
    assert max(abs(c) for c in log.crosstrack) < 0.08
    assert log.sign_flips() <= 2
    assert log.spin_toggles() == 0
    # Actually cruising, not crawling.
    moving = [v for v in log.vx if v > 0]
    assert np.mean(moving) > 0.3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_corner_bounded_cut_and_recovery(seed: int):
    """90° corner: bounded inside-cut, re-acquire the line, no spin toggling."""
    log = _run(_l_corner(), Pose2D(0.0, 0.0, 0.0), cfg=_robot_cfg(), seed=seed)
    assert log.reached
    # Longer L cuts a bit more of the corner; still well inside clearance preference.
    assert max(abs(c) for c in log.crosstrack) < 0.32
    # Once past the corner (heading north on x=3), we are back on the line.
    after = [c for c, v in zip(log.crosstrack, log.vx) if v > 0][-30:]
    assert max(abs(c) for c in after) < 0.12
    assert log.spin_toggles() == 0
    # One corner → at most one real direction change plus small settling.
    assert log.sign_flips() <= 3


@pytest.mark.parametrize("seed", [0, 1])
def test_slew_limited_corner_still_tracks(seed: int):
    """The final slew gate must smooth commands without loosening tracking."""
    log = _run(
        _l_corner(),
        Pose2D(0.0, 0.0, 0.0),
        cfg=_robot_cfg(),
        seed=seed,
        slew_limit=True,
    )
    assert log.reached
    assert max(abs(c) for c in log.crosstrack) < 0.32
    after = [c for c, v in zip(log.crosstrack, log.vx) if v > 0][-30:]
    assert max(abs(c) for c in after) < 0.12
    assert log.spin_toggles() == 0
    assert log.sign_flips() <= 3


def test_offset_start_converges_without_overshoot():
    """Start 0.35 m left of the path: converge smoothly, no oscillation across it."""
    log = _run(_straight(), Pose2D(0.0, 0.35, 0.0), cfg=_robot_cfg())
    assert log.reached
    # Longer lookahead + κ smoothing overshoots a few more cm than Stanley-style
    # gain, but must not reverse into a snake.
    assert min(log.crosstrack) > -0.12
    # Settled near the line (inside / near the crosstrack deadband).
    assert max(abs(c) for c in log.crosstrack[-30:]) < 0.10
    assert log.sign_flips() <= 2


def test_misaligned_start_rotates_then_drives():
    """Facing 120° off: rotate to heading first, then translate — no toggling."""
    log = _run(_straight(), Pose2D(0.0, 0.0, math.radians(120.0)), cfg=_robot_cfg())
    assert log.reached
    # Early ticks are pure rotation.
    first_moving = next(i for i, v in enumerate(log.vx) if v > 0)
    assert first_moving >= 5
    assert all(v == 0.0 for v in log.vx[:first_moving])
    # After starting to translate, never fall back to a spin.
    assert all(v > 0.0 for v in log.vx[first_moving:])
    assert max(abs(c) for c in log.crosstrack[-30:]) < 0.10


@pytest.mark.parametrize("seed", [0, 1])
def test_corner_while_obstacle_slowing_keeps_arc(seed: int):
    """Steady 'slow' band (wall at 0.7 m, stop 0.5, slow 1.0) must not tighten the
    arc or trip the base sanitizer into pure spins — the rc94 'crazy arcs'."""
    obstacle = ObstacleConfig(enabled=True, stop_distance_m=0.5, slow_distance_m=1.0)
    cfg = _robot_cfg(obstacle=obstacle)
    scan = _uniform_scan(0.7)
    log = _run(_l_corner(), Pose2D(0.0, 0.0, 0.0), cfg=cfg, scan=scan, seed=seed, max_ticks=4000)
    assert log.reached
    # Slowed, but every translating command stays drivable (vx ≥ crawl floor).
    moving = [v for v in log.vx if v > 0]
    assert 0.12 <= min(moving)
    assert np.mean(moving) < 0.3
    assert log.spin_toggles() == 0
    assert max(abs(c) for c in log.crosstrack) < 0.25
    assert log.sign_flips() <= 3


def test_latency_two_ticks_still_stable():
    """Heavier command latency (200 ms) must degrade gracefully, not oscillate."""
    log = _run(_l_corner(), Pose2D(0.0, 0.0, 0.0), cfg=_robot_cfg(), latency_ticks=2)
    assert log.reached
    assert max(abs(c) for c in log.crosstrack) < 0.3
    assert log.sign_flips() <= 4
    assert log.spin_toggles() == 0


# --- Real-base model: 5 Hz loop, 0.4 s pose latency, 3 cm / 3° SLAM noise,
# actuator lag, yaw deadband + slip, and the wheeled base's inner-wheel
# "nearly 0 RPM" rejection. This is what rc95 failed on the robot with:
# regulated corner arcs at 0.15 m/s / 0.7 rad/s were rejected and retried as
# pure spins → heading jumps → rotate-to-heading → stop-spin-go.

_HARSH = dict(base=HARSH, dt=0.2, noise_xy_m=0.03, noise_yaw_rad=math.radians(3.0))


def _dogleg() -> Path2D:
    return Path2D(
        points=_densify([(0.0, 0.0), (2.0, 0.0), (3.5, 1.2), (6.0, 1.2)]), goal_theta=0.0
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harsh_corner_never_rejected_by_base(seed: int):
    log = _run(_l_corner(), Pose2D(0.0, 0.0, 0.0), cfg=_robot_cfg(), seed=seed, **_HARSH)
    assert log.reached
    # Every translating command is inside the wheel envelope → no spin retries.
    assert log.rejections == 0
    assert log.spin_toggles() == 0
    # One continuous arc: bounded cut, and heading never diverges from the path.
    assert max(abs(c) for c in log.crosstrack) < 0.3
    assert log.max_heading_err < math.radians(110.0)
    assert log.sign_flips() <= 3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harsh_straight_keeps_heading(seed: int):
    """SLAM noise at 5 Hz must not become heading wag: the base should see
    almost no yaw commands, and actual heading stays within a few degrees."""
    log = _run(_straight(), Pose2D(0.0, 0.0, math.radians(4.0)), cfg=_robot_cfg(), seed=seed, **_HARSH)
    assert log.reached
    assert log.rejections == 0
    # Longer L + κ EMA lag a bit under harsh 5 Hz noise; still no zig-zag.
    assert max(abs(c) for c in log.crosstrack) < 0.22
    assert log.max_heading_err < math.radians(13.0)
    assert log.sign_flips() <= 2


@pytest.mark.parametrize("seed", [0, 1])
def test_harsh_dogleg_and_slow_band(seed: int):
    log = _run(_dogleg(), Pose2D(0.0, 0.0, 0.0), cfg=_robot_cfg(), seed=seed, **_HARSH)
    assert log.reached and log.rejections == 0 and log.spin_toggles() == 0
    assert max(abs(c) for c in log.crosstrack) < 0.28
    obstacle = ObstacleConfig(enabled=True, stop_distance_m=0.5, slow_distance_m=1.0)
    slow = _run(
        _l_corner(),
        Pose2D(0.0, 0.0, 0.0),
        cfg=_robot_cfg(obstacle=obstacle),
        scan=_uniform_scan(0.75),
        seed=seed,
        max_ticks=4000,
        **_HARSH,
    )
    assert slow.reached and slow.rejections == 0 and slow.spin_toggles() == 0
    assert max(abs(c) for c in slow.crosstrack) < 0.32

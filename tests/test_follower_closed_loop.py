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
    update_speed_estimate,
)
from src.nav_builtin.path_utils import signed_crosstrack_m
from src.nav_builtin.types import Path2D, Pose2D
from src.nav_builtin.viam_io import _sanitize_base_cmd
from src.ros import conversions as conv


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


@dataclass
class RunLog:
    crosstrack: List[float]
    omega: List[float]
    vx: List[float]
    final: Pose2D
    reached: bool
    ticks: int

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
) -> RunLog:
    rng = random.Random(seed)
    true = start
    goal = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
    queue: deque = deque([(0.0, 0.0)] * latency_ticks)
    rotate_active = False
    last_vx = 0.0
    log = RunLog([], [], [], true, False, 0)

    for tick in range(max_ticks):
        meas = Pose2D(
            true.x + rng.gauss(0.0, noise_xy_m),
            true.y + rng.gauss(0.0, noise_xy_m),
            conv.normalize_angle(true.theta + rng.gauss(0.0, noise_yaw_rad)),
        )
        cmd, progress = compute_path_command(
            meas,
            path,
            cfg=cfg,
            scan=scan,
            speed_mps=last_vx,
            rotate_active=rotate_active,
        )
        rotate_active = bool(progress.get("rotate_to_heading"))
        last_vx = update_speed_estimate(last_vx, cmd.vx)  # mirrors supervisor
        vx, _vy, w = _sanitize_base_cmd(cmd.vx, cmd.vy, cmd.vtheta)
        queue.append((vx, w))
        vx_a, w_a = queue.popleft()
        # Unicycle integration (midpoint heading).
        th_mid = true.theta + 0.5 * w_a * dt
        true = Pose2D(
            true.x + vx_a * math.cos(th_mid) * dt,
            true.y + vx_a * math.sin(th_mid) * dt,
            conv.normalize_angle(true.theta + w_a * dt),
        )
        ct, _ = signed_crosstrack_m(true, path)
        log.crosstrack.append(ct)
        log.omega.append(w_a)
        log.vx.append(vx_a)
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
    # Corner cut is bounded well inside the planner's clearance preference.
    assert max(abs(c) for c in log.crosstrack) < 0.25
    # Once past the corner (heading north on x=3), we are back on the line.
    after = [c for c, v in zip(log.crosstrack, log.vx) if v > 0][-30:]
    assert max(abs(c) for c in after) < 0.08
    assert log.spin_toggles() == 0
    # One corner → at most one real direction change plus small settling.
    assert log.sign_flips() <= 3


def test_offset_start_converges_without_overshoot():
    """Start 0.35 m left of the path: converge smoothly, no oscillation across it."""
    log = _run(_straight(), Pose2D(0.0, 0.35, 0.0), cfg=_robot_cfg())
    assert log.reached
    # Never swing past the line by more than a few cm.
    assert min(log.crosstrack) > -0.06
    # Settled on the line for the last stretch.
    assert max(abs(c) for c in log.crosstrack[-30:]) < 0.06
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
    assert max(abs(c) for c in log.crosstrack[-30:]) < 0.08


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

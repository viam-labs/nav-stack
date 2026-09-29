"""Footprint collision guard (Nav2 RPP / DWB-style) for a rectangular base.

The last stage of every control tick. Given the nominal twist (pursuit or
DWA), project the *oriented rectangle* along the commanded arc against the
actual obstacles — live lidar returns plus lethal cells of the local costmap
(static map walls, unknown space, persisted scan marks) — and:

1. regulate speed by time-to-collision along the arc (curvature preserved);
2. if the nominal arc is blocked within a short distance, pick the forward
   arc closest to the pursuit target that is collision-free;
3. if nothing forward is free, allow a rotation in place only when the
   rectangle's sweep is free; otherwise stop and report ``blocked``.

It never reverses: backing up is a supervisor recovery, not a reflex.

Obstacles are checked as points against the padded rectangle, so a 0.85 m
doorway is passable for a 0.59 m wide robot as long as the body actually
clears the jambs — no circular approximation of a rectangle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..geom import conversions as conv
from .costmap import LETHAL
from .types import Pose2D


@dataclass(frozen=True)
class GuardConfig:
    length_m: float
    width_m: float
    # Keep-out past the body for planning motion (lidar noise, cell
    # quantisation, localisation). Obstacles already inside this band at the
    # start pose may be passed alongside, but never closer than ``min_gap_m``.
    padding_m: float = 0.05
    min_gap_m: float = 0.02
    # Speed regulation: allow ``v`` only while the free arc length covers
    # ``v * time_to_collision_s + stop_gap_m``.
    time_to_collision_s: float = 1.2
    stop_gap_m: float = 0.04
    horizon_m: float = 0.8
    step_m: float = 0.025
    rot_step_rad: float = 0.04
    rot_horizon_rad: float = 0.6
    obstacle_radius_m: float = 2.0
    # Alternative forward arcs when the nominal arc is blocked.
    alt_speed_mps: float = 0.13
    alt_min_free_m: float = 0.20
    alt_max_curvature: float = 2.5
    alt_samples: int = 21
    # Base sanitizer: vx < 0.12 with |w| > 0.25 becomes a pure spin.
    base_spin_vx_floor: float = 0.125
    base_spin_w_cap: float = 0.25


@dataclass
class GuardResult:
    vx: float
    vtheta: float
    state: str  # clear | slow | steer | rotate | blocked
    free_m: float
    rotation_blocked: bool


def obstacle_points(
    pose: Pose2D,
    scan: Optional[conv.LaserScan2D],
    local_view,
    *,
    radius_m: float,
) -> np.ndarray:
    """World-frame obstacle points near ``pose``.

    Live scan returns (exact, current) plus *static* lethal cells of the local
    view (map walls). Persisted scan marks are skipped: they are
    projected with a stale pose, smear walls a cell inward, and in a 0.9 m
    corridor that alone reads as a collision.
    """
    chunks = []
    if scan is not None:
        pts = scan.to_points()
        if pts.size:
            ref = scan.capture_pose or pose
            c, s = math.cos(ref.theta), math.sin(ref.theta)
            wx = ref.x + c * pts[:, 0] - s * pts[:, 1]
            wy = ref.y + s * pts[:, 0] + c * pts[:, 1]
            chunks.append(np.stack([wx, wy], axis=1))
    if local_view is not None:
        costs = np.asarray(local_view.costs)
        res = float(local_view.occ.resolution)
        r0, c0 = local_view.world_to_cell(pose.x - radius_m, pose.y - radius_m)
        r1, c1 = local_view.world_to_cell(pose.x + radius_m, pose.y + radius_m)
        h, w = costs.shape
        r0, r1 = max(0, r0), min(h, r1 + 1)
        c0, c1 = max(0, c0), min(w, c1 + 1)
        if r1 > r0 and c1 > c0:
            # Observed-occupied only: unknown (255) speckle on open floor is
            # not inflated by the planner, so routes pass beside it and a
            # guard that treated it as solid stalled there (Nav2 RPP likewise
            # ignores NO_INFORMATION). Real unmapped obstacles come via scan.
            lethal = costs[r0:r1, c0:c1] == LETHAL
            raw = getattr(local_view.occ, "grid", None)
            if raw is not None and np.shape(raw) == costs.shape:
                lethal &= np.asarray(raw)[r0:r1, c0:c1] <= 0
            ys, xs = np.nonzero(lethal)
            if ys.size:
                wx = local_view.origin_x + (xs + c0 + 0.5) * res
                wy = local_view.origin_y + (ys + r0 + 0.5) * res
                chunks.append(np.stack([wx, wy], axis=1))
    if not chunks:
        return np.empty((0, 2))
    pts = np.concatenate(chunks, axis=0)
    d2 = (pts[:, 0] - pose.x) ** 2 + (pts[:, 1] - pose.y) ** 2
    return pts[d2 <= radius_m * radius_m]


def _arc_poses(pose: Pose2D, v: float, w: float, dist_m: float, step_m: float) -> np.ndarray:
    n = max(1, int(math.ceil(dist_m / step_m)))
    s = np.arange(1, n + 1) * (dist_m / n) * (1.0 if v >= 0 else -1.0)
    kappa = w / v if abs(v) > 1e-9 else 0.0
    th0 = pose.theta
    if abs(kappa) < 1e-6:
        xs = pose.x + s * math.cos(th0)
        ys = pose.y + s * math.sin(th0)
        ths = np.full_like(s, th0)
    else:
        ths = th0 + kappa * s
        xs = pose.x + (np.sin(ths) - math.sin(th0)) / kappa
        ys = pose.y - (np.cos(ths) - math.cos(th0)) / kappa
    return np.stack([xs, ys, ths], axis=1)


def _rot_poses(pose: Pose2D, sign: float, angle: float, step: float) -> np.ndarray:
    n = max(1, int(math.ceil(angle / step)))
    ths = pose.theta + sign * np.arange(1, n + 1) * (angle / n)
    return np.stack([np.full(n, pose.x), np.full(n, pose.y), ths], axis=1)


def _inside(poses: np.ndarray, pts: np.ndarray, hl: float, hw: float) -> np.ndarray:
    """(K, N) bool: point n inside the rectangle at pose k."""
    dx = pts[None, :, 0] - poses[:, 0:1]
    dy = pts[None, :, 1] - poses[:, 1:2]
    c = np.cos(poses[:, 2:3])
    s = np.sin(poses[:, 2:3])
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    return (np.abs(bx) <= hl) & (np.abs(by) <= hw)


class FootprintGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self._hl = cfg.length_m / 2.0
        self._hw = cfg.width_m / 2.0

    # --- primitives ---------------------------------------------------
    def _split(self, pose: Pose2D, pts: np.ndarray):
        """Split into (far, near) points relative to the current pose.

        Points inside the body itself are dropped (Nav2 footprint clearing:
        self-returns from the chassis/mast, or a stale mark we already
        occupy — a real obstacle there would already be a collision).
        Points between the body and the padding are only checked at
        ``min_gap_m`` so the robot can move along/away from a close wall.
        """
        p = self.cfg.padding_m
        if pts.size == 0:
            return pts, pts
        here = np.array([[pose.x, pose.y, pose.theta]])
        body = _inside(here, pts, self._hl, self._hw)[0]
        pts = pts[~body]
        near = _inside(here, pts, self._hl + p, self._hw + p)[0]
        return pts[~near], pts[near]

    def _first_hit(self, poses: np.ndarray, far: np.ndarray, near: np.ndarray) -> int:
        """Index of the first colliding pose, or len(poses)."""
        k = len(poses)
        hit = np.zeros(k, dtype=bool)
        p, g = self.cfg.padding_m, self.cfg.min_gap_m
        if far.size:
            hit |= _inside(poses, far, self._hl + p, self._hw + p).any(axis=1)
        if near.size:
            hit |= _inside(poses, near, self._hl + g, self._hw + g).any(axis=1)
        idx = np.flatnonzero(hit)
        return int(idx[0]) if idx.size else k

    def free_distance(
        self, pose: Pose2D, v: float, w: float, pts: np.ndarray, horizon_m: float
    ) -> float:
        far, near = self._split(pose, pts)
        poses = _arc_poses(pose, v, w, horizon_m, self.cfg.step_m)
        k = self._first_hit(poses, far, near)
        if k >= len(poses):
            return math.inf
        return k * (horizon_m / len(poses))

    def _free_distances_batch(
        self,
        pose: Pose2D,
        v: float,
        kappas: np.ndarray,
        pts: np.ndarray,
        horizon_m: float,
    ) -> np.ndarray:
        """Free arc length for many curvatures in one vectorised pass.

        Must sample at ``cfg.step_m`` like :meth:`free_distance`: the veto
        re-checks the chosen arc with it, and a coarser step here picks arcs
        the veto then zeroes (a dead stop beside the obstacle).
        """
        step_m = self.cfg.step_m
        out = np.full(len(kappas), math.inf)
        far, near = self._split(pose, pts)
        reach = horizon_m + math.hypot(self._hl, self._hw) + self.cfg.padding_m
        if far.size:
            d2 = (far[:, 0] - pose.x) ** 2 + (far[:, 1] - pose.y) ** 2
            far = far[d2 <= reach * reach]
        if far.size == 0 and near.size == 0:
            return out
        poses = np.concatenate(
            [_arc_poses(pose, v, v * float(k), horizon_m, step_m) for k in kappas]
        )
        k = len(poses) // len(kappas)
        p, g = self.cfg.padding_m, self.cfg.min_gap_m
        hit = np.zeros(len(poses), dtype=bool)
        if far.size:
            hit |= _inside(poses, far, self._hl + p, self._hw + p).any(axis=1)
        if near.size:
            hit |= _inside(poses, near, self._hl + g, self._hw + g).any(axis=1)
        hit = hit.reshape(len(kappas), k)
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        out[any_hit] = first[any_hit] * (horizon_m / k)
        return out

    def free_rotation(self, pose: Pose2D, sign: float, pts: np.ndarray) -> float:
        far, near = self._split(pose, pts)
        ang = self.cfg.rot_horizon_rad
        poses = _rot_poses(pose, sign, ang, self.cfg.rot_step_rad)
        k = self._first_hit(poses, far, near)
        if k >= len(poses):
            return math.inf
        return k * (ang / len(poses))

    def clearance(self, pose: Pose2D, pts: np.ndarray) -> float:
        """Distance from the (unpadded) body rectangle to the nearest point."""
        if pts.size == 0:
            return math.inf
        dx = pts[:, 0] - pose.x
        dy = pts[:, 1] - pose.y
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        bx = np.abs(c * dx + s * dy) - self._hl
        by = np.abs(-s * dx + c * dy) - self._hw
        d = np.hypot(np.maximum(bx, 0.0), np.maximum(by, 0.0))
        return float(d.min())

    # --- policy -------------------------------------------------------
    def _regulate(self, v: float, free_m: float) -> float:
        cfg = self.cfg
        if not math.isfinite(free_m):
            return abs(v)
        return max(0.0, min(abs(v), (free_m - cfg.stop_gap_m) / cfg.time_to_collision_s))

    def _drivable(self, v: float, w: float) -> Tuple[float, float]:
        """Keep an arc that the base sanitizer will not turn into a spin."""
        cfg = self.cfg
        if 0.0 < abs(v) < cfg.base_spin_vx_floor and abs(w) > cfg.base_spin_w_cap:
            w = math.copysign(cfg.base_spin_w_cap, w)
        return v, w

    def guard(
        self,
        pose: Pose2D,
        vx: float,
        vtheta: float,
        pts: np.ndarray,
        *,
        target: Optional[Pose2D] = None,
        max_dist_m: Optional[float] = None,
        prefer_sign: float = 0.0,
        allow_steer: bool = True,
    ) -> GuardResult:
        cfg = self.cfg
        horizon = cfg.horizon_m
        if max_dist_m is not None:
            horizon = max(0.1, min(horizon, float(max_dist_m) + cfg.stop_gap_m + 0.05))
        if abs(vx) < 1e-3:
            if abs(vtheta) < 1e-3:
                return GuardResult(0.0, 0.0, "clear", math.inf, False)
            free = self.free_rotation(pose, math.copysign(1.0, vtheta), pts)
            if free >= min(cfg.rot_horizon_rad, abs(vtheta) * 0.5 + cfg.rot_step_rad):
                return GuardResult(0.0, vtheta, "clear", math.inf, False)
            if allow_steer:
                alt = self.best_forward_arc(pose, pts, target, horizon, prefer_sign)
                if alt is not None:
                    return GuardResult(alt[0], alt[1], "steer", alt[2], True)
            return GuardResult(0.0, 0.0, "blocked", 0.0, True)

        free = self.free_distance(pose, vx, vtheta, pts, horizon)
        v_ok = self._regulate(vx, free)
        need = cfg.alt_min_free_m if allow_steer else cfg.stop_gap_m + 0.02
        if (
            allow_steer
            and vx > 0.0
            and math.isfinite(free)
            and free < horizon
        ):
            # Collision somewhere on the nominal arc: steer early if a clearly
            # freer forward arc exists, instead of closing to the stop gap.
            alt = self.best_forward_arc(
                pose, pts, target, horizon, prefer_sign, speed=abs(vx)
            )
            if alt is not None and alt[2] > free + 0.2:
                return GuardResult(alt[0], alt[1], "steer", alt[2], False)
        if v_ok >= min(abs(vx), 0.05) and free >= min(need, horizon):
            scale = v_ok / abs(vx)
            v, w = self._drivable(math.copysign(v_ok, vx), vtheta * scale)
            state = "clear" if scale >= 0.999 else "slow"
            return GuardResult(v, w, state, free, False)
        if vx > 0.0 and allow_steer:
            alt = self.best_forward_arc(pose, pts, target, horizon, prefer_sign)
            if alt is not None:
                return GuardResult(alt[0], alt[1], "steer", alt[2], False)
            # No better arc: the nominal one is short but free — crawl it.
            if v_ok >= 0.03:
                scale = v_ok / abs(vx)
                v, w = self._drivable(math.copysign(v_ok, vx), vtheta * scale)
                return GuardResult(v, w, "slow", free, False)
            # Nothing forward: rotate toward the target only if the sweep is free.
            if target is not None:
                bearing = conv.normalize_angle(
                    math.atan2(target.y - pose.y, target.x - pose.x) - pose.theta
                )
                sign = math.copysign(1.0, bearing) if abs(bearing) > 1e-3 else (
                    prefer_sign or 1.0
                )
                for sg in (sign, -sign):
                    if self.free_rotation(pose, sg, pts) >= cfg.rot_horizon_rad * 0.5:
                        return GuardResult(0.0, sg * 0.4, "rotate", free, False)
        return GuardResult(0.0, 0.0, "blocked", free, True)

    def best_forward_arc(
        self,
        pose: Pose2D,
        pts: np.ndarray,
        target: Optional[Pose2D],
        horizon: float,
        prefer_sign: float,
        speed: float = 0.0,
    ) -> Optional[Tuple[float, float, float]]:
        cfg = self.cfg
        v = max(cfg.alt_speed_mps, min(float(speed), 0.25))
        kappas = np.linspace(-cfg.alt_max_curvature, cfg.alt_max_curvature, cfg.alt_samples)
        frees = self._free_distances_batch(pose, v, kappas, pts, horizon)
        best = None
        best_score = math.inf
        for kappa, free in zip(kappas, frees):
            w = v * float(kappa)
            if free < cfg.alt_min_free_m:
                continue
            travel = min(free, horizon, 0.4)
            end = _arc_poses(pose, v, w, travel, travel)[-1]
            if target is not None:
                score = math.hypot(target.x - end[0], target.y - end[1])
                score += 0.3 * abs(
                    conv.normalize_angle(
                        math.atan2(target.y - end[1], target.x - end[0]) - end[2]
                    )
                )
            else:
                score = abs(float(kappa))
            score -= 0.3 * min(free, horizon)
            if prefer_sign and kappa * prefer_sign < 0:
                score += 0.15
            if score < best_score:
                best_score = score
                v_ok = max(0.05, min(v, self._regulate(v, free)))
                best = (*self._drivable(v_ok, w * v_ok / v), free)
        return best

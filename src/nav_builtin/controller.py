"""Path-following controller built on simple_motion P-control + obstacle cone."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..nav.simple_motion import (
    DriveCommand,
    ObstacleConfig,
    SimpleMotionConfig,
    apply_obstacle_avoidance,
    apply_velocity_floor,
    cone_min_range,
    distance_m,
    forward_clearance_m,
    heading_error_rad,
)
from ..geom import conversions as conv
from .local_costmap import LocalCostmapView
from .local_planner import LocalPlannerConfig, compute_local_command
from .path_utils import closest_point_on_path, signed_crosstrack_m
from .types import Path2D, Pose2D


@dataclass
class FollowerConfig:
    """Regulated Pure Pursuit (Nav2-style) for diff-drive / skid-steer.

    Lookahead is velocity-scaled: ``L = clamp(v · lookahead_time_s, min, max)``;
    ``lookahead_m`` is used when the commanded speed is unknown or ~0. Steering
    is geometric (``κ = 2·y_l / L²``, ``ω = v·κ``) so slowing down never
    tightens the turn, and the lookahead low-passes SLAM pose jitter.
    """

    # Longer lookahead damps soft-loc / heading jitter on skid-steer. Mid-path
    # S-curves still hunt around L≈1.0 m; 1.1–1.8 m trades a bit more corner
    # cut (still inside clearance_preference) for a straighter trail.
    lookahead_m: float = 1.35
    min_lookahead_m: float = 1.1
    max_lookahead_m: float = 1.8
    lookahead_time_s: float = 3.0
    # EMA weight on *new* curvature (rest from previous tick). Lower = calmer
    # mid-path ω; 0.25 ≈ 0.4 s memory at 10 Hz.
    curvature_smoothing: float = 0.25
    # Ignore |y_l| below this when computing κ so pose noise and densify jogs
    # do not flip vθ every tick on a long straight. Rotate-to-heading still
    # uses the raw bearing.
    crosstrack_deadband_m: float = 0.06
    approach_dist_m: float = 0.35
    waypoint_tolerance_m: float = 0.15
    # Rotate-to-heading: stop translating when the lookahead bearing exceeds
    # ``rotate_in_place_rad``; keep rotating until it drops under
    # ``rotate_exit_rad`` (hysteresis so we do not toggle spin/translate).
    # 60° (not Nav2's 45°): with curvature-regulated speed a 90° corner is a
    # slow tight arc, and a 45° trigger turned every corner into stop-spin-go.
    rotate_in_place_rad: float = math.radians(60.0)
    rotate_exit_rad: float = math.radians(25.0)
    rotate_vel_rad_s: float = 0.6
    # Curvature regulation: below this turn radius, scale v by r / r_min so the
    # robot slows into corners instead of carving them at cruise.
    regulated_min_radius_m: float = 0.7
    regulated_min_speed_mps: float = 0.15
    # Skid-steer wheel constraint. Inner wheel speed is ``vx - |vθ|·half_track``;
    # Viam wheeled bases reject commands whose inner wheel is "nearly 0" RPM and
    # the retry path turns the arc into a pure spin — which is how a regulated
    # corner arc became stop-spin-go on the robot. Arcs are kept such that
    # ``vx - |vθ|·half_track >= wheel_min_speed`` and radius ≥ min_turn_radius.
    # Supervisor sets half_track from robot_radius (≈0.6·r).
    wheel_half_track_m: float = 0.27
    wheel_min_speed_mps: float = 0.06
    min_turn_radius_m: Optional[float] = None
    # Command slew limits. The base has no onboard ramp, so every handoff
    # between command sources (pursuit / DWA / reactive avoid / planning stop)
    # used to step vx and vθ in a single tick — that is the visible jerk.
    # Braking to a standstill is exempt (see ``limit_twist_rate``).
    max_linear_accel_mps2: float = 0.8
    max_linear_decel_mps2: float = 1.2
    max_angular_accel_rad_s2: float = 2.0
    motion: SimpleMotionConfig = field(default_factory=SimpleMotionConfig)
    obstacle: Optional[ObstacleConfig] = None

    def effective_min_turn_radius_m(self) -> float:
        if self.min_turn_radius_m is not None:
            return max(0.05, float(self.min_turn_radius_m))
        return float(self.wheel_half_track_m) + 0.15


# ``ViamWorldIO._sanitize_base_cmd`` zeroes ``vx < 0.12`` when ``|vθ| > 0.25``
# (pure spin). Any translating arc we emit must clear that floor.
_BASE_CRAWL_FLOOR_MPS = 0.125


def _path_length(path: Path2D) -> float:
    pts = path.points
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _near_goal_command(
    *,
    yaw_err: float,
    bearing: float,
    dist: float,
    motion: SimpleMotionConfig,
) -> DriveCommand:
    """Hold / crawl near the goal without bearing-RIP oscillation.

    Outside ``xy_tolerance`` the normal law tracks bearing-to-point; a 1–2 cm
    overshoot makes that bearing ≈ ±π and commands ±max_vel_theta, then the
    next tick (back inside the ball) flips to final-yaw with the opposite
    sign — classic goal-swing.

    Keep closing XY until a few centimetres from the point, then pure-spin for
    final yaw. Supervisor may still accept XY-only after a yaw align timeout
    once inside the acceptance ball. We do not stop translating at the outer
    ``xy_tolerance`` ball — that left ~0.1–0.25 m residuals while hunting heading.

    Translating cmds must survive ``ViamWorldIO`` base sanitizer: it zeros
    ``|vx| < 0.12`` when ``|vθ| > 0.25`` (and ``|vx| < 0.05`` always). Tiny
    reverse crawls with yaw hunt therefore become pure spin — end wiggle
    with no XY progress.

    Inside ``xy_tolerance``, never pure-spin to face the *point* (that flip-flops
    with final-yaw spin). Soft-crawl or, when already close enough, spin for
    goal θ only.
    """
    xy_tol = motion.xy_tolerance_m
    yaw_tol = motion.yaw_tolerance_rad
    # Nail the point before hunting goal θ (acceptance ball stays xy_tol).
    xy_settle_m = xy_settle_radius_m(xy_tol)
    # Once inside acceptance, prefer final yaw over point-facing RIP so we do
    # not oscillate: face-point → crawl → overshoot → final-yaw → drift → repeat.
    final_yaw_hand_off_m = max(0.12, min(xy_tol * 0.5, 0.15))
    yaw_cap = min(0.40, motion.max_angular_rad_s)
    # Keep |vθ| under the sanitizer's 0.25 cut when also translating.
    translate_yaw_cap = min(yaw_cap, 0.22)
    # Floor above sanitizer lin_eps (0.05) and the tiny+turn kill (0.12).
    crawl_floor = 0.12
    soft_floor = 0.05  # lin_eps; pair with small vθ so soft crawl survives

    def _spin_yaw() -> DriveCommand:
        vtheta = _clamp(yaw_err * 0.85, yaw_cap)
        if abs(yaw_err) > yaw_tol and abs(vtheta) < 0.10:
            vtheta = math.copysign(0.10, yaw_err)
        return DriveCommand(0.0, 0.0, vtheta, False)

    def _close_xy(*, soft: bool = False) -> DriveCommand:
        """Close distance to the goal point without pure-spin face-the-point RIP.

        Soft mode never stops to rotate in place: that was the end-of-path
        swing (spin to point bearing, crawl, then spin the other way for θ).
        """
        floor = soft_floor if soft else crawl_floor
        # Soft: scale speed with remaining gap so a 6 cm approach doesn't
        # blast through the settle radius at 0.12 m/s.
        if soft:
            max_crawl = min(0.10, max(floor, dist * 1.2))
        else:
            max_crawl = 0.18
        if abs(bearing) > math.radians(100.0):
            crawl = min(
                max_crawl, max(floor, motion.max_linear_mps * (0.20 if soft else 0.25))
            )
            rev_bearing = conv.normalize_angle(bearing + math.pi)
            # Soft: reverse with almost no yaw so sanitizer keeps vx.
            vth = (
                _clamp(rev_bearing * 0.6, 0.15)
                if soft
                else _clamp(rev_bearing * 1.2, translate_yaw_cap)
            )
            return apply_velocity_floor(
                DriveCommand(
                    -max(floor, min(crawl, dist * 0.8)),
                    0.0,
                    vth,
                    False,
                ),
                motion,
            )
        if abs(bearing) > math.radians(45.0) and not soft:
            # Outside acceptance: face the point first.
            return apply_velocity_floor(
                DriveCommand(0.0, 0.0, _clamp(bearing * 1.5, yaw_cap), False),
                motion,
            )
        crawl = min(
            max_crawl, max(floor, motion.max_linear_mps * (0.20 if soft else 0.30))
        )
        vx = max(floor, min(crawl, dist * (1.2 if soft else 0.8)))
        vth = _clamp(bearing * (1.0 if soft else 1.5), translate_yaw_cap)
        if soft:
            # Keep |vθ| low so soft |vx| survives the sanitizer — even with
            # large bearing (crawl while turning instead of RIP).
            vth = _clamp(bearing * 0.7, 0.15)
            if abs(bearing) > math.radians(60.0):
                # Prefer slow reverse/forward over a long in-place swing.
                vx = max(floor, min(vx, 0.08))
        elif abs(bearing) < math.radians(25.0):
            vth = _clamp(bearing * 0.8, 0.15)
        return apply_velocity_floor(
            DriveCommand(vx, 0.0, vth, False),
            motion,
        )

    # Tight on the point: only final yaw remains.
    if dist <= xy_settle_m:
        return _spin_yaw()

    # Inside acceptance ball: soft-close XY, or hand off to final yaw once
    # close enough that another face-point spin would start the end wiggle.
    if dist <= xy_tol:
        if abs(yaw_err) > yaw_tol and dist <= final_yaw_hand_off_m:
            return _spin_yaw()
        return _close_xy(soft=True)

    # Still outside XY tolerance: ignore final yaw and close on the point.
    return _close_xy(soft=False)


def xy_settle_radius_m(xy_tolerance_m: float) -> float:
    """Radius at which near-goal stops translating and spins for final yaw.

    Kept small so we park on the point when possible. Below ~3 cm, soft crawl
    (sanitizer lin_eps 0.05 m/s) mostly overshoots, so we spin instead.
    """
    del xy_tolerance_m  # acceptance ball is separate; settle is physical
    return 0.03


def _effective_lookahead(
    cfg: FollowerConfig,
    *,
    speed_mps: float,
    near_goal: bool = False,
) -> float:
    """Velocity-scaled lookahead (RPP): ``clamp(v · t_lookahead, lo, hi)``."""
    lo = min(cfg.min_lookahead_m, cfg.max_lookahead_m)
    hi = max(cfg.min_lookahead_m, cfg.max_lookahead_m)
    if near_goal:
        return lo
    if abs(speed_mps) < 0.05:
        # Starting / after rotate-to-heading: no speed to scale from.
        return max(lo, min(hi, cfg.lookahead_m))
    return max(lo, min(hi, abs(speed_mps) * cfg.lookahead_time_s))


def update_speed_estimate(prev: float, cmd_vx: float, *, alpha: float = 0.25) -> float:
    """Smoothed forward-speed estimate for the velocity-scaled lookahead.

    Nav2 scales lookahead from *measured* odometry, which the robot's inertia
    smooths. We only have the last command; feeding it back raw creates a
    limit cycle (slow → shorter L → less curvature → fast → longer L → ...)
    that shows up as alternating vθ every tick through a corner. A ~0.4 s EMA
    stands in for inertia.
    """
    return (1.0 - alpha) * float(prev) + alpha * max(0.0, float(cmd_vx))


def drivable_min_speed(kappa: float, cfg: FollowerConfig) -> float:
    """Slowest ``vx`` at which an arc of curvature ``kappa`` keeps the inner
    wheel above ``wheel_min_speed`` (and clears the base sanitizer floor)."""
    denom = 1.0 - float(cfg.wheel_half_track_m) * abs(kappa)
    if denom <= 1e-3:
        return math.inf
    return max(_BASE_CRAWL_FLOOR_MPS, float(cfg.wheel_min_speed_mps) / denom)


def keep_arc_drivable(cmd: DriveCommand, cfg: Optional[FollowerConfig] = None) -> DriveCommand:
    """Keep a translating arc inside the region the base will actually execute.

    Obstacle slow-down scales ``vx`` toward zero; below the crawl floor the
    sanitizer, and below the inner-wheel minimum the base itself, turn the arc
    into a pure spin (the "crazy arcs" / stop-spin-go at corners). Preserve
    curvature instead: cap it at the minimum turn radius, then raise ``vx`` to
    the slowest speed that still drives that arc and rescale ``vθ`` with it.
    """
    if cmd.done or cmd.vx <= 0.0:
        return cmd
    cfg = cfg if cfg is not None else FollowerConfig()
    kappa = cmd.vtheta / cmd.vx
    kappa_max = 1.0 / cfg.effective_min_turn_radius_m()
    if abs(kappa) > kappa_max:
        kappa = math.copysign(kappa_max, kappa)
    vx = max(cmd.vx, drivable_min_speed(kappa, cfg))
    vx = min(vx, cfg.motion.max_linear_mps) if cfg.motion.max_linear_mps > 0 else vx
    if vx == cmd.vx and kappa == cmd.vtheta / cmd.vx:
        return cmd
    return DriveCommand(vx, cmd.vy, vx * kappa, False)


def limit_twist_rate(
    cmd: DriveCommand,
    prev: Optional[DriveCommand],
    *,
    cfg: FollowerConfig,
    dt_s: float,
) -> DriveCommand:
    """Bound how far a command may move from the one already on the base.

    Applied as the last gate before ``SetVelocity``. Ramping     ``vx`` alone would
    tighten the arc (ω/vx grows), so a translating command keeps its curvature
    by scaling ``vθ`` with the limited speed, then gets re-checked against the
    skid-steer wheel envelope.

    Any command that asks translation to stop takes effect on the same tick —
    the reactive stop bubble, costmap hard stop, wait, and pre-replan stop must
    never be slewed. Only speeding up, changing speed between nonzero values,
    and yaw changes are rate limited.
    """
    if cmd.done:
        return cmd
    dt = max(1e-3, float(dt_s))
    target_vx = float(cmd.vx)
    target_w = float(cmd.vtheta)
    if abs(target_vx) < 1e-6 and abs(target_w) < 1e-6:
        return cmd
    prev_vx = 0.0 if prev is None else float(prev.vx)
    prev_w = 0.0 if prev is None else float(prev.vtheta)

    if abs(target_vx) < 1e-6:
        vx = 0.0
        w = target_w
    else:
        speeding_up = abs(target_vx) > abs(prev_vx) and target_vx * prev_vx >= 0.0
        max_dv = dt * float(
            cfg.max_linear_accel_mps2 if speeding_up else cfg.max_linear_decel_mps2
        )
        vx = prev_vx + max(-max_dv, min(max_dv, target_vx - prev_vx))
        # Preserve the requested arc while the speed ramps.
        w = target_w * (vx / target_vx)
    max_dw = dt * float(cfg.max_angular_accel_rad_s2)
    w = prev_w + max(-max_dw, min(max_dw, w - prev_w))

    out = DriveCommand(vx, cmd.vy, w, False)
    if vx > 0.0:
        # May raise vx to the slowest speed that still drives this arc; never
        # past what the caller asked for.
        out = keep_arc_drivable(out, cfg)
        out = DriveCommand(min(out.vx, target_vx), out.vy, out.vtheta, False)
    return apply_velocity_floor(out, cfg.motion)


def _prev_curvature(prev_cmd: Optional[DriveCommand]) -> Optional[float]:
    if prev_cmd is None or prev_cmd.done or prev_cmd.vx <= 1e-3:
        return None
    return float(prev_cmd.vtheta) / float(prev_cmd.vx)


def pursuit_command(
    current: Pose2D,
    target: Pose2D,
    *,
    cfg: FollowerConfig,
    rotate_active: bool = False,
    prev_cmd: Optional[DriveCommand] = None,
) -> Tuple[DriveCommand, bool]:
    """Regulated pure pursuit toward the lookahead point ``target``.

    Returns ``(cmd, rotating)`` where ``rotating`` is the rotate-to-heading
    state to feed back next tick (hysteresis). ``prev_cmd`` (last issued
    command) enables curvature smoothing across ticks.
    """
    motion = cfg.motion
    max_linear = motion.max_linear_mps
    max_angular = motion.max_angular_rad_s

    dx = target.x - current.x
    dy = target.y - current.y
    cth = math.cos(current.theta)
    sth = math.sin(current.theta)
    # Lookahead point in the robot frame.
    x_l = cth * dx + sth * dy
    y_l = -sth * dx + cth * dy
    l2 = x_l * x_l + y_l * y_l
    if l2 < 1e-6:
        return DriveCommand(0.0, 0.0, 0.0, False), False
    alpha = math.atan2(y_l, x_l)

    enter = abs(alpha) > cfg.rotate_in_place_rad
    stay = rotate_active and abs(alpha) > cfg.rotate_exit_rad
    if enter or stay:
        rot_max = min(float(cfg.rotate_vel_rad_s), max_angular)
        rot_floor = min(rot_max, max(motion.min_angular_rad_s, 0.2))
        w = math.copysign(max(min(abs(alpha) * 1.5, rot_max), rot_floor), alpha)
        return DriveCommand(0.0, 0.0, w, False), True

    y_steer = y_l
    deadband = max(0.0, float(cfg.crosstrack_deadband_m))
    if deadband > 0.0 and abs(y_steer) < deadband:
        y_steer = 0.0
    kappa = 2.0 * y_steer / l2
    # Smooth κ across ticks (pose noise → κ flicker), unless we just started
    # translating (no previous arc to blend with).
    prev_kappa = _prev_curvature(prev_cmd)
    if prev_kappa is not None:
        a = min(1.0, max(0.05, float(cfg.curvature_smoothing)))
        kappa = a * kappa + (1.0 - a) * prev_kappa
    # Skid-steer cannot drive arcs tighter than min_turn_radius while
    # translating (inner wheel → 0 → base rejects → spin). Widen the arc; the
    # bearing then grows and rotate-to-heading takes over if really needed.
    kappa_max = 1.0 / cfg.effective_min_turn_radius_m()
    if abs(kappa) > kappa_max:
        kappa = math.copysign(kappa_max, kappa)

    v = max_linear
    if abs(kappa) > 1e-6:
        radius = 1.0 / abs(kappa)
        if radius < cfg.regulated_min_radius_m:
            v *= radius / cfg.regulated_min_radius_m
    v = max(v, float(cfg.regulated_min_speed_mps), drivable_min_speed(kappa, cfg))
    v = min(max_linear, v)
    w = v * kappa
    if abs(w) > max_angular:
        # Keep the arc; give up speed rather than curvature.
        w = math.copysign(max_angular, w)
        v = max_angular / abs(kappa)
    v = max(v, motion.min_linear_mps)
    return keep_arc_drivable(DriveCommand(v, 0.0, w, False), cfg), False


def lookahead_pose(
    current: Pose2D,
    path: Path2D,
    *,
    lookahead_m: float,
    waypoint_tolerance_m: float,
) -> Tuple[Pose2D, int, bool]:
    """Return (target_pose, waypoint_index, is_final).

    Projects the robot onto the polyline (closest point on any segment), then
    walks ``lookahead_m`` forward along the path. Vertex-only projection fails
    on sparse Lazy Theta* paths (2–4 waypoints spanning meters).
    """
    del waypoint_tolerance_m  # kept for call-site compatibility
    pts = path.points
    if not pts:
        return current, 0, True
    if len(pts) == 1:
        return Pose2D(pts[0][0], pts[0][1], path.goal_theta), 0, True

    # Build cumulative lengths for walking from the closest projection.
    seg_lens = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg_lens.append(L)
        cum.append(cum[-1] + L)
    total = cum[-1]
    if total < 1e-9:
        return Pose2D(pts[-1][0], pts[-1][1], path.goal_theta), len(pts) - 1, True

    _, _, seg_i, along = closest_point_on_path(current, path)
    target_along = min(total, along + max(lookahead_m, 0.05))

    # Walk to target_along.
    for i in range(len(pts) - 1):
        if cum[i + 1] + 1e-9 < target_along:
            continue
        seg = seg_lens[i]
        if seg < 1e-9:
            continue
        t = (target_along - cum[i]) / seg
        t = max(0.0, min(1.0, t))
        x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
        y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
        heading = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
        is_final = target_along >= total - 1e-6
        if is_final:
            return Pose2D(pts[-1][0], pts[-1][1], path.goal_theta), len(pts) - 1, True
        return Pose2D(x, y, heading), i, False

    return Pose2D(pts[-1][0], pts[-1][1], path.goal_theta), len(pts) - 1, True


def compute_follow_command(
    current: Pose2D,
    target: Pose2D,
    *,
    cfg: FollowerConfig,
    final_yaw: Optional[float] = None,
) -> DriveCommand:
    """One step toward ``target``: near-goal law if ``final_yaw`` else pursuit."""
    motion = cfg.motion
    dist = distance_m(current, target)
    heading_to_target = math.atan2(target.y - current.y, target.x - current.x)
    bearing = heading_error_rad(current.theta, heading_to_target)
    xy_tol = motion.xy_tolerance_m

    if final_yaw is not None:
        yaw_err = heading_error_rad(current.theta, final_yaw)
        if dist <= xy_tol and abs(yaw_err) <= motion.yaw_tolerance_rad:
            return DriveCommand(0.0, 0.0, 0.0, True)
        # Always use the near-goal law once we are pursuing the goal pose
        # (final_yaw set). Gating on 2× xy_tol left a dead band (~0.5–0.8 m)
        # where is_final still targets the goal but rotate_in_place commanded
        # ±max_vel_theta — classic near-goal butt-wiggle with vx=0.
        return _near_goal_command(
            yaw_err=yaw_err,
            bearing=bearing,
            dist=dist,
            motion=motion,
        )

    # Intermediate pursuit target reached — not navigation complete.
    if dist <= xy_tol * 0.5:
        return DriveCommand(0.0, 0.0, 0.0, False)

    cmd, _rotating = pursuit_command(current, target, cfg=cfg)
    return apply_velocity_floor(cmd, motion)


def compute_path_command(
    current: Pose2D,
    path: Path2D,
    *,
    cfg: FollowerConfig,
    scan: Optional[conv.LaserScan2D] = None,
    speed_mps: Optional[float] = None,
    local_view: Optional[LocalCostmapView] = None,
    local_planner: Optional[LocalPlannerConfig] = None,
    robot_radius_m: float = 0.22,
    min_cmd_vel_x: float = 0.0,
    min_cmd_vel_theta: float = 0.0,
    local_planner_active: bool = False,
    prev_local_cmd: Optional[DriveCommand] = None,
    rotate_active: bool = False,
    prev_cmd: Optional[DriveCommand] = None,
    force_local_planner: bool = False,
) -> Tuple[DriveCommand, dict]:
    """One control step along ``path``.

    ``rotate_active`` is the rotate-to-heading state from the previous tick
    (``progress["rotate_to_heading"]``); feed it back for hysteresis.
    ``prev_cmd`` is the last command actually issued (curvature smoothing).
    ``force_local_planner`` lets DWA run with a large bearing error when the
    path is locally blocked — otherwise rotate-in-place spins forever while
    replans fail and the detour never starts.
    """
    est_speed = cfg.motion.max_linear_mps * 0.5 if speed_mps is None else speed_mps
    goal_xy = Pose2D(path.points[-1][0], path.points[-1][1], 0.0)
    dist_goal = distance_m(current, goal_xy)
    # Keep local planner off and use final-yaw pursuit for the whole approach.
    near_goal = dist_goal <= max(
        cfg.motion.xy_tolerance_m * 2.0,
        cfg.approach_dist_m * 2.0,
        0.8,
    )
    lookahead = _effective_lookahead(
        cfg, speed_mps=est_speed, near_goal=near_goal
    )
    target, idx, is_final = lookahead_pose(
        current,
        path,
        lookahead_m=lookahead,
        waypoint_tolerance_m=cfg.waypoint_tolerance_m,
    )
    bearing = heading_error_rad(
        current.theta, math.atan2(target.y - current.y, target.x - current.x)
    )
    crosstrack, _path_yaw = signed_crosstrack_m(current, path)
    rotating = False

    local_active = False
    bearing_ok = abs(bearing) <= cfg.rotate_in_place_rad or force_local_planner
    if (
        local_view is not None
        and local_planner is not None
        and not near_goal
        and bearing_ok
    ):
        local_cmd = compute_local_command(
            current,
            path,
            local_view,
            cfg=local_planner,
            max_vel_x=cfg.motion.max_linear_mps,
            max_vel_theta=cfg.motion.max_angular_rad_s,
            robot_radius_m=robot_radius_m,
            min_cmd_vel_x=min_cmd_vel_x,
            min_cmd_vel_theta=min_cmd_vel_theta,
            local_planner_active=local_planner_active or force_local_planner,
            prev_cmd=prev_local_cmd if local_planner_active else None,
        )
        if local_cmd is not None:
            cmd = local_cmd
            cmd = apply_velocity_floor(cmd, cfg.motion)
            local_active = True

    if not local_active:
        if is_final or near_goal:
            goal = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
            cmd = compute_follow_command(
                current, goal, cfg=cfg, final_yaw=path.goal_theta
            )
            bearing = heading_error_rad(current.theta, path.goal_theta)
        else:
            cmd, rotating = pursuit_command(
                current,
                target,
                cfg=cfg,
                rotate_active=rotate_active,
                prev_cmd=prev_cmd,
            )
            cmd = apply_velocity_floor(cmd, cfg.motion)

    obstacle_state = "clear"
    forward_clearance = math.inf
    # Inside the XY acceptance ball, reactive slow/stop fights the soft crawl
    # (scales vx under sanitizer lin_eps) and blocks the yaw handoff. Clearance
    # is still reported when we have a scan; we just don't reshape the cmd.
    apply_obstacle = (
        not local_active
        and cfg.obstacle is not None
        and cfg.obstacle.enabled
        and dist_goal > cfg.motion.xy_tolerance_m
    )
    if apply_obstacle:
        cmd, obstacle_state, forward_clearance = apply_obstacle_avoidance(
            cmd,
            scan,
            cfg.obstacle,
            max_angular_rad_s=cfg.motion.max_angular_rad_s,
            prefer_bearing_rad=bearing,
        )
        if not cmd.done and (cmd.vx != 0.0 or cmd.vtheta != 0.0):
            # Slow-down scaled vx; keep the arc drivable before the floor so a
            # 0.05 m/s crawl with 0.2 rad/s does not become a base-side spin.
            cmd = apply_velocity_floor(keep_arc_drivable(cmd, cfg), cfg.motion)
    elif local_active:
        obstacle_state = "local_planner"
        if (
            scan is not None
            and cfg.obstacle is not None
            and cfg.obstacle.enabled
        ):
            forward_clearance = forward_clearance_m(scan, cfg.obstacle)
    elif (
        cfg.obstacle is not None
        and cfg.obstacle.enabled
        and dist_goal <= cfg.motion.xy_tolerance_m
        and scan is not None
    ):
        forward_clearance = forward_clearance_m(scan, cfg.obstacle)
        obstacle_state = "clear"

    # Hard safety: never translate into the stop bubble — including during DWA.
    if (
        cfg.obstacle is not None
        and cfg.obstacle.enabled
        and scan is not None
        and cmd.vx > 1e-6
        and dist_goal > cfg.motion.xy_tolerance_m
    ):
        if not math.isfinite(forward_clearance):
            forward_clearance = forward_clearance_m(scan, cfg.obstacle)
        if forward_clearance <= cfg.obstacle.stop_distance_m:
            left = cone_min_range(scan, 0.0, cfg.obstacle.side_cone_rad)
            right = cone_min_range(scan, -cfg.obstacle.side_cone_rad, 0.0)
            direction = 1.0 if left >= right else -1.0
            if abs(bearing) >= math.radians(12.0):
                prefer_left = bearing > 0.0
                preferred = left if prefer_left else right
                if preferred >= cfg.obstacle.stop_distance_m:
                    direction = 1.0 if prefer_left else -1.0
            cmd = DriveCommand(
                0.0, 0.0, direction * cfg.motion.max_angular_rad_s, False
            )
            obstacle_state = "avoid"

    # Costmap hard stop: lidar cone can look clear while the robot is already
    # driving into an inflated blob beside the nose (or while misaligned). If
    # the next ~stop_distance along the commanded heading is inscribed, freeze
    # translation.
    if (
        local_view is not None
        and cmd.vx > 1e-6
        and dist_goal > cfg.motion.xy_tolerance_m
    ):
        from .costmap import INSCRIBED
        from .local_costmap import max_cost_along_segment

        stop_m = (
            float(cfg.obstacle.stop_distance_m)
            if cfg.obstacle is not None and cfg.obstacle.enabled
            else max(0.35, float(robot_radius_m) + 0.1)
        )
        hx = current.x + math.cos(current.theta) * stop_m
        hy = current.y + math.sin(current.theta) * stop_m
        ahead = max_cost_along_segment(local_view, current.x, current.y, hx, hy)
        if ahead >= INSCRIBED:
            cmd = DriveCommand(0.0, 0.0, cmd.vtheta, False)
            if abs(cmd.vtheta) < 1e-6:
                # No yaw command: turn toward freer flank using path bearing.
                direction = 1.0 if bearing >= 0.0 else -1.0
                cmd = DriveCommand(
                    0.0, 0.0, direction * cfg.motion.max_angular_rad_s, False
                )
            obstacle_state = "avoid"

    progress = {
        "waypoint_index": idx,
        "is_final": is_final,
        "local_planner": local_active,
        "distance_remaining_m": distance_m(
            current, Pose2D(path.points[-1][0], path.points[-1][1], 0.0)
        ),
        "path_length_m": _path_length(path),
        "obstacle": obstacle_state,
        "forward_clearance_m": None if math.isinf(forward_clearance) else forward_clearance,
        "bearing_error_rad": bearing,
        "crosstrack_m": crosstrack,
        "lookahead_m": lookahead,
        "rotate_to_heading": rotating,
        "cmd_vx_mps": cmd.vx,
        "cmd_vy_mps": cmd.vy,
        "cmd_vtheta_rad_s": cmd.vtheta,
    }
    return cmd, progress

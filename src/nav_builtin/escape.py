"""Nose-escape policy shared by the controller and the supervisor.

The footprint guard never reverses. When the bumper is against an obstacle
the stack used to decide turn-vs-reverse in five places with three different
rear-clear checks. One policy, one reverse command, one rear-clear gate:

``hold`` → ``turn`` (heading/spin open) → ``reverse`` (rear open) → ``hold``.

Entry conditions still differ (wait after a failed replan, bumper with a
cheap path, crawl the base would drop). The *choice* once stuck is here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, TYPE_CHECKING

from ..geom import conversions as conv
from ..nav.simple_motion import DriveCommand, rear_clearance_m
from .local_costmap import LocalCostmapView, reverse_path_clear
from .types import Pose2D

if TYPE_CHECKING:
    from .controller import FollowerConfig
    from .footprint_guard import FootprintGuard

HOLD = "hold"
TURN = "turn"
REVERSE = "reverse"

# Match the historical nose-unstick / undrivable budgets.
UNSTICK_YAW_RAD = 0.8
UNSTICK_YAW_RATE = 0.45
REVERSE_CHECK_DIST_M = 0.30
RECTANGLE_REVERSE_VX = -0.15
# Bearing that is large enough to prefer a turn over a reverse creep.
TURN_BEARING_RAD = math.radians(12.0)


def rear_open(
    scan: Optional[conv.LaserScan2D],
    robot_radius_m: float,
    *,
    local_view: Optional[LocalCostmapView] = None,
    x_m: float = 0.0,
    y_m: float = 0.0,
    theta_rad: float = 0.0,
    reverse_dist_m: float = REVERSE_CHECK_DIST_M,
) -> bool:
    """True when a short reverse is safe per rear scan and local costmap.

    ``rear_clearance_m`` returns +inf when the rear arc has no returns (fully
    clear). That must count as open — requiring ``isfinite`` left the robot
    frozen in avoid+spin_block with an empty rear.

    When a local costmap is present, also require the reverse footprint path
    to stay clear so we do not back into persisted / inflated blobs that
    lidar is currently missing.
    """
    if scan is None:
        return False
    rear = rear_clearance_m(scan)
    rear_need = max(0.25, float(robot_radius_m) + 0.08)
    if math.isfinite(rear) and rear < rear_need:
        return False
    if local_view is not None:
        return reverse_path_clear(
            local_view,
            x_m,
            y_m,
            theta_rad,
            robot_radius_m=float(robot_radius_m),
            distance_m=float(reverse_dist_m),
            ignore_ahead=True,
        )
    return True


def rectangle_rear_open(
    guard: "FootprintGuard",
    pose: Pose2D,
    pts,
    *,
    distance_m: float,
    min_free_m: float = 0.12,
) -> bool:
    """True when the rectangle can back ``distance_m`` on raw obstacle points.

    Preferred over the inflated costmap check: a blob an inch ahead fills the
    reverse disc through the body while the rectangle still has room.
    """
    back_free = guard.free_distance(
        pose, RECTANGLE_REVERSE_VX, 0.0, pts, float(distance_m)
    )
    return not math.isfinite(back_free) or back_free >= float(min_free_m)


def reverse_clear(
    *,
    pose: Pose2D,
    scan: Optional[conv.LaserScan2D],
    robot_radius_m: float,
    local_view: Optional[LocalCostmapView] = None,
    guard: Optional["FootprintGuard"] = None,
    guard_pts=None,
    backup_dist_m: float = 0.35,
) -> bool:
    """Rectangle-first, cone/costmap-second rear clear for an escape reverse."""
    need = max(0.35, float(backup_dist_m))
    if guard is not None and guard_pts is not None:
        if rectangle_rear_open(guard, pose, guard_pts, distance_m=need, min_free_m=0.12):
            return True
    return rear_open(
        scan,
        robot_radius_m,
        local_view=local_view,
        x_m=pose.x,
        y_m=pose.y,
        theta_rad=pose.theta,
        reverse_dist_m=min(REVERSE_CHECK_DIST_M, need),
    )


def reverse_command(cfg: "FollowerConfig") -> DriveCommand:
    """Slow straight reverse the base will actually execute."""
    back = min(
        max(float(cfg.motion.min_linear_mps), 0.10),
        0.18,
    )
    return DriveCommand(-back, 0.0, 0.0, False)


def try_reverse(
    cfg: "FollowerConfig",
    scan: Optional[conv.LaserScan2D],
    robot_radius_m: float,
    *,
    local_view: Optional[LocalCostmapView],
    current: Pose2D,
) -> Optional[DriveCommand]:
    """Cone + costmap reverse when the rear is open; else None."""
    if rear_open(
        scan,
        robot_radius_m,
        local_view=local_view,
        x_m=current.x,
        y_m=current.y,
        theta_rad=current.theta,
    ):
        return reverse_command(cfg)
    return None


def turn_command(bearing_rad: float = 0.0, *, rate: float = UNSTICK_YAW_RATE) -> DriveCommand:
    """In-place yaw toward ``bearing_rad`` (or +rate when bearing is ~0)."""
    sign = math.copysign(1.0, bearing_rad) if abs(bearing_rad) > 1e-6 else 1.0
    return DriveCommand(0.0, 0.0, sign * float(rate), False)


def blocked_nose_unstick(
    *,
    failed_replans: int,
    nose_clear: bool,
    rear_open: bool,
    spin_clear: bool,
    heading_open: bool = False,
) -> str:
    """Motion while a blocked nose waits out the replan cooldown.

    ``hold`` for the first wait, so a person crossing can still move.
    After a replan fails: turn toward a heading the rectangle can drive
    when ``heading_open`` (raw scan, not the inflated spin disc).
    Otherwise ``reverse`` when the rear is open, else ``turn`` when an
    in-place spin would not sweep an obstacle.
    """
    if nose_clear or int(failed_replans) < 1:
        return HOLD
    if heading_open:
        return TURN
    if rear_open:
        return REVERSE
    if spin_clear:
        return TURN
    return HOLD


def bumper_wants_reverse(
    *,
    nose_clear: bool,
    spin_blocked: bool,
    cmd_vx: float,
    cmd_vtheta: float,
    rear_open: bool,
) -> bool:
    """Back up when the bumper is against an obstacle and a spin would hit it.

    The path centerline can still be cheap. The footprint guard will not
    reverse, so the cart sits at cmd 0 an inch off the obstacle with open
    space behind it.
    """
    return (
        not nose_clear
        and bool(spin_blocked)
        and abs(float(cmd_vx)) < 1e-6
        and abs(float(cmd_vtheta)) < 1e-6
        and bool(rear_open)
    )


def undrivable_nose(
    *,
    nose_clear: bool,
    waiting: bool,
    vx: float,
    vtheta: float,
    rear_open: bool,
    spin_clear: bool,
    bearing_rad: float,
) -> Optional[str]:
    """Motion when the only forward command would be dropped by the base.

    ``turn`` when a yaw toward the path is clear. ``reverse`` when the
    rectangle can back off the bumper. ``None`` while a person-crossing
    wait is still in its grace, or when neither escape is open.
    """
    from .base_envelope import twist_reaches_base

    if nose_clear or waiting or twist_reaches_base(vx, vtheta):
        return None
    if spin_clear and abs(float(bearing_rad)) >= TURN_BEARING_RAD:
        return TURN
    if rear_open:
        return REVERSE
    if spin_clear:
        return TURN
    return None


def choose(
    *,
    nose_clear: bool,
    rear_open: bool,
    spin_clear: bool = False,
    heading_open: bool = False,
    bearing_rad: float = 0.0,
    failed_replans: int = 1,
    waiting: bool = False,
    cmd_vx: float = 0.0,
    cmd_vtheta: float = 0.0,
    spin_blocked: bool = False,
    reason: str = "unstick",
) -> Optional[str]:
    """Single entry for the escape decision, by stuck reason.

    ``unstick`` — blocked-nose wait after a failed replan.
    ``bumper`` — nose blocked, spin blocked, cmd already zero, path may be free.
    ``undrivable`` — forward twist would not reach the base.
    """
    if reason == "bumper":
        return (
            REVERSE
            if bumper_wants_reverse(
                nose_clear=nose_clear,
                spin_blocked=spin_blocked,
                cmd_vx=cmd_vx,
                cmd_vtheta=cmd_vtheta,
                rear_open=rear_open,
            )
            else None
        )
    if reason == "undrivable":
        return undrivable_nose(
            nose_clear=nose_clear,
            waiting=waiting,
            vx=cmd_vx,
            vtheta=cmd_vtheta,
            rear_open=rear_open,
            spin_clear=spin_clear,
            bearing_rad=bearing_rad,
        )
    return blocked_nose_unstick(
        failed_replans=failed_replans,
        nose_clear=nose_clear,
        rear_open=rear_open,
        spin_clear=spin_clear,
        heading_open=heading_open,
    )


# ---------------------------------------------------------------------------
# Escape state machine
# ---------------------------------------------------------------------------

IDLE = "idle"
MODE_REVERSE = "reverse"
MODE_TURN = "turn"

# Small reverse that counts as "we moved" before preferring a turn instead.
_UNSTICK_REVERSE_NUDGE_M = 0.05
_UNSTICK_TURN_NUDGE_RAD = 0.2


@dataclass
class EscapeSense:
    """Everything the machine needs from one control tick."""

    now: float
    pose: Pose2D
    cmd: DriveCommand
    progress: Mapping
    nose_clear: bool
    waiting: bool
    failed_replans: int
    spin_blocked: bool
    heading_open: bool
    bearing_rad: float
    scan: Optional[conv.LaserScan2D]
    local_view: Optional[LocalCostmapView]
    guard: Optional["FootprintGuard"]
    guard_pts: Any
    robot_radius_m: float
    spin_radius_m: float
    follower: "FollowerConfig"
    backup_dist_m: float
    backup_cooldown_s: float
    loc_hold: bool = False
    corridor_active: bool = False
    # Preferred turn sign (corridor opening). Defaults to +1.
    prefer_yaw_sign: float = 1.0


@dataclass
class EscapeResult:
    """Command / progress / replan request for this tick."""

    cmd: Optional[DriveCommand] = None
    obstacle: Optional[str] = None
    replan_trigger: Optional[str] = None
    active_mode: Optional[str] = None
    # When set, the supervisor should stop and replan with this trigger.
    # ``reset_block_count`` clears failed_replan_while_blocked on success.
    reset_block_count: bool = False


class EscapeMachine:
    """One nose-escape state machine for bumper / wait-unstick / undrivable.

    States: ``idle`` → ``reverse`` | ``turn`` → (budget met) → cooldown → idle.
    Entry reasons differ; once active the same budget and rear-clear rules
    finish the motion and ask the supervisor to replan.
    """

    def __init__(self) -> None:
        self.mode: str = IDLE
        self.start: Optional[Pose2D] = None
        self.yaw_sign: float = 1.0
        self.cooldown_until: float = 0.0
        self.trigger: str = ""

    def reset(self) -> None:
        self.mode = IDLE
        self.start = None
        self.yaw_sign = 1.0
        self.trigger = ""

    def clear_episode(self) -> None:
        """Nose cleared or a new path landed — drop an in-flight escape."""
        self.reset()

    def _enter(self, mode: str, pose: Pose2D, trigger: str, yaw_sign: float = 1.0) -> None:
        self.mode = mode
        self.start = pose
        self.trigger = trigger
        self.yaw_sign = float(yaw_sign)

    def _begin_cooldown(self, now: float, cooldown_s: float) -> None:
        self.cooldown_until = now + max(1.5, float(cooldown_s))
        self.reset()

    def _cooling(self, now: float) -> bool:
        return now < self.cooldown_until

    def _rear_still_open(self, sense: EscapeSense, remain_m: float) -> bool:
        if sense.local_view is not None:
            if reverse_path_clear(
                sense.local_view,
                sense.pose.x,
                sense.pose.y,
                sense.pose.theta,
                robot_radius_m=sense.robot_radius_m,
                distance_m=remain_m,
                ignore_ahead=True,
            ):
                return True
        if sense.guard is not None and sense.guard_pts is not None:
            return rectangle_rear_open(
                sense.guard,
                sense.pose,
                sense.guard_pts,
                distance_m=remain_m,
                min_free_m=min(remain_m, 0.12),
            )
        return rear_open(
            sense.scan,
            sense.robot_radius_m,
            local_view=None,
            x_m=sense.pose.x,
            y_m=sense.pose.y,
            theta_rad=sense.pose.theta,
            reverse_dist_m=min(REVERSE_CHECK_DIST_M, remain_m),
        )

    def _spin_clear(self, sense: EscapeSense) -> bool:
        if sense.scan is None:
            return False
        from ..nav.simple_motion import spin_clearance_m
        from .local_costmap import spin_disc_blocked

        clear = spin_clearance_m(sense.scan) >= sense.spin_radius_m + 0.05
        if (
            clear
            and sense.spin_radius_m > sense.robot_radius_m
            and sense.local_view is not None
            and spin_disc_blocked(
                sense.local_view,
                sense.pose.x,
                sense.pose.y,
                spin_radius_m=sense.spin_radius_m,
                inscribed_radius_m=sense.robot_radius_m,
            )
        ):
            return False
        return clear

    def _guard_spin_clear(self, sense: EscapeSense, bearing: float) -> bool:
        if sense.guard is None or sense.guard_pts is None or abs(bearing) <= 1e-3:
            return False
        sign = 1.0 if bearing >= 0.0 else -1.0
        return (
            sense.guard.free_rotation(sense.pose, sign, sense.guard_pts)
            >= sense.guard.cfg.rot_horizon_rad * 0.5
        )

    def _finish(self, sense: EscapeSense) -> EscapeResult:
        trigger = self.trigger or "narrow_reverse_done"
        self._begin_cooldown(sense.now, sense.backup_cooldown_s)
        return EscapeResult(
            cmd=DriveCommand(0.0, 0.0, 0.0, False),
            obstacle="narrow_reverse_done" if "unstick" not in trigger else "planning",
            replan_trigger=trigger,
            reset_block_count=True,
            active_mode=None,
        )

    def _continue_reverse(self, sense: EscapeSense) -> EscapeResult:
        assert self.start is not None
        from ..nav.simple_motion import distance_m

        backed = distance_m(sense.pose, self.start)
        remain = max(0.12, float(sense.backup_dist_m) - backed)
        if backed >= sense.backup_dist_m or not self._rear_still_open(sense, remain):
            return self._finish(sense)
        cmd = reverse_command(sense.follower)
        # Undrivable reverse yaws while backing so the bumper does not re-hit.
        if self.trigger == "undrivable_nose" and abs(sense.bearing_rad) >= TURN_BEARING_RAD:
            cmd = DriveCommand(
                cmd.vx, 0.0, math.copysign(0.20, sense.bearing_rad), False
            )
        return EscapeResult(
            cmd=cmd,
            obstacle="narrow_reverse" if self.trigger != "blocked_nose_unstick" else "wait_reverse",
            active_mode=MODE_REVERSE,
        )

    def _continue_turn(self, sense: EscapeSense) -> EscapeResult:
        assert self.start is not None
        turned = abs(conv.normalize_angle(sense.pose.theta - self.start.theta))
        if turned >= UNSTICK_YAW_RAD:
            return self._finish(sense)
        cmd = DriveCommand(0.0, 0.0, self.yaw_sign * UNSTICK_YAW_RATE, False)
        obstacle = "avoid" if self.trigger == "undrivable_nose" else "wait_turn"
        return EscapeResult(cmd=cmd, obstacle=obstacle, active_mode=MODE_TURN)

    def tick_pre(self, sense: EscapeSense) -> EscapeResult:
        """Bumper adopt, reverse bound, wait inject, blocked-nose unstick."""
        if sense.loc_hold:
            return EscapeResult()

        # Cooldown: a reverse that restarts immediately after finishing used
        # to drive forever. Hold only the controller's per-tick reverse.
        obs = str(sense.progress.get("obstacle") or "")
        if self._cooling(sense.now) and obs == "narrow_reverse":
            self.reset()
            return EscapeResult(
                cmd=DriveCommand(0.0, 0.0, 0.0, False),
                obstacle="narrow_reverse_hold",
            )

        if self.mode == MODE_REVERSE:
            return self._continue_reverse(sense)
        if self.mode == MODE_TURN:
            # During wait unstick: if the controller already put us in reverse
            # without going through enter, adopt it.
            if (
                sense.cmd.vx < -1e-6
                and abs(sense.cmd.vtheta) < 1e-6
                and not sense.nose_clear
            ):
                self._enter(MODE_REVERSE, sense.pose, "blocked_nose_unstick")
                return self._continue_reverse(sense)
            return self._continue_turn(sense)

        # --- idle: look for an entry -----------------------------------
        if self._cooling(sense.now):
            return EscapeResult()

        # Bumper: nose blocked, spin blocked, cmd zero, rear clear.
        if bumper_wants_reverse(
            nose_clear=sense.nose_clear,
            spin_blocked=sense.spin_blocked,
            cmd_vx=sense.cmd.vx,
            cmd_vtheta=sense.cmd.vtheta,
            rear_open=True,
        ) and reverse_clear(
            pose=sense.pose,
            scan=sense.scan,
            robot_radius_m=sense.robot_radius_m,
            local_view=sense.local_view,
            guard=sense.guard,
            guard_pts=sense.guard_pts,
            backup_dist_m=sense.backup_dist_m,
        ):
            self._enter(MODE_REVERSE, sense.pose, "narrow_reverse_done")
            return EscapeResult(
                cmd=reverse_command(sense.follower),
                obstacle="narrow_reverse",
                active_mode=MODE_REVERSE,
            )

        # Controller already asked for narrow_reverse: adopt and bound it.
        if obs == "narrow_reverse" and sense.cmd.vx < -1e-6:
            self._enter(MODE_REVERSE, sense.pose, "narrow_reverse_done")
            return self._continue_reverse(sense)

        if not sense.waiting:
            return EscapeResult()

        # Wait: inject reverse when frozen on a spin-blocked / lethal nose.
        cmd = sense.cmd
        if (
            abs(cmd.vx) < 1e-6
            and abs(cmd.vtheta) < 1e-6
            and (sense.spin_blocked or obs == "in_lethal")
            and sense.local_view is not None
            and sense.scan is not None
        ):
            rev = try_reverse(
                sense.follower,
                sense.scan,
                sense.robot_radius_m,
                local_view=sense.local_view,
                current=sense.pose,
            )
            if rev is not None:
                cmd = rev

        # Blocked-nose unstick after a failed replan.
        if (
            not sense.nose_clear
            and sense.failed_replans >= 1
            and sense.scan is not None
            and sense.local_view is not None
            and abs(cmd.vx) < 1e-6
            and abs(cmd.vtheta) < 1e-6
        ):
            rev = try_reverse(
                sense.follower,
                sense.scan,
                sense.robot_radius_m,
                local_view=sense.local_view,
                current=sense.pose,
            )
            choice = blocked_nose_unstick(
                failed_replans=sense.failed_replans,
                nose_clear=sense.nose_clear,
                rear_open=rev is not None,
                spin_clear=self._spin_clear(sense),
                heading_open=sense.heading_open,
            )
            if choice == REVERSE and rev is not None:
                self._enter(MODE_REVERSE, sense.pose, "blocked_nose_unstick")
                return EscapeResult(
                    cmd=rev, obstacle="wait_reverse", active_mode=MODE_REVERSE
                )
            if choice == TURN:
                sign = math.copysign(1.0, sense.prefer_yaw_sign or 1.0)
                self._enter(
                    MODE_TURN,
                    sense.pose,
                    "blocked_nose_unstick",
                    yaw_sign=sign,
                )
                return EscapeResult(
                    cmd=DriveCommand(0.0, 0.0, sign * UNSTICK_YAW_RATE, False),
                    obstacle="wait_turn",
                    active_mode=MODE_TURN,
                )

        # Wait freeze: keep a reverse the inject produced, else zero translate.
        if abs(cmd.vx) > 1e-6 and cmd.vx < 0 and abs(cmd.vtheta) < 1e-6:
            if self.mode != MODE_REVERSE:
                self._enter(MODE_REVERSE, sense.pose, "blocked_nose_unstick")
            return EscapeResult(
                cmd=cmd, obstacle="wait_reverse", active_mode=MODE_REVERSE
            )
        keep_yaw = (
            abs(cmd.vx) < 1e-6
            and abs(cmd.vy) < 1e-6
            and abs(cmd.vtheta) > 1e-6
        )
        return EscapeResult(
            cmd=DriveCommand(0.0, 0.0, cmd.vtheta if keep_yaw else 0.0, False),
            obstacle="wait",
        )

    def tick_post(self, sense: EscapeSense) -> EscapeResult:
        """Undrivable-nose recovery after the footprint guard veto."""
        if sense.loc_hold or sense.corridor_active or self._cooling(sense.now):
            if sense.nose_clear and self.mode in (MODE_REVERSE, MODE_TURN):
                if self.trigger == "undrivable_nose":
                    self.reset()
            return EscapeResult()

        if self.mode == MODE_REVERSE and self.trigger == "undrivable_nose":
            return self._continue_reverse(sense)
        if self.mode == MODE_TURN and self.trigger == "undrivable_nose":
            return self._continue_turn(sense)

        if self.mode != IDLE:
            return EscapeResult()

        # Soft rear threshold: any budge off the bumper counts.
        rear = False
        if sense.guard is not None and sense.guard_pts is not None:
            rear = rectangle_rear_open(
                sense.guard,
                sense.pose,
                sense.guard_pts,
                distance_m=max(0.35, sense.backup_dist_m),
                min_free_m=0.05,
            )
        choice = undrivable_nose(
            nose_clear=sense.nose_clear,
            waiting=sense.waiting,
            vx=sense.cmd.vx,
            vtheta=sense.cmd.vtheta,
            rear_open=rear,
            spin_clear=self._guard_spin_clear(sense, sense.bearing_rad),
            bearing_rad=sense.bearing_rad,
        )
        if choice == TURN:
            sign = math.copysign(1.0, sense.bearing_rad or 1.0)
            self._enter(MODE_TURN, sense.pose, "undrivable_nose", yaw_sign=sign)
            return EscapeResult(
                cmd=DriveCommand(0.0, 0.0, sign * UNSTICK_YAW_RATE, False),
                obstacle="avoid",
                active_mode=MODE_TURN,
            )
        if choice == REVERSE:
            self._enter(MODE_REVERSE, sense.pose, "undrivable_nose")
            return self._continue_reverse(sense)
        if sense.nose_clear:
            self.reset()
        return EscapeResult()

"""Multi-point manoeuvres for a turn the body cannot make where it stands.

A rectangular body beside furniture often cannot turn toward its route in
place: the tail swings into whatever is next to it. A person would turn the
other way a little, back up, and then turn. This module finds such a sequence
by searching over short moves (forward, back, turn left, turn right), each
checked against the live obstacle points with the footprint guard's own
collision rules, until the body faces the route with room to drive.

The search replaces single-purpose rules ("back out when the route is
behind", "keep the direction you started in"): those each fixed one layout
and pecked in place in the next.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from ...geom import conversions as conv
from ..motion.base_envelope import ANG_EPS_RAD_S, LIN_EPS_MPS
from ..types import Pose2D

if TYPE_CHECKING:
    from .footprint_guard import FootprintGuard

FORWARD, BACK, LEFT, RIGHT = "F", "B", "L", "R"

STEP_M = 0.05
STEP_RAD = math.radians(5.0)
# Dedupe grid for visited states.
_CELL_M = 0.025
# The search stays this close to where it started: it is for getting out of a
# tight spot, not for route planning.
SEARCH_RADIUS_M = 0.7
MAX_EXPANSIONS = 2500
# Done when the body faces the route this closely with room to drive on
# (inside the follower's 25 degree rotate-to-heading exit).
GOAL_TOL_RAD = math.radians(15.0)
# Slowest speeds the executor will command; a step is only planned where the
# guard would still allow motion at these speeds.
MIN_SPEED_MPS = max(LIN_EPS_MPS, 0.05) + 0.01
MIN_TURN_RAD_S = max(ANG_EPS_RAD_S, 0.08) + 0.04
SPEED_MPS = 0.12
TURN_RAD_S = 0.5
# Ticks of zero command between segments, so the rate limiter does not blend
# the end of a turn into the start of a straight (an arc nobody checked).
SETTLE_TICKS = 3
_SEG_DONE_M = 0.015
_SEG_DONE_RAD = math.radians(2.5)

# Search costs, in steps. Reversing costs a little more than driving forward,
# and changing what the robot is doing costs more than continuing, so plans
# are a few long moves rather than many short ones.
_COST = {FORWARD: 1.0, BACK: 1.3, LEFT: 1.0, RIGHT: 1.0}
_SWITCH_COST = 4.0
_REVERSAL_COST = 8.0
_OPPOSITE = {FORWARD: BACK, BACK: FORWARD, LEFT: RIGHT, RIGHT: LEFT}
_HEURISTIC_WEIGHT = 1.0


@dataclass(frozen=True)
class Segment:
    kind: str  # F | B | L | R
    amount: float  # metres or radians


def _margins(guard: "FootprintGuard") -> Tuple[float, float]:
    """Free travel / sweep a move needs beyond itself for the guard to allow it."""
    cfg = guard.cfg
    translate = cfg.stop_gap_m + MIN_SPEED_MPS * cfg.time_to_collision_s
    rotate = MIN_TURN_RAD_S * 0.5 + cfg.rot_step_rad
    return translate, rotate


def _moves_free(
    guard: "FootprintGuard", x: float, y: float, th: float, pts: np.ndarray
) -> Tuple[bool, bool, bool, bool]:
    """Whether one step forward / back / left / right is allowed from a pose."""
    from .footprint_guard import _arc_poses, _rot_poses

    cfg = guard.cfg
    pose = Pose2D(x, y, th)
    t_margin, r_margin = _margins(guard)
    fwd = _arc_poses(pose, 1.0, 0.0, STEP_M + t_margin, cfg.step_m)
    back = _arc_poses(pose, -1.0, 0.0, STEP_M + t_margin, cfg.step_m)
    left = _rot_poses(pose, 1.0, STEP_RAD + r_margin, cfg.rot_step_rad)
    right = _rot_poses(pose, -1.0, STEP_RAD + r_margin, cfg.rot_step_rad)
    far, near, floor = guard._split(pose, pts)  # noqa: SLF001
    lin = guard._hits(np.concatenate([fwd, back]), far, near, floor)  # noqa: SLF001
    p_rot = min(cfg.rot_padding_m, cfg.padding_m)
    if abs(p_rot - cfg.padding_m) > 1e-9:
        far, near, floor = guard._split(pose, pts, p_rot)  # noqa: SLF001
    rot = guard._hits(  # noqa: SLF001
        np.concatenate([left, right]), far, near, floor, p_rot
    )
    nf, nl = len(fwd), len(left)
    return (
        not lin[:nf].any(),
        not lin[nf:].any(),
        not rot[:nl].any(),
        not rot[nl:].any(),
    )


def target_is_free(guard: "FootprintGuard", pts: np.ndarray, target: Pose2D) -> bool:
    """Whether the body could stand on ``target`` at all.

    When the route runs through something that is not on the map, the point
    the follower steers for sits inside it. Turning to face that point is
    pointless and fights the blocked-path logic, which is turning the robot
    away to go round. A manoeuvre is only worth planning toward a point the
    body could reach.
    """
    if not pts.size:
        return True
    d2 = (pts[:, 0] - target.x) ** 2 + (pts[:, 1] - target.y) ** 2
    return bool(d2.min() > guard._hw**2)  # noqa: SLF001


def plan_maneuver(
    guard: "FootprintGuard",
    pose: Pose2D,
    pts: np.ndarray,
    target: Pose2D,
    *,
    max_expansions: int = MAX_EXPANSIONS,
    radius_m: float = SEARCH_RADIUS_M,
    goal_tol_rad: float = GOAL_TOL_RAD,
) -> Optional[List[Segment]]:
    """Moves that leave the body facing ``target`` with room to drive, or None."""
    if pts.size:
        reach = radius_m + STEP_M + 0.3 + math.hypot(guard._hl, guard._hw)  # noqa: SLF001
        d2 = (pts[:, 0] - pose.x) ** 2 + (pts[:, 1] - pose.y) ** 2
        pts = pts[d2 <= reach * reach]

    def err(x: float, y: float, th: float) -> float:
        return abs(conv.normalize_angle(math.atan2(target.y - y, target.x - x) - th))

    def key(x: float, y: float, k: int) -> Tuple[int, int, int]:
        return (round((x - pose.x) / _CELL_M), round((y - pose.y) / _CELL_M), k % 72)

    # Heading is tracked as an integer count of steps from the start heading,
    # so it never drifts off the 5 degree lattice.
    start = (pose.x, pose.y, 0)
    best = {key(pose.x, pose.y, 0): 0.0}
    parent: dict = {}
    counter = 0
    heap: List[tuple] = [
        (
            _HEURISTIC_WEIGHT * err(pose.x, pose.y, pose.theta) / STEP_RAD,
            0,
            0.0,
            start,
            "",
        )
    ]
    expansions = 0
    while heap and expansions < max_expansions:
        _f, _c, g, node, last = heapq.heappop(heap)
        x, y, k = node
        node_key = key(x, y, k)
        if g > best.get(node_key, math.inf) + 1e-9:
            continue
        th = pose.theta + k * STEP_RAD
        expansions += 1
        ok_f, ok_b, ok_l, ok_r = _moves_free(guard, x, y, th, pts)
        if node != start and ok_f and err(x, y, th) <= goal_tol_rad:
            return _segments(parent, node_key)
        c, s = math.cos(th), math.sin(th)
        children = (
            (FORWARD, ok_f, (x + STEP_M * c, y + STEP_M * s, k)),
            (BACK, ok_b, (x - STEP_M * c, y - STEP_M * s, k)),
            (LEFT, ok_l, (x, y, k + 1)),
            (RIGHT, ok_r, (x, y, k - 1)),
        )
        for kind, ok, child in children:
            if not ok:
                continue
            cx, cy, ck = child
            if math.hypot(cx - pose.x, cy - pose.y) > radius_m:
                continue
            step = _COST[kind]
            if last and kind != last:
                step += _REVERSAL_COST if _OPPOSITE[kind] == last else _SWITCH_COST
            ng = g + step
            child_key = key(cx, cy, ck)
            if ng >= best.get(child_key, math.inf) - 1e-9:
                continue
            best[child_key] = ng
            parent[child_key] = (node_key, kind)
            counter += 1
            h = err(cx, cy, pose.theta + ck * STEP_RAD) / STEP_RAD
            heapq.heappush(heap, (ng + _HEURISTIC_WEIGHT * h, counter, ng, child, kind))
    return None


def _segments(parent: dict, goal_key) -> List[Segment]:
    kinds: List[str] = []
    cur = goal_key
    while cur in parent:
        cur, kind = parent[cur]
        kinds.append(kind)
    kinds.reverse()
    out: List[Segment] = []
    for kind in kinds:
        amount = STEP_M if kind in (FORWARD, BACK) else STEP_RAD
        if out and out[-1].kind == kind:
            out[-1] = Segment(kind, out[-1].amount + amount)
        else:
            out.append(Segment(kind, amount))
    return out


class ManeuverRunner:
    """Executes a planned manoeuvre one control tick at a time.

    ``step`` returns ``(vx, vtheta)`` while the plan is running and ``None``
    once it has finished or can no longer be followed. Every command is
    re-checked against the live obstacle points, with the limits the guard's
    own veto applies, so a plan made on an older scan cannot drive into
    something that has since appeared.
    """

    # Ticks to wait before searching again after a search found nothing: a
    # failed search is the expensive kind and the answer rarely changes in
    # the next 50 ms.
    RETRY_TICKS = 30
    # Ticks to hold still after a plan had to be abandoned (something moved,
    # or the pose shifted) before planning again from the new situation.
    ABORT_HOLD_TICKS = 6

    def __init__(self) -> None:
        self._plan: List[Segment] = []
        self._i = 0
        self._seg_start: Optional[Pose2D] = None
        self._last_theta = 0.0
        self._turned = 0.0
        self._settle = 0
        self._retry_in = 0
        self._hold = 0
        self.plans_made = 0

    @property
    def active(self) -> bool:
        return self._i < len(self._plan)

    @property
    def plan(self) -> List[Segment]:
        return list(self._plan[self._i :])

    def cancel(self) -> None:
        self._plan = []
        self._i = 0
        self._seg_start = None
        self._settle = 0

    @property
    def holding(self) -> bool:
        """True for a few ticks after an abandoned plan: stand still."""
        return self._hold > 0

    def tick(self) -> None:
        if self._retry_in > 0:
            self._retry_in -= 1
        if self._hold > 0:
            self._hold -= 1

    def _abort(self) -> None:
        self.cancel()
        self._hold = self.ABORT_HOLD_TICKS

    def start(
        self,
        guard: "FootprintGuard",
        pose: Pose2D,
        pts: np.ndarray,
        target: Optional[Pose2D],
    ) -> Optional[Tuple[float, float]]:
        """Plan from ``pose`` and return the first command, or None."""
        self.cancel()
        if target is None or self._retry_in > 0 or self._hold > 0:
            return None
        if not target_is_free(guard, pts, target):
            return None
        plan = plan_maneuver(guard, pose, pts, target)
        if not plan:
            self._retry_in = self.RETRY_TICKS
            return None
        self._plan = plan
        self.plans_made += 1
        return self.step(guard, pose, pts)

    def step(
        self, guard: "FootprintGuard", pose: Pose2D, pts: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        cfg = guard.cfg
        while self.active:
            if self._settle > 0:
                self._settle -= 1
                return 0.0, 0.0
            seg = self._plan[self._i]
            if self._seg_start is None:
                self._seg_start = pose
                self._last_theta = pose.theta
                self._turned = 0.0
            s0 = self._seg_start
            if seg.kind in (FORWARD, BACK):
                sign = 1.0 if seg.kind == FORWARD else -1.0
                travelled = sign * (
                    (pose.x - s0.x) * math.cos(s0.theta)
                    + (pose.y - s0.y) * math.sin(s0.theta)
                )
                remaining = seg.amount - travelled
                if remaining <= _SEG_DONE_M:
                    self._next_segment()
                    continue
                free = guard.free_distance(
                    pose,
                    sign * SPEED_MPS,
                    0.0,
                    pts,
                    min(cfg.horizon_m, remaining + 0.3),
                )
                # Ease off toward the end of the move: the pose lags the
                # robot by a few tenths of a second on a real base.
                v = min(SPEED_MPS, max(MIN_SPEED_MPS, remaining * 1.5))
                if math.isfinite(free):
                    v = min(v, 0.9 * (free - cfg.stop_gap_m) / cfg.time_to_collision_s)
                if v < LIN_EPS_MPS:
                    if remaining <= STEP_M:
                        # Close enough: the last few cm were the margin.
                        self._next_segment()
                        continue
                    self._abort()
                    return None
                return sign * v, 0.0
            sign = 1.0 if seg.kind == LEFT else -1.0
            # Summed tick by tick: a turn the long way round is over 180 deg.
            self._turned += sign * conv.normalize_angle(pose.theta - self._last_theta)
            self._last_theta = pose.theta
            remaining = seg.amount - self._turned
            if remaining <= _SEG_DONE_RAD:
                self._next_segment()
                continue
            free = guard.free_rotation(pose, sign, pts)
            w = min(TURN_RAD_S, max(MIN_TURN_RAD_S, remaining * 2.0))
            if math.isfinite(free):
                # A little under what the veto allows, so the rate limiter's
                # lag never pushes the command over it.
                w = min(w, 0.9 * (free - cfg.rot_step_rad) / 0.5)
            if w < ANG_EPS_RAD_S:
                if remaining <= STEP_RAD:
                    self._next_segment()
                    continue
                self._abort()
                return None
            return 0.0, sign * w
        self.cancel()
        return None

    def _next_segment(self) -> None:
        self._i += 1
        self._seg_start = None
        self._settle = SETTLE_TICKS if self.active else 0

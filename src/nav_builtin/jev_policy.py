"""Optional TypeSafe/Jev policy overlay for local-block nav decisions.

Modes (``builtin.nav_policy``):

- ``heuristic`` — existing wait / keep_dwa / replan rules only (default).
- ``shadow`` — call Jev, log heuristic vs Jev, **execute heuristic**.
- ``jev`` — call Jev; if confidence ≥ threshold execute Jev's choice,
  else fall back to heuristic (still log both).

Never runs on the free-path control tick — only when the supervisor already
sees a local block and would consult ``_local_block_action``.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

NAV_POLICY_HEURISTIC = "heuristic"
NAV_POLICY_SHADOW = "shadow"
NAV_POLICY_JEV = "jev"
NAV_POLICIES = (NAV_POLICY_HEURISTIC, NAV_POLICY_SHADOW, NAV_POLICY_JEV)

# Supervisor local-block actions (must stay in sync with _local_block_action).
ACTION_WAIT = "wait"
ACTION_KEEP_DWA = "keep_dwa"
ACTION_REPLAN = "replan"
ACTION_BACKUP = "backup"
ACTIONS = (ACTION_WAIT, ACTION_KEEP_DWA, ACTION_REPLAN, ACTION_BACKUP)


def normalize_nav_policy(raw: Any) -> str:
    name = str(raw or NAV_POLICY_HEURISTIC).strip().lower()
    if name in ("typesafe", "ai"):
        return NAV_POLICY_JEV
    if name not in NAV_POLICIES:
        raise ValueError(
            f"builtin.nav_policy must be one of {list(NAV_POLICIES)}; got {raw!r}"
        )
    return name


@dataclass
class ObstacleSample:
    """One tick of forward / path-block context for mover-vs-fixed judgment."""

    t: float
    nose_clear: bool
    forward_clearance_m: float
    path_ahead_cost: int
    obstacle_state: str
    nearest_range_m: Optional[float] = None
    nearest_bearing_rad: Optional[float] = None
    pose_x: Optional[float] = None
    pose_y: Optional[float] = None
    pose_theta: Optional[float] = None


@dataclass
class LocalBlockContext:
    """Inputs for a single wait / peel / replan / backup decision."""

    heuristic_action: str
    nose_clear: bool
    blocked_for_s: float
    wait_before_replan_s: float
    replan_cooldown_ready: bool
    path_ahead_cost: int
    obstacle_state: str
    forward_clearance_m: float
    failed_replan_while_blocked: int = 0
    remaining_path_m: Optional[float] = None
    nearest_range_m: Optional[float] = None
    nearest_bearing_rad: Optional[float] = None
    pose_xy: Optional[tuple[float, float]] = None
    goal_xy: Optional[tuple[float, float]] = None
    rear_clearance_m: Optional[float] = None
    backup_feasible: bool = False
    backup_attempts_remaining: int = 0
    backup_cooldown_ready: bool = True
    # Richer stuck / peel-loop signals (optional; supervisor fills when known).
    block_reasons: Sequence[str] = ()
    pose_cost: int = 0
    left_clearance_m: Optional[float] = None
    right_clearance_m: Optional[float] = None
    cmd_vx_mps: Optional[float] = None
    cmd_vtheta_rad_s: Optional[float] = None
    bearing_error_rad: Optional[float] = None
    local_planner_active: bool = False
    backup_denial_reason: str = ""
    last_replan_accepted: Optional[str] = None
    last_replan_attempts: Sequence[str] = ()
    last_replan_error: str = ""
    pose_theta: Optional[float] = None
    # Age of the last replan attempt (None = none this block).
    last_replan_age_s: Optional[float] = None


@dataclass
class PolicyDecision:
    """Result of consulting (or skipping) Jev for one local-block tick."""

    applied_action: str
    heuristic_action: str
    jev_action: Optional[str] = None
    mode: str = NAV_POLICY_HEURISTIC
    confidence: Optional[float] = None
    fallback_reason: str = ""
    queried: bool = False
    latency_s: Optional[float] = None
    features: Dict[str, Any] = field(default_factory=dict)
    answers: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mode": self.mode,
            "applied_action": self.applied_action,
            "heuristic_action": self.heuristic_action,
            "queried": self.queried,
        }
        if self.jev_action is not None:
            out["jev_action"] = self.jev_action
        if self.confidence is not None:
            out["confidence"] = round(float(self.confidence), 4)
        if self.fallback_reason:
            out["fallback_reason"] = self.fallback_reason
        if self.latency_s is not None:
            out["latency_s"] = round(float(self.latency_s), 4)
        if self.features:
            out["features"] = dict(self.features)
        if self.answers:
            out["answers"] = dict(self.answers)
        if self.error:
            out["error"] = self.error
        return out


def obstacle_motion_features(
    history: Sequence[ObstacleSample], *, window_s: float = 2.5
) -> Dict[str, Any]:
    """Summarize recent obstacle samples for mover-vs-fixed judgment.

    Pure feature extraction — unit-tested without TypeSafe.
    """
    if not history:
        return {
            "samples": 0,
            "dwell_s": 0.0,
            "range_delta_m": 0.0,
            "bearing_span_rad": 0.0,
            "clearance_trend_m": 0.0,
            "motion_score": 0.0,
            "likely_mover": False,
        }
    t_now = history[-1].t
    window = [s for s in history if t_now - s.t <= window_s]
    if not window:
        window = list(history[-1:])
    dwell_s = max(0.0, window[-1].t - window[0].t)
    ranges = [s.nearest_range_m for s in window if s.nearest_range_m is not None]
    bearings = [
        s.nearest_bearing_rad for s in window if s.nearest_bearing_rad is not None
    ]
    clearances = [float(s.forward_clearance_m) for s in window]
    range_delta = 0.0
    if len(ranges) >= 2:
        range_delta = float(ranges[-1] - ranges[0])
    bearing_span = 0.0
    if len(bearings) >= 2:
        bearing_span = float(max(bearings) - min(bearings))
    clearance_trend = 0.0
    if len(clearances) >= 2:
        clearance_trend = float(clearances[-1] - clearances[0])
    # Heuristic motion score in [0, 1]: moving range/bearing → mover-like.
    motion = 0.0
    motion += min(1.0, abs(range_delta) / 0.4) * 0.45
    motion += min(1.0, abs(bearing_span) / 0.35) * 0.35
    motion += min(1.0, abs(clearance_trend) / 0.3) * 0.20
    likely_mover = motion >= 0.35 and dwell_s >= 0.3
    return {
        "samples": len(window),
        "dwell_s": round(dwell_s, 3),
        "range_delta_m": round(range_delta, 3),
        "bearing_span_rad": round(bearing_span, 3),
        "clearance_trend_m": round(clearance_trend, 3),
        "motion_score": round(motion, 3),
        "likely_mover": bool(likely_mover),
        "nose_clear_last": bool(window[-1].nose_clear),
        "path_ahead_cost_last": int(window[-1].path_ahead_cost),
        "obstacle_state_last": str(window[-1].obstacle_state),
    }


def _wrap_pi(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def stuck_progress_features(
    history: Sequence[ObstacleSample], *, window_s: float = 3.0
) -> Dict[str, Any]:
    """Raw motion facts while blocked: meters advanced and yaw change.

    No verdicts — Jev decides what "stuck" means given what we were doing.
    """
    empty = {
        "progress_m": 0.0,
        "yaw_delta_rad": 0.0,
        "window_s": 0.0,
    }
    if not history:
        return empty
    t_now = history[-1].t
    window = [
        s
        for s in history
        if t_now - s.t <= window_s
        and s.pose_x is not None
        and s.pose_y is not None
    ]
    if len(window) < 2:
        return empty
    first, last = window[0], window[-1]
    assert first.pose_x is not None and first.pose_y is not None
    assert last.pose_x is not None and last.pose_y is not None
    progress_m = math.hypot(last.pose_x - first.pose_x, last.pose_y - first.pose_y)
    yaw_delta = 0.0
    if first.pose_theta is not None and last.pose_theta is not None:
        yaw_delta = abs(_wrap_pi(float(last.pose_theta) - float(first.pose_theta)))
    return {
        "progress_m": round(progress_m, 3),
        "yaw_delta_rad": round(yaw_delta, 3),
        "window_s": round(max(0.0, last.t - first.t), 3),
    }


def classify_block_reasons(
    *,
    path_ahead_cost: int,
    activate_cost: int,
    pose_cost: int,
    inscribed_cost: int,
    obstacle_state: str,
    reactive_avoid_for_s: float = 0.0,
) -> List[str]:
    """Why local_blocked is true (may be multiple)."""
    reasons: List[str] = []
    if int(path_ahead_cost) >= int(activate_cost):
        reasons.append("path_cost")
    if int(pose_cost) >= int(inscribed_cost):
        reasons.append("pose_inscribed")
    if str(obstacle_state) == "in_lethal":
        reasons.append("in_lethal")
    if str(obstacle_state) == "avoid" and float(reactive_avoid_for_s) >= 0.8:
        reasons.append("reactive_avoid")
    return reasons or ["unknown"]


def backup_denial_reason(
    *,
    enabled: bool,
    feasible: bool,
    attempts_remaining: int,
    cooldown_ready: bool,
    has_scan: bool,
    rear_clearance_m: Optional[float],
    rear_clear_min_m: float,
    reverse_footprint_ok: Optional[bool],
) -> str:
    """Why backup_feasible is false (empty string when feasible)."""
    if feasible:
        return ""
    if not enabled:
        return "disabled"
    if int(attempts_remaining) <= 0:
        return "attempts_exhausted"
    if not cooldown_ready:
        return "cooldown"
    if not has_scan:
        return "no_scan"
    if rear_clearance_m is None:
        return "no_rear"
    if float(rear_clearance_m) < float(rear_clear_min_m):
        return "rear_blocked"
    if reverse_footprint_ok is False:
        return "reverse_footprint"
    return "unknown"


def build_policy_state(
    ctx: LocalBlockContext,
    motion: Mapping[str, Any],
    *,
    recent_actions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact JSON state for TypeSafe ``system_one``.

    Facts only. No "prefer X when Y" hints — Jev is the decider; code keeps
    just the hard safety gates (see ``map_jev_to_action``).
    """
    stuck_keys = ("progress_m", "yaw_delta_rad", "window_s")
    state: Dict[str, Any] = {
        "task": "mobile_robot_local_navigation_block",
        "heuristic_action": ctx.heuristic_action,
        "nose_clear": ctx.nose_clear,
        "blocked_for_s": round(float(ctx.blocked_for_s), 3),
        "wait_before_replan_s": round(float(ctx.wait_before_replan_s), 3),
        "replan_cooldown_ready": bool(ctx.replan_cooldown_ready),
        "path_ahead_cost": int(ctx.path_ahead_cost),
        "pose_cost": int(ctx.pose_cost),
        "block_reasons": list(ctx.block_reasons) or ["unknown"],
        "obstacle_state": ctx.obstacle_state,
        "forward_clearance_m": round(float(ctx.forward_clearance_m), 3),
        "failed_replan_while_blocked": int(ctx.failed_replan_while_blocked),
        "obstacle_motion": {
            k: v for k, v in dict(motion).items() if k not in stuck_keys
        },
        # Robot displacement / yaw over the recent window. Read alongside
        # recent_actions: zero progress while waiting is expected, zero
        # progress while keep_dwa means peeling is not working.
        "motion_while_blocked": {k: motion[k] for k in stuck_keys if k in motion},
        "recent_actions": dict(recent_actions or {}),
        "actions": {
            ACTION_WAIT: (
                "Stop translating (rotate-to-heading still allowed). Costs time; "
                "only helps if the blocker moves away."
            ),
            ACTION_KEEP_DWA: (
                "Keep the current global path and let the local planner "
                "inch/peel around the block. Cheap if a gap exists; makes no "
                "progress if the corridor is truly sealed."
            ),
            ACTION_REPLAN: (
                "Plan a new global path from the current pose with the block "
                "painted in. The detour may be longer; recent attempt outcomes "
                "are in last_replan."
            ),
            ACTION_BACKUP: (
                "Reverse a short distance, then replan from the new pose. Only "
                "executable when backup.feasible is true."
            ),
        },
        "backup": {
            "feasible": bool(ctx.backup_feasible),
            "attempts_remaining": int(ctx.backup_attempts_remaining),
            "cooldown_ready": bool(ctx.backup_cooldown_ready),
            "denial_reason": str(ctx.backup_denial_reason or ""),
        },
        "motion_cmd": {
            "vx_mps": (
                round(float(ctx.cmd_vx_mps), 3)
                if ctx.cmd_vx_mps is not None
                else None
            ),
            "vtheta_rad_s": (
                round(float(ctx.cmd_vtheta_rad_s), 3)
                if ctx.cmd_vtheta_rad_s is not None
                else None
            ),
            "bearing_error_rad": (
                round(float(ctx.bearing_error_rad), 3)
                if ctx.bearing_error_rad is not None
                else None
            ),
            "local_planner_active": bool(ctx.local_planner_active),
        },
        "last_replan": {
            "accepted": ctx.last_replan_accepted,
            "error": str(ctx.last_replan_error or ""),
            "attempts": list(ctx.last_replan_attempts)[:6],
            "age_s": (
                round(float(ctx.last_replan_age_s), 1)
                if ctx.last_replan_age_s is not None
                else None
            ),
        },
    }
    if ctx.remaining_path_m is not None:
        state["remaining_path_m"] = round(float(ctx.remaining_path_m), 2)
    if ctx.nearest_range_m is not None:
        state["nearest_range_m"] = round(float(ctx.nearest_range_m), 3)
    if ctx.nearest_bearing_rad is not None:
        state["nearest_bearing_rad"] = round(float(ctx.nearest_bearing_rad), 3)
    if ctx.rear_clearance_m is not None:
        state["rear_clearance_m"] = round(float(ctx.rear_clearance_m), 3)
    if ctx.left_clearance_m is not None:
        state["left_clearance_m"] = round(float(ctx.left_clearance_m), 3)
    if ctx.right_clearance_m is not None:
        state["right_clearance_m"] = round(float(ctx.right_clearance_m), 3)
    if ctx.pose_xy is not None:
        state["pose_xy"] = [round(ctx.pose_xy[0], 3), round(ctx.pose_xy[1], 3)]
    if ctx.goal_xy is not None:
        state["goal_xy"] = [round(ctx.goal_xy[0], 3), round(ctx.goal_xy[1], 3)]
    return state


def build_policy_questions() -> Dict[str, Any]:
    """TypeSafe questions for one local-block consult (lazy SDK import inside)."""
    from typesafe_sdk import Choice, Noul, Score

    return {
        "action": Choice(
            instructions=(
                "A mobile robot following a planned path is locally blocked. "
                "Choose the single next action that gets it to the goal soonest "
                "without collision. The state gives obstacle clearances, obstacle "
                "motion history, robot displacement over the recent window, which "
                "actions the controller has already been applying and for how "
                "long (recent_actions), and the outcomes of recent global replan "
                "attempts (last_replan). backup is only executable when "
                "backup.feasible is true. Weigh each action by whether it is "
                "likely to change the situation given what has already been tried."
            ),
            criteria={
                ACTION_WAIT: "Stop and hold position",
                ACTION_KEEP_DWA: "Keep the path; local planner peels/inches",
                ACTION_REPLAN: "Compute a new global path from here",
                ACTION_BACKUP: "Reverse briefly, then replan",
            },
        ),
        "temporary_mover": Noul(
            instructions=(
                "Is the blocker likely a temporary mover (person, cart) rather than "
                "a fixed obstacle? Use obstacle_motion.motion_score, range_delta_m, "
                "bearing_span_rad, and how long the block has lasted."
            )
        ),
        "gap_worth_trying": Noul(
            instructions=(
                "Is it plausible the robot can inch/peel through or around the "
                "block from here, given forward/left/right clearances and what "
                "keep_dwa has achieved so far (recent_actions, motion_while_blocked)?"
            )
        ),
        "should_backup": Noul(
            instructions=(
                "Would reversing a short distance and replanning from there "
                "likely open a path that is not available from the current pose? "
                "Only meaningful when backup.feasible is true."
            )
        ),
        "replan_urgency": Score(
            instructions=(
                "How likely is a fresh global replan from the current pose to "
                "find a usable path, given last_replan outcomes and their age?"
            ),
            criteria=["low", "medium", "high"],
        ),
    }


def _answers_to_dict(result: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    choices = getattr(result, "choices", None) or {}
    nouls = getattr(result, "nouls", None) or {}
    scores = getattr(result, "scores", None) or {}
    for name, ans in choices.items():
        out[name] = {
            "type": "choice",
            "choice": getattr(ans, "choice", None),
            "confidence": getattr(ans, "confidence", None),
            "probabilities": dict(getattr(ans, "probabilities", {}) or {}),
        }
    for name, ans in nouls.items():
        out[name] = {
            "type": "noul",
            "noul": getattr(ans, "noul", None),
        }
    for name, ans in scores.items():
        out[name] = {
            "type": "score",
            "score": getattr(ans, "score", None),
            "confidence": getattr(ans, "confidence", None),
        }
    return out


def map_jev_to_action(
    *,
    choice: Optional[str],
    heuristic_action: str,
    backup_feasible: bool = False,
    **_ignored: Any,
) -> str:
    """Turn Jev's Choice into a supervisor action.

    Jev decides. Code applies only hard safety gates:
    - unknown choice → heuristic
    - ``backup`` when reverse is not physically feasible → heuristic (or wait)
    Reactive stops in the follower always win regardless of this mapping.
    """
    action = str(choice or "").strip().lower()
    if action not in ACTIONS:
        return heuristic_action if heuristic_action in ACTIONS else ACTION_WAIT
    if action == ACTION_BACKUP and not backup_feasible:
        # Never reverse into unknown rear space.
        return (
            heuristic_action
            if heuristic_action in ACTIONS and heuristic_action != ACTION_BACKUP
            else ACTION_WAIT
        )
    return action


QueryFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]


class JevNavPolicy:
    """Rate-limited Jev consult for local-block decisions."""

    def __init__(
        self,
        *,
        mode: str = NAV_POLICY_HEURISTIC,
        min_confidence: float = 0.7,
        timeout_s: float = 1.25,
        min_period_s: float = 1.0,
        history_s: float = 3.0,
        model: str = "jev-latest",
        api_key: Optional[str] = None,
        query_fn: Optional[QueryFn] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.mode = normalize_nav_policy(mode)
        self.min_confidence = float(min_confidence)
        self.timeout_s = max(0.1, float(timeout_s))
        self.min_period_s = max(0.0, float(min_period_s))
        self.history_s = max(0.5, float(history_s))
        self.model = str(model or "jev-latest")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self._query_fn = query_fn
        self._log = logger
        self._history: Deque[ObstacleSample] = deque(maxlen=64)
        self._last_query_at = 0.0
        self._last_decision: Optional[PolicyDecision] = None
        self._client = None
        self._decision_log: Deque[Dict[str, Any]] = deque(maxlen=256)
        self._decision_seq = 0
        self._run_id = 0
        # What we have actually been applying during this block (facts for Jev).
        self._applied_history: Deque[tuple[float, str]] = deque(maxlen=400)
        self._last_state_key: Optional[tuple] = None

    def _emit(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
                return
            except Exception:  # noqa: BLE001
                pass
        LOGGER.info(msg)

    def record_obstacle(self, sample: ObstacleSample) -> None:
        self._history.append(sample)
        cutoff = sample.t - self.history_s
        while self._history and self._history[0].t < cutoff:
            self._history.popleft()

    def clear_history(self) -> None:
        self._history.clear()
        self._applied_history.clear()
        self._last_state_key = None

    def start_run(self) -> int:
        """Begin a new navigate run; clears the decision timeline for the UI."""
        self._run_id += 1
        self._decision_log.clear()
        self._decision_seq = 0
        self._last_decision = None
        self.clear_history()
        return self._run_id

    def _record_applied(self, t: float, action: str) -> None:
        self._applied_history.append((t, action))
        cutoff = t - max(self.history_s, 10.0)
        while self._applied_history and self._applied_history[0][0] < cutoff:
            self._applied_history.popleft()

    def _recent_actions(self, now: float) -> Dict[str, Any]:
        """Facts about what the controller has been doing during this block."""
        if not self._applied_history:
            return {"current": None, "current_for_s": 0.0, "time_share_s": {}}
        current = self._applied_history[-1][1]
        current_since = self._applied_history[-1][0]
        for t, a in reversed(self._applied_history):
            if a != current:
                break
            current_since = t
        share: Dict[str, float] = {}
        prev_t: Optional[float] = None
        prev_a: Optional[str] = None
        for t, a in self._applied_history:
            if prev_t is not None and prev_a is not None:
                share[prev_a] = share.get(prev_a, 0.0) + max(0.0, t - prev_t)
            prev_t, prev_a = t, a
        if prev_t is not None and prev_a is not None:
            share[prev_a] = share.get(prev_a, 0.0) + max(0.0, now - prev_t)
        return {
            "current": current,
            "current_for_s": round(max(0.0, now - current_since), 2),
            "time_share_s": {k: round(v, 2) for k, v in share.items()},
        }

    @staticmethod
    def _state_key(ctx: LocalBlockContext) -> tuple:
        """Coarse situation fingerprint; a change invalidates a reused answer."""
        return (
            bool(ctx.nose_clear),
            bool(ctx.backup_feasible),
            tuple(ctx.block_reasons),
            int(ctx.failed_replan_while_blocked),
            ctx.last_replan_accepted is not None,
        )

    def clear_decision_log(self) -> None:
        self._decision_log.clear()
        self._decision_seq = 0

    def decision_log(self) -> List[Dict[str, Any]]:
        return list(self._decision_log)

    def _append_decision_log(
        self, ctx: LocalBlockContext, decision: PolicyDecision
    ) -> None:
        if self.mode == NAV_POLICY_HEURISTIC:
            return
        # Skip pure rate-limit echoes that change nothing (noise for the UI).
        if (
            not decision.queried
            and decision.fallback_reason.startswith("rate_limited")
            and decision.applied_action == decision.heuristic_action
            and self._decision_log
            and self._decision_log[-1].get("applied_action")
            == decision.applied_action
            and self._decision_log[-1].get("jev_action") == decision.jev_action
        ):
            return
        self._decision_seq += 1
        entry = decision.to_dict()
        entry.update(
            {
                "seq": self._decision_seq,
                "run_id": self._run_id,
                "t_wall": time.time(),
                "t_mono": time.monotonic(),
            }
        )
        if ctx.pose_xy is not None:
            entry["pose"] = {
                "x": round(ctx.pose_xy[0], 4),
                "y": round(ctx.pose_xy[1], 4),
            }
        if ctx.goal_xy is not None:
            entry["goal"] = {
                "x": round(ctx.goal_xy[0], 4),
                "y": round(ctx.goal_xy[1], 4),
            }
        if ctx.nearest_range_m is not None:
            entry["nearest_range_m"] = round(float(ctx.nearest_range_m), 3)
        if ctx.nearest_bearing_rad is not None:
            entry["nearest_bearing_rad"] = round(float(ctx.nearest_bearing_rad), 3)
        if ctx.rear_clearance_m is not None:
            entry["rear_clearance_m"] = round(float(ctx.rear_clearance_m), 3)
        entry["backup_feasible"] = bool(ctx.backup_feasible)
        self._decision_log.append(entry)

    @property
    def last_decision(self) -> Optional[PolicyDecision]:
        return self._last_decision

    def decide(self, ctx: LocalBlockContext) -> PolicyDecision:
        """Return the action to apply; always logs both sides in shadow/jev."""
        now = time.monotonic()
        motion = {
            **obstacle_motion_features(list(self._history)),
            **stuck_progress_features(list(self._history)),
        }
        recent = self._recent_actions(now)
        features = {
            **motion,
            "nose_clear": ctx.nose_clear,
            "blocked_for_s": round(float(ctx.blocked_for_s), 3),
            "path_ahead_cost": int(ctx.path_ahead_cost),
            "pose_cost": int(ctx.pose_cost),
            "block_reasons": list(ctx.block_reasons) or ["unknown"],
            "failed_replan_while_blocked": int(ctx.failed_replan_while_blocked),
            "forward_clearance_m": round(float(ctx.forward_clearance_m), 3),
            "replan_cooldown_ready": bool(ctx.replan_cooldown_ready),
            "backup_feasible": bool(ctx.backup_feasible),
            "backup_denial_reason": str(ctx.backup_denial_reason or ""),
            "local_planner_active": bool(ctx.local_planner_active),
            "recent_actions": recent,
        }
        if ctx.remaining_path_m is not None:
            features["remaining_path_m"] = round(float(ctx.remaining_path_m), 2)
        if ctx.left_clearance_m is not None:
            features["left_clearance_m"] = round(float(ctx.left_clearance_m), 3)
        if ctx.right_clearance_m is not None:
            features["right_clearance_m"] = round(float(ctx.right_clearance_m), 3)
        if ctx.cmd_vx_mps is not None:
            features["cmd_vx_mps"] = round(float(ctx.cmd_vx_mps), 3)
        if ctx.cmd_vtheta_rad_s is not None:
            features["cmd_vtheta_rad_s"] = round(float(ctx.cmd_vtheta_rad_s), 3)
        if ctx.bearing_error_rad is not None:
            features["bearing_error_rad"] = round(float(ctx.bearing_error_rad), 3)
        if ctx.last_replan_attempts:
            features["last_replan_attempts"] = list(ctx.last_replan_attempts)[:6]
        if ctx.last_replan_age_s is not None:
            features["last_replan_age_s"] = round(float(ctx.last_replan_age_s), 1)

        if self.mode == NAV_POLICY_HEURISTIC:
            decision = PolicyDecision(
                applied_action=ctx.heuristic_action,
                heuristic_action=ctx.heuristic_action,
                mode=self.mode,
                features=features,
            )
            self._last_decision = decision
            self._record_applied(now, decision.applied_action)
            return decision

        state_key = self._state_key(ctx)
        situation_changed = (
            self._last_state_key is not None and state_key != self._last_state_key
        )
        rate_limited = (
            self.min_period_s > 0
            and (now - self._last_query_at) < self.min_period_s
        )
        if rate_limited and not situation_changed:
            # Reuse last Jev suggestion if still fresh; else heuristic.
            prev = self._last_decision
            reused = prev.jev_action if prev and prev.jev_action else None
            if self.mode == NAV_POLICY_SHADOW:
                applied = ctx.heuristic_action
                reason = "rate_limited_shadow"
            elif (
                reused
                and prev
                and prev.confidence is not None
                and prev.confidence >= self.min_confidence
            ):
                applied = reused
                reason = "rate_limited_reuse_jev"
            else:
                applied = ctx.heuristic_action
                reason = "rate_limited_heuristic"
            decision = PolicyDecision(
                applied_action=applied,
                heuristic_action=ctx.heuristic_action,
                jev_action=reused,
                mode=self.mode,
                confidence=prev.confidence if prev else None,
                fallback_reason=reason,
                queried=False,
                features=features,
                answers=dict(prev.answers) if prev else {},
            )
            self._last_decision = decision
            self._record_applied(now, decision.applied_action)
            self._append_decision_log(ctx, decision)
            self._emit(f"jev_policy {decision.to_dict()}")
            return decision

        state = build_policy_state(ctx, motion, recent_actions=recent)
        t0 = time.monotonic()
        try:
            result = self._query(state)
            latency = time.monotonic() - t0
            self._last_query_at = time.monotonic()
            self._last_state_key = state_key
            answers = _answers_to_dict(result)
            choice_ans = answers.get("action") or {}
            jev_choice = choice_ans.get("choice")
            confidence = choice_ans.get("confidence")
            mapped = map_jev_to_action(
                choice=str(jev_choice) if jev_choice is not None else None,
                heuristic_action=ctx.heuristic_action,
                backup_feasible=bool(ctx.backup_feasible),
            )
            fallback = "" if not situation_changed else "requeried_state_change"
            if self.mode == NAV_POLICY_SHADOW:
                applied = ctx.heuristic_action
                fallback = "shadow"
            else:
                conf = float(confidence) if confidence is not None else 0.0
                if confidence is None or conf < self.min_confidence:
                    applied = ctx.heuristic_action
                    fallback = f"low_confidence:{conf:.2f}"
                else:
                    applied = mapped
            decision = PolicyDecision(
                applied_action=applied,
                heuristic_action=ctx.heuristic_action,
                jev_action=mapped,
                mode=self.mode,
                confidence=float(confidence) if confidence is not None else None,
                fallback_reason=fallback,
                queried=True,
                latency_s=latency,
                features=features,
                answers=answers,
            )
        except Exception as exc:  # noqa: BLE001 - never break nav
            latency = time.monotonic() - t0
            decision = PolicyDecision(
                applied_action=ctx.heuristic_action,
                heuristic_action=ctx.heuristic_action,
                mode=self.mode,
                fallback_reason="query_error",
                queried=True,
                latency_s=latency,
                features=features,
                error=str(exc),
            )
        self._last_decision = decision
        self._record_applied(now, decision.applied_action)
        self._append_decision_log(ctx, decision)
        self._emit(f"jev_policy {decision.to_dict()}")
        return decision

    def _query(self, state: Mapping[str, Any]) -> Any:
        if self._query_fn is not None:
            return self._query_fn(state, build_policy_questions())
        client = self._ensure_client()
        from typesafe_sdk import RetryPolicy

        return client.system_one(
            state,
            build_policy_questions(),
            retry=RetryPolicy(max_retries=0, timeout=self.timeout_s),
        )

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "typesafe-sdk is required for nav_policy shadow/jev; "
                "pip install typesafe-sdk"
            ) from exc
        if not self.api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY (or builtin.jev_api_key) required for "
                f"nav_policy={self.mode!r}"
            )
        self._client = TypeSafeClient(api_key=self.api_key, model=self.model)
        return self._client


def nearest_scan_obstacle(
    scan: Any, *, half_cone_rad: float = 0.6
) -> tuple[Optional[float], Optional[float]]:
    """Return (range_m, bearing_rad) of nearest hit in a forward cone."""
    if scan is None:
        return None, None
    ranges = getattr(scan, "ranges", None)
    if ranges is None:
        return None, None
    try:
        angle_min = float(getattr(scan, "angle_min", -math.pi))
        angle_inc = float(getattr(scan, "angle_increment", 0.0))
        range_min = float(getattr(scan, "range_min", 0.05))
        range_max = float(getattr(scan, "range_max", 20.0))
    except (TypeError, ValueError):
        return None, None
    best_r: Optional[float] = None
    best_b: Optional[float] = None
    for i, raw in enumerate(ranges):
        try:
            r = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(r) or r < range_min or r > range_max:
            continue
        bearing = angle_min + i * angle_inc
        # Wrap to [-pi, pi]
        while bearing > math.pi:
            bearing -= 2 * math.pi
        while bearing < -math.pi:
            bearing += 2 * math.pi
        if abs(bearing) > half_cone_rad:
            continue
        if best_r is None or r < best_r:
            best_r = r
            best_b = bearing
    return best_r, best_b

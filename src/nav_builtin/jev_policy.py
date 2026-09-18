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
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence

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


def build_policy_state(
    ctx: LocalBlockContext, motion: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compact JSON state for TypeSafe ``system_one``."""
    state: Dict[str, Any] = {
        "task": "mobile_robot_local_navigation_block",
        "heuristic_action": ctx.heuristic_action,
        "nose_clear": ctx.nose_clear,
        "blocked_for_s": round(float(ctx.blocked_for_s), 3),
        "wait_before_replan_s": round(float(ctx.wait_before_replan_s), 3),
        "replan_cooldown_ready": bool(ctx.replan_cooldown_ready),
        "path_ahead_cost": int(ctx.path_ahead_cost),
        "obstacle_state": ctx.obstacle_state,
        "forward_clearance_m": round(float(ctx.forward_clearance_m), 3),
        "failed_replan_while_blocked": int(ctx.failed_replan_while_blocked),
        "obstacle_motion": dict(motion),
        "actions": {
            ACTION_WAIT: (
                "Stop briefly — blocker may be a person/mover that will clear"
            ),
            ACTION_KEEP_DWA: (
                "Inch/peel forward with the local planner — gap may be passable; "
                "avoid a long global detour"
            ),
            ACTION_REPLAN: (
                "Commit to a new global path — static block or sustained failure; "
                "detour may be longer but progress resumes"
            ),
            ACTION_BACKUP: (
                "Reverse a short distance to free the footprint, then replan — "
                "use when nose is jammed / spinning in place and rear is clear"
            ),
        },
        "efficiency_hint": (
            "Prefer keep_dwa when the nose is clear or the gap may fit. "
            "Prefer wait when obstacle_motion.likely_mover is true. "
            "Prefer backup when jammed nose-on into a static block, rear is clear, "
            "and backup_feasible is true — then replan from the new pose. "
            "Prefer replan when the block looks static and grace/cooldown elapsed, "
            "or failed_replan_while_blocked is high — but avoid huge detours if "
            "inching could clear a temporary pinch. Do not choose backup when "
            "backup_feasible is false."
        ),
        "backup": {
            "feasible": bool(ctx.backup_feasible),
            "attempts_remaining": int(ctx.backup_attempts_remaining),
            "cooldown_ready": bool(ctx.backup_cooldown_ready),
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
                "Given this robot's local navigation block, choose the single best "
                "next action for efficiency and safety. Use obstacle_motion to judge "
                "movers vs fixed obstacles. Prefer keep_dwa (inch/peel) over a long "
                "replan when the nose is clear or a gap may fit. Prefer wait for "
                "likely people/movers. Prefer backup when jammed into a static nose "
                "block with backup.feasible true and rear clear. Prefer replan for "
                "sustained static blocks when peeling/backup will not help. Never "
                "choose backup when backup.feasible is false."
            ),
            criteria={
                ACTION_WAIT: "Stop and wait for a dynamic blocker to clear",
                ACTION_KEEP_DWA: "Keep short path; peel/inch with local planner",
                ACTION_REPLAN: "Escalate to a new global path (may be longer)",
                ACTION_BACKUP: "Reverse briefly (rear clear) then replan",
            },
        ),
        "temporary_mover": Noul(
            instructions=(
                "Is the blocker likely a temporary mover (person, cart) rather than "
                "a fixed obstacle? Use obstacle_motion.motion_score, range_delta_m, "
                "and bearing_span_rad."
            )
        ),
        "gap_worth_trying": Noul(
            instructions=(
                "Is it worth inching/peeling forward to test whether the robot can "
                "fit, instead of immediately taking a much longer global detour?"
            )
        ),
        "should_backup": Noul(
            instructions=(
                "Should the robot reverse a short distance to unstick before "
                "replanning? Only yes if backup.feasible is true, the nose is "
                "jammed on a likely-static block, and waiting/peeling looks futile."
            )
        ),
        "replan_urgency": Score(
            instructions="How urgently should we replan globally?",
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
    temporary_mover: Optional[float],
    gap_worth_trying: Optional[float],
    should_backup: Optional[float],
    heuristic_action: str,
    backup_feasible: bool = False,
) -> str:
    """Compose Choice + Nouls into a supervisor action (code remains in control)."""
    action = str(choice or "").strip().lower()
    if action not in ACTIONS:
        action = heuristic_action
    # Soft overrides when Choice is weakly aligned with physics cues.
    if temporary_mover is not None and temporary_mover >= 0.7 and action == ACTION_REPLAN:
        return ACTION_WAIT
    if (
        should_backup is not None
        and should_backup >= 0.7
        and backup_feasible
        and action in (ACTION_REPLAN, ACTION_KEEP_DWA, ACTION_WAIT)
    ):
        return ACTION_BACKUP
    if action == ACTION_BACKUP and not backup_feasible:
        # Never reverse into unknown rear space.
        return (
            heuristic_action
            if heuristic_action in ACTIONS and heuristic_action != ACTION_BACKUP
            else ACTION_WAIT
        )
    if (
        gap_worth_trying is not None
        and gap_worth_trying >= 0.7
        and action == ACTION_REPLAN
        and heuristic_action == ACTION_KEEP_DWA
    ):
        return ACTION_KEEP_DWA
    return action if action in ACTIONS else heuristic_action


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

    @property
    def last_decision(self) -> Optional[PolicyDecision]:
        return self._last_decision

    def decide(self, ctx: LocalBlockContext) -> PolicyDecision:
        """Return the action to apply; always logs both sides in shadow/jev."""
        motion = obstacle_motion_features(list(self._history))
        features = {
            **motion,
            "nose_clear": ctx.nose_clear,
            "blocked_for_s": round(float(ctx.blocked_for_s), 3),
            "path_ahead_cost": int(ctx.path_ahead_cost),
            "failed_replan_while_blocked": int(ctx.failed_replan_while_blocked),
            "forward_clearance_m": round(float(ctx.forward_clearance_m), 3),
            "replan_cooldown_ready": bool(ctx.replan_cooldown_ready),
        }
        if ctx.remaining_path_m is not None:
            features["remaining_path_m"] = round(float(ctx.remaining_path_m), 2)

        if self.mode == NAV_POLICY_HEURISTIC:
            decision = PolicyDecision(
                applied_action=ctx.heuristic_action,
                heuristic_action=ctx.heuristic_action,
                mode=self.mode,
                features=features,
            )
            self._last_decision = decision
            return decision

        now = time.monotonic()
        if self.min_period_s > 0 and (now - self._last_query_at) < self.min_period_s:
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
            self._emit(f"jev_policy {decision.to_dict()}")
            return decision

        state = build_policy_state(ctx, motion)
        t0 = time.monotonic()
        try:
            result = self._query(state)
            latency = time.monotonic() - t0
            self._last_query_at = time.monotonic()
            answers = _answers_to_dict(result)
            choice_ans = answers.get("action") or {}
            jev_choice = choice_ans.get("choice")
            confidence = choice_ans.get("confidence")
            mover = (answers.get("temporary_mover") or {}).get("noul")
            gap = (answers.get("gap_worth_trying") or {}).get("noul")
            should_backup = (answers.get("should_backup") or {}).get("noul")
            mapped = map_jev_to_action(
                choice=str(jev_choice) if jev_choice is not None else None,
                temporary_mover=float(mover) if mover is not None else None,
                gap_worth_trying=float(gap) if gap is not None else None,
                should_backup=float(should_backup)
                if should_backup is not None
                else None,
                heuristic_action=ctx.heuristic_action,
                backup_feasible=bool(ctx.backup_feasible),
            )
            fallback = ""
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

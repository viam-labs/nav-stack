"""BuiltinNavigator: navigate / compute_path / cancel / status (no Nav2)."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from ..config import NavConfig
from ..geom import conversions as conv
from .jev_policy import JevNavPolicy, normalize_nav_policy
from .runtime_kwargs import builtin_nav_runtime_kwargs
from .supervisor import NavSupervisor
from .types import Pose2D
from .world_io import WorldIO


class BuiltinNavigator:
    """Builtin path follower: navigate / compute_path / cancel / nav_status."""

    def __init__(
        self,
        world: WorldIO,
        nav_cfg: Optional[NavConfig] = None,
        *,
        logger=None,
        **overrides,
    ):
        self._world = world
        self._logger = logger
        self._kwargs = builtin_nav_runtime_kwargs(nav_cfg, **overrides)
        self._algorithm = self._kwargs["algorithm"]
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._supervisor: Optional[NavSupervisor] = None
        self._last_preview: Optional[Dict] = None
        self._last_status: Dict = {
            "state": "idle",
            "active": False,
            "motion": "builtin",
            "goal": None,
            "pose": None,
        }
        _world_log = getattr(world, "log", None)
        self._jev_policy = JevNavPolicy(
            mode=normalize_nav_policy(self._kwargs.get("nav_policy", "heuristic")),
            min_confidence=float(self._kwargs.get("jev_min_confidence", 0.7)),
            timeout_s=float(self._kwargs.get("jev_timeout_s", 1.25)),
            min_period_s=float(self._kwargs.get("jev_min_period_s", 1.0)),
            history_s=float(self._kwargs.get("jev_history_s", 3.0)),
            model=str(self._kwargs.get("jev_model") or "jev-latest"),
            api_key=(
                str(self._kwargs["jev_api_key"])
                if self._kwargs.get("jev_api_key")
                else None
            ),
            logger=_world_log if callable(_world_log) else None,
        )
        self._kwargs["jev_policy"] = self._jev_policy

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger(msg)
                return
            except Exception:  # noqa: BLE001
                pass
        # Fallback: no-op when no logger is wired.

    def _new_supervisor(self) -> NavSupervisor:
        return NavSupervisor(self._world, **self._kwargs)

    def jev_decision_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._jev_policy.decision_log()

    def clear_jev_decision_log(self) -> None:
        with self._lock:
            self._jev_policy.clear_decision_log()

    def jev_policy_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self._jev_policy.mode,
                "min_confidence": self._jev_policy.min_confidence,
                "run_id": self._jev_policy._run_id,
                "entries": len(self._jev_policy.decision_log()),
            }

    def navigate(self, x: float, y: float, theta: float) -> None:
        """Start following a goal in a background thread (non-blocking)."""
        goal = Pose2D(float(x), float(y), float(theta))
        with self._lock:
            if self._supervisor is not None:
                self._supervisor.request_cancel()
            if self._worker is not None and self._worker.is_alive():
                # Best-effort join briefly so we don't stack workers.
                self._worker.join(timeout=0.5)
            supervisor = self._new_supervisor()
            self._supervisor = supervisor
            self._last_status = {
                "state": "active",
                "active": True,
                "motion": "builtin",
                "goal": {"x": goal.x, "y": goal.y, "theta": goal.theta},
                "pose": None,
            }

            def _run() -> None:
                try:
                    supervisor.run_goal(goal)
                finally:
                    st = supervisor.status()
                    with self._lock:
                        self._last_status = st.to_dict()
                        if self._supervisor is supervisor:
                            self._supervisor = None

            self._worker = threading.Thread(
                target=_run, name="builtin-nav", daemon=True
            )
            self._worker.start()
        self._log(f"builtin navigate to ({x:.3f}, {y:.3f}, {theta:.3f})")

    def compute_path(
        self,
        x: float,
        y: float,
        theta: float = 0.0,
        *,
        planner_id: str = "LazyThetaStar",
        start: Optional[conv.Pose2D] = None,
        timeout_s: float = 20.0,
        max_points: int = 400,
    ) -> Dict:
        del timeout_s  # planning is in-process and fast
        from .planner import planner_id_for

        supervisor = self._new_supervisor()
        goal = Pose2D(float(x), float(y), float(theta))
        result = supervisor.plan(goal, start=start)
        preview = result.to_preview_dict(
            goal=(x, y, theta),
            start=start,
            planner_id=planner_id_for(planner_id or self._algorithm),
            max_points=max_points,
        )
        self._last_preview = preview
        try:
            if result.costmap_viz is not None:
                self._world.set_viz_costmap(result.costmap_viz)
            if preview.get("feasible"):
                self._world.set_viz_plan(
                    tuple((p["x"], p["y"]) for p in preview["path"]),
                    (float(x), float(y), float(theta)),
                )
        except Exception:  # noqa: BLE001
            pass
        return preview

    def last_preview_plan(self) -> Optional[Dict]:
        return dict(self._last_preview) if self._last_preview else None

    def cancel(self) -> None:
        with self._lock:
            if self._supervisor is not None:
                self._supervisor.request_cancel()
            try:
                self._world.stop()
            except Exception:  # noqa: BLE001
                pass
            self._last_status = {
                **self._last_status,
                "state": "canceled",
                "active": False,
                "motion": "builtin",
            }

    def nav_status(self) -> Dict:
        with self._lock:
            if self._supervisor is not None:
                return self._supervisor.status().to_dict()
            return dict(self._last_status)

    def control_stats(self) -> Optional[Dict]:
        with self._lock:
            if self._supervisor is None:
                return None
            return self._supervisor.control_stats()

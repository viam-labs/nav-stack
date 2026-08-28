"""BuiltinNavigator: navigate / compute_path / cancel / status (no Nav2)."""
from __future__ import annotations

import threading
from typing import Dict, Optional

from ..ros import conversions as conv
from .supervisor import NavSupervisor
from .types import Pose2D
from .world_io import WorldIO


class BuiltinNavigator:
    """Drop-in backend for RosManager.navigate / compute_path / cancel / nav_status."""

    def __init__(
        self,
        world: WorldIO,
        *,
        inflation_radius_m: float = 0.55,
        robot_radius_m: float = 0.22,
        cost_scaling_factor: float = 4.0,
        algorithm: str = "lazy_theta_star",
        replan_period_s: float = 1.0,
        lookahead_m: float = 0.6,
        min_lookahead_m: float = 0.25,
        max_lookahead_m: float = 0.7,
        approach_dist_m: float = 0.35,
        xy_tolerance_m: float = 0.25,
        yaw_tolerance_rad: float = 0.35,
        max_vel_x: float = 0.4,
        max_vel_theta: float = 1.0,
        min_cmd_vel_x: float = 0.0,
        min_cmd_vel_theta: float = 0.0,
        timeout_s: float = 300.0,
        avoid_obstacles: bool = True,
        stop_distance_m: float = 0.4,
        slow_distance_m: float = 1.0,
        scan_max_age_s: float = 2.0,
        smooth_path: bool = True,
        smooth_sample_spacing_m: float = 0.10,
        local_costmap_enabled: bool = True,
        local_costmap_width_m: float = 4.0,
        local_costmap_height_m: float = 4.0,
        local_costmap_resolution: float = 0.05,
        local_inflation_radius_m: float = 0.4,
        local_planner_enabled: bool = True,
        local_planner_sim_time_s: float = 1.2,
        local_planner_activate_cost: int = 200,
        logger=None,
    ):
        self._world = world
        self._logger = logger
        self._algorithm = algorithm
        self._kwargs = dict(
            inflation_radius_m=inflation_radius_m,
            robot_radius_m=robot_radius_m,
            cost_scaling_factor=cost_scaling_factor,
            algorithm=algorithm,
            replan_period_s=replan_period_s,
            lookahead_m=lookahead_m,
            xy_tolerance_m=xy_tolerance_m,
            yaw_tolerance_rad=yaw_tolerance_rad,
            max_vel_x=max_vel_x,
            max_vel_theta=max_vel_theta,
            min_cmd_vel_x=min_cmd_vel_x,
            min_cmd_vel_theta=min_cmd_vel_theta,
            timeout_s=timeout_s,
            avoid_obstacles=avoid_obstacles,
            stop_distance_m=stop_distance_m,
            slow_distance_m=slow_distance_m,
            scan_max_age_s=scan_max_age_s,
            smooth_path=smooth_path,
            smooth_sample_spacing_m=smooth_sample_spacing_m,
            local_costmap_enabled=local_costmap_enabled,
            local_costmap_width_m=local_costmap_width_m,
            local_costmap_height_m=local_costmap_height_m,
            local_costmap_resolution=local_costmap_resolution,
            local_inflation_radius_m=local_inflation_radius_m,
            local_planner_enabled=local_planner_enabled,
            local_planner_sim_time_s=local_planner_sim_time_s,
            local_planner_activate_cost=local_planner_activate_cost,
        )
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

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger(msg)
                return
            except Exception:  # noqa: BLE001
                pass
        # Fallback: no-op (RosManager wraps its own logger).

    def _new_supervisor(self) -> NavSupervisor:
        return NavSupervisor(self._world, **self._kwargs)

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

"""Goal lifecycle: plan → follow → replan → succeed / fail / cancel."""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

from ..config import NavConfig
from .runtime_kwargs import builtin_nav_runtime_kwargs
from ..nav.simple_motion import (
    DriveCommand,
    ObstacleConfig,
    SimpleMotionConfig,
    cone_min_range,
    distance_m,
    rear_clearance_m,
)
from ..geom import conversions as conv
from .controller import (
    FollowerConfig,
    compute_path_command,
    limit_twist_rate,
    update_speed_estimate,
)
from .local_costmap import (
    LocalCostmap,
    LocalCostmapConfig,
    footprint_max_cost,
    reverse_backup_feasible,
    reverse_path_clear,
)
from .local_planner import LocalPlannerConfig
from .planner import (
    path_blocked,
    path_blocked_local,
    path_blocked_on_costmap,
    paths_meaningfully_differ,
    plan_path,
    connect_plan_start,
)
from .smoother import smooth_path, smooth_plan_path
from .types import NavStatus, Path2D, PlanResult, Pose2D
from .world_io import WorldIO


class NavSupervisor:
    """Blocking control loop intended to run on a background worker thread."""

    def __init__(
        self,
        world: WorldIO,
        nav_cfg: Optional[NavConfig] = None,
        **overrides: Any,
    ):
        kw = builtin_nav_runtime_kwargs(nav_cfg, **overrides)
        inflation_radius_m = kw["inflation_radius_m"]
        robot_radius_m = kw["robot_radius_m"]
        spin_radius_m = kw["spin_radius_m"]
        nose_offset_m = kw["nose_offset_m"]
        wheel_half_track_m = kw["wheel_half_track_m"]
        cost_scaling_factor = kw["cost_scaling_factor"]
        clearance_preference_m = kw["clearance_preference_m"]
        algorithm = kw["algorithm"]
        replan_period_s = kw["replan_period_s"]
        lookahead_m = kw["lookahead_m"]
        min_lookahead_m = kw["min_lookahead_m"]
        max_lookahead_m = kw["max_lookahead_m"]
        approach_dist_m = kw["approach_dist_m"]
        xy_tolerance_m = kw["xy_tolerance_m"]
        yaw_tolerance_rad = kw["yaw_tolerance_rad"]
        max_vel_x = kw["max_vel_x"]
        max_vel_theta = kw["max_vel_theta"]
        min_cmd_vel_x = kw["min_cmd_vel_x"]
        min_cmd_vel_theta = kw["min_cmd_vel_theta"]
        timeout_s = kw["timeout_s"]
        poll_interval_s = kw["poll_interval_s"]
        avoid_obstacles = kw["avoid_obstacles"]
        stop_distance_m = kw["stop_distance_m"]
        slow_distance_m = kw["slow_distance_m"]
        scan_max_age_s = kw["scan_max_age_s"]
        smooth_path = kw["smooth_path"]
        smooth_sample_spacing_m = kw["smooth_sample_spacing_m"]
        local_costmap_enabled = kw["local_costmap_enabled"]
        local_costmap_width_m = kw["local_costmap_width_m"]
        local_costmap_height_m = kw["local_costmap_height_m"]
        local_costmap_resolution = kw["local_costmap_resolution"]
        local_inflation_radius_m = kw["local_inflation_radius_m"]
        local_costmap_rate_hz = kw["local_costmap_rate_hz"]
        local_planner_enabled = kw["local_planner_enabled"]
        local_planner_sim_time_s = kw["local_planner_sim_time_s"]
        local_planner_activate_cost = kw["local_planner_activate_cost"]
        local_planner_max_vel_x_mps = kw["local_planner_max_vel_x_mps"]
        local_planner_max_vel_x_reverse_m = kw["local_planner_max_vel_x_reverse_m"]
        backup_enabled = kw["backup_enabled"]
        backup_stuck_time_s = kw["backup_stuck_time_s"]
        backup_dist_m = kw["backup_dist_m"]
        backup_speed_mps = kw["backup_speed_mps"]
        backup_rear_clear_m = kw["backup_rear_clear_m"]
        backup_max_attempts = kw["backup_max_attempts"]
        backup_cooldown_s = kw["backup_cooldown_s"]
        recovery_wait_duration_s = kw["recovery_wait_duration_s"]
        replan_local_blocked_time_s = kw["replan_local_blocked_time_s"]
        replan_local_min_period_s = kw["replan_local_min_period_s"]
        drive_timeout_streak = kw["drive_timeout_streak"]
        yaw_align_timeout_s = kw["yaw_align_timeout_s"]
        max_goal_snap_m = kw["max_goal_snap_m"]
        max_linear_accel_mps2 = kw["max_linear_accel_mps2"]
        max_linear_decel_mps2 = kw["max_linear_decel_mps2"]
        max_angular_accel_rad_s2 = kw["max_angular_accel_rad_s2"]
        self._nav_loc_refine = bool(kw.get("nav_loc_refine_on_disagree", True))
        self._nav_loc_refine_margin_m = max(
            0.1, float(kw.get("nav_loc_refine_margin_m", 0.8))
        )
        self._nav_loc_refine_map_max_m = max(
            0.3, float(kw.get("nav_loc_refine_map_max_m", 2.5))
        )
        self._nav_loc_refine_lidar_min_m = max(
            0.2, float(kw.get("nav_loc_refine_lidar_min_m", 1.2))
        )
        self._nav_loc_refine_min_frac = min(
            1.0, max(0.05, float(kw.get("nav_loc_refine_min_frac", 0.30)))
        )
        self._nav_loc_refine_min_beams = max(
            1, int(kw.get("nav_loc_refine_min_beams", 6))
        )
        self._nav_loc_refine_max_tries = max(
            1, int(kw.get("nav_loc_refine_max_tries", 2))
        )
        self._nav_loc_refine_cooldown_s = max(
            0.0, float(kw.get("nav_loc_refine_cooldown_s", 12.0))
        )
        self._nav_loc_refine_period_s = max(
            0.0, float(kw.get("nav_loc_refine_period_s", 1.5))
        )
        self._nav_loc_refine_settle_s = max(
            0.0, float(kw.get("nav_loc_refine_settle_s", 0.35))
        )
        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = 0.0
        self._loc_refine_last_check = 0.0
        self._world = world
        self._inflation = inflation_radius_m
        # Driving clearance (half-width when a footprint is configured). Every
        # costmap / path / footprint check uses this.
        self._robot_radius = robot_radius_m
        # Rotation clearance (half-diagonal): what the body sweeps turning in
        # place. Larger than _robot_radius on a non-square robot, so it gates
        # spins without sealing gaps the robot can drive through.
        self._spin_radius = max(
            float(robot_radius_m),
            float(spin_radius_m) if spin_radius_m else float(robot_radius_m),
        )
        # Centre-to-bumper distance for the forward stop bubble.
        nose_offset = (
            float(nose_offset_m) if nose_offset_m else float(robot_radius_m)
        )
        half_track = (
            float(wheel_half_track_m)
            if wheel_half_track_m
            else 0.6 * float(robot_radius_m)
        )
        self._cost_scaling = cost_scaling_factor
        self._clearance_preference_m = max(0.0, float(clearance_preference_m))
        self._yaw_align_timeout_s = max(0.0, float(yaw_align_timeout_s))
        self._max_goal_snap_m = max(0.0, float(max_goal_snap_m))
        self._algorithm = algorithm
        self._replan_period = replan_period_s
        self._timeout_s = timeout_s
        self._scan_max_age = scan_max_age_s
        self._smooth_path = smooth_path
        self._smooth_spacing = smooth_sample_spacing_m
        self._local_costmap_enabled = local_costmap_enabled
        self._drive_timeout_streak = max(1, int(drive_timeout_streak))
        self._local_planner = LocalPlannerConfig(
            enabled=local_planner_enabled,
            sim_time_s=local_planner_sim_time_s,
            activate_cost_threshold=local_planner_activate_cost,
            max_detour_forward_mps=local_planner_max_vel_x_mps,
            max_vel_x_reverse_m=local_planner_max_vel_x_reverse_m,
        )
        self._backup_enabled = backup_enabled
        self._backup_stuck_time_s = backup_stuck_time_s
        self._backup_dist_m = backup_dist_m
        self._backup_speed_mps = backup_speed_mps
        self._backup_rear_clear_m = backup_rear_clear_m
        self._backup_max_attempts = max(0, int(backup_max_attempts))
        self._backup_cooldown_s = backup_cooldown_s
        self._recovery_wait_duration_s = max(0.0, float(recovery_wait_duration_s))
        self._replan_local_blocked_time_s = max(0.0, float(replan_local_blocked_time_s))
        self._replan_local_min_period_s = max(
            1.0, float(replan_local_min_period_s)
        )
        self._local_planner_activate_cost = local_planner_activate_cost
        self._local_costmap = (
            LocalCostmap(
                LocalCostmapConfig(
                    width_m=local_costmap_width_m,
                    height_m=local_costmap_height_m,
                    resolution=local_costmap_resolution,
                    inflation_radius_m=local_inflation_radius_m,
                    robot_radius_m=robot_radius_m,
                    cost_scaling_factor=cost_scaling_factor,
                )
            )
            if local_costmap_enabled
            else None
        )
        # Cached global costmap for local window. Full-map inflate is expensive;
        # rebuild off the control thread and only when the map generation changes.
        self._global_occ_cache = None
        self._global_costs_cache = None
        self._global_cache_at = 0.0
        self._global_cache_generation = None
        self._global_cache_lock = threading.Lock()
        self._global_refresh_inflight = False
        self._local_view_cache = None
        self._local_view_at = 0.0
        # Independent of control rate — lidar ~10 Hz; 5 Hz local is enough for DWA.
        rate = max(0.5, float(local_costmap_rate_hz))
        self._local_update_period_s = 1.0 / rate
        self._local_scan_max_age_s = min(float(scan_max_age_s), 0.5)
        self._global_cache_period_s = 1.0
        self._local_costmap_updates = 0
        self._global_costmap_rebuilds = 0
        self._cancel = threading.Event()
        self._status = NavStatus()
        self._status_lock = threading.Lock()
        self._last_cmd_vx = 0.0
        # Last twist actually handed to the base, for slew limiting.
        self._last_sent_cmd: Optional[DriveCommand] = None
        self._last_sent_at: Optional[float] = None
        self._last_replan_error = ""
        # Last replan diagnostics for get_status (trigger + attempt outcomes).
        self._last_replan_trigger = ""
        self._last_replan_info: dict = {}
        self._io_timeout_streak = 0
        # Control-loop soak metrics (wall-clock tick vs configured period).
        self._control_ticks = 0
        self._control_overruns = 0
        self._control_tick_last_s: Optional[float] = None
        self._control_tick_ema_s: Optional[float] = None
        self._control_period_s = max(1e-3, float(poll_interval_s))
        self._follower = FollowerConfig(
            lookahead_m=lookahead_m,
            min_lookahead_m=min_lookahead_m,
            max_lookahead_m=max_lookahead_m,
            approach_dist_m=approach_dist_m,
            waypoint_tolerance_m=max(0.1, xy_tolerance_m),
            # Keeps translating arcs above the base's inner-wheel "nearly 0 RPM"
            # rejection. From the footprint width when configured, else ≈0.6·r.
            wheel_half_track_m=max(0.08, half_track),
            max_linear_accel_mps2=max(0.05, float(max_linear_accel_mps2)),
            max_linear_decel_mps2=max(0.05, float(max_linear_decel_mps2)),
            max_angular_accel_rad_s2=max(0.1, float(max_angular_accel_rad_s2)),
            motion=SimpleMotionConfig(
                poll_interval_s=poll_interval_s,
                xy_tolerance_m=xy_tolerance_m,
                yaw_tolerance_rad=yaw_tolerance_rad,
                default_linear_mps=max_vel_x,
                max_linear_mps=max_vel_x,
                max_angular_rad_s=max_vel_theta,
                min_linear_mps=min_cmd_vel_x,
                min_angular_rad_s=min_cmd_vel_theta,
                timeout_s=timeout_s,
            ),
            obstacle=ObstacleConfig(
                enabled=avoid_obstacles,
                # Lidar stop must clear the bumper, which is a half-*length* out
                # from the centre — not a half-diagonal disc, which stopped a
                # 0.72 m robot 0.5 m short of everything.
                stop_distance_m=max(float(stop_distance_m), nose_offset + 0.05),
                slow_distance_m=max(
                    float(slow_distance_m),
                    max(float(stop_distance_m), nose_offset + 0.05) + 0.35,
                ),
                # Anything inside the body's swept corridor counts as "ahead" —
                # the ±35° cone alone let shoulder-side bins slide past.
                # Padding beyond the inscribed radius covers light shoulder
                # grazes (half-width + ~12 cm) without sealing every doorway.
                footprint_half_width_m=float(robot_radius_m) + 0.12,
                max_age_s=scan_max_age_s,
            )
            if avoid_obstacles
            else ObstacleConfig(enabled=False),
        )

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel

    def request_cancel(self) -> None:
        self._cancel.set()

    def status(self) -> NavStatus:
        with self._status_lock:
            s = self._status
            return NavStatus(
                state=s.state,
                active=s.active,
                goal=dict(s.goal) if s.goal else None,
                pose=dict(s.pose) if s.pose else None,
                error_msg=s.error_msg,
                path=list(s.path) if s.path is not None else None,
                length_m=s.length_m,
                motion=s.motion,
                progress=dict(s.progress) if s.progress is not None else None,
            )

    def control_stats(self) -> dict:
        """Control-loop timing for higher-rate soak (tick vs period)."""
        period = self._control_period_s
        with self._global_cache_lock:
            global_gen = self._global_cache_generation
        return {
            "period_s": round(period, 4),
            "target_hz": round(1.0 / period, 2),
            "ticks": self._control_ticks,
            "overruns": self._control_overruns,
            "tick_last_s": (
                None
                if self._control_tick_last_s is None
                else round(self._control_tick_last_s, 4)
            ),
            "tick_ema_s": (
                None
                if self._control_tick_ema_s is None
                else round(self._control_tick_ema_s, 4)
            ),
            "local_costmap_period_s": round(self._local_update_period_s, 4),
            "local_costmap_updates": self._local_costmap_updates,
            "global_costmap_rebuilds": self._global_costmap_rebuilds,
            "global_costmap_generation": global_gen,
        }

    def _note_control_tick(self, work_s: float) -> None:
        self._control_ticks += 1
        self._control_tick_last_s = work_s
        if self._control_tick_ema_s is None:
            self._control_tick_ema_s = work_s
        else:
            self._control_tick_ema_s = 0.8 * self._control_tick_ema_s + 0.2 * work_s
        if work_s > self._control_period_s:
            self._control_overruns += 1

    def _sleep_control_period(self, tick_started: float) -> None:
        """Sleep the remainder of the control period (not a full period after work)."""
        work_s = time.monotonic() - tick_started
        self._note_control_tick(work_s)
        remaining = self._control_period_s - work_s
        if remaining > 0.0:
            time.sleep(remaining)

    def _kick_global_costmap_refresh(self, *, allow_inline: bool = False) -> None:
        """Rebuild the full inflated map off the control thread when stale.

        The first call may run inline when there is no cache yet so the local
        window is not empty of static walls on the first tick.
        """
        if self._global_refresh_inflight:
            return
        now = time.monotonic()
        with self._global_cache_lock:
            have_cache = self._global_costs_cache is not None
        if have_cache and now - self._global_cache_at < self._global_cache_period_s:
            return

        def _job() -> None:
            try:
                try:
                    map_data = self._world.get_map()
                except TimeoutError:
                    return
                if map_data is None:
                    return
                generation = map_data.get("generation")
                with self._global_cache_lock:
                    if (
                        generation is not None
                        and generation == self._global_cache_generation
                        and self._global_costs_cache is not None
                    ):
                        self._global_cache_at = time.monotonic()
                        return
                from .costmap import build_costmap, occupancy_from_map_dict

                occ = occupancy_from_map_dict(map_data)
                costs = build_costmap(
                    occ,
                    inflation_radius_m=self._inflation,
                    robot_radius_m=self._robot_radius,
                    cost_scaling_factor=self._cost_scaling,
                    clearance_preference_m=self._clearance_preference_m,
                )
                with self._global_cache_lock:
                    self._global_occ_cache = occ
                    self._global_costs_cache = costs
                    self._global_cache_generation = generation
                    self._global_cache_at = time.monotonic()
                    self._global_costmap_rebuilds += 1
            except Exception:  # noqa: BLE001 - keep driving on last cache
                pass
            finally:
                self._global_refresh_inflight = False

        self._global_refresh_inflight = True
        if allow_inline and not have_cache:
            _job()
            return
        threading.Thread(
            target=_job, name="nav-global-costmap", daemon=True
        ).start()

    def _xy_at_nav_goal(self, pose: Pose2D, goal: Pose2D, path: Path2D) -> bool:
        """True when XY is close enough to the *requested* goal (not a far snap)."""
        xy_tol = self._follower.motion.xy_tolerance_m
        if distance_m(pose, goal) <= xy_tol:
            return True
        if not path.points:
            return False
        end = Pose2D(path.points[-1][0], path.points[-1][1], goal.theta)
        # Path may end on a small free-cell snap; accept that only when the snap
        # itself stayed near the requested goal.
        return (
            distance_m(end, goal) <= self._max_goal_snap_m
            and distance_m(pose, end) <= xy_tol
        )

    def _set_status(self, **kwargs) -> None:
        with self._status_lock:
            for k, v in kwargs.items():
                setattr(self._status, k, v)

    def plan(
        self,
        goal: Pose2D,
        start: Optional[Pose2D] = None,
        scan: Optional[conv.LaserScan2D] = None,
        *,
        blocked_path: Optional[Path2D] = None,
        blocked_path_pose: Optional[Pose2D] = None,
        local_view=None,
        paint_corridor: bool = True,
    ) -> PlanResult:
        pose = start if start is not None else self._world.get_pose()
        if pose is None:
            return PlanResult(
                feasible=False, error_code=5, error_msg="map pose unavailable"
            )
        map_data = self._world.get_map()
        if map_data is None:
            return PlanResult(
                feasible=False, error_code=6, error_msg="occupancy map unavailable"
            )
        result = plan_path(
            map_data,
            pose,
            goal,
            inflation_radius_m=self._inflation,
            robot_radius_m=self._robot_radius,
            cost_scaling_factor=self._cost_scaling,
            clearance_preference_m=self._clearance_preference_m,
            algorithm=self._algorithm,
            scan=scan,
            scan_pose=pose if scan is not None else None,
            blocked_path=blocked_path,
            blocked_path_pose=blocked_path_pose,
            local_view=local_view,
            paint_corridor=paint_corridor,
            dynamic_obstacle_radius_m=max(0.05, min(self._robot_radius, 0.12)),
            max_goal_snap_m=self._max_goal_snap_m,
        )
        if result.feasible:
            result = connect_plan_start(
                map_data,
                pose,
                result,
                inflation_radius_m=self._inflation,
                robot_radius_m=self._robot_radius,
                cost_scaling_factor=self._cost_scaling,
                clearance_preference_m=self._clearance_preference_m,
                algorithm=self._algorithm,
                xy_tolerance_m=self._follower.motion.xy_tolerance_m,
                scan=scan,
                local_view=local_view,
            )
        if result.feasible and self._smooth_path:
            if result.planning_costs is not None and result.planning_occ is not None:
                # Smooth on the costmap the planner actually used (static +
                # scan + local overlay). A static-only rebuild here used to
                # string-pull the detour straight back through the live
                # obstacle, so every replan was rejected as still-blocked.
                smoothed = smooth_path(
                    result.path,
                    result.planning_costs,
                    result.planning_occ,
                    enabled=True,
                    sample_spacing_m=self._smooth_spacing,
                )
            else:
                smoothed = smooth_plan_path(
                    result.path,
                    map_data,
                    inflation_radius_m=self._inflation,
                    robot_radius_m=self._robot_radius,
                    cost_scaling_factor=self._cost_scaling,
                    clearance_preference_m=self._clearance_preference_m,
                    enabled=True,
                    sample_spacing_m=self._smooth_spacing,
                )
            result.path = smoothed
        return result

    def _publish_plan_viz(self, result: PlanResult, goal: Pose2D, start: Optional[Pose2D]) -> None:
        preview = result.to_preview_dict(
            goal=(goal.x, goal.y, goal.theta), start=start
        )
        try:
            if result.costmap_viz is not None:
                self._world.set_viz_costmap(result.costmap_viz)
            if preview.get("feasible"):
                self._world.set_viz_plan(
                    tuple((p["x"], p["y"]) for p in preview["path"]),
                    (goal.x, goal.y, goal.theta),
                )
        except Exception:  # noqa: BLE001 - viz is best-effort
            pass
        return preview

    def _stop_before_replan(self, trigger: str = "") -> None:
        """Zero the base before a blocking replan on the control thread.

        Replan can take hundreds of ms (sometimes >1 s on a big map). Leaving
        the previous cmd_vel running for that whole window is what made the
        robot plow into a live obstacle and only *then* get a new path.
        """
        if trigger:
            self._last_replan_trigger = str(trigger)
        # Keep status consistent with the command actually sent while planning.
        # Previously progress kept advertising the prior 0.5 m/s command while
        # last_drive correctly showed zero, which hid stop/replan churn.
        with self._status_lock:
            progress = dict(self._status.progress or {})
            progress.update(
                obstacle="planning",
                local_planner=False,
                cmd_vx_mps=0.0,
                cmd_vtheta_rad_s=0.0,
                last_replan_trigger=self._last_replan_trigger,
            )
            self._status.progress = progress
        try:
            self._world.set_velocity(0.0, 0.0, 0.0)
        except Exception:  # noqa: BLE001 - never skip replan because stop failed
            try:
                self._world.stop()
            except Exception:  # noqa: BLE001
                pass
        self._note_base_stopped()

    def _note_base_stopped(self) -> None:
        """Record that the base is at rest so the next command ramps from zero."""
        self._last_sent_cmd = DriveCommand(0.0, 0.0, 0.0, False)
        self._last_sent_at = time.monotonic()
        self._last_cmd_vx = 0.0

    def _loc_refine_map(self):
        """Occupancy for scan-vs-map consistency (cached global grid when available)."""
        with self._global_cache_lock:
            occ = self._global_occ_cache
        if occ is not None:
            return occ
        try:
            map_data = self._world.get_map()
        except TimeoutError:
            return None
        from .loc_consistency import occupancy_for_consistency

        return occupancy_for_consistency(map_data)

    def _lidar_scan_for_loc_refine(self):
        try:
            return self._world.get_scan(
                self._scan_max_age, include_obstacles_only=False
            )
        except TimeoutError:
            return None

    def _publish_loc_refine_progress(
        self,
        pose: Pose2D,
        dist_goal: float,
        *,
        verdict,
        status: str,
    ) -> None:
        detail = verdict.to_dict() if verdict is not None else {}
        self._set_status(
            pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
            progress={
                "obstacle": "loc_refine",
                "local_planner": False,
                "forward_clearance_m": None,
                "cmd_vx_mps": 0.0,
                "cmd_vtheta_rad_s": 0.0,
                "bearing_error_rad": 0.0,
                "distance_remaining_m": dist_goal,
                "waypoint_index": 0,
                "localization_refine": {
                    "status": status,
                    "try": self._loc_refine_tries,
                    "max_tries": self._nav_loc_refine_max_tries,
                    **detail,
                },
            },
        )

    def _call_check_localization(self) -> Optional[dict]:
        fn = getattr(self._world, "check_localization", None)
        if not callable(fn):
            return None
        try:
            result = fn(
                allow_during_navigation=True,
                full_map_escalation="still_bad",
            )
        except TimeoutError:
            return {"status": "error", "reason": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc).strip() or type(exc).__name__}
        return result if isinstance(result, dict) else None

    def _measure_loc_disagreement(self, pose: Pose2D):
        from .loc_consistency import localization_looks_bad

        return localization_looks_bad(
            pose,
            self._lidar_scan_for_loc_refine(),
            self._loc_refine_map(),
            margin_m=self._nav_loc_refine_margin_m,
            map_max_m=self._nav_loc_refine_map_max_m,
            min_frac=self._nav_loc_refine_min_frac,
            min_beams=self._nav_loc_refine_min_beams,
        )

    def _maybe_pause_and_refine_localization(
        self, pose: Pose2D, now: float, dist_goal: float
    ) -> Optional[str]:
        """Stop + local refine when the scan is a poor fit for the published pose.

        Returns ``None`` (keep following), ``hold`` (stay stopped), ``resume``
        (replan after a correction), or ``fail`` (goal already marked failed).
        """
        if not self._nav_loc_refine:
            return None
        holding = self._loc_refine_tries > 0
        if now < self._loc_refine_cooldown_until:
            return "hold" if holding else None
        if now - self._loc_refine_last_check < self._nav_loc_refine_period_s:
            return "hold" if holding else None
        self._loc_refine_last_check = now

        verdict = self._measure_loc_disagreement(pose)
        if not verdict.disagree:
            if holding:
                self._loc_refine_tries = 0
                return "resume"
            return None

        self._world.stop()
        self._note_base_stopped()
        self._publish_loc_refine_progress(
            pose, dist_goal, verdict=verdict, status="pausing"
        )
        if self._nav_loc_refine_settle_s > 0.0:
            time.sleep(self._nav_loc_refine_settle_s)

        result = self._call_check_localization()
        if result is None:
            return "hold" if holding else None
        status = str(result.get("status") or "")
        if status == "awaiting_confirm":
            self._publish_loc_refine_progress(
                pose, dist_goal, verdict=verdict, status="awaiting_confirm"
            )
            if self._nav_loc_refine_settle_s > 0.0:
                time.sleep(max(0.4, self._nav_loc_refine_settle_s))
            else:
                time.sleep(0.05)
            result = self._call_check_localization() or result
            status = str(result.get("status") or "")
        if status in ("skipped", "unconfigured"):
            reason = str(result.get("reason") or "")
            if reason in ("spinning", "stale_scan"):
                return "hold"
            return "hold" if holding else None

        pose_after = self._world.get_pose() or pose
        after = self._measure_loc_disagreement(pose_after)
        if after.disagree:
            self._loc_refine_tries += 1
            self._loc_refine_cooldown_until = (
                time.monotonic() + self._nav_loc_refine_cooldown_s
            )
            if self._loc_refine_tries >= self._nav_loc_refine_max_tries:
                self._world.stop()
                self._set_status(
                    state="failed",
                    active=False,
                    error_msg="localization_lost",
                    pose={
                        "x": pose_after.x,
                        "y": pose_after.y,
                        "theta": pose_after.theta,
                    },
                    progress={
                        "obstacle": "loc_refine",
                        "localization_refine": {
                            "status": "failed",
                            "try": self._loc_refine_tries,
                            "max_tries": self._nav_loc_refine_max_tries,
                            **after.to_dict(),
                        },
                    },
                )
                return "fail"
            self._publish_loc_refine_progress(
                pose_after, dist_goal, verdict=after, status="retry"
            )
            return "hold"

        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = (
            time.monotonic() + self._nav_loc_refine_cooldown_s
        )
        self._publish_loc_refine_progress(
            pose_after, dist_goal, verdict=after, status="resumed"
        )
        return "resume"

    def _rate_limited(self, cmd: DriveCommand) -> DriveCommand:
        """Slew-limit ``cmd`` against the twist already on the base.

        ``dt`` comes from the real gap since the last command, not the nominal
        period: a slow tick (costmap rebuild, blocking replan) should be allowed
        a proportionally larger step rather than crawling back up to speed.
        """
        now = time.monotonic()
        if self._last_sent_at is None:
            dt = self._control_period_s
        else:
            dt = min(
                max(now - self._last_sent_at, self._control_period_s),
                4.0 * self._control_period_s,
            )
        limited = limit_twist_rate(
            cmd, self._last_sent_cmd, cfg=self._follower, dt_s=dt
        )
        self._last_sent_cmd = limited
        self._last_sent_at = now
        return limited

    def _forced_side_detour(
        self,
        goal: Pose2D,
        pose: Pose2D,
        path: Path2D,
        scan: Optional[conv.LaserScan2D],
        local_view,
    ) -> Optional[tuple[Path2D, PlanResult, str]]:
        """Plan pose → side via → goal when corridor seals keep returning the
        same route. Picks the shorter left/right peel that stays moderate.
        """
        from .controller import _path_length
        from .local_costmap import footprint_collides
        from .local_planner import path_block_distance_m, path_point_ahead
        from .planner import _merge_path_prefix

        if path.empty or len(path.points) < 2:
            return None
        block_dist = 0.9
        if local_view is not None:
            bd = path_block_distance_m(
                pose,
                path,
                local_view,
                threshold=self._local_planner.activate_cost_threshold,
                lookahead_m=self._local_planner.path_clearance_lookahead_m,
            )
            if bd is not None:
                block_dist = max(0.4, float(bd))
        bx, by = path_point_ahead(path, pose.x, pose.y, block_dist)
        bx2, by2 = path_point_ahead(path, pose.x, pose.y, block_dist + 0.25)
        yaw = math.atan2(by2 - by, bx2 - bx)
        nx, ny = -math.sin(yaw), math.cos(yaw)
        remaining = max(0.5, _path_length(path))
        best: Optional[tuple[float, Path2D, PlanResult, str]] = None
        for side in (0.45, -0.45, 0.65, -0.65, 0.35, -0.35):
            via = Pose2D(bx + side * nx, by + side * ny, yaw)
            if local_view is not None and footprint_collides(
                local_view,
                via.x,
                via.y,
                robot_radius_m=max(0.05, min(0.08, self._robot_radius)),
            ):
                continue
            to_via = self.plan(
                via,
                start=pose,
                scan=scan,
                blocked_path=path,
                blocked_path_pose=pose,
                local_view=local_view,
                paint_corridor=False,
            )
            if not to_via.feasible:
                continue
            to_goal = self.plan(
                goal,
                start=via,
                scan=scan,
                local_view=local_view,
                paint_corridor=False,
            )
            if not to_goal.feasible:
                continue
            merged = _merge_path_prefix(to_via.path, to_goal.path)
            if not paths_meaningfully_differ(path, merged, tol_m=0.12):
                continue
            new_len = _path_length(merged)
            # Mild peel only — reject room-scale loops.
            if new_len > remaining * 2.2:
                continue
            label = f"via{'L' if side > 0 else 'R'}{abs(side):.2f}"
            result = PlanResult(
                feasible=True,
                path=merged,
                planning_time_s=to_via.planning_time_s + to_goal.planning_time_s,
                costmap_viz=to_goal.costmap_viz or to_via.costmap_viz,
            )
            if best is None or new_len < best[0]:
                best = (new_len, merged, result, label)
        if best is None:
            return None
        return best[1], best[2], best[3]

    @staticmethod
    def _local_block_action(
        *,
        nose_clear: bool,
        blocked_for_s: float,
        wait_before_replan_s: float,
        replan_cooldown_ready: bool,
    ) -> str:
        """Choose wait / keep_dwa / replan while the local path cost is high.

        ``keep_dwa``: clear forward cone — stay on the short global path and
        peel with the local planner (fit doorway inflation pinch).
        ``wait``: blocked nose — freeze briefly for dynamic crossers.
        ``replan``: grace/cooldown elapsed — escalate to a new global path.
        """
        if blocked_for_s < wait_before_replan_s:
            return "wait" if not nose_clear else "keep_dwa"
        if replan_cooldown_ready:
            return "replan"
        # Still in replan cooldown: keep peeling if the nose is open, else hold.
        return "keep_dwa" if nose_clear else "wait"

    def _try_replan(
        self,
        goal: Pose2D,
        pose: Pose2D,
        path: Path2D,
        scan: Optional[conv.LaserScan2D],
        *,
        require_different: bool = True,
        failed_count: int = 0,
        local_view=None,
        trigger: str = "",
    ) -> Optional[Path2D]:
        """Replan around a live block: mild peel first, forced side via last.

        Policy:
        - Prefer scan+local (and optional corridor paint) that actually leaves
          the old route (tol 0.12 m — 0.25 m was rejecting useful peels as
          ``same route``).
        - Reject any candidate whose in-window local path cost is still at or
          above the activate threshold (peeling past one blob into another).
        - Cap length at ~1.8× remaining on the blocked-corridor attempt so we
          don't take room-scale loops when a milder peel already exists.
        - If every attempt is same-route/infeasible, force left/right vias
          around the first blocked path sample — that is the static-box case
          where Lazy Theta* stubbornly hugs the old corridor.
        """
        from .controller import _path_length
        from .local_planner import path_cost_in_local_window
        from .path_utils import closest_point_on_path

        old_len = _path_length(path)
        self._last_replan_trigger = str(trigger or "")
        attempts: list[tuple[str, bool]] = [("scan+local", False)]
        if failed_count >= 1:
            attempts.append(("blocked-corridor", True))
        reasons: list[str] = []
        _, _, _, along = closest_point_on_path(pose, path)
        remaining = max(0.5, old_len - along)
        best: Optional[tuple[float, Path2D, PlanResult, str]] = None
        differ_tol = 0.12 if local_view is not None else 0.25
        block_cost = int(self._local_planner_activate_cost)
        footprint_skip = max(0.05, float(self._robot_radius))

        def _still_local_blocked(candidate: Path2D) -> Optional[int]:
            if local_view is None:
                return None
            cost = int(
                path_cost_in_local_window(
                    pose,
                    candidate,
                    local_view,
                    start_offset_m=footprint_skip,
                )
            )
            if cost >= block_cost:
                return cost
            return None

        for label, paint in attempts:
            replanned = self.plan(
                goal,
                start=pose,
                scan=scan,
                blocked_path=path,
                blocked_path_pose=pose,
                local_view=local_view,
                paint_corridor=paint,
            )
            if not replanned.feasible:
                reasons.append(f"{label}: {replanned.error_msg or 'infeasible'}")
                continue
            if require_different and not paths_meaningfully_differ(
                path, replanned.path, tol_m=differ_tol
            ):
                reasons.append(
                    f"{label}: same route ({_path_length(replanned.path):.1f} m)"
                )
                continue
            stuck_cost = _still_local_blocked(replanned.path)
            if stuck_cost is not None:
                reasons.append(
                    f"{label}: still local-blocked (cost={stuck_cost})"
                )
                continue
            new_len = _path_length(replanned.path)
            if paint and new_len > remaining * 1.8 and best is not None:
                reasons.append(
                    f"{label}: too long ({new_len:.1f} m > {remaining * 1.8:.1f} m)"
                )
                continue
            if best is None or new_len < best[0]:
                best = (new_len, replanned.path, replanned, label)
            if not paint:
                break
        if best is None and local_view is not None:
            forced = self._forced_side_detour(goal, pose, path, scan, local_view)
            if forced is not None:
                new_path, result, label = forced
                stuck_cost = _still_local_blocked(new_path)
                if stuck_cost is not None:
                    reasons.append(
                        f"{label}: still local-blocked (cost={stuck_cost})"
                    )
                else:
                    new_len = _path_length(new_path)
                    self._last_replan_error = ""
                    self._last_replan_info = {
                        "trigger": self._last_replan_trigger,
                        "accepted": label,
                        "old_length_m": round(old_len, 3),
                        "new_length_m": round(new_len, 3),
                        "require_different": bool(require_different),
                        "attempts": list(reasons),
                    }
                    preview = self._publish_plan_viz(result, goal, start=pose)
                    self._set_status(path=preview["path"], length_m=preview["length_m"])
                    return new_path
            else:
                reasons.append("forced-via: none feasible")
        if best is None:
            self._last_replan_error = "; ".join(reasons)
            self._last_replan_info = {
                "trigger": self._last_replan_trigger,
                "accepted": None,
                "old_length_m": round(old_len, 3),
                "new_length_m": None,
                "require_different": bool(require_different),
                "attempts": list(reasons),
            }
            return None
        self._last_replan_error = ""
        new_len, new_path, result, label = best
        self._last_replan_info = {
            "trigger": self._last_replan_trigger,
            "accepted": label,
            "old_length_m": round(old_len, 3),
            "new_length_m": round(new_len, 3),
            "require_different": bool(require_different),
            "attempts": list(reasons),
        }
        preview = self._publish_plan_viz(result, goal, start=pose)
        self._set_status(path=preview["path"], length_m=preview["length_m"])
        return new_path

    def run_goal(self, goal: Pose2D) -> None:
        """Plan and follow until success, failure, or cancel. Blocking."""
        self._cancel.clear()
        self._last_replan_error = ""
        self._last_replan_trigger = ""
        self._last_replan_info = {}
        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = 0.0
        self._loc_refine_last_check = 0.0
        goal_dict = {"x": float(goal.x), "y": float(goal.y), "theta": float(goal.theta)}
        self._set_status(
            state="active",
            active=True,
            goal=goal_dict,
            error_msg="",
            path=None,
            length_m=0.0,
        )

        try:
            result = self.plan(goal)
            if not result.feasible:
                self._set_status(
                    state="failed",
                    active=False,
                    error_msg=result.error_msg or "no feasible path",
                )
                return

            path = result.path
            preview = self._publish_plan_viz(result, goal, start=None)
            self._set_status(path=preview["path"], length_m=preview["length_m"])

            deadline = time.monotonic() + self._timeout_s
            last_replan = time.monotonic()
            last_progress_pose: Optional[Pose2D] = None
            last_progress_at = time.monotonic()
            last_progress_dist = float("inf")
            last_progress_bearing = float("inf")
            spin_stuck_since: Optional[float] = None
            backup_active = False
            backup_start: Optional[Pose2D] = None
            backup_start_cost = 0
            backup_attempts = 0
            backup_cooldown_until = 0.0
            # Controller ``narrow_reverse`` is per-tick; bound it like backup so
            # we reverse ~backup_dist then replan instead of driving forever.
            narrow_rev_start: Optional[Pose2D] = None
            narrow_rev_cooldown_until = 0.0
            local_blocked_since: Optional[float] = None
            last_local_replan_at = 0.0
            failed_replan_while_blocked = 0
            failed_static_replan = 0
            local_planner_active = False
            reactive_avoid_since: Optional[float] = None
            last_obstacle_state = ""
            prev_local_cmd: Optional[DriveCommand] = None
            prev_cmd: Optional[DriveCommand] = None
            rotate_active = False
            vx_sign_history: list[tuple[float, int]] = []
            xy_ok_since: Optional[float] = None
            last_tick_pose: Optional[Pose2D] = None
            # Only validate the next few metres — full-path static checks on
            # long goals trip on far unknown/inflation and abort immediately.
            path_block_horizon_m = 5.0
            # Abort only when the route stays blocked *and* lidar is not clear.
            # Clearance + stall/timeout still end hopeless runs.
            static_replan_fail_limit = 5
            pose_jump_replan_m = 1.5
            was_loc_holding = False
            pending_loc_replan = False

            while time.monotonic() < deadline:
                tick_started = time.monotonic()
                if self._cancel.is_set():
                    self._world.stop()
                    self._set_status(state="canceled", active=False, error_msg="canceled")
                    return

                pose = self._world.get_pose()
                if pose is None:
                    self._world.stop()
                    self._set_status(
                        state="failed",
                        active=False,
                        error_msg="map pose unavailable",
                    )
                    return

                self._set_status(
                    pose={"x": pose.x, "y": pose.y, "theta": pose.theta}
                )

                # Large pose-jump awaiting confirm: stop until SLAM applies or
                # rejects. Driving on the old pose while turning is how we plow.
                loc_hold = None
                hold_fn = getattr(self._world, "get_localization_hold", None)
                if callable(hold_fn):
                    try:
                        loc_hold = hold_fn()
                    except Exception:  # noqa: BLE001
                        loc_hold = None
                holding_for_localize = isinstance(loc_hold, dict)
                entering_loc_hold = holding_for_localize and not was_loc_holding
                if was_loc_holding and not holding_for_localize:
                    pending_loc_replan = True
                was_loc_holding = holding_for_localize

                # Goal reached? Use the *requested* goal — path[-1] can be a
                # free-cell snap that used to let us "succeed" a metre away.
                dist_goal_chk = distance_m(pose, goal)
                xy_tol = self._follower.motion.xy_tolerance_m
                xy_ok = self._xy_at_nav_goal(pose, goal, path)
                yaw_ok = (
                    abs(conv.normalize_angle(pose.theta - goal.theta))
                    <= self._follower.motion.yaw_tolerance_rad
                )
                now = time.monotonic()
                if xy_ok and yaw_ok:
                    self._world.stop()
                    self._set_status(state="succeeded", active=False, error_msg="")
                    return
                # Start the yaw give-up clock once inside XY acceptance — not only
                # after the ~3 cm settle — so end-wiggle cannot run forever while
                # oscillating just outside settle.
                if xy_ok:
                    if xy_ok_since is None:
                        xy_ok_since = now
                    elif (
                        self._yaw_align_timeout_s > 0.0
                        and now - xy_ok_since >= self._yaw_align_timeout_s
                    ):
                        # Close enough in XY; final yaw will not lock cleanly.
                        self._world.stop()
                        self._set_status(
                            state="succeeded",
                            active=False,
                            error_msg="",
                        )
                        return
                else:
                    xy_ok_since = None

                if holding_for_localize:
                    # Stop once on entry — repeating SetVelocity(0) every control
                    # tick (esp. at 20 Hz) queues behind lidar/odom RPCs and
                    # surfaces as ``Viam IO timed out`` / stalled navigation.
                    if entering_loc_hold:
                        self._world.stop()
                        self._note_base_stopped()
                    last_progress_at = now
                    last_progress_pose = pose
                    last_progress_dist = dist_goal_chk
                    last_progress_bearing = float("inf")
                    spin_stuck_since = None
                    self._set_status(
                        pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
                        progress={
                            "obstacle": "loc_hold",
                            "local_planner": False,
                            "forward_clearance_m": None,
                            "cmd_vx_mps": 0.0,
                            "cmd_vtheta_rad_s": 0.0,
                            "bearing_error_rad": 0.0,
                            "distance_remaining_m": dist_goal_chk,
                            "waypoint_index": 0,
                            "localization_hold": {
                                "status": loc_hold.get("status"),
                                "confirm_count": loc_hold.get("confirm_count"),
                                "confirm_needed": loc_hold.get("confirm_needed"),
                                "shift_m": loc_hold.get("shift_m")
                                or loc_hold.get("jump_shift_m"),
                                "shift_deg": loc_hold.get("shift_deg")
                                or loc_hold.get("jump_shift_deg"),
                            },
                        },
                    )
                    self._sleep_control_period(tick_started)
                    continue

                now = time.monotonic()
                loc_outcome = self._maybe_pause_and_refine_localization(
                    pose, now, dist_goal_chk
                )
                if loc_outcome == "fail":
                    return
                if loc_outcome in ("hold", "resume"):
                    if loc_outcome == "resume":
                        pending_loc_replan = True
                    last_progress_at = now
                    last_progress_pose = pose
                    last_progress_dist = dist_goal_chk
                    last_progress_bearing = float("inf")
                    spin_stuck_since = None
                    self._sleep_control_period(tick_started)
                    continue

                need_scan = (
                    self._follower.obstacle is not None
                    and self._follower.obstacle.enabled
                ) or self._local_costmap is not None
                refresh_local = self._local_costmap is not None and (
                    now - self._local_view_at >= self._local_update_period_s
                    or self._local_view_cache is None
                )

                scan = None
                if need_scan:
                    try:
                        # One merge for reactive cone (+ reused on local update
                        # ticks). Depth in the cone is useful; DWA still works
                        # with the same scan (lidar dominates).
                        scan = self._world.get_scan(self._scan_max_age)
                    except TimeoutError:
                        scan = None

                local_view = self._local_view_cache
                if refresh_local:
                    self._kick_global_costmap_refresh(allow_inline=True)
                    # Missing scan must not wipe live marks — that made
                    # path_blocked_local flicker false under IO load while
                    # reactive avoid still saw the obstacle on the next tick.
                    if scan is None and self._local_view_cache is not None:
                        local_view = self._local_view_cache
                    else:
                        # Costmap must not ingest obstacles_only depth — that
                        # contradicts viam_io's contract and paints phantom
                        # inscribed cells (pose_cost / spin) while lidar is clear.
                        costmap_scan = None
                        try:
                            costmap_scan = self._world.get_scan(
                                self._scan_max_age, include_obstacles_only=False
                            )
                        except Exception:  # noqa: BLE001
                            costmap_scan = None
                        if costmap_scan is None:
                            # Do not fall back to fused/depth — keep last view.
                            local_view = self._local_view_cache
                        else:
                            if costmap_scan.capture_pose is None:
                                costmap_scan = conv.LaserScan2D(
                                    ranges=costmap_scan.ranges,
                                    angle_min=costmap_scan.angle_min,
                                    angle_increment=costmap_scan.angle_increment,
                                    range_min=costmap_scan.range_min,
                                    range_max=costmap_scan.range_max,
                                    sensor_pose=costmap_scan.sensor_pose,
                                    capture_pose=pose,
                                )
                            with self._global_cache_lock:
                                global_occ = self._global_occ_cache
                                global_costs = self._global_costs_cache
                            local_view = self._local_costmap.update(
                                pose,
                                costmap_scan,
                                global_occ=global_occ,
                                global_costs=global_costs,
                            )
                            self._local_view_cache = local_view
                            self._local_view_at = now
                            self._local_costmap_updates += 1
                            if self._local_costmap_enabled:
                                from .costmap import local_view_viz_dict

                                try:
                                    self._world.set_viz_local_costmap(
                                        local_view_viz_dict(local_view)
                                    )
                                except Exception:  # noqa: BLE001 - viz is best-effort
                                    pass

                path_ahead_cost = 0
                pose_cost = 0
                local_blocked = False
                from .costmap import INSCRIBED

                if local_view is not None:
                    from .local_planner import path_cost_ahead as _path_cost_ahead

                    path_ahead_cost = int(
                        _path_cost_ahead(
                            pose,
                            path,
                            local_view,
                            lookahead_m=self._local_planner.path_clearance_lookahead_m,
                        )
                    )
                    # Local costs are footprint-inflated: center cell >= inscribed
                    # means the body already overlaps an obstacle, even when the
                    # path centerline ahead is still free (off-path drift).
                    pose_cost = int(local_view.cost_at_world(pose.x, pose.y))
                    local_blocked = (
                        path_ahead_cost >= self._local_planner_activate_cost
                        or pose_cost >= INSCRIBED
                    )
                # Reactive avoid spinning with a clear-looking path still means
                # the robot cannot proceed — escalate to the blocked/replan path.
                # Exception: path centerline free + lidar nose clear — depth
                # phantoms in the fused corridor used to force avoid→replan
                # forever while path_cost=0 (live rc21: planning churn).
                if (
                    last_obstacle_state == "avoid"
                    and reactive_avoid_since is not None
                    and now - reactive_avoid_since >= 0.8
                    and (
                        path_ahead_cost >= self._local_planner_activate_cost
                        or pose_cost >= INSCRIBED
                    )
                ):
                    local_blocked = True
                    if local_blocked_since is None:
                        local_blocked_since = reactive_avoid_since
                # Also treat "already in lethal" from the follower as blocked so
                # we enter keep_dwa / replan instead of stalling on a clear path.
                if last_obstacle_state == "in_lethal":
                    local_blocked = True
                    if local_blocked_since is None:
                        local_blocked_since = now
                # Front-vs-side policy (not motion classification):
                # - Blocked nose: brief wait (people crossing), then replan.
                # - Clear nose + local path cost: inflation pinch / side hit —
                #   keep the short global path and let DWA peel first. Immediate
                #   replan here was sealing fit doorways (scan paint → 40 m+
                #   room loops). Escalate to replan only after the grace window.
                wait_before_replan_s = max(
                    self._recovery_wait_duration_s,
                    self._replan_local_blocked_time_s,
                )
                nose_clear = True
                obs_cfg = self._follower.obstacle
                lidar_only = None
                if obs_cfg is not None and obs_cfg.enabled:
                    # Wait / nose_clear must not trust depth phantoms. Fused
                    # scan can report fwd≈0 while lidar still sees free space.
                    # Use the front cone on lidar-only — corridor half-width
                    # would treat a shoulder pinch as a "person on the nose".
                    try:
                        lidar_only = self._world.get_scan(
                            self._scan_max_age, include_obstacles_only=False
                        )
                    except Exception:  # noqa: BLE001
                        lidar_only = None
                    nose_scan = lidar_only
                    if nose_scan is not None:
                        half = float(obs_cfg.front_cone_half_rad)
                        nose_range = cone_min_range(nose_scan, -half, half)
                        nose_clear = (
                            (not math.isfinite(nose_range))
                            or nose_range > obs_cfg.stop_distance_m
                        )
                    # If lidar-only is unavailable, do NOT fall back to fused
                    # for wait policy — that reintroduces depth phantoms.
                waiting_for_clear = False
                if local_blocked:
                    if local_blocked_since is None:
                        local_blocked_since = now
                    blocked_for = now - local_blocked_since
                    cooldown_ready = (
                        now - last_local_replan_at >= self._replan_local_min_period_s
                    )
                    action = self._local_block_action(
                        nose_clear=nose_clear,
                        blocked_for_s=blocked_for,
                        wait_before_replan_s=wait_before_replan_s,
                        replan_cooldown_ready=cooldown_ready,
                    )
                    # Contradiction: path centerline free (path_cost low) but we
                    # still "wait for nose" — freezes forever on phantom/side
                    # blocks while replan also fails. Prefer DWA peel instead.
                    if (
                        action == "wait"
                        and path_ahead_cost < self._local_planner_activate_cost
                    ):
                        action = "keep_dwa"
                    # Same trap for replan: pathc=0 + nose_clear + failed
                    # "cannot reach plan start" was a stop/replan death spiral
                    # (rc21 live). Keep peeling / reverse instead.
                    if (
                        action == "replan"
                        and path_ahead_cost < self._local_planner_activate_cost
                        and nose_clear
                    ):
                        action = "keep_dwa"
                    if action == "wait":
                        waiting_for_clear = True
                    elif action == "replan":
                        # Stop only long enough to plan. Paint + forced-via
                        # kick in so we do not keep accepting the same corridor.
                        _trig = (
                            f"local_blocked cost={path_ahead_cost} "
                            f"nose_clear={nose_clear}"
                        )
                        self._stop_before_replan(_trig)
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=max(1, failed_replan_while_blocked),
                            require_different=failed_replan_while_blocked < 3,
                            local_view=local_view,
                            trigger=_trig,
                        )
                        # Start cooldown when planning *finishes*. Planning can
                        # take >2 s; stamping its start with a 0.5 s cooldown
                        # made the next control tick replan again immediately.
                        replan_finished = time.monotonic()
                        last_local_replan_at = replan_finished
                        last_replan = replan_finished
                        if new_path is not None:
                            path = new_path
                            local_blocked_since = None
                            failed_replan_while_blocked = 0
                            backup_attempts = 0
                            vx_sign_history.clear()
                            spin_stuck_since = None
                            reactive_avoid_since = None
                            last_obstacle_state = ""
                            prev_local_cmd = None
                            local_planner_active = False
                            prev_cmd = None
                            rotate_active = False
                        else:
                            failed_replan_while_blocked += 1
                else:
                    local_blocked_since = None
                    failed_replan_while_blocked = 0

                # Keep DWA available when the path is blocked but the nose is
                # clear (or after a failed detour) — otherwise we only spin in
                # reactive avoid / sit in wait. force_local also bypasses the
                # ±60° bearing gate so large heading error cannot block DWA.
                allow_local_planner = (
                    self._local_costmap_enabled
                    and not waiting_for_clear
                    and (
                        not local_blocked
                        or failed_replan_while_blocked >= 1
                        or nose_clear
                    )
                )
                force_local = bool(local_blocked and allow_local_planner)
                # Reactive stop/slow uses fused scan so depth can catch low /
                # lidar-blind hits. Wait / nose_clear / costmap stay lidar-only
                # (above); replan guards refuse avoid→replan death spirals when
                # the lidar nose and path are already clear.
                cmd, progress = compute_path_command(
                    pose,
                    path,
                    cfg=self._follower,
                    scan=scan,
                    speed_mps=self._last_cmd_vx,
                    local_view=local_view,
                    local_planner=self._local_planner if allow_local_planner else None,
                    robot_radius_m=self._robot_radius,
                    spin_radius_m=self._spin_radius,
                    min_cmd_vel_x=self._follower.motion.min_linear_mps,
                    min_cmd_vel_theta=self._follower.motion.min_angular_rad_s,
                    local_planner_active=local_planner_active,
                    prev_local_cmd=prev_local_cmd,
                    rotate_active=rotate_active,
                    prev_cmd=prev_cmd,
                    force_local_planner=force_local,
                )
                rotate_active = bool(progress.get("rotate_to_heading"))
                local_planner_active = bool(progress.get("local_planner"))
                if local_planner_active:
                    prev_local_cmd = cmd
                else:
                    prev_local_cmd = None

                obs_state = str(progress.get("obstacle") or "")
                if obs_state == "avoid":
                    if reactive_avoid_since is None:
                        reactive_avoid_since = now
                else:
                    reactive_avoid_since = None
                last_obstacle_state = obs_state
                progress = {
                    **progress,
                    "local_blocked": bool(local_blocked),
                    "path_cost_ahead": int(path_ahead_cost),
                    "pose_cost": int(pose_cost),
                    "failed_replan_while_blocked": int(failed_replan_while_blocked),
                    "last_replan_error": self._last_replan_error,
                    "last_replan_trigger": self._last_replan_trigger,
                    "last_replan_info": dict(self._last_replan_info),
                    "nose_clear": bool(nose_clear),
                    "local_replan_cooldown_s": round(
                        max(
                            0.0,
                            self._replan_local_min_period_s
                            - (time.monotonic() - last_local_replan_at),
                        ),
                        2,
                    ),
                    "distance_remaining_m": distance_m(pose, goal),
                }

                # Bound per-tick ``narrow_reverse``: reverse ~backup_dist, then
                # replan. Without this the controller reverses forever while the
                # spin disc stays occupied (rc16 nearly backed into a wall).
                obs_state = str(progress.get("obstacle") or "")
                if now < narrow_rev_cooldown_until and obs_state == "narrow_reverse":
                    cmd = DriveCommand(0.0, 0.0, 0.0, False)
                    progress = {
                        **progress,
                        "obstacle": "narrow_reverse_hold",
                        "local_planner": False,
                        "cmd_vx_mps": 0.0,
                        "cmd_vtheta_rad_s": 0.0,
                    }
                    narrow_rev_start = None
                elif obs_state == "narrow_reverse" and cmd.vx < -1e-6:
                    if narrow_rev_start is None:
                        narrow_rev_start = pose
                    backed_m = distance_m(pose, narrow_rev_start)
                    remain = max(0.12, float(self._backup_dist_m) - backed_m)
                    rear_cost_ok = True
                    if local_view is not None:
                        rear_cost_ok = reverse_path_clear(
                            local_view,
                            pose.x,
                            pose.y,
                            pose.theta,
                            robot_radius_m=self._robot_radius,
                            distance_m=remain,
                            ignore_ahead=True,
                        )
                    if backed_m >= self._backup_dist_m or not rear_cost_ok:
                        narrow_rev_start = None
                        narrow_rev_cooldown_until = now + max(
                            1.5, float(self._backup_cooldown_s)
                        )
                        self._stop_before_replan("narrow_reverse_done")
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=failed_replan_while_blocked,
                            local_view=local_view,
                            trigger="narrow_reverse_done",
                        )
                        replan_finished = time.monotonic()
                        last_local_replan_at = replan_finished
                        last_replan = replan_finished
                        if new_path is not None:
                            path = new_path
                            last_progress_at = now
                            local_blocked_since = None
                            failed_replan_while_blocked = 0
                            vx_sign_history.clear()
                        cmd = DriveCommand(0.0, 0.0, 0.0, False)
                        progress = {
                            **progress,
                            "obstacle": "narrow_reverse_done",
                            "local_planner": False,
                            "cmd_vx_mps": 0.0,
                            "cmd_vtheta_rad_s": 0.0,
                        }
                else:
                    narrow_rev_start = None

                if waiting_for_clear:
                    # Freeze forward motion for dynamic crossers, but keep a
                    # spin-blocked reverse crawl (otherwise avoid→spin_block→
                    # wait deadlocks with cmd stuck at zero). Still allow
                    # rotate-to-heading when that is the only command.
                    #
                    # If the follower did not already reverse (fused phantom
                    # nose held it), inject reverse when spin is blocked —
                    # wait must not contradict the spin-gate escape.
                    if (
                        abs(cmd.vx) < 1e-6
                        and abs(cmd.vtheta) < 1e-6
                        and (
                            bool(progress.get("spin_blocked"))
                            or progress.get("obstacle") == "in_lethal"
                        )
                        and local_view is not None
                    ):
                        from .controller import _try_narrow_reverse

                        # Prefer lidar-only for reverse gate — fused depth can
                        # invent a rear wall that keeps wait at cmd=0.
                        rev_scan = lidar_only if lidar_only is not None else scan
                        if rev_scan is not None:
                            rev = _try_narrow_reverse(
                                self._follower,
                                rev_scan,
                                self._robot_radius,
                                local_view=local_view,
                                current=pose,
                            )
                            if rev is not None:
                                cmd = rev
                    if cmd.vx < -1e-6 and abs(cmd.vtheta) < 1e-6:
                        progress = {
                            **progress,
                            "obstacle": "wait_reverse",
                            "local_planner": False,
                            "cmd_vx_mps": cmd.vx,
                            "cmd_vtheta_rad_s": 0.0,
                        }
                    else:
                        keep_yaw = (
                            abs(cmd.vx) < 1e-6
                            and abs(cmd.vy) < 1e-6
                            and abs(cmd.vtheta) > 1e-6
                        )
                        cmd = DriveCommand(
                            0.0, 0.0, cmd.vtheta if keep_yaw else 0.0, False
                        )
                        progress = {
                            **progress,
                            "obstacle": "wait",
                            "local_planner": False,
                            "cmd_vx_mps": 0.0,
                            "cmd_vtheta_rad_s": cmd.vtheta,
                        }
                    last_progress_at = now
                    last_progress_pose = pose
                    last_progress_dist = distance_m(pose, goal)
                    last_progress_bearing = float("inf")

                allow_backup = (
                    self._backup_enabled
                    and scan is not None
                    and local_view is not None
                    and progress.get("local_planner")
                    and abs(cmd.vx) < 0.05
                    and abs(cmd.vtheta) > 0.1
                    and backup_attempts < self._backup_max_attempts
                    and now >= backup_cooldown_until
                    and not waiting_for_clear
                    and (
                        not local_blocked
                        or failed_replan_while_blocked >= 2
                    )
                )

                if backup_active and backup_start is not None:
                    backed_m = distance_m(pose, backup_start)
                    pose_cost = (
                        footprint_max_cost(
                            local_view,
                            pose.x,
                            pose.y,
                            robot_radius_m=self._robot_radius,
                        )
                        if local_view is not None
                        else 0
                    )
                    rear = (
                        rear_clearance_m(scan)
                        if scan is not None
                        else math.inf
                    )
                    worse = (
                        local_view is not None
                        and pose_cost > backup_start_cost + 5
                    )
                    if (
                        backed_m >= self._backup_dist_m
                        or rear < self._backup_rear_clear_m * 0.85
                        or worse
                    ):
                        backup_active = False
                        backup_start = None
                        spin_stuck_since = None
                        backup_cooldown_until = now + self._backup_cooldown_s
                        self._stop_before_replan("backup_done")
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=failed_replan_while_blocked,
                            local_view=local_view,
                            trigger="backup_done",
                        )
                        replan_finished = time.monotonic()
                        last_local_replan_at = replan_finished
                        last_replan = replan_finished
                        if new_path is not None:
                            path = new_path
                            last_progress_at = now
                            local_blocked_since = None
                            failed_replan_while_blocked = 0
                            backup_attempts = 0
                            vx_sign_history.clear()
                        cmd = DriveCommand(0.0, 0.0, 0.0, False)
                    else:
                        cmd = DriveCommand(
                            -self._backup_speed_mps, 0.0, 0.0, False
                        )
                        progress = {
                            **progress,
                            "local_planner": False,
                            "obstacle": "backup",
                            "cmd_vx_mps": cmd.vx,
                            "cmd_vtheta_rad_s": cmd.vtheta,
                        }
                elif allow_backup:
                    if spin_stuck_since is None:
                        spin_stuck_since = now
                    elif now - spin_stuck_since >= self._backup_stuck_time_s:
                        rear_ok = rear_clearance_m(scan) >= self._backup_rear_clear_m
                        costmap_ok = reverse_backup_feasible(
                            local_view,
                            pose.x,
                            pose.y,
                            pose.theta,
                            robot_radius_m=self._robot_radius,
                            distance_m=self._backup_dist_m,
                        )
                        if rear_ok and costmap_ok:
                            backup_active = True
                            backup_start = pose
                            backup_start_cost = footprint_max_cost(
                                local_view,
                                pose.x,
                                pose.y,
                                robot_radius_m=self._robot_radius,
                            )
                            backup_attempts += 1
                            spin_stuck_since = None
                            cmd = DriveCommand(
                                -self._backup_speed_mps, 0.0, 0.0, False
                            )
                            progress = {
                                **progress,
                                "local_planner": False,
                                "obstacle": "backup",
                                "cmd_vx_mps": cmd.vx,
                                "cmd_vtheta_rad_s": cmd.vtheta,
                            }
                        else:
                            spin_stuck_since = None
                else:
                    spin_stuck_since = None

                if abs(cmd.vx) > 0.05:
                    sign = 1 if cmd.vx > 0 else -1
                    vx_sign_history.append((now, sign))
                    vx_sign_history = [
                        (t, s) for t, s in vx_sign_history if now - t <= 3.0
                    ]
                oscillating = False
                if len(vx_sign_history) >= 4:
                    flips = sum(
                        1
                        for i in range(1, len(vx_sign_history))
                        if vx_sign_history[i][1] != vx_sign_history[i - 1][1]
                    )
                    oscillating = flips >= 3

                replan_due = now - last_replan >= self._replan_period
                map_data = None
                static_blocked = False
                pose_jumped = pending_loc_replan
                if last_tick_pose is not None:
                    pose_jumped = pose_jumped or (
                        distance_m(pose, last_tick_pose) >= pose_jump_replan_m
                    )
                last_tick_pose = pose
                if replan_due or pose_jumped:
                    # Prefer the already-inflated global cache (background thread).
                    # Falling back to path_blocked() rebuilds the full map costmap
                    # (~0.4 s on a large grid) on the control thread.
                    static_blocked = False
                    with self._global_cache_lock:
                        cached_occ = self._global_occ_cache
                        cached_costs = self._global_costs_cache
                    if cached_occ is not None and cached_costs is not None:
                        static_blocked = path_blocked_on_costmap(
                            cached_occ,
                            cached_costs,
                            path,
                            robot_radius_m=self._robot_radius,
                            from_pose=pose,
                            ahead_m=path_block_horizon_m,
                        )
                    else:
                        map_data = self._world.get_map()
                        static_blocked = map_data is not None and path_blocked(
                            map_data,
                            path,
                            inflation_radius_m=self._inflation,
                            robot_radius_m=self._robot_radius,
                            from_pose=pose,
                            ahead_m=path_block_horizon_m,
                        )
                    # Large localization corrections invalidate the old polyline;
                    # force a replan even if the first few metres still look free.
                    if pose_jumped:
                        static_blocked = True

                backup_exhausted = (
                    backup_attempts >= self._backup_max_attempts and local_blocked
                )
                # Sign-flip rock (narrow crawl ↔ reverse) can happen with the
                # path centerline still "clear" on costs — still force a replan.
                spin_rock = bool(progress.get("spin_blocked")) and oscillating
                should_replan = (replan_due or pose_jumped) and (
                    static_blocked
                    or (oscillating and local_blocked)
                    or spin_rock
                    or backup_exhausted
                )
                # Periodic check said "still clear" — still advance the timer.
                # Otherwise replan_due stays true and the expensive static check
                # (or a full inflate fallback) runs on every subsequent tick.
                if replan_due and not should_replan:
                    last_replan = now
                if should_replan:
                    # On static/pose-jump recovery, accept any feasible plan —
                    # require_different would reject a valid near-identical route
                    # and count it as "replan failed".
                    _trig = (
                        f"periodic static_blocked={static_blocked} "
                        f"pose_jumped={pose_jumped} "
                        f"oscillating={oscillating}"
                    )
                    self._stop_before_replan(_trig)
                    new_path = self._try_replan(
                        goal,
                        pose,
                        path,
                        scan,
                        require_different=not static_blocked,
                        failed_count=failed_replan_while_blocked if local_blocked else 0,
                        local_view=local_view,
                        trigger=_trig,
                    )
                    replan_finished = time.monotonic()
                    last_local_replan_at = replan_finished
                    last_replan = replan_finished
                    if new_path is not None:
                        path = new_path
                        last_progress_at = now
                        local_blocked_since = None
                        failed_replan_while_blocked = 0
                        failed_static_replan = 0
                        backup_attempts = 0
                        vx_sign_history.clear()
                        spin_stuck_since = None
                        pending_loc_replan = False
                    elif static_blocked:
                        failed_static_replan += 1
                        clearance = progress.get("forward_clearance_m")
                        has_room = clearance is None or float(clearance) >= 0.35
                        obstacle = str(progress.get("obstacle") or "")
                        # "wait"/"slow" still mean lidar sees space; only hard
                        # stop / proximity hold / missing scan should force fail.
                        lidar_open = (
                            obstacle not in ("stop", "hold", "no_scan") and has_room
                        )
                        # Keep following while the robot can still see open space;
                        # a mid-route localization jump often fails a few replans
                        # before the map/pose settle.
                        if (
                            failed_static_replan >= static_replan_fail_limit
                            and not lidar_open
                        ):
                            self._world.stop()
                            self._set_status(
                                state="failed",
                                active=False,
                                error_msg="replan failed (path blocked)",
                            )
                            return
                    else:
                        last_replan = time.monotonic()

                # Final gate: slew-limit against the twist already on the base so
                # source handoffs (pursuit ↔ DWA ↔ avoid ↔ post-replan resume)
                # ramp instead of stepping. Status and stall detection below use
                # the limited command, which is what the base actually gets.
                cmd = self._rate_limited(cmd)
                progress = {
                    **progress,
                    "cmd_vx_mps": cmd.vx,
                    "cmd_vtheta_rad_s": cmd.vtheta,
                }

                self._set_status(
                    pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
                    progress={
                        k: progress[k]
                        for k in (
                            "obstacle",
                            "local_planner",
                            "local_blocked",
                            "path_cost_ahead",
                            "failed_replan_while_blocked",
                            "last_replan_error",
                            "nose_clear",
                            "spin_blocked",
                            "local_replan_cooldown_s",
                            "forward_clearance_m",
                            "cmd_vx_mps",
                            "cmd_vtheta_rad_s",
                            "bearing_error_rad",
                            "distance_remaining_m",
                            "waypoint_index",
                        )
                        if k in progress
                    },
                )
                if progress.get("obstacle") == "no_scan":
                    # Fail closed: stop forward; brief wait then continue.
                    self._world.set_velocity(0.0, 0.0, cmd.vtheta)
                    self._last_sent_cmd = DriveCommand(0.0, 0.0, cmd.vtheta, False)
                    self._last_sent_at = time.monotonic()
                    self._sleep_control_period(tick_started)
                    continue

                if cmd.done:
                    at_goal = self._xy_at_nav_goal(pose, goal, path) and (
                        abs(conv.normalize_angle(pose.theta - goal.theta))
                        <= self._follower.motion.yaw_tolerance_rad
                    )
                    if at_goal:
                        self._world.stop()
                        self._set_status(state="succeeded", active=False, error_msg="")
                        return

                # Stall detection.
                # Pure spin with no bearing improvement must not reset the timer
                # forever (stuck local-planner / RIP loops need to replan). But
                # intentional align spins that shrink |bearing| are real progress.
                # While the route is locally blocked, give recovery room —
                # stop/replan/forced-via look like "no progress" and were
                # aborting with cmd_vx still showing a corridor charge.
                dist_goal = distance_m(pose, goal)
                near_goal_stall = (
                    dist_goal <= self._follower.motion.xy_tolerance_m * 2.0
                    or self._xy_at_nav_goal(pose, goal, path)
                )
                # Near goal, tiny crawls are real progress — don't require 0.05 m/s.
                translating_floor = 0.03 if near_goal_stall else 0.05
                stall_scale = 2.0 if near_goal_stall else 1.0
                if local_blocked:
                    stall_scale = max(stall_scale, 4.0)
                stall_limit_s = self._follower.motion.stall_timeout_s * stall_scale
                bearing_err = abs(float(progress.get("bearing_error_rad", 0.0)))
                spinning = (
                    abs(float(cmd.vtheta)) >= 0.08
                    and abs(float(cmd.vx)) < translating_floor
                )

                def _mark_progress() -> None:
                    nonlocal last_progress_pose, last_progress_at
                    nonlocal last_progress_dist, last_progress_bearing
                    last_progress_pose = pose
                    last_progress_at = now
                    last_progress_dist = dist_goal
                    last_progress_bearing = bearing_err

                def _stall_replan_or_fail(error_msg: str) -> bool:
                    """Try replan (incl. forced via). True ⇒ caller should return."""
                    nonlocal path, last_progress_at, last_progress_dist
                    nonlocal last_progress_bearing, last_replan
                    nonlocal last_local_replan_at, local_blocked_since, backup_attempts
                    nonlocal failed_replan_while_blocked
                    # Path centerline free + lidar nose clear: another stop/replan
                    # cannot help (rc21: "cannot reach plan start" forever). Let
                    # the follower crawl/reverse on the existing path instead.
                    if (
                        path_ahead_cost < self._local_planner_activate_cost
                        and nose_clear
                    ):
                        last_progress_at = now
                        return False
                    _trig = f"stall:{error_msg}"
                    self._stop_before_replan(_trig)
                    new_path = self._try_replan(
                        goal,
                        pose,
                        path,
                        scan,
                        failed_count=max(2, failed_replan_while_blocked),
                        local_view=local_view,
                        trigger=_trig,
                    )
                    replan_finished = time.monotonic()
                    last_local_replan_at = replan_finished
                    last_replan = replan_finished
                    if new_path is not None:
                        path = new_path
                        last_progress_at = now
                        last_progress_dist = dist_goal
                        last_progress_bearing = bearing_err
                        local_blocked_since = None
                        failed_replan_while_blocked = 0
                        backup_attempts = 0
                        return False
                    # Still recovering: do not abort while local_blocked and
                    # we have not exhausted several forced-via attempts.
                    if local_blocked and failed_replan_while_blocked < 5:
                        failed_replan_while_blocked += 1
                        last_progress_at = now
                        return False
                    self._world.stop()
                    self._set_status(
                        state="failed",
                        active=False,
                        error_msg=error_msg,
                    )
                    return True

                if last_progress_pose is None:
                    _mark_progress()
                else:
                    moved = distance_m(pose, last_progress_pose)
                    closing = dist_goal < last_progress_dist - 0.01
                    translating = abs(float(cmd.vx)) >= translating_floor
                    bearing_improved = bearing_err < last_progress_bearing - math.radians(
                        3.0
                    )
                    turned = abs(
                        conv.normalize_angle(pose.theta - last_progress_pose.theta)
                    )
                    if closing or (
                        moved >= self._follower.motion.stall_progress_m
                        and (
                            translating
                            or moved >= self._follower.motion.stall_progress_m * 2
                        )
                    ):
                        _mark_progress()
                    elif translating:
                        if turned >= self._follower.motion.stall_progress_rad:
                            _mark_progress()
                        elif now - last_progress_at >= stall_limit_s:
                            if _stall_replan_or_fail("navigation stalled"):
                                return
                    elif spinning and (
                        turned >= self._follower.motion.stall_progress_rad
                        or bearing_improved
                    ):
                        _mark_progress()
                    elif now - last_progress_at >= stall_limit_s:
                        if _stall_replan_or_fail(
                            "navigation stalled (no forward progress)"
                        ):
                            return

                try:
                    self._world.set_velocity(cmd.vx, cmd.vy, cmd.vtheta)
                    self._io_timeout_streak = 0
                except TimeoutError:
                    # Transient event-loop starvation — don't abort the goal on
                    # a single missed cmd_vel (common when SLAM mapping + lidar
                    # gRPC share the module loop with Base.SetVelocity).
                    self._io_timeout_streak += 1
                    if self._io_timeout_streak >= self._drive_timeout_streak:
                        raise
                    self._sleep_control_period(tick_started)
                    continue
                except Exception as exc:  # noqa: BLE001
                    # Motor "nearly 0 RPM" / similar drive rejects: skip tick.
                    # Also treat gRPC GOAWAY / unavailable as transient while
                    # resources flap (lidar USB reconnect storms).
                    msg = str(exc).lower()
                    if "nearly 0" in msg or "rpm" in msg:
                        try:
                            self._world.stop()
                        except Exception:  # noqa: BLE001
                            pass
                        self._note_base_stopped()
                        self._sleep_control_period(tick_started)
                        continue
                    if (
                        "goaway" in msg
                        or "unavailable" in msg
                        or "connection" in msg
                    ):
                        self._io_timeout_streak += 1
                        if self._io_timeout_streak >= max(
                            8, self._drive_timeout_streak
                        ):
                            raise
                        self._sleep_control_period(tick_started)
                        continue
                    raise
                # Smoothed speed for the velocity-scaled lookahead (see
                # ``update_speed_estimate``): raw cmd feedback limit-cycles.
                self._last_cmd_vx = update_speed_estimate(self._last_cmd_vx, cmd.vx)
                prev_cmd = cmd
                self._sleep_control_period(tick_started)

            self._world.stop()
            self._set_status(
                state="failed", active=False, error_msg="navigation timed out"
            )
        except Exception as exc:  # noqa: BLE001 - surface as failed status
            try:
                self._world.stop()
            except Exception:  # noqa: BLE001
                pass
            msg = str(exc).strip() or type(exc).__name__
            self._set_status(state="failed", active=False, error_msg=msg)

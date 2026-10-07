"""Goal lifecycle: plan → follow → replan → succeed / fail / cancel."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
from collections import deque
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
    spin_clearance_m,
)
from ..geom import conversions as conv
from .controller import (
    FollowerConfig,
    compute_path_command,
    limit_twist_rate,
    update_speed_estimate,
)
from .above_cart import AboveCartMemory, above_cart_frames
from .depth_memory import DepthObstacleMemory, depth_frames
from .footprint_guard import FootprintGuard, GuardConfig, obstacle_points
from .local_costmap import (
    LocalCostmap,
    LocalCostmapConfig,
    footprint_max_cost,
    reverse_backup_feasible,
    reverse_path_clear,
    spin_disc_blocked,
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


_BLOCKED_REPLAN_FAIL_LIMIT = 8
# After a blocked-nose replan fails, turn this far (then replan) when the
# rear is not open. ~46° is enough to face a different corridor.
_NOSE_UNSTICK_YAW_RAD = 0.8
_NOSE_UNSTICK_YAW_RATE = 0.45

# Last ~60 s of guarded control ticks (20 Hz), kept across goals so a graze
# can be inspected after the operator cancels (``get_trace`` DoCommand).
_TRACE: "deque[dict]" = deque(maxlen=1200)


def recent_trace(seconds: float = 30.0) -> list:
    entries = list(_TRACE)
    if not entries:
        return []
    cutoff = entries[-1]["t"] - max(0.0, float(seconds))
    return [e for e in entries if e["t"] >= cutoff]


# Heading error must fall by this much to count the final spin as still working.
# Smaller than yaw tolerance so jitter does not refresh the give-up clock.
_YAW_ALIGN_PROGRESS_RAD = math.radians(6.0)


def yaw_align_give_up(
    now: float,
    yaw_err_abs: float,
    timeout_s: float,
    best_err: Optional[float],
    progress_at: Optional[float],
) -> tuple[bool, Optional[float], Optional[float]]:
    """Whether to accept XY and stop hunting final heading.

    The clock runs only while a spin is not closing the heading. A rotation
    that keeps reducing the error does not hit ``timeout_s``. Returns
    ``(give_up, best_err, progress_at)``.
    """
    if best_err is None or progress_at is None:
        return False, yaw_err_abs, now
    if yaw_err_abs < best_err - _YAW_ALIGN_PROGRESS_RAD:
        best_err = yaw_err_abs
        progress_at = now
    stalled = timeout_s > 0.0 and (now - progress_at) >= timeout_s
    return stalled, best_err, progress_at


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
            1.0, max(0.05, float(kw.get("nav_loc_refine_min_frac", 0.22)))
        )
        self._nav_loc_refine_min_beams = max(
            1, int(kw.get("nav_loc_refine_min_beams", 6))
        )
        self._nav_loc_refine_max_tries = max(
            1, int(kw.get("nav_loc_refine_max_tries", 2))
        )
        self._nav_loc_refine_cooldown_s = max(
            0.0, float(kw.get("nav_loc_refine_cooldown_s", 5.0))
        )
        self._nav_loc_refine_period_s = max(
            0.0, float(kw.get("nav_loc_refine_period_s", 0.75))
        )
        self._nav_loc_refine_check_every_m = max(
            0.0, float(kw.get("nav_loc_refine_check_every_m", 2.0))
        )
        self._nav_loc_refine_retry_travel_m = max(
            self._nav_loc_refine_check_every_m,
            float(kw.get("nav_loc_refine_retry_travel_m", 8.0)),
        )
        self._nav_loc_refine_apply_max_m = max(
            0.2, float(kw.get("nav_loc_refine_apply_max_m", 1.0))
        )
        self._nav_loc_refine_apply_max_deg = max(
            5.0, float(kw.get("nav_loc_refine_apply_max_deg", 30.0))
        )
        self._nav_loc_refine_apply_min_score = min(
            1.0, max(0.0, float(kw.get("nav_loc_refine_apply_min_score", 0.35)))
        )
        self._nav_loc_refine_settle_s = max(
            0.0, float(kw.get("nav_loc_refine_settle_s", 0.35))
        )
        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = 0.0
        self._loc_refine_last_check = 0.0
        self._loc_refine_last_pose: Optional[Pose2D] = None
        self._loc_refine_start_pose: Optional[Pose2D] = None
        self._loc_refine_need_travel = False
        self._loc_refine_travel_need_m = 0.0
        self._world = world
        self._inflation = inflation_radius_m
        # Driving clearance (half-width when a footprint is configured). Every
        # costmap / path / footprint check uses this.
        self._robot_radius = robot_radius_m
        # Physical body (no clearance). Defaults to the hard radius so callers
        # that only override robot_radius_m keep a single-ring costmap.
        self._body_radius = min(
            max(0.0, float(kw.get("body_radius_m", robot_radius_m))),
            float(robot_radius_m),
        )
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
        # Rectangular collision guard: exact body (half-length = nose offset,
        # half-width = body radius); clearance_m sets the keep-out padding.
        self._guard: Optional[FootprintGuard] = None
        if bool(kw.get("footprint_guard", True)):
            clearance = max(0.0, float(robot_radius_m) - float(self._body_radius))
            self._guard = FootprintGuard(
                GuardConfig(
                    length_m=2.0 * nose_offset,
                    width_m=2.0 * float(self._body_radius),
                    padding_m=min(max(clearance, 0.04), 0.15),
                )
            )
        self._depth_memory = DepthObstacleMemory(
            length_m=2.0 * nose_offset, width_m=2.0 * float(self._body_radius)
        )
        self._avoid_above = bool(kw.get("avoid_obstacles_above_cart", True))
        raw_height = kw.get("cart_height_m")
        try:
            self._cart_height_m = float(raw_height) if raw_height else 0.0
        except (TypeError, ValueError):
            self._cart_height_m = 0.0
        self._above_memory: Optional[AboveCartMemory] = None
        if self._avoid_above:
            # No cart height: the depth camera's own band, up to z_max.
            self._above_memory = AboveCartMemory(
                cart_height_m=self._cart_height_m if self._cart_height_m > 0.0 else None
            )
        setter = getattr(self._world, "set_above_cart", None)
        if callable(setter):
            setter(
                self._above_memory is not None,
                self._cart_height_m if self._cart_height_m > 0.0 else None,
            )
        self._cost_scaling = cost_scaling_factor
        self._clearance_preference_m = max(0.0, float(clearance_preference_m))
        self._yaw_align_timeout_s = max(0.0, float(yaw_align_timeout_s))
        self._max_goal_snap_m = max(0.0, float(max_goal_snap_m))
        self._algorithm = algorithm
        self._replan_period = replan_period_s
        self._timeout_s = timeout_s
        self._max_vel_x = max(0.05, float(max_vel_x))
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
            0.1, float(replan_local_min_period_s)
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
                    body_radius_m=self._body_radius,
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
        self._global_cache_zone_rev = -1
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
        # One local pose match per blocked episode when the planner cannot
        # leave the current cell. Cleared once the nose is free again.
        self._stuck_pose_refine_used = False
        self._stuck_pose_refine: dict = {}
        # After accepting a longer detour, ban the abandoned short corridor for
        # the rest of this goal (not a wall-clock timer — that just delayed the
        # short↔long flip). Lifted only if the detour itself dies and no other
        # route works.
        self._detour_ban_path: Optional[Path2D] = None
        self._detour_min_length_m = 0.0
        self._detour_short_ratio = 0.75
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
                zone_rev = self._zone_revision()
                with self._global_cache_lock:
                    if (
                        generation is not None
                        and generation == self._global_cache_generation
                        and zone_rev == self._global_cache_zone_rev
                        and self._global_costs_cache is not None
                    ):
                        self._global_cache_at = time.monotonic()
                        return
                from .costmap import build_costmap, occupancy_from_map_dict
                from .types import OccupancyGrid

                occ = occupancy_from_map_dict(map_data)
                masks = self._zone_masks(map_data)
                if masks is not None and masks.keepout.shape == occ.grid.shape:
                    from ..nav.zones import apply_keepout_to_grid

                    occ = OccupancyGrid(
                        grid=apply_keepout_to_grid(occ.grid, masks.keepout),
                        resolution=occ.resolution,
                        origin_x=occ.origin_x,
                        origin_y=occ.origin_y,
                    )
                costs = build_costmap(
                    occ,
                    inflation_radius_m=self._inflation,
                    robot_radius_m=self._robot_radius,
                    body_radius_m=self._body_radius,
                    cost_scaling_factor=self._cost_scaling,
                    clearance_preference_m=self._clearance_preference_m,
                    mapping=self._slam_is_mapping(),
                )
                with self._global_cache_lock:
                    self._global_occ_cache = occ
                    self._global_costs_cache = costs
                    self._global_cache_generation = generation
                    self._global_cache_zone_rev = zone_rev
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

    def _zone_masks(self, map_data: Optional[dict] = None):
        """Keepout / speed masks for the active map, if zones are configured."""
        getter = getattr(self._world, "zone_masks_for", None)
        if not callable(getter):
            return None
        try:
            return getter(map_data)
        except Exception:  # noqa: BLE001
            return None

    def _zone_revision(self) -> int:
        getter = getattr(self._world, "zone_mask_revision", None)
        if not callable(getter):
            return 0
        try:
            return int(getter())
        except Exception:  # noqa: BLE001
            return 0

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
        mapping = self._slam_is_mapping()
        masks = self._zone_masks(map_data)
        keepout = None if masks is None else masks.keepout
        result = plan_path(
            map_data,
            pose,
            goal,
            inflation_radius_m=self._inflation,
            robot_radius_m=self._robot_radius,
            body_radius_m=self._body_radius,
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
            mapping=mapping,
            keepout_mask=keepout,
        )
        if result.feasible:
            result = connect_plan_start(
                map_data,
                pose,
                result,
                inflation_radius_m=self._inflation,
                robot_radius_m=self._robot_radius,
                body_radius_m=self._body_radius,
                cost_scaling_factor=self._cost_scaling,
                clearance_preference_m=self._clearance_preference_m,
                algorithm=self._algorithm,
                xy_tolerance_m=self._follower.motion.xy_tolerance_m,
                scan=scan,
                local_view=local_view,
                mapping=mapping,
                keepout_mask=keepout,
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
                    mapping=mapping,
                    keepout_mask=keepout,
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

    def _call_check_localization(self, *, apply: Optional[bool] = None) -> Optional[dict]:
        fn = getattr(self._world, "check_localization", None)
        if not callable(fn):
            return None
        try:
            result = fn(
                allow_during_navigation=True,
                full_map_escalation="still_bad",
                apply=apply,
            )
        except TypeError:
            # Older WorldIO doubles without the apply kwarg.
            try:
                result = fn(
                    allow_during_navigation=True,
                    full_map_escalation="still_bad",
                )
            except TimeoutError:
                return {"status": "error", "reason": "timeout"}
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "reason": str(exc).strip() or type(exc).__name__,
                }
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

    def _slam_is_mapping(self) -> bool:
        fn = getattr(self._world, "slam_mode", None)
        if not callable(fn):
            return False
        try:
            return str(fn() or "") == "mapping"
        except Exception:  # noqa: BLE001 - never block drive on a mode read
            return False

    def _maybe_pause_and_refine_localization(
        self, pose: Pose2D, now: float, dist_goal: float
    ) -> Optional[str]:
        """Local refine when the scan is a poor fit for the published pose.

        Returns ``None`` (keep following), ``hold`` (stay stopped),
        ``resume`` (pose moved: replan), or ``continue`` (pose unchanged:
        keep the current path). Does not fail the goal — a leftover residual
        or refused yank is common in hallways.

        A straight, already-moving base runs the check without stopping.
        Stopping is for a hard turn (the scan would be rejected as spinning)
        and for applying a pose correction. An unfixed residual waits
        ``retry_travel_m`` before the next check.
        """
        if not self._nav_loc_refine:
            return None
        # The check compares the live scan to a finished map. While mapping,
        # new rays disagree with the partial grid and check_localization
        # returns not_localizing, so the pause never corrects the pose.
        if self._slam_is_mapping():
            return None
        holding = self._loc_refine_tries > 0
        if now < self._loc_refine_cooldown_until:
            return "hold" if holding else None
        traveled = 0.0
        if self._loc_refine_last_pose is not None:
            traveled = distance_m(pose, self._loc_refine_last_pose)
        if self._loc_refine_need_travel:
            need_m = self._loc_refine_travel_need_m
            if need_m <= 0.0:
                need_m = max(0.5, float(self._nav_loc_refine_check_every_m) or 2.0)
            if traveled < need_m:
                return None
            self._loc_refine_need_travel = False
        due_by_time = now - self._loc_refine_last_check >= self._nav_loc_refine_period_s
        due_by_dist = (
            self._nav_loc_refine_check_every_m > 0.0
            and traveled >= self._nav_loc_refine_check_every_m
        )
        if not due_by_time and not due_by_dist:
            return "hold" if holding else None
        self._loc_refine_last_check = now
        self._loc_refine_last_pose = pose

        verdict = self._measure_loc_disagreement(pose)
        if not verdict.disagree:
            self._loc_refine_need_travel = False
            if holding:
                self._loc_refine_tries = 0
                return self._loc_refine_resume_kind(self._loc_refine_start_pose, pose)
            return None
        if not holding:
            self._loc_refine_start_pose = pose

        # SLAM skips the match above ~0.35 rad/s of yaw. Under that, a cruise
        # command is a valid scan — don't halt just to ask. Stop when the
        # base is turning hard so the settle can bring yaw down first.
        calm = self._loc_refine_yaw_calm()
        if not calm:
            self._pause_for_loc_refine(pose, dist_goal, verdict, settle=True)

        result = self._call_check_localization()
        if result is None:
            return "hold" if holding else None
        status = str(result.get("status") or "")
        if status == "awaiting_confirm":
            if calm:
                self._pause_for_loc_refine(pose, dist_goal, verdict, settle=False)
            self._publish_loc_refine_progress(
                pose, dist_goal, verdict=verdict, status="awaiting_confirm"
            )
            if self._nav_loc_refine_settle_s > 0.0:
                time.sleep(max(0.4, self._nav_loc_refine_settle_s))
            else:
                time.sleep(0.05)
            result = self._call_check_localization() or result
            status = str(result.get("status") or "")
        from .loc_consistency import small_local_match_worth_applying

        if small_local_match_worth_applying(
            result,
            max_shift_m=self._nav_loc_refine_apply_max_m,
            max_shift_deg=self._nav_loc_refine_apply_max_deg,
            min_score=self._nav_loc_refine_apply_min_score,
        ):
            if calm:
                self._pause_for_loc_refine(pose, dist_goal, verdict, settle=False)
            self._publish_loc_refine_progress(
                pose, dist_goal, verdict=verdict, status="applying"
            )
            applied = self._call_check_localization(apply=True)
            if applied is not None:
                result = applied
                status = str(result.get("status") or "")
        if status in ("skipped", "unconfigured"):
            reason = str(result.get("reason") or "")
            if reason in ("spinning", "stale_scan"):
                if calm:
                    self._world.stop()
                    self._note_base_stopped()
                return "hold"
            return "hold" if holding else None

        from .loc_consistency import residual_is_lost

        pose_after = self._world.get_pose() or pose
        after = self._measure_loc_disagreement(pose_after)
        start = self._loc_refine_start_pose or pose
        if after.disagree:
            self._loc_refine_tries += 1
            self._loc_refine_cooldown_until = (
                time.monotonic() + self._nav_loc_refine_cooldown_s
            )
            # A second look only helps when the scan still says "lost"; a
            # moderate residual SLAM could not fix is map change, not drift.
            if (
                self._loc_refine_tries < self._nav_loc_refine_max_tries
                and residual_is_lost(after)
            ):
                self._publish_loc_refine_progress(
                    pose_after, dist_goal, verdict=after, status="retry"
                )
                return "hold"
            # Keep the published pose and the goal. Hallway residuals and
            # refused large jumps are not ``localization_lost``.
            self._loc_refine_tries = 0
            self._loc_refine_need_travel = True
            self._loc_refine_travel_need_m = self._nav_loc_refine_retry_travel_m
            self._publish_loc_refine_progress(
                pose_after, dist_goal, verdict=after, status="continue"
            )
            return self._loc_refine_resume_kind(start, pose_after)

        self._loc_refine_tries = 0
        self._loc_refine_need_travel = False
        self._loc_refine_cooldown_until = (
            time.monotonic() + self._nav_loc_refine_cooldown_s
        )
        self._publish_loc_refine_progress(
            pose_after, dist_goal, verdict=after, status="resumed"
        )
        return self._loc_refine_resume_kind(start, pose_after)

    # Under SLAM's spinning skip (~0.35 rad/s). A cruise yaw is a usable scan.
    _LOC_REFINE_CALM_YAW_RAD_S = 0.25

    def _loc_refine_yaw_calm(self) -> bool:
        """True when the last command is not a hard turn."""
        cmd = self._last_sent_cmd
        if cmd is None:
            return False
        return abs(float(cmd.vtheta)) < self._LOC_REFINE_CALM_YAW_RAD_S

    def _pause_for_loc_refine(self, pose, dist_goal, verdict, *, settle: bool) -> None:
        self._world.stop()
        self._note_base_stopped()
        self._publish_loc_refine_progress(
            pose, dist_goal, verdict=verdict, status="pausing"
        )
        if settle and self._nav_loc_refine_settle_s > 0.0:
            time.sleep(self._nav_loc_refine_settle_s)

    def _goal_timeout_s(self, length_m: float) -> float:
        """``timeout_s``, or 3x the full-speed drive time for long routes."""
        return max(float(self._timeout_s), 3.0 * max(0.0, length_m) / self._max_vel_x)

    @staticmethod
    def _loc_refine_resume_kind(before: Optional[Pose2D], after: Pose2D) -> str:
        """``resume`` (replan) only when the refine actually moved the pose."""
        if before is None:
            return "resume"
        moved = distance_m(before, after) >= 0.15 or abs(
            conv.normalize_angle(after.theta - before.theta)
        ) >= math.radians(5.0)
        return "resume" if moved else "continue"

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
    def _path_locally_blocked(
        *,
        path_ahead_cost: int,
        pose_cost: int,
        activate_cost: int,
        nose_clear: bool,
        forward_clearance_m: Optional[float] = None,
        comfortable_clearance_m: float = 1.0,
    ) -> bool:
        """Whether the local path should force DWA / blocked recovery.

        True lethal on the path, or the body already in hard cost, always
        counts. Soft / inscribed path cost with a clear nose must not —
        that is C-space wall pinch, and waking DWA there is the
        hunt-and-peck (forward/back/spin) in doorways. Pursuit + reactive
        slow/stop handles clear-nose squeezes; DWA is for blocked nose or
        real lethal on the route.
        """
        from .costmap import LETHAL, is_hard

        if is_hard(pose_cost) or int(path_ahead_cost) >= int(LETHAL):
            return True
        if nose_clear:
            return False
        if int(path_ahead_cost) < int(activate_cost):
            return False
        return True

    @staticmethod
    def _local_block_action(
        *,
        nose_clear: bool,
        blocked_for_s: float,
        wait_before_replan_s: float,
        replan_cooldown_ready: bool,
        peel_stuck_s: float = 0.0,
        peel_stuck_limit_s: float = 6.0,
    ) -> str:
        """Choose wait / keep_dwa / replan while the local path cost is high.

        ``keep_dwa``: clear forward cone — stay on the short global path and
        peel with the local planner (fit doorway / wall-inflation pinch).
        Do not stop-replan on the first clear-nose tick (that was the mid-nav
        stutter). If peeling makes no progress for ``peel_stuck_limit_s``,
        escalate — otherwise DWA crawls forever at path_cost=253.
        ``wait``: blocked nose — freeze briefly for dynamic crossers.
        ``replan``: blocked nose + grace, or clear-nose peel stuck.
        """
        if nose_clear:
            if (
                peel_stuck_limit_s > 0.0
                and peel_stuck_s >= peel_stuck_limit_s
                and replan_cooldown_ready
            ):
                return "replan"
            return "keep_dwa"
        if blocked_for_s < wait_before_replan_s:
            return "wait"
        if replan_cooldown_ready:
            return "replan"
        return "wait"

    @staticmethod
    def _blocked_nose_unstick(
        *,
        failed_replans: int,
        nose_clear: bool,
        rear_open: bool,
        spin_clear: bool,
    ) -> str:
        """Motion while a blocked nose waits out the replan cooldown.

        ``hold`` for the first wait, so a person crossing can still move.
        After a replan fails: ``reverse`` when the rear is open, otherwise
        ``turn`` when an in-place spin would not sweep an obstacle.
        """
        if nose_clear or int(failed_replans) < 1:
            return "hold"
        if rear_open:
            return "reverse"
        if spin_clear:
            return "turn"
        return "hold"

    @staticmethod
    def _bumper_spin_reverse(
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

    def _abort_blocked_replans(self, count: int, *, nose_clear: bool) -> bool:
        """Fail the goal once blocked-nose replans are exhausted."""
        if count < _BLOCKED_REPLAN_FAIL_LIMIT or nose_clear:
            return False
        self._world.stop()
        with self._status_lock:
            prev_progress = dict(self._status.progress or {})
        if self._stuck_pose_refine:
            prev_progress["stuck_pose_refine"] = dict(self._stuck_pose_refine)
        self._set_status(
            state="failed",
            active=False,
            error_msg=(
                "path blocked: no route around obstacle "
                f"({count} replans failed)"
            ),
            progress=prev_progress or None,
        )
        return True

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
        allow_lift_ban: bool = True,
        force_lift_short_flip: bool = False,
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
        - After accepting a longer detour, ban the abandoned short corridor for
          the rest of this goal: keep painting it and reject much-shorter /
          matching candidates. A wall-clock hold only delayed the flip; lift
          the ban only when the detour itself is dead and nothing else works.
        """
        from .controller import _path_length
        from .local_planner import path_cost_in_local_window
        from .path_utils import closest_point_on_path

        old_len = _path_length(path)
        self._last_replan_trigger = str(trigger or "")
        hold_active = self._detour_ban_path is not None
        # Prefer painting the abandoned short corridor while the ban is live;
        # otherwise only escalate to corridor paint after a failed peel.
        attempts: list[tuple[str, bool]] = [("scan+local", False)]
        if hold_active or failed_count >= 1:
            attempts.append(("blocked-corridor", True))
        reasons: list[str] = []
        _, _, _, along = closest_point_on_path(pose, path)
        remaining = max(0.5, old_len - along)
        best: Optional[tuple[float, Path2D, PlanResult, str]] = None
        differ_tol = 0.12 if local_view is not None else 0.25
        block_cost = int(self._local_planner_activate_cost)
        footprint_skip = max(0.05, float(self._robot_radius))
        ban_path = self._detour_ban_path if hold_active else None

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

        def _reject_short_flip(candidate: Path2D, new_len: float) -> Optional[str]:
            if not hold_active:
                return None
            if (
                self._detour_min_length_m > 0.0
                and new_len < self._detour_min_length_m
            ):
                return (
                    f"short-flip hysteresis ({new_len:.1f} m < "
                    f"{self._detour_min_length_m:.1f} m ban)"
                )
            if ban_path is not None and not paths_meaningfully_differ(
                ban_path, candidate, tol_m=differ_tol
            ):
                return "short-flip hysteresis (matches abandoned corridor)"
            return None

        for label, paint in attempts:
            # During a detour ban, always seal the abandoned short corridor —
            # never the path we just committed to.
            do_paint = paint or hold_active
            paint_path = ban_path if (hold_active and ban_path is not None) else path
            replanned = self.plan(
                goal,
                start=pose,
                scan=scan,
                blocked_path=paint_path if do_paint else path,
                blocked_path_pose=pose,
                local_view=local_view,
                paint_corridor=do_paint,
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
            flip = _reject_short_flip(replanned.path, new_len)
            if flip is not None:
                reasons.append(f"{label}: {flip}")
                continue
            if paint and new_len > remaining * 1.8 and best is not None:
                reasons.append(
                    f"{label}: too long ({new_len:.1f} m > {remaining * 1.8:.1f} m)"
                )
                continue
            if best is None or new_len < best[0]:
                best = (new_len, replanned.path, replanned, label)
            if not paint and not hold_active:
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
                    flip = _reject_short_flip(new_path, new_len)
                    if flip is not None:
                        reasons.append(f"{label}: {flip}")
                    else:
                        self._commit_replan_path(
                            path, new_path, new_len, old_len, label, reasons,
                            require_different=require_different,
                            result=result,
                            goal=goal,
                            pose=pose,
                        )
                        return new_path
            else:
                reasons.append("forced-via: none feasible")
        if best is None:
            # Detour itself is dead: drop the corridor ban once and retry.
            # Normally do NOT lift when we only rejected short-flip candidates
            # (that is the thrash). Exception: path is actually blocked /
            # pose-jump recovery needs any escape (force_lift_short_flip).
            ban_was_only_reason = any("short-flip" in r for r in reasons)
            if hold_active and allow_lift_ban and (
                not ban_was_only_reason or force_lift_short_flip
            ):
                reasons.append("detour-ban: lifting (no alternate)")
                self._detour_ban_path = None
                self._detour_min_length_m = 0.0
                return self._try_replan(
                    goal,
                    pose,
                    path,
                    scan,
                    require_different=require_different,
                    failed_count=failed_count,
                    local_view=local_view,
                    trigger=trigger,
                    allow_lift_ban=False,
                    force_lift_short_flip=False,
                )
            self._last_replan_error = "; ".join(reasons)
            self._last_replan_info = {
                "trigger": self._last_replan_trigger,
                "accepted": None,
                "old_length_m": round(old_len, 3),
                "new_length_m": None,
                "require_different": bool(require_different),
                "detour_hold": bool(hold_active),
                "attempts": list(reasons),
            }
            return None
        new_len, new_path, result, label = best
        self._commit_replan_path(
            path, new_path, new_len, old_len, label, reasons,
            require_different=require_different,
            result=result,
            goal=goal,
            pose=pose,
        )
        return new_path

    def _commit_replan_path(
        self,
        old_path: Path2D,
        new_path: Path2D,
        new_len: float,
        old_len: float,
        label: str,
        reasons: list[str],
        *,
        require_different: bool,
        result: PlanResult,
        goal: Pose2D,
        pose: Pose2D,
    ) -> None:
        """Publish an accepted replan and ban the abandoned short corridor."""
        self._last_replan_error = ""
        hold = self._detour_ban_path is not None
        # Commit to a longer / different corridor for the rest of this goal.
        if (
            new_len > old_len * 1.15
            and paths_meaningfully_differ(old_path, new_path, tol_m=0.12)
        ):
            self._detour_min_length_m = max(
                new_len * self._detour_short_ratio, old_len * 1.05
            )
            self._detour_ban_path = old_path
            hold = True
        self._last_replan_info = {
            "trigger": self._last_replan_trigger,
            "accepted": label,
            "old_length_m": round(old_len, 3),
            "new_length_m": round(new_len, 3),
            "require_different": bool(require_different),
            "detour_hold": bool(hold),
            "attempts": list(reasons),
        }
        preview = self._publish_plan_viz(result, goal, start=pose)
        self._set_status(path=preview["path"], length_m=preview["length_m"])

    def _replan_blocked_at_start(self) -> bool:
        err = self._last_replan_error or ""
        return (
            "cannot reach plan start" in err
            or "start pose is in lethal" in err
        )

    def _refine_stuck_pose(self) -> Optional[Pose2D]:
        """Ask SLAM for one small local match. Returns the pose when it moved."""
        fn = getattr(self._world, "refine_stuck_pose", None)
        if not callable(fn):
            return None
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - a match timeout must not kill the loop
            self._stuck_pose_refine = {
                "status": "error",
                "reason": str(exc).strip() or type(exc).__name__,
                "corrected": False,
            }
            return None
        if not isinstance(result, dict):
            self._stuck_pose_refine = {"status": "error", "corrected": False}
            return None
        self._stuck_pose_refine = dict(result)
        if not result.get("corrected"):
            return None
        matched = result.get("pose")
        if isinstance(matched, dict) and "x" in matched and "y" in matched:
            try:
                return Pose2D(
                    float(matched["x"]),
                    float(matched["y"]),
                    float(matched.get("theta") or 0.0),
                )
            except (TypeError, ValueError):
                pass
        return self._world.get_pose()

    def _recover_unreachable_start(
        self,
        goal: Pose2D,
        pose: Pose2D,
        path: Path2D,
        scan: Optional[conv.LaserScan2D],
        *,
        failed_count: int,
        local_view,
        trigger: str,
    ) -> Optional[Path2D]:
        """Replan once from a corrected pose when the start cell is unreachable.

        Only when the last replan died because the planner could not leave the
        current cell. One attempt per blocked episode. The retry accepts the
        same corridor: painting it blocked is what sealed the hallway.
        """
        del pose, failed_count
        if self._stuck_pose_refine_used or not self._replan_blocked_at_start():
            return None
        self._stuck_pose_refine_used = True
        corrected = self._refine_stuck_pose()
        if corrected is None:
            return None
        return self._try_replan(
            goal,
            corrected,
            path,
            scan,
            require_different=False,
            failed_count=0,
            local_view=local_view,
            trigger=f"{trigger} after stuck_pose_refine",
        )

    def run_goal(self, goal: Pose2D) -> None:
        """Plan and follow until success, failure, or cancel. Blocking."""
        self._cancel.clear()
        self._last_replan_error = ""
        self._last_replan_trigger = ""
        self._last_replan_info = {}
        self._stuck_pose_refine_used = False
        self._stuck_pose_refine = {}
        self._detour_ban_path = None
        self._detour_min_length_m = 0.0
        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = 0.0
        self._loc_refine_last_check = 0.0
        self._loc_refine_last_pose = None
        self._loc_refine_start_pose = None
        self._loc_refine_need_travel = False
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

            deadline = time.monotonic() + self._goal_timeout_s(
                float(preview.get("length_m") or 0.0)
            )
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
            # Blocked-nose unstick between replans: one reverse or turn, then
            # a replan from the pose that motion reached.
            nose_unstick_start: Optional[Pose2D] = None
            nose_unstick_mode = ""
            nose_unstick_yaw_sign = 1.0
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
            align_best_err: Optional[float] = None
            align_progress_at: Optional[float] = None
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
            # Consecutive failed blocked-replans; reset only by a successful
            # replan or real progress toward the goal (not by a one-tick
            # local_blocked flicker).
            blocked_fail_count = 0
            blocked_fail_dist: Optional[float] = None

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
                if (
                    blocked_fail_dist is not None
                    and distance_m(pose, goal) < blocked_fail_dist - 0.5
                ):
                    blocked_fail_count = 0
                    blocked_fail_dist = None

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
                xy_ok = self._xy_at_nav_goal(pose, goal, path)
                yaw_err_abs = abs(conv.normalize_angle(pose.theta - goal.theta))
                yaw_ok = yaw_err_abs <= self._follower.motion.yaw_tolerance_rad
                now = time.monotonic()
                if xy_ok and yaw_ok:
                    self._world.stop()
                    self._set_status(state="succeeded", active=False, error_msg="")
                    return
                # Give up on final heading only when a spin inside the XY ball
                # stops reducing the error. A rotation that is still closing
                # the gap keeps going past the stall window.
                if xy_ok:
                    give_up, align_best_err, align_progress_at = yaw_align_give_up(
                        now,
                        yaw_err_abs,
                        self._yaw_align_timeout_s,
                        align_best_err,
                        align_progress_at,
                    )
                    if give_up:
                        self._world.stop()
                        self._set_status(
                            state="succeeded",
                            active=False,
                            error_msg="",
                        )
                        return
                else:
                    align_best_err = None
                    align_progress_at = None

                if holding_for_localize:
                    hold_status = str((loc_hold or {}).get("status") or "")
                    # A leftover mid-nav ``nav_hold`` (refused hallway yank)
                    # must not skip refine forever. ``awaiting_confirm`` still
                    # stops until SLAM applies or rejects.
                    if hold_status == "nav_hold" and self._nav_loc_refine:
                        holding_for_localize = False
                        entering_loc_hold = False
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
                    # Localization stops do not count against the goal timeout.
                    deadline += time.monotonic() - tick_started
                    continue

                now = time.monotonic()
                loc_outcome = self._maybe_pause_and_refine_localization(
                    pose, now, dist_goal_chk
                )
                if loc_outcome == "fail":
                    return
                if loc_outcome in ("hold", "resume", "continue"):
                    if loc_outcome == "resume":
                        pending_loc_replan = True
                    last_progress_at = now
                    last_progress_pose = pose
                    last_progress_dist = dist_goal_chk
                    last_progress_bearing = float("inf")
                    spin_stuck_since = None
                    self._sleep_control_period(tick_started)
                    deadline += time.monotonic() - tick_started
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
                        # Fused lidar + obstacles_only depth: reactive cone,
                        # local costmap / DWA, and nose_clear. Loc refine still
                        # reads lidar-only (the map was built from lidar).
                        scan = self._world.get_scan(self._scan_max_age)
                    except TimeoutError:
                        scan = None
                mem_pts = None
                if self._guard is not None:
                    self._depth_memory.update(
                        depth_frames(self._world), pose, time.monotonic()
                    )
                    mem_pts = self._depth_memory.points()
                if self._above_memory is not None:
                    self._above_memory.update(
                        above_cart_frames(self._world), pose, time.monotonic()
                    )
                    above_pts = self._above_memory.points()
                    if above_pts.size:
                        mem_pts = (
                            above_pts
                            if mem_pts is None or not np.size(mem_pts)
                            else np.vstack([np.asarray(mem_pts), above_pts])
                        )

                local_view = self._local_view_cache
                if refresh_local:
                    self._kick_global_costmap_refresh(allow_inline=True)
                    # Missing scan must not wipe live marks — that made
                    # path_blocked_local flicker false under IO load while
                    # reactive avoid still saw the obstacle on the next tick.
                    if scan is None:
                        local_view = self._local_view_cache
                    else:
                        costmap_scan = scan
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
                            extra_points=mem_pts,
                            mapping=self._slam_is_mapping(),
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
                # Nose / clearance first — inscribed path cost with a clear
                # nose must not force DWA (hunt-and-peck in doorways).
                nose_clear = True
                forward_clearance_m: Optional[float] = None
                guard_pts = None
                obs_cfg = self._follower.obstacle
                if self._guard is not None:
                    # Nose = can the rectangle move forward at all (straight
                    # or on some arc)? Not a padded lidar cone.
                    guard_pts = obstacle_points(
                        pose,
                        scan,
                        local_view,
                        radius_m=self._guard.cfg.obstacle_radius_m,
                        extra=mem_pts,
                    )
                    straight = self._guard.free_distance(
                        pose, 0.2, 0.0, guard_pts, 2.0
                    )
                    if math.isfinite(straight):
                        forward_clearance_m = straight + self._guard.cfg.length_m / 2.0
                    nose_clear = straight >= self._guard.cfg.alt_min_free_m or (
                        self._guard.best_forward_arc(
                            pose, guard_pts, None, self._guard.cfg.horizon_m, 0.0
                        )
                        is not None
                    )
                elif obs_cfg is not None and obs_cfg.enabled and scan is not None:
                    half = float(obs_cfg.front_cone_half_rad)
                    nose_range = cone_min_range(scan, -half, half)
                    if math.isfinite(nose_range):
                        forward_clearance_m = float(nose_range)
                    nose_clear = (
                        forward_clearance_m is None
                        or forward_clearance_m > obs_cfg.stop_distance_m
                    )
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
                    # Local costs are footprint-inflated: hard pose cost means
                    # the body already overlaps keep-out.
                    pose_cost = int(local_view.cost_at_world(pose.x, pose.y))
                    local_blocked = self._path_locally_blocked(
                        path_ahead_cost=path_ahead_cost,
                        pose_cost=pose_cost,
                        activate_cost=int(self._local_planner_activate_cost),
                        nose_clear=nose_clear,
                        forward_clearance_m=forward_clearance_m,
                    )
                # Reactive avoid spinning with a clear-looking path still means
                # the robot cannot proceed — escalate to the blocked/replan path.
                # Exception: path centerline free — a side/corridor phantom
                # used to force avoid→replan forever while path_cost=0
                # (live rc21: planning churn). Require real local cost.
                if (
                    last_obstacle_state == "avoid"
                    and reactive_avoid_since is not None
                    and now - reactive_avoid_since >= 0.8
                    and self._path_locally_blocked(
                        path_ahead_cost=path_ahead_cost,
                        pose_cost=pose_cost,
                        activate_cost=int(self._local_planner_activate_cost),
                        nose_clear=nose_clear,
                        forward_clearance_m=forward_clearance_m,
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
                # - Clear nose: pursuit + reactive slow (no DWA peel).
                wait_before_replan_s = max(
                    self._recovery_wait_duration_s,
                    self._replan_local_blocked_time_s,
                )
                waiting_for_clear = False
                if local_blocked:
                    if local_blocked_since is None:
                        local_blocked_since = now
                    blocked_for = now - local_blocked_since
                    # Exponential backoff between failed blocked-replans: each
                    # attempt is a full-map plan, and retrying every 0.5 s
                    # forever starves the control loop.
                    cooldown_ready = now - last_local_replan_at >= min(
                        8.0,
                        self._replan_local_min_period_s
                        * (2.0 ** min(blocked_fail_count, 4)),
                    )
                    # How long since real motion while peeling with a clear nose.
                    peel_stuck_s = (
                        max(0.0, now - last_progress_at)
                        if nose_clear
                        else 0.0
                    )
                    action = self._local_block_action(
                        nose_clear=nose_clear,
                        blocked_for_s=blocked_for,
                        wait_before_replan_s=wait_before_replan_s,
                        replan_cooldown_ready=cooldown_ready,
                        peel_stuck_s=peel_stuck_s,
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
                        if new_path is None:
                            recovered = self._recover_unreachable_start(
                                goal,
                                pose,
                                path,
                                scan,
                                failed_count=max(1, failed_replan_while_blocked),
                                local_view=local_view,
                                trigger=_trig,
                            )
                            if recovered is not None:
                                new_path = recovered
                                fresh = self._world.get_pose()
                                if fresh is not None:
                                    pose = fresh
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
                            blocked_fail_count = 0
                            blocked_fail_dist = None
                            nose_unstick_start = None
                            nose_unstick_mode = ""
                        else:
                            failed_replan_while_blocked += 1
                            blocked_fail_count += 1
                            if blocked_fail_dist is None:
                                blocked_fail_dist = distance_m(pose, goal)
                            # Nav2 BT semantics: bounded retries, then fail —
                            # not an infinite replan loop in front of a plug.
                            if self._abort_blocked_replans(
                                blocked_fail_count, nose_clear=nose_clear
                            ):
                                return
                else:
                    local_blocked_since = None
                    failed_replan_while_blocked = 0
                    nose_unstick_start = None
                    nose_unstick_mode = ""
                    self._stuck_pose_refine_used = False

                # DWA only when the route is actually blocked (or after a failed
                # detour). Clear-nose C-space pinch: pursuit + reactive slow —
                # DWA reverse/spin samples were the doorway hunt-and-peck.
                allow_local_planner = (
                    self._local_costmap_enabled
                    and not waiting_for_clear
                    and (
                        local_blocked
                        or failed_replan_while_blocked >= 1
                    )
                )
                force_local = bool(local_blocked and allow_local_planner)
                # Reactive stop/slow, nose_clear, and the local costmap all use
                # the fused scan so depth can catch low / lidar-blind hits.
                # Loc refine stays lidar-only. Replan guards still refuse
                # avoid→replan death spirals when the nose and path are clear.
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
                    guard=self._guard,
                    guard_extra_pts=mem_pts,
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
                    **(
                        {"stuck_pose_refine": dict(self._stuck_pose_refine)}
                        if self._stuck_pose_refine
                        else {}
                    ),
                    "nose_clear": bool(nose_clear),
                    "above_cart": (
                        int(len(self._above_memory))
                        if self._above_memory is not None
                        else None
                    ),
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

                # Bumper against an obstacle, path itself still free: the
                # footprint guard will not reverse, and the blocked-nose wait
                # never starts because path cost is low. Spin is blocked, so
                # back up when the rectangle itself can move backward. The
                # rear lidar cone and the inflated costmap both treat the
                # blob an inch ahead as occupying the reverse, which left
                # this case at cmd 0 with obstacle "avoid".
                if (
                    now >= narrow_rev_cooldown_until
                    and self._bumper_spin_reverse(
                        nose_clear=nose_clear,
                        spin_blocked=bool(progress.get("spin_blocked")),
                        cmd_vx=cmd.vx,
                        cmd_vtheta=cmd.vtheta,
                        rear_open=True,
                    )
                ):
                    from .controller import _narrow_reverse_command, _try_narrow_reverse

                    back_free = 0.0
                    if self._guard is not None and guard_pts is not None:
                        back_free = self._guard.free_distance(
                            pose,
                            -0.15,
                            0.0,
                            guard_pts,
                            max(0.35, float(self._backup_dist_m)),
                        )
                    rectangle_clear = (
                        not math.isfinite(back_free) or back_free >= 0.12
                    )
                    cone_clear = False
                    if (
                        not rectangle_clear
                        and scan is not None
                        and local_view is not None
                    ):
                        cone_clear = (
                            _try_narrow_reverse(
                                self._follower,
                                scan,
                                self._robot_radius,
                                local_view=local_view,
                                current=pose,
                            )
                            is not None
                        )
                    if rectangle_clear or cone_clear:
                        cmd = _narrow_reverse_command(self._follower)
                        progress = {
                            **progress,
                            "obstacle": "narrow_reverse",
                            "local_planner": False,
                            "cmd_vx_mps": cmd.vx,
                            "cmd_vtheta_rad_s": 0.0,
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
                    # Inflation of the blob on the nose reaches backward
                    # through the body and fails the costmap check while the
                    # rectangle still has room to back away from it.
                    if (
                        not rear_cost_ok
                        and self._guard is not None
                        and guard_pts is not None
                    ):
                        back_free = self._guard.free_distance(
                            pose, -0.15, 0.0, guard_pts, remain
                        )
                        rear_cost_ok = (
                            not math.isfinite(back_free)
                            or back_free >= min(remain, 0.12)
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

                        # Same fused scan as nose_clear. Forward depth does
                        # not invent rear hits; body-near phantoms are filtered.
                        rev_scan = scan
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
                    # After a blocked-nose replan fails, the cooldown wait
                    # used to sit at cmd 0 until the next attempt. Back up
                    # when the rear is open, otherwise turn when the spin
                    # disc is clear, then replan from that new pose.
                    unstick_done = False
                    if (
                        not nose_clear
                        and failed_replan_while_blocked >= 1
                        and scan is not None
                        and local_view is not None
                        and abs(cmd.vx) < 1e-6
                        and abs(cmd.vtheta) < 1e-6
                    ):
                        from .controller import _try_narrow_reverse

                        rev = _try_narrow_reverse(
                            self._follower,
                            scan,
                            self._robot_radius,
                            local_view=local_view,
                            current=pose,
                        )
                        spin_clear = spin_clearance_m(scan) >= self._spin_radius + 0.05
                        if (
                            spin_clear
                            and self._spin_radius > self._robot_radius
                            and spin_disc_blocked(
                                local_view,
                                pose.x,
                                pose.y,
                                spin_radius_m=self._spin_radius,
                                inscribed_radius_m=self._robot_radius,
                            )
                        ):
                            spin_clear = False
                        choice = self._blocked_nose_unstick(
                            failed_replans=failed_replan_while_blocked,
                            nose_clear=nose_clear,
                            rear_open=rev is not None,
                            spin_clear=spin_clear,
                        )
                        if choice == "reverse" and rev is not None:
                            if nose_unstick_mode != "reverse":
                                nose_unstick_start = pose
                                nose_unstick_mode = "reverse"
                            cmd = rev
                        elif choice == "turn":
                            backed = (
                                distance_m(pose, nose_unstick_start)
                                if nose_unstick_mode == "reverse"
                                and nose_unstick_start is not None
                                else 0.0
                            )
                            if backed >= 0.05:
                                unstick_done = True
                            else:
                                if nose_unstick_mode != "turn":
                                    nose_unstick_start = pose
                                    nose_unstick_mode = "turn"
                                cmd = DriveCommand(
                                    0.0,
                                    0.0,
                                    nose_unstick_yaw_sign * _NOSE_UNSTICK_YAW_RATE,
                                    False,
                                )
                        elif nose_unstick_start is not None:
                            moved = False
                            if nose_unstick_mode == "reverse":
                                moved = (
                                    distance_m(pose, nose_unstick_start) >= 0.05
                                )
                            elif nose_unstick_mode == "turn":
                                moved = (
                                    abs(
                                        conv.normalize_angle(
                                            pose.theta - nose_unstick_start.theta
                                        )
                                    )
                                    >= 0.2
                                )
                            if moved:
                                unstick_done = True
                            else:
                                nose_unstick_start = None
                                nose_unstick_mode = ""
                    if (
                        not nose_clear
                        and failed_replan_while_blocked >= 1
                        and cmd.vx < -1e-6
                        and nose_unstick_mode not in ("reverse", "turn")
                    ):
                        nose_unstick_start = pose
                        nose_unstick_mode = "reverse"
                    if (
                        not unstick_done
                        and nose_unstick_start is not None
                        and nose_unstick_mode == "reverse"
                        and cmd.vx < -1e-6
                    ):
                        if (
                            distance_m(pose, nose_unstick_start)
                            >= self._backup_dist_m
                        ):
                            unstick_done = True
                    elif (
                        not unstick_done
                        and nose_unstick_start is not None
                        and nose_unstick_mode == "turn"
                        and abs(cmd.vtheta) > 1e-6
                    ):
                        if (
                            abs(
                                conv.normalize_angle(
                                    pose.theta - nose_unstick_start.theta
                                )
                            )
                            >= _NOSE_UNSTICK_YAW_RAD
                        ):
                            unstick_done = True
                    if unstick_done:
                        self._stop_before_replan("blocked_nose_unstick")
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=max(1, failed_replan_while_blocked),
                            require_different=failed_replan_while_blocked < 3,
                            local_view=local_view,
                            trigger="blocked_nose_unstick",
                        )
                        replan_finished = time.monotonic()
                        last_local_replan_at = replan_finished
                        last_replan = replan_finished
                        nose_unstick_start = None
                        nose_unstick_mode = ""
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
                            blocked_fail_count = 0
                            blocked_fail_dist = None
                        else:
                            failed_replan_while_blocked += 1
                            blocked_fail_count += 1
                            if blocked_fail_dist is None:
                                blocked_fail_dist = distance_m(pose, goal)
                            if self._abort_blocked_replans(
                                blocked_fail_count, nose_clear=nose_clear
                            ):
                                return
                        cmd = DriveCommand(0.0, 0.0, 0.0, False)
                        progress = {
                            **progress,
                            "obstacle": "planning",
                            "local_planner": False,
                            "nose_unstick": None,
                            "cmd_vx_mps": 0.0,
                            "cmd_vtheta_rad_s": 0.0,
                        }
                    elif cmd.vx < -1e-6 and abs(cmd.vtheta) < 1e-6:
                        progress = {
                            **progress,
                            "obstacle": "wait_reverse",
                            "local_planner": False,
                            "nose_unstick": nose_unstick_mode or None,
                            "cmd_vx_mps": cmd.vx,
                            "cmd_vtheta_rad_s": 0.0,
                        }
                    elif (
                        nose_unstick_mode == "turn" and abs(cmd.vtheta) > 1e-6
                    ):
                        progress = {
                            **progress,
                            "obstacle": "wait_turn",
                            "local_planner": False,
                            "nose_unstick": "turn",
                            "cmd_vx_mps": 0.0,
                            "cmd_vtheta_rad_s": cmd.vtheta,
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
                            "nose_unstick": None,
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
                        masks = self._zone_masks(map_data)
                        keepout = None if masks is None else masks.keepout
                        static_blocked = map_data is not None and path_blocked(
                            map_data,
                            path,
                            inflation_radius_m=self._inflation,
                            robot_radius_m=self._robot_radius,
                            from_pose=pose,
                            ahead_m=path_block_horizon_m,
                            mapping=self._slam_is_mapping(),
                            keepout_mask=keepout,
                        )
                    # Large localization corrections used to force a replan even
                    # when the polyline was still free — that stop+replan looped
                    # forever under the detour ban (short-flip rejected every
                    # shorter candidate, pending_loc_replan never cleared).
                    # Soft loc nudge with a clear path: keep following.
                    if pose_jumped and not static_blocked:
                        pending_loc_replan = False

                backup_exhausted = (
                    backup_attempts >= self._backup_max_attempts and local_blocked
                )
                # Sign-flip rock (narrow crawl ↔ reverse) with a *blocked* nose
                # still forces a replan. With a clear nose, the same forward/
                # back jig is normal DWA peel in a tight corridor — stop-replan
                # there was the "crazy jig then freeze" (live: oscillating=True,
                # static_blocked=False, nose_clear). peel_stuck covers escalate.
                osc_replan = bool(
                    oscillating and local_blocked and not nose_clear
                )
                spin_rock = bool(
                    progress.get("spin_blocked") and oscillating and not nose_clear
                )
                should_replan = (replan_due or pose_jumped) and (
                    static_blocked
                    or osc_replan
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
                        # Path is actually blocked: do not let the goal-lifetime
                        # detour ban refuse every shorter escape.
                        allow_lift_ban=True,
                        force_lift_short_flip=bool(static_blocked),
                    )
                    replan_finished = time.monotonic()
                    last_local_replan_at = replan_finished
                    last_replan = replan_finished
                    # Always drop the loc-replan latch — otherwise a failed
                    # attempt (short-flip / infeasible) stop-replans forever.
                    pending_loc_replan = False
                    if new_path is not None:
                        path = new_path
                        last_progress_at = now
                        local_blocked_since = None
                        failed_replan_while_blocked = 0
                        failed_static_replan = 0
                        backup_attempts = 0
                        vx_sign_history.clear()
                        spin_stuck_since = None
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
                pre_limit = cmd
                cmd = self._rate_limited(cmd)
                if self._guard is not None and not cmd.done:
                    # Reduce-only veto on whatever is about to go to the base
                    # (slew-limited pursuit, backup, recovery spins).
                    veto_pts = guard_pts

                    def _veto(c: DriveCommand):
                        return self._guard.guard(
                            pose,
                            c.vx,
                            c.vtheta,
                            veto_pts,
                            max_dist_m=distance_m(pose, goal),
                            allow_steer=False,
                        )

                    def _reduced(c: DriveCommand, v) -> bool:
                        return abs(v.vx) < abs(c.vx) - 1e-9 or abs(v.vtheta) < abs(
                            c.vtheta
                        ) - 1e-9

                    veto = _veto(cmd)
                    if _reduced(cmd, veto) and abs(pre_limit.vx) > 1e-6:
                        # The slew limiter ramps ω slower than v, which
                        # straightens the arc the guard approved — through a
                        # doorway that clips the jamb. Keep the approved
                        # curvature at the ramped speed instead.
                        same_arc = DriveCommand(
                            cmd.vx,
                            cmd.vy,
                            pre_limit.vtheta * (cmd.vx / pre_limit.vx),
                            cmd.done,
                        )
                        arc_veto = _veto(same_arc)
                        if not _reduced(same_arc, arc_veto):
                            cmd, veto = same_arc, arc_veto
                            self._last_sent_cmd = cmd
                    if _reduced(cmd, veto):
                        cmd = DriveCommand(veto.vx, cmd.vy, veto.vtheta, cmd.done)
                        self._last_sent_cmd = cmd
                    near_d, near_x, near_y = self._guard.nearest(pose, guard_pts)
                    _TRACE.append(
                        {
                            "t": round(time.time(), 3),
                            "x": round(pose.x, 3),
                            "y": round(pose.y, 3),
                            "th": round(pose.theta, 4),
                            "vx": round(cmd.vx, 3),
                            "w": round(cmd.vtheta, 3),
                            "obs": progress.get("obstacle"),
                            "near_m": round(near_d, 3) if math.isfinite(near_d) else None,
                            "near_bx": round(near_x, 3) if math.isfinite(near_x) else None,
                            "near_by": round(near_y, 3) if math.isfinite(near_y) else None,
                            "pts": int(len(guard_pts)),
                            "mem": int(len(mem_pts)) if mem_pts is not None else 0,
                            "wp": progress.get("waypoint_index"),
                        }
                    )
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
                # Clear-nose C-space pinch (pathc=253, local_blocked=false after
                # rc22): slow pursuit must get the same grace or stall aborts
                # the goal while crawling a doorway.
                elif nose_clear and path_ahead_cost >= self._local_planner_activate_cost:
                    from .costmap import LETHAL

                    if path_ahead_cost < int(LETHAL):
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
                    from .costmap import LETHAL

                    # Clear nose + non-lethal path: C-space pinch / slow crawl.
                    # Stop-replan cannot help and was failing goals (live rc22:
                    # stall with pathc=253, nose_clear, local_blocked=false).
                    if nose_clear and path_ahead_cost < int(LETHAL):
                        last_progress_at = now
                        return False
                    # Nose on the bumper and spin disc blocked: the escape is
                    # reverse, not another plan of the same free corridor.
                    if (
                        not nose_clear
                        and bool(progress.get("spin_blocked"))
                        and cmd.vx < -1e-6
                    ):
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
                    # Cap linear speed inside speed_limit zones (percent of cmd).
                    masks = self._zone_masks()
                    if masks is not None:
                        pct = masks.speed_pct_at(pose.x, pose.y)
                        if pct is not None and pct < 100.0:
                            scale = pct / 100.0
                            cmd = DriveCommand(
                                vx=cmd.vx * scale,
                                vy=cmd.vy * scale,
                                vtheta=cmd.vtheta,
                                done=cmd.done,
                            )
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

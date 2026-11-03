"""Goal lifecycle: plan → follow → replan → succeed / fail / cancel."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from ...config import NavConfig
from ...geom import conversions as conv
from ..io.world_io import WorldIO
from ..motion.base_envelope import LIN_EPS_MPS, lift_sub_deadband
from ..motion.simple_motion import (
    DriveCommand,
    ObstacleConfig,
    SimpleMotionConfig,
    distance_m,
)
from ..planning.local_costmap import (
    LocalCostmap,
    LocalCostmapConfig,
)
from ..planning.local_planner import LocalPlannerConfig
from ..planning.planner import (
    connect_plan_start,
    paths_meaningfully_differ,
    plan_path,
)
from ..planning.smoother import smooth_path, smooth_plan_path
from ..stores.zones import apply_speed_zone_limit
from ..thresholds import (
    AHEAD_HALF_WIDTH_PAD_M,
    BLOCKED_REPLAN_FAIL_LIMIT,
    CORRIDOR_ALIGN_RAD,
    CORRIDOR_DRIVE_MPS,
    CORRIDOR_MAX_YAW_RAD,
    CORRIDOR_MIN_TRAVEL_M,
    CORRIDOR_SPIN_RAD,
    NOSE_STOP_MARGIN_M,
    YAW_ALIGN_PROGRESS_RAD,
)
from ..types import NavStatus, Path2D, PlanResult, Pose2D
from .above_cart import AboveCartMemory
from .backup_recovery import run_backup_recovery
from .block_policy import run_block_policy
from .controller import (
    FollowerConfig,
    limit_twist_rate,
    lookahead_pose,
)
from .depth_memory import DepthObstacleMemory
from .escape import (
    UNSTICK_YAW_RATE as _NOSE_UNSTICK_YAW_RATE,
)
from .escape import (
    blocked_nose_unstick,
    bumper_wants_reverse,
    undrivable_nose,
)
from .final_gate import run_final_gate
from .footprint_guard import (
    FootprintGuard,
    GuardConfig,
    obstacle_points,
    open_corridor_yaw,
)
from .goal_episode import GoalEpisode
from .loc_hold_tick import run_loc_hold_tick
from .path_follow import run_path_follow
from .periodic_replan import run_periodic_replan
from .replan_controller import apply_blocked_replan
from .runtime_kwargs import resolve_nav_cfg
from .sense_tick import elapsed_ms, run_sense_tick
from .stall_watch import run_stall_watch
from .tick_output import finish_tick_output, send_tick_velocity

# Nose-unstick yaw budget/rate live in ``escape`` (_NOSE_UNSTICK_* aliases).
# Corridor / yaw-align / blocked-replan policy numbers live in ``thresholds``.

# Last ~60 s of guarded control ticks (20 Hz), kept across goals so a graze
# can be inspected after the operator cancels (``get_trace`` DoCommand).
_TRACE: "deque[dict]" = deque(maxlen=1200)


def recent_trace(seconds: float = 30.0) -> list:
    entries = list(_TRACE)
    if not entries:
        return []
    cutoff = entries[-1]["t"] - max(0.0, float(seconds))
    return [e for e in entries if e["t"] >= cutoff]


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
    if yaw_err_abs < best_err - YAW_ALIGN_PROGRESS_RAD:
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
        # footprint_guard is supervisor-only; strip before resolve.
        footprint_guard = bool(overrides.pop("footprint_guard", True))
        self._cfg = resolve_nav_cfg(nav_cfg, **overrides)
        cfg = self._cfg
        b = cfg.builtin

        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = 0.0
        self._loc_refine_last_check = 0.0
        self._loc_refine_last_pose: Optional[Pose2D] = None
        self._loc_refine_start_pose: Optional[Pose2D] = None
        self._loc_refine_need_travel = False
        self._loc_refine_travel_need_m = 0.0
        # One stopped recovery attempt per goal (wider/trusted re-localize).
        self._loc_refine_recovery_used = False
        # Require consecutive clear checks before resuming from hold_bad_loc.
        self._loc_refine_clear_streak = 0
        self._loc_refine_clear_needed = 2
        # Localization matches run off the control thread. The follow loop
        # only starts them and polls; it never waits, so obstacle avoidance
        # keeps ticking while SLAM is matching.
        self._loc_phase = "idle"
        self._loc_step = ""
        self._loc_job: Optional[threading.Thread] = None
        self._loc_job_done: Optional[threading.Event] = None
        self._loc_job_result: Optional[dict] = None
        self._loc_job_gen = 0
        self._loc_settle_until = 0.0
        self._loc_next_step = ""
        self._loc_next_kwargs: dict = {}
        self._loc_refine_force_stop = False
        self._loc_probe_severe = False
        self._loc_probe_calm = False
        self._loc_verdict = None
        self._world = world

        nose_offset = cfg.nose_offset_m()
        half_track = cfg.wheel_half_track_m()
        robot_radius_m = cfg.hard_clearance_radius_m()
        body_radius_m = cfg.inscribed_radius_m()

        # Rectangular collision guard: exact body; clearance_m sets keep-out pad.
        self._guard: Optional[FootprintGuard] = None
        if footprint_guard:
            clearance = max(0.0, float(robot_radius_m) - float(body_radius_m))
            self._guard = FootprintGuard(
                GuardConfig(
                    length_m=2.0 * nose_offset,
                    width_m=2.0 * float(body_radius_m),
                    padding_m=min(max(clearance, 0.04), 0.15),
                )
            )
        self._depth_memory = DepthObstacleMemory(
            length_m=2.0 * nose_offset, width_m=2.0 * float(body_radius_m)
        )
        self._above_memory: Optional[AboveCartMemory] = None
        if cfg.avoid_obstacles_above_cart:
            cart_h = float(cfg.cart_height_m) if cfg.cart_height_m else 0.0
            self._above_memory = AboveCartMemory(
                cart_height_m=cart_h if cart_h > 0.0 else None
            )
        setter = getattr(self._world, "set_above_cart", None)
        if callable(setter):
            cart_h = float(cfg.cart_height_m) if cfg.cart_height_m else 0.0
            setter(
                self._above_memory is not None,
                cart_h if cart_h > 0.0 else None,
            )

        self._local_planner = LocalPlannerConfig(
            enabled=b.local_planner_enabled,
            sim_time_s=b.local_planner_sim_time_s,
            activate_cost_threshold=b.local_planner_activate_cost,
            max_detour_forward_mps=b.local_planner_max_vel_x_mps,
            max_vel_x_reverse_m=b.local_planner_max_vel_x_reverse_m,
        )
        self._local_costmap = (
            LocalCostmap(
                LocalCostmapConfig(
                    width_m=b.local_costmap_width_m,
                    height_m=b.local_costmap_height_m,
                    resolution=b.local_costmap_resolution,
                    inflation_radius_m=cfg.effective_local_inflation_radius_m(),
                    robot_radius_m=robot_radius_m,
                    body_radius_m=body_radius_m,
                    cost_scaling_factor=b.cost_scaling_factor,
                )
            )
            if b.local_costmap_enabled
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
        self._local_update_period_s = 1.0 / float(b.local_costmap_rate_hz)
        self._local_scan_max_age_s = min(float(cfg.simple_scan_max_age), 0.5)
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
        # After accepting a longer detour, ban the abandoned short corridor so
        # we don't thrash short↔long. A small per-goal budget allows flipping
        # back when the short corridor looks clear again (e.g. person left /
        # blocked the long path). After the budget is spent, lift only if the
        # detour itself dies and no other route works.
        self._detour_ban_path: Optional[Path2D] = None
        self._detour_min_length_m = 0.0
        self._detour_short_ratio = 0.75
        self._detour_short_flips_left = b.detour_short_flip_budget
        self._io_timeout_streak = 0
        # Control-loop soak metrics (wall-clock tick vs configured period).
        self._control_ticks = 0
        self._control_overruns = 0
        self._control_tick_last_s: Optional[float] = None
        self._control_tick_ema_s: Optional[float] = None
        self._control_period_s = max(1e-3, float(cfg.control_period_s()))
        # Last tick's sense/costmap/guard/dwa/gate wall times (ms).
        self._phase_ms_last: dict[str, float] = {}

        stop_m = max(float(cfg.simple_stop_distance), nose_offset + NOSE_STOP_MARGIN_M)
        self._follower = FollowerConfig(
            lookahead_m=b.lookahead_m,
            min_lookahead_m=b.min_lookahead_m,
            max_lookahead_m=b.max_lookahead_m,
            approach_dist_m=b.approach_dist_m,
            waypoint_tolerance_m=max(0.1, b.xy_goal_tolerance),
            # Keeps translating arcs above the base's inner-wheel "nearly 0 RPM"
            # rejection. From the footprint width when configured, else ≈0.6·r.
            wheel_half_track_m=max(0.08, half_track),
            max_linear_accel_mps2=b.max_linear_accel_mps2,
            max_linear_decel_mps2=b.max_linear_decel_mps2,
            max_angular_accel_rad_s2=b.max_angular_accel_rad_s2,
            motion=SimpleMotionConfig(
                poll_interval_s=cfg.control_period_s(),
                xy_tolerance_m=b.xy_goal_tolerance,
                yaw_tolerance_rad=b.yaw_goal_tolerance,
                default_linear_mps=cfg.max_linear_velocity,
                max_linear_mps=cfg.max_linear_velocity,
                max_angular_rad_s=cfg.max_vel_theta,
                min_linear_mps=cfg.min_cmd_vel_x,
                min_angular_rad_s=cfg.min_cmd_vel_theta,
                timeout_s=b.timeout_s,
            ),
            obstacle=ObstacleConfig(
                enabled=cfg.simple_avoid_obstacles,
                # Lidar stop must clear the bumper, which is a half-*length* out
                # from the centre — not a half-diagonal disc, which stopped a
                # 0.72 m robot 0.5 m short of everything.
                stop_distance_m=stop_m,
                slow_distance_m=max(
                    float(cfg.simple_slow_distance),
                    stop_m + 0.35,
                ),
                # Anything inside the body's swept corridor counts as "ahead" —
                # the ±35° cone alone let shoulder-side bins slide past.
                # Padding beyond the inscribed radius covers light shoulder
                # grazes (half-width + ~12 cm) without sealing every doorway.
                footprint_half_width_m=float(robot_radius_m) + AHEAD_HALF_WIDTH_PAD_M,
                max_age_s=cfg.simple_scan_max_age,
            )
            if cfg.simple_avoid_obstacles
            else ObstacleConfig(enabled=False),
        )

    # -- Config accessors (read ``self._cfg``; no kwargs copy) ----------------
    @property
    def _inflation(self) -> float:
        return self._cfg.effective_inflation_radius_m()

    @property
    def _robot_radius(self) -> float:
        return self._cfg.hard_clearance_radius_m()

    @property
    def _body_radius(self) -> float:
        return self._cfg.inscribed_radius_m()

    @property
    def _spin_radius(self) -> float:
        return max(
            self._cfg.hard_clearance_radius_m(),
            self._cfg.circumscribed_radius_m(),
        )

    @property
    def _cost_scaling(self) -> float:
        return self._cfg.builtin.cost_scaling_factor

    @property
    def _clearance_preference_m(self) -> float:
        return self._cfg.builtin.clearance_preference_m

    @property
    def _yaw_align_timeout_s(self) -> float:
        return self._cfg.builtin.yaw_align_timeout_s

    @property
    def _max_goal_snap_m(self) -> float:
        return self._cfg.builtin.max_goal_snap_m

    @property
    def _algorithm(self) -> str:
        return self._cfg.builtin.planner

    @property
    def _replan_period(self) -> float:
        return self._cfg.builtin.replan_period_s

    @property
    def _timeout_s(self) -> float:
        return self._cfg.builtin.timeout_s

    @property
    def _max_vel_x(self) -> float:
        return self._cfg.max_linear_velocity

    @property
    def _scan_max_age(self) -> float:
        return self._cfg.simple_scan_max_age

    @property
    def _smooth_path(self) -> bool:
        return self._cfg.builtin.smooth_path

    @property
    def _smooth_spacing(self) -> float:
        return self._cfg.builtin.smooth_sample_spacing_m

    @property
    def _local_costmap_enabled(self) -> bool:
        return self._cfg.builtin.local_costmap_enabled

    @property
    def _drive_timeout_streak(self) -> int:
        return self._cfg.builtin.drive_timeout_streak

    @property
    def _backup_enabled(self) -> bool:
        return self._cfg.builtin.backup_enabled

    @property
    def _backup_stuck_time_s(self) -> float:
        return self._cfg.builtin.backup_stuck_time_s

    @property
    def _backup_dist_m(self) -> float:
        return self._cfg.builtin.backup_dist_m

    @property
    def _backup_speed_mps(self) -> float:
        return self._cfg.builtin.backup_speed_mps

    @property
    def _backup_rear_clear_m(self) -> float:
        return self._cfg.builtin.backup_rear_clear_m

    @property
    def _backup_max_attempts(self) -> int:
        return self._cfg.builtin.backup_max_attempts

    @property
    def _backup_cooldown_s(self) -> float:
        return self._cfg.builtin.backup_cooldown_s

    @property
    def _recovery_wait_duration_s(self) -> float:
        return self._cfg.builtin.recovery_wait_duration_s

    @property
    def _replan_local_blocked_time_s(self) -> float:
        return self._cfg.builtin.replan_local_blocked_time_s

    @property
    def _replan_local_min_period_s(self) -> float:
        return self._cfg.builtin.replan_local_min_period_s

    @property
    def _replan_max_early_detour_ratio(self) -> float:
        return self._cfg.builtin.replan_max_early_detour_ratio

    @property
    def _replan_allow_long_detour_after_fails(self) -> int:
        return self._cfg.builtin.replan_allow_long_detour_after_fails

    @property
    def _detour_short_flip_budget(self) -> int:
        return self._cfg.builtin.detour_short_flip_budget

    @property
    def _local_planner_activate_cost(self) -> int:
        return self._cfg.builtin.local_planner_activate_cost

    @property
    def _avoid_above(self) -> bool:
        return self._cfg.avoid_obstacles_above_cart

    @property
    def _cart_height_m(self) -> float:
        return float(self._cfg.cart_height_m) if self._cfg.cart_height_m else 0.0

    @property
    def _nav_loc_refine(self) -> bool:
        return self._cfg.builtin.nav_loc_refine_on_disagree

    @property
    def _nav_loc_refine_margin_m(self) -> float:
        return self._cfg.builtin.nav_loc_refine_margin_m

    @property
    def _nav_loc_refine_map_max_m(self) -> float:
        return self._cfg.builtin.nav_loc_refine_map_max_m

    @property
    def _nav_loc_refine_lidar_min_m(self) -> float:
        return self._cfg.builtin.nav_loc_refine_lidar_min_m

    @property
    def _nav_loc_refine_min_frac(self) -> float:
        return self._cfg.builtin.nav_loc_refine_min_frac

    @property
    def _nav_loc_refine_min_beams(self) -> int:
        return self._cfg.builtin.nav_loc_refine_min_beams

    @property
    def _nav_loc_refine_max_tries(self) -> int:
        return self._cfg.builtin.nav_loc_refine_max_tries

    @property
    def _nav_loc_refine_cooldown_s(self) -> float:
        return self._cfg.builtin.nav_loc_refine_cooldown_s

    @property
    def _nav_loc_refine_period_s(self) -> float:
        return self._cfg.builtin.nav_loc_refine_period_s

    @property
    def _nav_loc_refine_check_every_m(self) -> float:
        return self._cfg.builtin.nav_loc_refine_check_every_m

    @property
    def _nav_loc_refine_retry_travel_m(self) -> float:
        return self._cfg.builtin.nav_loc_refine_retry_travel_m

    @property
    def _nav_loc_refine_apply_max_m(self) -> float:
        return self._cfg.builtin.nav_loc_refine_apply_max_m

    @property
    def _nav_loc_refine_apply_max_deg(self) -> float:
        return self._cfg.builtin.nav_loc_refine_apply_max_deg

    @property
    def _nav_loc_refine_apply_min_score(self) -> float:
        return self._cfg.builtin.nav_loc_refine_apply_min_score

    @property
    def _nav_loc_refine_settle_s(self) -> float:
        return self._cfg.builtin.nav_loc_refine_settle_s

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
            "phase_ms": dict(self._phase_ms_last) if self._phase_ms_last else None,
        }

    def _note_phase_ms(self, phase_ms: dict) -> None:
        """Stash the latest per-phase tick timings for ``control_stats`` / GetStatus."""
        self._phase_ms_last = {str(k): float(v) for k, v in phase_ms.items()}

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
                from ..planning.costmap import build_costmap, occupancy_from_map_dict
                from ..types import OccupancyGrid

                occ = occupancy_from_map_dict(map_data)
                masks = self._zone_masks(map_data)
                if masks is not None and masks.keepout.shape == occ.grid.shape:
                    from ..stores.zones import apply_keepout_to_grid

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
        threading.Thread(target=_job, name="nav-global-costmap", daemon=True).start()

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

    def _speed_pct_at(self, pose: Pose2D) -> Optional[float]:
        masks = self._zone_masks()
        if masks is None:
            return None
        return masks.speed_pct_at(pose.x, pose.y)

    def _lift_sub_deadband(
        self, cmd: DriveCommand, forward_clearance_m: Optional[float] = None
    ) -> DriveCommand:
        """Make a nonzero twist survive ``sanitize_base_cmd`` (shared envelope)."""
        guard = getattr(self, "_guard", None)
        body_len = guard.cfg.length_m if guard is not None else None
        lifted = lift_sub_deadband(
            cmd,
            forward_clearance_m=forward_clearance_m,
            body_length_m=body_len,
        )
        if lifted is not cmd:
            self._last_sent_cmd = lifted
        return lifted

    def _apply_speed_zone(self, pose: Pose2D, cmd: DriveCommand) -> DriveCommand:
        """Cap linear speed at the zone maximum. Slower commands pass through."""
        pct = self._speed_pct_at(pose)
        if pct is None or pct >= 100.0 or cmd.done:
            return cmd
        vx, vy, vtheta = apply_speed_zone_limit(
            cmd.vx,
            cmd.vy,
            cmd.vtheta,
            pct,
            max_linear_mps=self._max_vel_x,
        )
        return DriveCommand(vx, vy, vtheta, cmd.done)

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

    def _publish_plan_viz(
        self, result: PlanResult, goal: Pose2D, start: Optional[Pose2D]
    ) -> None:
        preview = result.to_preview_dict(goal=(goal.x, goal.y, goal.theta), start=start)
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

    def _call_check_localization(
        self,
        *,
        apply: Optional[bool] = None,
        stopped_recovery: bool = False,
    ) -> Optional[dict]:
        fn = getattr(self._world, "check_localization", None)
        if not callable(fn):
            return None
        try:
            result = fn(
                allow_during_navigation=True,
                full_map_escalation="still_bad",
                apply=apply,
                stopped_recovery=bool(stopped_recovery),
            )
        except TypeError:
            # Older WorldIO doubles without the newer kwargs.
            try:
                result = fn(
                    allow_during_navigation=True,
                    full_map_escalation="still_bad",
                    apply=apply,
                )
            except TypeError:
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
                return {
                    "status": "error",
                    "reason": str(exc).strip() or type(exc).__name__,
                }
        except TimeoutError:
            return {"status": "error", "reason": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc).strip() or type(exc).__name__}
        return result if isinstance(result, dict) else None

    def _launch_loc_check(self, step: str, **kwargs: Any) -> str:
        """Start a localization match without waiting for it."""
        self._loc_step = step
        self._loc_job_gen += 1
        gen = self._loc_job_gen
        done = threading.Event()
        self._loc_job_done = done
        self._loc_job_result = None

        def _run() -> None:
            try:
                result = self._call_check_localization(**kwargs)
            except Exception as exc:  # noqa: BLE001 - worker must not die silently
                result = {
                    "status": "error",
                    "reason": str(exc).strip() or type(exc).__name__,
                }
            if gen != self._loc_job_gen:
                return
            self._loc_job_result = result if isinstance(result, dict) else None
            done.set()

        self._loc_phase = "checking"
        job = threading.Thread(target=_run, name="loc-refine", daemon=True)
        self._loc_job = job
        job.start()
        return "hold" if self._loc_refine_force_stop else None

    def _take_loc_check_result(self) -> Optional[dict]:
        """Return a finished match, or None if it is still running."""
        done = self._loc_job_done
        if self._loc_phase != "checking" or done is None or not done.is_set():
            return None
        result = self._loc_job_result
        self._loc_phase = "idle"
        self._loc_job = None
        self._loc_job_done = None
        self._loc_job_result = None
        return result

    def _schedule_loc_check(self, step: str, **kwargs: Any) -> str:
        """Run ``step`` after the settle window, without sleeping here."""
        self._loc_next_step = step
        self._loc_next_kwargs = dict(kwargs)
        wait = float(self._nav_loc_refine_settle_s)
        if wait <= 0.0:
            return self._launch_loc_check(step, **kwargs)
        self._loc_settle_until = time.monotonic() + wait
        self._loc_phase = "settling"
        return "hold"

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

        Returns ``None`` (keep following), ``hold`` (command zero this tick),
        ``resume`` (pose moved: replan), or ``continue`` (pose unchanged:
        keep the current path). Does not fail the goal — a leftover residual
        or refused yank is common in hallways.

        The match itself never runs here. A straight, already-moving base may
        *start* a mild check without stopping; obstacle avoidance keeps
        commanding while that match runs. Severe disagreement always stops.
        Stopping is also required for a hard turn and for applying a pose
        correction. An unfixed *thin* residual waits ``retry_travel_m``
        before the next check; a still-lost pose keeps refusing forward.
        """
        if not self._nav_loc_refine:
            return None
        if self._loc_phase == "checking":
            result = self._take_loc_check_result()
            if result is None:
                return "hold" if self._loc_refine_force_stop else None
            return self._loc_refine_consume(pose, now, dist_goal, result)
        if self._loc_phase == "settling":
            if now < self._loc_settle_until:
                return "hold"
            self._loc_phase = "idle"
            return self._launch_loc_check(self._loc_next_step, **self._loc_next_kwargs)
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

        from .loc_consistency import (
            loc_blocks_forward,
        )

        verdict = self._measure_loc_disagreement(pose)
        if not verdict.disagree:
            self._loc_refine_need_travel = False
            if holding:
                # Anti-thrash: one fleeting clear frame must not resume drive.
                self._loc_refine_clear_streak += 1
                if self._loc_refine_clear_streak < self._loc_refine_clear_needed:
                    self._publish_loc_refine_progress(
                        pose, dist_goal, verdict=verdict, status="clearing"
                    )
                    return "hold"
                self._loc_refine_clear_streak = 0
                self._loc_refine_tries = 0
                return self._loc_refine_resume_kind(self._loc_refine_start_pose, pose)
            self._loc_refine_clear_streak = 0
            return None
        self._loc_refine_clear_streak = 0
        if not holding:
            self._loc_refine_start_pose = pose

        # SLAM skips the match above ~0.35 rad/s of yaw. Under that, a mild
        # cruise check is fine. Severe unexplained pose always stops first.
        calm = self._loc_refine_yaw_calm()
        severe = loc_blocks_forward(verdict)
        self._loc_probe_calm = calm
        self._loc_probe_severe = severe
        self._loc_verdict = verdict
        self._loc_refine_force_stop = bool(severe or not calm)
        if self._loc_refine_force_stop:
            self._pause_for_loc_refine(pose, dist_goal, verdict, settle=False)
        # Hard yaw: wait out the settle window before matching, but do not
        # sleep on this thread. Mild or already-stopped checks start now.
        if not calm and self._nav_loc_refine_settle_s > 0.0:
            return self._schedule_loc_check("probe")
        return self._launch_loc_check("probe")

    def _loc_refine_consume(
        self, pose: Pose2D, now: float, dist_goal: float, result: Optional[dict]
    ) -> Optional[str]:
        """Apply one finished match and maybe start the next. Never blocks."""
        from .loc_consistency import (
            loc_blocks_forward,
            residual_is_lost,
            slam_pose_is_trusted,
            small_local_match_worth_applying,
        )

        step = self._loc_step
        verdict = self._loc_verdict
        calm = self._loc_probe_calm
        severe = self._loc_probe_severe
        if result is None:
            return "hold" if self._loc_refine_tries > 0 else None
        status = str(result.get("status") or "")
        if status == "awaiting_confirm" and step in ("probe", "recovery"):
            self._loc_refine_force_stop = True
            if calm or step == "recovery":
                self._pause_for_loc_refine(pose, dist_goal, verdict, settle=False)
            self._publish_loc_refine_progress(
                pose, dist_goal, verdict=verdict, status="awaiting_confirm"
            )
            nxt = "confirm" if step == "probe" else "recovery_confirm"
            kwargs = {"stopped_recovery": True} if step == "recovery" else {}
            return self._schedule_loc_check(nxt, **kwargs)

        if step in ("probe", "confirm") and small_local_match_worth_applying(
            result,
            max_shift_m=self._nav_loc_refine_apply_max_m,
            max_shift_deg=self._nav_loc_refine_apply_max_deg,
            min_score=self._nav_loc_refine_apply_min_score,
        ):
            self._loc_refine_force_stop = True
            if calm:
                self._pause_for_loc_refine(pose, dist_goal, verdict, settle=False)
            self._publish_loc_refine_progress(
                pose, dist_goal, verdict=verdict, status="applying"
            )
            return self._launch_loc_check("apply", apply=True)

        if (
            step == "recovery"
            and (status not in ("corrected", "ok") or not result.get("corrected"))
            and bool(result.get("good_match"))
        ):
            return self._launch_loc_check(
                "recovery_apply", apply=True, stopped_recovery=True
            )

        if status in ("skipped", "unconfigured") and step in (
            "probe",
            "confirm",
            "apply",
        ):
            reason = str(result.get("reason") or "")
            if reason in ("spinning", "stale_scan"):
                if calm:
                    self._loc_refine_force_stop = True
                    self._world.stop()
                    self._note_base_stopped()
                return "hold"
            return "hold" if self._loc_refine_tries > 0 else None

        pose_after = self._world.get_pose() or pose
        after = self._measure_loc_disagreement(pose_after)
        start = self._loc_refine_start_pose or pose
        block_forward = loc_blocks_forward(after, result)
        if (
            block_forward
            and not self._loc_refine_recovery_used
            and step not in ("recovery", "recovery_confirm", "recovery_apply")
        ):
            if calm or not severe:
                self._loc_refine_force_stop = True
                self._pause_for_loc_refine(pose_after, dist_goal, after, settle=False)
            self._loc_refine_recovery_used = True
            self._publish_loc_refine_progress(
                pose_after, dist_goal, verdict=after, status="recovering"
            )
            return self._launch_loc_check("recovery", stopped_recovery=True)

        if block_forward:
            # Recovery already ran (or this result is the recovery). Do not
            # park for the rest of the goal: a refused jump is not applied,
            # but staying stopped cannot improve the match. Drive with
            # avoidance and look again after ``retry_travel_m``.
            self._loc_refine_tries = 0
            self._loc_refine_force_stop = False
            self._loc_refine_need_travel = True
            self._loc_refine_travel_need_m = self._nav_loc_refine_retry_travel_m
            self._loc_refine_clear_streak = 0
            self._loc_refine_cooldown_until = (
                time.monotonic() + self._nav_loc_refine_cooldown_s
            )
            self._publish_loc_refine_progress(
                pose_after, dist_goal, verdict=after, status="continue_unfixed"
            )
            return self._loc_refine_resume_kind(start, pose_after)
        if after.disagree:
            self._loc_refine_tries += 1
            self._loc_refine_cooldown_until = (
                time.monotonic() + self._nav_loc_refine_cooldown_s
            )
            self._loc_refine_force_stop = False
            if slam_pose_is_trusted(result):
                self._loc_refine_tries = 0
                self._loc_refine_need_travel = True
                self._loc_refine_travel_need_m = self._nav_loc_refine_retry_travel_m
                self._loc_refine_clear_streak = 0
                self._publish_loc_refine_progress(
                    pose_after, dist_goal, verdict=after, status="continue_trusted"
                )
                return self._loc_refine_resume_kind(start, pose_after)
            if (
                self._loc_refine_tries < self._nav_loc_refine_max_tries
                and residual_is_lost(after)
            ):
                self._loc_refine_force_stop = True
                self._publish_loc_refine_progress(
                    pose_after, dist_goal, verdict=after, status="retry"
                )
                return "hold"
            self._loc_refine_tries = 0
            self._loc_refine_need_travel = True
            self._loc_refine_travel_need_m = self._nav_loc_refine_retry_travel_m
            self._publish_loc_refine_progress(
                pose_after, dist_goal, verdict=after, status="continue"
            )
            return self._loc_refine_resume_kind(start, pose_after)

        self._loc_refine_tries = 0
        self._loc_refine_need_travel = False
        self._loc_refine_clear_streak = 0
        self._loc_refine_force_stop = False
        self._loc_refine_cooldown_until = (
            time.monotonic() + self._nav_loc_refine_cooldown_s
        )
        status_name = "recovered" if step.startswith("recovery") else "resumed"
        self._publish_loc_refine_progress(
            pose_after, dist_goal, verdict=after, status=status_name
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
        # Settle is a deadline polled by the control loop. Sleeping here used
        # to freeze obstacle avoidance for the whole pause.

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
        from ..planning.local_costmap import footprint_collides
        from ..planning.local_planner import path_block_distance_m, path_point_ahead
        from ..planning.planner import _merge_path_prefix
        from .controller import _path_length

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
        from ..planning.costmap import LETHAL, is_hard

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

        ``wait``: freeze for dynamic crossers (person in front for a second).
        Applies for the grace window even if the nose just cleared — leftover
        lethal in the costmap must not immediately DWA hunt-and-peck / spin.
        ``keep_dwa``: after grace, clear forward cone — stay on the short
        global path and peel (doorway / static blob). If peeling makes no
        progress for ``peel_stuck_limit_s``, escalate.
        ``replan``: grace elapsed + blocked nose, or clear-nose peel stuck.
        """
        # People crossing: stop first. Resume on the same path when clear.
        if blocked_for_s < wait_before_replan_s:
            return "wait"
        if nose_clear:
            if (
                peel_stuck_limit_s > 0.0
                and peel_stuck_s >= peel_stuck_limit_s
                and replan_cooldown_ready
            ):
                return "replan"
            return "keep_dwa"
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
        heading_open: bool = False,
    ) -> str:
        """Motion while a blocked nose waits out the replan cooldown."""
        return blocked_nose_unstick(
            failed_replans=failed_replans,
            nose_clear=nose_clear,
            rear_open=rear_open,
            spin_clear=spin_clear,
            heading_open=heading_open,
        )

    def _corridor_escape_yaw(
        self,
        pose: Pose2D,
        scan: conv.LaserScan2D,
        goal: Pose2D,
        path: Optional[Path2D] = None,
        *,
        commit_heading: Optional[float] = None,
        scan_world_pts: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Yaw (rad, +CCW) onto a scan corridor the rectangle can enter.

        Prefers the path lookahead. The straight-line goal can point through
        the obstacle while the route is open the other way, and a few degrees
        toward that goal used to beat a wide opening on the path. Once a
        corridor is picked, ``commit_heading`` (its map heading) is preferred
        while it stays open.

        None when no heading within 90° of the nose has ``CORRIDOR_MIN_TRAVEL_M``
        of body-width clearance, or the rectangle cannot spin onto it. Uses
        raw returns, so inflated costmap cells cannot veto a geometrically
        clear turn.
        """
        guard = self._guard
        if guard is None:
            return None
        body = scan.to_points()
        if body.size == 0:
            return None
        if commit_heading is not None:
            bearing = conv.normalize_angle(commit_heading - pose.theta)
        else:
            bearing = self._corridor_prefer_rad(pose, goal, path)
        yaw = open_corridor_yaw(
            body,
            half_length_m=guard._hl,
            half_width_m=guard._hw + guard.cfg.padding_m,
            prefer_rad=bearing,
            min_travel_m=CORRIDOR_MIN_TRAVEL_M,
            max_yaw_rad=CORRIDOR_MAX_YAW_RAD,
        )
        if yaw is None or abs(yaw) < 1e-3:
            return yaw
        world = obstacle_points(
            pose,
            scan,
            None,
            radius_m=guard.cfg.obstacle_radius_m,
            scan_world_pts=scan_world_pts,
        )
        free = guard.free_rotation(pose, math.copysign(1.0, yaw), world)
        # The next slice of the sweep has to be clear. Requiring the whole
        # target up front refuses a notch whose first degrees clip padding;
        # each tick rechecks and stops when the rectangle meets a return.
        need = min(abs(yaw), max(guard.cfg.rot_step_rad * 2.0, 0.12))
        if free + 1e-3 < need:
            return None
        return float(yaw)

    def _corridor_prefer_rad(
        self,
        pose: Pose2D,
        goal: Pose2D,
        path: Optional[Path2D],
    ) -> float:
        """Body-frame bearing of the route, else the straight-line goal."""
        follower = getattr(self, "_follower", None)
        if path is not None and follower is not None and len(path.points) >= 2:
            target, _, _ = lookahead_pose(
                pose,
                path,
                lookahead_m=float(follower.lookahead_m),
                waypoint_tolerance_m=float(follower.waypoint_tolerance_m),
            )
            return conv.normalize_angle(
                math.atan2(target.y - pose.y, target.x - pose.x) - pose.theta
            )
        return conv.normalize_angle(
            math.atan2(goal.y - pose.y, goal.x - pose.x) - pose.theta
        )

    def _corridor_motion(
        self,
        pose: Pose2D,
        scan: Optional[conv.LaserScan2D],
        yaw: float,
        *,
        scan_world_pts: Optional[np.ndarray] = None,
    ) -> tuple[DriveCommand, str]:
        """Twist that takes the scan corridor without a full stop when it can.

        A few degrees is a creep with yaw, the way Nav2 regulated pure
        pursuit closes a carrot bearing under ~45°. The arc is checked on
        the raw scan; if the nose is already touching, fall back to an
        in-place spin so the guard does not zero the command.
        """
        sign = math.copysign(1.0, yaw if abs(yaw) > 1e-6 else 1.0)
        spin = DriveCommand(0.0, 0.0, sign * _NOSE_UNSTICK_YAW_RATE, False)
        if abs(yaw) < CORRIDOR_ALIGN_RAD:
            return DriveCommand(CORRIDOR_DRIVE_MPS, 0.0, 0.0, False), "corridor"
        if abs(yaw) >= CORRIDOR_SPIN_RAD:
            return spin, "yaw_open"
        w = math.copysign(min(0.35, max(0.12, abs(yaw) / 0.5)), yaw)
        arc = DriveCommand(CORRIDOR_DRIVE_MPS, 0.0, w, False)
        guard = self._guard
        if guard is None or scan is None:
            return arc, "corridor"
        pts = obstacle_points(
            pose,
            scan,
            None,
            radius_m=guard.cfg.obstacle_radius_m,
            scan_world_pts=scan_world_pts,
        )
        free = guard.free_distance(pose, arc.vx, arc.vtheta, pts, guard.cfg.horizon_m)
        if guard._regulate(arc.vx, free) >= LIN_EPS_MPS:
            return arc, "corridor"
        return spin, "yaw_open"

    @staticmethod
    def _defer_periodic_replan(
        *,
        static_blocked: bool,
        nose_clear: bool,
        corridor_open: bool,
        pose_jumped: bool,
        osc_replan: bool,
        spin_rock: bool,
        backup_exhausted: bool,
    ) -> bool:
        """Skip a blocking global replan the scan can already drive through.

        Nav2's controller keeps publishing velocity while ComputePathToPose
        fails. An inscribed centerline stays ``static_blocked`` for a whole
        corridor, and zeroing the base for that search every second is the
        start/stop. Still replan on a pose jump, a rock, or a real stuck
        nose with no scan opening.
        """
        if not static_blocked:
            return False
        if pose_jumped or osc_replan or spin_rock or backup_exhausted:
            return False
        return bool(nose_clear) or bool(corridor_open)

    def _published_pose_misregistered(self) -> bool:
        """Map walls are at the wrong place: a large scan match was refused."""
        from .loc_consistency import published_pose_misregistered

        getter = getattr(self._world, "get_localization_check", None)
        if not callable(getter):
            return False
        try:
            check = getter()
        except Exception:  # noqa: BLE001 - a status read must not stop the tick
            return False
        return published_pose_misregistered(check)

    @staticmethod
    def _bumper_spin_reverse(
        *,
        nose_clear: bool,
        spin_blocked: bool,
        cmd_vx: float,
        cmd_vtheta: float,
        rear_open: bool,
    ) -> bool:
        """Back up when the bumper is against an obstacle and a spin would hit it."""
        return bumper_wants_reverse(
            nose_clear=nose_clear,
            spin_blocked=spin_blocked,
            cmd_vx=cmd_vx,
            cmd_vtheta=cmd_vtheta,
            rear_open=rear_open,
        )

    @staticmethod
    def _undrivable_nose_recovery(
        *,
        nose_clear: bool,
        waiting: bool,
        vx: float,
        vtheta: float,
        rear_open: bool,
        spin_clear: bool,
        bearing_rad: float,
    ) -> Optional[str]:
        """Motion when the only forward command would be dropped by the base."""
        return undrivable_nose(
            nose_clear=nose_clear,
            waiting=waiting,
            vx=vx,
            vtheta=vtheta,
            rear_open=rear_open,
            spin_clear=spin_clear,
            bearing_rad=bearing_rad,
        )

    def _abort_blocked_replans(self, count: int, *, nose_clear: bool) -> bool:
        """Fail the goal once blocked-nose replans are exhausted."""
        if count < BLOCKED_REPLAN_FAIL_LIMIT or nose_clear:
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
                f"path blocked: no route around obstacle ({count} replans failed)"
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
        - On early blocked-replans, refuse candidates longer than
          ``replan_max_early_detour_ratio`` × remaining so a brief person-block
          keeps waiting / local peel instead of a room-scale loop. After enough
          failed replans (or force-lift), longer detours are allowed.
        - Cap length on the blocked-corridor attempt when a milder peel already
          exists so we prefer the peel.
        - If every attempt is same-route/infeasible, force left/right vias
          around the first blocked path sample — that is the static-box case
          where Lazy Theta* stubbornly hugs the old corridor.
        - After accepting a longer detour, ban the abandoned short corridor:
          keep painting it and reject much-shorter / matching candidates.
          A small per-goal short-flip budget allows returning to a clear short
          corridor a few times; after that, lift only when the detour is dead.
        """
        from ..planning.local_planner import path_cost_in_local_window
        from ..planning.path_utils import closest_point_on_path
        from .controller import _path_length

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
        early_ratio = float(self._replan_max_early_detour_ratio)
        allow_long = (
            hold_active
            or force_lift_short_flip
            or failed_count >= self._replan_allow_long_detour_after_fails
        )
        early_limit = remaining * early_ratio

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
            reason: Optional[str] = None
            if self._detour_min_length_m > 0.0 and new_len < self._detour_min_length_m:
                reason = (
                    f"short-flip hysteresis ({new_len:.1f} m < "
                    f"{self._detour_min_length_m:.1f} m ban)"
                )
            elif ban_path is not None and not paths_meaningfully_differ(
                ban_path, candidate, tol_m=differ_tol
            ):
                reason = "short-flip hysteresis (matches abandoned corridor)"
            if reason is None:
                return None
            # Candidate already passed local-blocked → short corridor looks
            # clear. Spend one budgeted flip instead of staying on the long path.
            if self._detour_short_flips_left > 0:
                return None
            return reason

        def _reject_too_long(new_len: float, *, paint: bool) -> Optional[str]:
            if not allow_long and new_len > early_limit:
                return (
                    f"too long early ({new_len:.1f} m > {early_limit:.1f} m; "
                    f"fail={failed_count}/"
                    f"{self._replan_allow_long_detour_after_fails})"
                )
            # Prefer a milder peel when one already exists.
            if paint and best is not None and new_len > early_limit:
                return f"too long ({new_len:.1f} m > {early_limit:.1f} m)"
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
                reasons.append(f"{label}: still local-blocked (cost={stuck_cost})")
                continue
            new_len = _path_length(replanned.path)
            flip = _reject_short_flip(replanned.path, new_len)
            if flip is not None:
                reasons.append(f"{label}: {flip}")
                continue
            too_long = _reject_too_long(new_len, paint=paint)
            if too_long is not None:
                reasons.append(f"{label}: {too_long}")
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
                    reasons.append(f"{label}: still local-blocked (cost={stuck_cost})")
                else:
                    new_len = _path_length(new_path)
                    flip = _reject_short_flip(new_path, new_len)
                    if flip is not None:
                        reasons.append(f"{label}: {flip}")
                    else:
                        too_long = _reject_too_long(new_len, paint=False)
                        if too_long is not None:
                            reasons.append(f"{label}: {too_long}")
                        else:
                            self._commit_replan_path(
                                path,
                                new_path,
                                new_len,
                                old_len,
                                label,
                                reasons,
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
            if (
                hold_active
                and allow_lift_ban
                and (not ban_was_only_reason or force_lift_short_flip)
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
            path,
            new_path,
            new_len,
            old_len,
            label,
            reasons,
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
        """Publish an accepted replan and update detour ban / short-flip budget."""
        self._last_replan_error = ""
        hold = self._detour_ban_path is not None
        short_flip = False
        if hold:
            matches_ban = (
                self._detour_ban_path is not None
                and not paths_meaningfully_differ(
                    self._detour_ban_path, new_path, tol_m=0.12
                )
            )
            under_min = (
                self._detour_min_length_m > 0.0 and new_len < self._detour_min_length_m
            )
            if matches_ban or under_min:
                # Budgeted return to the abandoned short corridor.
                short_flip = True
                if self._detour_short_flips_left > 0:
                    self._detour_short_flips_left -= 1
                self._detour_ban_path = None
                self._detour_min_length_m = 0.0
                hold = False
        # Commit to a longer / different corridor (ban abandoned short path).
        if (
            not short_flip
            and new_len > old_len * 1.15
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
            "short_flip": bool(short_flip),
            "short_flips_left": int(self._detour_short_flips_left),
            "attempts": list(reasons),
        }
        preview = self._publish_plan_viz(result, goal, start=pose)
        self._set_status(path=preview["path"], length_m=preview["length_m"])

    def _replan_blocked_at_start(self) -> bool:
        err = self._last_replan_error or ""
        return "cannot reach plan start" in err or "start pose is in lethal" in err

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
        self._detour_short_flips_left = self._detour_short_flip_budget
        self._loc_refine_tries = 0
        self._loc_refine_cooldown_until = 0.0
        self._loc_refine_last_check = 0.0
        self._loc_refine_last_pose = None
        self._loc_refine_start_pose = None
        self._loc_refine_need_travel = False
        self._loc_refine_recovery_used = False
        self._loc_refine_clear_streak = 0
        self._loc_job_gen += 1
        self._loc_phase = "idle"
        self._loc_refine_force_stop = False
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

            _now0 = time.monotonic()
            ep = GoalEpisode(
                deadline=_now0
                + self._goal_timeout_s(float(preview.get("length_m") or 0.0)),
                last_replan=_now0,
                last_progress_at=_now0,
            )

            while time.monotonic() < ep.deadline:
                tick_started = time.monotonic()
                if self._cancel.is_set():
                    self._world.stop()
                    self._set_status(
                        state="canceled", active=False, error_msg="canceled"
                    )
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
                    ep.blocked_fail_dist is not None
                    and distance_m(pose, goal) < ep.blocked_fail_dist - 0.5
                ):
                    ep.on_goal_progress()

                self._set_status(pose={"x": pose.x, "y": pose.y, "theta": pose.theta})

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
                    give_up, ep.align_best_err, ep.align_progress_at = (
                        yaw_align_give_up(
                            now,
                            yaw_err_abs,
                            self._yaw_align_timeout_s,
                            ep.align_best_err,
                            ep.align_progress_at,
                        )
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
                    ep.align_best_err = None
                    ep.align_progress_at = None

                loc = run_loc_hold_tick(
                    self,
                    ep,
                    pose=pose,
                    dist_goal_m=dist_goal_chk,
                    tick_started=tick_started,
                )
                if loc.abort:
                    return
                now = loc.now
                holding_for_localize = loc.holding_for_localize
                loc_hold = loc.loc_hold

                tick = run_sense_tick(
                    self, ep, now=now, pose=pose, path=path, goal=goal
                )
                tick.loc_hold_motion = loc.loc_hold_motion
                block = run_block_policy(self, ep, tick)
                if block.action == "replan":
                    if apply_blocked_replan(self, ep, tick).abort:
                        return

                t_dwa = time.perf_counter()
                if run_path_follow(self, ep, tick).abort:
                    return
                tick.phase_ms["dwa_ms"] = elapsed_ms(t_dwa)
                run_backup_recovery(self, ep, tick)
                if run_periodic_replan(self, ep, tick).abort:
                    return

                t_gate = time.perf_counter()
                gate = run_final_gate(self, ep, tick)
                tick.phase_ms["gate_ms"] = elapsed_ms(t_gate)
                out = finish_tick_output(
                    self,
                    tick,
                    gate,
                    holding_for_localize=holding_for_localize,
                    loc_hold=loc_hold,
                    tick_started=tick_started,
                )
                if out.succeeded:
                    return
                if out.skip_rest:
                    path = tick.path
                    continue

                if run_stall_watch(self, ep, tick).abort:
                    return

                if send_tick_velocity(self, ep, tick, tick_started=tick_started):
                    path = tick.path
                    continue
                path = tick.path

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

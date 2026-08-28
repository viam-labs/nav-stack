"""Goal lifecycle: plan → follow → replan → succeed / fail / cancel."""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

from ..nav.simple_motion import (
    DriveCommand,
    ObstacleConfig,
    SimpleMotionConfig,
    distance_m,
    rear_clearance_m,
)
from ..ros import conversions as conv
from .controller import FollowerConfig, compute_path_command
from .local_costmap import (
    LocalCostmap,
    LocalCostmapConfig,
    footprint_max_cost,
    reverse_backup_feasible,
)
from .local_planner import LocalPlannerConfig
from .planner import (
    path_blocked,
    path_blocked_local,
    paths_meaningfully_differ,
    plan_path,
    connect_plan_start,
)
from .smoother import smooth_plan_path
from .types import NavStatus, Path2D, PlanResult, Pose2D
from .world_io import WorldIO


class NavSupervisor:
    """Blocking control loop intended to run on a background worker thread."""

    def __init__(
        self,
        world: WorldIO,
        *,
        inflation_radius_m: float = 0.25,
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
        max_vel_x: float = 0.6,
        max_vel_theta: float = 1.5,
        min_cmd_vel_x: float = 0.0,
        min_cmd_vel_theta: float = 0.0,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.1,
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
        local_inflation_radius_m: float = 0.25,
        local_planner_enabled: bool = True,
        local_planner_sim_time_s: float = 1.2,
        local_planner_activate_cost: int = 200,
        local_planner_max_vel_x_reverse_m: float = 0.15,
        backup_enabled: bool = True,
        backup_stuck_time_s: float = 3.0,
        backup_dist_m: float = 0.30,
        backup_speed_mps: float = 0.12,
        backup_rear_clear_m: float = 0.45,
        backup_max_attempts: int = 1,
        backup_cooldown_s: float = 4.0,
        replan_local_blocked_time_s: float = 0.3,
        replan_local_min_period_s: float = 0.5,
    ):
        self._world = world
        self._inflation = inflation_radius_m
        self._robot_radius = robot_radius_m
        self._cost_scaling = cost_scaling_factor
        self._algorithm = algorithm
        self._replan_period = replan_period_s
        self._timeout_s = timeout_s
        self._scan_max_age = scan_max_age_s
        self._smooth_path = smooth_path
        self._smooth_spacing = smooth_sample_spacing_m
        self._local_costmap_enabled = local_costmap_enabled
        self._local_planner = LocalPlannerConfig(
            enabled=local_planner_enabled,
            sim_time_s=local_planner_sim_time_s,
            activate_cost_threshold=local_planner_activate_cost,
            max_vel_x_reverse_m=local_planner_max_vel_x_reverse_m,
        )
        self._backup_enabled = backup_enabled
        self._backup_stuck_time_s = backup_stuck_time_s
        self._backup_dist_m = backup_dist_m
        self._backup_speed_mps = backup_speed_mps
        self._backup_rear_clear_m = backup_rear_clear_m
        self._backup_max_attempts = max(0, int(backup_max_attempts))
        self._backup_cooldown_s = backup_cooldown_s
        self._replan_local_blocked_time_s = replan_local_blocked_time_s
        self._replan_local_min_period_s = replan_local_min_period_s
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
        # Cached global costmap for local window (rebuild ~1 Hz, not every tick).
        self._global_occ_cache = None
        self._global_costs_cache = None
        self._global_cache_at = 0.0
        self._local_view_cache = None
        self._local_view_at = 0.0
        self._local_update_period_s = 0.1
        self._local_scan_max_age_s = min(float(scan_max_age_s), 0.5)
        self._global_cache_period_s = 1.0
        self._cancel = threading.Event()
        self._status = NavStatus()
        self._status_lock = threading.Lock()
        self._last_cmd_vx = 0.0
        self._io_timeout_streak = 0
        self._follower = FollowerConfig(
            lookahead_m=lookahead_m,
            min_lookahead_m=min_lookahead_m,
            max_lookahead_m=max_lookahead_m,
            approach_dist_m=approach_dist_m,
            waypoint_tolerance_m=max(0.1, xy_tolerance_m),
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
                stop_distance_m=stop_distance_m,
                slow_distance_m=slow_distance_m,
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
            algorithm=self._algorithm,
            scan=scan,
            scan_pose=pose if scan is not None else None,
            blocked_path=blocked_path,
            blocked_path_pose=blocked_path_pose,
            dynamic_obstacle_radius_m=max(self._inflation + 0.1, 0.35),
        )
        if result.feasible:
            result = connect_plan_start(
                map_data,
                pose,
                result,
                inflation_radius_m=self._inflation,
                robot_radius_m=self._robot_radius,
                cost_scaling_factor=self._cost_scaling,
                algorithm=self._algorithm,
                xy_tolerance_m=self._follower.motion.xy_tolerance_m,
                scan=scan,
            )
        if result.feasible and self._smooth_path:
            smoothed = smooth_plan_path(
                result.path,
                map_data,
                inflation_radius_m=self._inflation,
                robot_radius_m=self._robot_radius,
                cost_scaling_factor=self._cost_scaling,
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

    def _try_replan(
        self,
        goal: Pose2D,
        pose: Pose2D,
        path: Path2D,
        scan: Optional[conv.LaserScan2D],
        *,
        require_different: bool = True,
        failed_count: int = 0,
    ) -> Optional[Path2D]:
        """Replan from ``pose``, optionally marking live scan hits on the map."""
        replanned = self.plan(
            goal,
            start=pose,
            scan=scan,
            blocked_path=path if failed_count >= 1 else None,
            blocked_path_pose=pose if failed_count >= 1 else None,
        )
        if not replanned.feasible:
            return None
        if require_different and not paths_meaningfully_differ(path, replanned.path):
            return None
        preview = self._publish_plan_viz(replanned, goal, start=pose)
        self._set_status(path=preview["path"], length_m=preview["length_m"])
        return replanned.path

    def run_goal(self, goal: Pose2D) -> None:
        """Plan and follow until success, failure, or cancel. Blocking."""
        self._cancel.clear()
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
            spin_stuck_since: Optional[float] = None
            backup_active = False
            backup_start: Optional[Pose2D] = None
            backup_start_cost = 0
            backup_attempts = 0
            backup_cooldown_until = 0.0
            local_blocked_since: Optional[float] = None
            last_local_replan_at = 0.0
            failed_replan_while_blocked = 0
            vx_sign_history: list[tuple[float, int]] = []
            poll = self._follower.motion.poll_interval_s

            while time.monotonic() < deadline:
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

                # Goal reached?
                goal_pose = Pose2D(path.points[-1][0], path.points[-1][1], path.goal_theta)
                if (
                    distance_m(pose, goal_pose) <= self._follower.motion.xy_tolerance_m
                    and abs(conv.normalize_angle(pose.theta - goal_pose.theta))
                    <= self._follower.motion.yaw_tolerance_rad
                ):
                    self._world.stop()
                    self._set_status(state="succeeded", active=False, error_msg="")
                    return

                now = time.monotonic()

                scan = None
                if (
                    (
                        self._follower.obstacle is not None
                        and self._follower.obstacle.enabled
                    )
                    or self._local_costmap is not None
                ):
                    try:
                        scan = self._world.get_scan(self._scan_max_age)
                    except TimeoutError:
                        scan = None

                local_view = self._local_view_cache
                if self._local_costmap is not None and (
                    now - self._local_view_at >= self._local_update_period_s
                    or local_view is None
                ):
                    costmap_scan = scan
                    if costmap_scan is None and self._local_scan_max_age_s > 0:
                        try:
                            costmap_scan = self._world.get_scan(
                                self._local_scan_max_age_s
                            )
                        except TimeoutError:
                            costmap_scan = None
                    if (
                        costmap_scan is not None
                        and costmap_scan.capture_pose is None
                    ):
                        costmap_scan = conv.LaserScan2D(
                            ranges=costmap_scan.ranges,
                            angle_min=costmap_scan.angle_min,
                            angle_increment=costmap_scan.angle_increment,
                            range_min=costmap_scan.range_min,
                            range_max=costmap_scan.range_max,
                            sensor_pose=costmap_scan.sensor_pose,
                            capture_pose=pose,
                        )
                    if now - self._global_cache_at >= self._global_cache_period_s or (
                        self._global_costs_cache is None
                    ):
                        try:
                            map_data = self._world.get_map()
                        except TimeoutError:
                            map_data = None
                        if map_data is not None:
                            from .costmap import build_costmap, occupancy_from_bridge_map

                            self._global_occ_cache = occupancy_from_bridge_map(map_data)
                            self._global_costs_cache = build_costmap(
                                self._global_occ_cache,
                                inflation_radius_m=self._inflation,
                                robot_radius_m=self._robot_radius,
                                cost_scaling_factor=self._cost_scaling,
                            )
                            self._global_cache_at = now
                    local_view = self._local_costmap.update(
                        pose,
                        costmap_scan,
                        global_occ=self._global_occ_cache,
                        global_costs=self._global_costs_cache,
                    )
                    self._local_view_cache = local_view
                    self._local_view_at = now
                    if self._local_costmap_enabled:
                        from .costmap import local_view_viz_dict

                        try:
                            self._world.set_viz_local_costmap(
                                local_view_viz_dict(local_view)
                            )
                        except Exception:  # noqa: BLE001 - viz is best-effort
                            pass

                local_blocked = (
                    local_view is not None
                    and path_blocked_local(
                        pose,
                        path,
                        local_view,
                        cost_threshold=self._local_planner_activate_cost,
                    )
                )
                if local_blocked:
                    if local_blocked_since is None:
                        local_blocked_since = now
                    if (
                        now - local_blocked_since
                        >= self._replan_local_blocked_time_s
                        and now - last_local_replan_at
                        >= self._replan_local_min_period_s
                    ):
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=failed_replan_while_blocked,
                        )
                        last_local_replan_at = now
                        last_replan = now
                        if new_path is not None:
                            path = new_path
                            local_blocked_since = None
                            failed_replan_while_blocked = 0
                            backup_attempts = 0
                            vx_sign_history.clear()
                            spin_stuck_since = None
                        else:
                            failed_replan_while_blocked += 1
                else:
                    local_blocked_since = None
                    failed_replan_while_blocked = 0

                cmd, progress = compute_path_command(
                    pose,
                    path,
                    cfg=self._follower,
                    scan=scan,
                    speed_mps=self._last_cmd_vx,
                    local_view=local_view,
                    local_planner=self._local_planner
                    if self._local_costmap_enabled and not local_blocked
                    else None,
                    robot_radius_m=self._robot_radius,
                    min_cmd_vel_x=self._follower.motion.min_linear_mps,
                    min_cmd_vel_theta=self._follower.motion.min_angular_rad_s,
                )

                allow_backup = (
                    self._backup_enabled
                    and scan is not None
                    and local_view is not None
                    and progress.get("local_planner")
                    and abs(cmd.vx) < 0.05
                    and abs(cmd.vtheta) > 0.1
                    and backup_attempts < self._backup_max_attempts
                    and now >= backup_cooldown_until
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
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=failed_replan_while_blocked,
                        )
                        if new_path is not None:
                            path = new_path
                            last_replan = now
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
                if replan_due:
                    map_data = self._world.get_map()
                    static_blocked = map_data is not None and path_blocked(
                        map_data,
                        path,
                        inflation_radius_m=self._inflation,
                        robot_radius_m=self._robot_radius,
                    )

                sustained_local = (
                    local_blocked
                    and local_blocked_since is not None
                    and now - local_blocked_since
                    >= self._replan_local_blocked_time_s
                )
                backup_exhausted = (
                    backup_attempts >= self._backup_max_attempts and local_blocked
                )
                should_replan = replan_due and (
                    static_blocked
                    or (oscillating and local_blocked)
                    or backup_exhausted
                )
                if should_replan:
                    new_path = self._try_replan(
                        goal,
                        pose,
                        path,
                        scan,
                        failed_count=failed_replan_while_blocked if local_blocked else 0,
                    )
                    if new_path is not None:
                        path = new_path
                        last_replan = now
                        last_progress_at = now
                        local_blocked_since = None
                        failed_replan_while_blocked = 0
                        backup_attempts = 0
                        vx_sign_history.clear()
                        spin_stuck_since = None
                    elif static_blocked:
                        self._world.stop()
                        self._set_status(
                            state="failed",
                            active=False,
                            error_msg="replan failed (path blocked)",
                        )
                        return
                    else:
                        last_replan = now

                self._set_status(
                    pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
                    progress={
                        k: progress[k]
                        for k in (
                            "obstacle",
                            "local_planner",
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
                    time.sleep(poll)
                    continue

                if cmd.done:
                    goal_pose = Pose2D(
                        path.points[-1][0], path.points[-1][1], path.goal_theta
                    )
                    at_goal = (
                        distance_m(pose, goal_pose)
                        <= self._follower.motion.xy_tolerance_m
                        and abs(conv.normalize_angle(pose.theta - goal_pose.theta))
                        <= self._follower.motion.yaw_tolerance_rad
                    )
                    if at_goal:
                        self._world.stop()
                        self._set_status(state="succeeded", active=False, error_msg="")
                        return

                # Stall detection.
                # Pure spin (vx≈0) must not reset the stall timer — otherwise a
                # stuck local-planner / rotate-in-place loop never replans.
                if last_progress_pose is None:
                    last_progress_pose = pose
                    last_progress_at = now
                else:
                    moved = distance_m(pose, last_progress_pose)
                    translating = abs(float(cmd.vx)) >= 0.05
                    if moved >= self._follower.motion.stall_progress_m and (
                        translating or moved >= self._follower.motion.stall_progress_m * 2
                    ):
                        last_progress_pose = pose
                        last_progress_at = now
                    elif translating:
                        turned = abs(
                            conv.normalize_angle(pose.theta - last_progress_pose.theta)
                        )
                        if turned >= self._follower.motion.stall_progress_rad:
                            last_progress_pose = pose
                            last_progress_at = now
                        elif (
                            now - last_progress_at
                            >= self._follower.motion.stall_timeout_s
                        ):
                            new_path = self._try_replan(
                                goal,
                                pose,
                                path,
                                scan,
                                failed_count=failed_replan_while_blocked,
                            )
                            if new_path is not None:
                                path = new_path
                                last_progress_at = now
                                last_replan = now
                                local_blocked_since = None
                                backup_attempts = 0
                            else:
                                self._world.stop()
                                self._set_status(
                                    state="failed",
                                    active=False,
                                    error_msg="navigation stalled",
                                )
                                return
                    elif now - last_progress_at >= self._follower.motion.stall_timeout_s:
                        new_path = self._try_replan(
                            goal,
                            pose,
                            path,
                            scan,
                            failed_count=failed_replan_while_blocked,
                        )
                        if new_path is not None:
                            path = new_path
                            last_progress_at = now
                            last_replan = now
                            local_blocked_since = None
                            backup_attempts = 0
                        else:
                            self._world.stop()
                            self._set_status(
                                state="failed",
                                active=False,
                                error_msg="navigation stalled (no forward progress)",
                            )
                            return

                try:
                    self._world.set_velocity(cmd.vx, cmd.vy, cmd.vtheta)
                    self._io_timeout_streak = 0
                except TimeoutError:
                    # Transient event-loop starvation — don't abort the goal on
                    # a single missed cmd_vel (common when local costmap was
                    # rebuilding the full global map every tick).
                    self._io_timeout_streak += 1
                    if self._io_timeout_streak >= 5:
                        raise
                    time.sleep(poll)
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
                        time.sleep(poll)
                        continue
                    if (
                        "goaway" in msg
                        or "unavailable" in msg
                        or "connection" in msg
                    ):
                        self._io_timeout_streak += 1
                        if self._io_timeout_streak >= 8:
                            raise
                        time.sleep(poll)
                        continue
                    raise
                self._last_cmd_vx = cmd.vx
                time.sleep(poll)

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

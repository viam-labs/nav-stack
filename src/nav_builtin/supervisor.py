"""Goal lifecycle: plan → follow → replan → succeed / fail / cancel."""
from __future__ import annotations

import threading
import time
from typing import Optional

from ..nav.simple_motion import ObstacleConfig, SimpleMotionConfig, distance_m
from ..ros import conversions as conv
from .controller import FollowerConfig, compute_path_command
from .planner import path_blocked, plan_path
from .types import NavStatus, PlanResult, Pose2D
from .world_io import WorldIO


class NavSupervisor:
    """Blocking control loop intended to run on a background worker thread."""

    def __init__(
        self,
        world: WorldIO,
        *,
        inflation_radius_m: float = 0.55,
        robot_radius_m: float = 0.22,
        cost_scaling_factor: float = 4.0,
        algorithm: str = "lazy_theta_star",
        replan_period_s: float = 1.0,
        lookahead_m: float = 1.0,
        xy_tolerance_m: float = 0.25,
        yaw_tolerance_rad: float = 0.25,
        max_vel_x: float = 0.4,
        max_vel_theta: float = 1.0,
        min_cmd_vel_x: float = 0.0,
        min_cmd_vel_theta: float = 0.0,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.1,
        avoid_obstacles: bool = True,
        stop_distance_m: float = 0.4,
        slow_distance_m: float = 1.0,
        scan_max_age_s: float = 2.0,
    ):
        self._world = world
        self._inflation = inflation_radius_m
        self._robot_radius = robot_radius_m
        self._cost_scaling = cost_scaling_factor
        self._algorithm = algorithm
        self._replan_period = replan_period_s
        self._timeout_s = timeout_s
        self._scan_max_age = scan_max_age_s
        self._cancel = threading.Event()
        self._status = NavStatus()
        self._status_lock = threading.Lock()
        self._follower = FollowerConfig(
            lookahead_m=lookahead_m,
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

    def plan(self, goal: Pose2D, start: Optional[Pose2D] = None) -> PlanResult:
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
        return plan_path(
            map_data,
            pose,
            goal,
            inflation_radius_m=self._inflation,
            robot_radius_m=self._robot_radius,
            cost_scaling_factor=self._cost_scaling,
            algorithm=self._algorithm,
        )

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
            preview = result.to_preview_dict(goal=(goal.x, goal.y, goal.theta), start=None)
            self._set_status(path=preview["path"], length_m=preview["length_m"])
            try:
                self._world.set_viz_plan(
                    tuple((p["x"], p["y"]) for p in preview["path"]),
                    (goal.x, goal.y, goal.theta),
                )
            except Exception:  # noqa: BLE001 - viz is best-effort
                pass

            deadline = time.monotonic() + self._timeout_s
            last_replan = time.monotonic()
            last_progress_pose: Optional[Pose2D] = None
            last_progress_at = time.monotonic()
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
                need_replan_check = now - last_replan >= self._replan_period
                if need_replan_check:
                    last_replan = now
                    map_data = self._world.get_map()
                    # Only replan when the current path is blocked. Soft periodic
                    # replan was flipping Lazy Theta* routes every second and
                    # making differential bases spin in place chasing a new
                    # bearing.
                    if map_data is not None and path_blocked(
                        map_data,
                        path,
                        inflation_radius_m=self._inflation,
                        robot_radius_m=self._robot_radius,
                    ):
                        replanned = self.plan(goal, start=pose)
                        if replanned.feasible:
                            path = replanned.path
                            preview = replanned.to_preview_dict(
                                goal=(goal.x, goal.y, goal.theta), start=pose
                            )
                            self._set_status(
                                path=preview["path"], length_m=preview["length_m"]
                            )
                            try:
                                self._world.set_viz_plan(
                                    tuple((p["x"], p["y"]) for p in preview["path"]),
                                    (goal.x, goal.y, goal.theta),
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        else:
                            self._world.stop()
                            self._set_status(
                                state="failed",
                                active=False,
                                error_msg=replanned.error_msg or "replan failed",
                            )
                            return

                scan = None
                if self._follower.obstacle is not None and self._follower.obstacle.enabled:
                    scan = self._world.get_scan(self._scan_max_age)

                cmd, progress = compute_path_command(
                    pose, path, cfg=self._follower, scan=scan
                )
                self._set_status(
                    pose={"x": pose.x, "y": pose.y, "theta": pose.theta},
                    progress={
                        k: progress[k]
                        for k in (
                            "obstacle",
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
                    self._world.stop()
                    self._set_status(state="succeeded", active=False, error_msg="")
                    return

                # Stall detection.
                if last_progress_pose is None:
                    last_progress_pose = pose
                    last_progress_at = now
                else:
                    moved = distance_m(pose, last_progress_pose)
                    turned = abs(
                        conv.normalize_angle(pose.theta - last_progress_pose.theta)
                    )
                    if (
                        moved >= self._follower.motion.stall_progress_m
                        or turned >= self._follower.motion.stall_progress_rad
                    ):
                        last_progress_pose = pose
                        last_progress_at = now
                    elif now - last_progress_at >= self._follower.motion.stall_timeout_s:
                        # Try a replan once on stall before failing.
                        replanned = self.plan(goal, start=pose)
                        if replanned.feasible and replanned.path.points != path.points:
                            path = replanned.path
                            last_progress_at = now
                            last_replan = now
                        else:
                            self._world.stop()
                            self._set_status(
                                state="failed",
                                active=False,
                                error_msg="navigation stalled",
                            )
                            return

                self._world.set_velocity(cmd.vx, cmd.vy, cmd.vtheta)
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
            self._set_status(state="failed", active=False, error_msg=str(exc))

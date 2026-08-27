"""Duck-typed RosManager stand-in for ROS-free builtin navigation."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import NAV_BACKEND_BUILTIN, NavConfig
from ..ros import conversions as conv
from .navigator import BuiltinNavigator
from .viz_store import NavVizStore
from .world_io import WorldIO


def make_builtin_navigator(
    world: WorldIO,
    nav_cfg: NavConfig,
    *,
    logger=None,
) -> BuiltinNavigator:
    bcfg = nav_cfg.builtin
    xy_tol = (
        bcfg.xy_goal_tolerance
        if bcfg.xy_goal_tolerance is not None
        else nav_cfg.nav2.xy_goal_tolerance
    )
    yaw_tol = (
        bcfg.yaw_goal_tolerance
        if bcfg.yaw_goal_tolerance is not None
        else nav_cfg.nav2.yaw_goal_tolerance
    )
    return BuiltinNavigator(
        world,
        inflation_radius_m=nav_cfg.inflation_radius,
        robot_radius_m=nav_cfg.robot_radius,
        cost_scaling_factor=bcfg.cost_scaling_factor,
        algorithm=bcfg.planner,
        replan_period_s=bcfg.replan_period_s,
        lookahead_m=bcfg.lookahead_m,
        xy_tolerance_m=xy_tol,
        yaw_tolerance_rad=yaw_tol,
        max_vel_x=nav_cfg.max_vel_x,
        max_vel_theta=nav_cfg.max_vel_theta,
        min_cmd_vel_x=nav_cfg.min_cmd_vel_x,
        min_cmd_vel_theta=nav_cfg.min_cmd_vel_theta,
        timeout_s=bcfg.timeout_s,
        avoid_obstacles=nav_cfg.simple_avoid_obstacles,
        stop_distance_m=nav_cfg.simple_stop_distance,
        slow_distance_m=nav_cfg.simple_slow_distance,
        scan_max_age_s=nav_cfg.simple_scan_max_age,
        logger=logger,
    )


class BuiltinNavHost:
    """Nav surface used by ``NavServiceBase`` when there is no ROS/RosManager.

    Implements the methods ``nav_core`` calls on ``runtime.manager`` for
    navigate / plan / pose / scan / status. ``node`` is None (zones / Nav2
    filters are not published).
    """

    def __init__(
        self,
        navigator: BuiltinNavigator,
        world: WorldIO,
        viz: NavVizStore,
        *,
        nav_cfg: NavConfig,
    ):
        self._builtin_nav = navigator
        self._world = world
        self._viz = viz
        self._nav_cfg = nav_cfg
        self.node = None

    @property
    def viz(self) -> NavVizStore:
        return self._viz

    def set_nav_config(self, nav_cfg: NavConfig) -> None:
        self._nav_cfg = nav_cfg
        # Rebuild navigator with updated limits/planner if config changes.
        self._builtin_nav.cancel()
        self._builtin_nav = make_builtin_navigator(
            self._world, nav_cfg, logger=self._builtin_nav._logger  # noqa: SLF001
        )

    def ensure_nav2(self, nav_cfg: NavConfig, params_path) -> None:
        del params_path
        self.set_nav_config(nav_cfg)

    def ensure_nav2_async(self, nav_cfg: NavConfig, params_path) -> None:
        self.ensure_nav2(nav_cfg, params_path)

    def stop_nav2(self) -> None:
        self._builtin_nav.cancel()

    def nav_backend(self) -> str:
        return NAV_BACKEND_BUILTIN

    def nav2_running(self) -> bool:
        return False

    def nav2_startup_in_progress(self) -> bool:
        return False

    def nav_action_ready(self) -> bool:
        return True

    def navigate(self, x: float, y: float, theta: float) -> None:
        self._builtin_nav.navigate(x, y, theta)

    def compute_path(self, *args, **kwargs) -> Dict:
        return self._builtin_nav.compute_path(*args, **kwargs)

    def last_preview_plan(self) -> Optional[Dict]:
        return self._builtin_nav.last_preview_plan()

    def cancel(self) -> None:
        self._builtin_nav.cancel()

    def nav_status(self) -> Dict:
        status = self._builtin_nav.nav_status()
        status["nav_backend"] = NAV_BACKEND_BUILTIN
        return status

    def nav2_diagnostics(self, fast: bool = False) -> Dict:
        del fast
        return {
            "nav_backend": NAV_BACKEND_BUILTIN,
            "nav2_processes_running": False,
            "nav2_startup_in_progress": False,
            "nav_action_ready": True,
            "core_nodes_present": True,
            "missing_core_nodes": [],
        }

    def get_pose_in_map(self) -> Optional[conv.Pose2D]:
        return self._world.get_pose()

    def get_base_scan(self, max_age_s: float = 1.0) -> Optional[conv.LaserScan2D]:
        return self._world.get_scan(max_age_s)

    def publish_zone_masks(
        self,
        keepout_mask: np.ndarray,
        speed_mask: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        # Builtin costmap does not consume Nav2 keepout/speed topics yet.
        del keepout_mask, speed_mask, resolution, origin_x, origin_y

    def shutdown(self) -> None:
        try:
            self._builtin_nav.cancel()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._world.stop()
        except Exception:  # noqa: BLE001
            pass

"""Single wiring path: NavConfig → NavSupervisor / BuiltinNavigator kwargs."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import NavConfig


def default_nav_cfg() -> NavConfig:
    """Minimal NavConfig for unit tests that only pass runtime overrides."""
    return NavConfig(slam_service="slam", base="base")


def builtin_nav_runtime_kwargs(
    nav_cfg: Optional[NavConfig] = None, **overrides: Any
) -> Dict[str, Any]:
    """Build NavSupervisor kwargs from config; ``overrides`` win (tests / callers)."""
    cfg = nav_cfg if nav_cfg is not None else default_nav_cfg()
    bcfg = cfg.builtin
    kwargs: Dict[str, Any] = {
        "inflation_radius_m": cfg.effective_inflation_radius_m(),
        # Driving clearance uses the half-width; rotation uses the half-diagonal.
        "robot_radius_m": cfg.inscribed_radius_m(),
        "spin_radius_m": cfg.circumscribed_radius_m(),
        "nose_offset_m": cfg.nose_offset_m(),
        "wheel_half_track_m": cfg.wheel_half_track_m(),
        "cost_scaling_factor": bcfg.cost_scaling_factor,
        "clearance_preference_m": bcfg.clearance_preference_m,
        "algorithm": bcfg.planner,
        "replan_period_s": bcfg.replan_period_s,
        "lookahead_m": bcfg.lookahead_m,
        "min_lookahead_m": bcfg.min_lookahead_m,
        "max_lookahead_m": bcfg.max_lookahead_m,
        "approach_dist_m": bcfg.approach_dist_m,
        "xy_tolerance_m": bcfg.xy_goal_tolerance,
        "yaw_tolerance_rad": bcfg.yaw_goal_tolerance,
        "max_vel_x": cfg.max_vel_x,
        "max_vel_theta": cfg.max_vel_theta,
        "min_cmd_vel_x": cfg.min_cmd_vel_x,
        "min_cmd_vel_theta": cfg.min_cmd_vel_theta,
        "timeout_s": bcfg.timeout_s,
        "poll_interval_s": cfg.control_period_s(),
        "avoid_obstacles": cfg.simple_avoid_obstacles,
        "stop_distance_m": cfg.simple_stop_distance,
        "slow_distance_m": cfg.simple_slow_distance,
        "scan_max_age_s": cfg.simple_scan_max_age,
        "smooth_path": bcfg.smooth_path,
        "smooth_sample_spacing_m": bcfg.smooth_sample_spacing_m,
        "local_costmap_enabled": bcfg.local_costmap_enabled,
        "local_costmap_width_m": bcfg.local_costmap_width_m,
        "local_costmap_height_m": bcfg.local_costmap_height_m,
        "local_costmap_resolution": bcfg.local_costmap_resolution,
        "local_inflation_radius_m": cfg.effective_local_inflation_radius_m(),
        "local_costmap_rate_hz": bcfg.local_costmap_rate_hz,
        "local_planner_enabled": bcfg.local_planner_enabled,
        "local_planner_sim_time_s": bcfg.local_planner_sim_time_s,
        "local_planner_activate_cost": bcfg.local_planner_activate_cost,
        "local_planner_max_vel_x_mps": bcfg.local_planner_max_vel_x_mps,
        "local_planner_max_vel_x_reverse_m": bcfg.local_planner_max_vel_x_reverse_m,
        "backup_enabled": bcfg.backup_enabled,
        "backup_stuck_time_s": bcfg.backup_stuck_time_s,
        "backup_dist_m": bcfg.backup_dist_m,
        "backup_speed_mps": bcfg.backup_speed_mps,
        "backup_rear_clear_m": bcfg.backup_rear_clear_m,
        "backup_max_attempts": bcfg.backup_max_attempts,
        "backup_cooldown_s": bcfg.backup_cooldown_s,
        "recovery_wait_duration_s": bcfg.recovery_wait_duration_s,
        "replan_local_blocked_time_s": bcfg.replan_local_blocked_time_s,
        "replan_local_min_period_s": bcfg.replan_local_min_period_s,
        "nav_policy": bcfg.nav_policy,
        "jev_min_confidence": bcfg.jev_min_confidence,
        "jev_timeout_s": bcfg.jev_timeout_s,
        "jev_min_period_s": bcfg.jev_min_period_s,
        "jev_history_s": bcfg.jev_history_s,
        "jev_model": bcfg.jev_model,
        "jev_api_key": bcfg.jev_api_key,
        "drive_timeout_streak": bcfg.drive_timeout_streak,
        "yaw_align_timeout_s": bcfg.yaw_align_timeout_s,
        "max_goal_snap_m": bcfg.max_goal_snap_m,
        "max_linear_accel_mps2": bcfg.max_linear_accel_mps2,
        "max_linear_decel_mps2": bcfg.max_linear_decel_mps2,
        "max_angular_accel_rad_s2": bcfg.max_angular_accel_rad_s2,
    }
    kwargs.update(overrides)
    return kwargs

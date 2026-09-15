
import pytest

from src.config import (
    DIFFERENTIAL,
    OMNI,
    ExternalNavConfig,
    NavConfig,
    SlamConfig,
    body_linear_to_viam_mm_s,
    body_twist_to_viam_set_velocity,
    body_vtheta_to_viam_angular_deg_s,
    sensor_twist_to_body,
)


def test_slam_config_single_lidar_string():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert [l.name for l in cfg.lidars] == ["front"]
    assert cfg.required_dependencies() == ["b", "front"]


def test_slam_config_scan_max_age_default_and_override():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert cfg.scan_max_age_s == 2.0
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front", "scan_max_age_s": 0.75})
    assert cfg.scan_max_age_s == 0.75


def test_slam_config_lidar_scan_source():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {"name": "livox-pc", "scan_source": "point_cloud"},
        }
    )
    assert cfg.lidars[0].scan_source == "point_cloud"
    assert cfg.scan_accumulation_s == pytest.approx(0.3)
    assert cfg.heading_only_odom is False
    assert cfg.imu_odom_mode == "accel_only"
    assert cfg.lidar_odom_enabled is True
    assert cfg.lidar_odom_range_flow_only is True
    assert cfg.map.minimum_travel_distance == pytest.approx(0.15)
    assert cfg.map.minimum_travel_heading == pytest.approx(0.12)
    assert cfg.slam_params.get("minimum_time_interval") == pytest.approx(0.3)
    assert cfg.slam_params.get("correlation_search_space_dimension") == pytest.approx(0.6)
    with pytest.raises(ValueError, match="scan_source"):
        SlamConfig.from_dict(
            {"base": "b", "lidar": {"name": "x", "scan_source": "invalid"}}
        )


def test_slam_config_multi_lidar_with_mounts():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": [
                {"name": "front", "mount": {"x": 0.2, "theta": 0.0}},
                {"name": "rear", "mount": {"x": -0.2, "theta": 3.14159}},
            ],
            "movement_sensor": "odom",
            "mode": "localizing",
        }
    )
    assert len(cfg.lidars) == 2
    assert cfg.lidars[0].x == 0.2
    assert cfg.lidars[1].theta == pytest.approx(3.14159)
    assert "odom" in cfg.required_dependencies()
    assert cfg.movement_sensor_yaw_deg == 0.0


def test_lidar_config_mount_pitch_roll():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {
                "name": "livox",
                "scan_source": "point_cloud",
                "mount": {"x": 0.463, "z": 1.129, "theta": 0.0, "pitch": 0.035},
            },
        }
    )
    assert cfg.lidars[0].pitch == pytest.approx(0.035)
    assert cfg.lidars[0].roll == 0.0
    assert cfg.lidars[0].shm_name == "/viam-pc-livox"
    assert cfg.lidars[0].shm_required is False


def test_lidar_shm_name_empty_disables_default():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": {"name": "rplidar", "shm_name": ""}}
    )
    assert cfg.lidars[0].shm_name is None


def test_imu_shm_defaults_from_heading_sensor():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "heading_sensor": "wit"}
    )
    assert cfg.imu_shm_name == "/viam-imu-wit"
    cfg2 = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "heading_sensor": "wit",
            "imu_shm_name": "",
        }
    )
    assert cfg2.imu_shm_name is None


def test_lidar_config_shm_fields():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {
                "name": "front",
                "shm_name": "/viam-pc-lidar",
                "shm_required": True,
                "shm_region_size": 4 * 1024 * 1024,
            },
        }
    )
    assert cfg.lidars[0].shm_name == "/viam-pc-lidar"
    assert cfg.lidars[0].shm_required is True
    assert cfg.lidars[0].shm_region_size == 4 * 1024 * 1024


def test_slam_config_map_when_still_livox_defaults():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {"name": "livox", "scan_source": "point_cloud"},
            "map_when_still": True,
        }
    )
    assert cfg.map_when_still is True
    assert cfg.scan_accumulation_s == pytest.approx(1.0)
    assert cfg.map_when_still_dwell_s == pytest.approx(1.0)
    assert cfg.map.minimum_travel_distance == pytest.approx(0.0)
    # MapSettings travel gates for dense lidars.
    assert cfg.slam_params.get("correlation_search_space_dimension") == pytest.approx(
        1.0
    )
    assert cfg.slam_params.get("coarse_search_angle_offset") == pytest.approx(0.52)
    assert cfg.slam_params.get("link_match_minimum_response_fine") == pytest.approx(
        0.25
    )
    assert cfg.slam_params.get("loop_match_minimum_chain_size") == 10
    assert cfg.slam_params.get("loop_search_maximum_distance") == pytest.approx(5.0)
    assert cfg.slam_params.get("loop_match_minimum_response_fine") == pytest.approx(
        0.45
    )
    assert cfg.slam_params.get("do_loop_closing") is True
    assert cfg.slam_params.get("angle_variance_penalty") == pytest.approx(1.0)
    assert cfg.lidar_odom_enabled is False
    assert cfg.heading_only_odom is False
    assert cfg.imu_odom_mode == "accel_only"
    assert cfg.wall_yaw_correction is True
    assert cfg.wall_yaw_min_length_m == pytest.approx(2.0)
    assert cfg.wall_yaw_max_step_deg == pytest.approx(2.0)
    assert cfg.wall_yaw_blend == pytest.approx(0.5)
    # Mapping-time revisit check defaults on for stop-and-go Livox carts.
    assert cfg.mapping_revisit_check is True
    assert cfg.mapping_revisit_interval_s == pytest.approx(20.0)
    assert cfg.mapping_revisit_search_radius_m == pytest.approx(5.0)
    assert cfg.mapping_revisit_wide_radius_m == pytest.approx(12.0)
    assert cfg.mapping_revisit_min_score == pytest.approx(0.6)
    assert cfg.mapping_revisit_full_map_min_score == pytest.approx(0.75)
    assert cfg.mapping_revisit_min_shift_m == pytest.approx(1.0)
    assert cfg.mapping_revisit_max_shift_m == pytest.approx(10.0)
    assert cfg.mapping_revisit_full_map_fallback is True
    # Multi-height-slice verification defaults (3D lidar).
    assert cfg.mapping_revisit_slice_verify is True
    assert cfg.mapping_revisit_slice_bands == [[0.15, 0.45], [1.6, 2.4]]
    assert cfg.mapping_revisit_slice_min_hit_rate == pytest.approx(0.4)
    assert cfg.mapping_revisit_slice_resolution_m == pytest.approx(0.15)
    assert cfg.mapping_revisit_keyframes is True
    assert cfg.mapping_revisit_keyframe_min_score == pytest.approx(0.55)
    assert cfg.mapping_revisit_keyframe_max == 250
    # Strict stop-and-go: mid-pivot scans off unless explicitly enabled.
    assert cfg.map_when_still_yaw_step_deg == pytest.approx(0.0)
    assert cfg.map_when_still_max_drift_m == pytest.approx(0.03)
    assert cfg.map_when_still_max_drift_deg == pytest.approx(1.5)


def test_slam_config_map_when_still_overrides_user_travel_gates():
    """Continuous Livox travel gates must not survive with map_when_still."""
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {"name": "livox", "scan_source": "point_cloud"},
            "map_when_still": True,
            "map": {
                "resolution": 0.05,
                "minimum_travel_distance": 0.15,
                "minimum_travel_heading": 0.15,
            },
        }
    )
    assert cfg.map.minimum_travel_distance == pytest.approx(0.0)
    assert cfg.map.minimum_travel_heading == pytest.approx(0.0)


def test_slam_config_map_when_still_default_off_for_mir_style():
    """MiR-style laser scan lidars must keep continuous mapping defaults."""
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert cfg.map_when_still is False
    assert cfg.wall_yaw_correction is False
    assert cfg.mapping_revisit_check is True
    assert cfg.mapping_revisit_while_moving is True
    assert cfg.scan_accumulation_s == pytest.approx(0.0)


def test_slam_config_revisit_while_moving_off_for_map_when_still():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {"name": "livox", "scan_source": "point_cloud"},
            "map_when_still": True,
            "slam_backend": "builtin",
        }
    )
    assert cfg.mapping_revisit_while_moving is False
    cfg_on = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {"name": "livox", "scan_source": "point_cloud"},
            "map_when_still": True,
            "slam_backend": "builtin",
            "mapping_revisit_while_moving": True,
        }
    )
    assert cfg_on.mapping_revisit_while_moving is True


def test_slam_config_movement_sensor_yaw_deg():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": {"name": "x", "scan_source": "point_cloud"},
            "movement_sensor": "imu",
            "movement_sensor_yaw_deg": -90,
            "movement_sensor_upside_down": True,
            "heading_sensor_invert": True,
        }
    )
    assert cfg.movement_sensor_yaw_deg == pytest.approx(-90.0)
    assert cfg.movement_sensor_upside_down is True
    assert cfg.heading_sensor_invert is True
    assert cfg.heading_sensor_yaw_deg == 0.0


def test_slam_config_global_localize_on_start_options():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "mode": "localizing",
            "global_localize_on_start": True,
            "global_localize_on_start_delay_s": 4.5,
            "global_localize_on_start_options": {
                "full_map": True,
                "map_source": "live",
            },
            "global_localize_on_start_refine": True,
            "global_localize_on_start_refine_delay_s": 6.0,
            "global_localize_on_start_refine_options": {
                "full_map": False,
                "local_yaw_window_deg": 90.0,
            },
        }
    )
    assert cfg.global_localize_on_start is True
    assert cfg.global_localize_on_start_delay_s == pytest.approx(4.5)
    assert cfg.global_localize_on_start_readiness_timeout_s == pytest.approx(90.0)
    assert cfg.global_localize_on_start_options["full_map"] is True
    assert cfg.global_localize_on_start_options["map_source"] == "live"
    assert cfg.global_localize_on_start_refine is True
    assert cfg.global_localize_on_start_refine_delay_s == pytest.approx(6.0)
    assert cfg.global_localize_on_start_refine_max_passes == 3
    assert cfg.global_localize_on_start_target_score == pytest.approx(0.7)
    assert cfg.global_localize_on_start_target_ray_mae_m == pytest.approx(0.4)
    assert cfg.global_localize_on_start_post_apply_refine is True
    assert cfg.global_localize_on_start_post_apply_refine_delay_s == pytest.approx(8.0)
    assert cfg.global_localize_on_start_post_apply_refine_options["map_source"] == "live"
    assert cfg.global_localize_on_start_refine_options["full_map"] is False


def test_slam_config_periodic_relocalize_defaults():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front", "mode": "localizing"})
    assert cfg.periodic_relocalize is True
    assert cfg.periodic_relocalize_interval_s == pytest.approx(20.0)
    assert cfg.periodic_relocalize_nav_interval_s == pytest.approx(25.0)
    assert cfg.periodic_relocalize_soft_hold_max_s == pytest.approx(20.0)
    assert cfg.periodic_relocalize_max_yaw_rate_rad_s == pytest.approx(0.35)
    assert cfg.periodic_relocalize_max_scan_age_s == pytest.approx(0.75)
    assert cfg.periodic_relocalize_min_score == pytest.approx(0.5)
    assert cfg.periodic_relocalize_max_ray_mae_m == pytest.approx(1.0)
    assert cfg.periodic_relocalize_recovery_min_score == pytest.approx(0.45)
    assert cfg.periodic_relocalize_min_shift_m == pytest.approx(0.2)
    assert cfg.periodic_relocalize_min_shift_deg == pytest.approx(10.0)
    assert cfg.periodic_relocalize_nav_recoveries_threshold == 2
    assert cfg.periodic_relocalize_full_map_on_low_quality is True
    assert cfg.periodic_relocalize_during_navigation is True
    assert cfg.periodic_relocalize_options["search_radius_m"] == pytest.approx(3.0)
    assert cfg.periodic_relocalize_options["auto_full_map_fallback"] is True


def test_slam_config_periodic_relocalize_overrides():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "mode": "localizing",
            "periodic_relocalize": True,
            "periodic_relocalize_interval_s": 30.0,
            "periodic_relocalize_min_score": 0.6,
            "periodic_relocalize_during_navigation": True,
            "periodic_relocalize_options": {"search_radius_m": 2.0},
        }
    )
    assert cfg.periodic_relocalize is True
    assert cfg.periodic_relocalize_interval_s == pytest.approx(30.0)
    assert cfg.periodic_relocalize_min_score == pytest.approx(0.6)
    assert cfg.periodic_relocalize_during_navigation is True
    assert cfg.periodic_relocalize_options["search_radius_m"] == pytest.approx(2.0)


def test_slam_config_global_localize_on_start_defaults_enabled():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front", "mode": "localizing"})
    assert cfg.global_localize_on_start is True
    assert cfg.global_localize_on_start_delay_s == pytest.approx(4.0)
    assert cfg.global_localize_on_start_options == {
        "full_map": True,
        "map_source": "live",
    }
    assert cfg.global_localize_on_start_refine is True
    assert cfg.global_localize_on_start_refine_delay_s == pytest.approx(8.0)
    assert cfg.global_localize_on_start_refine_max_passes == 3
    assert cfg.global_localize_on_start_target_score == pytest.approx(0.7)
    assert cfg.global_localize_on_start_target_ray_mae_m == pytest.approx(0.4)
    assert cfg.global_localize_on_start_post_apply_refine is True
    assert cfg.global_localize_on_start_post_apply_refine_delay_s == pytest.approx(8.0)
    assert cfg.global_localize_on_start_post_apply_refine_options == {"map_source": "live"}
    assert cfg.global_localize_on_start_refine_options == {
        "full_map": False,
        "map_source": "live",
        "local_yaw_window_deg": 120.0,
        "search_radius_m": 6.0,
    }


def test_slam_config_requires_lidar():
    with pytest.raises(ValueError):
        SlamConfig.from_dict({"base": "b"})


def test_slam_config_obstacles_only_lidar():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": [
                {"name": "rplidar"},
                {
                    "name": "depth",
                    "scan_source": "point_cloud",
                    "obstacles_only": True,
                    "max_range": 4.0,
                },
            ],
        }
    )
    assert cfg.lidars[0].obstacles_only is False
    assert cfg.lidars[1].obstacles_only is True
    assert [l.name for l in cfg.slam_lidars()] == ["rplidar"]
    # Mapping lidar is not point_cloud-only → do not apply Livox SLAM defaults.
    assert cfg.map_when_still is False
    assert cfg.imu_odom_mode == "coast"


def test_slam_config_rejects_all_obstacles_only():
    with pytest.raises(ValueError, match="obstacles_only"):
        SlamConfig.from_dict(
            {
                "base": "b",
                "lidars": [
                    {"name": "depth", "obstacles_only": True},
                ],
            }
        )


def test_slam_config_bad_mode():
    with pytest.raises(ValueError):
        SlamConfig.from_dict({"base": "b", "lidar": "f", "mode": "wat"})


def test_nav_config_defaults_and_deps():
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    assert cfg.kinematics == DIFFERENTIAL
    assert cfg.required_dependencies() == ["slam", "b"]
    assert cfg.nav_backend == "builtin"
    assert cfg.uses_builtin_nav() is True


def test_nav_config_nav_backend_nav2_rejected():
    with pytest.raises(ValueError, match="pre-ros-removal|nav2|no longer supported"):
        NavConfig.from_dict(
            {"slam_service": "slam", "base": "b", "nav_backend": "nav2"}
        )


def test_nav_config_legacy_nav2_block_alias():
    """Older machine configs used ``nav2`` for what is now ``builtin``."""
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "nav2": {
                "xy_goal_tolerance": 0.25,
                "yaw_goal_tolerance": 0.5,
            },
        }
    )
    assert cfg.builtin.xy_goal_tolerance == pytest.approx(0.25)
    assert cfg.builtin.yaw_goal_tolerance == pytest.approx(0.5)
    # Explicit ``builtin`` wins over legacy ``nav2``.
    prefer = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "builtin": {"yaw_goal_tolerance": 0.4},
            "nav2": {"yaw_goal_tolerance": 0.9},
        }
    )
    assert prefer.builtin.yaw_goal_tolerance == pytest.approx(0.4)


def test_nav_config_top_level_goal_tolerances():
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "xy_goal_tolerance": 0.3,
            "yaw_goal_tolerance": 0.55,
        }
    )
    assert cfg.builtin.xy_goal_tolerance == pytest.approx(0.3)
    assert cfg.builtin.yaw_goal_tolerance == pytest.approx(0.55)
    # Nested builtin wins over top-level.
    nested = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "xy_goal_tolerance": 0.3,
            "builtin": {"xy_goal_tolerance": 0.18},
        }
    )
    assert nested.builtin.xy_goal_tolerance == pytest.approx(0.18)


def test_slam_config_top_level_resolution():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "mode": "localizing",
            "resolution": 0.08,
            "max_laser_range": 12.0,
        }
    )
    assert cfg.map.resolution == pytest.approx(0.08)
    assert cfg.map.max_laser_range == pytest.approx(12.0)
    nested = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "mode": "localizing",
            "resolution": 0.08,
            "map": {"resolution": 0.05},
        }
    )
    assert nested.map.resolution == pytest.approx(0.05)


def test_slam_config_legacy_slam_toolbox_block_alias():
    """Older machine configs used ``slam_toolbox`` for what is now ``map``."""
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "mode": "localizing",
            "slam_toolbox": {"resolution": 0.08},
        }
    )
    assert cfg.map.resolution == pytest.approx(0.08)
    prefer = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "mode": "localizing",
            "map": {"resolution": 0.05},
            "slam_toolbox": {"resolution": 0.2},
        }
    )
    assert prefer.map.resolution == pytest.approx(0.05)


def test_builtin_recovery_wait_defaults():
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    assert cfg.builtin.recovery_wait_duration_s == pytest.approx(2.0)
    tuned = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "builtin": {"recovery_wait_duration_s": 3.5},
        }
    )
    assert tuned.builtin.recovery_wait_duration_s == pytest.approx(3.5)


def test_builtin_follower_snake_defaults():
    """Defaults tuned to damp mid-path S-curve hunting on skid-steer."""
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    assert cfg.builtin.lookahead_m == pytest.approx(1.35)
    assert cfg.builtin.min_lookahead_m == pytest.approx(1.1)
    assert cfg.builtin.max_lookahead_m == pytest.approx(1.8)
    assert cfg.builtin.smooth_sample_spacing_m == pytest.approx(0.20)
    # Partial override must not resurrect the old from_dict fallbacks.
    partial = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "builtin": {"replan_period_s": 0.8},
        }
    )
    assert partial.builtin.lookahead_m == pytest.approx(1.35)
    assert partial.builtin.smooth_sample_spacing_m == pytest.approx(0.20)


def test_nav_config_bad_nav_backend():
    with pytest.raises(ValueError, match="nav_backend"):
        NavConfig.from_dict(
            {"slam_service": "slam", "base": "b", "nav_backend": "magic"}
        )


def test_nav_config_omni():
    cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "kinematics": "omni", "max_vel_y": 0.3}
    )
    assert cfg.kinematics == OMNI
    assert cfg.max_vel_y == 0.3


def test_nav_config_bad_kinematics():
    with pytest.raises(ValueError):
        NavConfig.from_dict({"slam_service": "s", "base": "b", "kinematics": "legs"})


def test_map_settings_from_attributes():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "f",
            "mode": "localizing",
            "map": {
                "resolution": 0.1,
                "max_laser_range": 30.0,
                "minimum_travel_distance": 0.5,
            },
        }
    )
    assert cfg.mode == "localizing"
    assert cfg.map.resolution == 0.1
    assert cfg.map.max_laser_range == 30.0
    assert cfg.map.minimum_travel_distance == 0.5


def test_base_velocity_convention_viam_default():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f"})
    assert cfg.base_velocity_convention == "viam"
    lx, ly = body_linear_to_viam_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(-100.0)
    assert ly == pytest.approx(500.0)


def test_base_velocity_convention_ros_alias_normalizes_to_x_forward():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "base_velocity_convention": "ros"}
    )
    assert cfg.base_velocity_convention == "x_forward"
    lx, ly = body_linear_to_viam_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(500.0)
    assert ly == pytest.approx(-100.0)


def test_base_velocity_convention_x_forward():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "base_velocity_convention": "x_forward"}
    )
    assert cfg.base_velocity_convention == "x_forward"
    lx, ly = body_linear_to_viam_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(500.0)
    assert ly == pytest.approx(-100.0)


def test_base_velocity_convention_mir_alias_normalizes_to_viam():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "base_velocity_convention": "mir"}
    )
    assert cfg.base_velocity_convention == "viam"
    lx, ly = body_linear_to_viam_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(-100.0)
    assert ly == pytest.approx(500.0)



def test_body_vtheta_to_viam_angular_is_degrees():
    import math
    # -1 rad/s must become ~-57.3 deg/s at Base.SetVelocity (Viam API).
    assert body_vtheta_to_viam_angular_deg_s(-1.0) == pytest.approx(-math.degrees(1.0))
    lx, ly, az = body_twist_to_viam_set_velocity(0.5, 0.0, -1.0, "viam")
    assert lx == pytest.approx(0.0)  # Y-forward: body vx -> linear.y
    assert ly == pytest.approx(500.0)
    assert az == pytest.approx(-math.degrees(1.0))

def test_sensor_twist_to_body_viam_y_forward():
    # Sensor: forward on y=0.019, no lateral → ROS forward on vx.
    ros_vx, ros_vy = sensor_twist_to_body(0.0, 0.019, "viam")
    assert ros_vx == pytest.approx(0.019)
    assert ros_vy == pytest.approx(0.0)
    ros_vx, ros_vy = sensor_twist_to_body(0.5, -0.1, "ros")
    assert ros_vx == pytest.approx(0.5)
    assert ros_vy == pytest.approx(-0.1)


def test_min_cmd_vel_defaults_and_legacy_alias():
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    # Default off — nonzero floors were a MiR regression (cart-only stiction).
    assert cfg.min_cmd_vel_x == pytest.approx(0.0)
    assert cfg.min_cmd_vel_theta == pytest.approx(0.0)

    cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "min_cmd_vel_x": 0.25, "min_cmd_vel_theta": 0.5}
    )
    assert cfg.min_cmd_vel_x == pytest.approx(0.25)
    assert cfg.min_cmd_vel_theta == pytest.approx(0.5)

    # Legacy names still accepted.
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "simple_min_vel_x": 0.3,
            "simple_min_vel_theta": 0.6,
        }
    )
    assert cfg.min_cmd_vel_x == pytest.approx(0.3)
    assert cfg.min_cmd_vel_theta == pytest.approx(0.6)


def test_base_velocity_convention_invalid():
    with pytest.raises(ValueError):
        SlamConfig.from_dict(
            {"base": "b", "lidar": "f", "base_velocity_convention": "sideways"}
        )


def test_external_nav_config_builds_bridge_and_nav():
    d = {
        "slam_service": "rtabmap",
        "base": "base",
        "lidars": [{"name": "mid360"}],
        "movement_sensor": "mid360-imu",
        "kinematics": "differential",
        "max_vel_x": 0.5,
    }
    cfg = ExternalNavConfig.from_dict(d)
    # bridge SlamConfig carries the sensor deps
    assert [l.name for l in cfg.bridge.lidars] == ["mid360"]
    assert cfg.bridge.movement_sensor == "mid360-imu"
    # nav NavConfig carries navigation behavior + the SLAM dep name
    assert cfg.nav.slam_service == "rtabmap"
    assert cfg.nav.base == "base"
    assert cfg.nav.max_vel_x == 0.5
    # reader flags default off (Position tar pit ignored)
    assert cfg.trust_movement_sensor_pose is False
    assert cfg.snap_heading is False


def test_external_nav_config_required_deps_union_dedup():
    d = {
        "slam_service": "rtabmap",
        "base": "base",
        "lidars": [{"name": "a"}, {"name": "b"}],
        "movement_sensor": "imu",
    }
    deps = ExternalNavConfig.from_dict(d).required_dependencies()
    # slam_service + base + lidars + movement_sensor, no duplicates
    assert deps[0] == "rtabmap"
    assert set(deps) == {"rtabmap", "base", "a", "b", "imu"}
    assert len(deps) == len(set(deps))


def test_external_nav_config_reader_flags_parse():
    d = {
        "slam_service": "s",
        "base": "base",
        "lidars": [{"name": "l"}],
        "trust_movement_sensor_pose": True,
        "snap_heading": True,
    }
    cfg = ExternalNavConfig.from_dict(d)
    assert cfg.trust_movement_sensor_pose is True
    assert cfg.snap_heading is True


def test_slam_config_external_nav_tunables_default_and_override():
    # These back navigation-external; must actually parse (were dead getattrs).
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "l"})
    assert cfg.external_pose_rate_hz == 10.0
    assert cfg.external_grid_rate_hz == 1.5
    assert cfg.external_transform_timeout_s == 0.2
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "l",
            "external_pose_rate_hz": 5.0,
            "external_grid_rate_hz": 0.5,
            "external_transform_timeout_s": 0.3,
        }
    )
    assert cfg.external_pose_rate_hz == 5.0
    assert cfg.external_grid_rate_hz == 0.5
    assert cfg.external_transform_timeout_s == 0.3

import math

import pytest

from src.config import (
    DIFFERENTIAL,
    OMNI,
    ExternalNavConfig,
    NavConfig,
    SlamConfig,
    ros_cmd_vel_to_viam_linear_mm_s,
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
    assert cfg.slam_toolbox.minimum_travel_distance == pytest.approx(0.15)
    assert cfg.slam_toolbox.minimum_travel_heading == pytest.approx(0.12)
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
    assert cfg.lidars[0].shm_name is None
    assert cfg.lidars[0].shm_required is False


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
    assert cfg.slam_toolbox.minimum_travel_distance == pytest.approx(0.0)
    # Real slam_toolbox matcher knobs (use_odometry / use_tf_* are not real params).
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
            "slam_toolbox": {
                "resolution": 0.05,
                "minimum_travel_distance": 0.15,
                "minimum_travel_heading": 0.15,
            },
        }
    )
    assert cfg.slam_toolbox.minimum_travel_distance == pytest.approx(0.0)
    assert cfg.slam_toolbox.minimum_travel_heading == pytest.approx(0.0)


def test_slam_config_map_when_still_default_off_for_mir_style():
    """MiR-style laser scan lidars must keep continuous mapping defaults."""
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert cfg.map_when_still is False
    assert cfg.wall_yaw_correction is False
    assert cfg.mapping_revisit_check is False
    assert cfg.scan_accumulation_s == pytest.approx(0.0)


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
    assert cfg.periodic_relocalize_nav_interval_s == pytest.approx(15.0)
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


def test_slam_config_bad_mode():
    with pytest.raises(ValueError):
        SlamConfig.from_dict({"base": "b", "lidar": "f", "mode": "wat"})


def test_nav_config_defaults_and_deps():
    cfg = NavConfig.from_dict({"slam_service": "slam", "base": "b"})
    assert cfg.kinematics == DIFFERENTIAL
    assert cfg.required_dependencies() == ["slam", "b"]


def test_nav_config_omni():
    cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "kinematics": "omni", "max_vel_y": 0.3}
    )
    assert cfg.kinematics == OMNI
    assert cfg.max_vel_y == 0.3


def test_nav_config_bad_kinematics():
    with pytest.raises(ValueError):
        NavConfig.from_dict({"slam_service": "s", "base": "b", "kinematics": "legs"})


def test_slam_toolbox_config_from_attributes():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "f",
            "mode": "localizing",
            "slam_toolbox": {
                "resolution": 0.1,
                "max_laser_range": 30.0,
                "minimum_travel_distance": 0.5,
            },
        }
    )
    assert cfg.mode == "localizing"
    assert cfg.slam_toolbox.resolution == 0.1
    assert cfg.slam_toolbox.max_laser_range == 30.0
    assert cfg.slam_toolbox.minimum_travel_distance == 0.5


def test_slam_params_use_viam_config(tmp_path):
    from pathlib import Path

    from src.ros.manager import RosManager

    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "f",
            "maps_dir": str(tmp_path),
            "slam_toolbox": {"resolution": 0.08, "scan_topic": "/scan_merged"},
            "slam_params": {"map_update_interval": 2.0},
        }
    )
    params = RosManager(cfg)._slam_params(Path(tmp_path) / "map", "mapping")
    rp = params["slam_toolbox"]["ros__parameters"]
    assert rp["mode"] == "mapping"
    assert rp["resolution"] == 0.08
    assert rp["scan_topic"] == "/scan_merged"
    assert rp["map_update_interval"] == 2.0


def test_apply_nav2_tuning_only_updates_planner_tolerance():
    from src.models.navigation import _apply_nav2_tuning

    params = {
        "planner_server": {"ros__parameters": {"GridBased": {"tolerance": 0.5}}},
        "smoother_server": {
            "ros__parameters": {"simple_smoother": {"tolerance": 1.0e-10}}
        },
        "general_goal_checker": {"xy_goal_tolerance": 0.25},
    }
    _apply_nav2_tuning(
        params, {"tolerance": 0.35, "xy_goal_tolerance": 0.4, "robot_radius": 0.3}
    )
    assert params["planner_server"]["ros__parameters"]["GridBased"]["tolerance"] == 0.35
    assert (
        params["smoother_server"]["ros__parameters"]["simple_smoother"]["tolerance"]
        == 1.0e-10
    )
    assert params["general_goal_checker"]["xy_goal_tolerance"] == 0.4


def test_apply_velocity_limits_wires_mppi_and_smoother():
    from src.models.navigation import _apply_velocity_limits

    params = {
        "controller_server": {
            "ros__parameters": {
                "FollowPath": {"vx_max": 0.4, "vx_min": -0.4, "wz_max": 1.0}
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "max_velocity": [0.5, 0.0, 2.0],
                "max_accel": [2.5, 0.0, 3.2],
            }
        },
    }
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "kinematics": "differential",
            "max_vel_x": 0.35,
            "max_vel_theta": 0.8,
            "acc_lim_x": 0.5,
            "acc_lim_theta": 1.0,
        }
    )
    _apply_velocity_limits(params, cfg)

    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    assert fp["motion_model"] == "DiffDrive"
    assert fp["vx_max"] == 0.35
    assert fp["vx_min"] == -0.15
    assert fp["wz_max"] == 0.8
    assert fp["ax_max"] == 0.5
    vs = params["velocity_smoother"]["ros__parameters"]
    assert vs["max_velocity"] == [0.35, 0.0, 0.8]
    # Smoother reverse must match MPPI — not full -max_vel_x.
    assert vs["min_velocity"] == [-0.15, -0.0, -0.8]
    assert vs["max_accel"] == [0.5, 0.0, 1.0]
    assert vs["max_decel"][0] == -0.75


def test_apply_velocity_limits_reverse_scales_with_low_max_vel():
    from src.models.navigation import _apply_velocity_limits

    params = {
        "controller_server": {
            "ros__parameters": {"FollowPath": {}}
        },
        "velocity_smoother": {"ros__parameters": {}},
    }
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "kinematics": "differential",
            "max_vel_x": 0.08,
            "max_vel_theta": 0.4,
            "acc_lim_x": 0.3,
            "acc_lim_theta": 0.5,
        }
    )
    _apply_velocity_limits(params, cfg)
    assert params["controller_server"]["ros__parameters"]["FollowPath"]["vx_min"] == -0.08
    assert params["velocity_smoother"]["ros__parameters"]["min_velocity"][0] == -0.08


def test_sync_smoother_reverse_follows_user_vx_min_override():
    from src.models.navigation import _sync_smoother_reverse_to_mppi

    params = {
        "controller_server": {
            "ros__parameters": {"FollowPath": {"vx_min": 0.0}}
        },
        "velocity_smoother": {
            "ros__parameters": {"min_velocity": [-0.15, 0.0, -1.0]}
        },
    }
    _sync_smoother_reverse_to_mppi(params)
    assert params["velocity_smoother"]["ros__parameters"]["min_velocity"][0] == 0.0


def test_diffdrive_uses_regulated_pure_pursuit():
    from src.models.navigation import _apply_diffdrive_controller

    params = {
        "controller_server": {
            "ros__parameters": {
                "FollowPath": {
                    "plugin": "nav2_mppi_controller::MPPIController",
                    "vx_max": 0.4,
                    "PathAngleCritic": {"enabled": True},
                },
                "progress_checker": {
                    "required_movement_radius": 0.25,
                    "movement_time_allowance": 10.0,
                },
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "feedback": "CLOSED_LOOP",
                "min_velocity": [-0.4, 0.0, -1.0],
                "deadband_velocity": [0.03, 0.0, 0.05],
            }
        },
    }
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "kinematics": "differential",
            "max_vel_x": 0.25,
            "max_vel_theta": 1.5,
            "acc_lim_x": 0.4,
            "acc_lim_theta": 2.0,
        }
    )
    _apply_diffdrive_controller(params, cfg)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    assert "regulated_pure_pursuit" in fp["plugin"]
    assert fp["use_rotate_to_heading"] is False
    assert fp["allow_reversing"] is False
    assert fp["use_collision_detection"] is False
    assert fp["desired_linear_vel"] == 0.25
    assert fp["max_angular_vel"] == 1.5
    assert fp["min_linear_vel"] == -0.15
    assert fp["regulated_linear_scaling_min_speed"] == 0.125
    # Cart-scale regression guard: default robot_radius (0.22) must keep the
    # carpet-tested geometry exactly.
    assert fp["regulated_linear_scaling_min_radius"] == 0.9
    assert fp["lookahead_dist"] == 0.6
    assert fp["min_lookahead_dist"] == 0.3
    assert fp["max_lookahead_dist"] == 0.9
    assert fp["max_angular_accel"] == 2.0
    pc = params["controller_server"]["ros__parameters"]["progress_checker"]
    assert pc["required_movement_radius"] == 0.15
    assert pc["movement_time_allowance"] == 30.0
    vs = params["velocity_smoother"]["ros__parameters"]
    assert vs["min_velocity"][0] == -0.15
    assert vs["feedback"] == "OPEN_LOOP"
    assert vs["deadband_velocity"] == [0.0, 0.0, 0.0]


def _diffdrive_params_fixture():
    return {
        "controller_server": {
            "ros__parameters": {
                "FollowPath": {
                    "plugin": "nav2_mppi_controller::MPPIController",
                    "vx_max": 0.4,
                },
                "progress_checker": {
                    "required_movement_radius": 0.25,
                    "movement_time_allowance": 10.0,
                },
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "feedback": "CLOSED_LOOP",
                "min_velocity": [-0.4, 0.0, -1.0],
                "max_accel": [1.0, 0.0, 2.0],
                "max_decel": [-1.5, 0.0, -3.0],
                "deadband_velocity": [0.03, 0.0, 0.05],
            }
        },
    }


def test_diffdrive_small_base_profile():
    from src.models.navigation import _apply_diffdrive_controller

    params = _diffdrive_params_fixture()
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "viam_base",
            "kinematics": "differential",
            "robot_radius": 0.1,
        }
    )
    _apply_diffdrive_controller(params, cfg)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    # Speed floor must never inflate omega (RPP does omega = v * curvature
    # after flooring v): 0.10, not half of max_vel_x.
    assert fp["regulated_linear_scaling_min_speed"] == 0.10
    assert fp["regulated_linear_scaling_min_radius"] == 0.35
    # Velocity-scaled lookahead must not collapse to 0.3 m at low speed.
    assert fp["min_lookahead_dist"] == 0.45
    assert fp["lookahead_dist"] == 0.55
    # A small rover pivots cleanly; large heading errors rotate in place
    # instead of arcing tighter than the robot's own footprint.
    assert fp["use_rotate_to_heading"] is True
    assert fp["allow_reversing"] is False
    # Yaw slew must track RPP demand (4 * max_vel_theta), in RPP and smoother.
    assert fp["max_angular_accel"] == 4.0
    vs = params["velocity_smoother"]["ros__parameters"]
    assert vs["max_accel"] == [1.0, 0.0, 4.0]
    assert vs["max_decel"] == [-1.5, 0.0, -6.0]


def test_diffdrive_user_followpath_overrides_survive():
    from src.models.navigation import _apply_diffdrive_controller

    params = _diffdrive_params_fixture()
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "kinematics": "differential",
            "robot_radius": 0.1,
        }
    )
    user_params = {
        "controller_server": {
            "ros__parameters": {
                "FollowPath": {
                    "min_lookahead_dist": 0.6,
                    "regulated_linear_scaling_min_speed": 0.05,
                },
                "progress_checker": {"required_movement_radius": 0.3},
            }
        },
        "velocity_smoother": {"ros__parameters": {"feedback": "CLOSED_LOOP"}},
    }
    _apply_diffdrive_controller(params, cfg, user_params)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    assert "regulated_pure_pursuit" in fp["plugin"]
    assert fp["min_lookahead_dist"] == 0.6
    assert fp["regulated_linear_scaling_min_speed"] == 0.05
    pc = params["controller_server"]["ros__parameters"]["progress_checker"]
    assert pc["required_movement_radius"] == 0.3
    vs = params["velocity_smoother"]["ros__parameters"]
    assert vs["feedback"] == "CLOSED_LOOP"


def test_diffdrive_user_plugin_choice_skips_swap():
    from src.models.navigation import _apply_diffdrive_controller

    params = _diffdrive_params_fixture()
    cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "kinematics": "differential"}
    )
    user_params = {
        "controller_server": {
            "ros__parameters": {
                "FollowPath": {"plugin": "nav2_mppi_controller::MPPIController"}
            }
        }
    }
    _apply_diffdrive_controller(params, cfg, user_params)
    fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    assert fp["plugin"] == "nav2_mppi_controller::MPPIController"
    assert "regulated_linear_scaling_min_speed" not in fp
    # Explicit plugin choice means we leave the smoother alone too.
    vs = params["velocity_smoother"]["ros__parameters"]
    assert vs["feedback"] == "CLOSED_LOOP"


def test_diffdrive_controller_noop_for_omni():
    from src.models.navigation import _apply_diffdrive_controller

    params = {
        "controller_server": {
            "ros__parameters": {
                "FollowPath": {"plugin": "nav2_mppi_controller::MPPIController"}
            }
        }
    }
    cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "kinematics": "omni"}
    )
    _apply_diffdrive_controller(params, cfg)
    assert (
        params["controller_server"]["ros__parameters"]["FollowPath"]["plugin"]
        == "nav2_mppi_controller::MPPIController"
    )


def test_nav2_config_from_attributes():
    cfg = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "b",
            "nav2": {"xy_goal_tolerance": 0.4, "local_costmap_width": 6.0},
        }
    )
    assert cfg.nav2.xy_goal_tolerance == 0.4
    assert cfg.nav2.local_costmap_width == 6.0
    assert "width" not in cfg.nav2.to_override_dict()


def test_local_costmap_width_height_are_integers():
    from src.models.navigation import _apply_local_costmap_size

    params = {
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {"width": 4, "height": 4},
            }
        }
    }
    nav2 = NavConfig.from_dict(
        {"slam_service": "slam", "base": "b", "nav2": {"local_costmap_width": 6.0}}
    ).nav2
    _apply_local_costmap_size(params, nav2)
    lc = params["local_costmap"]["local_costmap"]["ros__parameters"]
    assert lc["width"] == 6
    assert lc["height"] == 4
    assert isinstance(lc["width"], int)
    assert isinstance(lc["height"], int)


def test_base_velocity_convention_viam_default():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f"})
    assert cfg.base_velocity_convention == "viam"
    lx, ly = ros_cmd_vel_to_viam_linear_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(-100.0)
    assert ly == pytest.approx(500.0)


def test_base_velocity_convention_ros_uses_x_forward():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "base_velocity_convention": "ros"}
    )
    assert cfg.base_velocity_convention == "ros"
    lx, ly = ros_cmd_vel_to_viam_linear_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(500.0)
    assert ly == pytest.approx(-100.0)


def test_base_velocity_convention_mir_alias_normalizes_to_viam():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "base_velocity_convention": "mir"}
    )
    assert cfg.base_velocity_convention == "viam"
    lx, ly = ros_cmd_vel_to_viam_linear_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(-100.0)
    assert ly == pytest.approx(500.0)


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


def test_nav2_template_lifecycle_manager_excludes_collision_monitor():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    params_file = Path(__file__).resolve().parents[1] / "params" / "nav2_params.yaml"
    data = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    nodes = data["lifecycle_manager_navigation"]["ros__parameters"]["node_names"]
    assert "collision_monitor" not in nodes



def test_normalize_nav2_user_params_wraps_missing_ros_parameters():
    from src.models.navigation import _normalize_nav2_user_params

    template = {
        "controller_server": {"ros__parameters": {"controller_frequency": 20.0}},
        "local_costmap": {
            "local_costmap": {"ros__parameters": {"width": 4}}
        },
    }
    user = {
        "controller_server": {"controller_frequency": 5.0},
        "local_costmap": {"width": 6},
    }

    normalized = _normalize_nav2_user_params(user, template)

    assert normalized["controller_server"] == {
        "ros__parameters": {"controller_frequency": 5.0}
    }
    assert normalized["local_costmap"] == {
        "local_costmap": {"ros__parameters": {"width": 6}}
    }


def test_normalize_nav2_user_params_keeps_wrapped_form():
    from src.models.navigation import _normalize_nav2_user_params

    template = {
        "controller_server": {"ros__parameters": {"controller_frequency": 20.0}},
    }
    user = {"controller_server": {"ros__parameters": {"controller_frequency": 5.0}}}

    normalized = _normalize_nav2_user_params(user, template)

    assert normalized == user


def test_normalize_nav2_user_params_coerces_types_to_template():
    # Viam attributes arrive via protobuf Structs: every number is a double.
    # bt_navigator declares default_server_timeout as int and dies on 200.0.
    from src.models.navigation import _normalize_nav2_user_params

    template = {
        "bt_navigator": {
            "ros__parameters": {
                "default_server_timeout": 20,
                "bt_loop_duration": 10,
                "action_server_result_timeout": 900.0,
                "use_sim_time": False,
            }
        },
    }
    user = {
        "bt_navigator": {
            "ros__parameters": {
                "default_server_timeout": 200.0,  # int in template
                "action_server_result_timeout": 300,  # float in template
                "use_sim_time": True,  # bool must survive untouched
                "unknown_param": 5.0,  # not in template: left as-is
            }
        },
    }

    rp = _normalize_nav2_user_params(user, template)["bt_navigator"][
        "ros__parameters"
    ]

    assert rp["default_server_timeout"] == 200
    assert isinstance(rp["default_server_timeout"], int)
    assert rp["action_server_result_timeout"] == 300.0
    assert isinstance(rp["action_server_result_timeout"], float)
    assert rp["use_sim_time"] is True
    assert rp["unknown_param"] == 5.0


def test_validate_nav2_params_structure_rejects_bad_tree():
    from src.models.navigation import _validate_nav2_params_structure

    with pytest.raises(ValueError, match="before 'ros__parameters'"):
        _validate_nav2_params_structure(
            {"controller_server": {"controller_frequency": 5.0}}
        )
    with pytest.raises(ValueError, match="outside"):
        _validate_nav2_params_structure(
            {
                "controller_server": {
                    "ros__parameters": {"a": 1},
                    "stray": 2,
                }
            }
        )


def test_validate_nav2_params_structure_accepts_template():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    from src.models.navigation import _validate_nav2_params_structure

    params_file = Path(__file__).resolve().parents[1] / "params" / "nav2_params.yaml"
    _validate_nav2_params_structure(yaml.safe_load(params_file.read_text()))


def test_normalize_nav2_user_params_relocates_plugin_sections():
    from src.models.navigation import _normalize_nav2_user_params

    template = {
        "controller_server": {
            "ros__parameters": {
                "controller_frequency": 20.0,
                "FollowPath": {"vx_max": 0.4, "vy_max": 0.0},
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "inflation_layer": {"inflation_radius": 0.45}
                }
            }
        },
        "global_costmap": {
            "global_costmap": {
                "ros__parameters": {
                    "inflation_layer": {"inflation_radius": 0.45}
                }
            }
        },
    }
    user = {
        "FollowPath": {"vy_max": 0.2},
        "inflation_layer": {"inflation_radius": 0.6},
    }

    normalized = _normalize_nav2_user_params(user, template)

    assert normalized["controller_server"]["ros__parameters"]["FollowPath"] == {
        "vy_max": 0.2
    }
    for costmap in ("local_costmap", "global_costmap"):
        assert (
            normalized[costmap][costmap]["ros__parameters"]["inflation_layer"][
                "inflation_radius"
            ]
            == 0.6
        )


def test_normalize_nav2_user_params_relocation_merges_with_node_override():
    from src.models.navigation import _normalize_nav2_user_params

    template = {
        "controller_server": {
            "ros__parameters": {
                "controller_frequency": 20.0,
                "FollowPath": {"vx_max": 0.4},
            }
        },
    }
    user = {
        "controller_server": {"controller_frequency": 10.0},
        "FollowPath": {"vx_max": 0.7},
    }

    normalized = _normalize_nav2_user_params(user, template)

    rp = normalized["controller_server"]["ros__parameters"]
    assert rp["controller_frequency"] == 10.0
    assert rp["FollowPath"] == {"vx_max": 0.7}


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
    # nav NavConfig carries Nav2 behavior + the SLAM dep name
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

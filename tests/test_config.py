import pytest

from src.config import DIFFERENTIAL, OMNI, NavConfig, SlamConfig, ros_cmd_vel_to_viam_linear_mm_s


def test_slam_config_single_lidar_string():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert [l.name for l in cfg.lidars] == ["front"]
    assert cfg.required_dependencies() == ["b", "front"]


def test_slam_config_scan_max_age_default_and_override():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert cfg.scan_max_age_s == 2.0
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front", "scan_max_age_s": 0.75})
    assert cfg.scan_max_age_s == 0.75


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
    assert vs["max_accel"] == [0.5, 0.0, 1.0]
    assert vs["max_decel"][0] == -0.75


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


def test_base_velocity_convention_ros_default():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "f"})
    assert cfg.base_velocity_convention == "ros"
    lx, ly = ros_cmd_vel_to_viam_linear_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(500.0)
    assert ly == pytest.approx(-100.0)


def test_base_velocity_convention_mir_swaps_axes():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "f", "base_velocity_convention": "mir"}
    )
    lx, ly = ros_cmd_vel_to_viam_linear_mm_s(0.5, -0.1, cfg.base_velocity_convention)
    assert lx == pytest.approx(-100.0)
    assert ly == pytest.approx(500.0)


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

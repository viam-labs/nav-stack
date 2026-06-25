import pytest

from src.config import DIFFERENTIAL, OMNI, NavConfig, SlamConfig, ros_cmd_vel_to_viam_linear_mm_s


def test_slam_config_single_lidar_string():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert [l.name for l in cfg.lidars] == ["front"]
    assert cfg.required_dependencies() == ["b", "front"]


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
    assert cfg.nav2.to_override_dict()["width"] == 6.0


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


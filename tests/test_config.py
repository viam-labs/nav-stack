import pytest

from src.config import DIFFERENTIAL, OMNI, NavConfig, SlamConfig


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

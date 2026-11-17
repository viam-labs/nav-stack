"""Vision config validation on SlamConfig."""

from __future__ import annotations

import pytest

from src.config import SlamConfig


def _base_lidars():
    return [
        {"name": "front-lidar", "scan_source": "get_laser_scan"},
        {
            "name": "depth-cam",
            "scan_source": "point_cloud",
            "obstacles_only": True,
            "cloud_frame": "camera_optical",
            "mount": {"x": 0.15, "z": 0.4},
        },
    ]


def test_vision_disabled_by_default():
    cfg = SlamConfig.from_dict({"base": "b", "lidars": _base_lidars()})
    assert cfg.vision_enabled is False
    assert cfg.vision_cameras == []
    assert cfg.vision_lidar_configs() == []


def test_vision_enabled_ok():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": _base_lidars(),
            "vision_enabled": True,
            "vision_cameras": ["depth-cam"],
            "vision_rate_hz": 2.0,
            "vision_survey_only": True,
        }
    )
    assert cfg.vision_enabled is True
    assert cfg.vision_cameras == ["depth-cam"]
    assert cfg.vision_survey_only is True
    cams = cfg.vision_lidar_configs()
    assert len(cams) == 1
    assert cams[0].obstacles_only is True


def test_vision_density_defaults():
    cfg = SlamConfig.from_dict({"base": "b", "lidars": _base_lidars()})
    assert cfg.vision_rate_hz == 2.0
    assert cfg.vision_round_robin is True
    assert cfg.vision_input_size == 448
    assert cfg.vision_survey_only is False
    # The camera-only pose applier is off; lidar-checked recovery is on.
    assert cfg.vision_recovery is True
    assert cfg.vision_place_enabled is False
    assert cfg.vision_place_retrieval is False
    assert cfg.vision_place_min_score == 0.55
    assert cfg.vision_place_max_jump_m == 0.75
    assert cfg.vision_yolo_conf == 0.10
    assert cfg.vision_detector_classes == []


def test_vision_detector_classes_from_config():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": _base_lidars(),
            "vision_detector_classes": ["painting", "door", "clock"],
        }
    )
    assert cfg.vision_detector_classes == ["painting", "door", "clock"]


def test_settings_of_the_removed_landmark_layer_are_ignored():
    """A robot config written for an earlier build still loads."""
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidars": _base_lidars(),
            "vision_pose_writeback": True,
            "vision_landmark_labels": ["painting"],
            "landmark_fix_max_range_m": 6.0,
        }
    )
    assert not hasattr(cfg, "vision_pose_writeback")
    assert not hasattr(cfg, "vision_landmark_labels")


def test_vision_requires_obstacles_only_optical():
    with pytest.raises(ValueError, match="obstacles_only"):
        SlamConfig.from_dict(
            {
                "base": "b",
                "lidars": [
                    {"name": "front-lidar"},
                    {
                        "name": "depth-cam",
                        "obstacles_only": False,
                        "cloud_frame": "camera_optical",
                        "scan_source": "point_cloud",
                    },
                ],
                "vision_enabled": True,
                "vision_cameras": ["depth-cam"],
            }
        )
    with pytest.raises(ValueError, match="cloud_frame"):
        SlamConfig.from_dict(
            {
                "base": "b",
                "lidars": [
                    {"name": "front-lidar"},
                    {
                        "name": "depth-cam",
                        "obstacles_only": True,
                        "cloud_frame": "sensor",
                        "scan_source": "point_cloud",
                    },
                ],
                "vision_enabled": True,
                "vision_cameras": ["depth-cam"],
            }
        )


def test_vision_enabled_needs_camera_list():
    with pytest.raises(ValueError, match="vision_cameras"):
        SlamConfig.from_dict(
            {
                "base": "b",
                "lidars": _base_lidars(),
                "vision_enabled": True,
            }
        )

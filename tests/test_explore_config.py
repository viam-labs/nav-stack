import pytest

from src.config import ExploreConfig


def test_explore_config_defaults():
    cfg = ExploreConfig.from_dict(
        {"slam_service": "slam", "navigation_service": "navigation", "base": "base"}
    )
    assert cfg.max_linear_mps == 0.4
    assert cfg.max_angular_dps == 30.0
    assert cfg.home_tolerance_m == 0.5
    assert cfg.structure_fraction == 0.15
    assert cfg.revisit_fraction == 0.1
    assert cfg.well_seen_m == 4.0
    assert cfg.keyframe_max == 2000
    assert cfg.required_dependencies() == ["slam", "navigation", "base"]


def test_explore_config_requires_names():
    with pytest.raises(ValueError):
        ExploreConfig.from_dict({"slam_service": "slam"})

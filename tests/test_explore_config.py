import pytest

from src.config import ExploreConfig


def test_explore_config_defaults():
    cfg = ExploreConfig.from_dict(
        {"slam_service": "slam", "navigation_service": "navigation", "base": "base"}
    )
    assert cfg.max_linear_mps == 0.4
    assert cfg.home_tolerance_m == 0.5
    assert cfg.min_frontier_m == 0.5
    assert cfg.required_dependencies() == ["slam", "navigation", "base"]


def test_explore_config_requires_names():
    with pytest.raises(ValueError):
        ExploreConfig.from_dict({"slam_service": "slam"})

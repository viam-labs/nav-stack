from src.config import NavConfig
from src.models.nav_core import nav_limits_for_status


def test_nav_limits_from_config():
    cfg = NavConfig.from_dict(
        {"slam_service": "slam", "base": "base", "max_linear_velocity": 0.5}
    )
    assert nav_limits_for_status(cfg) == {"max_linear_mps": 0.5, "max_angular_rad_s": 1.5}

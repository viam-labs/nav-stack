"""Overhead depth returns stay remembered after they leave the vertical view."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import LidarConfig, NavConfig
from src.geom.conversions import Pose2D
from src.nav_builtin.above_cart import AboveCartMemory

CAM = LidarConfig(
    name="camera",
    x=0.2,
    y=0.0,
    z=0.45,
    min_range=0.2,
    max_range=4.0,
    obstacles_only=True,
    fov_deg=87.0,
    vfov_deg=58.0,
)
# 0.70 m above the camera. In view at 2 m, above the image once ~0.45 m ahead.
BEAM = (2.2, 0.0, 1.15)


def _mem() -> AboveCartMemory:
    return AboveCartMemory(cart_height_m=1.0)


def test_overhead_return_is_kept_after_it_leaves_the_vertical_view():
    m = _mem()
    far = Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, [BEAM], CAM, far)], far, 1.0)
    assert len(m) == 1
    # Closed the gap: the beam is now above the camera's vertical field, and
    # this frame has no return. It must still block the spot ahead.
    close = Pose2D(1.55, 0.0, 0.0)
    m.update([(2.0, [], CAM, close)], close, 2.0)
    pts = m.points()
    assert len(pts) == 1
    assert np.allclose(pts[0], BEAM[:2], atol=0.02)


def test_empty_look_clears_a_height_the_camera_can_still_see():
    m = _mem()
    far = Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, [BEAM], CAM, far)], far, 1.0)
    m.update([(2.0, [], CAM, far)], far, 2.0)
    assert len(m) == 0


def test_same_frame_is_not_reapplied_after_the_robot_turns():
    m = _mem()
    far = Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, [BEAM], CAM, far)], far, 1.0)
    turned = Pose2D(0.0, 0.0, 1.2)
    m.update([(1.0, [], CAM, turned)], turned, 1.1)
    assert len(m) == 1


def test_forgets_when_far_or_old():
    m = _mem()
    far = Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, [BEAM], CAM, far)], far, 1.0)
    away = Pose2D(0.0, 6.0, 0.0)
    m.update([], away, 2.0)
    assert len(m) == 0
    m.update([(3.0, [BEAM], CAM, far)], far, 3.0)
    m.update([], far, 3.0 + 25.0)
    assert len(m) == 0


def test_without_cart_height_the_camera_band_is_remembered():
    """Unset height keeps every return, up to the cloud the camera already clipped."""
    m = AboveCartMemory()
    far = Pose2D(0.0, 0.0, 0.0)
    ankle = (2.2, 0.3, 0.4)
    m.update([(1.0, [BEAM, ankle], CAM, far)], far, 1.0)
    assert len(m) == 2
    close = Pose2D(1.55, 0.0, 0.0)
    m.update([(2.0, [], CAM, close)], close, 2.0)
    pts = m.points()
    # The high return left the vertical view. The ankle is still in view, so
    # the empty frame clears only that one.
    assert len(pts) == 1
    assert np.allclose(pts[0], BEAM[:2], atol=0.02)


def test_body_height_return_is_ignored():
    m = _mem()
    far = Pose2D(0.0, 0.0, 0.0)
    m.update([(1.0, [(2.7, 0.0, 0.4)], CAM, far)], far, 1.0)
    assert len(m) == 0


def test_avoid_obstacles_above_cart_defaults_on():
    nav = NavConfig.from_dict({"slam_service": "slam", "base": "base"})
    assert nav.avoid_obstacles_above_cart is True
    assert nav.cart_height_m is None
    set_h = NavConfig.from_dict(
        {
            "slam_service": "slam",
            "base": "base",
            "cart_height_m": 1.1,
            "avoid_obstacles_above_cart": False,
        }
    )
    assert set_h.avoid_obstacles_above_cart is False
    assert set_h.cart_height_m == pytest.approx(1.1)


def test_cart_height_rejects_non_positive():
    with pytest.raises(ValueError, match="cart_height_m"):
        NavConfig.from_dict(
            {"slam_service": "slam", "base": "base", "cart_height_m": -1}
        )

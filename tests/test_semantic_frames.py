"""Intrinsics / FOV helpers for the vision cameras."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.semantic.frames import (
    intrinsics_from_camera_properties,
    intrinsics_from_size,
)


def test_intrinsics_from_camera_properties_scales_resolution():
    props = SimpleNamespace(
        intrinsic_parameters=SimpleNamespace(
            width_px=1280,
            height_px=720,
            focal_x_px=920.0,
            focal_y_px=918.0,
            center_x_px=640.0,
            center_y_px=360.0,
        )
    )
    K = intrinsics_from_camera_properties(props, image_width=640, image_height=360)
    assert K is not None
    assert K.width == 640 and K.height == 360
    assert K.fx == pytest.approx(460.0)
    assert K.fy == pytest.approx(459.0)
    assert K.cx == pytest.approx(320.0)
    assert K.cy == pytest.approx(180.0)


def test_intrinsics_from_camera_properties_rejects_empty():
    props = SimpleNamespace(
        intrinsic_parameters=SimpleNamespace(
            width_px=0,
            height_px=0,
            focal_x_px=0.0,
            focal_y_px=0.0,
            center_x_px=0.0,
            center_y_px=0.0,
        )
    )
    assert (
        intrinsics_from_camera_properties(props, image_width=640, image_height=480)
        is None
    )


def test_intrinsics_use_separate_vfov():
    K = intrinsics_from_size(640, 480, fov_deg=69.0, vfov_deg=42.0)
    assert K.fx != pytest.approx(K.fy, rel=0.01)
    # Wider HFOV → smaller fx than a square-pixel fy from narrower VFOV?
    # 69° on 640 vs 42° on 480:
    fx_exp = (0.5 * 640) / math.tan(math.radians(34.5))
    fy_exp = (0.5 * 480) / math.tan(math.radians(21.0))
    assert K.fx == pytest.approx(fx_exp, rel=1e-6)
    assert K.fy == pytest.approx(fy_exp, rel=1e-6)

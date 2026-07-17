"""Tests for the ROS-free nav-camera renderer (src/ros/nav_view.py)."""
import io

import numpy as np
import pytest
from PIL import Image

from src.ros.nav_view import (
    NavViewOptions,
    _build_frame,
    _colorize,
    legend,
    legend_text,
    placeholder_png,
    render_nav_view,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _costmap(h=20, w=30, res=0.05, ox=1.0, oy=2.0):
    grid = np.full((h, w), -1, dtype=np.int16)
    grid[5:15, 10:20] = 0  # a free rectangle
    grid[0, 0] = 100  # a lethal cell
    grid[8, 12] = 50  # inflation
    return {"grid": grid, "resolution": res, "origin_x": ox, "origin_y": oy}


def _full_snapshot():
    return {
        "costmap": _costmap(),
        "global_plan": [(1.0, 2.0), (1.3, 2.2), (1.5, 2.3)],
        "local_plan": [(1.1, 2.1), (1.2, 2.15)],
        "plan_history": [[(1.0, 2.0), (1.4, 2.4)], [(1.0, 2.0), (1.35, 2.25)]],
        "footprint": [(1.2, 2.2), (1.25, 2.2), (1.25, 2.25), (1.2, 2.25)],
        "pose": (1.2, 2.2, 0.5),
        "goal": (1.5, 2.3, 1.0),
    }


def _decode(png: bytes) -> Image.Image:
    assert png[:8] == _PNG_MAGIC
    return Image.open(io.BytesIO(png))


def test_renders_valid_png_with_expected_dims():
    png = render_nav_view(_full_snapshot(), NavViewOptions(max_dim=700))
    img = _decode(png)
    # scale = 700 / max(30, 20) => out is 700 x round(20*700/30) = 700 x 467.
    assert img.size == (700, 467)
    assert img.mode in ("RGB", "RGBA")


def test_placeholder_when_no_grid():
    png = render_nav_view({}, NavViewOptions())
    img = _decode(png)
    assert img.size == (420, 300)


def test_placeholder_helper():
    img = _decode(placeholder_png("hello"))
    assert img.size[0] > 0


def test_falls_back_to_map_when_no_costmap():
    snap = {"map": _costmap()}
    img = _decode(render_nav_view(snap, NavViewOptions(max_dim=300)))
    assert img.size[0] == 300  # width is the long edge (w=30 > h=20)


def test_frame_world_to_pixel_flip():
    grid_info = _costmap(h=20, w=30, res=0.05, ox=1.0, oy=2.0)
    _img, frame = _build_frame(grid_info, 700)

    # Grid origin (world bottom-left) maps to pixel bottom-left (within 1px of
    # out_h, which is the rounded grid extent).
    px, py = frame.to_px(1.0, 2.0)
    assert px == pytest.approx(0.0, abs=1e-6)
    assert py == pytest.approx(frame.out_h, abs=1.0)

    # Far corner (world top-right) maps to pixel top-right; top edge is exact.
    px, py = frame.to_px(1.0 + 30 * 0.05, 2.0 + 20 * 0.05)
    assert px == pytest.approx(frame.out_w, abs=1.0)
    assert py == pytest.approx(0.0, abs=1e-6)

    # Increasing world y moves *up* the image (smaller pixel y): no vertical flip bug.
    _, low = frame.to_px(1.0, 2.1)
    _, high = frame.to_px(1.0, 2.4)
    assert high < low


def test_px_len_scales_with_resolution():
    _img, frame = _build_frame(_costmap(res=0.05), 700)
    # 0.10 m is two 0.05 m cells; px length is 2 * scale.
    assert frame.px_len(0.10) == pytest.approx(2 * frame.scale, rel=1e-6)


def test_colorize_value_buckets():
    grid = np.array([[-1, 0, 50, 100]], dtype=np.int16)
    rgb = _colorize(grid)
    assert tuple(rgb[0, 0]) == (70, 70, 70)  # unknown
    assert tuple(rgb[0, 1]) == (245, 245, 245)  # free
    assert tuple(rgb[0, 3]) == (35, 35, 40)  # lethal
    # inflation (50) is between the light and warm endpoints, not pure grey/black.
    mid = tuple(int(v) for v in rgb[0, 2])
    assert mid not in ((70, 70, 70), (245, 245, 245), (35, 35, 40))


def test_toggles_off_do_not_crash():
    opts = NavViewOptions(
        show_global_plan=False,
        show_local_plan=False,
        show_pose=False,
        show_footprint=False,
        show_goal=False,
        show_history=False,
    )
    _decode(render_nav_view(_full_snapshot(), opts))


def test_missing_optional_fields_are_safe():
    # Only a costmap, everything else absent/empty.
    snap = {"costmap": _costmap(), "global_plan": (), "plan_history": []}
    _decode(render_nav_view(snap, NavViewOptions()))


def test_legend_entries_are_serializable():
    entries = legend()
    assert len(entries) >= 8
    for e in entries:
        assert set(e) == {"label", "color", "rgb"}
        assert isinstance(e["color"], str) and e["color"]
        assert len(e["rgb"]) == 3 and all(0 <= v <= 255 for v in e["rgb"])
    # Colours match what the renderer draws (free space -> _C_FREE = 245,245,245).
    free = next(e for e in entries if e["label"] == "free space")
    assert free["rgb"] == [245, 245, 245]
    assert free["color"] == "white"


def test_legend_text_uses_color_names_not_hex():
    txt = legend_text()
    assert "legend" in txt.lower()
    assert txt.count("\n") >= len(legend())
    assert "#" not in txt  # rough colour names, no hex codes
    assert "green" in txt and "magenta" in txt


def test_footprint_fallback_circle_uses_pose():
    # No footprint published, but a pose exists -> should still render.
    snap = {"costmap": _costmap(), "pose": (1.2, 2.2, 0.0)}
    _decode(render_nav_view(snap, NavViewOptions(show_footprint=True)))

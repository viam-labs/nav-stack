"""Tests for the ROS-free nav-camera renderer (src/ros/nav_view.py)."""
import io

import numpy as np
import pytest
from PIL import Image

from src.ros.nav_view import (
    NavViewOptions,
    _build_frame,
    _colorize,
    _compute_window,
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


def test_window_full_is_default_and_none():
    assert _compute_window(_full_snapshot(), NavViewOptions(), _costmap()) is None


def test_window_follow_centers_on_pose():
    snap = {"costmap": _costmap(), "pose": (1.5, 2.3, 0.0)}
    opts = NavViewOptions(window_mode="follow", window_size_m=0.5)
    win = _compute_window(snap, opts, snap["costmap"])
    assert win == pytest.approx((1.25, 2.05, 1.75, 2.55))


def test_window_follow_falls_back_to_goal_then_grid_center():
    cm = _costmap(h=20, w=30, res=0.05, ox=1.0, oy=2.0)
    opts = NavViewOptions(window_mode="follow", window_size_m=1.0)
    # no pose -> use goal
    win = _compute_window({"costmap": cm, "goal": (2.0, 3.0, 0.0)}, opts, cm)
    assert win == pytest.approx((1.5, 2.5, 2.5, 3.5))
    # no pose or goal -> grid centre (ox + w*res/2, oy + h*res/2) = (1.75, 2.5)
    win = _compute_window({"costmap": cm}, opts, cm)
    assert win == pytest.approx((1.25, 2.0, 2.25, 3.0))


def test_window_region_bbox_normalized():
    cm = _costmap()
    opts = NavViewOptions(
        window_mode="region",
        window_min_x=1.6, window_min_y=2.4, window_max_x=1.2, window_max_y=2.1,
    )
    # min/max get sorted regardless of input order.
    assert _compute_window({"costmap": cm}, opts, cm) == pytest.approx((1.2, 2.1, 1.6, 2.4))


def test_window_region_missing_bounds_falls_back_to_full():
    cm = _costmap()
    opts = NavViewOptions(window_mode="region", window_min_x=1.2, window_min_y=2.1)
    assert _compute_window({"costmap": cm}, opts, cm) is None


def test_windowed_render_dims_and_alignment():
    cm = _costmap(h=20, w=30, res=0.05, ox=1.0, oy=2.0)
    # 0.4m x 0.3m region -> 8 x 6 cells.
    win = (1.2, 2.1, 1.6, 2.4)
    img, frame = _build_frame(cm, 300, win)
    # long edge (8 cells) scaled to 300.
    assert img.size == (300, 225)
    assert frame.out_w == 300 and frame.out_h == 225
    # window centre maps near the image centre.
    cx, cy = frame.to_px(1.4, 2.25)
    assert cx == pytest.approx(frame.out_w / 2, abs=frame.scale)
    assert cy == pytest.approx(frame.out_h / 2, abs=frame.scale)


def test_window_outside_grid_is_padded_not_crashing():
    cm = _costmap()
    # region entirely off the grid.
    _img, _frame = _build_frame(cm, 200, (100.0, 100.0, 101.0, 101.0))
    # follow mode render end-to-end with a pose near an edge.
    snap = {"costmap": cm, "pose": (1.0, 2.0, 0.0)}
    _decode(render_nav_view(snap, NavViewOptions(window_mode="follow", window_size_m=2.0)))


def test_footprint_fallback_circle_uses_pose():
    # No footprint published, but a pose exists -> should still render.
    snap = {"costmap": _costmap(), "pose": (1.2, 2.2, 0.0)}
    _decode(render_nav_view(snap, NavViewOptions(show_footprint=True)))

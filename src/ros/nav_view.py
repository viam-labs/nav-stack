"""Occupancy + plan renderer for the ``nav-camera`` component.

Turns a plain-Python snapshot (occupancy grid, plans, robot pose, goal) into a
PNG the Viam app can stream. Deliberately **ROS-free** and side-effect-free so
it unit-tests without a ROS install: the bridge collects the snapshot; this
module only draws it.

Coordinate model: occupancy grids are ``nav_msgs/OccupancyGrid``-style, row-major
starting at ``(origin_x, origin_y)`` with +y increasing up the rows. Images have
row 0 at the top, so the grid is flipped vertically once and every world->pixel
mapping uses the same flip -> world-up renders as image-up (matches rviz).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Cost values in a nav_msgs/OccupancyGrid costmap: -1 unknown, 0 free,
# 1..98 inflation gradient, 99 inscribed, 100 lethal. A raw SLAM /map uses the
# same convention with 100 = occupied, so the same colouring serves both.
_INSCRIBED_COST = 99

# Colours (RGB).
_C_FREE = (245, 245, 245)
_C_UNKNOWN = (70, 70, 70)
_C_LETHAL = (35, 35, 40)
_C_INFLATE_LO = (210, 210, 210)
_C_INFLATE_HI = (250, 120, 60)
_C_GLOBAL_PLAN = (40, 200, 90)
_C_LOCAL_PLAN = (255, 140, 0)
_C_HISTORY = (150, 150, 150)
_C_GOAL = (230, 60, 200)
_C_POSE = (230, 40, 40)
_C_FOOTPRINT = (60, 120, 255)
_C_PLACEHOLDER_BG = (40, 40, 45)
_C_PLACEHOLDER_FG = (200, 200, 200)


@dataclass
class NavViewOptions:
    """Render toggles for the nav-camera; mirrors the model's config."""

    max_dim: int = 700  # longest output edge in pixels
    show_global_plan: bool = True
    show_local_plan: bool = True
    show_pose: bool = True
    show_footprint: bool = True
    show_goal: bool = True
    show_history: bool = True
    robot_radius_m: float = 0.22  # fallback footprint / pose-arrow size


@dataclass
class _Frame:
    """Grid geometry + the pixel scale, shared by every world->pixel mapping."""

    origin_x: float
    origin_y: float
    resolution: float
    height: int  # grid rows
    scale: float  # output pixels per grid cell
    out_w: int
    out_h: int

    def to_px(self, wx: float, wy: float) -> Tuple[float, float]:
        col = (wx - self.origin_x) / self.resolution
        row = (wy - self.origin_y) / self.resolution
        # Flip about the grid's true (unrounded) pixel extent so world-top maps
        # to image y=0 exactly, independent of out_h rounding.
        return (col * self.scale, (self.height - row) * self.scale)

    def px_len(self, meters: float) -> float:
        return (meters / self.resolution) * self.scale


def _colorize(grid: np.ndarray) -> np.ndarray:
    """Map an (H, W) int occupancy grid to an (H, W, 3) uint8 RGB array."""
    g = np.asarray(grid)
    h, w = g.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)

    unknown = g < 0
    lethal = g >= _INSCRIBED_COST
    free = g == 0
    mid = ~unknown & ~lethal & ~free  # inflation gradient (1..98)

    rgb[free] = _C_FREE
    rgb[unknown] = _C_UNKNOWN
    rgb[lethal] = _C_LETHAL
    if np.any(mid):
        t = (g[mid].astype(np.float32) / float(_INSCRIBED_COST - 1)).clip(0.0, 1.0)
        for i in range(3):
            rgb[mid, i] = (
                _C_INFLATE_LO[i] + t * (_C_INFLATE_HI[i] - _C_INFLATE_LO[i])
            ).astype(np.uint8)
    return rgb


def _pick_background(snapshot: Dict) -> Optional[Dict]:
    """Prefer the global costmap (shows inflation the planner sees); else /map."""
    cm = snapshot.get("costmap")
    if cm is not None and cm.get("grid") is not None:
        return cm
    mp = snapshot.get("map")
    if mp is not None and mp.get("grid") is not None:
        return mp
    return None


def _build_frame(grid_info: Dict, max_dim: int) -> Tuple[Image.Image, _Frame]:
    grid = np.asarray(grid_info["grid"])
    h, w = grid.shape
    scale = max(1, int(max_dim)) / float(max(w, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))

    # Grid row 0 is world-bottom; flip so image row 0 is world-top (rviz-like).
    rgb = np.flipud(_colorize(grid))
    img = Image.fromarray(rgb, mode="RGB").resize((out_w, out_h), Image.NEAREST)

    frame = _Frame(
        origin_x=float(grid_info["origin_x"]),
        origin_y=float(grid_info["origin_y"]),
        resolution=float(grid_info["resolution"]),
        height=h,
        scale=scale,
        out_w=out_w,
        out_h=out_h,
    )
    return img, frame


def _poly_px(frame: _Frame, pts: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [frame.to_px(x, y) for (x, y) in pts]


def _draw_polyline(draw, frame, pts, color, width, alpha=255):
    if not pts or len(pts) < 2:
        return
    line = _poly_px(frame, pts)
    draw.line(line, fill=(*color, alpha), width=max(1, int(round(width))), joint="curve")


def _draw_pose(draw, frame, pose, radius_m, color):
    x, y, theta = pose
    px, py = frame.to_px(x, y)
    r = max(4.0, frame.px_len(radius_m))
    # Heading arrow: tip ahead, two tail corners. Image y is down, so the
    # world-frame direction is negated in y to point the right way on screen.
    tip = (px + r * math.cos(theta), py - r * math.sin(theta))
    left = (
        px + r * math.cos(theta + 2.5) * 0.7,
        py - r * math.sin(theta + 2.5) * 0.7,
    )
    right = (
        px + r * math.cos(theta - 2.5) * 0.7,
        py - r * math.sin(theta - 2.5) * 0.7,
    )
    draw.polygon([tip, left, right], fill=(*color, 255))


def _draw_goal(draw, frame, goal, color):
    x, y, theta = goal
    px, py = frame.to_px(x, y)
    r = 7.0
    draw.ellipse([px - r, py - r, px + r, py + r], outline=(*color, 255), width=3)
    draw.line(
        [(px, py), (px + 1.8 * r * math.cos(theta), py - 1.8 * r * math.sin(theta))],
        fill=(*color, 255),
        width=2,
    )


def _placeholder(text: str, size=(420, 300)) -> bytes:
    img = Image.new("RGB", size, _C_PLACEHOLDER_BG)
    draw = ImageDraw.Draw(img)
    draw.text((16, size[1] // 2 - 6), text, fill=_C_PLACEHOLDER_FG)
    return _encode(img)


def legend() -> List[Dict]:
    """The colour key for a rendered frame, as plain dicts (single source of
    truth for both the renderer and the nav-camera's ``legend`` DoCommand).

    ``color`` is a rough human name for what shows on screen; ``rgb`` is the
    exact value the renderer draws.
    """
    entries = [
        ("free space", "white", _C_FREE),
        ("unknown", "dark grey", _C_UNKNOWN),
        ("obstacle inflation (rising cost)", "light grey -> orange", _C_INFLATE_HI),
        ("lethal / inscribed obstacle", "near-black", _C_LETHAL),
        ("global plan", "green", _C_GLOBAL_PLAN),
        ("local plan", "orange", _C_LOCAL_PLAN),
        ("superseded plans (oldest->faintest)", "faded grey", _C_HISTORY),
        ("robot pose (heading arrow)", "red", _C_POSE),
        ("robot footprint", "blue", _C_FOOTPRINT),
        ("goal", "magenta", _C_GOAL),
    ]
    return [
        {"label": label, "color": color, "rgb": list(rgb)}
        for label, color, rgb in entries
    ]


def legend_text() -> str:
    """The colour key as a printable multi-line string."""
    lines = ["nav-camera legend:"]
    lines += [f"  {e['color']:<22}{e['label']}" for e in legend()]
    return "\n".join(lines)


def placeholder_png(text: str) -> bytes:
    """A standalone status image (e.g. navigation not running yet)."""
    return _placeholder(text)


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def render_nav_view(
    snapshot: Dict, options: Optional[NavViewOptions] = None
) -> bytes:
    """Render a nav-camera snapshot to PNG bytes.

    ``snapshot`` keys (all optional / may be ``None``/empty):
      ``costmap``/``map``: ``{grid, resolution, origin_x, origin_y}`` (grid is an
      (H, W) int array); ``global_plan``/``local_plan``: sequences of (x, y);
      ``plan_history``: list of such sequences, oldest first; ``footprint``:
      sequence of (x, y); ``pose``/``goal``: (x, y, theta). Coordinates are in
      the same map frame as the grid origin.
    """
    opts = options or NavViewOptions()
    bg = _pick_background(snapshot)
    if bg is None:
        return _placeholder("nav-camera: waiting for costmap / map...")

    img, frame = _build_frame(bg, opts.max_dim)
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Older plans first (faded), so the current plan draws on top.
    if opts.show_history:
        history = snapshot.get("plan_history") or []
        n = len(history)
        for i, plan in enumerate(history):
            alpha = int(50 + (100 * (i + 1) / n)) if n else 100
            _draw_polyline(draw, frame, plan, _C_HISTORY, 2, alpha=alpha)

    if opts.show_local_plan:
        _draw_polyline(draw, frame, snapshot.get("local_plan"), _C_LOCAL_PLAN, 2)
    if opts.show_global_plan:
        _draw_polyline(draw, frame, snapshot.get("global_plan"), _C_GLOBAL_PLAN, 3)

    pose = snapshot.get("pose")
    if opts.show_footprint:
        fp = snapshot.get("footprint")
        if fp and len(fp) >= 3:
            draw.polygon(
                _poly_px(frame, fp), outline=(*_C_FOOTPRINT, 255)
            )
        elif pose is not None:
            # No footprint published yet: draw the configured radius as a circle.
            px, py = frame.to_px(pose[0], pose[1])
            r = frame.px_len(opts.robot_radius_m)
            draw.ellipse(
                [px - r, py - r, px + r, py + r], outline=(*_C_FOOTPRINT, 255), width=2
            )

    if opts.show_goal and snapshot.get("goal") is not None:
        _draw_goal(draw, frame, snapshot["goal"], _C_GOAL)
    if opts.show_pose and pose is not None:
        _draw_pose(draw, frame, pose, opts.robot_radius_m, _C_POSE)

    return _encode(Image.alpha_composite(img, overlay))

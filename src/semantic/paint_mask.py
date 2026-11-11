"""Dynamic-object discs that gate lidar endpoints out of map paint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Disc:
    """Axis-aligned exclusion disc in base_link XY."""

    x: float
    y: float
    radius_m: float


def filter_scan_endpoints(
    scan_xy: np.ndarray,
    discs: Sequence[Disc],
) -> np.ndarray:
    """Drop endpoints whose XY falls inside any disc.

    ``scan_xy`` is (N, 2) base_link. Returns a copy with survivors only.
    """
    pts = np.asarray(scan_xy, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"scan_xy must be (N, 2), got {pts.shape}")
    if not discs:
        return pts[:, :2].copy()
    keep = np.ones(len(pts), dtype=bool)
    for disc in discs:
        r = max(0.0, float(disc.radius_m))
        if r <= 0.0:
            continue
        d2 = (pts[:, 0] - float(disc.x)) ** 2 + (pts[:, 1] - float(disc.y)) ** 2
        keep &= d2 > r * r
    return pts[keep, :2].copy()


def _disc_xyr(disc) -> Tuple[float, float, float]:
    if isinstance(disc, tuple):
        x, y, r = disc
    else:
        x, y, r = disc.x, disc.y, disc.radius_m
    return float(x), float(y), float(r)


def mask_scan_ranges(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    discs: Sequence,
) -> np.ndarray:
    """Blank ranges whose endpoint falls inside a dynamic-object disc.

    The endpoint is ``(r cos a, r sin a)``, the same frame ``insert_scan``
    paints. A blank ray is neither occupied nor free, so a person is not
    stored as a wall and the wall behind them is not erased.
    """
    out = np.asarray(ranges, dtype=float).copy()
    if out.size == 0 or not discs:
        return out
    packed = []
    for disc in discs:
        x, y, r = _disc_xyr(disc)
        if r > 0.0:
            packed.append((x, y, r))
    if not packed:
        return out
    ang = float(angle_min) + np.arange(out.size, dtype=float) * float(angle_increment)
    finite = np.isfinite(out) & (out > 0.05)
    xs = out * np.cos(ang)
    ys = out * np.sin(ang)
    drop = np.zeros(out.shape, dtype=bool)
    for x, y, r in packed:
        drop |= finite & ((xs - x) ** 2 + (ys - y) ** 2 <= r * r)
    out[drop] = np.nan
    return out


def detection_to_disc(
    *,
    u_center: float,
    v_center: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    box_width_px: float,
    mount_xy_theta: Tuple[float, float, float],
    min_radius_m: float = 0.25,
    mount_z: float = 0.0,
    mount_pitch: float = 0.0,
    mount_roll: float = 0.0,
) -> Disc:
    """Rough base_link disc from a detection center + depth (optical → base)."""
    from ..geom.conversions import (
        camera_optical_to_sensor_frame,
        transform_lidar_mount_to_base_link,
    )

    ox = (float(u_center) - float(cx)) / max(1.0, float(fx))
    oy = (float(v_center) - float(cy)) / max(1.0, float(fy))
    z = max(0.05, float(depth_m))
    optical = np.array([[ox * z, oy * z, z]], dtype=float)
    sensor = camera_optical_to_sensor_frame(optical)
    mx, my, mth = mount_xy_theta
    base = transform_lidar_mount_to_base_link(
        sensor,
        x=mx,
        y=my,
        z=float(mount_z),
        theta=mth,
        pitch=float(mount_pitch),
        roll=float(mount_roll),
    )[0]
    # Angular half-width → radius at this depth, floored.
    half_w = 0.5 * float(box_width_px) / max(1.0, float(fx))
    radius = max(float(min_radius_m), abs(half_w) * z)
    return Disc(x=float(base[0]), y=float(base[1]), radius_m=float(radius))


def discs_from_detections(
    frame,
    detections: Sequence,
    labels: Sequence[str],
    *,
    min_range_m: float = 0.4,
    max_range_m: float = 6.0,
    min_valid_px: int = 12,
) -> list:
    """Base-link discs for detections whose label is in ``labels``.

    Depth is the median of valid pixels in the box. A box with no depth,
    or a surface outside ``min_range_m``..``max_range_m``, makes no disc.
    """
    want = {str(x).strip().lower() for x in labels if str(x).strip()}
    depth = getattr(frame, "depth_m", None)
    if not want or depth is None:
        return []
    depth = np.asarray(depth)
    if depth.ndim != 2:
        return []
    h, w = depth.shape
    intrinsics = frame.intrinsics
    discs = []
    for det in detections:
        if str(det.label).strip().lower() not in want:
            continue
        x0 = int(np.clip(math.floor(min(float(det.x0), float(det.x1))), 0, w - 1))
        x1 = int(np.clip(math.ceil(max(float(det.x0), float(det.x1))), 0, w))
        y0 = int(np.clip(math.floor(min(float(det.y0), float(det.y1))), 0, h - 1))
        y1 = int(np.clip(math.ceil(max(float(det.y0), float(det.y1))), 0, h))
        if x1 <= x0 or y1 <= y0:
            continue
        win = depth[y0:y1, x0:x1]
        valid = win[np.isfinite(win) & (win > 0.05)]
        if int(valid.size) < int(min_valid_px):
            continue
        dist = float(np.median(valid))
        if dist < float(min_range_m) or dist > float(max_range_m):
            continue
        discs.append(
            detection_to_disc(
                u_center=0.5 * (float(det.x0) + float(det.x1)),
                v_center=0.5 * (float(det.y0) + float(det.y1)),
                depth_m=dist,
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.cx),
                cy=float(intrinsics.cy),
                box_width_px=abs(float(det.x1) - float(det.x0)),
                mount_xy_theta=(
                    float(frame.mount_x),
                    float(frame.mount_y),
                    float(frame.mount_theta),
                ),
                mount_z=float(frame.mount_z),
                mount_pitch=float(frame.mount_pitch),
                mount_roll=float(frame.mount_roll),
            )
        )
    return discs

"""Dynamic-object discs that gate lidar endpoints out of map paint."""

from __future__ import annotations

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
) -> Disc:
    """Rough base_link disc from a detection center + depth (optical → base).

    Used by the paint gate; full mount pitch/roll handled in patches path.
    """
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
        sensor, x=mx, y=my, z=0.0, theta=mth, pitch=0.0, roll=0.0
    )[0]
    # Angular half-width → radius at this depth, floored.
    half_w = 0.5 * float(box_width_px) / max(1.0, float(fx))
    radius = max(float(min_radius_m), abs(half_w) * z)
    return Disc(x=float(base[0]), y=float(base[1]), radius_m=float(radius))

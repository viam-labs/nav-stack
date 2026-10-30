"""Per-height map layers so one sensor's beam cannot clear another's hit.

A depth camera and a lidar share a 2D map but not a height. Raytracing them
as one scan lets the lidar, after the camera looks away, mark free every cell
up to its own farther hit. Each height band is raytraced alone. The published
grid is occupied wherever any band is occupied.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from ..geom import conversions as conv
from . import occupancy as occ
from .types import LogOddsGrid

# One hit is ``L_OCC`` (0.85). A band the sensor never looked at stays 0 and
# must not cancel another band's free or occupied value.
_OCC = 0.5
_FREE = -0.2
BAND_M = 0.25


def band_of_z(z: float) -> int:
    return int(math.floor(float(z) / BAND_M))


def band_scans(
    clouds: Iterable[np.ndarray],
    *,
    num_bins: int,
    range_min: float,
    range_max: float,
) -> Dict[int, conv.LaserScan2D]:
    """Closest return in each height band, across every cloud."""
    buckets: Dict[int, list] = {}
    for raw in clouds:
        pts = np.asarray(raw, dtype=float)
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 3:
            continue
        bands = np.floor(pts[:, 2] / BAND_M).astype(np.int32)
        for band in np.unique(bands):
            buckets.setdefault(int(band), []).append(pts[bands == band])
    out: Dict[int, conv.LaserScan2D] = {}
    for band, parts in buckets.items():
        stacked = np.vstack(parts)
        out[band] = conv.points_to_scan(
            stacked,
            angle_min=-math.pi,
            angle_max=math.pi,
            num_bins=num_bins,
            range_min=range_min,
            range_max=range_max,
        )
    return out


def rebase(
    grid: LogOddsGrid, origin_x: float, origin_y: float, height: int, width: int
) -> LogOddsGrid:
    """Copy ``grid`` into a canvas with the given origin and shape."""
    out = np.zeros((int(height), int(width)), dtype=np.float32)
    res = float(grid.resolution)
    dc = int(round((grid.origin_x - origin_x) / res))
    dr = int(round((grid.origin_y - origin_y) / res))
    src_r0 = max(0, -dr)
    src_c0 = max(0, -dc)
    dst_r0 = max(0, dr)
    dst_c0 = max(0, dc)
    h = min(grid.height - src_r0, int(height) - dst_r0)
    w = min(grid.width - src_c0, int(width) - dst_c0)
    if h > 0 and w > 0:
        out[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w] = grid.log_odds[
            src_r0 : src_r0 + h, src_c0 : src_c0 + w
        ]
    return LogOddsGrid(
        log_odds=out,
        resolution=res,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
    )


def cover_bands(bands: Mapping[int, LogOddsGrid]) -> Dict[int, LogOddsGrid]:
    """Give every band the same origin and shape."""
    if not bands:
        return {}
    res = float(next(iter(bands.values())).resolution)
    min_x = min(g.origin_x for g in bands.values())
    min_y = min(g.origin_y for g in bands.values())
    max_x = max(g.origin_x + g.width * g.resolution for g in bands.values())
    max_y = max(g.origin_y + g.height * g.resolution for g in bands.values())
    width = max(1, int(round((max_x - min_x) / res)))
    height = max(1, int(round((max_y - min_y) / res)))
    return {
        band: rebase(grid, min_x, min_y, height, width)
        for band, grid in bands.items()
    }


def compress_bands(bands: Mapping[int, LogOddsGrid]) -> LogOddsGrid:
    """Occupied if any band is occupied; otherwise free if any band is free."""
    aligned = cover_bands(bands)
    first = next(iter(aligned.values()))
    if len(aligned) == 1:
        return first
    stack = np.stack([g.log_odds for g in aligned.values()])
    occupied = stack >= _OCC
    free = stack <= _FREE
    any_occ = occupied.any(axis=0)
    any_free = free.any(axis=0)
    out = np.zeros(first.log_odds.shape, dtype=np.float32)
    out[any_occ] = stack.max(axis=0)[any_occ]
    rest = ~any_occ & any_free
    out[rest] = stack.min(axis=0)[rest]
    return LogOddsGrid(
        log_odds=out,
        resolution=first.resolution,
        origin_x=first.origin_x,
        origin_y=first.origin_y,
    )


def paint_bands(
    bands: Mapping[int, LogOddsGrid],
    clouds: Iterable[np.ndarray],
    pose: conv.Pose2D,
    *,
    resolution: float,
    num_bins: int,
    range_min: float,
    range_max: float,
    seed: Optional[LogOddsGrid] = None,
) -> Dict[int, LogOddsGrid]:
    """Raytrace ``clouds`` into ``bands``. Bands with no points this time stay."""
    scans = band_scans(
        clouds, num_bins=num_bins, range_min=range_min, range_max=range_max
    )
    if not scans:
        return dict(bands)
    painted: Dict[int, LogOddsGrid] = {
        band: occ.copy_grid(grid) for band, grid in bands.items()
    }
    template = next(iter(painted.values()), None)
    if template is None and seed is not None:
        template = seed
    for band, scan in scans.items():
        base = painted.get(band)
        if base is None:
            if template is not None:
                base = LogOddsGrid(
                    log_odds=np.zeros_like(template.log_odds),
                    resolution=template.resolution,
                    origin_x=template.origin_x,
                    origin_y=template.origin_y,
                )
            else:
                base = occ.empty_grid(resolution=resolution)
        painted[band] = occ.insert_scan(
            base,
            pose.x,
            pose.y,
            pose.theta,
            np.asarray(scan.ranges, dtype=float),
            float(scan.angle_min),
            float(scan.angle_increment),
            range_min=float(scan.range_min),
            range_max=float(scan.range_max) if math.isfinite(scan.range_max) else 30.0,
        )
    return cover_bands(painted)

"""Load / save map.yaml + map.pgm (+ last_pose.json) for builtin SLAM."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..geom import conversions as conv
from ..nav.global_localize import OccupancyMap, load_occupancy_from_map_dir
from . import occupancy as occ
from .types import LogOddsGrid

LAST_POSE_FILENAME = "last_pose.json"


def load_log_odds(map_dir: Path) -> Optional[LogOddsGrid]:
    om = load_occupancy_from_map_dir(map_dir)
    if om is None:
        return None
    return occ.from_occupancy_int16(
        om.grid,
        resolution=om.resolution,
        origin_x=om.origin_x,
        origin_y=om.origin_y,
    )


def load_occupancy_map(map_dir: Path) -> Optional[OccupancyMap]:
    return load_occupancy_from_map_dir(map_dir)


def last_pose_path(map_dir: Path) -> Path:
    return Path(map_dir) / LAST_POSE_FILENAME


def save_last_pose(map_dir: Path, pose: conv.Pose2D) -> None:
    """Atomic write of map-frame pose so a restart can resume near here."""
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": float(pose.x),
        "y": float(pose.y),
        "theta": float(pose.theta),
        "saved_unix": time.time(),
    }
    path = last_pose_path(map_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_last_pose(
    map_dir: Path, *, max_age_s: float = 0.0
) -> Optional[conv.Pose2D]:
    """Return the last saved pose, or None if missing/stale/corrupt.

    ``max_age_s`` <= 0 disables the age gate (always accept a valid file).
    """
    path = last_pose_path(map_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        x = float(data["x"])
        y = float(data["y"])
        theta = float(data["theta"])
        if not all(math.isfinite(v) for v in (x, y, theta)):
            return None
        if max_age_s > 0.0:
            saved = float(data.get("saved_unix", 0.0))
            if saved <= 0.0 or (time.time() - saved) > max_age_s:
                return None
        return conv.Pose2D(x, y, theta)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_occupancy(
    map_dir: Path,
    grid: LogOddsGrid,
    *,
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
) -> None:
    """Write Nav2-style ``map.yaml`` + ``map.pgm`` (image top-down)."""
    map_dir.mkdir(parents=True, exist_ok=True)
    int16 = occ.to_occupancy_int16(grid)
    # OccupancyGrid is bottom-up; map_server PGM is top-down.
    pgm_grid = np.flipud(int16)

    # Encode: unknown=205, free=254, occupied=0 (map_server convention, negate=0).
    pixels = np.full(pgm_grid.shape, 205, dtype=np.uint8)
    pixels[pgm_grid == 0] = 254
    pixels[pgm_grid >= 50] = 0
    mid = (pgm_grid > 0) & (pgm_grid < 50)
    if np.any(mid):
        # Intermediate costs -> greyscale between free and occupied.
        pixels[mid] = (254 - (pgm_grid[mid].astype(np.float32) / 100.0) * 254).astype(
            np.uint8
        )

    h, w = pixels.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    pgm_path = map_dir / "map.pgm"
    pgm_path.write_bytes(header + pixels.tobytes())

    yaml_text = (
        f"image: map.pgm\n"
        f"mode: trinary\n"
        f"resolution: {grid.resolution:.6f}\n"
        f"origin: [{grid.origin_x:.6f}, {grid.origin_y:.6f}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: {occupied_thresh}\n"
        f"free_thresh: {free_thresh}\n"
    )
    (map_dir / "map.yaml").write_text(yaml_text, encoding="utf-8")

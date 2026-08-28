"""Load / save map.yaml + map.pgm for builtin SLAM."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..nav.global_localize import OccupancyMap, load_occupancy_from_map_dir
from . import occupancy as occ
from .types import LogOddsGrid


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

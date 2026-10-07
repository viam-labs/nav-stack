"""Named-zones store (CRUD) + rasterizer, scoped to a single map.

Two zone types are supported:

* ``keepout``     - virtual no-go regions. Mask cells are 0 (free) or 100
                    (keepout). Builtin nav paints these as occupied before
                    costmap inflation so the planner routes around them.
* ``speed_limit`` - regions where the robot slows to ``speed_pct`` of max speed.
                    Mask cells hold the percentage (1-100); 0 means no limit.
                    Builtin nav scales linear velocity by this percentage.

Geometry is expressed in the map frame (meters). Supported shapes:

* ``{"type": "circle", "center": [x, y], "radius": r}``
* ``{"type": "box", "center": [x, y], "size": [w, h], "rotation": theta}``
  (``rotation`` optional, radians counterclockwise from +x)
* ``{"type": "polygon", "points": [[x, y], ...]}``

The design leaves room for additional zone types (e.g. directional/preferred-lane)
using the same masking plumbing.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KEEPOUT = "keepout"
SPEED_LIMIT = "speed_limit"
ZONE_TYPES = {KEEPOUT, SPEED_LIMIT}

from .naming import validate_name


def validate_zone_name(name: str) -> str:
    return validate_name(name, "zone")


def _validate_geometry(geometry: Dict) -> Dict:
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be an object")
    shape = geometry.get("type")
    if shape == "circle":
        if "center" not in geometry or "radius" not in geometry:
            raise ValueError("circle geometry requires 'center' and 'radius'")
    elif shape == "box":
        if "center" not in geometry or "size" not in geometry:
            raise ValueError("box geometry requires 'center' and 'size'")
    elif shape == "polygon":
        pts = geometry.get("points")
        if not pts or len(pts) < 3:
            raise ValueError("polygon geometry requires >= 3 'points'")
    else:
        raise ValueError(f"unsupported geometry type {shape!r}")
    return geometry


@dataclass
class Zone:
    name: str
    type: str
    geometry: Dict
    speed_pct: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Zone":
        return cls(
            name=d["name"],
            type=d["type"],
            geometry=d["geometry"],
            speed_pct=d.get("speed_pct"),
        )


class ZoneStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._zones: Dict[str, Zone] = {}
        self._load()

    def _load(self) -> None:
        self._zones = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raw = []
            for item in raw:
                z = Zone.from_dict(item)
                self._zones[z.name] = z

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [z.to_dict() for z in self._zones.values()]
        self.path.write_text(json.dumps(data, indent=2))

    # -- CRUD ----------------------------------------------------------------
    def add(
        self,
        name: str,
        type: str,
        geometry: Dict,
        speed_pct: Optional[float] = None,
    ) -> Zone:
        name = validate_zone_name(name)
        if name in self._zones:
            raise ValueError(f"zone {name!r} already exists")
        zone = self._make_zone(name, type, geometry, speed_pct)
        self._zones[name] = zone
        self._save()
        return zone

    @staticmethod
    def _make_zone(
        name: str, type: str, geometry: Dict, speed_pct: Optional[float]
    ) -> Zone:
        if type not in ZONE_TYPES:
            raise ValueError(f"zone type must be one of {sorted(ZONE_TYPES)}")
        _validate_geometry(geometry)
        if type == SPEED_LIMIT:
            if speed_pct is None:
                raise ValueError("speed_limit zones require 'speed_pct' (1-100)")
            if not (0 < float(speed_pct) <= 100):
                raise ValueError("speed_pct must be in (0, 100]")
        return Zone(name=name, type=type, geometry=geometry, speed_pct=speed_pct)

    def get(self, name: str) -> Zone:
        if name not in self._zones:
            raise KeyError(f"zone {name!r} not found")
        return self._zones[name]

    def list(self, type: Optional[str] = None) -> List[Zone]:
        zones = list(self._zones.values())
        if type is not None:
            zones = [z for z in zones if z.type == type]
        return zones

    def update(
        self,
        name: str,
        type: Optional[str] = None,
        geometry: Optional[Dict] = None,
        speed_pct: Optional[float] = None,
        new_name: Optional[str] = None,
    ) -> Zone:
        existing = self.get(name)
        new_type = type if type is not None else existing.type
        new_geom = geometry if geometry is not None else existing.geometry
        new_speed = speed_pct if speed_pct is not None else existing.speed_pct
        updated = self._make_zone(existing.name, new_type, new_geom, new_speed)
        if new_name is not None and new_name != name:
            new_name = validate_zone_name(new_name)
            if new_name in self._zones:
                raise ValueError(f"zone {new_name!r} already exists")
            del self._zones[name]
            updated.name = new_name
        self._zones[updated.name] = updated
        self._save()
        return updated

    def delete(self, name: str) -> None:
        if name not in self._zones:
            raise KeyError(f"zone {name!r} not found")
        del self._zones[name]
        self._save()

    def delete_all(self, type: Optional[str] = None) -> None:
        if type is None:
            self._zones.clear()
        else:
            self._zones = {n: z for n, z in self._zones.items() if z.type != type}
        self._save()


# ---------------------------------------------------------------------------
# Rasterization to costmap-filter masks
# ---------------------------------------------------------------------------
def _cell_centers(
    width: int, height: int, resolution: float, origin_x: float, origin_y: float
):
    xs = origin_x + (np.arange(width) + 0.5) * resolution
    ys = origin_y + (np.arange(height) + 0.5) * resolution
    return np.meshgrid(xs, ys)  # X, Y each shape (height, width)


def _mask_for_geometry(
    geometry: Dict, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    shape = geometry["type"]
    if shape == "circle":
        cx, cy = geometry["center"]
        r = float(geometry["radius"])
        return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    if shape == "box":
        cx, cy = geometry["center"]
        w, h = geometry["size"]
        rot = float(geometry.get("rotation", 0.0))
        # Transform cell centers into the box's local frame.
        c, s = math.cos(-rot), math.sin(-rot)
        lx = c * (X - cx) - s * (Y - cy)
        ly = s * (X - cx) + c * (Y - cy)
        return (np.abs(lx) <= w / 2.0) & (np.abs(ly) <= h / 2.0)
    if shape == "polygon":
        return _point_in_polygon(geometry["points"], X, Y)
    raise ValueError(f"unsupported geometry type {shape!r}")


def _point_in_polygon(points: Sequence[Sequence[float]], X: np.ndarray, Y: np.ndarray):
    """Vectorized even-odd ray-casting point-in-polygon test over a grid."""
    poly = np.asarray(points, dtype=float)
    n = len(poly)
    inside = np.zeros(X.shape, dtype=bool)
    px, py = X, Y
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= cond
        j = i
    return inside


def rasterize_zones(
    zones: Sequence[Zone],
    zone_type: str,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    """Rasterize zones of ``zone_type`` into a mask grid (row-major, int8).

    * keepout: cells are 0 (free) or 100 (no-go).
    * speed_limit: cells hold the speed percentage (1-100); 0 means no limit. When
      zones overlap, the most restrictive (lowest) non-zero speed wins.
    """
    X, Y = _cell_centers(width, height, resolution, origin_x, origin_y)
    mask = np.zeros((height, width), dtype=np.int16)

    for zone in zones:
        if zone.type != zone_type:
            continue
        covered = _mask_for_geometry(zone.geometry, X, Y)
        if zone_type == KEEPOUT:
            mask[covered] = 100
        elif zone_type == SPEED_LIMIT:
            val = int(round(float(zone.speed_pct)))
            current = mask[covered]
            # 0 means "unset"; otherwise keep the more restrictive (lower) value.
            replace = (current == 0) | (val < current)
            sel = np.zeros_like(mask, dtype=bool)
            sel[covered] = replace
            mask[sel] = val
    return mask.astype(np.int8)


def rasterize_for_map(
    zones: Sequence[Zone], map_data: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize keepout and speed masks for a bridge-style map dict."""
    grid = np.asarray(map_data["grid"])
    height, width = int(grid.shape[0]), int(grid.shape[1])
    res = float(map_data["resolution"])
    ox = float(map_data["origin_x"])
    oy = float(map_data["origin_y"])
    keepout = rasterize_zones(zones, KEEPOUT, width, height, res, ox, oy)
    speed = rasterize_zones(zones, SPEED_LIMIT, width, height, res, ox, oy)
    return keepout, speed


def apply_keepout_to_grid(grid: np.ndarray, keepout: np.ndarray) -> np.ndarray:
    """Return a copy of ``grid`` with keepout cells marked occupied (100)."""
    if keepout.shape != grid.shape:
        raise ValueError(
            f"keepout mask shape {keepout.shape} != occupancy shape {grid.shape}"
        )
    out = np.asarray(grid, dtype=np.int16).copy()
    out[np.asarray(keepout) >= 100] = 100
    return out


def speed_pct_at(
    speed_mask: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    x: float,
    y: float,
) -> Optional[float]:
    """Speed limit percent (1-100) at map-frame ``(x, y)``, or None if unset."""
    col = int((float(x) - float(origin_x)) / float(resolution))
    row = int((float(y) - float(origin_y)) / float(resolution))
    h, w = speed_mask.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    val = int(speed_mask[row, col])
    if val <= 0:
        return None
    return float(max(1, min(100, val)))


@dataclass
class ZoneMaskSet:
    """Rasterized keepout / speed masks aligned to one occupancy map."""

    keepout: np.ndarray
    speed: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def matches_map(self, map_data: dict) -> bool:
        grid = np.asarray(map_data["grid"])
        return (
            self.keepout.shape == grid.shape
            and self.speed.shape == grid.shape
            and abs(self.resolution - float(map_data["resolution"])) < 1e-9
            and abs(self.origin_x - float(map_data["origin_x"])) < 1e-9
            and abs(self.origin_y - float(map_data["origin_y"])) < 1e-9
        )

    def speed_pct_at(self, x: float, y: float) -> Optional[float]:
        return speed_pct_at(
            self.speed, self.resolution, self.origin_x, self.origin_y, x, y
        )


class ZoneMaskPublisher:
    """Thread-safe zone geometries + cached rasters for the active map."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._zones: List[Zone] = []
        self._masks: Optional[ZoneMaskSet] = None
        self._revision: int = 0

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def set_zones(self, zones: Sequence[Zone]) -> None:
        with self._lock:
            self._zones = list(zones)
            self._masks = None
            self._revision += 1

    def publish_masks(
        self,
        keepout_mask: np.ndarray,
        speed_mask: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        with self._lock:
            self._masks = ZoneMaskSet(
                keepout=np.asarray(keepout_mask),
                speed=np.asarray(speed_mask),
                resolution=float(resolution),
                origin_x=float(origin_x),
                origin_y=float(origin_y),
            )
            self._revision += 1

    def masks_for(self, map_data: Optional[dict]) -> Optional[ZoneMaskSet]:
        """Return masks for ``map_data``, re-rasterizing when the map changes."""
        if map_data is None or map_data.get("grid") is None:
            return None
        with self._lock:
            if self._masks is not None and self._masks.matches_map(map_data):
                return self._masks
            keepout, speed = rasterize_for_map(self._zones, map_data)
            self._masks = ZoneMaskSet(
                keepout=keepout,
                speed=speed,
                resolution=float(map_data["resolution"]),
                origin_x=float(map_data["origin_x"]),
                origin_y=float(map_data["origin_y"]),
            )
            return self._masks

    def zones(self) -> List[Zone]:
        with self._lock:
            return list(self._zones)

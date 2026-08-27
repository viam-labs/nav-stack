"""Multi-height-slice verification for mapping-time revisit matching.

The 2D occupancy map only contains the primary z-band silhouette, and office
furniture is self-similar in exactly that band — a wrong corridor can score
well against the map. The Livox cloud carries much more discriminating
structure at other heights (monitors and shelving at head height, open air
above low furniture), so we accumulate sparse per-band occupancy grids from
pause scans during a mapping session and require a revisit-correction pose to
also agree with every band that has reference data nearby.

Pure Python/numpy (no ROS) so it unit-tests like conversions.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..ros import conversions as conv


@dataclass(frozen=True)
class SliceBand:
    """Height band in base_link meters (z above floor)."""

    z_min: float
    z_max: float

    @classmethod
    def parse_bands(cls, raw: Sequence) -> List["SliceBand"]:
        bands: List[SliceBand] = []
        for pair in raw:
            lo, hi = float(pair[0]), float(pair[1])
            if hi <= lo:
                raise ValueError(f"slice band must have z_max > z_min, got {pair}")
            bands.append(cls(lo, hi))
        return bands


class SliceGrid:
    """Sparse occupied-cell grid for one height band, in map-frame cells."""

    def __init__(self, resolution_m: float = 0.15):
        self.resolution_m = float(resolution_m)
        self._cells: set = set()

    def __len__(self) -> int:
        return len(self._cells)

    def _to_cells(self, pts_xy: np.ndarray, pose: conv.Pose2D) -> np.ndarray:
        """Transform base_link XY points by ``pose`` and quantize to cells."""
        pts_xy = np.asarray(pts_xy, dtype=float)
        if pts_xy.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        mx = pose.x + c * pts_xy[:, 0] - s * pts_xy[:, 1]
        my = pose.y + s * pts_xy[:, 0] + c * pts_xy[:, 1]
        return np.stack(
            [
                np.floor(mx / self.resolution_m).astype(np.int64),
                np.floor(my / self.resolution_m).astype(np.int64),
            ],
            axis=1,
        )

    def add_points(self, pts_xy: np.ndarray, pose: conv.Pose2D) -> int:
        """Splat base_link points observed at map pose ``pose``; returns cell count."""
        cells = self._to_cells(pts_xy, pose)
        for ix, iy in cells:
            self._cells.add((int(ix), int(iy)))
        return len(self._cells)

    def cells_near(self, pose: conv.Pose2D, radius_m: float) -> int:
        """Stored cells within ``radius_m`` of ``pose`` (coverage probe)."""
        if not self._cells:
            return 0
        r_cells = radius_m / self.resolution_m
        cx = pose.x / self.resolution_m
        cy = pose.y / self.resolution_m
        count = 0
        for ix, iy in self._cells:
            if (ix - cx) ** 2 + (iy - cy) ** 2 <= r_cells * r_cells:
                count += 1
        return count

    def hit_rate(self, pts_xy: np.ndarray, pose: conv.Pose2D) -> Optional[float]:
        """Fraction of points landing on (or next to) a stored cell.

        Returns ``None`` when there are no query points. Neighborhood check is
        3x3 cells, i.e. tolerance ~= 1.5 * resolution.
        """
        cells = self._to_cells(pts_xy, pose)
        if cells.shape[0] == 0:
            return None
        hits = 0
        for ix, iy in cells:
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (int(ix) + dx, int(iy) + dy) in self._cells:
                        found = True
                        break
                if found:
                    break
            if found:
                hits += 1
        return hits / cells.shape[0]


def slice_points_by_bands(
    points: np.ndarray,
    bands: Sequence[SliceBand],
    *,
    max_points_per_band: int = 200,
) -> List[np.ndarray]:
    """Split a base_link cloud into per-band XY arrays (subsampled)."""
    points = np.asarray(points, dtype=float)
    out: List[np.ndarray] = []
    for band in bands:
        if points.size == 0 or points.shape[1] < 3:
            out.append(np.empty((0, 2)))
            continue
        z = points[:, 2]
        sel = points[(z >= band.z_min) & (z <= band.z_max)][:, :2]
        if max_points_per_band > 0 and sel.shape[0] > max_points_per_band:
            idx = np.linspace(
                0, sel.shape[0] - 1, max_points_per_band, dtype=np.int64
            )
            sel = sel[idx]
        out.append(sel)
    return out


class SliceLibrary:
    """Per-band reference grids accumulated over one mapping session."""

    def __init__(
        self,
        bands: Sequence[SliceBand],
        *,
        resolution_m: float = 0.15,
        min_hit_rate: float = 0.4,
        min_query_points: int = 30,
        min_reference_cells_near: int = 40,
        coverage_radius_m: float = 8.0,
    ):
        self.bands = list(bands)
        self.min_hit_rate = float(min_hit_rate)
        self.min_query_points = int(min_query_points)
        self.min_reference_cells_near = int(min_reference_cells_near)
        self.coverage_radius_m = float(coverage_radius_m)
        self._grids = [SliceGrid(resolution_m) for _ in self.bands]
        self.scans_recorded = 0

    def record(self, band_points: Sequence[np.ndarray], pose: conv.Pose2D) -> None:
        """Store a pause scan's band points at a trusted map pose."""
        for grid, pts in zip(self._grids, band_points):
            if np.asarray(pts).size:
                grid.add_points(pts, pose)
        self.scans_recorded += 1

    def verify(
        self, band_points: Sequence[np.ndarray], pose: conv.Pose2D
    ) -> Dict:
        """Check whether ``pose`` is consistent with stored band geometry.

        Per band: skipped when the current scan has too few points there or
        the library has too little reference data near ``pose`` (unknown area
        must not veto). Overall ``pass``: False if any checked band scores
        below ``min_hit_rate``; None when no band could be checked.
        """
        per_band: List[Dict] = []
        checked = 0
        failed = 0
        for band, grid, pts in zip(self.bands, self._grids, band_points):
            pts = np.asarray(pts, dtype=float)
            entry: Dict = {
                "band": [band.z_min, band.z_max],
                "query_points": int(pts.shape[0]) if pts.size else 0,
            }
            if pts.size == 0 or pts.shape[0] < self.min_query_points:
                entry["status"] = "too_few_points"
                per_band.append(entry)
                continue
            nearby = grid.cells_near(pose, self.coverage_radius_m)
            entry["reference_cells_near"] = nearby
            if nearby < self.min_reference_cells_near:
                entry["status"] = "no_reference_data"
                per_band.append(entry)
                continue
            rate = grid.hit_rate(pts, pose)
            entry["hit_rate"] = None if rate is None else round(rate, 3)
            checked += 1
            if rate is not None and rate < self.min_hit_rate:
                entry["status"] = "failed"
                failed += 1
            else:
                entry["status"] = "passed"
            per_band.append(entry)
        overall: Optional[bool]
        if checked == 0:
            overall = None
        else:
            overall = failed == 0
        return {
            "pass": overall,
            "bands_checked": checked,
            "bands_failed": failed,
            "per_band": per_band,
            "scans_recorded": self.scans_recorded,
        }

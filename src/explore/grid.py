"""Occupancy grid analysis for exploration: frontiers, visibility, coverage."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..slam_client import parse_get_grid

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


@dataclass
class Candidate:
    x: float
    y: float
    kind: str


@dataclass
class OccGrid:
    cells: np.ndarray
    resolution: float
    x_min: float
    y_min: float

    @classmethod
    def from_get_grid(cls, resp: Mapping) -> Optional["OccGrid"]:
        parsed = parse_get_grid(dict(resp))
        if parsed is None:
            return None
        rows, cols, cell_size, x_min, y_min, cells = parsed
        arr = np.asarray(cells, dtype=np.int8).reshape(rows, cols)
        return cls(cells=arr, resolution=cell_size, x_min=x_min, y_min=y_min)

    @property
    def rows(self) -> int:
        return int(self.cells.shape[0])

    @property
    def cols(self) -> int:
        return int(self.cells.shape[1])

    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor((y - self.y_min) / self.resolution + 1e-9)),
            int(math.floor((x - self.x_min) / self.resolution + 1e-9)),
        )

    def to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (
            self.x_min + (col + 0.5) * self.resolution,
            self.y_min + (row + 0.5) * self.resolution,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def known_free_count(self) -> int:
        return int(np.count_nonzero(self.cells == FREE))

    def free_mask(self) -> np.ndarray:
        return self.cells == FREE

    def occupied_mask(self) -> np.ndarray:
        return self.cells >= 50

    def unknown_mask(self) -> np.ndarray:
        return self.cells < 0


def _disk_offsets(radius_cells: int) -> List[Tuple[int, int]]:
    r = int(radius_cells)
    return [
        (dr, dc)
        for dr in range(-r, r + 1)
        for dc in range(-r, r + 1)
        if dr * dr + dc * dc <= r * r
    ]


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    for dr, dc in _disk_offsets(radius_cells):
        src = mask[
            max(0, -dr) : rows - max(0, dr),
            max(0, -dc) : cols - max(0, dc),
        ]
        out[
            max(0, dr) : rows - max(0, -dr),
            max(0, dc) : cols - max(0, -dc),
        ] |= src
    return out


def clear_mask(grid: OccGrid, clearance_m: float) -> np.ndarray:
    """Free cells with no occupied or unknown cell within ``clearance_m``."""
    radius = int(math.ceil(clearance_m / grid.resolution))
    blocked = ~grid.free_mask()
    return grid.free_mask() & ~_dilate(blocked, radius)


def _frontier_mask(grid: OccGrid) -> np.ndarray:
    free = grid.free_mask()
    unknown = grid.unknown_mask()
    padded = np.pad(unknown, 1, constant_values=False)
    near_unknown = (
        padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    )
    return free & near_unknown


def _clusters(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    clusters: List[List[Tuple[int, int]]] = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        queue = deque([(int(r0), int(c0))])
        cluster: List[Tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cluster.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        clusters.append(cluster)
    return clusters


def frontier_goals(
    grid: OccGrid, robot_radius_m: float, min_cluster_cells: int = 6
) -> List[Candidate]:
    clear = clear_mask(grid, robot_radius_m)
    clear_cells = np.argwhere(clear)
    if clear_cells.size == 0:
        return []
    search_cells = 1.0 / grid.resolution
    goals: List[Candidate] = []
    for cluster in _clusters(_frontier_mask(grid)):
        if len(cluster) < min_cluster_cells:
            continue
        centroid = np.mean(np.asarray(cluster, dtype=float), axis=0)
        d = np.hypot(clear_cells[:, 0] - centroid[0], clear_cells[:, 1] - centroid[1])
        i = int(np.argmin(d))
        if d[i] > search_cells:
            continue
        x, y = grid.to_world(int(clear_cells[i, 0]), int(clear_cells[i, 1]))
        goals.append(Candidate(x, y, "frontier"))
    return goals


def visible_new_cells(
    grid: OccGrid, x: float, y: float, max_range_m: float = 8.0, n_rays: int = 72
) -> int:
    r0, c0 = grid.to_cell(x, y)
    if not grid.in_bounds(r0, c0):
        return 0
    step = grid.resolution / 2.0
    steps = int(max_range_m / step)
    seen = set()
    occupied = grid.occupied_mask()
    unknown = grid.unknown_mask()
    for k in range(n_rays):
        ang = 2.0 * math.pi * k / n_rays
        dx, dy = math.cos(ang), math.sin(ang)
        for s in range(1, steps + 1):
            r, c = grid.to_cell(x + dx * s * step, y + dy * s * step)
            if not grid.in_bounds(r, c) or occupied[r, c]:
                break
            if unknown[r, c]:
                seen.add((r, c))
    return len(seen)


def under_seen_goals(
    grid: OccGrid,
    visited: Sequence[Tuple[float, float]],
    well_seen_m: float,
    spacing_m: float = 2.0,
) -> List[Candidate]:
    step = max(1, int(round(spacing_m / grid.resolution)))
    free = grid.free_mask()
    vis = np.asarray(visited, dtype=float).reshape(-1, 2)
    goals: List[Candidate] = []
    for r in range(step // 2, grid.rows, step):
        for c in range(step // 2, grid.cols, step):
            if not free[r, c]:
                continue
            x, y = grid.to_world(r, c)
            if vis.size and np.min(np.hypot(vis[:, 0] - x, vis[:, 1] - y)) <= well_seen_m:
                continue
            goals.append(Candidate(x, y, "under_seen"))
    return goals

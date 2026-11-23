"""Occupancy grid analysis for exploration: reachable frontiers from one wavefront.

One pass per decision: a wavefront from the robot over traversable known-free
cells gives the approximate path length to every reachable cell, and the
frontier (known free cells next to unknown space) is clustered and scored from
that same field. No per-candidate path planning.
"""

from __future__ import annotations

import base64
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import numpy as np

from ..slam_client import parse_get_grid

UNKNOWN = -1
FREE = 0
OCCUPIED = 100
OCCUPIED_THRESHOLD = 50


@dataclass
class OccGrid:
    cells: np.ndarray
    resolution: float
    x_min: float
    y_min: float

    @classmethod
    def from_get_grid(cls, resp: Mapping) -> Optional["OccGrid"]:
        data = resp.get("data") if isinstance(resp, Mapping) else None
        if isinstance(data, (str, bytes, bytearray)):
            try:
                rows, cols = int(resp["rows"]), int(resp["cols"])
                cell_size = float(resp.get("cellSize", resp.get("cell_size")))
                x_min = float(resp.get("xMin", resp.get("x_min")))
                y_min = float(resp.get("yMin", resp.get("y_min")))
                raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
            except (KeyError, TypeError, ValueError):
                return None
            if rows <= 0 or cols <= 0 or cell_size <= 0 or len(raw) != rows * cols:
                return None
            arr = np.clip(np.frombuffer(raw, dtype=np.int8), -1, 100).reshape(
                rows, cols
            )
            return cls(cells=arr, resolution=cell_size, x_min=x_min, y_min=y_min)
        parsed = parse_get_grid(dict(resp))
        if parsed is None:
            return None
        rows, cols, cell_size, x_min, y_min, cells = parsed
        arr = np.asarray(cells, dtype=np.int8).reshape(rows, cols)
        return cls(cells=arr, resolution=cell_size, x_min=x_min, y_min=y_min)

    @classmethod
    def from_map_dict(cls, m: Optional[Mapping]) -> Optional["OccGrid"]:
        """Builtin SLAM's in-process ``get_map()`` dict (-1 unknown, 0..100)."""
        if not isinstance(m, Mapping) or m.get("grid") is None:
            return None
        try:
            cells = np.clip(np.asarray(m["grid"]), -1, 100).astype(np.int8)
            res = float(m["resolution"])
            x0, y0 = float(m["origin_x"]), float(m["origin_y"])
        except (KeyError, TypeError, ValueError):
            return None
        if cells.ndim != 2 or res <= 0:
            return None
        return cls(cells=cells, resolution=res, x_min=x0, y_min=y0)

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
        return int(np.count_nonzero(self.free_mask()))

    def free_mask(self) -> np.ndarray:
        return (self.cells >= 0) & (self.cells < OCCUPIED_THRESHOLD)

    def occupied_mask(self) -> np.ndarray:
        return self.cells >= OCCUPIED_THRESHOLD

    def unknown_mask(self) -> np.ndarray:
        return self.cells < 0

    def cropped(self, include: Tuple[float, float], pad_cells: int = 2) -> "OccGrid":
        """Smallest window holding every known cell and ``include``, plus a pad.

        SLAM grids are a fixed canvas much larger than the mapped area; the
        wavefront only needs the part that has been seen.
        """
        known = self.cells >= 0
        rows_any = np.flatnonzero(known.any(axis=1))
        cols_any = np.flatnonzero(known.any(axis=0))
        r, c = self.to_cell(*include)
        if rows_any.size == 0:
            r_lo = r_hi = r
            c_lo = c_hi = c
        else:
            r_lo, r_hi = min(int(rows_any[0]), r), max(int(rows_any[-1]), r)
            c_lo, c_hi = min(int(cols_any[0]), c), max(int(cols_any[-1]), c)
        r_lo, c_lo = max(0, r_lo - pad_cells), max(0, c_lo - pad_cells)
        r_hi, c_hi = (
            min(self.rows, r_hi + pad_cells + 1),
            min(self.cols, c_hi + pad_cells + 1),
        )
        if r_lo >= r_hi or c_lo >= c_hi:
            return self
        return OccGrid(
            cells=self.cells[r_lo:r_hi, c_lo:c_hi],
            resolution=self.resolution,
            x_min=self.x_min + c_lo * self.resolution,
            y_min=self.y_min + r_lo * self.resolution,
        )


def _disk_offsets(radius_cells: int) -> List[Tuple[int, int]]:
    r = int(radius_cells)
    return [
        (dr, dc)
        for dr in range(-r, r + 1)
        for dc in range(-r, r + 1)
        if dr * dr + dc * dc <= r * r
    ]


def _shift_or(out: np.ndarray, src: np.ndarray, dr: int, dc: int) -> None:
    rows, cols = src.shape
    out[max(0, dr) : rows - max(0, -dr), max(0, dc) : cols - max(0, -dc)] |= src[
        max(0, -dr) : rows - max(0, dr), max(0, -dc) : cols - max(0, dc)
    ]


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    out = mask.copy()
    for dr, dc in _disk_offsets(radius_cells):
        if dr or dc:
            _shift_or(out, mask, dr, dc)
    return out


def traversable_mask(grid: OccGrid, robot_radius_m: float) -> np.ndarray:
    """Known free cells farther than ``robot_radius_m`` from any occupied cell.

    Unknown space is not inflated (nav's costmap does the same), so cells right
    at the frontier stay usable as goals.
    """
    radius = int(math.ceil(robot_radius_m / grid.resolution))
    return grid.free_mask() & ~_dilate(grid.occupied_mask(), radius)


def wavefront(trav: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """Ring index from ``start`` over ``trav`` (-1 where unreachable).

    Alternating 4- and 8-connected rings grow an octagon, so ring count times
    resolution tracks Euclidean path length within about 6 %. Each step only
    expands the previous ring's cells, so the cost follows the reached area.
    """
    rows, cols = trav.shape
    r0, c0 = start
    if not (0 <= r0 < rows and 0 <= c0 < cols) or not trav[r0, c0]:
        return np.full(trav.shape, -1, dtype=np.int32)
    w = cols + 2  # one-cell untraversable border: neighbours never leave the array
    ok = np.zeros((rows + 2, w), dtype=bool)
    ok[1:-1, 1:-1] = trav
    ok = ok.ravel()
    dist = np.full(ok.size, -1, dtype=np.int32)
    axis = np.array([1, -1, w, -w])
    both = np.concatenate([axis, [w + 1, w - 1, -w + 1, -w - 1]])
    ring = np.array([(r0 + 1) * w + c0 + 1])
    dist[ring] = 0
    k = 0
    while ring.size:
        k += 1
        nb = (ring[:, None] + (both if k % 2 == 0 else axis)[None, :]).ravel()
        nb = nb[ok[nb]]
        nb = np.unique(nb[dist[nb] < 0])
        dist[nb] = k
        ring = nb
    return dist.reshape(rows + 2, w)[1:-1, 1:-1].copy()


def frontier_mask(grid: OccGrid, fill_cells: int = 0) -> np.ndarray:
    """Known free cells with an unknown 4-neighbour.

    ``fill_cells`` > 0 first closes unknown holes up to about twice that many
    cells wide (gaps between lidar rays, speckle), so only real openings into
    unseen space count.
    """
    unknown = grid.unknown_mask()
    if fill_cells > 0:
        known = _dilate(~unknown, fill_cells)
        unknown = _dilate(~known, fill_cells)  # closing: known grows, then shrinks back
    padded = np.pad(unknown, 1, constant_values=False)
    near_unknown = (
        padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    )
    return grid.free_mask() & near_unknown


def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected component labels (0 = background). Mask cells are few."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    rows, cols = mask.shape
    n = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if labels[r0, c0]:
            continue
        n += 1
        labels[r0, c0] = n
        queue = deque([(int(r0), int(c0))])
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and mask[nr, nc]
                        and not labels[nr, nc]
                    ):
                        labels[nr, nc] = n
                        queue.append((nr, nc))
    return labels, n


def _grow_labels(
    labels: np.ndarray, allowed: np.ndarray, steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Spread each label up to ``steps`` cells (8-connected) over ``allowed``.

    Returns the grown labels and each labelled cell's step count (0 = seed).
    """
    rows, cols = labels.shape
    w = cols + 2
    ok = np.zeros((rows + 2, w), dtype=bool)
    ok[1:-1, 1:-1] = allowed
    ok = ok.ravel()
    lab = np.zeros((rows + 2, w), dtype=np.int32)
    lab[1:-1, 1:-1] = labels
    lab = lab.ravel()
    depth = np.full(lab.size, -1, dtype=np.int32)
    offs = np.array([1, -1, w, -w, w + 1, w - 1, -w + 1, -w - 1])
    ring = np.flatnonzero(lab)
    depth[ring] = 0
    for step in range(1, steps + 1):
        if ring.size == 0:
            break
        nb = (ring[:, None] + offs[None, :]).ravel()
        src = np.repeat(lab[ring], offs.size)
        keep = ok[nb] & (lab[nb] == 0)
        nb, src = nb[keep], src[keep]
        nb, first = np.unique(nb, return_index=True)
        lab[nb] = src[first]
        depth[nb] = step
        ring = nb
    inner = (slice(1, -1), slice(1, -1))
    return lab.reshape(rows + 2, w)[inner].copy(), depth.reshape(rows + 2, w)[
        inner
    ].copy()


@dataclass
class Frontier:
    """One reachable frontier cluster.

    ``goal_*`` is where nav is sent: of the reachable cells closest to the
    frontier, the one nearest the robot. ``x``/``y`` is the frontier cell
    nearest that goal (what the goal is meant to reveal).
    """

    goal_x: float
    goal_y: float
    x: float
    y: float
    size_m: float
    path_m: float


@dataclass
class Waypoint:
    """Where to send nav next on the way to a frontier goal.

    ``theta`` is the direction of travel at the waypoint, so a robot that does
    reach it is already facing along the rest of the route. ``final`` means
    the waypoint is the frontier goal itself.
    """

    x: float
    y: float
    theta: float
    final: bool


# Neighbour offsets for walking the wavefront back toward the robot.
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass
class FrontierScan:
    grid: OccGrid
    frontier: np.ndarray
    rings: np.ndarray
    frontiers: List[Frontier] = field(default_factory=list)
    start_found: bool = True
    # Traversable cells with extra room from walls, preferred for waypoints.
    roomy: Optional[np.ndarray] = None

    def frontier_near(self, x: float, y: float, radius_m: float) -> bool:
        g = self.grid
        r, c = g.to_cell(x, y)
        k = int(math.ceil(radius_m / g.resolution))
        r_lo, r_hi = max(0, r - k), min(g.rows, r + k + 1)
        c_lo, c_hi = max(0, c - k), min(g.cols, c + k + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return False
        rr, cc = np.nonzero(self.frontier[r_lo:r_hi, c_lo:c_hi])
        if rr.size == 0:
            return False
        return bool(np.min((rr + r_lo - r) ** 2 + (cc + c_lo - c) ** 2) <= k * k)

    def _reached_cell(
        self, x: float, y: float, search_m: float = 0.3
    ) -> Optional[Tuple[int, int]]:
        """The reached cell nearest (x, y) within ``search_m``."""
        g = self.grid
        r, c = g.to_cell(x, y)
        k = max(1, int(math.ceil(search_m / g.resolution)))
        r_lo, r_hi = max(0, r - k), min(g.rows, r + k + 1)
        c_lo, c_hi = max(0, c - k), min(g.cols, c + k + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return None
        rr, cc = np.nonzero(self.rings[r_lo:r_hi, c_lo:c_hi] >= 0)
        if rr.size == 0:
            return None
        i = int(np.argmin((rr + r_lo - r) ** 2 + (cc + c_lo - c) ** 2))
        return int(rr[i] + r_lo), int(cc[i] + c_lo)

    def path_m(self, x: float, y: float, search_m: float = 0.3) -> Optional[float]:
        """Approximate path length from the robot to (x, y), None if unreachable."""
        cell = self._reached_cell(x, y, search_m)
        if cell is None:
            return None
        return float(self.rings[cell]) * self.grid.resolution

    def _descend(self, cell: Tuple[int, int], stop_ring: int) -> List[Tuple[int, int]]:
        """Cells from ``cell`` back along the wavefront until ring <= stop_ring."""
        rings = self.rings
        rows, cols = rings.shape
        r, c = cell
        out = [cell]
        while rings[r, c] > stop_ring:
            here = rings[r, c]
            best = None
            for dr, dc in _STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and 0 <= rings[nr, nc] < here:
                    if best is None or rings[nr, nc] < rings[best]:
                        best = (nr, nc)
            if best is None:
                break
            r, c = best
            out.append(best)
        return out

    def waypoint(self, x: float, y: float, max_m: float) -> Optional[Waypoint]:
        """A point at most ``max_m`` of path from the robot toward (x, y).

        Walks the wavefront back from (x, y) to where the remaining path is
        ``max_m``, then prefers a nearby cell with more room from walls (the
        shortest route hugs corners). Short nav goals keep each of nav's plans
        short: plan time grows quickly with distance.
        """
        g = self.grid
        res = g.resolution
        goal = self._reached_cell(x, y)
        if goal is None:
            return None
        target = int(max_m / res)
        if self.rings[goal] <= target:
            # Final leg: face the way the route arrives.
            back = self._descend(goal, max(0, int(self.rings[goal]) - 10))
            bx, by = g.to_world(*back[-1])
            gx, gy = g.to_world(*goal)
            theta = math.atan2(gy - by, gx - bx) if back[-1] != goal else 0.0
            return Waypoint(x, y, theta, True)
        route = self._descend(goal, target)
        wr, wc = route[-1]
        if self.roomy is not None and not self.roomy[wr, wc]:
            k = max(1, int(round(0.4 / res)))
            r_lo, r_hi = max(0, wr - k), min(g.rows, wr + k + 1)
            c_lo, c_hi = max(0, wc - k), min(g.cols, wc + k + 1)
            ring_win = self.rings[r_lo:r_hi, c_lo:c_hi]
            ok = (
                self.roomy[r_lo:r_hi, c_lo:c_hi]
                & (ring_win >= 0)
                & (ring_win <= target)
            )
            rr, cc = np.nonzero(ok)
            if rr.size:
                i = int(np.argmin((rr + r_lo - wr) ** 2 + (cc + c_lo - wc) ** 2))
                wr, wc = int(rr[i] + r_lo), int(cc[i] + c_lo)
        ahead = route[max(0, len(route) - 11)]  # ~0.5 m further along toward the goal
        wx, wy = g.to_world(wr, wc)
        ax, ay = g.to_world(*ahead)
        theta = math.atan2(ay - wy, ax - wx) if (ax, ay) != (wx, wy) else 0.0
        return Waypoint(wx, wy, theta, False)


def _nearest_traversable(
    trav: np.ndarray, cell: Tuple[int, int], max_cells: int
) -> Optional[Tuple[int, int]]:
    r, c = cell
    rows, cols = trav.shape
    if 0 <= r < rows and 0 <= c < cols and trav[r, c]:
        return cell
    r_lo, r_hi = max(0, r - max_cells), min(rows, r + max_cells + 1)
    c_lo, c_hi = max(0, c - max_cells), min(cols, c + max_cells + 1)
    if r_lo >= r_hi or c_lo >= c_hi:
        return None
    rr, cc = np.nonzero(trav[r_lo:r_hi, c_lo:c_hi])
    if rr.size == 0:
        return None
    d2 = (rr + r_lo - r) ** 2 + (cc + c_lo - c) ** 2
    i = int(np.argmin(d2))
    if d2[i] > max_cells * max_cells:
        return None
    return int(rr[i] + r_lo), int(cc[i] + c_lo)


def scan_frontiers(
    grid: OccGrid,
    x: float,
    y: float,
    robot_radius_m: float,
    *,
    min_frontier_m: float = 0.5,
    start_search_m: float = 1.0,
    fill_m: float = 0.1,
    blind_m: Optional[float] = None,
) -> FrontierScan:
    """Reachable frontier clusters around the robot at (x, y), nearest first.

    Frontier cells within ``blind_m`` of the robot (default: its radius plus
    0.2 m) are ignored: that ring is the body and the lidar's minimum range,
    which the next few centimetres of motion reveal anyway.
    """
    res = grid.resolution
    fill_cells = int(round(fill_m / res))
    g = grid.cropped((x, y), pad_cells=fill_cells + 2)
    trav = traversable_mask(g, robot_radius_m)
    roomy = traversable_mask(g, robot_radius_m + 0.2)
    front = frontier_mask(g, fill_cells)
    blind = int(math.ceil((robot_radius_m + 0.2 if blind_m is None else blind_m) / res))
    r0, c0 = g.to_cell(x, y)
    rr0, cc0 = np.ogrid[: g.rows, : g.cols]
    front &= (rr0 - r0) ** 2 + (cc0 - c0) ** 2 > blind * blind
    start = _nearest_traversable(
        trav, g.to_cell(x, y), int(math.ceil(start_search_m / res))
    )
    if start is None:
        empty = np.full(trav.shape, -1, dtype=np.int32)
        return FrontierScan(g, front, empty, [], start_found=False, roomy=roomy)
    rings = wavefront(trav, start)
    reached = rings >= 0

    # Bridge 1-cell gaps along a frontier, then grow each cluster's label over
    # free cells far enough to reach a cell the robot can stand on.
    free = g.free_mask()
    labels, n = _label(_dilate(front, 1) & free)
    if n == 0:
        return FrontierScan(g, front, rings, [], roomy=roomy)
    labels, depth = _grow_labels(labels, free, int(math.ceil(robot_radius_m / res)) + 2)

    sizes = np.bincount(labels[front], minlength=n + 1)
    cand = reached & (labels > 0)
    rr, cc = np.nonzero(cand)
    out: List[Frontier] = []
    if rr.size:
        labs = labels[rr, cc]
        ring = rings[rr, cc]
        # Per label: closest to the frontier first, then nearest the robot.
        order = np.lexsort((ring, depth[rr, cc], labs))
        first = np.ones(order.size, dtype=bool)
        first[1:] = labs[order][1:] != labs[order][:-1]
        fr_r, fr_c = np.nonzero(front)
        fr_lab = labels[fr_r, fr_c]
        for i in order[first]:
            lab = int(labs[i])
            size_m = float(sizes[lab]) * res
            if size_m < min_frontier_m:
                continue
            gr, gc = int(rr[i]), int(cc[i])
            mine = fr_lab == lab
            d2 = (fr_r[mine] - gr) ** 2 + (fr_c[mine] - gc) ** 2
            j = int(np.argmin(d2))
            gx, gy = g.to_world(gr, gc)
            fx, fy = g.to_world(int(fr_r[mine][j]), int(fr_c[mine][j]))
            out.append(Frontier(gx, gy, fx, fy, size_m, float(ring[i]) * res))
    out.sort(key=lambda f: f.path_m)
    return FrontierScan(g, front, rings, out, roomy=roomy)

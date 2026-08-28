"""Global planners on an inflated costmap (A* and Lazy Theta*)."""
from __future__ import annotations

import heapq
import math
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from .costmap import (
    INSCRIBED,
    build_costmap,
    costmap_viz_dict,
    is_traversable,
    nearest_free_cell,
    occupancy_from_bridge_map,
)
from .types import OccupancyGrid, Path2D, PlanResult, Pose2D

PLANNER_ASTAR = "astar"
PLANNER_LAZY_THETA = "lazy_theta_star"
PLANNER_IDS = frozenset({PLANNER_ASTAR, PLANNER_LAZY_THETA})
DEFAULT_PLANNER = PLANNER_LAZY_THETA
DEFAULT_PLANNER_ID = "LazyThetaStar"

# 8-connected neighbors (dx, dy, step_cost)
_NEIGHBORS = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (-1, -1, math.sqrt(2.0)),
)

Cell = Tuple[int, int]


def normalize_planner(name: Optional[str]) -> str:
    """Map config / planner_id aliases onto an internal algorithm key."""
    from ..config import normalize_builtin_planner

    return normalize_builtin_planner(name)


def planner_id_for(algorithm: str) -> str:
    algo = normalize_planner(algorithm)
    return "LazyThetaStar" if algo == PLANNER_LAZY_THETA else "BuiltinAStar"


def _heuristic(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cell_step_cost(costs: np.ndarray, cell: Cell, base_step: float) -> float:
    c = int(costs[cell])
    if c >= INSCRIBED:
        return base_step
    penalty = 1.0 + (c / float(INSCRIBED)) * 0.5
    return base_step * penalty


def line_of_sight(costs: np.ndarray, a: Cell, b: Cell) -> bool:
    """True if every cell on the Bresenham line from ``a`` to ``b`` is traversable.

    Also rejects diagonal corner-cuts: when the line steps diagonally, both
    flanking orthogonal cells must be free (same rule as grid Theta*).
    """
    y0, x0 = a
    y1, x1 = b
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y1 >= y0 else -1
    sx = 1 if x1 >= x0 else -1
    err = dx - dy
    y, x = y0, x0
    h, w = costs.shape

    while True:
        if not (0 <= y < h and 0 <= x < w) or not is_traversable(int(costs[y, x])):
            return False
        if (y, x) == (y1, x1):
            return True
        e2 = 2 * err
        stepped_x = False
        stepped_y = False
        if e2 > -dy:
            err -= dy
            x += sx
            stepped_x = True
        if e2 < dx:
            err += dx
            y += sy
            stepped_y = True
        # Corner cut: diagonal step must not squeeze between two blocked cells.
        if stepped_x and stepped_y:
            if not (
                0 <= y - sy < h
                and 0 <= x < w
                and is_traversable(int(costs[y - sy, x]))
            ):
                return False
            if not (
                0 <= y < h
                and 0 <= x - sx < w
                and is_traversable(int(costs[y, x - sx]))
            ):
                return False


def _reconstruct(came_from: dict, goal: Cell) -> List[Cell]:
    path = [goal]
    current = goal
    while current in came_from and came_from[current] != current:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _astar(
    costs: np.ndarray,
    start: Cell,
    goal: Cell,
) -> Optional[List[Cell]]:
    h, w = costs.shape
    if not is_traversable(int(costs[start])) or not is_traversable(int(costs[goal])):
        return None

    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    g_score = {start: 0.0}
    came_from: dict = {}
    heapq.heappush(open_heap, (_heuristic(start, goal), counter, start))
    closed = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, goal)
        closed.add(current)

        cy, cx = current
        for dy, dx, step in _NEIGHBORS:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            neighbor = (ny, nx)
            if not is_traversable(int(costs[neighbor])):
                continue
            tentative = g_score[current] + _cell_step_cost(costs, neighbor, step)
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            f = tentative + _heuristic(neighbor, goal)
            heapq.heappush(open_heap, (f, counter, neighbor))

    return None


def _lazy_theta_star(
    costs: np.ndarray,
    start: Cell,
    goal: Cell,
) -> Optional[List[Cell]]:
    """Lazy Theta* (Nash et al.): any-angle paths with deferred LOS checks."""
    h, w = costs.shape
    if not is_traversable(int(costs[start])) or not is_traversable(int(costs[goal])):
        return None

    g_score: dict = {start: 0.0}
    parent: dict = {start: start}
    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (_heuristic(start, goal), counter, start))
    closed: set = set()
    in_open: set = {start}

    def _set_vertex(s: Cell) -> None:
        """Validate (or repair) the lazy parent assumption when expanding ``s``."""
        p = parent[s]
        if p == s or line_of_sight(costs, p, s):
            return
        # No LOS to assumed parent: fall back to best visible closed neighbor.
        best_p = None
        best_g = math.inf
        sy, sx = s
        for dy, dx, step in _NEIGHBORS:
            ny, nx = sy + dy, sx + dx
            n = (ny, nx)
            if n not in closed:
                continue
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not is_traversable(int(costs[n])):
                continue
            if not line_of_sight(costs, n, s):
                continue
            # Euclidean parent→s (n is an 8-neighbor, but keep any-angle form).
            cand = g_score[n] + math.hypot(sy - ny, sx - nx)
            if cand < best_g:
                best_g = cand
                best_p = n
        if best_p is None:
            # Should be rare; keep existing parent and g (A*-like local edge).
            return
        parent[s] = best_p
        g_score[s] = best_g

    def _compute_cost(s: Cell, sp: Cell) -> None:
        """Lazy update: assume LOS from parent(s) to ``sp`` (Path 2)."""
        ps = parent[s]
        # Euclidean any-angle cost from assumed parent.
        tentative = g_score[ps] + math.hypot(sp[0] - ps[0], sp[1] - ps[1])
        if tentative < g_score.get(sp, math.inf):
            parent[sp] = ps
            g_score[sp] = tentative

    while open_heap:
        _, _, s = heapq.heappop(open_heap)
        if s not in in_open:
            continue
        in_open.discard(s)
        _set_vertex(s)
        if s == goal:
            return _reconstruct(parent, goal)
        closed.add(s)

        sy, sx = s
        for dy, dx, _step in _NEIGHBORS:
            ny, nx = sy + dy, sx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            sp = (ny, nx)
            if sp in closed:
                continue
            if not is_traversable(int(costs[sp])):
                continue
            if sp not in in_open and sp not in g_score:
                g_score[sp] = math.inf
            g_old = g_score.get(sp, math.inf)
            _compute_cost(s, sp)
            if g_score[sp] < g_old:
                counter += 1
                f = g_score[sp] + _heuristic(sp, goal)
                heapq.heappush(open_heap, (f, counter, sp))
                in_open.add(sp)

    return None


def _simplify(cells: List[Cell]) -> List[Cell]:
    """Drop colinear intermediate cells."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    for i in range(1, len(cells) - 1):
        y0, x0 = out[-1]
        y1, x1 = cells[i]
        y2, x2 = cells[i + 1]
        if (y1 - y0) * (x2 - x1) == (y2 - y1) * (x1 - x0) and (y1 - y0) * (
            x2 - x0
        ) == (y2 - y0) * (x1 - x0):
            continue
        out.append(cells[i])
    out.append(cells[-1])
    return out


def _search(algorithm: str) -> Callable[[np.ndarray, Cell, Cell], Optional[List[Cell]]]:
    algo = normalize_planner(algorithm)
    if algo == PLANNER_LAZY_THETA:
        return _lazy_theta_star
    return _astar


def plan_on_costmap(
    occ: OccupancyGrid,
    costs: np.ndarray,
    start: Pose2D,
    goal: Pose2D,
    *,
    snap_radius_cells: int = 40,
    algorithm: str = DEFAULT_PLANNER,
) -> PlanResult:
    t0 = time.perf_counter()
    sr, sc = occ.world_to_cell(start.x, start.y)
    gr, gc = occ.world_to_cell(goal.x, goal.y)

    start_cell = nearest_free_cell(costs, sr, sc, max_radius_cells=snap_radius_cells)
    goal_cell = nearest_free_cell(costs, gr, gc, max_radius_cells=snap_radius_cells)
    if start_cell is None:
        return PlanResult(
            feasible=False,
            error_code=1,
            error_msg="start pose is in lethal / unknown space",
            planning_time_s=time.perf_counter() - t0,
        )
    if goal_cell is None:
        return PlanResult(
            feasible=False,
            error_code=2,
            error_msg="goal pose is in lethal / unknown space",
            planning_time_s=time.perf_counter() - t0,
        )

    try:
        search = _search(algorithm)
    except ValueError as exc:
        return PlanResult(
            feasible=False,
            error_code=7,
            error_msg=str(exc),
            planning_time_s=time.perf_counter() - t0,
        )

    cells = search(costs, start_cell, goal_cell)
    if not cells:
        return PlanResult(
            feasible=False,
            error_code=3,
            error_msg="no feasible path",
            planning_time_s=time.perf_counter() - t0,
        )

    cells = _simplify(cells)
    world = tuple(occ.cell_to_world(r, c) for r, c in cells)
    # Ensure exact start/goal endpoints for the controller.
    if world:
        world = ((start.x, start.y),) + world[1:-1] + ((goal.x, goal.y),)
        if len(world) < 2:
            world = ((start.x, start.y), (goal.x, goal.y))

    return PlanResult(
        feasible=True,
        path=Path2D(points=world, goal_theta=goal.theta),
        planning_time_s=time.perf_counter() - t0,
    )


def plan_path(
    map_data: dict,
    start: Pose2D,
    goal: Pose2D,
    *,
    inflation_radius_m: float,
    robot_radius_m: float = 0.22,
    cost_scaling_factor: float = 4.0,
    algorithm: str = DEFAULT_PLANNER,
) -> PlanResult:
    """Plan from a bridge-style map dict.

    On success, ``result.costmap_viz`` holds an OccupancyGrid-style dict the
    nav-camera can render (inflated costs the planner actually used).
    """
    try:
        occ = occupancy_from_bridge_map(map_data)
    except (KeyError, TypeError, ValueError) as exc:
        return PlanResult(feasible=False, error_code=4, error_msg=f"bad map: {exc}")
    costs = build_costmap(
        occ,
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
        cost_scaling_factor=cost_scaling_factor,
    )
    result = plan_on_costmap(occ, costs, start, goal, algorithm=algorithm)
    result.costmap_viz = costmap_viz_dict(occ, costs)
    return result


def path_blocked(
    map_data: dict,
    path: Path2D,
    *,
    inflation_radius_m: float,
    robot_radius_m: float,
    sample_step_m: float = 0.15,
) -> bool:
    """True if any sample along ``path`` is non-traversable on a fresh costmap."""
    if path.empty:
        return True
    occ = occupancy_from_bridge_map(map_data)
    costs = build_costmap(
        occ,
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
    )
    pts = path.points
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg / sample_step_m)))
        for k in range(n + 1):
            t = k / n
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            r, c = occ.world_to_cell(x, y)
            if not occ.in_bounds(r, c) or not is_traversable(int(costs[r, c])):
                return True
    return False

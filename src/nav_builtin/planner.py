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
    footprint_traversable,
    is_traversable,
    mark_local_costmap_on_occupancy,
    mark_path_ahead_on_occupancy,
    mark_path_block_from_local,
    mark_scan_on_occupancy,
    nearest_free_cell,
    nearest_free_pose,
    occupancy_from_map_dict,
)
from .path_utils import closest_point_on_path
from .local_costmap import LocalCostmapView, overlay_local_costs_on_costmap
from .local_planner import path_cost_ahead
from .types import OccupancyGrid, Path2D, PlanResult, Pose2D
from ..geom import conversions as conv

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


def _cost_multiplier(cost: int) -> float:
    """Traversal multiplier for soft inflation (1 = free, ≫1 near obstacles).

    Soft cells stay traversable so narrow corridors remain solvable, but the
    multiplier must be strong enough that a modestly longer clear path beats a
    short hug of the inflation halo when open space is available.
    """
    c = int(cost)
    if c <= 0:
        return 1.0
    if c >= INSCRIBED:
        return 1e6
    t = c / float(INSCRIBED - 1)
    # Linear + steep quadratic: outer soft ≈ 10–20×, near-inscribed ≫50×.
    return 1.0 + 25.0 * t + 120.0 * (t * t)


# Any-angle LOS / string-pull may only shortcut through near-free cells.
# Preference costs (32–48) and the visible soft glow fail LOS so paths stay in
# clear space when a detour exists. Narrow gaps remain traversable via
# 8-connected steps through higher soft cells.
_LOS_MAX_SOFT_COST = 30


def _cell_step_cost(costs: np.ndarray, cell: Cell, base_step: float) -> float:
    return base_step * _cost_multiplier(int(costs[cell]))


def _segment_traversal_cost(costs: np.ndarray, a: Cell, b: Cell) -> float:
    """Bresenham path length with per-cell soft-inflation multipliers.

    Integrates step×multiplier (not Euclidean×peak) so Lazy Theta* pays for
    every soft cell the way A* does, and still heavily penalizes halo clips.
    """
    cells = bresenham_cells(a, b)
    if len(cells) <= 1:
        return 0.0
    total = 0.0
    peak = 1.0
    for i in range(1, len(cells)):
        y0, x0 = cells[i - 1]
        y1, x1 = cells[i]
        step = math.hypot(y1 - y0, x1 - x0)
        mult = _cost_multiplier(int(costs[cells[i]]))
        peak = max(peak, mult)
        total += step * mult
    # Peak floor: a long mostly-free chord that nicks soft still pays.
    euclid = math.hypot(b[0] - a[0], b[1] - a[1])
    return max(total, euclid * peak)


def line_of_sight(costs: np.ndarray, a: Cell, b: Cell) -> bool:
    """True if every cell on the Bresenham line from ``a`` to ``b`` is traversable.

    Also rejects diagonal corner-cuts: when the line steps diagonally, both
    flanking orthogonal cells must be free (same rule as grid Theta*).
    Soft-inflation cells above ``_LOS_MAX_SOFT_COST`` also fail LOS so
    any-angle shortcuts stay in clear space when a clear detour exists.
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
        if not (0 <= y < h and 0 <= x < w):
            return False
        cell_cost = int(costs[y, x])
        if not is_traversable(cell_cost):
            return False
        if cell_cost > _LOS_MAX_SOFT_COST:
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
                and int(costs[y - sy, x]) <= _LOS_MAX_SOFT_COST
            ):
                return False
            if not (
                0 <= y < h
                and 0 <= x - sx < w
                and is_traversable(int(costs[y, x - sx]))
                and int(costs[y, x - sx]) <= _LOS_MAX_SOFT_COST
            ):
                return False


def bresenham_cells(a: Cell, b: Cell) -> List[Cell]:
    """Inclusive Bresenham cell chain from ``a`` to ``b`` (no corner checks)."""
    y0, x0 = a
    y1, x1 = b
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y1 >= y0 else -1
    sx = 1 if x1 >= x0 else -1
    err = dx - dy
    y, x = y0, x0
    out: List[Cell] = []
    while True:
        out.append((y, x))
        if (y, x) == (y1, x1):
            return out
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def world_segment_traversable(
    costs: np.ndarray,
    occ: OccupancyGrid,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    sample_step_m: float = 0.05,
    max_cost: Optional[int] = None,
) -> bool:
    """True when the Euclidean segment stays in traversable costmap cells.

    When ``max_cost`` is set, cells above that soft cost also fail (used by
    string-pull so shortcuts stay in clear space).
    """
    seg = math.hypot(x1 - x0, y1 - y0)
    step = max(1e-3, float(sample_step_m))
    if seg < 1e-9:
        row, col = occ.world_to_cell(x0, y0)
        if not (occ.in_bounds(row, col) and is_traversable(int(costs[row, col]))):
            return False
        if max_cost is not None and int(costs[row, col]) > max_cost:
            return False
        return True
    n = max(1, int(math.ceil(seg / step)))
    for k in range(n + 1):
        t = k / n
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        row, col = occ.world_to_cell(x, y)
        if not occ.in_bounds(row, col) or not is_traversable(int(costs[row, col])):
            return False
        if max_cost is not None and int(costs[row, col]) > max_cost:
            return False
    return True


def _cells_8_adjacent(a: Cell, b: Cell) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 and a != b


def _repair_cell_path(costs: np.ndarray, cells: List[Cell]) -> List[Cell]:
    """Ensure consecutive path cells are LOS-connected (Lazy Theta can emit gaps).

    Invalid parent jumps are bridged with A* so the polyline of cell centers
    does not cut through the inflation halo.
    """
    if len(cells) < 2:
        return cells
    out: List[Cell] = [cells[0]]
    for nxt in cells[1:]:
        prev = out[-1]
        if prev == nxt:
            continue
        if _cells_8_adjacent(prev, nxt) and is_traversable(int(costs[nxt])):
            out.append(nxt)
            continue
        if line_of_sight(costs, prev, nxt):
            chain = bresenham_cells(prev, nxt)
            # Only accept Bresenham fill when every cell is free (LOS should
            # guarantee this; still guard against corner-cut mismatches).
            if all(is_traversable(int(costs[c])) for c in chain):
                out.extend(chain[1:])
                continue
        bridge = _astar(costs, prev, nxt)
        if bridge is None or len(bridge) < 2:
            # Last resort: keep the waypoint so planning still returns something.
            out.append(nxt)
        else:
            out.extend(bridge[1:])
    return out


def _densify_world_path(
    costs: np.ndarray,
    occ: OccupancyGrid,
    points: List[Tuple[float, float]],
    *,
    sample_step_m: float,
) -> List[Tuple[float, float]]:
    """Insert cell-center waypoints when a world chord clips non-traversable cells."""
    if len(points) < 2:
        return points
    out: List[Tuple[float, float]] = [points[0]]
    for nxt in points[1:]:
        prev = out[-1]
        if world_segment_traversable(
            costs, occ, prev[0], prev[1], nxt[0], nxt[1], sample_step_m=sample_step_m
        ):
            out.append(nxt)
            continue
        ca = occ.world_to_cell(prev[0], prev[1])
        cb = occ.world_to_cell(nxt[0], nxt[1])
        bridge = _astar(costs, ca, cb)
        if bridge is None or len(bridge) < 2:
            out.append(nxt)
            continue
        for cell in bridge[1:-1]:
            out.append(occ.cell_to_world(cell[0], cell[1]))
        out.append(nxt)
    return out


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
            # Euclidean parent→s weighted by soft inflation along the chord.
            cand = g_score[n] + _segment_traversal_cost(costs, n, s)
            if cand < best_g:
                best_g = cand
                best_p = n
        if best_p is None:
            # No LOS parent among closed cells: fall back to an 8-connected
            # closed neighbor (A*-style edge). Leaving the invalid lazy parent
            # produces chords that cut the inflation halo and fail path_blocked.
            for dy, dx, step in _NEIGHBORS:
                ny, nx = sy + dy, sx + dx
                n = (ny, nx)
                if n not in closed:
                    continue
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if not is_traversable(int(costs[n])):
                    continue
                cand = g_score[n] + _cell_step_cost(costs, s, step)
                if cand < best_g:
                    best_g = cand
                    best_p = n
            if best_p is None:
                return
        parent[s] = best_p
        g_score[s] = best_g

    def _compute_cost(s: Cell, sp: Cell) -> None:
        """Lazy update: assume LOS from parent(s) to ``sp`` (Path 2)."""
        ps = parent[s]
        # Any-angle length × inflation along the assumed parent chord.
        tentative = g_score[ps] + _segment_traversal_cost(costs, ps, sp)
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


def paths_meaningfully_differ(
    a: Path2D,
    b: Path2D,
    *,
    tol_m: float = 0.25,
    samples: int = 16,
) -> bool:
    """True when two paths diverge by more than ``tol_m`` anywhere along the route."""
    if a.empty or b.empty:
        return True
    if len(a.points) != len(b.points):
        # Different waypoint counts usually means a new Lazy Theta* route.
        return True

    def _sample(path: Path2D) -> list[tuple[float, float]]:
        pts = path.points
        if len(pts) < 2:
            return [pts[0]] if pts else []
        seg_lens = []
        cum = [0.0]
        for i in range(len(pts) - 1):
            length = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            seg_lens.append(length)
            cum.append(cum[-1] + length)
        total = cum[-1]
        if total < 1e-9:
            return [pts[0]]
        out: list[tuple[float, float]] = []
        n = max(2, int(samples))
        for k in range(n):
            target = total * k / (n - 1)
            for i in range(len(pts) - 1):
                if cum[i + 1] + 1e-9 < target:
                    continue
                seg = seg_lens[i]
                t = 0.0 if seg < 1e-9 else (target - cum[i]) / seg
                t = max(0.0, min(1.0, t))
                x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
                y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
                out.append((x, y))
                break
        return out

    def _point_to_path_dist(x: float, y: float, path: Path2D) -> float:
        pose = Pose2D(x, y, 0.0)
        px, py, _, _ = closest_point_on_path(pose, path)
        return math.hypot(x - px, y - py)

    sa = _sample(a)
    sb = _sample(b)
    worst = 0.0
    for x, y in sa:
        worst = max(worst, _point_to_path_dist(x, y, b))
    for x, y in sb:
        worst = max(worst, _point_to_path_dist(x, y, a))
    return worst > tol_m


def path_blocked_local(
    pose: Pose2D,
    path: Path2D,
    view: LocalCostmapView,
    *,
    cost_threshold: int = 200,
    lookahead_m: float = 1.5,
    margin_m: float = 0.0,
) -> bool:
    """True when live local costs block the global path ahead of the robot."""
    if path.empty:
        return True
    return (
        path_cost_ahead(
            pose, path, view, lookahead_m=lookahead_m, margin_m=margin_m
        )
        >= cost_threshold
    )


def _merge_path_prefix(prefix: Path2D, main: Path2D) -> Path2D:
    """Join two paths, dropping a duplicate junction point when they meet."""
    pp = prefix.points
    mp = main.points
    if not pp:
        return main
    if not mp:
        return prefix
    if math.hypot(pp[-1][0] - mp[0][0], pp[-1][1] - mp[0][1]) <= 1e-3:
        merged = pp[:-1] + mp
    else:
        merged = pp + mp
    return Path2D(points=merged, goal_theta=main.goal_theta)


def connect_plan_start(
    map_data: dict,
    pose: Pose2D,
    result: PlanResult,
    *,
    inflation_radius_m: float,
    robot_radius_m: float,
    cost_scaling_factor: float = 4.0,
    clearance_preference_m: float = 0.35,
    algorithm: str = DEFAULT_PLANNER,
    xy_tolerance_m: float = 0.15,
    scan: Optional[conv.LaserScan2D] = None,
) -> PlanResult:
    """Prepend a feasible segment when the robot cannot reach ``path[0]`` safely."""
    if not result.feasible or result.path.empty:
        return result
    try:
        occ = occupancy_from_map_dict(map_data)
    except (KeyError, TypeError, ValueError) as exc:
        return PlanResult(feasible=False, error_code=4, error_msg=f"bad map: {exc}")
    costs = build_costmap(
        occ,
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
        cost_scaling_factor=cost_scaling_factor,
        clearance_preference_m=clearance_preference_m,
    )
    sx, sy = result.path.points[0]
    at_start = math.hypot(pose.x - sx, pose.y - sy) <= xy_tolerance_m
    if at_start and footprint_traversable(
        costs,
        occ,
        pose.x,
        pose.y,
        robot_radius_m=min(float(robot_radius_m), float(occ.resolution)),
    ):
        return result
    bridge = plan_path(
        map_data,
        pose,
        Pose2D(sx, sy, pose.theta),
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
        cost_scaling_factor=cost_scaling_factor,
        clearance_preference_m=clearance_preference_m,
        algorithm=algorithm,
        scan=scan,
        scan_pose=pose if scan is not None else None,
    )
    if not bridge.feasible:
        return PlanResult(
            feasible=False,
            error_code=8,
            error_msg="cannot reach plan start from current pose",
        )
    merged = _merge_path_prefix(bridge.path, result.path)
    out = PlanResult(
        feasible=True,
        path=merged,
        planning_time_s=result.planning_time_s + bridge.planning_time_s,
        costmap_viz=result.costmap_viz,
    )
    return out


def plan_on_costmap(
    occ: OccupancyGrid,
    costs: np.ndarray,
    start: Pose2D,
    goal: Pose2D,
    *,
    snap_radius_cells: int = 40,
    max_goal_snap_m: float = 0.5,
    robot_radius_m: float = 0.22,
    algorithm: str = DEFAULT_PLANNER,
) -> PlanResult:
    t0 = time.perf_counter()
    if robot_radius_m > 0.0:
        # ``costs`` is already inflated by robot_radius, so the start only
        # needs its own cell (plus one cell of slack) traversable. Checking a
        # full robot disc here demanded 2x clearance and, next to a live
        # obstacle, reported "start pose is in lethal" on every replan.
        start_xy = nearest_free_pose(
            costs,
            occ,
            start.x,
            start.y,
            robot_radius_m=min(float(robot_radius_m), float(occ.resolution)),
            max_radius_cells=snap_radius_cells,
        )
        goal_xy = nearest_free_pose(
            costs,
            occ,
            goal.x,
            goal.y,
            robot_radius_m=robot_radius_m,
            max_radius_cells=snap_radius_cells,
        )
        if start_xy is None:
            return PlanResult(
                feasible=False,
                error_code=1,
                error_msg="start pose is in lethal / unknown space",
                planning_time_s=time.perf_counter() - t0,
            )
        if goal_xy is None:
            return PlanResult(
                feasible=False,
                error_code=2,
                error_msg="goal pose is in lethal / unknown space",
                planning_time_s=time.perf_counter() - t0,
            )
        snap_m = math.hypot(goal_xy[0] - goal.x, goal_xy[1] - goal.y)
        if snap_m > max(0.0, float(max_goal_snap_m)):
            return PlanResult(
                feasible=False,
                error_code=2,
                error_msg=(
                    f"goal snap {snap_m:.2f} m exceeds max_goal_snap_m="
                    f"{float(max_goal_snap_m):.2f} (goal blocked / over-inflated)"
                ),
                planning_time_s=time.perf_counter() - t0,
            )
        start_cell = occ.world_to_cell(start_xy[0], start_xy[1])
        goal_cell = occ.world_to_cell(goal_xy[0], goal_xy[1])
    else:
        sr, sc = occ.world_to_cell(start.x, start.y)
        gr, gc = occ.world_to_cell(goal.x, goal.y)
        start_cell = nearest_free_cell(
            costs, sr, sc, max_radius_cells=snap_radius_cells
        )
        goal_cell = nearest_free_cell(
            costs, gr, gc, max_radius_cells=snap_radius_cells
        )
        start_xy = (
            occ.cell_to_world(start_cell[0], start_cell[1])
            if start_cell is not None
            else None
        )
        goal_xy = (
            occ.cell_to_world(goal_cell[0], goal_cell[1])
            if goal_cell is not None
            else None
        )
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

    # Lazy Theta* can emit parent jumps without LOS; repair then densify so the
    # world polyline matches what path_blocked / the follower will sample.
    cells = _repair_cell_path(costs, cells)
    cells = _simplify(cells)
    world_list = [occ.cell_to_world(r, c) for r, c in cells]
    # Keep endpoints on snapped free cells so exact poses don't pull the path
    # through the inflation halo.
    if world_list and start_xy is not None and goal_xy is not None:
        if len(world_list) >= 2:
            world_list[0] = start_xy
            world_list[-1] = goal_xy
        else:
            world_list = [start_xy, goal_xy]
    sample_step = max(0.05, float(occ.resolution) * 0.5)
    world_list = _densify_world_path(
        costs, occ, world_list, sample_step_m=sample_step
    )

    return PlanResult(
        feasible=True,
        path=Path2D(points=tuple(world_list), goal_theta=goal.theta),
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
    clearance_preference_m: float = 0.35,
    algorithm: str = DEFAULT_PLANNER,
    scan: Optional[conv.LaserScan2D] = None,
    scan_pose: Optional[conv.Pose2D] = None,
    blocked_path: Optional[Path2D] = None,
    blocked_path_pose: Optional[Pose2D] = None,
    local_view: Optional[LocalCostmapView] = None,
    paint_corridor: bool = True,
    dynamic_obstacle_radius_m: float = 0.35,
    max_goal_snap_m: float = 0.5,
) -> PlanResult:
    """Plan from a bridge-style map dict.

    When ``scan`` is supplied, hits are marked on the map so replans can route
    around dynamic obstacles (people, chairs) not in the static SLAM map.
    When ``local_view`` is set, high-cost cells from the rolling local costmap
    are painted too — that is what already tripped the controller, and scan-only
    marking can miss it (beam gaps / novel-hit filter) and return the same route.
    Live local costs (≥200) are also overlaid onto the planning costmap (no
    second occupancy inflation) so a peel around one blob cannot thread a
    second blob already visible in the window.
    When ``blocked_path`` is set, the current route segment ahead of the robot
    is also marked so a retry must pick a different corridor.

    On success, ``result.costmap_viz`` holds an OccupancyGrid-style dict the
    nav-camera can render (inflated costs the planner actually used).
    """
    try:
        occ = occupancy_from_map_dict(map_data)
    except (KeyError, TypeError, ValueError) as exc:
        return PlanResult(feasible=False, error_code=4, error_msg=f"bad map: {exc}")
    # Paint lidar hits as occupied *cells* (small radius). build_costmap then
    # applies inflation once. Using inflation_radius here double-inflates walls
    # already on the map and can seal narrow corridors.
    if scan is not None and scan_pose is not None:
        hit_r = max(float(occ.resolution), min(float(dynamic_obstacle_radius_m), 0.12))
        occ = mark_scan_on_occupancy(
            occ, scan_pose, scan, obstacle_radius_m=hit_r
        )
    if local_view is not None:
        hit_r = max(float(occ.resolution), min(float(dynamic_obstacle_radius_m), 0.12))
        # Copy only raw, novel scan hits from the cached local view. Its merged
        # cost layer includes global inflation and must not be painted/reinflated.
        occ = mark_local_costmap_on_occupancy(
            occ,
            local_view,
            radius_m=hit_r,
        )
        if blocked_path is not None and blocked_path_pose is not None:
            seal_r = max(0.22, min(float(robot_radius_m) + 0.05, 0.35))
            occ = mark_path_block_from_local(
                occ,
                blocked_path,
                blocked_path_pose,
                local_view,
                cost_threshold=200,
                margin_m=0.0,
                radius_m=seal_r,
                lookahead_m=1.5,
                start_offset_m=float(robot_radius_m) + 0.05,
                end_offset_m=(
                    seal_r
                    + 2.0 * float(robot_radius_m)
                    + 2.0 * float(occ.resolution)
                ),
            )
    if (
        paint_corridor
        and blocked_path is not None
        and blocked_path_pose is not None
    ):
        block_r = max(float(occ.resolution), min(float(dynamic_obstacle_radius_m), 0.12))
        # Leave the robot's own footprint unpainted (start would otherwise be
        # lethal and snap sideways), and only paint the stretch the local
        # costmap actually flagged — not 2.5 m of route.
        occ = mark_path_ahead_on_occupancy(
            occ,
            blocked_path,
            blocked_path_pose,
            radius_m=block_r,
            lookahead_m=1.5,
            start_offset_m=float(robot_radius_m) + block_r + 2.0 * float(occ.resolution),
            end_offset_m=(
                block_r
                + 2.0 * float(robot_radius_m)
                + 2.0 * float(occ.resolution)
            ),
        )
    costs = build_costmap(
        occ,
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
        cost_scaling_factor=cost_scaling_factor,
        clearance_preference_m=clearance_preference_m,
    )
    if local_view is not None:
        costs = overlay_local_costs_on_costmap(costs, occ, local_view, min_cost=200)
    result = plan_on_costmap(
        occ,
        costs,
        start,
        goal,
        algorithm=algorithm,
        robot_radius_m=robot_radius_m,
        max_goal_snap_m=max_goal_snap_m,
    )
    result.costmap_viz = costmap_viz_dict(occ, costs)
    return result


def path_blocked(
    map_data: dict,
    path: Path2D,
    *,
    inflation_radius_m: float,
    robot_radius_m: float,
    sample_step_m: float = 0.15,
    from_pose: Optional[Pose2D] = None,
    ahead_m: Optional[float] = None,
) -> bool:
    """True if any sample along ``path`` is non-traversable on a fresh costmap.

    When ``from_pose`` is set, only the portion of the path from the closest
    projection forward is checked (optionally limited to ``ahead_m``). Checking
    the entire multi-tens-of-metres plan every second falsely trips on far
    unknown/inflation cells and hard-fails long goals immediately.
    """
    if path.empty:
        return True
    occ = occupancy_from_map_dict(map_data)
    costs = build_costmap(
        occ,
        inflation_radius_m=inflation_radius_m,
        robot_radius_m=robot_radius_m,
    )
    return path_blocked_on_costmap(
        occ,
        costs,
        path,
        robot_radius_m=robot_radius_m,
        sample_step_m=sample_step_m,
        from_pose=from_pose,
        ahead_m=ahead_m,
    )


def path_blocked_on_costmap(
    occ: OccupancyGrid,
    costs: np.ndarray,
    path: Path2D,
    *,
    robot_radius_m: float = 0.22,
    sample_step_m: float = 0.15,
    from_pose: Optional[Pose2D] = None,
    ahead_m: Optional[float] = None,
) -> bool:
    """Like ``path_blocked`` but reuses an already-built costmap (control-loop safe).

    ``robot_radius_m`` is accepted for API symmetry with ``path_blocked``; the
    footprint is already encoded in ``costs``.
    """
    _ = robot_radius_m
    if path.empty:
        return True
    pts = path.points
    start_seg = 0
    start_t = 0.0
    remaining_budget = float("inf") if ahead_m is None else max(0.0, float(ahead_m))
    if from_pose is not None and len(pts) >= 2:
        cx, cy, start_seg, _along = closest_point_on_path(from_pose, path)
        start_seg = min(max(0, start_seg), len(pts) - 2)
        x0, y0 = pts[start_seg]
        x1, y1 = pts[start_seg + 1]
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            start_t = 0.0
        else:
            start_t = max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / seg2))

    step = max(1e-3, float(sample_step_m))
    for i in range(start_seg, len(pts) - 1):
        if remaining_budget <= 0.0:
            break
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        t0 = start_t if i == start_seg else 0.0
        usable = seg * (1.0 - t0)
        if usable < 1e-9:
            continue
        check_len = min(usable, remaining_budget)
        # Sample the checked sub-segment (from t0 along usable).
        x_a = x0 + t0 * (x1 - x0)
        y_a = y0 + t0 * (y1 - y0)
        frac = check_len / seg if seg > 1e-9 else 0.0
        x_b = x0 + (t0 + frac) * (x1 - x0)
        y_b = y0 + (t0 + frac) * (y1 - y0)
        if not world_segment_traversable(
            costs, occ, x_a, y_a, x_b, y_b, sample_step_m=step
        ):
            return True
        remaining_budget -= check_len
    return False

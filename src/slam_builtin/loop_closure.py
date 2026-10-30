"""Loop-closure detection for builtin mapping.

The live grid already contains whatever the drifted pose has painted, so a
scan matched against it confirms the drift (the doubled wall). Detection here
renders a submap from *older* keyframes only — path far enough behind the
robot — and matches a short window of recent scans against it with a wide
correlative search. A good, unambiguous fit becomes a pose-graph edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..geom import conversions as conv
from ..nav.global_localize import _inflate_occupied
from . import occupancy as occ
from .pose_graph import GraphSnapshot, render_grid
from .types import LogOddsGrid


@dataclass
class LoopClosureParams:
    min_travel_gap_m: float = 6.0
    search_radius_m: float = 4.0
    # Wide enough that a several-meter return on a long loop is still inside
    # the submap. The match window below is the part that has to see the drift.
    submap_radius_m: float = 16.0
    submap_max_nodes: int = 80
    query_travel_m: float = 2.0
    query_max_points: int = 400
    query_max_range_m: float = 12.0
    window_m: float = 8.0
    window_deg: float = 45.0
    min_score: float = 0.55
    min_overlap: float = 0.3
    min_points: int = 60
    drift_fraction: float = 0.10
    max_std_m: float = 0.35
    max_std_deg: float = 4.0


@dataclass
class LoopMatch:
    pose: conv.Pose2D
    fit: float
    overlap: float
    prior_fit: float
    cov: np.ndarray
    points: int
    anchor: Optional[int] = None
    gap_m: float = 0.0


@dataclass
class _MatchGrid:
    """Flat score tables with a one-cell zero border (out of map scores 0)."""

    resolution: float
    origin_x: float
    origin_y: float
    height: int
    width: int
    coarse: np.ndarray
    fine: np.ndarray
    known: np.ndarray


def old_node_indices(
    snap: GraphSnapshot,
    cur: int,
    *,
    min_travel_gap_m: float,
    radius_m: float,
    max_nodes: int = 0,
) -> List[int]:
    """Nodes far enough behind ``cur`` along the path and near it in space."""
    if cur <= 0:
        return []
    limit = snap.travel[cur] - float(min_travel_gap_m)
    old = np.nonzero(snap.travel[:cur] <= limit)[0]
    if old.size == 0:
        return []
    d = np.hypot(
        snap.poses[old, 0] - snap.poses[cur, 0],
        snap.poses[old, 1] - snap.poses[cur, 1],
    )
    keep = d <= float(radius_m)
    old, d = old[keep], d[keep]
    if max_nodes > 0 and old.size > max_nodes:
        order = np.argsort(d)[:max_nodes]
        old = np.sort(old[order])
    return [int(i) for i in old]


def query_cloud(
    snap: GraphSnapshot,
    cur: int,
    *,
    travel_m: float,
    max_points: int,
    max_range_m: float,
    cell_m: float = 0.05,
) -> np.ndarray:
    """Endpoints of the recent scans in the ``cur`` node frame."""
    cur_inv = conv.invert_pose(snap.pose(cur))
    chunks = []
    k = cur
    while k >= 0 and snap.travel[cur] - snap.travel[k] <= float(travel_m):
        node_in_cur = conv.compose_poses(cur_inv, snap.pose(k))
        for s in snap.scans[k]:
            r = np.asarray(s.ranges, dtype=float)
            ang = s.angle_min + np.arange(r.size) * s.angle_increment
            ok = np.isfinite(r) & (r >= s.range_min) & (r <= min(s.range_max, max_range_m))
            if not np.any(ok):
                continue
            px = r[ok] * np.cos(ang[ok])
            py = r[ok] * np.sin(ang[ok])
            sp = conv.compose_poses(node_in_cur, s.rel)
            c, sn = math.cos(sp.theta), math.sin(sp.theta)
            chunks.append(
                np.stack([sp.x + c * px - sn * py, sp.y + sn * px + c * py], axis=1)
            )
        k -= 1
    if not chunks:
        return np.empty((0, 2))
    pts = np.concatenate(chunks, axis=0)
    keys = np.floor(pts / cell_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    pts = pts[np.sort(first)]
    if max_points > 0 and pts.shape[0] > max_points:
        pick = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[pick]
    return pts


def crop_grid(grid: LogOddsGrid, x: float, y: float, half_m: float) -> LogOddsGrid:
    """Copy of the square of ``grid`` within ``half_m`` of ``(x, y)``."""
    res = grid.resolution
    r0, c0 = grid.world_to_cell(x - half_m, y - half_m)
    r1, c1 = grid.world_to_cell(x + half_m, y + half_m)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(grid.height, r1 + 1), min(grid.width, c1 + 1)
    if r1 <= r0 or c1 <= c0:
        return occ.empty_grid(resolution=res, origin_x=x - 10.0, origin_y=y - 10.0)
    return LogOddsGrid(
        log_odds=np.array(grid.log_odds[r0:r1, c0:c1], dtype=np.float32, copy=True),
        resolution=res,
        origin_x=grid.origin_x + c0 * res,
        origin_y=grid.origin_y + r0 * res,
    )


# Score units per point: wall hit +1, endpoint in known free space -0.7.
_HIT = 10
_FREE = -7
_COARSE_INFLATE_CELLS = 3
_COARSE_YAW_DEG = 2.0
_FINE_XY_CELLS = 3
_FINE_YAW_DEG = 0.5
_FINE_YAW_STEPS = 4
# Coarse cells on a side. A short window stays at 0.1 m; a long-loop window
# steps up so the lattice does not grow with the square of the search.
_COARSE_MAX_STEPS = 20


def _coarse_cell_step(window_m: float, res: float) -> int:
    if res <= 0:
        return 2
    span = float(window_m) / res
    cells = 2
    if span / cells > _COARSE_MAX_STEPS:
        cells = max(2, int(math.ceil(span / _COARSE_MAX_STEPS)))
    return cells


def _match_grid(grid: LogOddsGrid) -> _MatchGrid:
    g = occ.to_occupancy_int16(grid)
    occupied = g >= 50
    hit_fine = _inflate_occupied(occupied, 1)
    hit_coarse = _inflate_occupied(occupied, _COARSE_INFLATE_CELLS)
    free = (g == 0) & ~hit_coarse
    known = (g >= 0) | hit_coarse

    def table(hit: np.ndarray) -> np.ndarray:
        t = np.where(hit, _HIT, np.where(free, _FREE, 0)).astype(np.int16)
        return np.pad(t, 1).ravel()

    return _MatchGrid(
        resolution=float(grid.resolution),
        origin_x=float(grid.origin_x),
        origin_y=float(grid.origin_y),
        height=int(g.shape[0]),
        width=int(g.shape[1]),
        coarse=table(hit_coarse),
        fine=table(hit_fine),
        known=np.pad(known.astype(np.int16), 1).ravel(),
    )


def _lattice_sum(
    mg: _MatchGrid,
    table: np.ndarray,
    pts: np.ndarray,
    x0: float,
    y0: float,
    yaw: float,
    offs: np.ndarray,
) -> np.ndarray:
    """``[j, i]`` = sum of ``table`` under ``pts`` at ``(x0 + offs[i]·res, y0 + offs[j]·res, yaw)``.

    Offsets are whole cells, so each point's cell is computed once per yaw and
    the lattice is a single integer gather.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rx = c * pts[:, 0] - s * pts[:, 1]
    ry = s * pts[:, 0] + c * pts[:, 1]
    col = np.floor((x0 + rx - mg.origin_x) / mg.resolution).astype(np.int64)
    row = np.floor((y0 + ry - mg.origin_y) / mg.resolution).astype(np.int64)
    cols = np.clip(col[:, None] + offs[None, :] + 1, 0, mg.width + 1)
    rows = np.clip(row[:, None] + offs[None, :] + 1, 0, mg.height + 1)
    flat = rows[:, :, None] * (mg.width + 2) + cols[:, None, :]
    return table.take(flat).sum(axis=0, dtype=np.int64)


def match_against(
    grid: LogOddsGrid,
    pts: np.ndarray,
    prior: conv.Pose2D,
    *,
    window_m: float,
    window_deg: float,
) -> Optional[LoopMatch]:
    """Coarse-to-fine correlative search of ``pts`` (prior frame) in ``grid``."""
    n = int(pts.shape[0])
    if n == 0:
        return None
    mg = _match_grid(grid)
    res = mg.resolution
    cells = _coarse_cell_step(window_m, res)
    k = int(round(window_m / (cells * res))) if res > 0 else 0
    offs = np.arange(-k, k + 1, dtype=np.int64) * cells
    yaw_step = math.radians(_COARSE_YAW_DEG)
    n_yaw = int(math.ceil(math.radians(window_deg) / yaw_step))
    dyaws = np.arange(-n_yaw, n_yaw + 1) * yaw_step
    surf = np.stack(
        [
            _lattice_sum(mg, mg.coarse, pts, prior.x, prior.y, prior.theta + d, offs)
            for d in dyaws
        ]
    ).astype(float) / _HIT
    # Equal scores are common along walls; break ties toward the prior
    # instead of argmax's first (most negative) offset. Far below one point.
    ti = (offs / max(1, offs.max())) ** 2
    ty = (dyaws / max(1e-9, abs(dyaws).max())) ** 2
    tie = 1e-3 * (ty[:, None, None] + ti[None, :, None] + ti[None, None, :])
    best = np.unravel_index(int(np.argmax(surf - tie)), surf.shape)
    best_obj = float(surf[best])
    if best_obj <= 0.0:
        return None

    # Spread of near-best candidates: wide along a corridor, tight on a corner.
    yi, ji, ii = np.nonzero(surf >= 0.9 * best_obj)
    sample = np.stack(
        [prior.x + offs[ii] * res, prior.y + offs[ji] * res, dyaws[yi]], axis=1
    )
    cov = np.cov(sample, rowvar=False) if sample.shape[0] >= 2 else np.zeros((3, 3))
    step = cells * res
    cov = cov + np.diag(
        [step**2 / 12.0 + 0.03**2, step**2 / 12.0 + 0.03**2,
         yaw_step**2 / 12.0 + math.radians(0.5) ** 2]
    )

    cx = prior.x + offs[best[2]] * res
    cy = prior.y + offs[best[1]] * res
    cth = prior.theta + dyaws[best[0]]
    # Cover a full coarse step so the fine pass can reach the next sample.
    fine_half = max(_FINE_XY_CELLS, cells)
    fine_offs = np.arange(-fine_half, fine_half + 1, dtype=np.int64)
    fx = cx + fine_offs * res
    fy = cy + fine_offs * res
    window2 = max(window_m, res) ** 2
    yaw_span2 = max(math.radians(window_deg), 1e-6) ** 2
    tie_xy = 1e-3 * _HIT * (
        (fy[:, None] - prior.y) ** 2 + (fx[None, :] - prior.x) ** 2
    ) / window2
    best_f = (-math.inf, cx, cy, cth, -math.inf)
    for kk in range(-_FINE_YAW_STEPS, _FINE_YAW_STEPS + 1):
        yaw = cth + math.radians(_FINE_YAW_DEG) * kk
        s = _lattice_sum(mg, mg.fine, pts, cx, cy, yaw, fine_offs).astype(float)
        tie_yaw = 1e-3 * _HIT * conv.normalize_angle(yaw - prior.theta) ** 2 / yaw_span2
        ranked = s - tie_xy - tie_yaw
        j, i = np.unravel_index(int(np.argmax(ranked)), s.shape)
        if ranked[j, i] > best_f[4]:
            best_f = (float(s[j, i]), float(fx[i]), float(fy[j]), yaw, float(ranked[j, i]))
    obj, bx, by, bth, _ = best_f
    zero = np.zeros(1, dtype=np.int64)
    known = float(_lattice_sum(mg, mg.known, pts, bx, by, bth, zero)[0, 0])
    if known <= 0:
        return None

    p_obj = float(_lattice_sum(mg, mg.fine, pts, prior.x, prior.y, prior.theta, zero)[0, 0])
    p_known = float(_lattice_sum(mg, mg.known, pts, prior.x, prior.y, prior.theta, zero)[0, 0])
    return LoopMatch(
        pose=conv.Pose2D(float(bx), float(by), conv.normalize_angle(bth)),
        fit=obj / _HIT / known,
        overlap=known / n,
        prior_fit=p_obj / _HIT / max(1.0, p_known),
        cov=cov,
        points=n,
    )


def detect(
    snap: GraphSnapshot,
    params: LoopClosureParams,
    *,
    resolution: float,
    base: Optional[LogOddsGrid] = None,
) -> Tuple[Optional[LoopMatch], str]:
    """Match the newest node against older keyframes. Returns ``(match, reason)``."""
    cur = len(snap) - 1
    if cur < 1:
        return None, "no_nodes"
    radius = params.search_radius_m + params.window_m
    old = old_node_indices(
        snap,
        cur,
        min_travel_gap_m=params.min_travel_gap_m,
        radius_m=params.submap_radius_m,
        max_nodes=params.submap_max_nodes,
    )
    near = [
        i
        for i in old
        if math.hypot(snap.poses[i, 0] - snap.poses[cur, 0], snap.poses[i, 1] - snap.poses[cur, 1])
        <= radius
    ]
    if not near and base is None:
        return None, "no_candidates"
    pts = query_cloud(
        snap,
        cur,
        travel_m=params.query_travel_m,
        max_points=params.query_max_points,
        max_range_m=params.query_max_range_m,
    )
    if pts.shape[0] < params.min_points:
        return None, "few_points"
    prior = snap.pose(cur)
    if base is not None:
        base = crop_grid(base, prior.x, prior.y, params.submap_radius_m + params.query_max_range_m)
    submap = render_grid(
        snap, resolution=resolution, nodes=old, base=base, center=(prior.x, prior.y)
    )
    m = match_against(
        submap, pts, prior, window_m=params.window_m, window_deg=params.window_deg
    )
    if m is None:
        return None, "no_match"
    if m.overlap < params.min_overlap:
        return m, "low_overlap"
    if m.fit < params.min_score:
        return m, "low_score"
    dx, dy = m.pose.x - prior.x, m.pose.y - prior.y
    dth = conv.normalize_angle(m.pose.theta - prior.theta)
    edge_m = params.window_m - 1.5 * resolution
    if abs(dx) >= edge_m or abs(dy) >= edge_m or abs(math.degrees(dth)) >= params.window_deg - 1.0:
        return m, "window_edge"
    evals = np.linalg.eigvalsh(m.cov[:2, :2])
    if math.sqrt(max(evals[0], 0.0)) > params.max_std_m:
        return m, "ambiguous"
    if math.degrees(math.sqrt(max(m.cov[2, 2], 0.0))) > params.max_std_deg:
        return m, "ambiguous_yaw"

    anchor = None
    best_d = math.inf
    for i in near:
        d = math.hypot(snap.poses[i, 0] - m.pose.x, snap.poses[i, 1] - m.pose.y)
        if d < best_d:
            anchor, best_d = i, d
    m.anchor = anchor
    m.gap_m = float(snap.travel[cur] - snap.travel[anchor]) if anchor is not None else 0.0
    shift = math.hypot(dx, dy)
    if anchor is not None:
        allowed = max(0.3, params.drift_fraction * m.gap_m)
        # Heading slip is not proportional to distance, so a spin can close
        # inside the search window. Translation stays a fraction of the loop:
        # a long loop may use the whole window, a short one may not.
        allowed_deg = params.window_deg
        if shift > allowed or abs(math.degrees(dth)) > allowed_deg:
            return m, "implausible_drift"
    return m, "ok"

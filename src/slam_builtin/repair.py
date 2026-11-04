"""Offline pose-graph repair for a torn builtin map.

Live loop closure has to decide in seconds with a few scans, so it keeps tight
drift / yaw caps and skips anything it cannot confirm — which is how a long
return pass can end up as a skewed second copy of a hallway. Repair replays
loop detection over every keyframe after the fact with looser caps, and makes
up for that with stricter evidence:

* a correction only counts when ``min_run`` consecutive keyframes agree on it
  (one lucky match on a repetitive corridor does not);
* each accepted run is optimized into the graph before later keyframes are
  matched, so a corrected stretch becomes reference for the next one;
* a run whose edges still disagree with the graph afterwards is rejected;
* the result is only worth applying when the re-rendered map is measurably
  healthier (see ``verdict``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..geom import conversions as conv
from . import loop_closure, map_health
from . import occupancy as occ
from . import pose_graph as pg
from .map_health import MapHealth
from .types import LogOddsGrid


@dataclass
class RepairParams:
    loop: loop_closure.LoopClosureParams
    min_run: int = 3
    # Consecutive corrections that land within this of each other agree.
    agree_m: float = 0.25
    agree_deg: float = 3.0
    # Unmatched keyframes tolerated inside a run.
    max_miss: int = 1
    # Runs whose correction is below this only add a holding edge (no optimize).
    hold_shift_m: float = 0.08
    hold_shift_deg: float = 1.0
    max_chi2: float = 25.0
    window_nodes: int = 900
    max_s: float = 600.0
    # Large absolute snaps need a longer agreeing chain (twin corridors often
    # look consistent for ~3 frames; a true multi-meter weld lasts longer).
    large_shift_m: float = 1.0
    large_shift_min_run: int = 5
    xlarge_shift_m: float = 2.5
    xlarge_shift_min_run: int = 7
    # After optimizing a large-shift cluster, require a local ghost drop in a
    # crop around the corrected stretch (whole-map ghost can stay flat while a
    # false twin weld slides the hallway).
    local_ghost_radius_m: float = 8.0
    local_ghost_min_drop: float = 0.02
    # ``verdict`` gates.
    min_ghost_drop_abs: float = 0.03
    min_ghost_drop_frac: float = 0.25
    # Tile wall-angle spread is noisy on small maps (pillars, clutter); only a
    # big rise that ends in the degrading band vetoes an otherwise clear win.
    max_angle_std_rise_deg: float = 5.0


def required_run(params: RepairParams, shift_m: float) -> int:
    """How many agreeing keyframes a correction of ``shift_m`` must collect."""
    s = float(shift_m)
    base = max(1, int(params.min_run))
    if s >= float(params.xlarge_shift_m):
        return max(base, int(params.xlarge_shift_min_run))
    if s >= float(params.large_shift_m):
        return max(base, int(params.large_shift_min_run))
    return base


def relaxed_loop_params(
    live: loop_closure.LoopClosureParams,
) -> loop_closure.LoopClosureParams:
    """Live detection params with the long-loop drift / yaw caps opened up.

    Those caps exist because live closure acts on two matches; repair needs a
    run of agreeing matches instead, so it can look at the drift live refused.
    """
    return replace(
        live,
        drift_fraction=max(float(live.drift_fraction), 0.3),
        max_yaw_long_gap_deg=float(live.window_deg),
        min_score=max(float(live.min_score), 0.6),
    )


@dataclass
class RepairPlan:
    poses: np.ndarray
    edges: List[pg.GraphEdge]
    clusters: List[dict] = field(default_factory=list)
    rejected: int = 0
    examined: int = 0
    matched: int = 0
    timed_out: bool = False
    stopped: bool = False
    elapsed_s: float = 0.0
    max_shift_m: float = 0.0
    max_shift_deg: float = 0.0

    def summary(self) -> dict:
        return {
            "clusters": len(self.clusters),
            "corrections": [dict(c) for c in self.clusters],
            "rejected": int(self.rejected),
            "new_edges": len(self.edges),
            "examined": int(self.examined),
            "matched": int(self.matched),
            "timed_out": bool(self.timed_out),
            "stopped": bool(self.stopped),
            "elapsed_s": round(self.elapsed_s, 2),
            "max_shift_m": round(self.max_shift_m, 3),
            "max_shift_deg": round(self.max_shift_deg, 2),
        }


@dataclass
class _Hit:
    node: int
    edge: pg.GraphEdge
    corr: conv.Pose2D
    shift_m: float
    shift_deg: float


def _agree(a: conv.Pose2D, b: conv.Pose2D, at: conv.Pose2D, p: RepairParams) -> bool:
    pa = conv.compose_poses(a, at)
    pb = conv.compose_poses(b, at)
    return math.hypot(pa.x - pb.x, pa.y - pb.y) <= p.agree_m and abs(
        conv.normalize_angle(pa.theta - pb.theta)
    ) <= math.radians(p.agree_deg)


def _prefix(snap: pg.GraphSnapshot, poses: np.ndarray, j: int) -> pg.GraphSnapshot:
    return pg.GraphSnapshot(
        poses=poses[: j + 1],
        travel=snap.travel[: j + 1],
        scans=snap.scans[: j + 1],
        edges=[],
        last_seq=snap.last_seq,
    )


def _local_ghost_frac(
    snap: pg.GraphSnapshot,
    poses: np.ndarray,
    *,
    resolution: float,
    center: Tuple[float, float],
    radius_m: float,
    base: Optional[LogOddsGrid],
) -> MapHealth:
    """Map-health on a crop around ``center`` rendered from ``poses``."""
    grid = pg.render_grid(
        snap,
        resolution=resolution,
        poses=poses,
        base=base,
        center=center,
    )
    # render_grid with center seeds a fixed-size window; keep the crop tight.
    half = max(2.0, float(radius_m))
    ox = float(center[0]) - half
    oy = float(center[1]) - half
    cells = occ.to_occupancy_int16(grid)
    # Mask cells outside the radius box to unknown so distant walls don't
    # dilute a hallway-local ghost score.
    res = float(grid.resolution)
    rows = np.arange(cells.shape[0], dtype=np.int32)[:, None]
    cols = np.arange(cells.shape[1], dtype=np.int32)[None, :]
    cx = grid.origin_x + (cols + 0.5) * res
    cy = grid.origin_y + (rows + 0.5) * res
    inside = (cx >= ox) & (cx <= ox + 2 * half) & (cy >= oy) & (cy <= oy + 2 * half)
    cropped = np.where(inside, cells, np.int16(-1))
    return map_health.assess_occupancy(
        cropped,
        resolution=res,
        origin_x=float(grid.origin_x),
        origin_y=float(grid.origin_y),
    )


def _local_ghost_improved(
    before: MapHealth, after: MapHealth, params: RepairParams
) -> bool:
    """True when a large-shift crop is healthier (or was too cold to judge)."""
    if before.occupied_cells < 200 or after.occupied_cells < 200:
        return True
    drop = float(params.local_ghost_min_drop)
    if before.status in ("watch", "degrading"):
        return after.ghost_frac <= before.ghost_frac - drop
    # Wasn't ghosty: still refuse if the weld made it worse.
    return after.ghost_frac <= before.ghost_frac + drop


def plan_repair(
    snap: pg.GraphSnapshot,
    *,
    resolution: float,
    params: RepairParams,
    base: Optional[LogOddsGrid] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> RepairPlan:
    """Find and optimize agreeing loop corrections over the whole graph."""
    started = time.monotonic()
    n = len(snap)
    poses = np.array(snap.poses, dtype=float, copy=True)
    edges: List[pg.GraphEdge] = list(snap.edges)
    plan = RepairPlan(poses=poses, edges=[])
    run: List[_Hit] = []
    misses = 0
    for j in range(1, n):
        if should_stop is not None and should_stop():
            plan.stopped = True
            break
        if time.monotonic() - started > params.max_s:
            plan.timed_out = True
            break
        if progress is not None:
            progress(j, n)
        sub = _prefix(snap, poses, j)
        match, reason = loop_closure.detect(
            sub, params.loop, resolution=resolution, base=base
        )
        plan.examined += 1
        if reason != "ok" or match is None:
            misses += 1
            if misses > params.max_miss:
                run = []
            continue
        misses = 0
        plan.matched += 1
        prior = sub.pose(j)
        corr = pg.world_correction(prior, match.pose)
        hit = _Hit(
            node=j,
            edge=loop_closure.edge_from_match(sub, j, match),
            corr=corr,
            shift_m=math.hypot(match.pose.x - prior.x, match.pose.y - prior.y),
            shift_deg=abs(
                math.degrees(conv.normalize_angle(match.pose.theta - prior.theta))
            ),
        )
        if run and not _agree(run[-1].corr, corr, prior, params):
            run = []
        run.append(hit)
        batch_shift = max(h.shift_m for h in run)
        if len(run) < required_run(params, batch_shift):
            continue
        batch, run = run, []
        if all(
            h.shift_m < params.hold_shift_m and h.shift_deg < params.hold_shift_deg
            for h in batch
        ):
            # Already consistent: keep one edge so later runs cannot pull it apart.
            edges.append(batch[-1].edge)
            plan.edges.append(batch[-1].edge)
            continue
        trial = [h.edge for h in batch]
        anchors = [e.i for e in trial if e.i >= 0]
        first_free = max(
            1, (min(anchors) + 1) if anchors else 1, n - params.window_nodes
        )
        poses_before = np.array(poses, dtype=float, copy=True)
        res = pg.optimize(poses, edges + trial, first_free=first_free, iterations=8)
        worst = max(pg.edge_chi2(res.poses, e) for e in trial)
        if not math.isfinite(worst) or worst > params.max_chi2:
            plan.rejected += 1
            continue
        batch_shift = max(h.shift_m for h in batch)
        if batch_shift >= float(params.large_shift_m):
            cx = float(np.mean([poses_before[h.node, 0] for h in batch]))
            cy = float(np.mean([poses_before[h.node, 1] for h in batch]))
            before_h = _local_ghost_frac(
                snap,
                poses_before,
                resolution=resolution,
                center=(cx, cy),
                radius_m=params.local_ghost_radius_m,
                base=base,
            )
            after_h = _local_ghost_frac(
                snap,
                res.poses,
                resolution=resolution,
                center=(cx, cy),
                radius_m=params.local_ghost_radius_m,
                base=base,
            )
            if not _local_ghost_improved(before_h, after_h, params):
                plan.rejected += 1
                continue
        poses = res.poses
        edges.extend(trial)
        plan.edges.extend(trial)
        plan.clusters.append(
            {
                "nodes": [batch[0].node, batch[-1].node],
                "anchor": min(anchors) if anchors else None,
                "shift_m": round(batch_shift, 3),
                "shift_deg": round(max(h.shift_deg for h in batch), 2),
                "chi2": round(worst, 2),
            }
        )

    if plan.clusters:
        poses, dropped = _settle(poses, edges, plan.edges, n, params)
        if dropped:
            plan.rejected += len(dropped)
            plan.edges = [e for e in plan.edges if not any(e is d for d in dropped)]
    plan.poses = poses
    d = np.hypot(poses[:, 0] - snap.poses[:, 0], poses[:, 1] - snap.poses[:, 1])
    dth = np.abs(
        np.arctan2(
            np.sin(poses[:, 2] - snap.poses[:, 2]),
            np.cos(poses[:, 2] - snap.poses[:, 2]),
        )
    )
    plan.max_shift_m = float(d.max()) if d.size else 0.0
    plan.max_shift_deg = float(np.degrees(dth.max())) if dth.size else 0.0
    plan.elapsed_s = time.monotonic() - started
    return plan


def _settle(
    poses: np.ndarray,
    edges: List[pg.GraphEdge],
    added: List[pg.GraphEdge],
    n: int,
    params: RepairParams,
) -> Tuple[np.ndarray, List[pg.GraphEdge]]:
    """One joint optimize; drop added edges that still disagree, re-solve once."""
    first_free = max(1, n - params.window_nodes)
    res = pg.optimize(poses, edges, first_free=first_free, iterations=8)
    bad = [e for e in added if pg.edge_chi2(res.poses, e) > params.max_chi2]
    if not bad:
        return res.poses, []
    kept = [e for e in edges if not any(e is b for b in bad)]
    res = pg.optimize(poses, kept, first_free=first_free, iterations=8)
    return res.poses, bad


_RANK = {"cold": 0, "ok": 0, "watch": 1, "degrading": 2}


def verdict(
    before: MapHealth,
    after: MapHealth,
    plan: RepairPlan,
    params: RepairParams,
) -> Tuple[bool, List[str]]:
    """``(confident, reasons_not)``: apply without asking only when confident."""
    reasons: List[str] = []
    if _RANK.get(before.status, 0) == 0:
        reasons.append("map_not_torn")
    if not plan.clusters:
        reasons.append("no_corrections")
    if plan.rejected:
        reasons.append("inconsistent_corrections")
    if plan.timed_out or plan.stopped:
        reasons.append("incomplete")
    need = max(
        params.min_ghost_drop_abs, params.min_ghost_drop_frac * before.ghost_frac
    )
    if after.ghost_frac > before.ghost_frac - need:
        reasons.append("ghosts_not_reduced_enough")
    if after.angle_std_deg > max(
        before.angle_std_deg + params.max_angle_std_rise_deg,
        map_health.DEGRADING_ANGLE_STD,
    ):
        reasons.append("wall_angles_worse")
    if _RANK.get(after.status, 0) > _RANK.get(before.status, 0):
        reasons.append("health_worse")
    return (not reasons), reasons


def acceptable(before: MapHealth, after: MapHealth, plan: RepairPlan) -> bool:
    """Bar for an operator-forced apply: something changed and nothing got worse."""
    return (
        bool(plan.clusters)
        and after.ghost_frac <= before.ghost_frac
        and _RANK.get(after.status, 0) <= _RANK.get(before.status, 0)
    )

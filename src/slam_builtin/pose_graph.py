"""Keyframe pose graph for builtin SLAM loop closure.

Nodes are spaced keyframes along the mapping path. Every map insert is kept
(up to ``scans_per_node``) relative to its node, so moving a node moves its
scans, and the grid can be re-rendered from optimized poses. Edges are the
odometry chain between consecutive nodes plus loop constraints found by
matching against older parts of the map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..geom import conversions as conv
from . import occupancy as occ
from .types import LogOddsGrid

PRIOR_NODE = -1


@dataclass
class GraphScan:
    seq: int
    rel: conv.Pose2D
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float


@dataclass
class GraphNode:
    pose: conv.Pose2D
    travel_m: float
    scans: List[GraphScan] = field(default_factory=list)


@dataclass
class GraphEdge:
    """Constraint ``inv(X_i) ∘ X_j == rel``; ``i == PRIOR_NODE`` pins ``X_j``."""

    i: int
    j: int
    rel: conv.Pose2D
    info: np.ndarray
    loop: bool = False


@dataclass
class GraphSnapshot:
    poses: np.ndarray
    travel: np.ndarray
    scans: List[List[GraphScan]]
    edges: List[GraphEdge]
    last_seq: int

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    def pose(self, index: int) -> conv.Pose2D:
        p = self.poses[index]
        return conv.Pose2D(float(p[0]), float(p[1]), float(p[2]))


def info_from_sigmas(sigma_xy: float, sigma_theta: float) -> np.ndarray:
    sxy = max(float(sigma_xy), 1e-4)
    sth = max(float(sigma_theta), 1e-5)
    return np.diag([1.0 / sxy**2, 1.0 / sxy**2, 1.0 / sth**2])


def odometry_info(rel: conv.Pose2D) -> np.ndarray:
    """Scan-matched odometry gets less certain with distance and turning."""
    dist = math.hypot(rel.x, rel.y)
    turn = abs(rel.theta)
    sigma_xy = 0.02 + 0.04 * dist + 0.02 * turn
    sigma_th = math.radians(0.5) + 0.03 * turn + 0.01 * dist
    return info_from_sigmas(sigma_xy, sigma_th)


def world_correction(old: conv.Pose2D, new: conv.Pose2D) -> conv.Pose2D:
    """Map-frame transform ``T`` with ``T ∘ old == new``."""
    return conv.compose_poses(new, conv.invert_pose(old))


class PoseGraph:
    """Session pose graph of mapping keyframes."""

    def __init__(
        self,
        *,
        max_nodes: int = 2000,
        node_spacing_m: float = 0.5,
        node_spacing_deg: float = 20.0,
        scans_per_node: int = 2,
        max_beams: int = 180,
    ):
        self.max_nodes = max(1, int(max_nodes))
        self.node_spacing_m = max(0.05, float(node_spacing_m))
        self.node_spacing_rad = math.radians(max(1.0, float(node_spacing_deg)))
        self.scans_per_node = max(1, int(scans_per_node))
        self.max_beams = int(max_beams)
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self._seq = 0
        self.full = False
        # Map-frame jump of the live pose since the last node (set_pose); keeps
        # the next odometry edge from encoding the jump as motion.
        self._rebase = conv.Pose2D(0.0, 0.0, 0.0)

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def last_seq(self) -> int:
        return self._seq

    def scan_count(self) -> int:
        return sum(len(n.scans) for n in self.nodes)

    def loop_edge_count(self) -> int:
        return sum(1 for e in self.edges if e.loop)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.full = False
        self._rebase = conv.Pose2D(0.0, 0.0, 0.0)

    def rebase(self, correction: conv.Pose2D) -> None:
        """The live pose was moved by ``correction`` (map frame) without the graph."""
        if self.nodes:
            self._rebase = conv.compose_poses(correction, self._rebase)

    # -- building -----------------------------------------------------------
    def _make_scan(
        self, rel: conv.Pose2D, scan: conv.LaserScan2D
    ) -> Optional[GraphScan]:
        ranges = np.asarray(scan.ranges, dtype=float)
        if ranges.size == 0:
            return None
        angle_min = float(scan.angle_min)
        angle_increment = float(scan.angle_increment)
        if self.max_beams > 1 and ranges.size > self.max_beams:
            pick = np.linspace(0, ranges.size - 1, self.max_beams, dtype=np.int32)
            ranges = ranges[pick]
            angle_increment = float(
                scan.angle_increment * (pick[-1] - pick[0]) / (self.max_beams - 1)
            )
        range_max = float(scan.range_max)
        if not math.isfinite(range_max):
            range_max = 30.0
        self._seq += 1
        return GraphScan(
            seq=self._seq,
            rel=rel,
            ranges=ranges.astype(np.float32),
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_min=float(scan.range_min),
            range_max=range_max,
        )

    def add_scan(self, pose: conv.Pose2D, scan: conv.LaserScan2D) -> bool:
        """Store a map insert; starts a new node after enough motion."""
        pose = conv.Pose2D(float(pose.x), float(pose.y), float(pose.theta))
        if not self.nodes:
            gs = self._make_scan(conv.Pose2D(0.0, 0.0, 0.0), scan)
            if gs is None:
                return False
            self.nodes.append(GraphNode(pose=pose, travel_m=0.0, scans=[gs]))
            return True
        last = self.nodes[-1]
        last_live = conv.compose_poses(self._rebase, last.pose)
        rel = conv.compose_poses(conv.invert_pose(last_live), pose)
        moved = math.hypot(rel.x, rel.y)
        turned = abs(rel.theta)
        if moved >= self.node_spacing_m or turned >= self.node_spacing_rad:
            if len(self.nodes) >= self.max_nodes:
                self.full = True
                return False
            gs = self._make_scan(conv.Pose2D(0.0, 0.0, 0.0), scan)
            if gs is None:
                return False
            info = odometry_info(rel)
            if (
                abs(self._rebase.x) > 1e-9
                or abs(self._rebase.y) > 1e-9
                or abs(self._rebase.theta) > 1e-9
            ):
                # Across a set_pose jump the two frames disagree by the jump;
                # a near-free edge keeps optimization from bending the older
                # map to absorb it.
                info = info * 1e-4
                self._rebase = conv.Pose2D(0.0, 0.0, 0.0)
            self.nodes.append(
                GraphNode(pose=pose, travel_m=last.travel_m + moved, scans=[gs])
            )
            self.edges.append(
                GraphEdge(len(self.nodes) - 2, len(self.nodes) - 1, rel, info)
            )
            return True
        if len(last.scans) >= self.scans_per_node:
            return False
        # Spread the kept scans across the node instead of the first few ticks.
        prev = last.scans[-1].rel
        step_m = self.node_spacing_m / self.scans_per_node
        step_rad = self.node_spacing_rad / self.scans_per_node
        if (
            math.hypot(rel.x - prev.x, rel.y - prev.y) < step_m
            and abs(conv.normalize_angle(rel.theta - prev.theta)) < step_rad
        ):
            return False
        gs = self._make_scan(rel, scan)
        if gs is None:
            return False
        last.scans.append(gs)
        return True

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    # -- queries ------------------------------------------------------------
    def snapshot(self) -> GraphSnapshot:
        poses = np.array(
            [[n.pose.x, n.pose.y, n.pose.theta] for n in self.nodes], dtype=float
        ).reshape(-1, 3)
        travel = np.array([n.travel_m for n in self.nodes], dtype=float)
        return GraphSnapshot(
            poses=poses,
            travel=travel,
            scans=[list(n.scans) for n in self.nodes],
            edges=list(self.edges),
            last_seq=self._seq,
        )

    def nearest_old_node(
        self, pose: conv.Pose2D, *, min_travel_gap_m: float, radius_m: float
    ) -> Optional[int]:
        """Closest node at least ``min_travel_gap_m`` of path behind the newest."""
        if not self.nodes:
            return None
        newest = self.nodes[-1].travel_m
        best, best_d = None, float(radius_m)
        for i, n in enumerate(self.nodes):
            if newest - n.travel_m < min_travel_gap_m:
                break
            d = math.hypot(n.pose.x - pose.x, n.pose.y - pose.y)
            if d <= best_d:
                best, best_d = i, d
        return best

    def earliest_node_near(
        self, pose: conv.Pose2D, *, radius_m: float = 1.5
    ) -> Optional[int]:
        for i, n in enumerate(self.nodes):
            if math.hypot(n.pose.x - pose.x, n.pose.y - pose.y) <= radius_m:
                return i
        return None

    # -- edits --------------------------------------------------------------
    def apply_poses(self, snap: GraphSnapshot, new_poses: np.ndarray) -> conv.Pose2D:
        """Write optimized poses for the snapshot's nodes.

        Nodes added after the snapshot ride along with the last snapshot node.
        Returns that map-frame correction for the live pose.
        """
        n = len(snap)
        if n == 0:
            return conv.Pose2D(0.0, 0.0, 0.0)
        old_last = snap.pose(n - 1)
        new_last = conv.Pose2D(*map(float, new_poses[n - 1]))
        tail = world_correction(old_last, new_last)
        for i in range(min(n, len(self.nodes))):
            p = new_poses[i]
            self.nodes[i].pose = conv.Pose2D(float(p[0]), float(p[1]), float(p[2]))
        for i in range(n, len(self.nodes)):
            self.nodes[i].pose = conv.compose_poses(tail, self.nodes[i].pose)
        self._rebase = conv.compose_poses(
            conv.compose_poses(tail, self._rebase), conv.invert_pose(tail)
        )
        return tail

    def apply_world_correction_from(self, start: int, correction: conv.Pose2D) -> None:
        """Rigidly move nodes ``start`` onward by a map-frame transform."""
        for i in range(max(0, int(start)), len(self.nodes)):
            self.nodes[i].pose = conv.compose_poses(correction, self.nodes[i].pose)

    # -- rendering ----------------------------------------------------------
    def scans_after(self, seq: int) -> List[Tuple[conv.Pose2D, GraphScan]]:
        out = []
        for n in self.nodes:
            for s in n.scans:
                if s.seq > seq:
                    out.append((conv.compose_poses(n.pose, s.rel), s))
        return out


def paint_scan(grid: LogOddsGrid, pose: conv.Pose2D, s: GraphScan) -> LogOddsGrid:
    return occ.insert_scan(
        grid,
        pose.x,
        pose.y,
        pose.theta,
        s.ranges,
        s.angle_min,
        s.angle_increment,
        range_min=s.range_min,
        range_max=s.range_max,
        max_beams=0,
    )


def render_grid(
    snap: GraphSnapshot,
    *,
    resolution: float,
    poses: Optional[np.ndarray] = None,
    nodes: Optional[Sequence[int]] = None,
    base: Optional[LogOddsGrid] = None,
    center: Optional[Tuple[float, float]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> LogOddsGrid:
    """Ray-cast stored scans at (optimized) node poses into a fresh grid."""
    if base is not None:
        grid = occ.copy_grid(base)
    elif center is not None:
        size = 20.0
        grid = occ.empty_grid(
            resolution=resolution,
            size_m=size,
            origin_x=center[0] - size / 2.0,
            origin_y=center[1] - size / 2.0,
        )
    else:
        grid = occ.empty_grid(resolution=resolution)
    poses = snap.poses if poses is None else poses
    order = range(len(snap)) if nodes is None else nodes
    total = len(order)
    for k, i in enumerate(order):
        p = poses[i]
        node_pose = conv.Pose2D(float(p[0]), float(p[1]), float(p[2]))
        for s in snap.scans[i]:
            grid = paint_scan(grid, conv.compose_poses(node_pose, s.rel), s)
        if progress is not None:
            progress(k + 1, total)
    return grid


# -- optimization -------------------------------------------------------------
@dataclass
class OptimizeResult:
    poses: np.ndarray
    iterations: int
    chi2_before: float
    chi2_after: float
    free_nodes: int
    loop_chi2: List[float]


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _edge_error(poses: np.ndarray, e: GraphEdge) -> np.ndarray:
    xj = poses[e.j]
    rz = _rot(e.rel.theta)
    if e.i == PRIOR_NODE:
        exy = rz.T @ (xj[:2] - np.array([e.rel.x, e.rel.y]))
        eth = conv.normalize_angle(xj[2] - e.rel.theta)
        return np.array([exy[0], exy[1], eth])
    xi = poses[e.i]
    ri = _rot(xi[2])
    exy = rz.T @ (ri.T @ (xj[:2] - xi[:2]) - np.array([e.rel.x, e.rel.y]))
    eth = conv.normalize_angle(xj[2] - xi[2] - e.rel.theta)
    return np.array([exy[0], exy[1], eth])


def edge_chi2(poses: np.ndarray, e: GraphEdge) -> float:
    err = _edge_error(poses, e)
    return float(err @ e.info @ err)


def total_chi2(poses: np.ndarray, edges: Sequence[GraphEdge]) -> float:
    return float(sum(edge_chi2(poses, e) for e in edges))


def optimize(
    poses: np.ndarray,
    edges: Sequence[GraphEdge],
    *,
    first_free: int = 1,
    iterations: int = 10,
    tol: float = 1e-4,
) -> OptimizeResult:
    """Gauss-Newton over nodes ``first_free`` onward; earlier nodes are fixed."""
    x = np.array(poses, dtype=float, copy=True)
    n = x.shape[0]
    first_free = max(0, min(int(first_free), n))
    nfree = n - first_free
    used = [
        e
        for e in edges
        if e.j < n and e.i < n and (e.j >= first_free or e.i >= first_free)
    ]
    chi_before = total_chi2(x, used)
    it = 0
    if nfree > 0 and used:
        dim = 3 * nfree
        for it in range(1, max(1, int(iterations)) + 1):
            H = np.zeros((dim, dim))
            b = np.zeros(dim)
            for e in used:
                err = _edge_error(x, e)
                rz_t = _rot(e.rel.theta).T
                xj = x[e.j]
                B = np.zeros((3, 3))
                if e.i == PRIOR_NODE:
                    B[:2, :2] = rz_t
                    B[2, 2] = 1.0
                    A = None
                else:
                    xi = x[e.i]
                    ri_t = _rot(xi[2]).T
                    c, s = math.cos(xi[2]), math.sin(xi[2])
                    dri_t = np.array([[-s, c], [-c, -s]])
                    A = np.zeros((3, 3))
                    A[:2, :2] = -rz_t @ ri_t
                    A[:2, 2] = rz_t @ dri_t @ (xj[:2] - xi[:2])
                    A[2, 2] = -1.0
                    B[:2, :2] = rz_t @ ri_t
                    B[2, 2] = 1.0
                om = e.info
                blocks = []
                if A is not None and e.i >= first_free:
                    blocks.append((3 * (e.i - first_free), A))
                if e.j >= first_free:
                    blocks.append((3 * (e.j - first_free), B))
                for ra, Ja in blocks:
                    b[ra : ra + 3] += Ja.T @ om @ err
                    for rb, Jb in blocks:
                        H[ra : ra + 3, rb : rb + 3] += Ja.T @ om @ Jb
            H[np.diag_indices(dim)] += 1e-6
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break
            dx = dx.reshape(nfree, 3)
            x[first_free:] += dx
            x[first_free:, 2] = np.arctan2(
                np.sin(x[first_free:, 2]), np.cos(x[first_free:, 2])
            )
            if float(np.max(np.abs(dx))) < tol:
                break
    return OptimizeResult(
        poses=x,
        iterations=it,
        chi2_before=chi_before,
        chi2_after=total_chi2(x, used),
        free_nodes=nfree,
        loop_chi2=[edge_chi2(x, e) for e in used if e.loop],
    )

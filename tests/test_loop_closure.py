"""Builtin SLAM automatic loop closure: drifted loop → no doubled walls."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.global_localize import _inflate_occupied
from src.nav.maps import MapStore
from src.slam_builtin import loop_closure
from src.slam_builtin import occupancy as occ
from src.slam_builtin import pose_graph as pg
from src.slam_builtin.engine import BuiltinSlamEngine

RES = 0.05
ORIGIN = (-2.0, -2.0)


class _Sensors:
    def get_scan(self, max_age_s: float = 2.0):
        return None

    def get_odom(self):
        return None


def _truth_world() -> np.ndarray:
    """Loop corridor around a block, with pillars so places are distinct."""
    w, h = int(20 / RES), int(16 / RES)
    g = np.zeros((h, w), dtype=bool)

    def box(x0, y0, x1, y1):
        c0, r0 = int((x0 - ORIGIN[0]) / RES), int((y0 - ORIGIN[1]) / RES)
        c1, r1 = int((x1 - ORIGIN[0]) / RES), int((y1 - ORIGIN[1]) / RES)
        g[r0 : r1 + 1, c0 : c1 + 1] = True

    def outline(x0, y0, x1, y1):
        box(x0, y0, x1, y0 + RES)
        box(x0, y1 - RES, x1, y1)
        box(x0, y0, x0 + RES, y1)
        box(x1 - RES, y0, x1, y1)

    outline(-1.0, -1.0, 17.0, 13.0)
    outline(3.0, 3.0, 13.0, 9.0)
    for (x, y) in [
        (5.0, -0.6), (9.0, 2.5), (11.5, -0.4), (15.6, 4.0), (14.2, 7.5),
        (13.6, 11.8), (8.0, 12.5), (4.5, 9.6), (-0.4, 7.0), (2.4, 4.8),
        (0.2, 2.6),
    ]:
        box(x - 0.15, y - 0.15, x + 0.15, y + 0.15)
    return g


def _raycast(truth: np.ndarray, pose: conv.Pose2D, n: int = 360, max_r: float = 8.0):
    angles = -math.pi + np.arange(n) * (2 * math.pi / n)
    t = np.arange(0.05, max_r, RES / 2)
    wx = pose.x + np.cos(pose.theta + angles)[:, None] * t[None, :]
    wy = pose.y + np.sin(pose.theta + angles)[:, None] * t[None, :]
    cols = np.floor((wx - ORIGIN[0]) / RES).astype(int)
    rows = np.floor((wy - ORIGIN[1]) / RES).astype(int)
    h, w = truth.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    hit = truth[np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)] & inside
    first = np.argmax(hit, axis=1)
    ranges = np.where(hit.any(axis=1), t[first], np.inf)
    return conv.LaserScan2D(ranges, -math.pi, 2 * math.pi / n, 0.05, max_r)


def _path():
    """True poses: drive the loop (0.1 m steps, 5 deg turns) and 4 m past start."""
    pts = [(1.0, 1.0), (15.0, 1.0), (15.0, 11.0), (1.0, 11.0), (1.0, 1.0), (5.0, 1.0)]
    poses = []
    th = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        target = math.atan2(y1 - y0, x1 - x0)
        while abs(conv.normalize_angle(target - th)) > 1e-6:
            step = max(-math.radians(5), min(math.radians(5), conv.normalize_angle(target - th)))
            th = conv.normalize_angle(th + step)
            poses.append(conv.Pose2D(x0, y0, th))
        d = math.hypot(x1 - x0, y1 - y0)
        k = int(round(d / 0.1))
        for i in range(1, k + 1):
            poses.append(conv.Pose2D(x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k, th))
    return poses


def _ghost_fraction(grid, truth: np.ndarray) -> float:
    """Share of mapped wall cells farther than 0.2 m from any real wall."""
    near_truth = _inflate_occupied(truth, 4)
    int16 = occ.to_occupancy_int16(grid)
    rows, cols = np.nonzero(int16 >= 65)
    if rows.size == 0:
        return 1.0
    wx = grid.origin_x + (cols + 0.5) * grid.resolution
    wy = grid.origin_y + (rows + 0.5) * grid.resolution
    tc = np.floor((wx - ORIGIN[0]) / RES).astype(int)
    tr = np.floor((wy - ORIGIN[1]) / RES).astype(int)
    h, w = truth.shape
    inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
    ok = np.zeros(rows.size, dtype=bool)
    ok[inside] = near_truth[tr[inside], tc[inside]]
    return float(1.0 - ok.mean())


def _drive(loop_closure_on: bool):
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": "/tmp",
            "mode": "mapping",
            "builtin_loop_closure": loop_closure_on,
        }
    )
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore("/tmp"), rate_hz=5.0)  # type: ignore[arg-type]
    truth = _truth_world()
    poses = _path()
    gyro_scale = 1.015
    engine._pose = poses[0]  # noqa: SLF001
    now = 0.0
    results = []
    for prev, cur in zip(poses, poses[1:]):
        rel = conv.compose_poses(conv.invert_pose(prev), cur)
        odom_rel = conv.Pose2D(rel.x, rel.y, rel.theta * gyro_scale)
        with engine._lock:  # noqa: SLF001
            engine._pose = conv.compose_poses(engine._pose, odom_rel)  # noqa: SLF001
        now += 0.2
        nodes = len(engine._graph)  # noqa: SLF001
        engine._maybe_insert_scan(_raycast(truth, cur), now)  # noqa: SLF001
        if len(engine._graph) > nodes:  # noqa: SLF001
            out = engine.run_loop_closure()
            results.append(out)
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
    end = engine.get_pose()
    err = math.hypot(end.x - poses[-1].x, end.y - poses[-1].y)
    return engine, _ghost_fraction(grid, truth), err, results


def test_drifted_loop_without_closure_doubles_walls():
    _, ghosts, err, _ = _drive(False)
    assert err > 0.4
    assert ghosts > 0.05


def test_loop_closure_removes_doubled_walls():
    engine, ghosts, err, results = _drive(True)
    statuses = [r.get("status") for r in results]
    assert "rebuilt" in statuses, statuses
    assert err < 0.15
    assert ghosts < 0.02
    diag = engine.diagnostics()["loop_closure"]
    assert diag["closures"] >= 1
    assert diag["loop_edges"] >= 2
    assert diag["rejected_inconsistent"] == 0


def test_featureless_corridor_out_and_back_does_not_snap_along_axis():
    """Along a blank corridor only the sideways correction is observable; the
    return pass must not be yanked along the corridor."""
    g = np.zeros((int(16 / RES), int(36 / RES)), dtype=bool)

    def box(x0, y0, x1, y1):
        c0, r0 = int((x0 - ORIGIN[0]) / RES), int((y0 - ORIGIN[1]) / RES)
        c1, r1 = int((x1 - ORIGIN[0]) / RES), int((y1 - ORIGIN[1]) / RES)
        g[r0 : r1 + 1, c0 : c1 + 1] = True

    box(-1, -1, 31, -1 + RES)
    box(-1, 1 - RES, 31, 1)
    box(-1, -1, -1 + RES, 1)
    box(31 - RES, -1, 31, 1)
    poses = [conv.Pose2D(0.5 + 0.1 * i, 0.0, 0.0) for i in range(280)]
    x = poses[-1].x
    poses += [conv.Pose2D(x, 0.0, conv.normalize_angle(math.radians(5 * k))) for k in range(1, 37)]
    poses += [conv.Pose2D(x - 0.1 * i, 0.0, math.pi) for i in range(1, 270)]

    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "mapping"})
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore("/tmp"), rate_hz=5.0)  # type: ignore[arg-type]
    engine._pose = poses[0]  # noqa: SLF001
    now = 0.0
    for prev, cur in zip(poses, poses[1:]):
        rel = conv.compose_poses(conv.invert_pose(prev), cur)
        with engine._lock:  # noqa: SLF001
            engine._pose = conv.compose_poses(  # noqa: SLF001
                engine._pose, conv.Pose2D(rel.x, rel.y, rel.theta * 1.02)  # noqa: SLF001
            )
        now += 0.2
        nodes = len(engine._graph)  # noqa: SLF001
        engine._maybe_insert_scan(_raycast(g, cur), now)  # noqa: SLF001
        if len(engine._graph) > nodes:  # noqa: SLF001
            engine.run_loop_closure()
    end = engine.get_pose()
    assert abs(end.x - poses[-1].x) < 0.15
    assert abs(end.y - poses[-1].y) < 0.15
    with engine._lock:  # noqa: SLF001
        assert _ghost_fraction(engine._grid, g) < 0.02  # noqa: SLF001


def test_match_against_recovers_offset():
    truth = _truth_world()
    graph = pg.PoseGraph(node_spacing_m=0.5)
    # Old pass along the bottom corridor.
    for i in range(60):
        p = conv.Pose2D(1.0 + 0.1 * i, 1.0, 0.0)
        graph.add_scan(p, _raycast(truth, p))
    snap = graph.snapshot()
    submap = pg.render_grid(snap, resolution=RES, center=(4.0, 1.0))
    true = conv.Pose2D(4.0, 1.0, 0.0)
    scan = _raycast(truth, true)
    pts = scan.to_points()
    prior = conv.Pose2D(4.4, 0.7, math.radians(3.0))
    m = loop_closure.match_against(submap, pts, prior, window_m=1.5, window_deg=15.0)
    assert m is not None
    assert math.hypot(m.pose.x - true.x, m.pose.y - true.y) < 0.1
    assert abs(math.degrees(m.pose.theta)) < 1.0
    assert m.fit > 0.55


def test_detect_rejects_when_no_old_geometry():
    truth = _truth_world()
    graph = pg.PoseGraph(node_spacing_m=0.5)
    for i in range(40):
        p = conv.Pose2D(1.0 + 0.1 * i, 1.0, 0.0)
        graph.add_scan(p, _raycast(truth, p))
    m, reason = loop_closure.detect(
        graph.snapshot(), loop_closure.LoopClosureParams(), resolution=RES
    )
    assert m is None
    assert reason == "no_candidates"


def test_set_pose_jump_does_not_become_odometry():
    graph = pg.PoseGraph(node_spacing_m=0.5)
    scan = conv.LaserScan2D(np.array([2.0]), 0.0, 0.0, 0.1, 10.0)
    graph.add_scan(conv.Pose2D(0.0, 0.0, 0.0), scan)
    graph.rebase(conv.Pose2D(3.0, 0.0, 0.0))
    graph.add_scan(conv.Pose2D(3.6, 0.0, 0.0), scan)
    assert len(graph) == 2
    edge = graph.edges[-1]
    assert edge.rel.x == pytest.approx(0.6)
    assert edge.info[0, 0] < pg.odometry_info(edge.rel)[0, 0] * 1e-3

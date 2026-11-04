"""Builtin SLAM automatic loop closure: drifted loop → no doubled walls."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.localize.global_localize import _inflate_occupied
from src.nav.stores.maps import MapStore
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
    for x, y in [
        (5.0, -0.6),
        (9.0, 2.5),
        (11.5, -0.4),
        (15.6, 4.0),
        (14.2, 7.5),
        (13.6, 11.8),
        (8.0, 12.5),
        (4.5, 9.6),
        (-0.4, 7.0),
        (2.4, 4.8),
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
            step = max(
                -math.radians(5),
                min(math.radians(5), conv.normalize_angle(target - th)),
            )
            th = conv.normalize_angle(th + step)
            poses.append(conv.Pose2D(x0, y0, th))
        d = math.hypot(x1 - x0, y1 - y0)
        k = int(round(d / 0.1))
        for i in range(1, k + 1):
            poses.append(
                conv.Pose2D(x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k, th)
            )
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


def _drive(loop_closure_on: bool, **extra):
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": "/tmp",
            "mode": "mapping",
            "builtin_loop_closure": loop_closure_on,
            **extra,
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
    poses += [
        conv.Pose2D(x, 0.0, conv.normalize_angle(math.radians(5 * k)))
        for k in range(1, 37)
    ]
    poses += [conv.Pose2D(x - 0.1 * i, 0.0, math.pi) for i in range(1, 270)]

    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "mapping"}
    )
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore("/tmp"), rate_hz=5.0)  # type: ignore[arg-type]
    engine._pose = poses[0]  # noqa: SLF001
    now = 0.0
    for prev, cur in zip(poses, poses[1:]):
        rel = conv.compose_poses(conv.invert_pose(prev), cur)
        with engine._lock:  # noqa: SLF001
            engine._pose = conv.compose_poses(  # noqa: SLF001
                engine._pose,
                conv.Pose2D(rel.x, rel.y, rel.theta * 1.02),  # noqa: SLF001
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


def test_long_loop_recovers_a_several_meter_drift():
    """A return several meters off (with modest yaw) is inside the search.

    The old ±1.5 m / ±15° window could not see this. Translation still has to
    stay under about 10% of the loop, which a long loop allows. Yaw on a long
    return is capped near ``max_yaw_long_gap_deg`` so a 22° snap cannot shear
    the corridor — inject only a few degrees here.
    """
    truth = _truth_world()
    graph = pg.PoseGraph(node_spacing_m=0.5)
    poses = _path()
    # Keep yaw inside the long-gap cap; lean on XY for the multi-meter drift.
    shift = conv.Pose2D(2.2, 1.6, math.radians(6.0))
    cut = len(poses) - 40
    for i, pose in enumerate(poses):
        stored = pose if i < cut else conv.compose_poses(shift, pose)
        graph.add_scan(stored, _raycast(truth, pose))
    snap = graph.snapshot()
    assert float(snap.travel[-1]) > 35.0
    wandered = math.hypot(
        snap.poses[-1, 0] - poses[-1].x, snap.poses[-1, 1] - poses[-1].y
    )
    assert wandered > 2.5
    match, reason = loop_closure.detect(
        snap, loop_closure.LoopClosureParams(), resolution=RES
    )
    assert reason == "ok", reason
    assert match is not None
    # Along the corridor a fraction of a meter is unobservable. The meters of
    # drift and the heading are not.
    residual = math.hypot(match.pose.x - poses[-1].x, match.pose.y - poses[-1].y)
    assert residual < 0.6
    assert residual < wandered / 2
    assert abs(conv.normalize_angle(match.pose.theta - poses[-1].theta)) < math.radians(
        4.0
    )


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


# Let a sub-meter revisit shift go through the graph instead of pose-only.
_REVISIT_CFG = {
    "mapping_revisit_min_shift_m": 0.05,
    "mapping_revisit_min_shift_deg": 1.0,
}


def test_revisit_correction_with_true_pose_fixes_drifted_loop():
    """A mapping-revisit match at the true pose goes in as a scored graph edge
    (not the old rigid 5 cm / 1 deg one) and un-doubles the walls."""
    engine, ghosts_before, err_before, _ = _drive(False, **_REVISIT_CFG)
    assert err_before > 0.4 and ghosts_before > 0.05
    truth = _truth_world()
    true_end = _path()[-1]
    out = engine.apply_map_pose_correction(true_end, score=0.8)
    assert out["applied"] is True
    assert out["method"] == "pose_graph"
    assert out["status"] == "rebuilt"
    assert "loop_chi2" in out
    end = engine.get_pose()
    assert math.hypot(end.x - true_end.x, end.y - true_end.y) < 0.15
    with engine._lock:  # noqa: SLF001
        assert _ghost_fraction(engine._grid, truth) < 0.03  # noqa: SLF001
    edge = engine._graph.edges[-1]  # noqa: SLF001
    assert edge.loop
    sigma_m, _ = engine._revisit_edge_sigmas(0.8)  # noqa: SLF001
    assert edge.info[0, 0] == pytest.approx(1.0 / sigma_m**2)
    revisit = engine.diagnostics()["loop_closure"]["revisit"]
    assert revisit["corrections"] == 1
    assert revisit["last"]["status"] == "rebuilt"


def test_revisit_edge_softens_with_weaker_score():
    weak_m, weak_rad = BuiltinSlamEngine._revisit_edge_sigmas(0.55)  # noqa: SLF001
    strong_m, strong_rad = BuiltinSlamEngine._revisit_edge_sigmas(0.95)  # noqa: SLF001
    default_m, _ = BuiltinSlamEngine._revisit_edge_sigmas(None)  # noqa: SLF001
    assert weak_m > default_m > strong_m
    assert weak_rad > strong_rad
    assert strong_m == pytest.approx(0.05 + 0.30 * 0.05)
    assert BuiltinSlamEngine._revisit_edge_sigmas(float("nan")) == (  # noqa: SLF001
        BuiltinSlamEngine._revisit_edge_sigmas(None)  # noqa: SLF001
    )


def test_revisit_correction_far_off_is_rejected_not_shifted_rigidly():
    """A match that would move the robot farther than the drift the path could
    have accrued is a wrong wall; the graph and pose stay put."""
    engine, _, _, _ = _drive(False, **_REVISIT_CFG)
    before_pose = engine.get_pose()
    with engine._lock:  # noqa: SLF001
        before_nodes = [n.pose for n in engine._graph.nodes]  # noqa: SLF001
        before_edges = len(engine._graph.edges)  # noqa: SLF001
    bogus = conv.Pose2D(before_pose.x + 4.0, before_pose.y + 3.0, before_pose.theta)
    out = engine.apply_map_pose_correction(bogus, score=0.7)
    assert out["applied"] is False
    assert out["rebuilt"] is False
    assert out["method"] == "rejected_implausible"
    assert out["status"] == "rejected_implausible"
    assert engine.get_pose() == before_pose
    with engine._lock:  # noqa: SLF001
        assert [n.pose for n in engine._graph.nodes] == before_nodes  # noqa: SLF001
        assert len(engine._graph.edges) == before_edges  # noqa: SLF001
    revisit = engine.diagnostics()["loop_closure"]["revisit"]
    assert revisit["rejected"] == 1
    assert revisit["reasons"] == {"rejected_implausible": 1}


def test_revisit_large_yaw_is_rejected_unless_forced():
    """A 180 deg revisit answer is a corridor flip. Only the operator's flip
    command applies it, and then rigidly."""
    engine, _, _, _ = _drive(False, **_REVISIT_CFG)
    before = engine.get_pose()
    flipped = conv.Pose2D(
        before.x, before.y, conv.normalize_angle(before.theta + math.pi)
    )
    out = engine.apply_map_pose_correction(flipped)
    assert out["applied"] is False
    assert out["method"] == "rejected_implausible"
    assert engine.get_pose() == before

    out = engine.apply_map_pose_correction(flipped, force=True)
    assert out["applied"] is True
    assert out["method"] == "rigid"
    assert out["status"] == "rebuilt"
    end = engine.get_pose()
    assert abs(conv.normalize_angle(end.theta - flipped.theta)) < 1e-6


def test_allowed_yaw_tightens_on_long_gap():
    """A long return must stay nearly parallel; short loops keep the window."""
    params = loop_closure.LoopClosureParams()
    assert loop_closure.allowed_yaw_deg(
        params, params.min_travel_gap_m
    ) == pytest.approx(params.window_deg)
    # The tear on tracer2a was 22.5° over a 64 m gap — that must not pass.
    assert loop_closure.allowed_yaw_deg(params, 64.0) == pytest.approx(
        params.max_yaw_long_gap_deg
    )
    assert loop_closure.allowed_yaw_deg(params, 64.0) < 22.5


def test_long_gap_large_yaw_is_implausible_drift():
    """Detect rejects a long-gap match whose yaw would shear the corridor."""
    truth = _truth_world()
    graph = pg.PoseGraph(node_spacing_m=0.5)
    poses = _path()
    # Drive the loop with a large yaw error on the return so the prior is off
    # by more than the long-gap yaw cap but still inside the search window.
    shift = conv.Pose2D(0.4, 0.3, math.radians(22.5))
    cut = len(poses) - 40
    for i, pose in enumerate(poses):
        stored = pose if i < cut else conv.compose_poses(shift, pose)
        graph.add_scan(stored, _raycast(truth, pose))
    snap = graph.snapshot()
    assert float(snap.travel[-1]) > 35.0
    match, reason = loop_closure.detect(
        snap,
        loop_closure.LoopClosureParams(max_yaw_long_gap_deg=8.0),
        resolution=RES,
    )
    # Either the matcher never finds this (low_score / ambiguous) or it finds
    # it and the yaw/gap gate rejects it. A rebuild with 22° must not pass.
    if reason == "ok":
        assert match is not None
        prior = snap.pose(len(snap) - 1)
        dth = abs(math.degrees(conv.normalize_angle(match.pose.theta - prior.theta)))
        assert dth <= 8.0 + 1e-6
    else:
        assert reason in {
            "implausible_drift",
            "low_score",
            "ambiguous",
            "ambiguous_yaw",
            "window_edge",
            "no_match",
            "low_overlap",
        }


def test_loop_pending_expires_instead_of_hanging(monkeypatch):
    """Disagreeing ok-matches must expire pending, not leave it true forever."""
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": "/tmp",
            "mode": "mapping",
            "builtin_loop_closure": True,
            "builtin_loop_pending_max_tries": 3,
            "builtin_loop_pending_max_s": 60.0,
        }
    )
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore("/tmp"), rate_hz=5.0)  # type: ignore[arg-type]
    scan = conv.LaserScan2D(np.full(180, 2.0), -math.pi, 2 * math.pi / 180, 0.05, 8.0)
    # Enough travel that detect would normally run; we stub detect itself.
    for i in range(20):
        engine._graph.add_scan(conv.Pose2D(0.5 * i, 0.0, 0.0), scan)  # noqa: SLF001
    engine._pose = conv.Pose2D(9.5, 0.0, 0.0)  # noqa: SLF001

    n = {"i": 0}

    def fake_detect(snap, params, resolution=0.05, base=None, **_kw):
        # Each call: ok match at a new node pose, yaw wobbling so they disagree.
        n["i"] += 1
        cur = len(snap) - 1
        prior = snap.pose(cur)
        yaw = math.radians(3.0 * ((n["i"] % 3) - 1))  # -3°, 0°, +3° — >2° apart
        matched = conv.Pose2D(
            prior.x + 0.05, prior.y, conv.normalize_angle(prior.theta + yaw)
        )
        cov = np.diag([0.05**2, 0.05**2, math.radians(1.0) ** 2])
        m = loop_closure.LoopMatch(
            pose=matched,
            fit=0.7,
            overlap=0.8,
            prior_fit=0.4,
            cov=cov,
            points=100,
            anchor=0,
            gap_m=10.0,
        )
        return m, "ok"

    monkeypatch.setattr(loop_closure, "detect", fake_detect)

    statuses = []
    for _ in range(6):
        # Advance a node each attempt so pending[1] != cur can be satisfied
        # structurally; the yaw wobble still prevents agreement.
        engine._graph.add_scan(  # noqa: SLF001
            conv.Pose2D(0.5 * (20 + n["i"]), 0.0, 0.0), scan
        )
        statuses.append(engine.run_loop_closure().get("status"))

    assert "pending" in statuses
    assert "pending_expired" in statuses
    diag = engine.diagnostics()["loop_closure"]
    assert diag["pending_expired"] >= 1
    assert diag["closures"] == 0
    # After expiry a fresh pending may be armed, but it must not be ancient.
    if diag["pending"]:
        assert diag["pending_age_s"] is not None
        assert diag["pending_age_s"] < 1.0
        assert diag["pending_tries"] < cfg.builtin_loop_pending_max_tries


def test_large_shift_loop_needs_three_agreeing_detections(monkeypatch):
    """A ≥1 m snap stays pending after the second agree; closes on the third."""
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": "/tmp",
            "mode": "mapping",
            "builtin_loop_closure": True,
            "builtin_loop_pending_max_tries": 10,
            "builtin_loop_pending_max_s": 60.0,
        }
    )
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore("/tmp"), rate_hz=5.0)  # type: ignore[arg-type]
    scan = conv.LaserScan2D(np.full(180, 2.0), -math.pi, 2 * math.pi / 180, 0.05, 8.0)
    for i in range(20):
        engine._graph.add_scan(conv.Pose2D(0.5 * i, 0.0, 0.0), scan)  # noqa: SLF001
    engine._pose = conv.Pose2D(9.5, 0.0, 0.0)  # noqa: SLF001

    n = {"i": 0}

    def fake_detect(snap, params, resolution=0.05, base=None, **_kw):
        n["i"] += 1
        cur = len(snap) - 1
        prior = snap.pose(cur)
        # Stable 1.5 m XY correction — large enough to need 3 agrees.
        matched = conv.Pose2D(prior.x + 1.5, prior.y, prior.theta)
        cov = np.diag([0.05**2, 0.05**2, math.radians(1.0) ** 2])
        return (
            loop_closure.LoopMatch(
                pose=matched,
                fit=0.8,
                overlap=0.8,
                prior_fit=0.4,
                cov=cov,
                points=100,
                anchor=0,
                gap_m=20.0,
            ),
            "ok",
        )

    monkeypatch.setattr(loop_closure, "detect", fake_detect)

    # Bypass optimize/render: we only care that the agree gate holds.
    monkeypatch.setattr(
        engine,
        "_optimize_and_render",
        lambda *a, **k: {"status": "rebuilt", "optimize_ms": 0, "render_ms": 0},
    )

    statuses = []
    for _ in range(3):
        engine._graph.add_scan(  # noqa: SLF001
            conv.Pose2D(0.5 * (20 + n["i"]), 0.0, 0.0), scan
        )
        statuses.append(engine.run_loop_closure().get("status"))

    assert statuses[0] == "pending"
    assert statuses[1] == "pending"  # second agree still shy of 3
    assert statuses[2] == "rebuilt"
    assert engine.diagnostics()["loop_closure"]["closures"] == 1


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

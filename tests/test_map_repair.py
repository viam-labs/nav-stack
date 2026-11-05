"""Saved keyframes, resume on them, and offline map repair."""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import graph_store, repair
from src.slam_builtin import occupancy as occ
from src.slam_builtin import pose_graph as pg
from src.slam_builtin.engine import MODE_LOCALIZING, MODE_MAPPING, BuiltinSlamEngine
from src.slam_builtin.map_health import MapHealth

from .test_loop_closure import (
    RES,
    _ghost_fraction,
    _path,
    _raycast,
    _Sensors,
    _truth_world,
)


def _engine(tmp_path, **extra) -> BuiltinSlamEngine:
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": str(tmp_path),
            "mode": "mapping",
            **extra,
        }
    )
    return BuiltinSlamEngine(cfg, _Sensors(), MapStore(str(tmp_path)), rate_hz=5.0)  # type: ignore[arg-type]


def _drive(engine, poses, *, gyro_scale: float, loop_closure: bool):
    truth = _truth_world()
    engine._pose = poses[0]  # noqa: SLF001
    now = 0.0
    for prev, cur in zip(poses, poses[1:]):
        rel = conv.compose_poses(conv.invert_pose(prev), cur)
        with engine._lock:  # noqa: SLF001
            engine._pose = conv.compose_poses(  # noqa: SLF001
                engine._pose,  # noqa: SLF001
                conv.Pose2D(rel.x, rel.y, rel.theta * gyro_scale),
            )
        now += 0.2
        nodes = len(engine._graph)  # noqa: SLF001
        engine._maybe_insert_scan(_raycast(truth, cur), now)  # noqa: SLF001
        if loop_closure and len(engine._graph) > nodes:  # noqa: SLF001
            engine.run_loop_closure()
    return truth


def _torn_loop(tmp_path, gyro_scale: float = 1.04, loop_closure: bool = False):
    engine = _engine(tmp_path, builtin_loop_closure=loop_closure)
    poses = _path()
    truth = _drive(engine, poses, gyro_scale=gyro_scale, loop_closure=loop_closure)
    return engine, truth, poses


def _end_error(engine, poses) -> float:
    end = engine.get_pose()
    return math.hypot(end.x - poses[-1].x, end.y - poses[-1].y)


def test_keyframes_roundtrip_and_stale_image(tmp_path):
    truth = _truth_world()
    graph = pg.PoseGraph(node_spacing_m=0.5)
    for i in range(30):
        p = conv.Pose2D(1.0 + 0.1 * i, 1.0, 0.01 * i)
        graph.add_scan(p, _raycast(truth, p), also=[(1, _raycast(truth, p))])
    graph.add_edge(
        pg.GraphEdge(0, 3, conv.Pose2D(1.0, 0.1, 0.2), np.eye(3) * 7.0, loop=True)
    )
    base = occ.empty_grid(resolution=RES, size_m=4.0, origin_x=-1.0, origin_y=-2.0)
    base.log_odds[3, 4] = 2.5
    edits = [("clear", 1.0, 2.0, 0.3), ("mark", -1.0, 0.5, 0.2)]
    snap = graph.snapshot()
    graph_store.save_graph(
        tmp_path,
        nodes=graph.nodes,
        edges=graph.edges,
        last_seq=graph.last_seq,
        resolution=RES,
        image_sha1="abc",
        base=base,
        edits=edits,
    )
    loaded = graph_store.load_graph(tmp_path, image_sha1="abc", resolution=RES)
    assert loaded is not None
    assert len(loaded.nodes) == len(snap)
    assert loaded.last_seq == graph.last_seq
    for node, (x, y, th) in zip(loaded.nodes, snap.poses):
        assert (node.pose.x, node.pose.y, node.pose.theta) == (x, y, th)
    a, b = graph.nodes[2].scans, loaded.nodes[2].scans
    assert [s.band for s in a] == [s.band for s in b]
    assert np.allclose(a[0].ranges, b[0].ranges, atol=0.02, equal_nan=True)
    assert [(e.i, e.j, e.loop) for e in loaded.edges] == [
        (e.i, e.j, e.loop) for e in graph.edges
    ]
    assert np.allclose(loaded.edges[-1].info, np.eye(3) * 7.0)
    assert loaded.edits == edits
    assert loaded.base is not None
    assert loaded.base.log_odds[3, 4] == pytest.approx(2.5, abs=0.01)
    assert (loaded.base.origin_x, loaded.base.origin_y) == (-1.0, -2.0)

    assert graph_store.load_graph(tmp_path, image_sha1="other", resolution=RES) is None
    assert graph_store.load_graph(tmp_path, image_sha1="abc", resolution=0.1) is None


def test_resume_reloads_saved_keyframes_as_fixed_nodes(tmp_path):
    engine = _engine(tmp_path, builtin_loop_closure=False)
    poses = _path()[:160]
    _drive(engine, poses, gyro_scale=1.0, loop_closure=False)
    map_dir = tmp_path / "m"
    out = engine.save_map(map_dir, repair=False)
    assert out["keyframes"] == "saved"
    n = len(engine._graph)  # noqa: SLF001
    assert n > 10

    resumed = _engine(tmp_path)
    resumed.configure_mode(MODE_MAPPING, map_dir)
    graph = resumed._graph  # noqa: SLF001
    assert len(graph) == n
    assert resumed._graph_frozen == n  # noqa: SLF001
    assert resumed._graph_base is None  # noqa: SLF001
    baseline = resumed._session_baseline  # noqa: SLF001
    assert baseline is not None
    assert np.array_equal(baseline.log_odds, resumed._grid.log_odds)  # noqa: SLF001
    health = resumed.refresh_map_health(force=True)
    assert health["scope"] == "session"
    assert "trusted" in health["baseline"]
    assert resumed.diagnostics()["loop_closure"]["saved_nodes"] == n
    assert resumed._live_first_free(1) == n  # noqa: SLF001

    # The first insert of the new session starts its own node, linked to the
    # saved path by a near-free edge and already "far" along the path.
    last = graph.nodes[-1]
    resumed._pose = last.pose  # noqa: SLF001
    assert graph.add_scan(last.pose, _raycast(_truth_world(), last.pose))
    assert len(graph) == n + 1
    assert graph.nodes[-1].travel_m >= last.travel_m + 6.0
    assert graph.edges[-1].info[0, 0] < pg.odometry_info(conv.Pose2D(0, 0, 0))[0, 0]


def test_resume_ignores_keyframes_saved_with_another_image(tmp_path):
    engine = _engine(tmp_path, builtin_loop_closure=False)
    _drive(engine, _path()[:120], gyro_scale=1.0, loop_closure=False)
    map_dir = tmp_path / "m"
    engine.save_map(map_dir, repair=False)
    pgm = map_dir / "map.pgm"
    data = bytearray(pgm.read_bytes())
    data[-1] ^= 0xFF
    pgm.write_bytes(bytes(data))

    resumed = _engine(tmp_path)
    resumed.configure_mode(MODE_MAPPING, map_dir)
    assert len(resumed._graph) == 0  # noqa: SLF001
    assert resumed._graph_base is not None  # noqa: SLF001


def test_untrusted_baseline_keeps_saved_keyframes_out_of_loop_closure(
    tmp_path, monkeypatch
):
    from src.slam_builtin import loop_closure

    seen = []

    def fake_detect(snap, params, *, resolution, base=None, min_node=0):
        seen.append((base, min_node))
        return None, "no_candidates"

    monkeypatch.setattr(loop_closure, "detect", fake_detect)
    engine = _engine(tmp_path)
    _drive(engine, _path()[:120], gyro_scale=1.0, loop_closure=False)
    with engine._lock:  # noqa: SLF001
        engine._graph_frozen = len(engine._graph) - 3  # noqa: SLF001
        engine._session_baseline = occ.copy_grid(engine._grid)  # noqa: SLF001
        engine._map_health_base_src = engine._session_baseline  # noqa: SLF001
        engine._map_health_base = {"trusted": False}  # noqa: SLF001
    engine.run_loop_closure()
    assert seen[-1] == (None, engine._graph_frozen)  # noqa: SLF001

    engine._map_health_base = {"trusted": True}  # noqa: SLF001
    engine.run_loop_closure()
    assert seen[-1][1] == 0


def test_untrusted_baseline_paints_extend_only_and_status_reports_it(tmp_path):
    engine = _engine(tmp_path, builtin_loop_closure=False)
    # Seed a known occupied wall that must not thicken under extend-only paint.
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
        r, c = grid.world_to_cell(1.0, 0.0)
        grid.log_odds[r, c] = occ.L_MAX
        before = float(grid.log_odds[r, c])
        engine._session_baseline = occ.copy_grid(grid)  # noqa: SLF001
        engine._map_health_base_src = engine._session_baseline  # noqa: SLF001
        engine._map_health_base = {"trusted": False}  # noqa: SLF001
        engine._mode = MODE_MAPPING  # noqa: SLF001

    scan = conv.LaserScan2D(
        ranges=np.array([1.0], dtype=float),
        angle_min=0.0,
        angle_increment=0.0,
        range_min=0.05,
        range_max=5.0,
    )
    engine._paint_mapping_scan(scan, conv.Pose2D(0.0, 0.0, 0.0), now=1.0)  # noqa: SLF001
    with engine._lock:  # noqa: SLF001
        assert float(engine._grid.log_odds[r, c]) == before  # noqa: SLF001
    paint = engine.diagnostics()["paint"]
    assert paint["extend_only"] is True
    assert paint["baseline_trusted"] is False
    assert paint["state"] == "extend_only"
    assert paint["last_fit"] == "fit"
    assert paint["map_updates"] >= 1


def test_untrusted_baseline_still_grows_walls_in_new_territory(tmp_path):
    engine = _engine(tmp_path, builtin_loop_closure=False)
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
        r, c = grid.world_to_cell(1.025, 0.025)
        grid.log_odds[r, c] = occ.L_MAX
        engine._session_baseline = occ.copy_grid(grid)  # noqa: SLF001
        engine._map_health_base_src = engine._session_baseline  # noqa: SLF001
        engine._map_health_base = {"trusted": False}  # noqa: SLF001
        engine._mode = MODE_MAPPING  # noqa: SLF001

    def scan(rng):
        return conv.LaserScan2D(
            ranges=np.array([rng], dtype=float),
            angle_min=0.0,
            angle_increment=0.0,
            range_min=0.05,
            range_max=10.0,
        )

    # First seen from afar (a free ray through y=3), then driven up to.
    far = conv.Pose2D(0.025, 0.025, math.pi / 2)
    engine._paint_mapping_scan(scan(5.0), far, now=1.0)  # noqa: SLF001
    near = conv.Pose2D(0.025, 2.025, math.pi / 2)
    for t in (2.0, 3.0):
        engine._paint_mapping_scan(scan(1.0), near, now=t)  # noqa: SLF001
    with engine._lock:  # noqa: SLF001
        g = engine._grid  # noqa: SLF001
        cells = occ.to_occupancy_int16(g)
        assert cells[g.world_to_cell(0.025, 3.025)] == 100
        assert float(g.log_odds[g.world_to_cell(1.025, 0.025)]) == occ.L_MAX
    assert engine.diagnostics()["paint"]["state"] == "extend_only"


def test_repair_fixes_a_torn_loop_then_undo_restores_it(tmp_path):
    engine, truth, poses = _torn_loop(tmp_path, gyro_scale=1.04)
    torn = _ghost_fraction(engine._grid, truth)  # noqa: SLF001
    assert torn > 0.3
    assert _end_error(engine, poses) > 1.0
    map_dir = tmp_path / "m"
    saved = engine.save_map(map_dir)
    assert saved["repair"] == {"status": "started"}
    engine._repair_thread.join(timeout=120)  # noqa: SLF001
    out = engine.repair_status()
    assert out["state"] == "done"
    assert out["reason"] == "save"
    assert out["status"] == "applied", out
    assert out["saved"] is True
    assert out["confident"] is True
    assert out["rejected"] == 0
    assert out["after"]["ghost_frac"] < out["before"]["ghost_frac"]
    assert _ghost_fraction(engine._grid, truth) < 0.08  # noqa: SLF001
    assert _end_error(engine, poses) < 0.2
    assert (map_dir / "pre_repair" / "keyframes.npz").is_file()
    saved = graph_store.load_graph(
        map_dir,
        image_sha1=graph_store.file_sha1(map_dir / "map.pgm"),
        resolution=RES,
    )
    assert saved is not None
    assert np.allclose(
        [[n.pose.x, n.pose.y] for n in saved.nodes],
        engine._graph.snapshot().poses[:, :2],  # noqa: SLF001
    )

    undone = engine.undo_repair(map_dir)
    assert undone["status"] == "restored"
    assert not (map_dir / "pre_repair").exists()
    assert _ghost_fraction(engine._grid, truth) > 0.3  # noqa: SLF001
    assert engine._graph_frozen == len(engine._graph) > 0  # noqa: SLF001
    assert engine.undo_repair(map_dir)["status"] == "no_backup"


def test_switching_to_localize_right_after_save_lets_the_repair_land(tmp_path):
    engine, truth, poses = _torn_loop(tmp_path, gyro_scale=1.04)
    map_dir = tmp_path / "m"
    assert engine.save_map(map_dir)["repair"] == {"status": "started"}
    engine.configure_mode(MODE_LOCALIZING, map_dir)
    out = engine.repair_status()
    assert out["state"] == "done"
    assert out["status"] == "applied", out
    assert engine._mode == MODE_LOCALIZING  # noqa: SLF001
    assert _ghost_fraction(engine._grid, truth) < 0.08  # noqa: SLF001
    assert _end_error(engine, poses) < 0.2


def test_switching_to_another_map_cancels_the_repair(tmp_path):
    engine = _engine(tmp_path)
    release = threading.Event()
    job = threading.Thread(
        target=lambda: engine._repair_cancel.wait(5.0) or release.wait(5.0)  # noqa: SLF001
    )
    job.start()
    engine._repair_thread = job  # noqa: SLF001
    engine._repair_map_dir = tmp_path / "m"  # noqa: SLF001
    engine.configure_mode(MODE_MAPPING, tmp_path / "other")
    assert engine._repair_cancel.is_set()  # noqa: SLF001
    job.join(timeout=5.0)
    assert not job.is_alive()


def test_repair_leaves_a_healthy_map_alone(tmp_path):
    engine, truth, _ = _torn_loop(tmp_path, gyro_scale=1.04, loop_closure=True)
    before = engine._graph.snapshot().poses.copy()  # noqa: SLF001
    out = engine.run_repair(apply="auto")
    assert out["status"] in ("kept", "no_corrections"), out
    assert np.array_equal(engine._graph.snapshot().poses, before)  # noqa: SLF001
    assert _ghost_fraction(engine._grid, truth) < 0.03  # noqa: SLF001


def test_save_map_starts_background_repair_only_while_mapping(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        BuiltinSlamEngine,
        "start_repair",
        lambda self, map_dir, apply="auto", reason="manual": (
            calls.append((map_dir, apply, reason)) or {"status": "started"}
        ),
    )
    engine = _engine(tmp_path)
    _drive(engine, _path()[:60], gyro_scale=1.0, loop_closure=False)
    out = engine.save_map(tmp_path / "m")
    assert out["repair"] == {"status": "started"}
    assert calls == [(tmp_path / "m", "auto", "save")]

    off = _engine(tmp_path, builtin_repair_on_save=False)
    _drive(off, _path()[:60], gyro_scale=1.0, loop_closure=False)
    assert "repair" not in off.save_map(tmp_path / "n")


def test_repair_skips_small_graphs_and_other_modes(tmp_path):
    engine = _engine(tmp_path)
    assert engine.start_repair(tmp_path)["reason"] == "too_few_keyframes"
    engine._mode = "localizing"  # noqa: SLF001
    assert engine.run_repair()["reason"] == "not_mapping"


def _health(status: str, ghost: float, angle: float = 5.0) -> MapHealth:
    return MapHealth(status, ghost, 0.1, angle, 5000, ())


def _plan(clusters: int = 2, rejected: int = 0) -> repair.RepairPlan:
    plan = repair.RepairPlan(poses=np.zeros((1, 3)), edges=[])
    plan.clusters = [{"nodes": [1, 3]}] * clusters
    plan.rejected = rejected
    return plan


def test_verdict_only_trusts_a_clear_win_on_a_torn_map():
    p = repair.RepairParams(loop=repair.relaxed_loop_params(_loop_params()))
    torn, fixed = _health("degrading", 0.35, 25.0), _health("ok", 0.08, 8.0)
    assert repair.verdict(torn, fixed, _plan(), p) == (True, [])
    assert (
        "map_not_torn"
        in repair.verdict(_health("ok", 0.05), _health("ok", 0.01), _plan(), p)[1]
    )
    assert (
        "inconsistent_corrections"
        in repair.verdict(torn, fixed, _plan(rejected=1), p)[1]
    )
    assert (
        "ghosts_not_reduced_enough"
        in repair.verdict(torn, _health("degrading", 0.33, 25.0), _plan(), p)[1]
    )
    assert (
        "wall_angles_worse"
        in repair.verdict(torn, _health("degrading", 0.1, 31.0), _plan(), p)[1]
    )
    assert "no_corrections" in repair.verdict(torn, fixed, _plan(0), p)[1]
    assert repair.acceptable(torn, fixed, _plan())
    assert not repair.acceptable(fixed, torn, _plan())


def test_required_run_scales_with_shift_size():
    p = repair.RepairParams(loop=repair.relaxed_loop_params(_loop_params()))
    assert repair.required_run(p, 0.2) == p.min_run
    assert repair.required_run(p, 1.0) == p.large_shift_min_run
    assert repair.required_run(p, 3.0) == p.xlarge_shift_min_run


def test_local_ghost_improved_rejects_flat_or_worse_hallway():
    p = repair.RepairParams(loop=repair.relaxed_loop_params(_loop_params()))
    torn = _health("degrading", 0.30)
    assert not repair._local_ghost_improved(torn, _health("degrading", 0.29), p)  # noqa: SLF001
    assert repair._local_ghost_improved(torn, _health("ok", 0.20), p)  # noqa: SLF001
    ok = _health("ok", 0.05)
    assert not repair._local_ghost_improved(ok, _health("ok", 0.10), p)  # noqa: SLF001
    cold = MapHealth("cold", 0.0, 0.0, 0.0, 50, ())
    assert repair._local_ghost_improved(cold, cold, p)  # noqa: SLF001


def test_plan_repair_rejects_large_shift_without_local_ghost_drop(monkeypatch):
    """Agreeing multi-meter twins that don't cut local ghost are rejected."""
    from src.slam_builtin import loop_closure

    truth = _truth_world()
    graph = pg.PoseGraph(node_spacing_m=0.5)
    # Straight path with scans so optimize / render have geometry.
    for i in range(12):
        p = conv.Pose2D(0.5 * i, 0.0, 0.0)
        graph.add_scan(p, _raycast(truth, p))
    snap = graph.snapshot()
    params = repair.RepairParams(
        loop=repair.relaxed_loop_params(_loop_params()),
        min_run=3,
        large_shift_m=1.0,
        large_shift_min_run=3,  # keep the stub short
        xlarge_shift_m=9.0,
        xlarge_shift_min_run=3,
        max_chi2=1e6,
    )

    def fake_detect(sub, loop_params, *, resolution, base=None, min_node=0):
        cur = len(sub) - 1
        if cur < 3:
            return None, "no_candidates"
        prior = sub.pose(cur)
        matched = conv.Pose2D(prior.x + 1.5, prior.y, prior.theta)
        cov = np.diag([0.05**2, 0.05**2, math.radians(1.0) ** 2])
        m = loop_closure.LoopMatch(
            pose=matched,
            fit=0.8,
            overlap=0.8,
            prior_fit=0.3,
            cov=cov,
            points=100,
            anchor=max(0, cur - 6),
            gap_m=3.0,
        )
        return m, "ok"

    monkeypatch.setattr(loop_closure, "detect", fake_detect)
    # Local crop stays torn: before and after both degrading with no drop.
    torn = _health("degrading", 0.30)

    def fake_local(*_a, **_k):
        return torn

    monkeypatch.setattr(repair, "_local_ghost_frac", fake_local)

    plan = repair.plan_repair(snap, resolution=RES, params=params)
    assert plan.clusters == []
    assert plan.rejected >= 1
    assert plan.matched >= 3


def _loop_params():
    from src.slam_builtin import loop_closure

    return loop_closure.LoopClosureParams()


def test_repair_apply_arg():
    from src.models.slam import _repair_apply_arg

    assert _repair_apply_arg("auto") == "auto"
    assert _repair_apply_arg(True) is True
    assert _repair_apply_arg("force") is True
    assert _repair_apply_arg("dry_run") is False
    with pytest.raises(ValueError):
        _repair_apply_arg("maybe")

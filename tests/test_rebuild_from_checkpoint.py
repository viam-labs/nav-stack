"""A map rebuild that draws only what moved, and gives the same map.

On tracer2a a loop-closure rebuild took 5.7 to 8.9 s at 800 keyframes, most
of them several times a minute. Profiled on that robot's saved graph (832
keyframes, 3,256 scans, a 1835 x 1639 cell grid): 57% of a rebuild was
clamping the whole grid after every scan, and every keyframe was drawn again
to move the last 15 to 85.
"""

from __future__ import annotations

import math

import numpy as np

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import occupancy as occ
from src.slam_builtin import pose_graph as pg
from src.slam_builtin.engine import BuiltinSlamEngine

from .test_loop_closure import _raycast, _Sensors, _truth_world

TRUTH = _truth_world()


def _pose(k: int) -> conv.Pose2D:
    return conv.Pose2D(0.6 * k, 0.1 * math.sin(k), 0.05 * k)


def _graph(keyframes: int) -> pg.PoseGraph:
    graph = pg.PoseGraph(node_spacing_m=0.5)
    for k in range(keyframes):
        graph.add_scan(_pose(k), _raycast(TRUTH, _pose(k)))
    return graph


def _same(a, b) -> bool:
    return (a.origin_x, a.origin_y) == (b.origin_x, b.origin_y) and np.array_equal(
        a.log_odds, b.log_odds
    )


# --- Clamping only what a scan touched ---------------------------------------


def test_painting_a_scan_leaves_every_cell_in_range_without_clamping_them_all():
    rng = np.random.default_rng(3)
    grid = occ.empty_grid(resolution=0.05, size_m=30.0)
    for _ in range(60):
        # Short ranges on purpose: rays overlap near the sensor and run a
        # cell far past the limit before it is clamped.
        ranges = rng.uniform(0.3, 6.0, 180)
        grid = occ.insert_scan(
            grid,
            float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-3.1, 3.1)),
            ranges,
            -math.pi,
            2 * math.pi / 180,
            range_min=0.1,
            range_max=10.0,
            max_beams=0,
        )
        lo = grid.log_odds
        assert float(lo.min()) >= occ.L_MIN and float(lo.max()) <= occ.L_MAX
    assert float(grid.log_odds.min()) == occ.L_MIN  # the limit was reached


# --- Starting a render from a checkpoint -------------------------------------


def test_a_render_from_a_checkpoint_is_the_full_render_cell_for_cell():
    graph = _graph(20)
    snap = graph.snapshot()
    kept: list = []
    first = pg.render_grid(snap, resolution=0.05, keep_at=12, kept=kept)
    assert _same(first, pg.render_grid(snap, resolution=0.05))
    assert len(kept) == 1 and kept[0].nodes == 12

    # A loop closure moves the keyframes since its anchor and no others.
    moved = snap.poses.copy()
    moved[14:, 0] += 0.07
    moved[14:, 2] += math.radians(1.0)
    fast = pg.render_grid(snap, resolution=0.05, poses=moved, resume=kept[0])
    assert _same(fast, pg.render_grid(snap, resolution=0.05, poses=moved))

    # More keyframes since: still the same map as drawing them all.
    for k in range(20, 26):
        graph.add_scan(_pose(k), _raycast(TRUTH, _pose(k)))
    grown = graph.snapshot()
    newer: list = []
    fast = pg.render_grid(
        grown, resolution=0.05, resume=kept[0], keep_at=18, kept=newer
    )
    assert _same(fast, pg.render_grid(grown, resolution=0.05))
    assert newer[0].nodes == 18


def test_a_checkpoint_is_not_used_once_what_it_covers_has_changed():
    graph = _graph(20)
    snap = graph.snapshot()
    kept: list = []
    pg.render_grid(snap, resolution=0.05, keep_at=12, kept=kept)
    ckpt = kept[0]
    assert ckpt.fits(snap, snap.poses, None, 0.05)

    early = snap.poses.copy()
    early[5, 0] += 0.07  # a long loop: a keyframe before the checkpoint moved
    assert not ckpt.fits(snap, early, None, 0.05)
    assert _same(
        pg.render_grid(snap, resolution=0.05, poses=early, resume=ckpt),
        pg.render_grid(snap, resolution=0.05, poses=early),
    )

    other = _graph(20)
    other.nodes[3].scans = other.nodes[3].scans[:0]  # not the scans it drew
    assert not ckpt.fits(other.snapshot(), snap.poses, None, 0.05)
    base = occ.empty_grid(resolution=0.05, size_m=20.0)
    assert not ckpt.fits(snap, snap.poses, base, 0.05)
    assert not ckpt.fits(snap, snap.poses, None, 0.10)


def test_box_clears_land_between_the_same_scans_from_a_checkpoint():
    graph = _graph(20)
    snap = graph.snapshot()
    seqs = [scans[-1].seq for scans in snap.scans]

    def clear(x0, y0, x1, y1):
        return lambda grid: occ.clear_rect(grid, x0, y0, x1, y1)

    clears = [
        (seqs[4], clear(1.0, -1.0, 3.0, 1.0)),  # before the checkpoint
        (seqs[15], clear(6.0, -1.0, 8.0, 1.0)),  # after it
    ]
    kept: list = []
    pg.render_grid(snap, resolution=0.05, clears=clears, keep_at=12, kept=kept)
    fast = pg.render_grid(snap, resolution=0.05, clears=clears, resume=kept[0])
    assert _same(fast, pg.render_grid(snap, resolution=0.05, clears=clears))


# --- In the engine ------------------------------------------------------------


def _engine(tmp_path, keyframes: int) -> BuiltinSlamEngine:
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": str(tmp_path),
            "mode": "mapping",
            "builtin_loop_closure": False,
            "builtin_repair_on_save": False,
        }
    )
    engine = BuiltinSlamEngine(cfg, _Sensors(), MapStore(str(tmp_path)), rate_hz=5.0)  # type: ignore[arg-type]
    for k in range(keyframes):
        _insert(engine, k)
    return engine


def _insert(engine: BuiltinSlamEngine, k: int) -> None:
    with engine._lock:  # noqa: SLF001
        engine._pose = _pose(k)  # noqa: SLF001
    engine._paint_mapping_scan(_raycast(TRUTH, _pose(k)), _pose(k), 0.2 * k)  # noqa: SLF001


def _rebuild(engine: BuiltinSlamEngine, first_free: int) -> dict:
    """What a loop closure does once it has its edge: optimize from ``first_free``."""
    with engine._graph_job:  # noqa: SLF001
        return engine._optimize_and_render(first_free)  # noqa: SLF001


def _is_the_full_render(engine: BuiltinSlamEngine) -> bool:
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
        fresh = pg.render_grid(
            engine._graph.snapshot(),  # noqa: SLF001
            resolution=grid.resolution,
            base=engine._graph_base,  # noqa: SLF001
            clears=engine._render_clears(engine._grid_edits),  # noqa: SLF001
        )
    return _same(grid, fresh)


def test_the_second_rebuild_draws_only_the_keyframes_since_the_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(BuiltinSlamEngine, "_RENDER_CHECKPOINT_LAG", 4)
    engine = _engine(tmp_path, keyframes=12)

    first = _rebuild(engine, first_free=10)
    assert first["status"] == "rebuilt" and first["drawn_nodes"] == 12
    assert _is_the_full_render(engine)

    for k in range(12, 16):
        _insert(engine, k)
    second = _rebuild(engine, first_free=14)
    # The first rebuild kept its render at keyframe 12 - 4 = 8.
    assert second["status"] == "rebuilt" and second["drawn_nodes"] == 16 - 8
    assert _is_the_full_render(engine)

    # A long loop moves a keyframe the checkpoint covers: all of them again.
    with engine._lock:  # noqa: SLF001
        node = engine._graph.nodes[3]  # noqa: SLF001
        node.pose = conv.Pose2D(node.pose.x + 0.1, node.pose.y, node.pose.theta)
    third = _rebuild(engine, first_free=14)
    assert third["drawn_nodes"] == 16 and _is_the_full_render(engine)


def test_a_new_map_does_not_start_from_the_old_maps_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(BuiltinSlamEngine, "_RENDER_CHECKPOINT_LAG", 4)
    engine = _engine(tmp_path, keyframes=12)
    _rebuild(engine, first_free=10)
    assert engine._render_ckpt is not None  # noqa: SLF001

    with engine._lock:  # noqa: SLF001
        engine._reset_graph_locked()  # noqa: SLF001
    assert engine._render_ckpt is None  # noqa: SLF001

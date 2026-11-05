"""Region clear: occupancy AABB + keyframe drop, graph-store v2 clear_box."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import graph_store, persistence
from src.slam_builtin import occupancy as occ
from src.slam_builtin import pose_graph as pg
from src.slam_builtin.engine import MODE_MAPPING, BuiltinSlamEngine

from .test_loop_closure import RES, _Sensors


def _engine(tmp_path, **extra) -> BuiltinSlamEngine:
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": str(tmp_path),
            "mode": "mapping",
            "builtin_repair_on_save": False,
            **extra,
        }
    )
    return BuiltinSlamEngine(cfg, _Sensors(), MapStore(str(tmp_path)), rate_hz=5.0)  # type: ignore[arg-type]


def _node(x: float, y: float, travel: float = 0.0) -> pg.GraphNode:
    return pg.GraphNode(
        pose=conv.Pose2D(x, y, 0.0),
        travel_m=travel,
        scans=[],
        inserts=1,
    )


def _chain_edges(n: int) -> list:
    info = np.eye(3) * 10.0
    edges = [pg.GraphEdge(pg.PRIOR_NODE, 0, conv.Pose2D(0.0, 0.0, 0.0), info.copy())]
    for i in range(n - 1):
        edges.append(pg.GraphEdge(i, i + 1, conv.Pose2D(1.0, 0.0, 0.0), info.copy()))
    # Loop that will be dropped when a middle node is removed.
    if n >= 4:
        edges.append(
            pg.GraphEdge(0, 3, conv.Pose2D(3.0, 0.0, 0.0), info.copy(), loop=True)
        )
    return edges


def test_pose_graph_remove_nodes_renumbers_edges():
    graph = pg.PoseGraph(node_spacing_m=0.5)
    graph.nodes = [_node(float(i), 0.0, float(i)) for i in range(5)]
    graph.edges = _chain_edges(5)
    # Drop nodes at x=1 and x=2 (indices 1, 2).
    removed, n_edges = graph.remove_nodes(lambda _i, node: 0.5 <= node.pose.x <= 2.5)
    assert removed == [1, 2]
    assert n_edges >= 3  # chain edges into/out of 1,2 plus loop 0->3
    assert [(n.pose.x, n.pose.y) for n in graph.nodes] == [
        (0.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
    ]
    # Kept {0,3,4} → {0,1,2}: prior→0, old 3→4 becomes 1→2, loop 0→3 → 0→1.
    pairs = {(e.i, e.j, e.loop) for e in graph.edges}
    assert pairs == {
        (pg.PRIOR_NODE, 0, False),
        (1, 2, False),
        (0, 1, True),
    }


def test_graph_store_clear_box_v2_roundtrip(tmp_path):
    graph = pg.PoseGraph(node_spacing_m=0.5)
    graph.nodes = [_node(0.0, 0.0), _node(1.0, 0.0)]
    graph.edges = _chain_edges(2)
    edits = [
        ("clear", 1.0, 2.0, 0.3),
        ("clear_box", 26.0, -20.0, 32.0, -11.0),
        ("mark", -1.0, 0.5, 0.2),
        ("clear_box", 1.0, 1.0, 2.0, 2.0, 7),
    ]
    graph_store.save_graph(
        tmp_path,
        nodes=graph.nodes,
        edges=graph.edges,
        last_seq=2,
        resolution=RES,
        image_sha1="abc",
        edits=edits,
    )
    with np.load(tmp_path / "keyframes.npz", allow_pickle=False) as z:
        assert int(z["version"]) == 2
        assert z["edit_params"].shape == (4, 4)
        assert z["edit_seq"].tolist() == [-1, -1, -1, 7]
        assert "edit_xyr" not in z.files
    loaded = graph_store.load_graph(tmp_path, image_sha1="abc", resolution=RES)
    assert loaded is not None
    assert loaded.edits == edits


def test_graph_store_loads_v1_edit_xyr(tmp_path):
    """Hand-built v1 keyframes.npz still loads (disk edits only)."""
    path = tmp_path / "keyframes.npz"
    np.savez_compressed(
        path,
        version=np.array(1, dtype=np.int32),
        image_sha1=np.array("abc"),
        resolution=np.array(RES),
        last_seq=np.array(0, dtype=np.int64),
        node_pose=np.zeros((0, 3), dtype=np.float64),
        node_travel=np.zeros(0, dtype=np.float64),
        node_inserts=np.zeros(0, dtype=np.int32),
        scan_node=np.zeros(0, dtype=np.int32),
        scan_seq=np.zeros(0, dtype=np.int64),
        scan_rel=np.zeros((0, 3), dtype=np.float64),
        scan_band=np.zeros(0, dtype=np.int16),
        scan_geom=np.zeros((0, 4), dtype=np.float64),
        scan_len=np.zeros(0, dtype=np.int32),
        scan_ranges=np.zeros(0, dtype=np.float16),
        edge_ij=np.zeros((0, 2), dtype=np.int32),
        edge_rel=np.zeros((0, 3), dtype=np.float64),
        edge_info=np.zeros((0, 3, 3), dtype=np.float64),
        edge_loop=np.zeros(0, dtype=bool),
        edit_kind=np.array([0, 1], dtype=np.int8),
        edit_xyr=np.array([[1.0, 2.0, 0.3], [-1.0, 0.5, 0.2]], dtype=np.float64),
    )
    loaded = graph_store.load_graph(tmp_path, image_sha1="abc", resolution=RES)
    assert loaded is not None
    assert loaded.edits == [
        ("clear", 1.0, 2.0, 0.3),
        ("mark", -1.0, 0.5, 0.2),
    ]


def test_clear_region_drops_inbox_nodes_and_persists(tmp_path):
    engine = _engine(tmp_path)
    # Occupancy: fill a band that overlaps the clear box and a cell outside.
    grid = occ.empty_grid(resolution=RES, size_m=20.0, origin_x=-2.0, origin_y=-2.0)
    for x in np.linspace(0.0, 8.0, 41):
        for y in np.linspace(-0.5, 0.5, 11):
            r, c = grid.world_to_cell(float(x), float(y))
            if 0 <= r < grid.log_odds.shape[0] and 0 <= c < grid.log_odds.shape[1]:
                grid.log_odds[r, c] = occ.L_MAX
    with engine._lock:  # noqa: SLF001
        engine._grid = grid  # noqa: SLF001
        # Nodes: 0 and 4 outside the hallway box; 1,2,3 inside.
        engine._graph.nodes = [  # noqa: SLF001
            _node(0.0, 0.0, 0.0),
            _node(2.0, 0.0, 2.0),
            _node(3.0, 0.0, 3.0),
            _node(4.0, 0.0, 4.0),
            _node(7.0, 0.0, 7.0),
        ]
        engine._graph.edges = _chain_edges(5)  # noqa: SLF001
        engine._graph_frozen = 5  # noqa: SLF001
        engine._session_baseline = occ.copy_grid(grid)  # noqa: SLF001
        engine._map_health_base_src = engine._session_baseline  # noqa: SLF001
        engine._map_health_base = {"trusted": False}  # noqa: SLF001

    out = engine.clear_region(1.5, -1.0, 4.5, 1.0)
    assert out["removed_nodes"] == 3
    assert out["remaining_nodes"] == 2
    assert out["frozen"] == 2  # both kept nodes were in the frozen prefix
    assert out["cleared_cells"] > 0
    assert out["removed_edges"] > 0

    poses = [(n.pose.x, n.pose.y) for n in engine._graph.nodes]  # noqa: SLF001
    assert poses == [(0.0, 0.0), (7.0, 0.0)]
    # Middle nodes removed: only the prior pin on node 0 survives.
    pairs = [(e.i, e.j) for e in engine._graph.edges]  # noqa: SLF001
    assert pairs == [(pg.PRIOR_NODE, 0)]

    after = occ.to_occupancy_int16(engine._grid)  # noqa: SLF001
    r_in, c_in = engine._grid.world_to_cell(3.0, 0.0)  # noqa: SLF001
    assert after[r_in, c_in] == 0
    # Nodes in this unit test carry no scans; a rebuild replaces the seeded
    # occupancy with an empty render + clear_box. Outside the box may be
    # unknown (-1) rather than the pre-seeded occupied cell.
    assert out.get("rebuilt") in (True, False)

    boxes = [e[:5] for e in engine._grid_edits if e[0] == "clear_box"]  # noqa: SLF001
    assert boxes == [("clear_box", 1.5, -1.0, 4.5, 1.0)]
    assert engine._map_health_base is None  # noqa: SLF001
    assert engine._session_baseline is not None  # noqa: SLF001
    assert np.array_equal(  # noqa: SLF001
        engine._session_baseline.log_odds,
        engine._grid.log_odds,  # noqa: SLF001
    )

    map_dir = tmp_path / "m"
    saved = engine.save_map(map_dir, repair=False)
    assert saved["keyframes"] == "saved"

    resumed = _engine(tmp_path)
    resumed.configure_mode(MODE_MAPPING, map_dir)
    assert len(resumed._graph) == 2  # noqa: SLF001
    assert resumed._graph_frozen == 2  # noqa: SLF001
    assert [(n.pose.x, n.pose.y) for n in resumed._graph.nodes] == [  # noqa: SLF001
        (0.0, 0.0),
        (7.0, 0.0),
    ]
    assert resumed._grid_edits == engine._grid_edits  # noqa: SLF001
    reloaded = occ.to_occupancy_int16(resumed._grid)  # noqa: SLF001
    r_in, c_in = resumed._grid.world_to_cell(3.0, 0.0)  # noqa: SLF001
    assert reloaded[r_in, c_in] == 0


def test_clear_region_rejects_empty_box(tmp_path):
    engine = _engine(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        engine.clear_region(1.0, 0.0, 1.0, 2.0)


def test_clear_region_enables_full_paint_despite_untrusted_baseline(tmp_path):
    """After clear_region, remapping must rewrite free cells (not extend_only)."""
    engine = _engine(tmp_path, builtin_loop_closure=False)
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
        # Seed a wall inside the upcoming clear box so clear paints it free.
        r_hole, c_hole = grid.world_to_cell(0.0, 0.0)
        grid.log_odds[r_hole, c_hole] = occ.L_MAX
        engine._session_baseline = occ.copy_grid(grid)  # noqa: SLF001
        engine._map_health_base_src = engine._session_baseline  # noqa: SLF001
        engine._map_health_base = {"trusted": False}  # noqa: SLF001
        engine._mode = MODE_MAPPING  # noqa: SLF001

    assert engine.diagnostics()["paint"]["extend_only"] is True

    out = engine.clear_region(-0.5, -0.5, 0.5, 0.5)
    assert out["operator_trusted_paint"] is True
    paint = engine.diagnostics()["paint"]
    assert paint["baseline_trusted"] is True
    assert paint["extend_only"] is False
    assert paint["operator_trusted"] is True
    guide = engine.diagnostics()["mapping_guidance"]
    assert guide["operator_trusted_paint"] is True
    assert guide["suggested_action"]
    assert (
        "drive" in guide["suggested_action"].lower()
        or "remap" in guide["suggested_action"].lower()
    )

    with engine._lock:  # noqa: SLF001
        hole = float(engine._grid.log_odds[r_hole, c_hole])  # noqa: SLF001
    assert hole == occ.CLEARED_LOG_ODDS

    # Full paint can rewrite the free hole; extend_only would leave it alone.
    scan = conv.LaserScan2D(
        ranges=np.array([1.0], dtype=float),
        angle_min=0.0,
        angle_increment=0.0,
        range_min=0.05,
        range_max=5.0,
    )
    engine._paint_mapping_scan(scan, conv.Pose2D(-1.0, 0.0, 0.0), now=1.0)  # noqa: SLF001
    with engine._lock:  # noqa: SLF001
        # One hit may not cross free→occupied; it must still move off the hole.
        hole = float(engine._grid.log_odds[r_hole, c_hole])  # noqa: SLF001
    assert hole > occ.CLEARED_LOG_ODDS
    assert engine.diagnostics()["paint"]["state"] == "painting"


def test_mapping_guidance_warns_when_tearing(tmp_path):
    engine = _engine(tmp_path)
    with engine._lock:  # noqa: SLF001
        engine._mode = MODE_MAPPING  # noqa: SLF001
        engine._map_health = {  # noqa: SLF001
            "status": "degrading",
            "ghost_frac": 0.4,
            "hotspots": [{"frac": 0.7, "x": 12.0, "y": -3.0}],
        }
        engine._map_health_base = {"trusted": False}  # noqa: SLF001
        engine._session_baseline = occ.copy_grid(engine._grid)  # noqa: SLF001
        engine._map_health_base_src = engine._session_baseline  # noqa: SLF001
        engine._operator_trusted_paint = False  # noqa: SLF001
    guide = engine.diagnostics()["mapping_guidance"]
    assert guide["health_status"] == "degrading"
    assert guide["extend_only"] is True
    assert guide["suggested_action"]
    assert (
        "tear" in guide["suggested_action"].lower()
        or "clear" in guide["suggested_action"].lower()
    )
    assert guide["hotspot_count"] == 1


def _banded_wall_scan(wall_x: float) -> conv.LaserScan2D:
    """A wall ``wall_x`` m ahead seen by a low and a high lidar (two bands)."""
    ang = np.radians(np.linspace(-30.0, 30.0, 721))
    r = wall_x / np.cos(ang)
    xyz = np.vstack(
        [
            np.stack([r * np.cos(ang), r * np.sin(ang), np.full_like(ang, z)], 1)
            for z in (0.21, 1.2)
        ]
    )
    scan = conv.points_to_scan(xyz, num_bins=360, range_min=0.05, range_max=20.0)
    scan.cloud_xyz = xyz
    scan.cloud_xy = xyz[:, :2]
    return scan


def _paint_banded(engine, pose: conv.Pose2D, wall_x: float, now: float) -> None:
    scan = _banded_wall_scan(wall_x)
    engine._paint_mapping_scan(scan, pose, now=now, frames=[(None, scan)])  # noqa: SLF001


def _odds(engine, x: float, y: float) -> float:
    """Strongest cell within two cells: keyframes keep a thinned scan."""
    grid = engine._grid  # noqa: SLF001
    r, c = grid.world_to_cell(x, y)
    return float(grid.log_odds[r - 2 : r + 3, c - 2 : c + 3].max())


def _banded_engine(tmp_path):
    engine = _engine(tmp_path, builtin_loop_closure=False)
    engine._mode = MODE_MAPPING  # noqa: SLF001
    for i in range(3):
        _paint_banded(engine, conv.Pose2D(0.0, 0.0, 0.0), 2.0, now=float(i))
    assert len(engine._height_bands) == 2  # noqa: SLF001
    assert _odds(engine, 2.0, 0.0) > 0.5
    return engine


def test_clear_region_sticks_after_next_banded_paint(tmp_path):
    """Live paint rebuilds the grid from height bands; the clear must be in them."""
    engine = _banded_engine(tmp_path)
    engine.clear_region(1.5, -1.0, 2.5, 1.0)
    assert _odds(engine, 2.0, 0.0) < 0.0

    _paint_banded(engine, conv.Pose2D(0.0, 3.0, 0.0), 4.0, now=10.0)
    assert _odds(engine, 2.0, 0.0) < 0.0


def test_clear_obstacles_sticks_after_next_banded_paint(tmp_path):
    engine = _banded_engine(tmp_path)
    engine.clear_obstacles(2.0, 0.0, 0.3)
    assert _odds(engine, 2.0, 0.0) < 0.0

    _paint_banded(engine, conv.Pose2D(0.0, 3.0, 0.0), 4.0, now=10.0)
    assert _odds(engine, 2.0, 0.0) < 0.0


def test_rerender_sticks_after_next_banded_paint(tmp_path):
    """A loop-closure render that moves walls must not be undone by the next scan."""
    engine = _banded_engine(tmp_path)
    with engine._lock:  # noqa: SLF001
        assert len(engine._graph) >= 1  # noqa: SLF001
        engine._graph.apply_world_correction_from(  # noqa: SLF001
            0, conv.Pose2D(-1.0, 0.0, 0.0)
        )
    out = engine._optimize_and_render(None)  # noqa: SLF001
    assert out["status"] == "rebuilt"
    assert _odds(engine, 1.0, 0.0) > 0.5
    assert _odds(engine, 2.0, 0.0) <= 0.5

    _paint_banded(engine, conv.Pose2D(0.0, 6.0, 0.0), 4.0, now=10.0)
    assert _odds(engine, 1.0, 0.0) > 0.5
    assert _odds(engine, 2.0, 0.0) <= 0.5


def _hallway_engine(tmp_path):
    engine = _engine(tmp_path)
    grid = occ.empty_grid(resolution=RES, size_m=20.0, origin_x=-2.0, origin_y=-2.0)
    for x in np.linspace(0.0, 8.0, 41):
        for y in np.linspace(-0.5, 0.5, 11):
            r, c = grid.world_to_cell(float(x), float(y))
            if 0 <= r < grid.log_odds.shape[0] and 0 <= c < grid.log_odds.shape[1]:
                grid.log_odds[r, c] = occ.L_MAX
    with engine._lock:  # noqa: SLF001
        engine._grid = grid  # noqa: SLF001
        engine._graph.nodes = [  # noqa: SLF001
            _node(0.0, 0.0, 0.0),
            _node(3.0, 0.0, 3.0),
            _node(7.0, 0.0, 7.0),
        ]
        engine._graph.edges = _chain_edges(3)  # noqa: SLF001
        engine._graph_frozen = 3  # noqa: SLF001
    return engine


def _refill(engine, x0: float, x1: float) -> None:
    with engine._lock:  # noqa: SLF001
        for x in np.linspace(x0, x1, 21):
            for y in np.linspace(-0.4, 0.4, 9):
                r, c = engine._grid.world_to_cell(float(x), float(y))  # noqa: SLF001
                h, w = engine._grid.log_odds.shape  # noqa: SLF001
                if 0 <= r < h and 0 <= c < w:
                    engine._grid.log_odds[r, c] = occ.L_MAX  # noqa: SLF001


def test_walls_remapped_after_clear_survive_reload(tmp_path):
    engine = _hallway_engine(tmp_path)
    engine.clear_region(1.5, -1.0, 4.5, 1.0)
    # Mapping lidar draws the hallway again before Save.
    _refill(engine, 2.0, 3.0)

    map_dir = tmp_path / "m"
    engine.save_map(map_dir, repair=False)
    resumed = _engine(tmp_path)
    resumed.configure_mode(MODE_MAPPING, map_dir)
    reloaded = occ.to_occupancy_int16(resumed._grid)  # noqa: SLF001
    r_wall, c_wall = resumed._grid.world_to_cell(2.5, 0.0)  # noqa: SLF001
    assert reloaded[r_wall, c_wall] == 100
    # The rest of the hole is still free and unjudged by matching.
    r_free, c_free = resumed._grid.world_to_cell(4.0, 0.0)  # noqa: SLF001
    assert reloaded[r_free, c_free] == 0
    hole = float(resumed._grid.log_odds[r_free, c_free])  # noqa: SLF001
    assert hole == occ.CLEARED_LOG_ODDS
    assert resumed._grid_edits == engine._grid_edits  # noqa: SLF001


def test_clear_box_saved_without_seq_still_wipes_on_reload(tmp_path):
    """Older saves could hold walls band paint put back; keep wiping those."""
    engine = _hallway_engine(tmp_path)
    engine.clear_region(1.5, -1.0, 4.5, 1.0)
    _refill(engine, 2.0, 4.0)
    with engine._lock:  # noqa: SLF001
        engine._grid_edits = [e[:5] for e in engine._grid_edits]  # noqa: SLF001

    map_dir = tmp_path / "m"
    engine.save_map(map_dir, repair=False)
    on_disk = persistence.load_log_odds(map_dir)
    assert on_disk is not None
    r_in, c_in = on_disk.world_to_cell(3.0, 0.0)
    assert occ.to_occupancy_int16(on_disk)[r_in, c_in] == 100

    resumed = _engine(tmp_path)
    resumed.configure_mode(MODE_MAPPING, map_dir)
    reloaded = occ.to_occupancy_int16(resumed._grid)  # noqa: SLF001
    r_in, c_in = resumed._grid.world_to_cell(3.0, 0.0)  # noqa: SLF001
    assert reloaded[r_in, c_in] == 0
    # Re-renders this session erase every loaded keyframe in the box.
    last_seq = resumed._graph.last_seq  # noqa: SLF001
    assert ("clear_box", 1.5, -1.0, 4.5, 1.0, last_seq) in resumed._grid_edits  # noqa: SLF001


def _two_wall_scan() -> conv.LaserScan2D:
    """Walls 3 m behind and 2 m ahead of the robot."""
    ranges = np.full(360, np.inf)
    angles = -np.pi + np.arange(360) * (2 * np.pi / 360)
    ahead = np.abs(angles) <= np.radians(30)
    behind = np.abs(np.abs(angles) - np.pi) <= np.radians(30)
    ranges[ahead] = 2.0 / np.cos(angles[ahead])
    ranges[behind] = 3.0 / np.abs(np.cos(angles[behind]))
    return conv.LaserScan2D(ranges, -np.pi, 2 * np.pi / 360, 0.05, 20.0)


def test_scan_into_cleared_hole_still_fits(tmp_path):
    """Returns from walls inside a cleared hole must not count as misses."""
    engine = _engine(tmp_path, builtin_loop_closure=False)
    grid = occ.empty_grid(resolution=RES, size_m=12.0, origin_x=-6.0, origin_y=-6.0)
    lo = grid.log_odds
    lo[:] = occ.L_MIN
    r, c = grid.world_to_cell(-3.0, 0.0)
    lo[:, c - 1 : c + 2] = occ.L_MAX
    with engine._lock:  # noqa: SLF001
        engine._grid = grid  # noqa: SLF001
        engine._invalidate_occ_cache()  # noqa: SLF001
    scan = _two_wall_scan()
    pose = conv.Pose2D(0.0, 0.0, 0.0)
    # No wall at x=2 on the map yet: those returns land in known free space.
    assert not engine._scan_fits_map(scan, pose)  # noqa: SLF001

    engine.clear_region(1.0, -3.0, 3.0, 3.0)
    assert engine._scan_fits_map(scan, pose)  # noqa: SLF001


def test_rerender_keeps_walls_painted_after_clear(tmp_path):
    """A re-render erases only the scans from before the clear."""
    engine = _banded_engine(tmp_path)
    engine.clear_region(1.5, -1.0, 2.5, 1.0)
    # Far enough from the first keyframe that the graph keeps these scans.
    for i in range(3):
        _paint_banded(engine, conv.Pose2D(0.0, 0.62, 0.0), 2.33, now=10.0 + i)
    assert _odds(engine, 2.33, 0.62) > 0.5

    out = engine._optimize_and_render(None)  # noqa: SLF001
    assert out["status"] == "rebuilt"
    assert _odds(engine, 2.33, 0.62) > 0.5
    assert _odds(engine, 2.0, 0.0) <= 0.5


def test_cleared_value_is_not_reachable_by_paint():
    """Live paint must never land on the sentinel matching treats as unjudged."""
    f = np.float32
    loaded = occ.from_occupancy_int16(
        np.array([[-1, 0, 100]], dtype=np.int16),
        resolution=RES,
        origin_x=0.0,
        origin_y=0.0,
    ).log_odds.ravel()
    seen = {f(v) for v in loaded} | {f(0.0), f(occ.L_MIN), f(occ.L_MAX)}
    frontier = set(seen)
    for _ in range(120):
        nxt = set()
        for v in frontier:
            for d in (f(occ.L_OCC), f(occ.L_FREE)):
                w = f(np.clip(f(v + d), f(occ.L_MIN), f(occ.L_MAX)))
                if w not in seen:
                    seen.add(w)
                    nxt.add(w)
        frontier = nxt
    assert f(occ.CLEARED_LOG_ODDS) not in seen
    assert (
        occ.to_occupancy_int16(
            occ.LogOddsGrid(
                log_odds=np.array([[occ.CLEARED_LOG_ODDS]], dtype=np.float32),
                resolution=RES,
                origin_x=0.0,
                origin_y=0.0,
            )
        )[0, 0]
        == 0
    )

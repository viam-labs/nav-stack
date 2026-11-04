"""Region clear: occupancy AABB + keyframe drop, graph-store v2 clear_box."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import graph_store
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
    edges = [
        pg.GraphEdge(pg.PRIOR_NODE, 0, conv.Pose2D(0.0, 0.0, 0.0), info.copy())
    ]
    for i in range(n - 1):
        edges.append(
            pg.GraphEdge(i, i + 1, conv.Pose2D(1.0, 0.0, 0.0), info.copy())
        )
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
    removed, n_edges = graph.remove_nodes(
        lambda _i, node: 0.5 <= node.pose.x <= 2.5
    )
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
        assert z["edit_params"].shape == (3, 4)
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
    r_out, c_out = engine._grid.world_to_cell(7.0, 0.0)  # noqa: SLF001
    assert after[r_out, c_out] == 100

    assert ("clear_box", 1.5, -1.0, 4.5, 1.0) in engine._grid_edits  # noqa: SLF001
    assert engine._map_health_base is None  # noqa: SLF001
    assert engine._session_baseline is not None  # noqa: SLF001
    assert np.array_equal(  # noqa: SLF001
        engine._session_baseline.log_odds, engine._grid.log_odds  # noqa: SLF001
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
    assert ("clear_box", 1.5, -1.0, 4.5, 1.0) in resumed._grid_edits  # noqa: SLF001
    reloaded = occ.to_occupancy_int16(resumed._grid)  # noqa: SLF001
    r_in, c_in = resumed._grid.world_to_cell(3.0, 0.0)  # noqa: SLF001
    assert reloaded[r_in, c_in] == 0
    r_out, c_out = resumed._grid.world_to_cell(7.0, 0.0)  # noqa: SLF001
    assert reloaded[r_out, c_out] == 100


def test_clear_region_rejects_empty_box(tmp_path):
    engine = _engine(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        engine.clear_region(1.0, 0.0, 1.0, 2.0)

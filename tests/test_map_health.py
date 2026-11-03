"""Unit tests for live occupancy map-health scoring."""

from __future__ import annotations

import numpy as np

from src.slam_builtin import map_health


def _blank(h: int, w: int, res: float = 0.05):
    grid = np.full((h, w), -1, dtype=np.int16)
    return grid, res


def _paint_wall(grid, r0, r1, c0, c1):
    grid[r0:r1, c0:c1] = 100


def test_cold_map_when_too_few_occupied_cells():
    grid, res = _blank(80, 80)
    _paint_wall(grid, 40, 41, 10, 30)
    h = map_health.assess_occupancy(grid, resolution=res, origin_x=0.0, origin_y=0.0)
    assert h.status == "cold"
    assert h.occupied_cells < 400


def test_clean_corridor_is_ok():
    # One ~2 m wide hallway: walls 40 cells apart (2.0 m) — not a ghost gap.
    grid, res = _blank(200, 200)
    grid[20:180, 20:180] = 0
    _paint_wall(grid, 20, 180, 40, 42)
    _paint_wall(grid, 20, 180, 80, 82)
    h = map_health.assess_occupancy(grid, resolution=res, origin_x=-5.0, origin_y=-5.0)
    assert h.occupied_cells >= 400
    assert h.status in ("ok", "watch")
    assert h.ghost_frac < map_health._DEGRADING_GHOST


def test_ghost_parallel_walls_raise_health():
    # Two copies of the same wall ~0.3 m apart with free between — classic smear.
    grid, res = _blank(220, 220)
    grid[20:200, 20:200] = 0
    for c in (60, 66, 100, 106, 140, 146):
        _paint_wall(grid, 30, 190, c, c + 2)
    h = map_health.assess_occupancy(grid, resolution=res, origin_x=0.0, origin_y=0.0)
    assert h.ghost_frac >= map_health._WATCH_GHOST
    assert h.status in ("watch", "degrading")
    assert h.as_dict()["status"] == h.status
    assert isinstance(h.as_dict()["hotspots"], list)


def test_rotated_tiles_increase_angle_std():
    grid, res = _blank(240, 240)
    grid[:, :] = 0
    # Axis-aligned block.
    _paint_wall(grid, 40, 42, 40, 100)
    _paint_wall(grid, 40, 100, 40, 42)
    # Diagonal-ish block elsewhere (45° PCA).
    for k in range(60):
        r = 140 + k
        c = 140 + k
        if 0 <= r < 240 and 0 <= c < 240:
            grid[r : r + 2, c : c + 2] = 100
            grid[r : r + 2, c + 8 : c + 10] = 100
    h = map_health.assess_occupancy(grid, resolution=res, origin_x=0.0, origin_y=0.0)
    assert h.angle_std_deg > 0.0


def _ghost_hallway(h: int = 220, w: int = 220):
    grid, res = _blank(h, w)
    grid[20:200, 20:200] = 0
    for c in (60, 66, 100, 106, 140, 146):
        _paint_wall(grid, 30, 190, c, c + 2)
    return grid, res


def test_baseline_hides_inherited_ghosts():
    grid, res = _ghost_hallway()
    whole = map_health.assess_occupancy(
        grid, resolution=res, origin_x=0.0, origin_y=0.0
    )
    session = map_health.assess_occupancy(
        grid, resolution=res, origin_x=0.0, origin_y=0.0, baseline=grid.copy()
    )
    assert whole.status in ("watch", "degrading")
    assert session.status == "cold"
    assert session.occupied_cells == 0


def test_new_copy_beside_baseline_wall_is_flagged():
    # Baseline: clean walls. Session repaints each one again 0.3 m over.
    base, res = _blank(220, 220)
    base[20:200, 20:200] = 0
    for c in (60, 100, 140):
        _paint_wall(base, 30, 190, c, c + 2)
    live = base.copy()
    for c in (66, 106, 146):
        _paint_wall(live, 30, 190, c, c + 2)
    h = map_health.assess_occupancy(
        live, resolution=res, origin_x=0.0, origin_y=0.0, baseline=base
    )
    assert h.status in ("watch", "degrading")
    assert h.ghost_frac >= map_health._WATCH_GHOST
    assert h.hotspots


def test_clean_new_walls_ok_on_a_torn_baseline():
    base, res = _ghost_hallway(220, 440)
    live = base.copy()
    live[20:200, 220:420] = 0
    _paint_wall(live, 30, 190, 260, 262)
    _paint_wall(live, 30, 190, 320, 322)
    h = map_health.assess_occupancy(
        live, resolution=res, origin_x=0.0, origin_y=0.0, baseline=base
    )
    assert h.status == "ok"
    assert h.ghost_frac < map_health._WATCH_GHOST


def test_align_grid_reindexes_by_origin():
    src = np.full((4, 5), -1, dtype=np.int16)
    src[1, 2] = 100
    out = map_health.align_grid(
        src,
        origin_x=1.0,
        origin_y=0.5,
        shape=(10, 10),
        to_origin_x=0.0,
        to_origin_y=0.0,
        resolution=0.5,
    )
    assert out.shape == (10, 10)
    assert out[2, 4] == 100
    assert int((out == 100).sum()) == 1
    clipped = map_health.align_grid(
        src,
        origin_x=-1.0,
        origin_y=0.0,
        shape=(3, 3),
        to_origin_x=0.0,
        to_origin_y=0.0,
        resolution=0.5,
    )
    assert clipped[1, 0] == 100


def test_engine_scores_session_against_resumed_map(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine(tmp_path)
    torn, res = _ghost_hallway()
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        torn, resolution=res, origin_x=-2.0, origin_y=-3.0
    )
    engine._graph_base = occ.copy_grid(engine._grid)  # noqa: SLF001
    out = engine.refresh_map_health(force=True)
    assert out["scope"] == "session"
    assert out["status"] == "cold"
    assert out["baseline"]["status"] in ("watch", "degrading")
    assert engine._map_health_boosts == 0  # noqa: SLF001

    # The session draws a ghost copy of the outer wall 0.3 m over.
    live = torn.copy()
    for c in (54, 94, 134):
        _paint_wall(live, 30, 190, c, c + 2)
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        live, resolution=res, origin_x=-2.0, origin_y=-3.0
    )
    engine._generation += 1  # noqa: SLF001
    out = engine.refresh_map_health(force=True)
    assert out["scope"] == "session"
    assert out["status"] in ("watch", "degrading")
    assert engine._map_health_boosts == 1  # noqa: SLF001
    assert (
        engine.diagnostics()["map_health"]["baseline"]["status"]
        == (out["baseline"]["status"])
    )


def _engine(tmp_path):
    from src.config import SlamConfig
    from src.nav.stores.maps import MapStore
    from src.slam_builtin.engine import BuiltinSlamEngine

    class _Sensors:
        def get_scan(self, max_age_s: float = 2.0, *, fresh: bool = False):
            return None

        def get_odom(self):
            return None

    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": str(tmp_path), "mode": "mapping"}
    )
    return BuiltinSlamEngine(cfg, _Sensors(), MapStore(str(tmp_path)), rate_hz=5.0)  # type: ignore[arg-type]


def test_engine_pending_and_revisit_react_to_health(tmp_path):
    engine = _engine(tmp_path)
    base_s, base_tries = engine._pending_limits()  # noqa: SLF001
    assert engine.mapping_revisit_interval_s(20.0) == 20.0

    engine._map_health = {"status": "watch"}  # noqa: SLF001
    watch_s, watch_tries = engine._pending_limits()  # noqa: SLF001
    assert watch_s > base_s
    assert watch_tries >= base_tries
    assert engine.mapping_revisit_interval_s(20.0) == 10.0

    engine._map_health = {"status": "degrading"}  # noqa: SLF001
    deg_s, deg_tries = engine._pending_limits()  # noqa: SLF001
    assert deg_s > watch_s
    assert deg_tries >= watch_tries
    assert engine.mapping_revisit_interval_s(20.0) == 6.0
    assert "map_health" in engine.diagnostics()

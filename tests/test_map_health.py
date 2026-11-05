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
    assert h.ghost_frac >= map_health._DEGRADING_GHOST
    # Immature maps stay cold with no live hotspots; raw still flags tear.
    assert h.mature is False
    assert h.status == "cold"
    assert h.raw_status == "degrading"
    assert h.hotspots == ()
    assert h.as_dict()["status"] == h.status
    assert h.as_dict()["hotspots"] == []


def test_mature_severe_ghost_is_degrading():
    grid, res = _blank(400, 400)
    grid[20:380, 20:380] = 0
    for c in (60, 66, 100, 106, 140, 146, 200, 206, 260, 266):
        _paint_wall(grid, 30, 370, c, c + 2)
    h = map_health.assess_occupancy(grid, resolution=res, origin_x=0.0, origin_y=0.0)
    assert h.mature is True
    assert h.occupied_cells >= map_health._MATURE_OCC_CELLS
    assert h.status == "degrading"
    assert h.raw_status == "degrading"


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
    assert whole.mature is False
    assert whole.status == "cold"
    assert whole.raw_status == "degrading"
    assert session.status == "cold"
    assert session.occupied_cells == 0


def test_new_copy_beside_baseline_wall_is_flagged():
    # Baseline: clean walls. Session repaints each one again 0.3 m over.
    # Use a mature-sized map so live status can escalate past cold.
    base, res = _blank(400, 400)
    base[20:380, 20:380] = 0
    for c in (60, 100, 140, 200, 260):
        _paint_wall(base, 30, 370, c, c + 2)
    live = base.copy()
    for c in (66, 106, 146, 206, 266):
        _paint_wall(live, 30, 370, c, c + 2)
    h = map_health.assess_occupancy(
        live, resolution=res, origin_x=0.0, origin_y=0.0, baseline=base
    )
    assert h.ghost_frac >= map_health._WATCH_GHOST
    assert h.raw_status in ("watch", "degrading")
    # Session-only cells can still be below mature; live may stay cold.
    if h.mature:
        assert h.status in ("watch", "degrading")
        assert h.hotspots
    else:
        assert h.status == "cold"
        assert h.hotspots == ()


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
    engine._session_baseline = engine._graph_base  # noqa: SLF001
    out = engine.refresh_map_health(force=True)
    assert out["scope"] == "session"
    assert out["status"] == "cold"
    assert out["baseline"]["raw_status"] == "degrading"
    assert out["baseline"]["trusted"] is False
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
    # Session-only new cells on a small map stay cold (no premature watch).
    assert out["status"] == "cold"
    assert out["raw_status"] in ("watch", "degrading", "cold", "ok")
    assert engine._map_health_boosts == 0  # noqa: SLF001
    assert (
        engine.diagnostics()["map_health"]["baseline"]["raw_status"]
        == out["baseline"]["raw_status"]
    )


def _resume_on(tmp_path, cells, res):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine(tmp_path)
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        cells, resolution=res, origin_x=-2.0, origin_y=-3.0
    )
    engine._graph_base = occ.copy_grid(engine._grid)  # noqa: SLF001
    engine._session_baseline = engine._graph_base  # noqa: SLF001
    engine.refresh_map_health(force=True)
    return engine


def _clean_hallway():
    grid, res = _blank(220, 220)
    grid[20:200, 20:200] = 0
    for c in (60, 100, 140):
        _paint_wall(grid, 30, 190, c, c + 2)
    return grid, res


def test_torn_resumed_map_is_not_used_by_loop_closure(tmp_path, monkeypatch):
    from src.slam_builtin import loop_closure
    from src.slam_builtin.engine import BuiltinSlamEngine

    seen = []

    def fake_detect(snap, params, *, resolution, base=None, **_kw):
        seen.append(base)
        return None, "no_candidates"

    monkeypatch.setattr(loop_closure, "detect", fake_detect)
    monkeypatch.setattr(BuiltinSlamEngine, "_loop_closure_enabled", lambda self: True)

    torn = _resume_on(tmp_path, *_ghost_hallway())
    health = torn.map_health()
    assert health["baseline"]["trusted"] is False
    assert health["recommendation"] == "clear_map"
    torn.run_loop_closure()
    assert seen[-1] is None

    clean = _resume_on(tmp_path, *_clean_hallway())
    health = clean.map_health()
    assert health["baseline"]["trusted"] is True
    assert "recommendation" not in health
    clean.run_loop_closure()
    assert seen[-1] is clean._graph_base  # noqa: SLF001


def test_revisit_does_not_snap_session_onto_torn_resumed_map(tmp_path):
    import math

    from src.geom import conversions as conv

    scan = conv.LaserScan2D(np.full(80, 2.0), -math.pi, 2 * math.pi / 80, 0.1, 10.0)
    start = conv.Pose2D(0.0, 0.0, 0.0)
    target = conv.Pose2D(1.8, 0.0, 0.0)

    torn = _resume_on(tmp_path, *_ghost_hallway())
    torn._graph.add_scan(start, scan)  # noqa: SLF001
    out = torn.apply_map_pose_correction(target, score=0.9)
    assert out["method"] == "rejected_untrusted_base"
    assert torn.get_pose() == start

    clean = _resume_on(tmp_path, *_clean_hallway())
    clean._graph.add_scan(start, scan)  # noqa: SLF001
    out = clean.apply_map_pose_correction(target, score=0.9)
    assert out["method"] == "rigid"


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


def _engine_heal(tmp_path, **extra):
    from src.config import SlamConfig
    from src.nav.stores.maps import MapStore
    from src.slam_builtin.engine import BuiltinSlamEngine

    class _Sensors:
        def get_scan(self, max_age_s: float = 2.0, *, fresh: bool = False):
            return None

        def get_odom(self):
            return None

    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": str(tmp_path),
            "mode": "mapping",
            "map_health_auto_heal": True,
            "map_health_auto_clear": True,
            "map_health_hotspot_persist_n": 2,
            "map_health_auto_clear_cooldown_s": 0.0,
            "map_health_auto_clear_pad_m": 2.5,
            "map_health_auto_clear_min_keyframes": 0,
            "map_health_auto_clear_near_m": 50.0,
            "builtin_repair_on_save": False,
            **extra,
        }
    )
    return BuiltinSlamEngine(cfg, _Sensors(), MapStore(str(tmp_path)), rate_hz=5.0)  # type: ignore[arg-type]


def _mature_ghost_hallway():
    grid, res = _blank(400, 400)
    grid[20:380, 20:380] = 0
    for c in (60, 66, 100, 106, 140, 146, 200, 206, 260, 266):
        _paint_wall(grid, 30, 370, c, c + 2)
    return grid, res


def test_default_auto_clear_follows_heal(tmp_path):
    from src.config import SlamConfig

    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": str(tmp_path), "mode": "mapping"}
    )
    assert cfg.map_health_auto_heal is True
    assert cfg.map_health_auto_clear is True
    off = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": str(tmp_path),
            "mode": "mapping",
            "map_health_auto_heal": False,
        }
    )
    assert off.map_health_auto_clear is False


def test_hotspot_auto_heal_tracks_then_clears(tmp_path):
    import time

    from src.geom import conversions as conv
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine_heal(tmp_path)
    torn, res = _mature_ghost_hallway()
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        torn, resolution=res, origin_x=0.0, origin_y=0.0
    )
    engine._session_baseline = None  # noqa: SLF001
    engine._pose = conv.Pose2D(7.0, 10.0, 0.0)  # noqa: SLF001

    t0 = time.monotonic()
    first = engine.refresh_map_health(force=True)
    assert first["status"] == "degrading"
    assert first["mature"] is True
    assert first["hotspots"]
    tracked = first["hotspot_heal"]["tracked"]
    assert tracked is not None
    assert tracked["streak"] == 1
    assert first["hotspot_heal"]["clears"] == 0
    assert engine.mapping_revisit_interval_s(20.0) == 6.0

    # Immediate re-score must not advance the clear streak (time gate).
    engine._generation += 1  # noqa: SLF001
    churn = engine.refresh_map_health(force=True)
    assert churn["hotspot_heal"]["tracked"]["streak"] == 1
    assert churn["hotspot_heal"]["clears"] == 0

    # Advance the streak clock, then clear.
    engine._hotspot_track["last_count_at"] = t0 - 6.0  # noqa: SLF001
    engine._generation += 1  # noqa: SLF001
    second = engine.refresh_map_health(force=True)
    assert second["hotspot_heal"]["clears"] == 1
    assert second.get("auto_cleared") is not None
    assert engine._operator_trusted_paint is True  # noqa: SLF001
    assert second["recommendation"] == "remap_cleared_region"
    assert second["hotspot_heal"]["tracked"] is None
    assert engine.diagnostics()["hotspot_heal"]["clears"] == 1


def test_hotspot_auto_heal_watch_does_not_clear(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine_heal(tmp_path, map_health_hotspot_persist_n=1)
    engine._mode = MODE_MAPPING  # noqa: SLF001
    cells = np.full((80, 80), 0, dtype=np.int16)
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        cells, resolution=0.05, origin_x=0.0, origin_y=0.0
    )
    engine._session_baseline = None  # noqa: SLF001

    def fake_assess(*_a, **_kw):
        return map_health.MapHealth(
            status="watch",
            ghost_frac=0.15,
            thick_frac=0.0,
            angle_std_deg=5.0,
            occupied_cells=500,
            hotspots=((0.4, 1.0, 2.0),),
            mature=False,
            raw_status="watch",
        )

    import src.slam_builtin.map_health as mh

    orig = mh.assess_occupancy
    mh.assess_occupancy = fake_assess  # type: ignore[assignment]
    try:
        out = engine.refresh_map_health(force=True)
    finally:
        mh.assess_occupancy = orig  # type: ignore[assignment]
    assert out["status"] == "watch"
    assert out["hotspot_heal"]["tracked"]["streak"] == 1
    assert out["hotspot_heal"]["clears"] == 0
    assert "auto_cleared" not in out


def test_hotspot_auto_clear_respects_cooldown(tmp_path):
    import time

    from src.geom import conversions as conv
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine_heal(
        tmp_path,
        map_health_hotspot_persist_n=1,
        map_health_auto_clear_cooldown_s=999.0,
    )
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._pose = conv.Pose2D(1.0, 2.0, 0.0)  # noqa: SLF001
    engine._hotspot_last_clear_at = time.monotonic()  # noqa: SLF001
    health = {
        "status": "degrading",
        "hotspots": [{"frac": 0.5, "x": 1.0, "y": 2.0}],
    }
    out = engine._maybe_auto_heal_hotspot(health, now=time.monotonic())  # noqa: SLF001
    assert out is None
    assert engine._hotspot_heal["skips"].get("cooldown", 0) >= 1  # noqa: SLF001
    assert engine._hotspot_heal["clears"] == 0  # noqa: SLF001


def test_hotspot_auto_clear_skips_when_too_few_keyframes(tmp_path):
    import time

    from src.geom import conversions as conv
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine_heal(
        tmp_path,
        map_health_hotspot_persist_n=1,
        map_health_auto_clear_min_keyframes=40,
    )
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._pose = conv.Pose2D(1.0, 2.0, 0.0)  # noqa: SLF001
    health = {
        "status": "degrading",
        "hotspots": [{"frac": 0.5, "x": 1.0, "y": 2.0}],
    }
    out = engine._maybe_auto_heal_hotspot(health, now=time.monotonic())  # noqa: SLF001
    assert out is None
    assert engine._hotspot_heal["skips"].get("too_few_keyframes", 0) >= 1  # noqa: SLF001


def test_hotspot_auto_clear_skips_when_too_far(tmp_path):
    import time

    from src.geom import conversions as conv
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine_heal(
        tmp_path,
        map_health_hotspot_persist_n=1,
        map_health_auto_clear_near_m=2.0,
    )
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._pose = conv.Pose2D(20.0, 20.0, 0.0)  # noqa: SLF001
    health = {
        "status": "degrading",
        "hotspots": [{"frac": 0.5, "x": 1.0, "y": 2.0}],
    }
    out = engine._maybe_auto_heal_hotspot(health, now=time.monotonic())  # noqa: SLF001
    assert out is None
    assert engine._hotspot_heal["skips"].get("too_far", 0) >= 1  # noqa: SLF001


def test_health_rate_limits_generation_churn(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine(tmp_path)
    torn, res = _ghost_hallway()
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        torn, resolution=res, origin_x=0.0, origin_y=0.0
    )
    engine._session_baseline = None  # noqa: SLF001
    first = engine.refresh_map_health(force=True)
    gen_at = engine._map_health_generation  # noqa: SLF001
    engine._generation += 5  # noqa: SLF001
    second = engine.refresh_map_health(force=False)
    # Cached within MIN_S despite generation moving.
    assert second["status"] == first["status"]
    assert engine._map_health_generation == gen_at  # noqa: SLF001


def test_hotspot_auto_clear_can_be_disabled(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine_heal(
        tmp_path,
        map_health_auto_clear=False,
        map_health_hotspot_persist_n=1,
    )
    torn, res = _mature_ghost_hallway()
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        torn, resolution=res, origin_x=0.0, origin_y=0.0
    )
    engine._session_baseline = None  # noqa: SLF001
    out = engine.refresh_map_health(force=True)
    assert out["status"] == "degrading"
    assert out["hotspot_heal"]["tracked"]["streak"] == 1
    assert out["hotspot_heal"]["auto_clear"] is False
    assert out["hotspot_heal"]["clears"] == 0
    assert engine._operator_trusted_paint is False  # noqa: SLF001


def test_immature_map_stays_cold_without_hotspots(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine(tmp_path)
    torn, res = _ghost_hallway()
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        torn, resolution=res, origin_x=0.0, origin_y=0.0
    )
    engine._session_baseline = None  # noqa: SLF001
    out = engine.refresh_map_health(force=True)
    assert out["mature"] is False
    assert out["status"] == "cold"
    assert out["hotspots"] == []
    assert out["raw_status"] == "degrading"
    assert out["hotspot_heal"]["tracked"] is None


def test_baseline_two_cell_slack_ignores_revisit_jitter():
    base, res = _blank(400, 400)
    base[20:380, 20:380] = 0
    for c in (60, 100, 140, 200, 260):
        _paint_wall(base, 30, 370, c, c + 2)
    # ~10 cm offset (2 cells) — normal re-drive jitter, should not be "new".
    live = base.copy()
    for c in (62, 102, 142, 202, 262):
        _paint_wall(live, 30, 370, c, c + 2)
    h = map_health.assess_occupancy(
        live, resolution=res, origin_x=0.0, origin_y=0.0, baseline=base
    )
    assert h.status in ("ok", "cold")
    assert h.ghost_frac < map_health._WATCH_GHOST


def test_fresh_map_ok_checkpoint_ignores_revisit_jitter(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine(tmp_path)
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._session_baseline = None  # noqa: SLF001

    clean, res = _blank(400, 400)
    clean[20:380, 20:380] = 0
    for c in (60, 100, 140, 200, 260):
        _paint_wall(clean, 30, 370, c, c + 2)
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        clean, resolution=res, origin_x=0.0, origin_y=0.0
    )
    first = engine.refresh_map_health(force=True)
    assert first["status"] == "ok"
    assert first["scope"] == "map"
    assert engine._map_health_ok_cells is not None  # noqa: SLF001

    # ~10 cm revisit thickening — full-map can look busier, but against the
    # ok checkpoint (+ 2-cell dilate) it must stay quiet.
    slipped = clean.copy()
    for c in (62, 102, 142, 202, 262):
        _paint_wall(slipped, 30, 370, c, c + 2)
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        slipped, resolution=res, origin_x=0.0, origin_y=0.0
    )
    engine._generation += 1  # noqa: SLF001
    second = engine.refresh_map_health(force=True)
    assert second["scope"] == "since_ok"
    assert second["status"] in ("ok", "cold")
    assert second["hotspots"] == []


def test_fresh_map_ok_checkpoint_still_flags_new_double(tmp_path):
    from src.slam_builtin import occupancy as occ
    from src.slam_builtin.engine import MODE_MAPPING

    engine = _engine(tmp_path)
    engine._mode = MODE_MAPPING  # noqa: SLF001
    engine._session_baseline = None  # noqa: SLF001

    clean, res = _blank(400, 400)
    clean[20:380, 20:380] = 0
    for c in (60, 100, 140, 200, 260):
        _paint_wall(clean, 30, 370, c, c + 2)
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        clean, resolution=res, origin_x=0.0, origin_y=0.0
    )
    assert engine.refresh_map_health(force=True)["status"] == "ok"

    # Clear ~30 cm double beyond dilate slack — real tear vs last ok snapshot.
    doubled = clean.copy()
    for c in (66, 106, 146, 206, 266):
        _paint_wall(doubled, 30, 370, c, c + 2)
    engine._grid = occ.from_occupancy_int16(  # noqa: SLF001
        doubled, resolution=res, origin_x=0.0, origin_y=0.0
    )
    engine._generation += 1  # noqa: SLF001
    out = engine.refresh_map_health(force=True)
    assert out["scope"] == "since_ok"
    assert out["status"] in ("watch", "degrading")
    assert out["hotspots"]

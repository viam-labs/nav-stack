"""A loop-closure map rebuild must not freeze the pose or drop the robot's motion.

A rebuild re-renders every keyframe of the session: five to nine seconds at
500 to 800 keyframes. Scans stored while that ran used to trigger a second
full render, this one holding the pose lock, so no tick could run. The pose
was then reset in a way that discarded the wheel motion since the last tick.
A robot driving at 0.5 m/s came out of a rebuild meters from where the map
said it was. On tracer2a every loss of the map during explore followed a
rebuild within seconds (2.2 m once; 2.8 m and 45 degrees another time).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import occupancy as occ
from src.slam_builtin import pose_graph as pg
from src.slam_builtin.engine import BuiltinSlamEngine

from .test_loop_closure import _raycast, _Sensors, _truth_world

TRUTH = _truth_world()


def _engine(tmp_path, keyframes: int = 8) -> BuiltinSlamEngine:
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
        _insert(engine, conv.Pose2D(0.6 * k, 0.0, 0.0), now=0.2 * k)
    return engine


def _insert(engine: BuiltinSlamEngine, pose: conv.Pose2D, now: float) -> None:
    with engine._lock:  # noqa: SLF001
        engine._pose = pose  # noqa: SLF001
    engine._paint_mapping_scan(_raycast(TRUTH, pose), pose, now)  # noqa: SLF001


def _during_first_render(monkeypatch, engine: BuiltinSlamEngine, act):
    """Run ``act`` while the rebuild's first render is drawing; return the calls."""
    real = pg.render_grid
    calls = []

    def render(snap, **kw):
        calls.append(bool(engine._lock._is_owned()))  # noqa: SLF001
        if len(calls) == 1:
            act()
        return real(snap, **kw)

    monkeypatch.setattr(pg, "render_grid", render)
    return calls


def test_scans_stored_during_a_rebuild_do_not_put_a_render_under_the_pose_lock(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    late = [conv.Pose2D(0.6 * k, 0.0, 0.0) for k in range(8, 14)]
    held = _during_first_render(
        monkeypatch,
        engine,
        lambda: [_insert(engine, p, now=5.0 + i) for i, p in enumerate(late)],
    )

    out = engine.optimize_pose_graph()

    assert out["status"] == "rebuilt"
    # No full render ran with the pose lock held.
    assert held and not any(held)
    # The scans that arrived meanwhile are on the map all the same: the
    # result is what rendering the whole graph again would give.
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
        fresh = pg.render_grid(
            engine._graph.snapshot(),  # noqa: SLF001
            resolution=grid.resolution,
            base=engine._graph_base,  # noqa: SLF001
        )
    assert (grid.origin_x, grid.origin_y) == (fresh.origin_x, fresh.origin_y)
    assert np.array_equal(occ.to_occupancy_int16(grid), occ.to_occupancy_int16(fresh))
    assert len(engine._graph) == 14  # noqa: SLF001


def test_a_rebuild_keeps_the_motion_made_while_it_ran(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    odom = [conv.Pose2D(0.0, 0.0, 0.0)]
    engine._sensors.get_odom = lambda: conv.OdomReading(  # noqa: SLF001
        0.5, 0.0, 0.0, pose=odom[0]
    )
    engine._tick()  # noqa: SLF001
    before = engine.get_pose()

    def drive():
        # The robot covers 0.4 m and turns 20 degrees while the map is being
        # redrawn. (More than half a meter between two ticks a few
        # milliseconds apart, as they are in this test, is refused as an
        # odometry jump.)
        odom[0] = conv.Pose2D(0.4, 0.0, math.radians(20.0))

    _during_first_render(monkeypatch, engine, drive)
    assert engine.optimize_pose_graph()["status"] == "rebuilt"
    engine._tick()  # noqa: SLF001

    after = engine.get_pose()
    assert math.hypot(after.x - before.x, after.y - before.y) == pytest.approx(
        0.4, abs=0.02
    )
    assert conv.normalize_angle(after.theta - before.theta) == pytest.approx(
        math.radians(20.0), abs=math.radians(1.0)
    )


def test_late_scans_at_two_heights_land_in_their_own_height_band(tmp_path, monkeypatch):
    from .test_paint_fit_heights import (
        _cluttered,
        _room_ranges,
        _scan,
        _two_height_engine,
    )

    walls = _room_ranges(conv.Pose2D(0.0, 0.0, 0.0))
    low = _cluttered(walls, share=0.34)
    engine = _two_height_engine(tmp_path, low, walls)
    merged = _scan(np.minimum(low, walls))

    def insert(k: int) -> None:
        pose = conv.Pose2D(0.6 * k - 1.2, 0.0, 0.0)
        with engine._lock:  # noqa: SLF001
            engine._pose = pose  # noqa: SLF001
        engine._paint_mapping_scan(merged, pose, 0.2 * k)  # noqa: SLF001

    for k in range(3):
        insert(k)
    held = _during_first_render(
        monkeypatch, engine, lambda: [insert(k) for k in range(3, 9)]
    )

    assert engine.optimize_pose_graph()["status"] == "rebuilt"

    assert held and not any(held)
    with engine._lock:  # noqa: SLF001
        grid = engine._grid  # noqa: SLF001
        bands = dict(engine._height_bands)  # noqa: SLF001
        fresh_bands: dict = {}
        fresh = pg.render_grid(
            engine._graph.snapshot(),  # noqa: SLF001
            resolution=grid.resolution,
            base=engine._graph_base,  # noqa: SLF001
            bands_out=fresh_bands,
        )
    assert len(bands) >= 2 and set(bands) == set(fresh_bands)
    assert np.array_equal(grid.log_odds, fresh.log_odds)
    for band, band_grid in bands.items():
        assert np.array_equal(band_grid.log_odds, fresh_bands[band].log_odds)


def test_a_stalled_update_loop_says_what_the_module_was_busy_with(tmp_path):
    engine = _engine(tmp_path)
    said = []
    engine._log = said.append  # noqa: SLF001
    now = 1000.0
    with engine._doing("map rebuild"):  # noqa: SLF001
        engine._note_stall(3.2, now)  # noqa: SLF001
        engine._note_stall(2.0, now + 1.0)  # noqa: SLF001  (too soon to say it again)
    engine._note_stall(1.5, now + 10.0)  # noqa: SLF001
    assert len(said) == 2
    assert "stalled 3.2 s" in said[0] and "map rebuild" in said[0]
    assert "nothing it tracks" in said[1]

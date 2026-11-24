"""A saved map must not be replaced by an empty one.

On tracer2a an explore run ended while a save-time map repair was running.
Its finish step asked SLAM to optimize the pose graph; that request passed
its "are we mapping" check and then waited seven minutes behind the repair.
Meanwhile the map was saved and SLAM was switched to localizing, which loads
the map and empties the keyframe graph. When the repair let go, the waiting
optimize rendered that empty graph, an empty 20 m grid, and made it the live
map. Explore's finish then saved it over map.pgm.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from src.config import MODE_LOCALIZING, SlamConfig
from src.geom import conversions as conv
from src.nav.stores.maps import MapStore
from src.slam_builtin import occupancy as occ
from src.slam_builtin import persistence
from src.slam_builtin.engine import BuiltinSlamEngine

from .test_loop_closure import _raycast, _Sensors, _truth_world

TRUTH = _truth_world()


def _engine(tmp_path) -> BuiltinSlamEngine:
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
    for k in range(8):
        pose = conv.Pose2D(0.6 * k, 0.0, 0.0)
        with engine._lock:  # noqa: SLF001
            engine._pose = pose  # noqa: SLF001
        engine._paint_mapping_scan(_raycast(TRUTH, pose), pose, 0.2 * k)  # noqa: SLF001
    return engine


def _walls(grid) -> int:
    return int((occ.to_occupancy_int16(grid) >= 50).sum())


def test_an_optimize_that_waited_through_a_mode_switch_does_not_empty_the_map(
    tmp_path,
):
    engine = _engine(tmp_path)
    map_dir = tmp_path / "m"
    engine.save_map(map_dir)
    walls = _walls(persistence.load_log_odds(map_dir))
    assert walls > 200

    # A map repair holds the graph for minutes; an optimize request queues
    # up behind it.
    engine._graph_job.acquire()  # noqa: SLF001
    result = {}
    waiting = threading.Thread(
        target=lambda: result.update(engine.optimize_pose_graph())
    )
    waiting.start()
    time.sleep(0.2)
    # Meanwhile SLAM is switched to localizing: the map is reloaded and the
    # keyframe graph emptied.
    engine.configure_mode(MODE_LOCALIZING, map_dir)
    assert len(engine._graph) == 0  # noqa: SLF001
    engine._graph_job.release()  # noqa: SLF001
    waiting.join(timeout=10.0)
    assert not waiting.is_alive()

    assert result["status"] == "skipped"
    assert _walls(engine._grid) == walls  # noqa: SLF001
    # And a save after it (explore's finish did one) leaves the map as it was.
    engine.save_map(map_dir)
    assert _walls(persistence.load_log_odds(map_dir)) == walls


def test_a_graph_with_no_keyframes_is_never_rendered_over_the_map(tmp_path):
    engine = _engine(tmp_path)
    map_dir = tmp_path / "m"
    engine.save_map(map_dir)
    engine.configure_mode(MODE_LOCALIZING, map_dir)
    grid = engine._grid  # noqa: SLF001
    with engine._graph_job:  # noqa: SLF001
        out = engine._optimize_and_render(1)  # noqa: SLF001
    assert out["status"] == "skipped"
    assert engine._grid is grid  # noqa: SLF001


def test_saving_an_empty_grid_over_a_saved_map_is_refused(tmp_path):
    engine = _engine(tmp_path)
    map_dir = tmp_path / "m"
    engine.save_map(map_dir)
    saved = (map_dir / "map.pgm").read_bytes()
    loaded = persistence.load_log_odds(map_dir)
    assert loaded is not None and _walls(loaded) > 200

    with engine._lock:  # noqa: SLF001
        engine._grid = occ.empty_grid(resolution=0.05)  # noqa: SLF001
    out = engine.save_map(map_dir)

    assert out["status"] == "refused_empty"
    assert (map_dir / "map.pgm").read_bytes() == saved
    # A new map with nothing on it yet can still be saved.
    fresh = tmp_path / "fresh"
    assert engine.save_map(fresh).get("status") != "refused_empty"
    assert (fresh / "map.pgm").is_file()


def test_an_optimize_does_not_wait_forever_behind_a_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(BuiltinSlamEngine, "_OPTIMIZE_WAIT_S", 0.2)
    engine = _engine(tmp_path)
    engine._graph_job.acquire()  # noqa: SLF001
    try:
        started = time.monotonic()
        out = engine.optimize_pose_graph()
    finally:
        engine._graph_job.release()  # noqa: SLF001
    assert out["status"] == "busy" and time.monotonic() - started < 2.0
    assert math.isfinite(float(_walls(engine._grid)))  # noqa: SLF001
    assert np.any(engine._grid.log_odds != 0.0)  # noqa: SLF001

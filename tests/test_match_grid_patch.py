"""The scan-match grid is patched where scans painted, and equals a full rebuild."""

from __future__ import annotations

import numpy as np

from src.nav.localize.global_localize import OccupancyMap
from src.slam_builtin import occupancy as occ
from src.slam_builtin import scan_match
from src.slam_builtin.engine import BuiltinSlamEngine, _MatchBase
from src.slam_builtin.types import LogOddsGrid


def _full(grid: LogOddsGrid, has_clears: bool = False):
    int16 = occ.to_occupancy_int16(grid)
    if has_clears:
        int16[grid.log_odds == np.float32(occ.CLEARED_LOG_ODDS)] = -1
    lookup = scan_match.occupied_lookup(
        OccupancyMap(grid=int16, resolution=grid.resolution, origin_x=0.0, origin_y=0.0)
    )
    return int16, lookup, int(np.count_nonzero(int16 >= 0))


def _grid(rng, h=120, w=160) -> LogOddsGrid:
    lo = np.zeros((h, w), dtype=np.float32)
    pick = rng.random((h, w))
    lo[pick < 0.3] = -2.0  # free
    lo[pick > 0.9] = 3.0  # walls
    lo[(pick > 0.5) & (pick < 0.55)] = 0.3  # in between
    return LogOddsGrid(log_odds=lo, resolution=0.05, origin_x=-4.0, origin_y=-3.0)


def test_a_patch_over_the_painted_boxes_equals_a_full_build():
    rng = np.random.default_rng(3)
    for trial in range(20):
        grid = _grid(rng)
        has_clears = trial % 2 == 1
        if has_clears:
            grid.log_odds[rng.random(grid.log_odds.shape) < 0.02] = occ.CLEARED_LOG_ODDS
        int16, lookup, known = _full(grid, has_clears)
        base = _MatchBase(("g",), int16, lookup, known, 0)
        after = occ.copy_grid(grid)
        boxes = []
        # Paints between two matches land near each other, as a robot's do.
        ar, ac = int(rng.integers(0, 90)), int(rng.integers(0, 130))
        for _ in range(int(rng.integers(1, 4))):
            r0, c0 = ar + int(rng.integers(0, 20)), ac + int(rng.integers(0, 20))
            r1, c1 = r0 + int(rng.integers(1, 10)), c0 + int(rng.integers(1, 10))
            after.log_odds[r0:r1, c0:c1] = rng.choice(
                [-2.0, 0.0, 0.3, 3.0], size=(r1 - r0, c1 - c0)
            ).astype(np.float32)
            boxes.append(
                (
                    after.origin_x + c0 * 0.05,
                    after.origin_y + r0 * 0.05,
                    after.origin_x + c1 * 0.05,
                    after.origin_y + r1 * 0.05,
                )
            )
        patched = BuiltinSlamEngine._patched_match_grid(base, after, boxes, has_clears)
        want = _full(after, has_clears)
        assert patched is not None
        assert np.array_equal(patched[0], want[0])
        assert np.array_equal(patched[1], want[1])
        assert patched[2] == want[2]
        # The base it was patched from is left as it was.
        assert np.array_equal(base.int16, int16)


def test_mapping_patches_the_match_grid_to_exactly_what_a_full_build_gives(monkeypatch):
    from tests.test_vision_eval import _RoomRun
    from tools.vision_eval.replay import replay_mapping

    real = BuiltinSlamEngine._patched_match_grid
    checked = {"patches": 0}

    def checking(base, grid, dirty, has_clears):
        out = real(base, grid, dirty, has_clears)
        if out is not None and dirty:
            want = _full(grid, has_clears)
            assert np.array_equal(out[0], want[0])
            assert np.array_equal(out[1], want[1])
            assert out[2] == want[2]
            checked["patches"] += 1
        return out

    monkeypatch.setattr(
        BuiltinSlamEngine, "_patched_match_grid", staticmethod(checking)
    )
    replay_mapping(_RoomRun(seconds=6.0), log=lambda *_: None)
    assert checked["patches"] > 20

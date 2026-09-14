from __future__ import annotations

import numpy as np

from src.slam_builtin import occupancy as occ


def test_clear_disk_frees_occupied_cells():
    grid = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    # Paint a small occupied blob around the origin.
    for x in np.linspace(-0.15, 0.15, 7):
        for y in np.linspace(-0.15, 0.15, 7):
            r, c = grid.world_to_cell(float(x), float(y))
            grid.log_odds[r, c] = occ.L_MAX

    before = occ.to_occupancy_int16(grid)
    assert np.any(before == 100)

    cleared = occ.clear_disk(grid, 0.0, 0.0, 0.25)
    assert cleared > 0
    after = occ.to_occupancy_int16(grid)
    # Center neighborhood should be free, not occupied.
    r, c = grid.world_to_cell(0.0, 0.0)
    assert after[r, c] == 0
    assert not np.any(after[r - 2 : r + 3, c - 2 : c + 3] == 100)


def test_clear_disk_rejects_non_positive_radius():
    grid = occ.empty_grid(resolution=0.05, size_m=2.0)
    try:
        occ.clear_disk(grid, 0.0, 0.0, 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass

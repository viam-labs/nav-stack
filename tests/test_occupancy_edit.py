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


def test_mark_disk_occupies_free_cells():
    grid = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    marked = occ.mark_disk(grid, 0.0, 0.0, 0.25)
    assert marked > 0
    after = occ.to_occupancy_int16(grid)
    r, c = grid.world_to_cell(0.0, 0.0)
    assert after[r, c] == 100
    assert np.all(after[r - 2 : r + 3, c - 2 : c + 3] == 100)


def test_clear_disk_rejects_non_positive_radius():
    grid = occ.empty_grid(resolution=0.05, size_m=2.0)
    try:
        occ.clear_disk(grid, 0.0, 0.0, 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_clear_rect_frees_box_and_leaves_outside():
    grid = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    # Paint by cell so float boundary rounding cannot miss the outside probe.
    for x in np.arange(-1.0, 1.05, 0.05):
        for y in np.arange(-1.0, 1.05, 0.05):
            r, c = grid.world_to_cell(float(x), float(y))
            grid.log_odds[r, c] = occ.L_MAX

    cleared = occ.clear_rect(grid, -0.4, -0.2, 0.4, 0.2)
    assert cleared > 0
    after = occ.to_occupancy_int16(grid)
    r0, c0 = grid.world_to_cell(0.0, 0.0)
    assert after[r0, c0] == 0
    r_out, c_out = grid.world_to_cell(0.85, 0.85)
    assert after[r_out, c_out] == 100


def test_clear_rect_rejects_empty_box():
    grid = occ.empty_grid(resolution=0.05, size_m=2.0)
    try:
        occ.clear_rect(grid, 0.0, 0.0, 0.0, 1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_insert_scan_extend_only_leaves_observed_cells_alone():
    grid = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    # Seed a known free cell and a known occupied cell along +x.
    r_free, c_free = grid.world_to_cell(0.5, 0.0)
    r_occ, c_occ = grid.world_to_cell(1.0, 0.0)
    grid.log_odds[r_free, c_free] = occ.L_MIN
    grid.log_odds[r_occ, c_occ] = occ.L_MAX
    free_before = float(grid.log_odds[r_free, c_free])
    occ_before = float(grid.log_odds[r_occ, c_occ])

    # Ray through free toward occupied, plus into unknown beyond.
    ranges = np.array([1.5], dtype=float)
    grid = occ.insert_scan(
        grid,
        0.0,
        0.0,
        0.0,
        ranges,
        0.0,
        0.0,
        range_min=0.05,
        range_max=5.0,
        max_beams=0,
        extend_only=True,
    )
    r_free, c_free = grid.world_to_cell(0.5, 0.0)
    r_occ, c_occ = grid.world_to_cell(1.0, 0.0)
    assert float(grid.log_odds[r_free, c_free]) == free_before
    assert float(grid.log_odds[r_occ, c_occ]) == occ_before
    r_new, c_new = grid.world_to_cell(1.5, 0.0)
    assert float(grid.log_odds[r_new, c_new]) > 0.0


_Y = 0.025  # cell center: boundary points can floor to a neighbour after growth
_OCC_LO = float(np.log(occ.OCC_THRESH / (1 - occ.OCC_THRESH)))


def _ray(grid, x, rng, **kw):
    return occ.insert_scan(
        grid,
        x,
        _Y,
        0.0,
        np.array([rng], dtype=float),
        0.0,
        0.0,
        range_min=0.05,
        range_max=10.0,
        max_beams=0,
        extend_only=True,
        **kw,
    )


def test_extend_only_with_protect_keeps_updating_cells_new_this_session():
    grid = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    r, c = grid.world_to_cell(0.525, _Y)
    grid.log_odds[r, c] = occ.L_MIN
    baseline = occ.copy_grid(grid)

    # A long glance marks x=1.225 free first; a later close scan hits a wall there.
    grid = _ray(grid, 0.0, 1.8, protect=baseline)
    grid = _ray(grid, 0.0, 1.225, protect=baseline)
    grid = _ray(grid, 0.0, 1.225, protect=baseline)
    r_wall, c_wall = grid.world_to_cell(1.225, _Y)
    assert float(grid.log_odds[r_wall, c_wall]) > _OCC_LO
    r, c = grid.world_to_cell(0.525, _Y)
    assert float(grid.log_odds[r, c]) == occ.L_MIN

    # Without protect the first glance freezes the cell as free.
    old = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    for rng in (1.8, 1.225, 1.225):
        old = _ray(old, 0.0, rng)
    assert float(old.log_odds[r_wall, c_wall]) < 0.0


def test_extend_only_protect_tracks_baseline_after_grid_grows():
    grid = occ.empty_grid(resolution=0.05, size_m=4.0, origin_x=-2.0, origin_y=-2.0)
    r, c = grid.world_to_cell(1.025, _Y)
    grid.log_odds[r, c] = occ.L_MIN
    baseline = occ.copy_grid(grid)

    # Ray reaches past the baseline so the grid expands and its origin shifts.
    grid = _ray(grid, 0.0, 6.0, protect=baseline)
    grid = _ray(grid, 0.0, 1.025, protect=baseline)
    assert (grid.origin_x, grid.origin_y) != (baseline.origin_x, baseline.origin_y)
    r, c = grid.world_to_cell(1.025, _Y)
    assert float(grid.log_odds[r, c]) == occ.L_MIN
    grid = _ray(grid, 0.0, 4.025, protect=baseline)
    grid = _ray(grid, 0.0, 4.025, protect=baseline)
    r, c = grid.world_to_cell(4.025, _Y)
    assert float(grid.log_odds[r, c]) > _OCC_LO

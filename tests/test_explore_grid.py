import base64

import numpy as np

from src.explore.grid import OccGrid, frontier_goals, under_seen_goals, visible_new_cells


def room(rows=60, cols=80, res=0.05):
    return OccGrid(
        cells=np.full((rows, cols), -1, dtype=np.int8), resolution=res, x_min=0.0, y_min=0.0
    )


def enclose(g):
    g.cells[20:40, 20:40] = 0
    g.cells[19, 19:41] = 100
    g.cells[40, 19:41] = 100
    g.cells[19:41, 19] = 100
    g.cells[19:41, 40] = 100


def test_from_get_grid_roundtrip():
    cells = np.zeros(12, dtype=np.int8)
    cells[5] = 100
    cells[0] = -1
    resp = {
        "rows": 3,
        "cols": 4,
        "cellSize": 0.05,
        "xMin": -1.0,
        "yMin": 2.0,
        "data": base64.b64encode(cells.tobytes()).decode(),
    }
    g = OccGrid.from_get_grid(resp)
    assert g.cells.shape == (3, 4) and g.cells[1, 1] == 100 and g.cells[0, 0] == -1
    assert g.to_cell(-1.0 + 0.075, 2.0 + 0.025) == (0, 1)
    assert g.to_world(0, 1) == (-0.925, 2.025)
    assert OccGrid.from_get_grid({"available": False}) is None


def test_frontier_goals_sit_on_known_free_cells():
    g = room()
    g.cells[20:40, 20:40] = 0
    goals = frontier_goals(g, robot_radius_m=0.25)
    assert goals
    for c in goals:
        r, col = g.to_cell(c.x, c.y)
        assert g.cells[r, col] == 0


def test_frontier_goals_ignore_walls():
    g = room()
    enclose(g)
    assert frontier_goals(g, robot_radius_m=0.25) == []


def test_frontier_goals_drop_small_clusters():
    g = room()
    enclose(g)
    g.cells[19, 30] = 0
    assert frontier_goals(g, robot_radius_m=0.25, min_cluster_cells=6) == []
    assert len(frontier_goals(g, robot_radius_m=0.25, min_cluster_cells=1)) == 1


def test_visible_new_cells_counts_unknown_only():
    g = room()
    g.cells[20:40, 20:40] = 0
    assert visible_new_cells(g, 1.5, 1.5, max_range_m=8.0) > 0
    g.cells[:, :] = 0
    assert visible_new_cells(g, 1.5, 1.5, max_range_m=8.0) == 0


def test_visible_new_cells_blocked_by_walls():
    g = room()
    enclose(g)
    assert visible_new_cells(g, 1.5, 1.5, max_range_m=8.0) == 0


def test_under_seen_goals_far_from_visits():
    g = room()
    g.cells[:, :] = 0
    goals = under_seen_goals(g, visited=[(0.5, 0.5)], well_seen_m=1.0, spacing_m=1.0)
    assert goals and all(np.hypot(c.x - 0.5, c.y - 0.5) > 1.0 for c in goals)
    assert under_seen_goals(g, visited=[(2.0, 1.5)], well_seen_m=10.0) == []

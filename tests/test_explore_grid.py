import base64

import numpy as np

from src.explore.grid import (
    OccGrid,
    frontier_mask,
    scan_frontiers,
    traversable_mask,
    wavefront,
)

RES = 0.05


def grid(rows=80, cols=120):
    return OccGrid(
        cells=np.full((rows, cols), -1, dtype=np.int8),
        resolution=RES,
        x_min=0.0,
        y_min=0.0,
    )


def room(g, r0=10, r1=50, c0=10, c1=60):
    """Known free room with walls on every side."""
    g.cells[r0:r1, c0:c1] = 0
    g.cells[r0 - 1, c0 - 1 : c1 + 1] = 100
    g.cells[r1, c0 - 1 : c1 + 1] = 100
    g.cells[r0 - 1 : r1 + 1, c0 - 1] = 100
    g.cells[r0 - 1 : r1 + 1, c1] = 100


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


def test_from_map_dict_reads_slam_get_map():
    m = {
        "grid": np.array([[-1, 0], [37, 100]], dtype=np.int16),
        "resolution": 0.05,
        "origin_x": 1.0,
        "origin_y": -2.0,
        "generation": 4,
    }
    g = OccGrid.from_map_dict(m)
    assert g.cells.dtype == np.int8 and g.x_min == 1.0 and g.y_min == -2.0
    # Mid-probability cells below the occupied threshold count as free.
    assert g.free_mask().tolist() == [[False, True], [True, False]]
    assert OccGrid.from_map_dict({"grid": None}) is None


def test_cropped_keeps_world_coordinates():
    g = grid()
    room(g)
    c = g.cropped((1.0, 1.0), pad_cells=2)
    assert c.cells.shape == (41 + 1 + 4, 51 + 1 + 4)
    assert c.to_world(*c.to_cell(1.0, 1.0)) == g.to_world(*g.to_cell(1.0, 1.0))


def test_wavefront_tracks_path_length_around_walls():
    trav = np.ones((60, 60), dtype=bool)
    d = wavefront(trav, (30, 10))
    assert d[30, 50] == 40  # straight
    assert abs(d[0, 40] * 1.0 - np.hypot(30, 30)) / np.hypot(30, 30) < 0.08  # diagonal
    trav[10:60, 30] = False  # wall with a gap at the top
    d = wavefront(trav, (50, 20))
    assert d[50, 40] > 70  # must go round through the gap
    trav[:, 30] = False
    assert wavefront(trav, (50, 20))[50, 40] == -1


def test_traversable_keeps_clear_of_walls_but_not_unknown():
    g = grid()
    room(g)
    t = traversable_mask(g, robot_radius_m=0.2)
    assert not t[12, 30] and t[20, 30]  # 3 cells from a wall vs 11
    g.cells[30:50, 20:40] = -1
    t = traversable_mask(g, robot_radius_m=0.2)
    assert t[29, 30]  # right next to unknown is fine


def test_frontier_fill_ignores_small_holes():
    g = grid()
    room(g)
    g.cells[30, 30] = -1  # single unseen cell inside the room
    assert frontier_mask(g)[29, 30]
    assert not frontier_mask(g, fill_cells=2).any()


def test_enclosed_room_has_no_frontiers():
    g = grid()
    room(g)
    assert scan_frontiers(g, 1.0, 1.0, robot_radius_m=0.2).frontiers == []


def test_doorway_frontier_goal_is_reachable_and_at_the_opening():
    g = grid()
    room(g)
    g.cells[25:35, 60] = 0  # 0.5 m doorway in the right wall onto unknown space
    scan = scan_frontiers(g, 1.0, 1.0, robot_radius_m=0.1)
    assert len(scan.frontiers) == 1
    f = scan.frontiers[0]
    assert 2.6 < f.goal_x <= 3.05 and 1.2 < f.goal_y < 1.8
    assert abs(f.path_m - 2.0) < 0.3 and f.size_m >= 0.5
    assert scan.frontier_near(f.x, f.y, 0.1)
    assert abs(scan.path_m(f.goal_x, f.goal_y) - f.path_m) < 0.06


def test_unreachable_and_short_frontiers_are_dropped():
    g = grid()
    room(g)
    g.cells[60:70, 30:90] = (
        0  # free strip seen through nothing: walled off from the robot
    )
    assert scan_frontiers(g, 1.0, 1.0, robot_radius_m=0.1).frontiers == []
    g = grid()
    room(g)
    g.cells[30:33, 60] = 0  # 0.15 m gap
    assert (
        scan_frontiers(g, 1.0, 1.0, robot_radius_m=0.1, min_frontier_m=0.5).frontiers
        == []
    )


def test_frontier_at_the_robot_is_ignored():
    g = grid()
    room(g)
    g.cells[19:22, 19:22] = -1  # unseen patch under the robot
    scan = scan_frontiers(g, 1.0, 1.0, robot_radius_m=0.2, fill_m=0.0)
    assert scan.frontiers == []


def test_robot_off_traversable_space_starts_from_nearest_cell():
    g = grid()
    room(g)
    g.cells[25:35, 60] = 0
    scan = scan_frontiers(g, 0.55, 1.5, robot_radius_m=0.2)  # 1 cell from the left wall
    assert scan.start_found and scan.frontiers
    assert not scan_frontiers(g, 5.5, 3.5, robot_radius_m=0.2).start_found

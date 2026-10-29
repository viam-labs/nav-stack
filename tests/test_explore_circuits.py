import math

import numpy as np

from src.explore.circuits import (
    circuit_closed,
    facing_wall_theta,
    inner_circuit_goals,
    next_wall_hop,
    path_length,
)
from src.explore.clients import Pose
from src.explore.grid import OccGrid


def walled_room(w_m=8.0, h_m=6.0, res=0.05):
    rows, cols = int(h_m / res), int(w_m / res)
    g = np.zeros((rows, cols), dtype=np.int8)
    g[0, :] = 100
    g[-1, :] = 100
    g[:, 0] = 100
    g[:, -1] = 100
    return OccGrid(cells=g, resolution=res, x_min=0.0, y_min=0.0)


def test_wall_hop_keeps_wall_on_left_and_moves_along_it():
    g = walled_room()
    hop = next_wall_hop(g, Pose(4.0, 0.8, math.pi))
    assert hop is not None
    assert hop.x < 3.0 and abs(hop.y - 0.8) < 0.3


def test_wall_hop_turns_corner():
    g = walled_room()
    hop = next_wall_hop(g, Pose(1.0, 0.8, math.pi))
    assert hop is not None and hop.x >= 0.6 and hop.y > 1.3


def test_wall_hop_none_without_wall_in_range():
    g = walled_room(w_m=20.0, h_m=20.0)
    assert next_wall_hop(g, Pose(10.0, 10.0, 0.0), search_m=3.0) is None


def test_facing_wall_theta_points_at_nearest_wall():
    g = walled_room()
    th = facing_wall_theta(g, Pose(4.0, 0.8, 0.0))
    assert abs(th - (-math.pi / 2)) < 0.2


def test_circuit_closed():
    s = Pose(1.0, 1.0, 0.0)
    assert not circuit_closed(5.0, Pose(1.2, 1.0, 0.0), s, far_m=5.0)
    assert not circuit_closed(20.0, Pose(5.0, 1.0, 0.0), s, far_m=5.0)
    assert circuit_closed(20.0, Pose(1.2, 1.0, 0.0), s, far_m=5.0)


def test_inner_circuit_goals_are_clear_of_walls_and_long_enough():
    g = walled_room()
    goals = inner_circuit_goals(g, Pose(1.5, 1.5, 0.0), n=3, clearance_m=1.0)
    assert len(goals) == 4
    assert (goals[-1].x, goals[-1].y, goals[-1].kind) == (1.5, 1.5, "circuit")
    for c in goals[:-1]:
        assert 1.0 <= c.x <= 7.0 and 1.0 <= c.y <= 5.0
    assert path_length([(1.5, 1.5)] + [(c.x, c.y) for c in goals]) >= 12.0


def test_wall_hops_keep_direction_around_the_room():
    g = walled_room()
    pose = Pose(1.0, 0.8, math.pi)
    prev = None
    hops = []
    for _ in range(12):
        hop = next_wall_hop(g, pose, prev_travel=prev)
        assert hop is not None
        travel = (hop.x - pose.x, hop.y - pose.y)
        if prev is not None:
            assert travel[0] * prev[0] + travel[1] * prev[1] >= -0.1
        hops.append((hop.x, hop.y))
        prev = travel
        pose = Pose(hop.x, hop.y, math.atan2(travel[1], travel[0]))
    xs = [h[0] for h in hops]
    ys = [h[1] for h in hops]
    assert max(xs) - min(xs) > 4.0 and max(ys) - min(ys) > 3.0


def test_circuit_closed_requires_excursion():
    s = Pose(1.0, 1.0, 0.0)
    assert not circuit_closed(20.0, Pose(1.2, 1.0, 0.0), s, far_m=1.0)
    assert circuit_closed(20.0, Pose(1.2, 1.0, 0.0), s, far_m=4.0)

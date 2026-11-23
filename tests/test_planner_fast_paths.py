"""The list-based planner helpers must match the numpy reference ones exactly."""

import numpy as np

from src.nav.planning import planner

COSTS = [0, 0, 0, 10, 25, 30, 31, 60, 120, 200, 252, 253, 254, 255]


def _maps():
    for seed in range(12):
        rng = np.random.default_rng(seed)
        c = rng.choice(COSTS, size=(40, 50)).astype(np.uint8)
        c[rng.random((40, 50)) < 0.6] = 0
        yield rng, c


def test_line_of_sight_and_chord_cost_match_reference():
    for rng, c in _maps():
        rows = planner._cost_rows(c)
        for _ in range(60):
            a = (int(rng.integers(40)), int(rng.integers(50)))
            b = (int(rng.integers(40)), int(rng.integers(50)))
            assert planner._line_of_sight_rows(
                rows, 40, 50, a, b
            ) == planner.line_of_sight(c, a, b)
            assert planner._segment_traversal_cost_rows(
                rows, a, b
            ) == planner._segment_traversal_cost(c, a, b)


def test_cost_rows_clipping_keeps_predicates():
    c = np.array([[-5, 0, 30, 31, 254, 255, 400]], dtype=np.int32)
    rows = planner._cost_rows(c)
    for col in range(c.shape[1]):
        raw = int(c[0, col])
        v = rows[0][col]
        assert planner._MULT[v] == planner._cost_multiplier(raw)
        assert planner._TRAV[v] == planner.is_traversable(raw)

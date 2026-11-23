import math

from src.explore.grid import Frontier
from src.explore.scoring import Blacklist, cost


def f(x, y, size_m=1.0, path_m=None):
    return Frontier(x, y, x, y, size_m, math.hypot(x, y) if path_m is None else path_m)


def test_blacklist_radius_and_ttl():
    b = Blacklist(radius_m=1.0, ttl_s=600.0)
    b.add(5.0, 5.0, now=100.0)
    assert b.blocked(5.5, 5.0, now=200.0)
    assert not b.blocked(6.5, 5.0, now=200.0)
    assert not b.blocked(5.0, 5.0, now=701.0)


def test_cost_prefers_short_paths():
    assert cost(f(2.0, 0.0), 0, 0, 0) < cost(f(5.0, 0.0), 0, 0, 0)


def test_cost_prefers_straight_ahead_over_behind():
    ahead, behind = f(3.0, 0.0), f(-3.0, 0.0)
    assert cost(ahead, 0, 0, 0) + 1.5 < cost(behind, 0, 0, 0)


def test_frontier_size_helps_but_is_capped():
    small, big = f(3.0, 0.0, size_m=0.5), f(3.0, 0.0, size_m=4.0)
    assert cost(big, 0, 0, 0) < cost(small, 0, 0, 0)
    huge_far = f(30.0, 0.0, size_m=500.0)
    assert cost(f(3.0, 0.0), 0, 0, 0) < cost(huge_far, 0, 0, 0)

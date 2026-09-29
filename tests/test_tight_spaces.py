"""Closed-loop tight-space scenarios for the real 0.72 x 0.59 m tracer footprint.

The real ``NavSupervisor`` drives a ground-truth rectangular robot (see
``tight_space_sim``). Invariants every scenario checks:
- the rectangular body never touches anything (map walls or unmapped bins);
- routes the body fits through succeed in bounded time;
- routes it does not fit through fail cleanly (still no contact);
- the committed global path does not flip short<->long.

Each scenario runs under several conditions: ideal sensing, noisy sensing
(1 cm lidar range noise, ~2 cm / 1 deg wandering localisation error), and the
default ``clearance_m`` (0.2) as well as the live tracer value (0.03).
"""
from __future__ import annotations

import pytest

from src.geom.conversions import Pose2D

from . import tight_space_sim as sim


def _room():
    return sim.blank(10.0, 6.0)


def _door_wall(g, *, x: float, y0: float, y1: float, thick: float = 0.1):
    sim.box(g, x, 0.0, x + thick, y0)
    sim.box(g, x, y1, x + thick, 6.0)


def bin_beside_route():
    """Live rc24: bin at the shoulder, not in the map, lidar sees it."""
    return _room(), Pose2D(1.0, 3.0, 0.0), Pose2D(8.5, 3.0, 0.0), [(4.3, 3.2, 4.7, 3.6)], True


def bin_on_route():
    return _room(), Pose2D(1.0, 3.0, 0.0), Pose2D(8.5, 3.0, 0.0), [(4.3, 2.8, 4.7, 3.2)], True


def door_straight(door_m):
    g = _room()
    _door_wall(g, x=5.0, y0=3.0 - door_m / 2, y1=3.0 + door_m / 2)
    return g, Pose2D(2.0, 3.0, 0.0), Pose2D(8.0, 3.0, 0.0), [], True


def door_oblique(door_m):
    g = _room()
    _door_wall(g, x=5.0, y0=3.0 - door_m / 2, y1=3.0 + door_m / 2)
    return g, Pose2D(2.0, 1.0, 0.0), Pose2D(8.0, 5.0, 0.0), [], True


def door_then_bin():
    """Tight door, then a bin half-blocking the far side (the live pinch)."""
    g = _room()
    _door_wall(g, x=5.0, y0=2.5, y1=3.5)
    return g, Pose2D(2.0, 3.0, 0.0), Pose2D(8.5, 3.0, 0.0), [(6.0, 3.05, 6.4, 3.45)], True


def corridor_bend():
    """0.9 m wide L corridor."""
    g = sim.blank(10.0, 8.0)
    sim.box(g, 0.0, 0.0, 10.0, 8.0)
    sim.box(g, 0.5, 0.5, 3.0, 2.5, v=0)
    sim.box(g, 3.0, 1.05, 6.5, 1.95, v=0)
    sim.box(g, 5.6, 1.05, 6.5, 6.0, v=0)
    sim.box(g, 4.5, 6.0, 9.5, 7.5, v=0)
    return g, Pose2D(1.5, 1.5, 0.0), Pose2D(8.0, 6.8, 0.0), [], True


def gap_too_narrow():
    g = _room()
    _door_wall(g, x=5.0, y0=2.72, y1=3.28)  # 0.56 m < 0.59 m body
    return g, Pose2D(2.0, 3.0, 0.0), Pose2D(8.0, 3.0, 0.0), [], False


def corridor_dead_end():
    """Unmapped bin fully plugs the only corridor: fail, never push into it."""
    g = sim.blank(10.0, 4.0)
    sim.box(g, 0.0, 0.0, 10.0, 4.0)
    sim.box(g, 0.5, 0.5, 3.0, 3.5, v=0)
    sim.box(g, 3.0, 1.55, 7.0, 2.45, v=0)
    sim.box(g, 7.0, 0.5, 9.5, 3.5, v=0)
    return g, Pose2D(1.5, 2.0, 0.0), Pose2D(8.5, 2.0, 0.0), [(4.8, 1.55, 5.2, 2.45)], False


def short_tight_vs_long_wide():
    """Short 0.85 m door vs a wide opening far away: commit, never thrash."""
    g = sim.blank(12.0, 8.0)
    sim.box(g, 6.0, 0.0, 6.1, 8.0)
    sim.box(g, 6.0, 3.6, 6.1, 4.45, v=0)
    sim.box(g, 6.0, 0.05, 6.1, 1.2, v=0)
    return g, Pose2D(3.0, 4.0, 0.0), Pose2D(9.0, 4.0, 0.0), [], True


SCENARIOS = {
    "bin_beside_route": bin_beside_route,
    "bin_on_route": bin_on_route,
    "door_1.00_straight": lambda: door_straight(1.0),
    "door_0.85_straight": lambda: door_straight(0.85),
    "door_1.00_oblique": lambda: door_oblique(1.0),
    "door_0.85_oblique": lambda: door_oblique(0.85),
    "door_then_bin": door_then_bin,
    "corridor_bend_0.9": corridor_bend,
    "gap_too_narrow": gap_too_narrow,
    "corridor_dead_end": corridor_dead_end,
    "short_tight_vs_long_wide": short_tight_vs_long_wide,
}

# Narrowest opening the route must pass (m).
NARROWEST = {
    "bin_beside_route": 2.6,
    "bin_on_route": 2.8,
    "door_1.00_straight": 1.0,
    "door_0.85_straight": 0.85,
    "door_1.00_oblique": 1.0,
    "door_0.85_oblique": 0.85,
    "door_then_bin": 1.0,
    "corridor_bend_0.9": 0.9,
    "gap_too_narrow": 0.56,
    "corridor_dead_end": 0.0,
    "short_tight_vs_long_wide": 1.15,
}

CONDITIONS = {
    "ideal": dict(noise=False, seed=0, clearance_m=0.03),
    "noisy1": dict(noise=True, seed=1, clearance_m=0.03),
    "noisy2": dict(noise=True, seed=2, clearance_m=0.03),
    "clear0.2": dict(noise=False, seed=0, clearance_m=0.2),
}


def _detail(r: sim.RunResult) -> str:
    return (
        f"state={r.state} err={r.error!r} t={r.sim_s:.1f}s contacts={r.contacts} "
        f"first={r.first_contact} flips={r.length_flips()} "
        f"lens={[round(x, 1) for x in r.path_lengths][:12]} "
        f"states={r.obstacle_states[:16]} rev={r.reversing_s:.1f}s "
        f"end=({r.final_pose.x:.2f},{r.final_pose.y:.2f})"
    )


@pytest.mark.parametrize("cond", list(CONDITIONS))
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_tight_space(monkeypatch, name, cond):
    grid, start, goal, extras, fits = SCENARIOS[name]()
    c = CONDITIONS[cond]
    # clearance_m is a hard planner minimum: gaps under body + 2*clearance
    # (plus a cell of quantisation) are correctly refused.
    if fits and NARROWEST[name] < sim.WIDTH + 2.0 * c["clearance_m"] + 2 * sim.RES:
        fits = False
    r = sim.run(
        grid,
        start,
        goal,
        extra_obstacles=extras,
        monkeypatch=monkeypatch,
        max_sim_s=90.0,
        noise=c["noise"],
        seed=c["seed"],
        clearance_m=c["clearance_m"],
    )
    d = _detail(r)
    assert r.contacts == 0, "body contact: " + d
    if fits:
        assert r.state == "succeeded", d
    else:
        # Fail on its own (bounded retries), not sit until the test cap.
        assert r.state == "failed", d
    assert r.length_flips() == 0, d

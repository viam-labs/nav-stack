"""Table-driven nav scenarios that assert on ``progress["obstacle"]`` sequences.

These sit on top of ``tight_space_sim`` (real ``NavSupervisor`` + rectangular
raycast world). Outcome checks (succeed / fail / no contact) still matter, but
the regressions we keep finding on tracer2 are *which recovery ran* — bumper
reverse vs freeze, wait vs replan thrash, loc-hold vs driving through a jump.

Each row is one closed-loop story: build a world, run a goal, require an
obstacle-state subsequence (and forbid a few bad ones).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import pytest

from src.geom.conversions import Pose2D

from . import tight_space_sim as sim


@dataclass(frozen=True)
class Scenario:
    name: str
    # Builder returns (nav_grid, start, goal).
    build: object
    expect_state: Union[str, Tuple[str, ...]] = "succeeded"
    # Subsequence of progress["obstacle"] transitions (intermediates OK).
    expect_order: Tuple[str, ...] = ()
    # Labels that must each appear somewhere (order free).
    expect_saw: Tuple[str, ...] = ()
    # Labels that must never appear.
    forbid: Tuple[str, ...] = ()
    # Unmapped extras painted into the truth grid only (lidar + body).
    extras: Tuple[Tuple[float, float, float, float], ...] = ()
    timed: Tuple[Tuple[float, float, float, float, float, float], ...] = ()
    # Seal the nav map after t_on so replans fail while truth already blocks.
    timed_map: Tuple[Tuple[float, float, float, float, float], ...] = ()
    loc_hold_window: Optional[Tuple[float, float]] = None
    clearance_m: float = 0.03
    max_sim_s: float = 60.0
    require_reverse: bool = False
    max_sim_s_soft: Optional[float] = None


def _room():
    return sim.blank(10.0, 6.0)


def _hallway(y0: float = 2.4, y1: float = 3.6):
    g = sim.blank(12.0, 6.0)
    sim.box(g, 0.0, 0.0, 12.0, y0)
    sim.box(g, 0.0, y1, 12.0, 6.0)
    return g


def _wide_hallway():
    return _hallway(y0=2.0, y1=4.0)


def _door_wall(g, *, x: float, door_m: float, y_mid: float = 3.0, thick: float = 0.1):
    half = door_m / 2.0
    sim.box(g, x, 0.0, x + thick, y_mid - half)
    sim.box(g, x, y_mid + half, x + thick, 6.0)


def _bumper_unmapped_wall():
    # Map is open floor. An unmapped thin wall sits an inch off the bumper:
    # path cost stays low so the blocked-nose wait never starts; the escape
    # machine must reverse then replan.
    return _room(), Pose2D(1.5, 3.0, 0.0), Pose2D(8.0, 3.0, 0.0)


def _person_crosses_hallway():
    return _hallway(), Pose2D(1.0, 3.0, 0.0), Pose2D(10.0, 3.0, 0.0)


def _door_0_85_straight():
    g = _room()
    _door_wall(g, x=5.0, door_m=0.85)
    return g, Pose2D(2.0, 3.0, 0.0), Pose2D(8.0, 3.0, 0.0)


def _open_floor_loc_hold():
    return _room(), Pose2D(1.0, 3.0, 0.0), Pose2D(8.0, 3.0, 0.0)


def _blocked_nose_unstick():
    # Wide hallway: unmapped plug ahead. After the robot closes up, seal the
    # same cells into the nav map so replans fail and wait-unstick must reverse.
    return _hallway(y0=2.2, y1=3.8), Pose2D(1.0, 3.0, 0.0), Pose2D(10.0, 3.0, 0.0)


def _corridor_yaw_open():
    # Offset unmapped blob in a wide hall: nose blocked, raw-scan opening to
    # the side → yaw_open / corridor instead of freezing forever.
    return _wide_hallway(), Pose2D(1.5, 3.0, 0.0), Pose2D(10.0, 3.0, 0.0)


SCENARIOS: Sequence[Scenario] = (
    Scenario(
        name="bumper_unmapped_wall",
        build=_bumper_unmapped_wall,
        expect_order=("narrow_reverse", "narrow_reverse_done"),
        extras=((1.92, 2.5, 2.02, 3.5),),
        require_reverse=True,
        max_sim_s=45.0,
    ),
    Scenario(
        name="person_crosses_hallway",
        build=_person_crosses_hallway,
        expect_order=("wait", "clear"),
        timed=((3.6, 2.4, 4.0, 3.6, 4.0, 6.0),),
        max_sim_s=50.0,
    ),
    Scenario(
        name="door_0.85_straight",
        build=_door_0_85_straight,
        forbid=("narrow_reverse", "backup", "in_lethal"),
        max_sim_s=40.0,
        max_sim_s_soft=30.0,
    ),
    Scenario(
        name="map_mismatch_loc_hold",
        build=_open_floor_loc_hold,
        expect_order=("loc_hold", "clear"),
        loc_hold_window=(3.0, 6.0),
        max_sim_s=40.0,
    ),
    Scenario(
        name="blocked_nose_unstick",
        build=_blocked_nose_unstick,
        # Wait crawl used to starve unstick; must reverse then replan, then
        # fail cleanly once the sealed plug leaves no route.
        expect_state="failed",
        expect_order=("wait_reverse", "planning"),
        extras=((4.5, 2.2, 4.75, 3.8),),
        timed_map=((4.5, 2.2, 4.75, 3.8, 6.0),),
        require_reverse=True,
        max_sim_s=70.0,
    ),
    Scenario(
        name="corridor_yaw_open",
        build=_corridor_yaw_open,
        expect_saw=("yaw_open",),
        extras=((3.5, 3.05, 3.9, 3.55),),
        forbid=("in_lethal",),
        max_sim_s=50.0,
    ),
)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_nav_scenario_obstacle_sequence(monkeypatch, scenario: Scenario):
    grid, start, goal = scenario.build()
    r = sim.run(
        grid,
        start,
        goal,
        extra_obstacles=scenario.extras,
        timed_obstacles=scenario.timed,
        timed_map_obstacles=scenario.timed_map,
        loc_hold_window=scenario.loc_hold_window,
        monkeypatch=monkeypatch,
        max_sim_s=scenario.max_sim_s,
        clearance_m=scenario.clearance_m,
    )
    d = r.detail()
    assert r.contacts == 0, "body contact: " + d
    allowed = (
        (scenario.expect_state,)
        if isinstance(scenario.expect_state, str)
        else scenario.expect_state
    )
    assert r.state in allowed, d
    if scenario.expect_order:
        assert r.saw_in_order(*scenario.expect_order), (
            f"missing subsequence {scenario.expect_order}: " + d
        )
    if scenario.expect_saw:
        assert r.saw(*scenario.expect_saw), f"missing {scenario.expect_saw}: " + d
    if scenario.forbid:
        assert r.never(*scenario.forbid), f"saw forbidden {scenario.forbid}: " + d
    if scenario.require_reverse:
        assert r.reversing_s > 0.05, "expected a reverse escape: " + d
    if scenario.max_sim_s_soft is not None:
        assert r.sim_s < scenario.max_sim_s_soft, "too slow: " + d


def test_run_result_sequence_helpers():
    r = sim.RunResult(
        state="succeeded",
        error="",
        sim_s=1.0,
        final_pose=Pose2D(0.0, 0.0, 0.0),
        contacts=0,
        first_contact=None,
        obstacle_states=["clear", "wait", "clear", "loc_hold", "clear"],
    )
    assert r.saw("wait", "loc_hold")
    assert r.never("backup", "in_lethal")
    assert r.saw_in_order("wait", "loc_hold")
    assert r.saw_in_order("clear", "clear", "clear")
    assert not r.saw_in_order("loc_hold", "wait")
    assert not r.saw("backup")

"""Exploration run: structure, fill, revisit, then save the map and go home."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Tuple

from .circuits import (
    circuit_closed,
    facing_wall_theta,
    inner_circuit_goals,
    next_wall_hop,
)
from .clients import NavClient, NavStatus, Pose, SlamClient
from .grid import Candidate, OccGrid, frontier_goals, under_seen_goals, visible_new_cells
from .legs import LegResult, LegRunner
from .scoring import Blacklist, Scored, rank, score

HOP_M = 1.5
STRUCTURE_KEYFRAME_SHARE = 0.6
STRUCTURE_PAUSE_S = 2.0
FILL_PAUSE_S = 1.0
CANDIDATE_CAP = 30
SCORE_RANGE_M = 8.0
FAILURES_BEFORE_WAIT = 5
FAILURE_WAIT_S = 30.0
FAILURES_BEFORE_FINISH = 20
LOST_POSE_POLL_S = 5.0
LOST_POSE_MAX_S = 60.0
IDLE_POLL_S = 5.0
MIN_LEG_M = 1.0


@dataclass
class RunParams:
    duration_s: float
    max_linear_mps: float
    max_angular_dps: float
    home_tolerance_m: float
    structure_fraction: float
    revisit_fraction: float
    well_seen_m: float
    keyframe_max: int
    min_score: float = 2.0
    robot_radius_m: float = 0.3


@dataclass
class Circuit:
    waypoints: List[Candidate]
    closures: int = 0


class _Finish(Exception):
    pass


@dataclass
class _State:
    phase: str = "idle"
    legs: int = 0
    legs_failed: int = 0
    consecutive_failures: int = 0
    coverage_cells: int = 0
    frontier_count: int = 0
    keyframes_used: int = 0
    last_error: str = ""
    home: Optional[Pose] = None
    visited: List[Tuple[float, float]] = field(default_factory=list)
    circuits: List[Circuit] = field(default_factory=list)


class ExploreRun:
    def __init__(
        self,
        nav: NavClient,
        slam: SlamClient,
        params: RunParams,
        *,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
        stop_base: Callable[[], Awaitable[None]],
    ):
        self._nav = nav
        self._slam = slam
        self._p = params
        self._clock = clock
        self._raw_sleep = sleep
        self._stop_base = stop_base
        self._stop_requested = False
        self._started_at: Optional[float] = None
        self._s = _State()
        self._blacklist = Blacklist()
        self._legs = LegRunner(
            nav,
            max_linear_mps=params.max_linear_mps,
            clock=clock,
            sleep=self._sleep,
            should_stop=lambda: self._stop_requested,
        )

    # -- public ----------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop_requested = True

    def status(self) -> dict:
        elapsed = 0.0 if self._started_at is None else self._clock() - self._started_at
        home = self._s.home
        return {
            "phase": self._s.phase,
            "elapsed_s": elapsed,
            "remaining_s": max(0.0, self._p.duration_s - elapsed),
            "legs": self._s.legs,
            "legs_failed": self._s.legs_failed,
            "coverage_cells": self._s.coverage_cells,
            "frontier_count": self._s.frontier_count,
            "keyframes_used": self._s.keyframes_used,
            "last_error": self._s.last_error,
            "home": None if home is None else {"x": home.x, "y": home.y, "theta": home.theta},
        }

    async def preflight(self) -> Optional[str]:
        slam = await self._slam.status()
        if slam.mode != "mapping":
            return f"slam mode must be mapping (got {slam.mode or 'unknown'})"
        nav = await self._nav.status()
        limit = math.radians(self._p.max_angular_dps)
        if nav.max_angular_rad_s is None:
            return "nav angular limit unknown; navigation get_status has no limits field"
        if nav.max_angular_rad_s > limit + 1e-9:
            return (
                f"nav angular limit {nav.max_angular_rad_s:.2f} rad/s exceeds explore "
                f"max_angular_dps {self._p.max_angular_dps:g} ({limit:.2f} rad/s)"
            )
        return None

    async def run(self) -> dict:
        self._started_at = self._clock()
        error = await self.preflight()
        if error is not None:
            self._s.last_error = error
            self._s.phase = "done"
            return self.status()
        try:
            pose = await self._require_pose()
            self._s.home = pose
            self._s.visited.append((pose.x, pose.y))
            await self._structure_phase()
            await self._fill_phase()
            await self._revisit_phase()
        except _Finish:
            pass
        await self._finish()
        return self.status()

    # -- helpers ---------------------------------------------------------------
    async def _sleep(self, s: float) -> None:
        await self._raw_sleep(s)
        await asyncio.sleep(0)

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    def _remaining(self) -> float:
        return self._p.duration_s - self._elapsed()

    def _check_budget(self) -> None:
        if self._stop_requested or self._remaining() <= 0.0:
            raise _Finish()

    async def _set_phase(self, phase: str) -> None:
        self._s.phase = phase
        await asyncio.sleep(0)

    async def _require_pose(self) -> Pose:
        waited = 0.0
        while True:
            nav = await self._nav.status()
            slam = await self._slam.status()
            self._s.keyframes_used = slam.keyframes
            if nav.localized and nav.pose is not None and slam.running:
                return nav.pose
            if waited >= LOST_POSE_MAX_S:
                self._s.last_error = (
                    f"lost pose for {LOST_POSE_MAX_S:g} s"
                    if not (nav.localized and nav.pose is not None)
                    else "slam not running"
                )
                raise _Finish()
            await self._sleep(LOST_POSE_POLL_S)
            waited += LOST_POSE_POLL_S

    async def _grid(self) -> Optional[OccGrid]:
        raw = await self._slam.grid()
        if raw is None:
            return None
        grid = OccGrid.from_get_grid(raw)
        if grid is not None:
            self._s.coverage_cells = grid.known_free_count()
        return grid

    async def _keyframes(self) -> int:
        st = await self._slam.status()
        self._s.keyframes_used = st.keyframes
        return st.keyframes

    async def _leg(
        self, goal: Candidate, *, planned: bool, length_m: float, theta: Optional[float] = None
    ) -> LegResult:
        self._check_budget()
        self._s.legs += 1
        result = await self._legs.run(
            goal.x, goal.y, theta if theta is not None else 0.0,
            planned=planned, planned_length_m=length_m,
        )
        if result.reason == "canceled" and self._stop_requested:
            raise _Finish()
        if result.reason == "lost_pose":
            await self._require_pose()
        if result.ok:
            self._s.consecutive_failures = 0
            if result.end_pose is not None:
                self._s.visited.append((result.end_pose.x, result.end_pose.y))
            return result
        self._s.legs_failed += 1
        self._s.consecutive_failures += 1
        self._blacklist.add(goal.x, goal.y, self._clock())
        if self._s.consecutive_failures >= FAILURES_BEFORE_FINISH:
            self._s.last_error = f"too many failed legs ({self._s.consecutive_failures} in a row)"
            raise _Finish()
        if self._s.consecutive_failures % FAILURES_BEFORE_WAIT == 0:
            await self._sleep(FAILURE_WAIT_S)
        return result

    async def _pose(self) -> Pose:
        st = await self._nav.status()
        if st.pose is None or not st.localized:
            return await self._require_pose()
        return st.pose

    # -- phases ----------------------------------------------------------------
    def _structure_budget_left(self, keyframes: int) -> bool:
        return (
            self._elapsed() < self._p.structure_fraction * self._p.duration_s
            and keyframes < STRUCTURE_KEYFRAME_SHARE * self._p.keyframe_max
        )

    async def _structure_phase(self) -> None:
        await self._set_phase("structure")
        start = await self._pose()
        outer = Circuit(waypoints=[])
        closures_before = (await self._slam.status()).closures
        path_m = 0.0
        pose = start
        while self._structure_budget_left(await self._keyframes()):
            self._check_budget()
            grid = await self._grid()
            hop = None if grid is None else next_wall_hop(grid, pose, hop_m=HOP_M)
            if hop is None or self._blacklist.blocked(hop.x, hop.y, self._clock()):
                break
            theta = facing_wall_theta(grid, Pose(hop.x, hop.y, 0.0))
            result = await self._leg(hop, planned=False, length_m=HOP_M, theta=theta)
            if result.ok and result.end_pose is not None:
                path_m += math.hypot(result.end_pose.x - pose.x, result.end_pose.y - pose.y)
                pose = result.end_pose
                outer.waypoints.append(hop)
            else:
                pose = await self._pose()
            await self._sleep(STRUCTURE_PAUSE_S)
            if circuit_closed(path_m, pose, start):
                break
        if outer.waypoints:
            outer.closures = (await self._slam.status()).closures - closures_before
            self._s.circuits.append(outer)
        inner = 0
        while inner < 3 and self._structure_budget_left(await self._keyframes()):
            grid = await self._grid()
            if grid is None:
                break
            goals = inner_circuit_goals(grid, await self._pose(), clearance_m=self._p.robot_radius_m * 3)
            if len(goals) < 2:
                break
            circuit = Circuit(waypoints=goals)
            before = (await self._slam.status()).closures
            await self._drive_circuit(circuit, pause_s=STRUCTURE_PAUSE_S, structure=True)
            circuit.closures = (await self._slam.status()).closures - before
            self._s.circuits.append(circuit)
            inner += 1

    async def _drive_circuit(self, circuit: Circuit, *, pause_s: float, structure: bool) -> int:
        driven = 0
        for goal in circuit.waypoints:
            if structure and not self._structure_budget_left(await self._keyframes()):
                return driven
            self._check_budget()
            if self._blacklist.blocked(goal.x, goal.y, self._clock()):
                continue
            here = await self._pose()
            if math.hypot(goal.x - here.x, goal.y - here.y) <= MIN_LEG_M:
                continue
            feasible, length = await self._nav.plan_to_point(goal.x, goal.y)
            if not feasible:
                continue
            grid = await self._grid()
            theta = None if grid is None else facing_wall_theta(grid, Pose(goal.x, goal.y, 0.0))
            await self._leg(goal, planned=True, length_m=length, theta=theta)
            driven += 1
            await self._sleep(pause_s)
        return driven

    async def _candidates(self) -> List[Scored]:
        grid = await self._grid()
        if grid is None:
            self._s.frontier_count = 0
            return []
        pose = await self._pose()
        now = self._clock()
        frontiers = frontier_goals(grid, self._p.robot_radius_m)
        self._s.frontier_count = len(frontiers)
        under = under_seen_goals(grid, self._s.visited, self._p.well_seen_m)
        pool = [c for c in frontiers + under if not self._blacklist.blocked(c.x, c.y, now)]
        pool.sort(key=lambda c: math.hypot(c.x - pose.x, c.y - pose.y))
        scored: List[Scored] = []
        for cand in pool[:CANDIDATE_CAP]:
            feasible, length = await self._nav.plan_to_point(cand.x, cand.y)
            if not feasible:
                continue
            gain = visible_new_cells(grid, cand.x, cand.y, max_range_m=SCORE_RANGE_M)
            scored.append(Scored(cand, gain, length, score(gain, length)))
        return rank(scored, self._p.min_score)

    async def _fill_phase(self) -> None:
        await self._set_phase("fill")
        while self._remaining() > self._p.revisit_fraction * self._p.duration_s:
            self._check_budget()
            ranked = await self._candidates()
            if not ranked:
                return
            best = ranked[0]
            await self._leg(best.candidate, planned=True, length_m=best.path_len_m)
            await self._sleep(FILL_PAUSE_S)

    async def _revisit_phase(self) -> None:
        slam = await self._slam.status()
        self._s.keyframes_used = slam.keyframes
        can_close = not slam.graph_full and slam.keyframes < self._p.keyframe_max
        if can_close and self._s.circuits:
            await self._set_phase("revisit")
            ordered = sorted(self._s.circuits, key=lambda c: c.closures)
            while self._remaining() > 0.0:
                driven = 0
                for circuit in ordered:
                    self._check_budget()
                    driven += await self._drive_circuit(
                        circuit, pause_s=STRUCTURE_PAUSE_S, structure=False
                    )
                if driven == 0:
                    await self._sleep(IDLE_POLL_S)
        else:
            await self._set_phase("fill")
        while self._remaining() > 0.0:
            self._check_budget()
            ranked = await self._candidates()
            if ranked:
                best = ranked[0]
                await self._leg(best.candidate, planned=True, length_m=best.path_len_m)
                await self._sleep(FILL_PAUSE_S)
            else:
                await self._sleep(IDLE_POLL_S)

    async def _finish(self) -> None:
        await self._set_phase("finishing")
        try:
            await self._nav.cancel()
            await self._slam.optimize()
            await self._slam.save_map()
            home = self._s.home
            went_home = False
            if home is not None:
                for _ in range(2):
                    went_home = await self._go_home(home)
                    if went_home:
                        break
            if not went_home:
                await self._stop_base()
                if home is None:
                    self._s.last_error = self._s.last_error or "return home skipped: no home pose"
                else:
                    msg = f"return home failed: {self._last_home_reason}"
                    self._s.last_error = f"{self._s.last_error}; {msg}" if self._s.last_error else msg
        except Exception as exc:  # noqa: BLE001
            self._s.last_error = f"finish failed: {exc}"
        await self._set_phase("done")

    _last_home_reason = ""

    async def _go_home(self, home: Pose) -> bool:
        pose = (await self._nav.status()).pose
        feasible, length = await self._nav.plan_to_point(home.x, home.y)
        if not feasible:
            length = 0.0 if pose is None else math.hypot(home.x - pose.x, home.y - pose.y)
        self._stop_requested = False
        result = await self._legs.run(
            home.x, home.y, home.theta,
            planned=True, planned_length_m=length, goal_tolerance_m=self._p.home_tolerance_m,
        )
        self._last_home_reason = result.reason
        return result.ok

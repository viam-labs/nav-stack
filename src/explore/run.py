"""Exploration run: drive to frontiers until none are left, then save and go home.

Every tick one wavefront scan finds the reachable frontiers and their path
lengths (no per-candidate planning). The robot keeps moving: the next goal is
handed to nav shortly before the current one is reached, a goal whose frontier
has already been seen is dropped unless it lies on the way to the next one,
and a goal that stops making progress is blacklisted.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Tuple

from .clients import NavClient, NavStatus, Pose, SlamClient
from .grid import Frontier, FrontierScan, scan_frontiers
from .legs import LegRunner
from .scoring import Blacklist, cost

TICK_S = 1.0
# Send the next goal once the current one is this close, so nav never stops
# and turns to a final heading in front of a frontier that is already seen.
HANDOFF_M = 1.0
# A goal's frontier counts as seen once no frontier cell is this close to it.
SEEN_RADIUS_M = 0.5
# A live goal is only replaced by a frontier at least this much cheaper.
REPLACE_MARGIN_M = 2.0
# Goals closer than this to the current one are the same frontier.
SAME_GOAL_M = 0.75
PROGRESS_TIMEOUT_S = 30.0
PROGRESS_MIN_M = 0.3
# Scans in a row with no reachable frontier before the space counts as mapped
# (lets the map catch up with the last scans).
EMPTY_SCANS_TO_FINISH = 3
FAILURES_BEFORE_FINISH = 10
LOST_POSE_POLL_S = 5.0
LOST_POSE_MAX_S = 60.0
# Ticks in a row with the robot nowhere near traversable mapped space before
# the run gives up (squeezed against an obstacle, or the map moved under it).
OFF_MAP_TICKS_MAX = 30
SLAM_STATUS_EVERY = 10


@dataclass
class RunParams:
    duration_s: Optional[float]
    max_linear_mps: float
    max_angular_dps: float
    home_tolerance_m: float
    robot_radius_m: float = 0.3
    min_frontier_m: float = 0.5


class _Finish(Exception):
    pass


@dataclass
class _Goal:
    frontier: Frontier
    sent_at: float
    best_path_m: float
    progress_at: float


@dataclass
class _State:
    phase: str = "idle"
    goals: int = 0
    goals_failed: int = 0
    consecutive_failures: int = 0
    coverage_cells: int = 0
    frontier_count: int = 0
    unreachable_frontiers: int = 0
    keyframes_used: int = 0
    decision_ms: float = 0.0
    last_error: str = ""
    home: Optional[Pose] = None
    goal: Optional[_Goal] = None
    ticks: int = 0
    off_map_ticks: int = 0


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
        self._abort_home = False
        self._started_at: Optional[float] = None
        self._ended_at: Optional[float] = None
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
        if self._s.phase == "finishing":
            self._abort_home = True

    def status(self) -> dict:
        """Run state for the ``status`` DoCommand.

        Keys with no value are left out rather than set to None: a DoCommand
        reply turns None into 0 on the client.
        """
        if self._started_at is None:
            elapsed = 0.0
        else:
            end = self._ended_at if self._ended_at is not None else self._clock()
            elapsed = end - self._started_at
        out = {
            "phase": self._s.phase,
            "elapsed_s": elapsed,
            "goals": self._s.goals,
            "goals_failed": self._s.goals_failed,
            "coverage_cells": self._s.coverage_cells,
            "frontier_count": self._s.frontier_count,
            # Frontiers left out after failed goals: the map is incomplete there.
            "unreachable_frontiers": self._s.unreachable_frontiers,
            "keyframes_used": self._s.keyframes_used,
            "decision_ms": round(self._s.decision_ms, 1),
            "last_error": self._s.last_error,
        }
        duration = self._p.duration_s
        if duration is not None:
            out["duration_s"] = duration
            out["remaining_s"] = max(0.0, duration - elapsed)
        goal = self._s.goal
        if goal is not None:
            out["goal"] = {"x": goal.frontier.goal_x, "y": goal.frontier.goal_y}
        home = self._s.home
        if home is not None:
            out["home"] = {"x": home.x, "y": home.y, "theta": home.theta}
        return out

    async def preflight(self) -> Optional[str]:
        slam = await self._slam.status()
        if slam.mode != "mapping":
            return f"slam mode must be mapping (got {slam.mode or 'unknown'})"
        nav = await self._nav.status()
        limit = math.radians(self._p.max_angular_dps)
        if nav.max_angular_rad_s is None:
            return (
                "nav angular limit unknown; navigation get_status has no limits field"
            )
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
            await self._explore()
        except _Finish:
            pass
        except Exception as exc:  # noqa: BLE001
            self._s.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            await self._finish()
            self._ended_at = self._clock()
        return self.status()

    # -- helpers ---------------------------------------------------------------
    async def _sleep(self, s: float) -> None:
        await self._raw_sleep(s)
        await asyncio.sleep(0)

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    def _check_budget(self) -> None:
        if self._stop_requested:
            raise _Finish()
        if self._p.duration_s is not None and self._elapsed() >= self._p.duration_s:
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

    # -- exploring -------------------------------------------------------------
    async def _explore(self) -> None:
        await self._set_phase("exploring")
        empty_scans = 0
        while True:
            self._check_budget()
            nav = await self._nav.status()
            if not nav.localized or nav.pose is None:
                await self._drop_goal(cancel=True)
                await self._require_pose()
                continue
            pose = nav.pose
            self._s.ticks += 1
            if self._s.ticks % SLAM_STATUS_EVERY == 1:
                self._s.keyframes_used = (await self._slam.status()).keyframes
            grid = await self._slam.occupancy()
            if grid is None:
                await self._sleep(TICK_S)
                continue
            self._s.coverage_cells = grid.known_free_count()
            t0 = time.perf_counter()
            scan = await asyncio.to_thread(
                scan_frontiers,
                grid,
                pose.x,
                pose.y,
                self._p.robot_radius_m,
                min_frontier_m=self._p.min_frontier_m,
            )
            self._s.decision_ms = (time.perf_counter() - t0) * 1000.0
            if not scan.start_found:
                # Not "nothing left": keep the current goal and let nav move.
                self._s.off_map_ticks += 1
                if self._s.off_map_ticks >= OFF_MAP_TICKS_MAX:
                    self._s.last_error = "robot is not in traversable mapped space"
                    raise _Finish()
                await self._sleep(TICK_S)
                continue
            self._s.off_map_ticks = 0
            now = self._clock()
            ranked = sorted(
                scan.frontiers, key=lambda f: cost(f, pose.x, pose.y, pose.theta)
            )
            nxt = await self._decide(nav, pose, scan, ranked, now)
            open_count = len(self._open(ranked, now))
            self._s.frontier_count = open_count
            self._s.unreachable_frontiers = len(ranked) - open_count
            if nxt is not None:
                await self._send(nxt, pose, now)
            if self._s.goal is None and self._best(ranked, now) is None:
                empty_scans += 1
                if empty_scans >= EMPTY_SCANS_TO_FINISH:
                    return
            else:
                empty_scans = 0
            await self._sleep(TICK_S)

    async def _decide(
        self,
        nav: NavStatus,
        pose: Pose,
        scan: FrontierScan,
        ranked: List[Frontier],
        now: float,
    ) -> Optional[Frontier]:
        """The frontier to send now, or None to keep the current goal.

        ``ranked`` is every reachable frontier, cheapest first; the best one is
        re-picked after any blacklist change so a failed goal is never resent.
        """
        g = self._s.goal
        if g is None:
            return self._best(ranked, now)
        f = g.frontier
        seen = not scan.frontier_near(f.x, f.y, SEEN_RADIUS_M)
        if not nav.active:
            if nav.state == "succeeded":
                if not seen:
                    # Arrived but the frontier is still there: it cannot be
                    # cleared from here (glass, a drop, sensor height).
                    self._blacklist.add(f.goal_x, f.goal_y, now)
                self._goal_reached()
            else:
                await self._goal_failed(cancel=False)
            return self._best(ranked, now)
        remaining = scan.path_m(f.goal_x, f.goal_y)
        stalled = False
        if remaining is not None and remaining < g.best_path_m - PROGRESS_MIN_M:
            g.best_path_m, g.progress_at = remaining, now
        elif now - g.progress_at > PROGRESS_TIMEOUT_S:
            stalled = True
        if remaining is None or stalled:
            # Unreachable now, or no progress: blacklist it and move on.
            self._blacklist.add(f.goal_x, f.goal_y, now)
            best = self._best(ranked, now)
            await self._goal_failed(cancel=best is None)
            return best
        best = self._best(ranked, now)
        if best is None:
            if seen:
                await self._drop_goal(cancel=True)
            return None
        if math.hypot(best.goal_x - f.goal_x, best.goal_y - f.goal_y) < SAME_GOAL_M:
            return None
        if remaining <= HANDOFF_M:
            self._goal_reached()
            return best
        if seen:
            # Keep driving only while the old goal is on the way to the new one.
            detour = remaining + math.hypot(
                best.goal_x - f.goal_x, best.goal_y - f.goal_y
            )
            if best.path_m < detour - HANDOFF_M:
                self._goal_reached()
                return best
            return None
        current = next(
            (
                c
                for c in scan.frontiers
                if math.hypot(c.goal_x - f.goal_x, c.goal_y - f.goal_y) < SAME_GOAL_M
            ),
            None,
        )
        if current is not None and (
            cost(best, pose.x, pose.y, pose.theta) + REPLACE_MARGIN_M
            < cost(current, pose.x, pose.y, pose.theta)
        ):
            return best
        return None

    def _open(self, ranked: List[Frontier], now: float) -> List[Frontier]:
        return [
            f for f in ranked if not self._blacklist.blocked(f.goal_x, f.goal_y, now)
        ]

    def _best(self, ranked: List[Frontier], now: float) -> Optional[Frontier]:
        open_ = self._open(ranked, now)
        return open_[0] if open_ else None

    async def _send(self, f: Frontier, pose: Pose, now: float) -> None:
        self._check_budget()
        theta = math.atan2(f.goal_y - pose.y, f.goal_x - pose.x)
        await self._nav.navigate_to_point(f.goal_x, f.goal_y, theta)
        self._s.goals += 1
        self._s.goal = _Goal(f, sent_at=now, best_path_m=f.path_m, progress_at=now)

    def _goal_reached(self) -> None:
        self._s.consecutive_failures = 0
        self._s.goal = None

    async def _drop_goal(self, *, cancel: bool) -> None:
        if self._s.goal is not None and cancel:
            await self._nav.cancel()
        self._s.goal = None

    async def _goal_failed(self, *, cancel: bool) -> None:
        g = self._s.goal
        if g is None:
            return
        self._blacklist.add(g.frontier.goal_x, g.frontier.goal_y, self._clock())
        await self._drop_goal(cancel=cancel)
        self._s.goals_failed += 1
        self._s.consecutive_failures += 1
        if self._s.consecutive_failures >= FAILURES_BEFORE_FINISH:
            self._s.last_error = (
                f"too many failed goals ({self._s.consecutive_failures} in a row)"
            )
            raise _Finish()

    # -- finish ----------------------------------------------------------------
    async def _finish(self) -> None:
        await self._set_phase("finishing")
        self._s.goal = None
        errors: List[str] = []
        for name, step in (
            ("cancel", self._nav.cancel),
            ("optimize", self._slam.optimize),
            ("save_map", self._slam.save_map),
        ):
            try:
                await step()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name} failed: {exc}")
        home = self._s.home
        went_home = False
        reason = "no home pose"
        if home is not None:
            for attempt in range(2):
                if attempt and self._abort_home:
                    reason = "stop requested"
                    break
                try:
                    went_home, reason = await self._go_home(home)
                except Exception as exc:  # noqa: BLE001
                    went_home, reason = False, f"{type(exc).__name__}: {exc}"
                if went_home:
                    break
        if not went_home:
            errors.append(f"return home failed: {reason}")
            try:
                await self._nav.cancel()
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._stop_base()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"base stop failed: {exc}")
        if errors:
            joined = "; ".join(errors)
            self._s.last_error = (
                f"{self._s.last_error}; {joined}" if self._s.last_error else joined
            )
        await self._set_phase("done")

    async def _go_home(self, home: Pose) -> Tuple[bool, str]:
        pose = (await self._nav.status()).pose
        feasible, length = await self._nav.plan_to_point(home.x, home.y)
        if not feasible:
            length = (
                0.0 if pose is None else math.hypot(home.x - pose.x, home.y - pose.y)
            )
        self._stop_requested = False
        result = await self._legs.run(
            home.x,
            home.y,
            home.theta,
            planned_length_m=length,
            goal_tolerance_m=self._p.home_tolerance_m,
        )
        return result.ok, result.reason

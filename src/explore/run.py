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
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Tuple

from viam.logging import getLogger

from .clients import NavClient, NavStatus, Pose, SlamClient
from .grid import Frontier, FrontierScan, Waypoint, scan_frontiers
from .legs import LegRunner
from .scoring import Blacklist, cost

LOGGER = getLogger(__name__)

TICK_S = 1.0
# Nav is sent waypoints at most this far (path length) toward the frontier.
# Nav stops the robot and plans the whole path before moving, and plan time
# grows quickly with distance: a 20 m plan took 10+ s on a busy cloud sim.
WAYPOINT_M = 4.0
# Send the next waypoint once the current one is this close, so nav never
# stops and turns to a final heading on the way.
HANDOFF_M = 1.0
# A goal's frontier counts as seen once no frontier cell is this close to it.
SEEN_RADIUS_M = 0.5
# A live goal is only replaced by a frontier at least this much cheaper.
REPLACE_MARGIN_M = 2.0
# ... and only after the live goal has been pursued this long. On tracer2a
# three goal changes in five seconds (west, south-east, west) spun the robot
# 112 degrees beside furniture and left it wedged.
GOAL_MIN_DWELL_S = 5.0
# A frontier must be found in this many scans in a row before it can become
# a goal. Small frontiers flicker in and out as the map settles; one that
# vanished 1.2 s after it was chosen caused the spin above.
FRONTIER_PERSIST_SCANS = 3
FRONTIER_TRACK_M = 1.0
# Goals closer than this to the current one are the same frontier.
SAME_GOAL_M = 0.75
# Nav plans on its own copy of the map, which can lag the live one by a moment:
# a goal at the edge of just-seen space may still be unknown to nav. A failure
# this soon after sending is retried once before the goal is blacklisted.
QUICK_FAIL_S = 3.0
PROGRESS_TIMEOUT_S = 30.0
# Two goals in a row that time out with the robot still in the same place
# mean it cannot move at all: stop and say so, every other goal needs it to.
STUCK_MOVE_M = 0.2
STUCK_TURN_RAD = math.radians(20.0)
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
    wp: Waypoint
    sent_at: float
    best_path_m: float
    progress_at: float
    started_at: float = 0.0


@dataclass
class _State:
    phase: str = "idle"
    goals: int = 0
    waypoints: int = 0
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
    last_target: Optional[Tuple[float, float]] = None
    quick_failed: Optional[Tuple[float, float]] = None
    # (x, y, scans seen in a row) for each frontier in the latest scan.
    tracks: List[Tuple[float, float, int]] = field(default_factory=list)
    # Pose at the last goal that timed out with no progress.
    stalled_at: Optional[Pose] = None
    skip_home: bool = False
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
            "waypoints": self._s.waypoints,
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
            out["waypoint"] = {"x": goal.wp.x, "y": goal.wp.y}
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
            self._track(ranked)
            nxt = await self._decide(nav, pose, scan, ranked, now)
            open_count = len(self._open(ranked, now))
            self._s.frontier_count = open_count
            self._s.unreachable_frontiers = len(ranked) - open_count
            if nxt is not None:
                await self._send(nxt, pose, now, scan)
            if self._s.goal is None and not self._open(ranked, now):
                empty_scans += 1
                if empty_scans >= EMPTY_SCANS_TO_FINISH:
                    LOGGER.info(
                        "explore: no reachable frontiers left (%d skipped after failures)",
                        self._s.unreachable_frontiers,
                    )
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
        """The frontier to head for now, or None to keep the current waypoint.

        ``ranked`` is every reachable frontier, cheapest first; the best one is
        re-picked after any blacklist change so a failed goal is never resent.
        Returning the current frontier sends the next waypoint toward it.
        """
        g = self._s.goal
        if g is None:
            return self._best(ranked, now)
        f = g.frontier
        seen = not scan.frontier_near(f.x, f.y, SEEN_RADIUS_M)
        if not nav.active:
            if nav.state == "succeeded":
                if g.wp.final and not seen:
                    # Arrived but the frontier is still there: it cannot be
                    # cleared from here (glass, a drop, sensor height).
                    self._blacklist.add(f.goal_x, f.goal_y, now)
                self._goal_reached()
            else:
                reason = f"nav {nav.state or 'stopped'}"
                if nav.error:
                    reason += f": {nav.error}"
                if now - g.sent_at <= QUICK_FAIL_S and not self._quick_failed_before(f):
                    LOGGER.info(
                        "explore: goal (%.2f, %.2f) failed right away (%s); retrying once",
                        f.goal_x,
                        f.goal_y,
                        reason,
                    )
                    self._s.quick_failed = (f.goal_x, f.goal_y)
                    self._s.goal = None
                else:
                    await self._goal_failed(reason, cancel=False)
            return self._best(ranked, now)
        remaining = scan.path_m(f.goal_x, f.goal_y)
        stalled = False
        if remaining is not None and remaining < g.best_path_m - PROGRESS_MIN_M:
            g.best_path_m, g.progress_at = remaining, now
        elif now - g.progress_at > PROGRESS_TIMEOUT_S:
            stalled = True
        if stalled:
            was = self._s.stalled_at
            if (
                was is not None
                and math.hypot(pose.x - was.x, pose.y - was.y) < STUCK_MOVE_M
                and abs(
                    math.atan2(
                        math.sin(pose.theta - was.theta),
                        math.cos(pose.theta - was.theta),
                    )
                )
                < STUCK_TURN_RAD
            ):
                self._s.last_error = (
                    f"robot cannot move from ({pose.x:.2f}, {pose.y:.2f}): two goals in a "
                    "row made no progress. It is probably boxed in and needs moving by hand"
                )
                self._s.skip_home = True
                LOGGER.info("explore: %s", self._s.last_error)
                await self._drop_goal(cancel=True)
                raise _Finish()
            self._s.stalled_at = pose
        if remaining is None or stalled:
            # Unreachable now, or no progress: blacklist it and move on.
            self._blacklist.add(f.goal_x, f.goal_y, now)
            best = self._best(ranked, now)
            reason = (
                "no longer reachable"
                if remaining is None
                else f"no progress for {PROGRESS_TIMEOUT_S:g} s"
            )
            await self._goal_failed(reason, cancel=best is None)
            return best
        best = self._best(ranked, now)
        if best is None:
            if seen:
                await self._drop_goal(cancel=True)
            return None
        to_wp = scan.path_m(g.wp.x, g.wp.y)
        near_wp = to_wp is None or to_wp <= HANDOFF_M
        if math.hypot(best.goal_x - f.goal_x, best.goal_y - f.goal_y) < SAME_GOAL_M:
            # Same frontier: on the last leg drive onto it, else hand over the
            # next waypoint toward it before reaching this one.
            return best if near_wp and not g.wp.final else None
        if near_wp:
            self._goal_reached()
            return best
        if seen:
            # Keep driving only while this waypoint is on the way to the new goal.
            detour = to_wp + math.hypot(best.goal_x - g.wp.x, best.goal_y - g.wp.y)
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
        if (
            current is not None
            and now - g.started_at >= GOAL_MIN_DWELL_S
            and cost(best, pose.x, pose.y, pose.theta) + REPLACE_MARGIN_M
            < cost(current, pose.x, pose.y, pose.theta)
        ):
            return best
        return None

    def _open(self, ranked: List[Frontier], now: float) -> List[Frontier]:
        return [
            f for f in ranked if not self._blacklist.blocked(f.goal_x, f.goal_y, now)
        ]

    def _best(self, ranked: List[Frontier], now: float) -> Optional[Frontier]:
        """Cheapest frontier that is not blacklisted and has persisted."""
        for f in self._open(ranked, now):
            if self._age(f) >= FRONTIER_PERSIST_SCANS:
                return f
        return None

    def _age(self, f: Frontier) -> int:
        return max(
            (
                n
                for x, y, n in self._s.tracks
                if math.hypot(f.goal_x - x, f.goal_y - y) < 1e-6
            ),
            default=0,
        )

    def _track(self, frontiers: List[Frontier]) -> None:
        """Count how many scans in a row each frontier has been found."""
        prev = self._s.tracks
        tracks: List[Tuple[float, float, int]] = []
        for f in frontiers:
            seen = max(
                (
                    n
                    for x, y, n in prev
                    if math.hypot(f.goal_x - x, f.goal_y - y) <= FRONTIER_TRACK_M
                ),
                default=0,
            )
            tracks.append((f.goal_x, f.goal_y, seen + 1))
        self._s.tracks = tracks

    async def _send(
        self, f: Frontier, pose: Pose, now: float, scan: FrontierScan
    ) -> None:
        """Send nav the next waypoint toward frontier ``f``."""
        self._check_budget()
        wp = scan.waypoint(f.goal_x, f.goal_y, WAYPOINT_M) or Waypoint(
            f.goal_x, f.goal_y, math.atan2(f.goal_y - pose.y, f.goal_x - pose.x), True
        )
        await self._nav.navigate_to_point(wp.x, wp.y, wp.theta)
        self._s.waypoints += 1
        last = self._s.last_target
        if (
            last is None
            or math.hypot(f.goal_x - last[0], f.goal_y - last[1]) >= SAME_GOAL_M
        ):
            self._s.goals += 1
            LOGGER.info(
                "explore: heading for a %.1f m frontier at (%.2f, %.2f), %.1f m away",
                f.size_m,
                f.goal_x,
                f.goal_y,
                f.path_m,
            )
        self._s.last_target = (f.goal_x, f.goal_y)
        prev = self._s.goal
        if prev is not None and (
            math.hypot(prev.frontier.goal_x - f.goal_x, prev.frontier.goal_y - f.goal_y)
            < SAME_GOAL_M
        ):
            # Next leg toward the same frontier: progress keeps counting from
            # the leg before, so a frontier that never gets closer still times out.
            best_path_m, progress_at = prev.best_path_m, prev.progress_at
            started_at = prev.started_at
        else:
            best_path_m, progress_at, started_at = f.path_m, now, now
        self._s.goal = _Goal(f, wp, now, best_path_m, progress_at, started_at)

    def _goal_reached(self) -> None:
        self._s.consecutive_failures = 0
        self._s.stalled_at = None
        self._s.goal = None

    async def _drop_goal(self, *, cancel: bool) -> None:
        if self._s.goal is not None and cancel:
            await self._nav.cancel()
        self._s.goal = None

    def _quick_failed_before(self, f: Frontier) -> bool:
        q = self._s.quick_failed
        return (
            q is not None and math.hypot(f.goal_x - q[0], f.goal_y - q[1]) < SAME_GOAL_M
        )

    async def _goal_failed(self, reason: str, *, cancel: bool) -> None:
        g = self._s.goal
        if g is None:
            return
        LOGGER.info(
            "explore: goal (%.2f, %.2f) failed (%s); skipping it for %.0f min",
            g.frontier.goal_x,
            g.frontier.goal_y,
            reason,
            self._blacklist.ttl_s / 60.0,
        )
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
        if self._s.skip_home:
            reason = "robot cannot move"
        elif home is not None:
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

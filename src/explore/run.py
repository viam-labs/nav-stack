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

from .clients import NavClient, NavStatus, PaintHold, PendingJump, Pose, SlamClient
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
# Ticks in a row in which a call to navigation or SLAM failed before the run
# ends on that error.
CALL_ERRORS_BEFORE_FINISH = 5
# The drive home starts within this long of the run ending, or not at all.
FINISH_HOME_WITHIN_S = 60.0
LOST_POSE_POLL_S = 5.0
LOST_POSE_MAX_S = 60.0
# Ticks in a row with the robot nowhere near traversable mapped space before
# the run gives up (squeezed against an obstacle, or the map moved under it).
OFF_MAP_TICKS_MAX = 30
# No scan has been painted for this long: stop driving. Nothing new is being
# mapped, and every meter driven on wheel odometry alone takes the pose farther
# from where it can be found again. SLAM's own position search starts at 3 s,
# so a slip it can fix is usually fixed before this.
LOST_PAUSE_S = 6.0
# Standing still, SLAM searches for the robot every two seconds.
LOST_STILL_S = 6.0
# Where the map was being painted, one pose every half meter, newest last.
TRAIL_SPACING_M = 0.5
TRAIL_MAX = 120
# How far back along that trail to drive, in turn, until scans are painted
# again. On one robot a person driving it back to ground it had just mapped
# got it painting again; nothing short of that did.
BACKTRACK_M = (2.0, 5.0, 10.0, 20.0)
BACKTRACK_TIMEOUT_S = 45.0
# Reads in a row with scans being painted before the robot counts as back.
PAINTING_CONFIRM_TICKS = 2
# Painting lost again within this distance of an earlier loss: the frontier
# the robot was heading for is skipped like any other failed goal.
LOST_SAME_PLACE_M = 3.0
# SLAM moves the pose by meters only after two looks that agree, and takes the
# second about 3 s after the first. Explore stands still that long. A jump
# still undecided after this is left to SLAM, and not waited on again for a
# while: a place that keeps matching two ways must not park the run.
JUMP_WAIT_MAX_S = 30.0
JUMP_IGNORE_S = 60.0
# The pose moved at least this far while the robot stood: SLAM took the jump.
JUMP_MOVED_M = 0.5
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
    skip_home_reason: str = "robot cannot move"
    # Poses where the map was being painted (see TRAIL_SPACING_M).
    trail: List[Pose] = field(default_factory=list)
    relocations: int = 0
    # Where the robot was each time it had to drive back to find the map.
    lost_at: List[Tuple[float, float]] = field(default_factory=list)
    ticks: int = 0
    off_map_ticks: int = 0
    empty_scans: int = 0


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
        self._hold = PaintHold()
        self._painting_ticks = 0
        self._jump: Optional[PendingJump] = None
        self._jump_since = 0.0
        self._jump_from: Optional[Pose] = None
        self._jump_ignore_until = 0.0
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
            # Times scans stopped fitting the map and the run had to stop.
            "relocations": self._s.relocations,
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
        """The pose, once navigation trusts one and SLAM is running.

        The wait is its own phase and is logged with its reason. On one
        robot a start was accepted, sat at "idle" for a minute with nothing
        in the log, and ended: it looked like a start that never happened.
        """
        waited = 0.0
        phase = self._s.phase
        while True:
            self._check_budget()
            nav = await self._nav.status()
            slam = await self._slam.status()
            self._s.keyframes_used = slam.keyframes
            if nav.localized and nav.pose is not None and slam.running:
                if waited > 0.0:
                    LOGGER.info("explore: has a pose again after %.0f s", waited)
                    await self._set_phase(phase)
                return nav.pose
            if nav.pose is None:
                why = "navigation has no pose"
            elif not nav.localized:
                why = nav.not_localized_reason or "navigation does not trust its pose"
            else:
                why = "SLAM is not running"
            if waited >= LOST_POSE_MAX_S:
                self._s.last_error = (
                    f"no trusted pose for {LOST_POSE_MAX_S:g} s ({why})"
                )
                LOGGER.warning("explore: %s", self._s.last_error)
                raise _Finish()
            if waited == 0.0:
                LOGGER.info(
                    "explore: waiting up to %g s for a pose (%s)", LOST_POSE_MAX_S, why
                )
                await self._set_phase("waiting_for_pose")
            await self._sleep(LOST_POSE_POLL_S)
            waited += LOST_POSE_POLL_S

    # -- exploring -------------------------------------------------------------
    async def _explore(self) -> None:
        await self._set_phase("exploring")
        errors = 0
        while True:
            try:
                done = await self._explore_tick()
                errors = 0
            except _Finish:
                raise
            except Exception as exc:  # noqa: BLE001
                # One call to navigation or SLAM that fails is not the end of
                # a mapping run: on one robot a single "context canceled",
                # while the module was busy for a few seconds, ended a run
                # that was going well.
                errors += 1
                if errors >= CALL_ERRORS_BEFORE_FINISH:
                    raise
                LOGGER.warning(
                    "explore: %s: %s (%d in a row); trying again",
                    type(exc).__name__,
                    exc,
                    errors,
                )
                done = False
            if done:
                return
            await self._sleep(TICK_S)

    async def _explore_tick(self) -> bool:
        """One look at the map and one decision. True when nothing is left to map."""
        self._check_budget()
        nav = await self._nav.status()
        if not nav.localized or nav.pose is None:
            await self._drop_goal(cancel=True)
            await self._require_pose()
            return False
        pose = nav.pose
        if await self._poll_jump() is not None:
            await self._stand_for_jump()
            return False
        hold = await self._slam.paint_hold()
        if hold.held_s <= 0.0:
            self._remember_painted(pose)
        elif hold.held_s >= LOST_PAUSE_S:
            await self._get_back_on_the_map(pose, hold)
            return False
        self._s.ticks += 1
        if self._s.ticks % SLAM_STATUS_EVERY == 1:
            self._s.keyframes_used = (await self._slam.status()).keyframes
        grid = await self._slam.occupancy()
        if grid is None:
            return False
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
            return False
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
            self._s.empty_scans += 1
            if self._s.empty_scans >= EMPTY_SCANS_TO_FINISH:
                LOGGER.info(
                    "explore: no reachable frontiers left (%d skipped after failures)",
                    self._s.unreachable_frontiers,
                )
                return True
        else:
            self._s.empty_scans = 0
        return False

    # -- lost while mapping -----------------------------------------------------
    def _remember_painted(self, pose: Pose) -> None:
        trail = self._s.trail
        if trail and math.hypot(pose.x - trail[-1].x, pose.y - trail[-1].y) < (
            TRAIL_SPACING_M
        ):
            return
        trail.append(pose)
        del trail[:-TRAIL_MAX]

    def _trail_point(self, back_m: float) -> Optional[Pose]:
        """The pose ``back_m`` back along the trail, or its start if shorter."""
        trail = self._s.trail
        if len(trail) < 2:
            return None
        walked = 0.0
        for newer, older in zip(reversed(trail), reversed(trail[:-1])):
            walked += math.hypot(newer.x - older.x, newer.y - older.y)
            if walked >= back_m:
                return older
        return trail[0] if walked >= 1.0 else None

    async def _painting(self) -> bool:
        """One tick later: are scans being painted again, and have they stayed so?"""
        self._check_budget()
        await self._sleep(TICK_S)
        self._hold = await self._slam.paint_hold()
        # Not while SLAM has the robot somewhere else: scans fitting here is
        # then the thing in doubt.
        jump = await self._poll_jump()
        painted = self._hold.held_s <= 0.0 and jump is None
        self._painting_ticks = self._painting_ticks + 1 if painted else 0
        return self._painting_ticks >= PAINTING_CONFIRM_TICKS

    async def _painting_within(self, seconds: float) -> bool:
        """Stand still up to ``seconds``; True once scans are painted again.

        The wait starts over while SLAM has a jump to take a second look at,
        so the robot is still standing here when it does.
        """
        waited = 0.0
        limit = seconds
        while waited < limit:
            if await self._painting():
                return True
            waited += TICK_S
            if self._jump is not None:
                limit = min(waited + seconds, seconds + 2.0 * JUMP_WAIT_MAX_S)
        return False

    # -- a jump SLAM is yet to confirm ---------------------------------------------
    async def _poll_jump(self) -> Optional[PendingJump]:
        """SLAM's jump awaiting its second look, if any.

        Says so in the log when one appears and when it ends, and gives up on
        one SLAM has not settled in ``JUMP_WAIT_MAX_S``.
        """
        now = self._clock()
        jump = None
        if now >= self._jump_ignore_until:
            jump = await self._slam.pending_jump()
        was, self._jump = self._jump, jump
        if jump is not None and was is None:
            self._jump_since = now
            self._jump_from = (await self._nav.status()).pose
            LOGGER.info(
                "explore: SLAM has the scan fitting a place %.1f m from the pose "
                "(look %d of %d); standing still for its next look",
                jump.shift_m,
                jump.confirm_count,
                jump.confirm_needed,
            )
        elif jump is None and was is not None:
            await self._jump_ended(now - self._jump_since)
        elif jump is not None and now - self._jump_since >= JUMP_WAIT_MAX_S:
            self._jump = None
            self._jump_ignore_until = now + JUMP_IGNORE_S
            LOGGER.warning(
                "explore: SLAM has not settled that jump in %g s; "
                "carrying on without waiting for it",
                JUMP_WAIT_MAX_S,
            )
        return self._jump

    async def _jump_ended(self, after_s: float) -> None:
        was, now = self._jump_from, (await self._nav.status()).pose
        moved = (
            0.0
            if was is None or now is None
            else math.hypot(now.x - was.x, now.y - was.y)
        )
        if moved < JUMP_MOVED_M:
            LOGGER.info("explore: SLAM dropped that jump after %.0f s", after_s)
            return
        # The trail was laid at poses SLAM has since moved away from.
        self._s.trail.clear()
        LOGGER.info(
            "explore: SLAM moved the pose %.2f m after %.0f s; carrying on from there",
            moved,
            after_s,
        )

    async def _stand_for_jump(self) -> None:
        """Stand still until SLAM has confirmed or dropped its pending jump.

        SLAM moves the pose by meters only after two looks that agree. On one
        robot the first look had the scan fitting a place 3.38 m from the
        pose. In the same second scans were painted again where the robot
        stood, explore set off, the second look found nothing wrong, and a
        hallway was mapped a second time beside the first.
        """
        await self._drop_goal(cancel=True)
        await self._stop_base()
        await self._set_phase("relocating")
        while self._jump is not None:
            self._check_budget()
            await self._sleep(TICK_S)
            await self._poll_jump()
        await self._set_phase("exploring")

    async def _drive_back(self, pose: Pose, target: Pose) -> bool:
        """Drive to ``target``; True as soon as scans are painted again."""
        await self._nav.navigate_to_point(target.x, target.y, target.theta)
        start = self._clock()
        far_m = math.hypot(target.x - pose.x, target.y - pose.y)
        timeout = max(BACKTRACK_TIMEOUT_S, 3.0 * far_m / self._p.max_linear_mps)
        started = False
        while True:
            if await self._painting():
                await self._nav.cancel()
                return True
            nav = await self._nav.status()
            started = started or nav.active
            if started and not nav.active:
                return False
            waited = self._clock() - start
            if waited > timeout or (not started and waited > QUICK_FAIL_S):
                await self._nav.cancel()
                return False

    async def _get_back_on_the_map(self, pose: Pose, hold: PaintHold) -> None:
        """Get scans painted again once they have stopped fitting the map.

        First stand still: SLAM searches for the robot by itself, and a
        stopped scan is the cleanest one it gets. Then drive back along the
        trail to where the map was last being painted, farther each time,
        and stop the moment scans are painted again. Nothing else counts as
        being back: on one robot SLAM reported that nowhere nearby fit better
        than the pose it had, exploring resumed on that, and the robot drove
        36 m in five minutes without mapping anything. If the trail runs out,
        the run ends where the robot stands: driving on, or home, would be on
        a pose known to be wrong.
        """
        goal = self._s.goal
        await self._drop_goal(cancel=True)
        await self._stop_base()
        await self._set_phase("relocating")
        self._s.relocations += 1
        self._hold = hold
        self._painting_ticks = 0
        LOGGER.info(
            "explore: no scan has been painted for %.0f s (%.1f m driven); "
            "stopped at (%.2f, %.2f) for SLAM to find the robot",
            hold.held_s,
            hold.travel_m,
            pose.x,
            pose.y,
        )
        how = "standing still"
        painting = await self._painting_within(LOST_STILL_S)
        drove_back = False
        last: Optional[Pose] = None
        for back_m in () if painting else BACKTRACK_M:
            target = self._trail_point(back_m)
            if target is None:
                break
            if last is not None and target is last:
                continue  # the trail is shorter than this; already been there
            last = target
            drove_back = True
            LOGGER.info(
                "explore: still not painting; driving back %.0f m to "
                "(%.2f, %.2f), where the map was last being painted",
                back_m,
                target.x,
                target.y,
            )
            how = f"driving back {back_m:g} m"
            painting = await self._drive_back(
                pose, target
            ) or await self._painting_within(LOST_STILL_S)
            if painting:
                break
        if not painting:
            self._s.last_error = (
                f"lost near ({pose.x:.2f}, {pose.y:.2f}): no scan has been "
                f"painted for {self._hold.held_s:.0f} s, and neither standing "
                "still nor driving back along the trail changed that "
                f"(SLAM: {self._hold.verdict or 'no search ran'}); relocalize "
                "the robot and start explore again"
            )
            self._s.skip_home = True
            self._s.skip_home_reason = "robot is lost"
            LOGGER.warning("explore: %s", self._s.last_error)
            raise _Finish()
        LOGGER.info("explore: painting again after %s; exploring again", how)
        if drove_back:
            await self._note_lost_place(pose, goal)
        await self._set_phase("exploring")

    async def _note_lost_place(self, pose: Pose, goal: Optional[_Goal]) -> None:
        """Skip a frontier that has now cost the map twice near the same place.

        Once can be a slip anywhere, so the first time the robot goes
        straight back for another try.
        """
        here = (pose.x, pose.y)
        again = any(
            math.hypot(here[0] - x, here[1] - y) < LOST_SAME_PLACE_M
            for x, y in self._s.lost_at
        )
        self._s.lost_at.append(here)
        if not again or goal is None:
            return
        self._s.goal = goal
        await self._goal_failed("scans stopped fitting the map there", cancel=False)

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
        began = self._clock()
        steps = [("cancel", self._nav.cancel)]
        # The map is explore's to optimize and save only while SLAM is still
        # mapping it. On one robot this step ran seven minutes late, after a
        # person had saved the map and switched SLAM to localizing.
        try:
            still_mapping = (await self._slam.status()).mode == "mapping"
        except Exception:  # noqa: BLE001
            still_mapping = True
        if still_mapping:
            steps += [
                ("optimize", self._slam.optimize),
                ("save_map", self._slam.save_map),
            ]
        else:
            errors.append("SLAM is no longer mapping; map left as it is")
            self._s.skip_home = True
            self._s.skip_home_reason = "SLAM is no longer mapping"
        for name, step in steps:
            try:
                await step()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name} failed: {exc}")
        if not self._s.skip_home and self._clock() - began > FINISH_HOME_WITHIN_S:
            # A robot that sets off by itself minutes after its run ended is
            # a surprise to whoever has walked up to it since.
            self._s.skip_home = True
            self._s.skip_home_reason = "finishing took too long"
        home = self._s.home
        went_home = False
        reason = "no home pose"
        if self._s.skip_home:
            reason = self._s.skip_home_reason
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

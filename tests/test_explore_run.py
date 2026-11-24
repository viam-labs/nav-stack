import asyncio
import math

from src.explore import run as run_mod
from src.explore.clients import PaintHold, Pose
from src.explore.grid import Frontier, Waypoint
from src.explore.run import ExploreRun, RunParams
from src.geom import conversions as conv
from src.sim.world import SimWorld, make_builtin_apartment, make_builtin_corridor
from tests.explore_fakes import FakeClock, FakeNav, FakeSlam, known_fraction


def params(duration_s=None, **kw):
    base = dict(
        duration_s=duration_s,
        max_linear_mps=0.4,
        max_angular_dps=30.0,
        home_tolerance_m=0.5,
    )
    base.update(kw)
    return RunParams(**base)


def setup(sim_map=None, seed=(1.0, 1.0, 0.0), **kw):
    w = SimWorld(sim_map or make_builtin_corridor(), seed_pose=conv.Pose2D(*seed))
    clk = FakeClock()
    slam = FakeSlam(w)
    nav = FakeNav(w, slam, clk)
    stops = []

    async def stop_base():
        stops.append(1)

    run = ExploreRun(
        nav, slam, params(**kw), clock=clk.now, sleep=clk.sleep, stop_base=stop_base
    )
    return w, clk, slam, nav, run, stops


def count_plans(nav):
    calls = []
    orig = nav.plan_to_point

    async def plan(x, y):
        calls.append((x, y))
        return await orig(x, y)

    nav.plan_to_point = plan
    return calls


def test_corridor_is_mapped_then_run_ends_on_its_own_and_goes_home():
    w, clk, slam, nav, run, _ = setup()
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and final["last_error"] == ""
    assert known_fraction(slam, w) > 0.95
    assert clk.now() < 120.0  # no time budget: it stops when nothing is left
    assert slam.saved == 1 and slam.optimized == 1
    pose = w.get_pose()
    assert abs(pose.x - 1.0) < 0.5 and abs(pose.y - 1.0) < 0.5
    assert final["home"] == {"x": 1.0, "y": 1.0, "theta": 0.0}


def test_apartment_is_mapped_through_doorways():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    final = asyncio.run(run.run())
    assert final["last_error"] == "" and known_fraction(slam, w) > 0.95
    assert clk.now() < 300.0 and final["goals_failed"] == 0


def test_exploring_never_plans_per_candidate():
    w, clk, slam, nav, run, _ = setup()
    plans = count_plans(nav)
    final = asyncio.run(run.run())
    assert final["goals"] >= 2
    assert len(plans) == 1  # only the drive home asks nav for a plan


def test_next_goal_is_handed_over_before_arrival():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    arrivals = []
    orig = nav._advance

    def advance():
        before = nav._state
        orig()
        if nav._state == "succeeded" and before == "active":
            arrivals.append(clk.now())

    nav._advance = advance
    final = asyncio.run(run.run())
    # Most goals are replaced while driving; arriving (and stopping) is rare.
    assert len(arrivals) <= final["goals"] // 2 + 1


def test_nav_goals_are_short_waypoints():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    sent = []
    orig = nav.navigate_to_point

    async def nav_to(x, y, theta):
        p = w.get_pose()
        sent.append(math.hypot(x - p.x, y - p.y))
        await orig(x, y, theta)

    nav.navigate_to_point = nav_to
    final = asyncio.run(run.run())
    exploring = sent[:-1]  # the last one is the drive home
    assert exploring and max(exploring) <= run_mod.WAYPOINT_M + 0.5
    assert final["waypoints"] == len(exploring) and final["waypoints"] > final["goals"]


def test_stop_mid_run_finishes_once():
    w, clk, slam, nav, run, _ = setup()

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["goals"] < 1:
            await asyncio.sleep(0)
        run.request_stop()
        return await task

    final = asyncio.run(go())
    assert final["phase"] == "done" and slam.saved == 1 and nav.cancels >= 1


def test_time_budget_ends_the_run():
    w, clk, slam, nav, run, _ = setup(
        make_builtin_apartment(), seed=(2.0, 2.0, 0.0), duration_s=20.0
    )
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and slam.saved == 1
    assert known_fraction(slam, w) < 0.9 and final["remaining_s"] == 0.0


def test_lost_pose_waits_then_finishes():
    w, clk, slam, nav, run, stops = setup()
    nav.lose_pose = True
    final = asyncio.run(run.run())
    assert final["last_error"].startswith("lost pose") and 60.0 <= clk.now() < 120.0
    assert slam.saved == 1 and stops == [1] and final["goals"] == 0


def test_wrong_slam_mode_refuses_before_moving():
    w, clk, slam, nav, run, _ = setup()
    slam.mode = "localizing"
    assert "mapping" in asyncio.run(run.preflight())
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and "mapping" in final["last_error"]
    assert final["goals"] == 0 and slam.saved == 0


def test_angular_limit_refuses_before_moving():
    w, clk, slam, nav, run, _ = setup()
    nav.max_angular_rad_s = 1.5
    final = asyncio.run(run.run())
    assert "angular" in final["last_error"] and final["goals"] == 0


def test_blocked_goal_is_blacklisted_and_run_survives():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    # An obstacle nav's planner does not know about: legs through it fail.
    w.add_obstacle(4.0, 2.0, 0.6)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and final["goals_failed"] >= 1
    assert final["goals"] > final["goals_failed"]


def test_stalled_goal_is_dropped_after_progress_timeout():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    nav.speed_mps = 0.0  # nav accepts goals but never gets anywhere

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["goals_failed"] < 1:
            await asyncio.sleep(0)
        failed_at = clk.now()
        run.request_stop()
        await task
        return failed_at

    failed_at = asyncio.run(go())
    assert run_mod.PROGRESS_TIMEOUT_S < failed_at < run_mod.PROGRESS_TIMEOUT_S + 5.0


def test_nav_refusing_every_goal_ends_early_and_reports_it():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["goals"] < 1:
            await asyncio.sleep(0)
        nav.block_all = True
        return await task

    final = asyncio.run(go())
    # Either the failure limit ends it, or every frontier is blacklisted; both
    # end quickly and say the map is incomplete.
    assert clk.now() < 300.0 and final["goals_failed"] >= 1
    assert "failed goals" in final["last_error"] or final["unreachable_frontiers"] >= 1


def test_home_failure_stops_in_place():
    w, clk, slam, nav, run, stops = setup()

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["phase"] != "finishing":
            await asyncio.sleep(0)
        nav.block_all = True
        return await task

    final = asyncio.run(go())
    assert final["phase"] == "done" and slam.saved == 1 and stops == [1]
    assert "home" in final["last_error"]


def test_docommand_exception_mid_run_still_finishes():
    w, clk, slam, nav, run, stops = setup()
    orig = nav.status

    async def boom():
        if run.status()["goals"] >= 1:
            raise RuntimeError("grpc unavailable")
        return await orig()

    nav.status = boom
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and "grpc unavailable" in final["last_error"]
    assert slam.saved == 1 and stops == [1]


def test_finish_saves_map_even_if_optimize_raises():
    w, clk, slam, nav, run, _ = setup(duration_s=5.0)

    async def bad_optimize():
        raise RuntimeError("optimize exploded")

    slam.optimize = bad_optimize
    final = asyncio.run(run.run())
    assert (
        final["phase"] == "done"
        and slam.saved == 1
        and "optimize" in final["last_error"]
    )


def test_stop_during_finishing_skips_home_retry():
    w, clk, slam, nav, run, stops = setup(duration_s=5.0)
    starts = {}

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["phase"] != "finishing":
            await asyncio.sleep(0)
        starts["before"] = nav.legs
        nav.block_all = True
        run.request_stop()
        return await task

    final = asyncio.run(go())
    assert (
        final["phase"] == "done" and nav.legs - starts["before"] == 1 and stops == [1]
    )


def test_status_reports_decision_time_and_current_goal():
    w, clk, slam, nav, run, _ = setup()
    snaps = []

    async def go():
        task = asyncio.ensure_future(run.run())
        while not task.done():
            st = run.status()
            if st["phase"] == "exploring" and "goal" in st:
                snaps.append(st)
            await asyncio.sleep(0)
        return await task

    asyncio.run(go())
    assert snaps and snaps[-1]["decision_ms"] > 0.0 and snaps[-1]["frontier_count"] >= 0


def test_robot_off_mapped_space_neither_finishes_nor_fails_its_goal():
    from src.explore import grid as grid_mod

    w, clk, slam, nav, run, _ = setup()
    orig = grid_mod.scan_frontiers
    off = {"on": False}

    def scan(*a, **kw):
        out = orig(*a, **kw)
        if off["on"]:
            out.frontiers, out.start_found = [], False
        return out

    run_mod.scan_frontiers = scan

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["goals"] < 1:
            await asyncio.sleep(0)
        off["on"] = True
        return await task

    try:
        final = asyncio.run(go())
    finally:
        run_mod.scan_frontiers = orig
    assert final["last_error"] == "robot is not in traversable mapped space"
    assert final["goals_failed"] == 0 and clk.now() >= run_mod.OFF_MAP_TICKS_MAX


def test_goal_rejected_right_away_is_retried_before_blacklisting():
    w, clk, slam, nav, run, _ = setup()
    orig = nav.navigate_to_point
    calls = []

    async def nav_to(x, y, theta):
        calls.append((x, y))
        await orig(x, y, theta)
        if len(calls) == 1:
            # Nav's map has not caught up yet: it rejects the first goal.
            nav._active, nav._state, nav._error = False, "failed", "goal in unknown"

    nav.navigate_to_point = nav_to
    final = asyncio.run(run.run())
    assert final["goals_failed"] == 0 and known_fraction(slam, w) > 0.95
    assert math.hypot(calls[1][0] - calls[0][0], calls[1][1] - calls[0][1]) < 0.75


def test_stuck_robot_ends_the_run_and_says_so():
    # Live tracer2a: wedged between furniture, every goal timed out after 30 s
    # and explore kept picking new ones. Two in a row from the same pose now
    # end the run with a clear error and no drive home.
    w, clk, slam, nav, run, stops = setup(
        make_builtin_apartment(), seed=(2.0, 2.0, 0.0)
    )
    nav.speed_mps = 0.0  # nav accepts goals but the robot never moves
    final = asyncio.run(run.run())
    assert "cannot move" in final["last_error"]
    assert final["goals_failed"] == 1  # the second stall ends the run instead
    assert (
        2 * run_mod.PROGRESS_TIMEOUT_S < clk.now() < 2 * run_mod.PROGRESS_TIMEOUT_S + 15
    )
    assert slam.saved == 1 and stops == [1]
    assert "robot cannot move" in final["last_error"].split("return home failed: ")[-1]


def test_a_frontier_must_persist_before_it_becomes_a_goal():
    w, clk, slam, nav, run, _ = setup()
    sent_at = []
    orig = nav.navigate_to_point

    async def nav_to(x, y, theta):
        sent_at.append(clk.now())
        await orig(x, y, theta)

    nav.navigate_to_point = nav_to
    asyncio.run(run.run())
    # The first goal waits for FRONTIER_PERSIST_SCANS scans, one per tick.
    assert sent_at[0] >= (run_mod.FRONTIER_PERSIST_SCANS - 1) * run_mod.TICK_S


def test_track_counts_consecutive_scans_and_drops_flicker():
    from src.explore.grid import Frontier

    w, clk, slam, nav, run, _ = setup()
    a = Frontier(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    moved = Frontier(1.4, 1.0, 1.4, 1.0, 1.0, 1.0)  # same frontier, robot moved
    other = Frontier(5.0, 5.0, 5.0, 5.0, 0.7, 6.0)
    run._track([a])
    run._track([moved, other])
    assert run._age(moved) == 2 and run._age(other) == 1
    assert run._best([other, moved], now=0.0) is None  # neither is old enough
    run._track([moved])  # `other` flickered out
    assert run._best([moved], now=0.0) is moved
    run._track([moved, other])
    assert run._age(other) == 1  # starts again from one


def test_live_goal_is_not_replaced_before_the_dwell_time():
    from src.explore.clients import NavStatus, Pose
    from src.explore.grid import Frontier, Waypoint

    w, clk, slam, nav, run, _ = setup()
    pose = Pose(0.0, 0.0, 0.0)
    far = Frontier(8.0, 0.0, 8.0, 0.0, 1.0, 8.0)
    near = Frontier(2.0, 0.0, 2.0, 0.0, 1.0, 2.0)

    class Scan:
        frontiers = [near, far]

        def frontier_near(self, x, y, r):
            return True  # the live goal's frontier is still there

        def path_m(self, x, y, search_m=0.3):
            return math.hypot(x, y)

    for _ in range(run_mod.FRONTIER_PERSIST_SCANS):
        run._track([near, far])
    active = NavStatus(True, "active", pose, True, 0.5, 0.3, "")

    def decide(now, started):
        run._s.goal = run_mod._Goal(
            far, Waypoint(4.0, 0.0, 0.0, False), started, 8.0, now, started
        )
        return asyncio.run(run._decide(active, pose, Scan(), [near, far], now))

    assert decide(now=103.0, started=100.0) is None  # 3 s in: keep going
    assert decide(now=106.0, started=100.0) is near  # past the dwell: switch


# --- Scans stop fitting the map -------------------------------------------------
# Live tracer2a: the pose slid 2.2 m while mapping. SLAM painted nothing more,
# and explore drove the robot between frontiers for five minutes on that pose.


def _record_goals(nav, run):
    """Every goal sent to nav, with the phase explore was in."""
    sent = []
    orig = nav.navigate_to_point

    async def go(x, y, theta):
        sent.append((run.status()["phase"], x, y))
        await orig(x, y, theta)

    nav.navigate_to_point = go
    return sent


def _lost_from(clk, at_s, placed):
    """A hold that starts at ``at_s`` and lasts until ``placed()`` is true."""
    done = []

    def hold():
        t = clk.now()
        if done or t < at_s:
            return PaintHold()
        if placed():
            done.append(t)
            return PaintHold()
        return PaintHold(held_s=t - at_s, travel_m=1.0, verdict="ambiguous")

    return hold


def test_a_lost_robot_stops_and_carries_on_once_slam_places_it():
    w, clk, slam, nav, run, stops = setup(
        make_builtin_apartment(), seed=(2.0, 2.0, 0.0)
    )
    stopped = []

    def placed():
        # SLAM finds the robot three seconds after it stops moving.
        if run.status()["phase"] == "relocating":
            stopped.append(clk.now())
        return bool(stopped) and clk.now() - stopped[0] >= 3.0

    slam.hold = _lost_from(clk, 20.0, placed)
    sent = _record_goals(nav, run)
    final = asyncio.run(run.run())
    assert final["last_error"] == "" and final["relocations"] == 1
    assert known_fraction(slam, w) > 0.95
    # Stopped within a tick of the hold reaching the limit, and never drove back.
    assert 20.0 + run_mod.LOST_PAUSE_S <= stopped[0] <= 22.0 + run_mod.LOST_PAUSE_S
    assert stops and not [goal for goal in sent if goal[0] == "relocating"]


def test_a_lost_robot_drives_back_to_where_it_was_painting():
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    visited = []
    stopped = []

    def placed():
        pose = w.get_pose()
        if clk.now() < 20.0:
            visited.append((pose.x, pose.y))
        if run.status()["phase"] != "relocating":
            return False
        stopped.append((pose.x, pose.y))
        # Scans fit again 1.5 m from where the robot stopped.
        return math.hypot(pose.x - stopped[0][0], pose.y - stopped[0][1]) >= 1.5

    slam.hold = lambda: (placed() if clk.now() < 20.0 else None) or lost()
    lost = _lost_from(clk, 20.0, placed)
    sent = _record_goals(nav, run)
    final = asyncio.run(run.run())
    assert final["last_error"] == "" and final["relocations"] == 1
    back = [goal for goal in sent if goal[0] == "relocating"]
    assert len(back) == 1
    # The way back is a place it had already been, before the scans stopped fitting.
    assert min(math.hypot(back[0][1] - x, back[0][2] - y) for x, y in visited) < 0.6
    # Exploring went on afterwards.
    assert sent.index(back[0]) < len(sent) - 1 and sent[-1][0] != "relocating"


def test_a_robot_that_stays_lost_ends_the_run_where_it_stands():
    w, clk, slam, nav, run, stops = setup(
        make_builtin_apartment(), seed=(2.0, 2.0, 0.0)
    )
    slam.hold = _lost_from(clk, 20.0, lambda: False)
    sent = _record_goals(nav, run)
    final = asyncio.run(run.run())
    assert final["last_error"].startswith("lost near (")
    assert "return home failed: robot is lost" in final["last_error"]
    # Stood still, drove back farther each time until the trail ran out, then
    # stopped for good: no drive home.
    back = [goal for goal in sent if goal[0] != "exploring"]
    assert 2 <= len(back) <= len(run_mod.BACKTRACK_M)
    assert {goal[0] for goal in back} == {"relocating"}
    assert len({goal[1:] for goal in back}) == len(back)  # never the same place twice
    assert slam.saved == 1 and stops
    assert clk.now() < 20.0 + 400.0


def test_slam_finding_nowhere_better_is_not_being_back_on_the_map():
    """Live tracer2a, 1.3.1-rc9: 36 m and five minutes with nothing painted.

    The robot started to drive back, SLAM's search reported that nowhere
    nearby fit better than the pose it had, explore took that for being back
    on the map, and turned around. Five times.
    """
    w, clk, slam, nav, run, _ = setup(make_builtin_apartment(), seed=(2.0, 2.0, 0.0))
    stopped = []
    painted_again = []

    def hold():
        t = clk.now()
        if painted_again or t < 20.0:
            return PaintHold()
        pose = w.get_pose()
        if run.status()["phase"] == "relocating":
            stopped.append((pose.x, pose.y))
        # Scans fit again 1.5 m back from where the robot stopped.
        if (
            stopped
            and math.hypot(pose.x - stopped[0][0], pose.y - stopped[0][1]) >= 1.5
        ):
            painted_again.append(t)
            return PaintHold()
        return PaintHold(held_s=t - 20.0, travel_m=1.0, verdict="no_better_place")

    slam.hold = hold
    sent = _record_goals(nav, run)
    final = asyncio.run(run.run())
    assert final["last_error"] == "" and final["relocations"] == 1
    # It stopped, and it drove back until scans were painted again.
    assert len([goal for goal in sent if goal[0] == "relocating"]) == 1
    assert painted_again


def test_a_second_loss_near_the_same_place_skips_that_frontier():
    _, clk, _, _, run, _ = setup()
    frontier = Frontier(5.0, 1.0, 5.2, 1.0, 1.0, 3.0)
    goal = run_mod._Goal(frontier, Waypoint(5.0, 1.0, 0.0, True), 0.0, 3.0, 0.0)  # noqa: SLF001

    def blocked():
        return run._blacklist.blocked(5.0, 1.0, clk.now())  # noqa: SLF001

    # Once can be a slip anywhere: straight back for another try.
    asyncio.run(run._note_lost_place(Pose(4.0, 1.0, 0.0), goal))  # noqa: SLF001
    assert run.status()["goals_failed"] == 0 and not blocked()
    # Somewhere else gets a first try of its own.
    asyncio.run(run._note_lost_place(Pose(4.0, 9.0, 0.0), goal))  # noqa: SLF001
    assert run.status()["goals_failed"] == 0 and not blocked()
    # Again near the first place: that frontier is a failed goal.
    asyncio.run(run._note_lost_place(Pose(4.5, 1.5, 0.0), goal))  # noqa: SLF001
    assert run.status()["goals_failed"] == 1 and blocked()


def test_the_trail_gives_a_pose_that_far_back_or_its_start():
    _, _, _, _, run, _ = setup()
    assert run._trail_point(2.0) is None  # noqa: SLF001
    for i in range(8):
        run._remember_painted(Pose(0.5 * i, 0.0, 0.0))  # noqa: SLF001
        run._remember_painted(Pose(0.5 * i + 0.1, 0.0, 0.0))  # noqa: SLF001  (too close)
    assert len(run._s.trail) == 8  # noqa: SLF001
    assert run._trail_point(2.0).x == 1.5  # noqa: SLF001
    assert run._trail_point(50.0).x == 0.0  # noqa: SLF001

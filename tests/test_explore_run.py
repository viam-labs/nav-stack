import asyncio
import math

from src.explore import run as run_mod
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


def test_goals_face_their_approach_direction():
    w, clk, slam, nav, run, _ = setup()
    seen = []
    orig = nav.navigate_to_point

    async def nav_to(x, y, theta):
        p = w.get_pose()
        seen.append((theta, math.atan2(y - p.y, x - p.x)))
        await orig(x, y, theta)

    nav.navigate_to_point = nav_to
    asyncio.run(run.run())
    assert seen and all(abs(t - a) < 1e-6 for t, a in seen[:-1])  # last is home


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

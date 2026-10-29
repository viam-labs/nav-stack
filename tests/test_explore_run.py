import asyncio

from src.explore.run import ExploreRun, RunParams
from src.geom import conversions as conv
from src.sim.world import SimWorld, make_builtin_corridor
from tests.explore_fakes import FakeClock, FakeNav, FakeSlam, known_fraction


def params(duration_s=600.0, **kw):
    base = dict(
        duration_s=duration_s,
        max_linear_mps=0.4,
        max_angular_dps=30.0,
        home_tolerance_m=0.5,
        structure_fraction=0.15,
        revisit_fraction=0.1,
        well_seen_m=4.0,
        keyframe_max=2000,
    )
    base.update(kw)
    return RunParams(**base)


def setup(**kw):
    w = SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 1.0, 0.0))
    clk = FakeClock()
    slam = FakeSlam(w, keyframe_max=kw.get("keyframe_max", 2000))
    nav = FakeNav(w, slam, clk)
    stops = []

    async def stop_base():
        stops.append(1)

    run = ExploreRun(nav, slam, params(**kw), clock=clk.now, sleep=clk.sleep, stop_base=stop_base)
    return w, clk, slam, nav, run, stops


async def run_watching(run):
    phases = []
    task = asyncio.ensure_future(run.run())
    while not task.done():
        phases.append(run.status()["phase"])
        await asyncio.sleep(0)
    return await task, phases


def test_full_run_covers_corridor_and_finishes_home():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and final["last_error"] == ""
    assert known_fraction(slam, w) > 0.7
    assert slam.saved == 1 and slam.optimized == 1
    pose = w.get_pose()
    assert abs(pose.x - 1.0) < 0.5 and abs(pose.y - 1.0) < 0.5
    assert final["legs"] > 3 and final["home"] == {"x": 1.0, "y": 1.0, "theta": 0.0}


def test_phases_progress_in_order():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0)
    final, seen = asyncio.run(run_watching(run))
    order = [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]]
    if order and order[0] == "idle":
        order = order[1:]
    assert order[:2] == ["structure", "fill"] and order[-2:] == ["finishing", "done"]


def test_stop_mid_leg_finishes_once():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0)

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["legs"] < 1:
            await asyncio.sleep(0)
        run.request_stop()
        return await task

    final = asyncio.run(go())
    assert final["phase"] == "done" and slam.saved == 1 and nav.cancels >= 1


def test_timeout_triggers_finish():
    w, clk, slam, nav, run, _ = setup(duration_s=60.0)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and clk.now() >= 60.0 and slam.saved == 1


def test_lost_pose_waits_then_finishes():
    w, clk, slam, nav, run, stops = setup(duration_s=900.0)
    nav.lose_pose = True
    final = asyncio.run(run.run())
    assert final["last_error"].startswith("lost pose") and 60.0 <= clk.now() < 120.0
    assert slam.saved == 1 and stops == [1] and final["legs"] == 0


def test_wrong_slam_mode_refuses_before_moving():
    w, clk, slam, nav, run, _ = setup()
    slam.mode = "localizing"
    assert "mapping" in asyncio.run(run.preflight())
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and "mapping" in final["last_error"]
    assert final["legs"] == 0 and slam.saved == 0


def test_angular_limit_refuses_before_moving():
    w, clk, slam, nav, run, _ = setup()
    nav.max_angular_rad_s = 1.5
    final = asyncio.run(run.run())
    assert "angular" in final["last_error"] and final["legs"] == 0


def test_keyframe_cap_ends_structure_and_skips_revisit():
    w, clk, slam, nav, run, _ = setup(duration_s=900.0, keyframe_max=10)
    final, phases = asyncio.run(run_watching(run))
    assert "revisit" not in phases and phases.count("structure") < phases.count("fill")


def test_failed_goals_are_blacklisted_and_run_survives():
    w, clk, slam, nav, run, _ = setup(duration_s=600.0)
    w.add_obstacle(4.0, 0.8, 0.8)
    final = asyncio.run(run.run())
    assert final["phase"] == "done" and final["legs_failed"] >= 1
    assert final["legs"] > final["legs_failed"]


def test_twenty_consecutive_failures_finish_early():
    w, clk, slam, nav, run, _ = setup(duration_s=3600.0)

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["legs"] < 1:
            await asyncio.sleep(0)
        nav.block_all = True
        return await task

    final = asyncio.run(go())
    assert "failed legs" in final["last_error"] and final["legs_failed"] >= 20
    assert clk.now() < 3600.0


def test_home_failure_stops_in_place():
    w, clk, slam, nav, run, stops = setup(duration_s=120.0)

    async def go():
        task = asyncio.ensure_future(run.run())
        while run.status()["phase"] != "finishing":
            await asyncio.sleep(0)
        nav.block_all = True
        return await task

    final = asyncio.run(go())
    assert final["phase"] == "done" and slam.saved == 1 and stops == [1]
    assert "home" in final["last_error"]

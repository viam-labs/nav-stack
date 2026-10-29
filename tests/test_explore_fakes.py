import asyncio

from src.geom import conversions as conv
from src.sim.world import SimWorld, make_builtin_corridor
from tests.explore_fakes import FakeClock, FakeNav, FakeSlam, known_fraction


def world():
    return SimWorld(make_builtin_corridor(), seed_pose=conv.Pose2D(1.0, 1.0, 0.0))


def test_world_obstacle_add_and_remove():
    w = world()
    before = w.get_scan().ranges.copy()
    oid = w.add_obstacle(2.0, 1.0, 0.2)
    assert w.get_scan().ranges.min() < before.min()
    w.remove_obstacle(oid)
    assert (w.get_scan().ranges == before).all()


def test_fake_slam_reveals_cells_and_counts_keyframes():
    w = world()
    s = FakeSlam(w)
    assert known_fraction(s, w) == 0.0
    s.reveal(conv.Pose2D(1.0, 1.0, 0.0))
    assert known_fraction(s, w) > 0.05
    s.reveal(conv.Pose2D(1.6, 1.0, 0.0))
    assert asyncio.run(s.status()).keyframes == 2
    assert asyncio.run(s.grid())["rows"] == w.map.height


def test_fake_nav_drives_to_goal_and_reveals():
    w = world()
    clk, s = FakeClock(), FakeSlam(w)
    nav = FakeNav(w, s, clk)

    async def go():
        assert await nav.plan_to_point(3.0, 1.0) == (True, 2.0)
        await nav.navigate_to_point(3.0, 1.0, 0.0)
        while (await nav.status()).active:
            await clk.sleep(0.5)
        return await nav.status()

    st = asyncio.run(go())
    assert st.state == "succeeded" and abs(st.pose.x - 3.0) < 0.05
    assert known_fraction(s, w) > 0.1


def test_fake_nav_fails_on_blocked_line():
    w = world()
    nav = FakeNav(w, FakeSlam(w), FakeClock())
    assert asyncio.run(nav.plan_to_point(7.0, 3.0)) == (False, 0.0)

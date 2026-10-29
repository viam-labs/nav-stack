import asyncio

from src.explore.clients import NavStatus, Pose
from src.explore.legs import LegRunner, leg_timeout_s


class FakeNav:
    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), []

    async def status(self):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    async def plan_to_point(self, x, y):
        return True, 1.0

    async def navigate_to_point(self, x, y, theta):
        self.calls.append(("nav", x, y))

    async def go_to_point(self, x, y, theta, velocity_mps):
        self.calls.append(("go", x, y, velocity_mps))

    async def cancel(self):
        self.calls.append(("cancel",))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    async def sleep(self, s):
        self.t += s


def st(active, state, x=0.0, y=0.0, localized=True):
    return NavStatus(
        active=active,
        state=state,
        pose=Pose(x, y, 0.0),
        localized=localized,
        max_angular_rad_s=0.5,
        robot_radius_m=0.3,
        error="",
    )


def runner(nav, clk, **kw):
    return LegRunner(nav, max_linear_mps=0.4, clock=clk.now, sleep=clk.sleep, **kw)


def go(r, planned=True):
    return asyncio.run(r.run(2.0, 0.0, 0.0, planned=planned, planned_length_m=2.0))


def test_timeout_formula():
    assert leg_timeout_s(2.0, 0.4) == 20.0
    assert leg_timeout_s(40.0, 0.4) == 300.0


def test_leg_succeeds_when_nav_reports_succeeded_near_goal():
    nav, clk = FakeNav([st(True, "active"), st(False, "succeeded", 1.9, 0.1)]), FakeClock()
    r = go(runner(nav, clk))
    assert r.ok and r.end_pose.x == 1.9 and nav.calls[0][0] == "nav"


def test_leg_far_from_goal_fails():
    nav, clk = FakeNav([st(True, "active"), st(False, "succeeded", 0.0, 0.0)]), FakeClock()
    r = go(runner(nav, clk))
    assert not r.ok and r.reason == "far_from_goal"


def test_leg_times_out_and_cancels():
    nav, clk = FakeNav([st(True, "active")]), FakeClock()
    r = go(runner(nav, clk))
    assert not r.ok and r.reason == "timeout" and ("cancel",) in nav.calls and clk.t >= 20.0


def test_leg_lost_pose_cancels():
    nav, clk = FakeNav([st(True, "active"), st(True, "active", localized=False)]), FakeClock()
    r = go(runner(nav, clk))
    assert not r.ok and r.reason == "lost_pose" and ("cancel",) in nav.calls


def test_leg_failed_state_reports_failed():
    nav, clk = FakeNav([st(True, "active"), st(False, "failed")]), FakeClock()
    assert go(runner(nav, clk)).reason == "failed"


def test_should_stop_cancels_leg():
    nav, clk = FakeNav([st(True, "active")]), FakeClock()
    r = go(runner(nav, clk, should_stop=lambda: True))
    assert r.reason == "canceled" and ("cancel",) in nav.calls


def test_unplanned_leg_uses_go_to_point():
    nav, clk = FakeNav([st(True, "active"), st(False, "succeeded", 2.0, 0.0)]), FakeClock()
    r = go(runner(nav, clk), planned=False)
    assert r.ok and nav.calls[0] == ("go", 2.0, 0.0, 0.4)

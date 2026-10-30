import asyncio

from src.config import ExploreConfig
from src.models.explore import ExploreService

class Rec:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    async def do_command(self, command, **kw):
        self.calls.append(dict(command))
        return self.replies.get(command["command"], {})

    async def stop(self):
        self.calls.append({"command": "base_stop"})


def make(slam_mode="mapping", ang=0.5):
    svc = ExploreService("explore")
    svc._slam = Rec(
        {
            "get_status": {"mode": slam_mode, "running": True, "keyframes": 0, "loop_closure": {}},
            "get_grid": {"available": False},
        }
    )
    svc._nav = Rec(
        {
            "get_status": {
                "active": False,
                "state": "idle",
                "pose": {"x": 0, "y": 0, "theta": 0},
                "limits": {"max_linear_mps": 0.4, "max_angular_rad_s": ang},
            }
        }
    )
    svc._base = Rec({})
    svc._cfg = ExploreConfig.from_dict(
        {"slam_service": "slam", "navigation_service": "navigation", "base": "base"}
    )
    return svc


def test_status_idle_before_start():
    assert asyncio.run(make().do_command({"command": "status"})) == {"phase": "idle"}


def test_start_refuses_when_slam_not_mapping():
    svc = make(slam_mode="localizing")
    out = asyncio.run(svc.do_command({"command": "start", "duration_s": 10}))
    assert "mapping" in out["error"]
    assert not any(c["command"] in ("navigate_to_point", "go_to_point") for c in svc._nav.calls)


def test_stop_without_run_is_an_error():
    assert asyncio.run(make().do_command({"command": "stop"})) == {"error": "no active run"}


def test_start_twice_is_an_error_and_stop_ends_run():
    svc = make()

    async def go():
        a = await svc.do_command({"command": "start", "duration_s": 300})
        b = await svc.do_command({"command": "start", "duration_s": 300})
        s = await svc.do_command({"command": "stop"})
        await svc._run_task
        return a, b, s, await svc.do_command({"command": "status"})

    a, b, s, st = asyncio.run(go())
    assert a == {"started": True, "duration_s": 300.0}
    assert "already" in b["error"] and s == {"stopping": True}
    assert st["phase"] == "done"


def test_close_ends_active_run_and_stops_base():
    svc = make()

    async def go():
        await svc.do_command({"command": "start", "duration_s": 300})
        await svc.close()
        return await svc.do_command({"command": "status"})

    st = asyncio.run(go())
    assert st["phase"] == "done"
    assert any(c["command"] == "base_stop" for c in svc._base.calls)

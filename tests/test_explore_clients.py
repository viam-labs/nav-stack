import asyncio

from src.explore.clients import ViamNavClient, ViamSlamClient


class Recorder:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def do_command(self, command, **kw):
        self.calls.append(dict(command))
        return self.reply


def test_nav_status_parses_pose_limits_and_localized():
    raw = {
        "active": False,
        "state": "succeeded",
        "pose": {"x": 1.0, "y": 2.0, "theta": 0.5},
        "limits": {"max_linear_mps": 0.5, "max_angular_rad_s": 1.5},
        "robot_radius_m": 0.3,
        "error_msg": "",
    }
    st = asyncio.run(ViamNavClient(Recorder(raw)).status())
    assert st.pose.x == 1.0 and st.max_angular_rad_s == 1.5 and st.robot_radius_m == 0.3
    assert st.localized and st.state == "succeeded"


def test_nav_status_without_pose_is_not_localized():
    st = asyncio.run(ViamNavClient(Recorder({"active": False, "state": "idle"})).status())
    assert st.pose is None and not st.localized and st.max_angular_rad_s is None


def test_nav_status_untrusted_is_not_localized():
    raw = {
        "active": False,
        "state": "idle",
        "pose": {"x": 0, "y": 0, "theta": 0},
        "localization_check": {"status": "nav_hold"},
    }
    assert not asyncio.run(ViamNavClient(Recorder(raw)).status()).localized


def test_plan_to_point_returns_feasible_and_length():
    rec = Recorder({"status": "planned", "feasible": True, "length_m": 7.5})
    assert asyncio.run(ViamNavClient(rec).plan_to_point(1.0, 2.0)) == (True, 7.5)
    assert rec.calls[0]["command"] == "plan_to_point" and rec.calls[0]["x"] == 1.0


def test_go_to_point_sends_velocity_and_no_wait():
    rec = Recorder({"status": "ok"})
    asyncio.run(ViamNavClient(rec).go_to_point(1.0, 2.0, 0.0, velocity_mps=0.4))
    assert rec.calls[0] == {
        "command": "go_to_point",
        "x": 1.0,
        "y": 2.0,
        "theta": 0.0,
        "wait": False,
        "velocity_mps": 0.4,
    }


def test_slam_status_and_grid():
    raw = {
        "mode": "mapping",
        "running": True,
        "keyframes": 12,
        "loop_closure": {"graph_full": False, "closures": 2},
    }
    st = asyncio.run(ViamSlamClient(Recorder(raw)).status())
    assert (st.mode, st.keyframes, st.graph_full, st.closures) == ("mapping", 12, False, 2)
    none_grid = ViamSlamClient(Recorder({"available": False, "reason": "no_map"}))
    assert asyncio.run(none_grid.grid()) is None

import asyncio

import numpy as np

from src import runtime
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
    st = asyncio.run(
        ViamNavClient(Recorder({"active": False, "state": "idle"})).status()
    )
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


def test_slam_status_skips_sensor_probes():
    rec = Recorder({"mode": "mapping", "running": True, "keyframes": 12})
    st = asyncio.run(ViamSlamClient(rec).status())
    assert (st.mode, st.running, st.keyframes) == ("mapping", True, 12)
    assert rec.calls[0] == {"command": "get_status", "probe_sensors": False}


def test_occupancy_falls_back_to_get_grid():
    none_grid = ViamSlamClient(
        Recorder({"available": False, "reason": "no_map"}), "slam"
    )
    assert asyncio.run(none_grid.occupancy()) is None


def test_occupancy_reads_in_process_slam_map_without_docommand():
    class Node:
        def get_map(self):
            return {
                "grid": np.zeros((4, 5), dtype=np.int16),
                "resolution": 0.05,
                "origin_x": 0.0,
                "origin_y": 0.0,
                "generation": 1,
            }

    class Manager:
        node = Node()

    rec = Recorder({})
    runtime.register_slam(
        "explore-test-slam", runtime.SlamRuntime(Manager(), None, None)
    )
    try:
        g = asyncio.run(ViamSlamClient(rec, "explore-test-slam").occupancy())
    finally:
        runtime.unregister_slam("explore-test-slam")
    assert g.cells.shape == (4, 5) and rec.calls == []


def test_paint_hold_is_read_in_process_and_is_empty_for_slam_elsewhere():
    class Node:
        def paint_hold(self):
            return {"held_s": 7.5, "travel_m": 3.2, "verdict": "ambiguous"}

    class Manager:
        node = Node()

    rec = Recorder({})
    runtime.register_slam(
        "explore-test-slam", runtime.SlamRuntime(Manager(), None, None)
    )
    try:
        hold = asyncio.run(ViamSlamClient(rec, "explore-test-slam").paint_hold())
    finally:
        runtime.unregister_slam("explore-test-slam")
    assert (hold.held_s, hold.travel_m, hold.verdict) == (7.5, 3.2, "ambiguous")
    # SLAM in another process: no hold, and no status call on every tick.
    none = asyncio.run(ViamSlamClient(rec, "slam-elsewhere").paint_hold())
    assert none.held_s == 0.0 and rec.calls == []

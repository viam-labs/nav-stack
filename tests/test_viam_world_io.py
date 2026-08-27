"""Tests for ViamWorldIO + get_grid encode/decode (ROS-free nav I/O)."""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.nav_builtin.viam_io import (
    ViamWorldIO,
    bridge_map_to_get_grid,
    get_grid_response_to_map,
)
from src.nav_builtin.viz_store import NavVizStore
from src.ros import conversions as conv


def test_bridge_map_get_grid_roundtrip():
    grid = np.array([[0, 100], [-1, 50]], dtype=np.int16)
    mp = {
        "grid": grid,
        "resolution": 0.05,
        "origin_x": -1.0,
        "origin_y": -2.0,
    }
    payload = bridge_map_to_get_grid(mp)
    assert payload["rows"] == 2
    assert payload["cols"] == 2
    assert payload["cellSize"] == pytest.approx(0.05)
    back = get_grid_response_to_map(payload)
    assert back is not None
    assert back["origin_x"] == pytest.approx(-1.0)
    assert back["origin_y"] == pytest.approx(-2.0)
    np.testing.assert_array_equal(back["grid"], grid)


@pytest.mark.asyncio
async def test_viam_world_io_map_pose_drive():
    loop = asyncio.get_event_loop()
    grid = np.zeros((4, 4), dtype=np.int16)
    grid[1, 1] = 100
    payload = bridge_map_to_get_grid(
        {
            "grid": grid,
            "resolution": 0.1,
            "origin_x": 0.0,
            "origin_y": 0.0,
        }
    )

    slam = MagicMock()
    slam.do_command = AsyncMock(return_value=payload)
    pose = SimpleNamespace(
        x=1500.0, y=-500.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=90.0
    )
    slam.get_position = AsyncMock(return_value=pose)

    base = MagicMock()
    base.set_velocity = AsyncMock()

    viz = NavVizStore()
    world = ViamWorldIO(
        slam=slam,
        base=base,
        loop=loop,
        cameras={},
        lidars=[],
        base_velocity_convention="viam",
        viz=viz,
    )

    mp = await asyncio.to_thread(world.get_map)
    assert mp is not None
    assert mp["grid"][1, 1] == 100
    assert viz.get_map() is not None

    p2 = await asyncio.to_thread(world.get_pose)
    assert p2 is not None
    assert p2.x == pytest.approx(1.5)
    assert p2.y == pytest.approx(-0.5)
    assert p2.theta == pytest.approx(math.pi / 2)

    await asyncio.to_thread(world.set_velocity, 0.2, 0.0, 0.1)
    base.set_velocity.assert_awaited()
    args = base.set_velocity.await_args
    # viam convention: ROS vx -> Viam linear.y
    assert args.kwargs["linear"].y == pytest.approx(200.0)
    assert args.kwargs["angular"].z == pytest.approx(math.degrees(0.1))


def test_nav_viz_store_snapshot_shape():
    viz = NavVizStore()
    viz.set_map(
        {
            "grid": np.zeros((2, 2), dtype=np.int16),
            "resolution": 0.05,
            "origin_x": 0.0,
            "origin_y": 0.0,
        }
    )
    viz.set_plan(((0.0, 0.0), (1.0, 0.0)), goal=(1.0, 0.0, 0.0))
    viz.set_pose(conv.Pose2D(0.1, 0.2, 0.3))
    snap = viz.viz_snapshot()
    assert snap["map"] is not None
    assert len(snap["global_plan"]) == 2
    assert snap["goal"] == (1.0, 0.0, 0.0)
    assert snap["pose"] == (0.1, 0.2, 0.3)


def test_builtin_nav_host_status():
    from src.config import NavConfig
    from src.nav_builtin.host import BuiltinNavHost, make_builtin_navigator

    class _World:
        def get_map(self):
            return None

        def get_pose(self):
            return conv.Pose2D(0.0, 0.0, 0.0)

        def get_scan(self, max_age_s=2.0):
            return None

        def set_velocity(self, vx, vy, vtheta):
            pass

        def stop(self):
            pass

        def set_viz_plan(self, path_xy, goal=None):
            pass

        def set_viz_costmap(self, costmap):
            pass

    cfg = NavConfig.from_dict({"slam_service": "s", "base": "b"})
    world = _World()
    viz = NavVizStore()
    nav = make_builtin_navigator(world, cfg)
    host = BuiltinNavHost(nav, world, viz, nav_cfg=cfg)
    assert host.nav_backend() == "builtin"
    assert host.nav_action_ready() is True
    status = host.nav_status()
    assert status["nav_backend"] == "builtin"
    diag = host.nav2_diagnostics()
    assert diag["nav2_processes_running"] is False

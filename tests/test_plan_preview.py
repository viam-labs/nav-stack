"""Tests for Nav2 plan-without-execute helpers and DoCommands."""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ros.conversions import path_length_m, path_msg_to_points


def _pose(x, y, yaw=0.0):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
        )
    )


def test_path_msg_to_points_empty():
    assert path_msg_to_points(None) == []
    assert path_msg_to_points(SimpleNamespace(poses=[])) == []


def test_path_msg_to_points_keeps_yaw():
    path = SimpleNamespace(poses=[_pose(0.0, 0.0, 0.0), _pose(1.0, 0.0, math.pi / 2)])
    pts = path_msg_to_points(path)
    assert len(pts) == 2
    assert pts[0]["x"] == pytest.approx(0.0)
    assert pts[1]["x"] == pytest.approx(1.0)
    assert pts[1]["theta"] == pytest.approx(math.pi / 2, abs=1e-6)


def test_path_msg_to_points_downsamples_preserving_ends():
    poses = [_pose(float(i), 0.0) for i in range(100)]
    pts = path_msg_to_points(SimpleNamespace(poses=poses), max_points=10)
    assert len(pts) == 10
    assert pts[0]["x"] == pytest.approx(0.0)
    assert pts[-1]["x"] == pytest.approx(99.0)


def test_path_length_m():
    pts = [{"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}]
    assert path_length_m(pts) == pytest.approx(5.0)


def test_plan_to_point_do_command_returns_path_without_navigate():
    pytest.importorskip("viam")
    from src.config import NavConfig
    from src.models.navigation import RosNavigation

    nav = RosNavigation("nav")
    nav._cfg = NavConfig(
        slam_service="slam",
        base="my-base",
        kinematics="differential",
        robot_radius=0.22,
        max_vel_x=0.4,
        max_vel_theta=1.0,
        inflation_radius=0.45,
    )
    preview = {
        "feasible": True,
        "error_code": 0,
        "error_msg": "",
        "planner_id": "GridBased",
        "planning_time_s": 0.01,
        "length_m": 2.5,
        "path": [{"x": 0.0, "y": 0.0, "theta": 0.0}, {"x": 2.5, "y": 0.0, "theta": 0.0}],
        "goal": {"x": 2.5, "y": 0.0, "theta": 0.0},
        "start": None,
        "point_count": 2,
    }
    mgr = MagicMock()
    mgr.compute_path = MagicMock(return_value=preview)
    mgr.navigate = MagicMock()
    mgr.last_preview_plan = MagicMock(return_value=preview)
    runtime = SimpleNamespace(manager=mgr, localization_check={})
    nav._resolve_runtime = MagicMock(return_value=runtime)  # type: ignore[method-assign]

    result = asyncio.run(
        nav.do_command({"command": "plan_to_point", "x": 2.5, "y": 0.0, "theta": 0.0})
    )

    assert result["status"] == "planned"
    assert result["feasible"] is True
    assert result["path"][1]["x"] == pytest.approx(2.5)
    mgr.compute_path.assert_called_once()
    mgr.navigate.assert_not_called()
    assert nav._last_preview_plan["length_m"] == pytest.approx(2.5)


def test_execute_plan_navigates_preview_goal():
    pytest.importorskip("viam")
    from src.config import NavConfig
    from src.models.navigation import RosNavigation

    nav = RosNavigation("nav")
    nav._cfg = NavConfig(
        slam_service="slam",
        base="my-base",
        kinematics="differential",
        robot_radius=0.22,
        max_vel_x=0.4,
        max_vel_theta=1.0,
        inflation_radius=0.45,
    )
    preview = {
        "feasible": True,
        "error_code": 0,
        "goal": {"x": 1.0, "y": 2.0, "theta": 0.5},
        "length_m": 3.0,
        "path": [{"x": 0.0, "y": 0.0, "theta": 0.0}, {"x": 1.0, "y": 2.0, "theta": 0.5}],
    }
    nav._last_preview_plan = preview
    mgr = MagicMock()
    mgr.navigate = MagicMock()
    mgr.last_preview_plan = MagicMock(return_value=None)
    runtime = SimpleNamespace(manager=mgr, localization_check={})
    nav._resolve_runtime = MagicMock(return_value=runtime)  # type: ignore[method-assign]

    result = asyncio.run(nav.do_command({"command": "execute_plan"}))

    assert result["status"] == "navigating"
    assert result["from_preview"] is True
    mgr.navigate.assert_called_once_with(1.0, 2.0, 0.5)


def test_move_on_map_preview_extra_does_not_navigate():
    pytest.importorskip("viam")
    from grpclib.exceptions import GRPCError
    from viam.proto.common import Pose

    from src.config import NavConfig
    from src.models.navigation import RosNavigation

    nav = RosNavigation("nav")
    nav._cfg = NavConfig(
        slam_service="slam",
        base="my-base",
        kinematics="differential",
        robot_radius=0.22,
        max_vel_x=0.4,
        max_vel_theta=1.0,
        inflation_radius=0.45,
    )
    preview = {
        "feasible": True,
        "error_code": 0,
        "error_msg": "",
        "path": [{"x": 0.0, "y": 0.0, "theta": 0.0}, {"x": 1.0, "y": 0.0, "theta": 0.0}],
        "goal": {"x": 1.0, "y": 0.0, "theta": 0.0},
        "length_m": 1.0,
    }
    mgr = MagicMock()
    mgr.compute_path = MagicMock(return_value=preview)
    mgr.navigate = MagicMock()
    runtime = SimpleNamespace(manager=mgr, localization_check={})
    nav._resolve_runtime = MagicMock(return_value=runtime)  # type: ignore[method-assign]
    nav._base = MagicMock()

    dest = Pose(x=1000.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    exec_id = asyncio.run(
        nav.move_on_map("my-base", dest, "slam", extra={"preview": True})
    )
    assert str(exec_id).startswith("preview-")
    mgr.navigate.assert_not_called()
    mgr.compute_path.assert_called_once()

    bad = dict(preview, feasible=False, error_msg="blocked", error_code=208)
    mgr.compute_path = MagicMock(return_value=bad)
    with pytest.raises(GRPCError):
        asyncio.run(nav.move_on_map("my-base", dest, "slam", extra={"plan_only": True}))

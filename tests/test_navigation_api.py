"""Typed Navigation service: units and the methods that replace DoCommand."""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("viam")

from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.proto.common import Pose
from viam.services.navigation import GoalState, MapType, NavMode, Position

from src.config import NavConfig
from src.geom.conversions import Pose2D
from src.models.nav_api import (
    geometry_m_to_mm,
    geometry_mm_to_m,
    plan_from_preview,
    pose2d_to_position,
    position_to_pose2d,
    status_from_dict,
)
from src.models.navigation import NavigationService


def _configured(nav_status=None) -> tuple[NavigationService, MagicMock]:
    nav = NavigationService("nav")
    nav._cfg = NavConfig(
        slam_service="slam",
        base="my-base",
        kinematics="differential",
        robot_radius=0.22,
        max_linear_velocity=0.4,
        max_vel_theta=1.0,
        clearance_m=0.2,
    )
    mgr = MagicMock()
    mgr.navigate = MagicMock()
    mgr.cancel = MagicMock()
    mgr.nav_status = MagicMock(
        return_value=nav_status
        if nav_status is not None
        else {
            "active": False,
            "state": "idle",
            "pose": {"x": 1.0, "y": 2.0, "theta": 0.5},
            "motion": "builtin",
        }
    )
    runtime = SimpleNamespace(manager=mgr, localization_check={})
    nav._resolve_runtime = MagicMock(return_value=runtime)  # type: ignore[method-assign]
    nav._base = MagicMock()
    return nav, mgr


def test_position_roundtrip_is_millimeters_and_degrees():
    pose = Pose2D(3.5, -1.0, math.pi / 2)
    position = pose2d_to_position(pose)
    assert position.pose.x == pytest.approx(3500.0)
    assert position.pose.y == pytest.approx(-1000.0)
    assert position.pose.theta == pytest.approx(90.0)
    assert position.pose.o_z == pytest.approx(1.0)
    back = position_to_pose2d(position)
    assert back.x == pytest.approx(pose.x)
    assert back.y == pytest.approx(pose.y)
    assert back.theta == pytest.approx(pose.theta)


def test_zone_geometry_lengths_convert_at_the_api_boundary():
    stored = {"type": "circle", "center": [1.5, -0.25], "radius": 0.8}
    wire = geometry_m_to_mm(stored)
    assert wire["center"] == pytest.approx([1500.0, -250.0])
    assert wire["radius"] == pytest.approx(800.0)
    assert geometry_mm_to_m(wire)["radius"] == pytest.approx(0.8)


def test_status_maps_active_goal_and_velocity():
    status = status_from_dict(
        {
            "active": True,
            "state": "active",
            "motion": "builtin",
            "pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
            "goal": {"x": 3.0, "y": 4.0, "theta": 0.0, "name": "wait"},
            "path": [{"x": 1.0, "y": 2.0, "theta": 0.0}, {"x": 3.0, "y": 4.0, "theta": 0.0}],
            "length_m": 2.8,
            "distance_remaining_m": 2.8,
            "cmd_vx_mps": 0.2,
            "cmd_vtheta_rad_s": math.radians(10.0),
            "obstacle": "clear",
        }
    )
    assert status.localized is True
    assert status.goal.state == GoalState.GOAL_STATE_ACTIVE
    assert status.goal.mode == NavMode.NAV_MODE_PLANNED
    assert status.goal.location == "wait"
    assert status.goal.target.pose.x == pytest.approx(3000.0)
    assert status.velocity.linear_m_per_sec == pytest.approx(0.2)
    assert status.velocity.angular_deg_per_sec == pytest.approx(10.0)
    assert len(status.goal.progress.path) == 2


def test_untrusted_pose_is_not_localized():
    status = status_from_dict(
        {
            "active": False,
            "state": "idle",
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "map_pose_untrusted": True,
        }
    )
    assert status.localized is False


def test_infeasible_plan_raises():
    with pytest.raises(GRPCError) as exc:
        plan_from_preview({"feasible": False, "error_msg": "no feasible path"})
    assert exc.value.status == Status.FAILED_PRECONDITION


def test_navigate_to_point_uses_meters_and_radians():
    nav, mgr = _configured()
    dest = Position(pose=Pose(x=3500, y=-1000, z=0, o_x=0, o_y=0, o_z=1, theta=90))
    status = asyncio.run(nav.navigate_to_point(dest))
    mgr.navigate.assert_called_once()
    args = mgr.navigate.call_args.args
    assert args[0] == pytest.approx(3.5)
    assert args[1] == pytest.approx(-1.0)
    assert args[2] == pytest.approx(math.pi / 2)
    assert status.localized is True


def test_finished_goal_does_not_keep_the_untrusted_flag():
    """A timeout leaves progress.map_pose_untrusted set. The pose is still published."""
    status = status_from_dict(
        {
            "active": False,
            "state": "failed",
            "error_msg": "navigation timed out",
            "pose": {"x": -7.8, "y": 2.4, "theta": 0.9},
            "progress": {"map_pose_untrusted": True},
            "localization_check": {
                "status": "ambiguous",
                "corrected": False,
                "shift_m": 0.0,
                "shift_deg": 0.0,
                "ray_mae_m": 0.22,
                "score": 0.83,
            },
        }
    )
    assert status.localized is True


def test_ambiguous_check_still_has_a_pose():
    """A refused full-map peak keeps the published pose. Short nav can start."""
    status = status_from_dict(
        {
            "active": False,
            "state": "idle",
            "pose": {"x": -4.2, "y": -0.5, "theta": 1.4},
            "localization_check": {"status": "ambiguous", "score": 0.3},
        }
    )
    assert status.localized is True


def test_idle_status_fills_pose_from_slam():
    nav, mgr = _configured(
        {"active": False, "state": "idle", "pose": None, "motion": "builtin"}
    )
    mgr.get_pose_in_map.return_value = Pose2D(-4.2, -0.47, 1.38)
    status = asyncio.run(nav.do_command({"command": "get_status"}))
    assert status["pose"]["x"] == pytest.approx(-4.2)
    assert status["pose"]["y"] == pytest.approx(-0.47)
    assert status["pose"]["theta"] == pytest.approx(1.38)


def test_navigate_to_point_refuses_when_pose_is_missing():
    nav, mgr = _configured(
        {"active": False, "state": "idle", "pose": None, "motion": "builtin"}
    )
    mgr.get_pose_in_map.return_value = None
    mgr.node = None
    dest = Position(pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0))
    with pytest.raises(GRPCError) as exc:
        asyncio.run(nav.navigate_to_point(dest))
    assert exc.value.status == Status.FAILED_PRECONDITION
    mgr.navigate.assert_not_called()


def test_resource_get_status_matches_the_do_command_payload():
    nav, _mgr = _configured(
        {
            "active": True,
            "state": "active",
            "motion": "builtin",
            "pose": {"x": 1.0, "y": 2.0, "theta": 0.0},
            "goal": {"x": 3.0, "y": 4.0, "theta": 0.0, "name": "wait"},
        }
    )
    raw = asyncio.run(nav.get_status())
    assert raw["active"] is True
    assert raw["goal"]["name"] == "wait"
    assert raw["pose"]["x"] == pytest.approx(1.0)


def test_get_properties_is_a_local_map():
    nav, _mgr = _configured()
    props = asyncio.run(nav.get_properties())
    assert props.map_type == MapType.MAP_TYPE_LOCAL
    assert props.planning_supported is True
    assert props.routes_supported is True
    assert props.robot_radius_m == pytest.approx(0.22 + 0.2)

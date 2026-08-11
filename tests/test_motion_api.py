"""Unit tests for Motion MoveOnMap on NavServiceBase."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("viam")

from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import PlanState
from viam.services.motion import Motion

from src.config import NavConfig
from src.models.navigation import RosNavigation, _nav_status_to_plan_state
from src.ros.conversions import Pose2D, viam_pose_to_pose2d


def test_nav_status_to_plan_state_mapping():
    assert _nav_status_to_plan_state({"active": True}) == PlanState.PLAN_STATE_IN_PROGRESS
    assert (
        _nav_status_to_plan_state({"active": False, "state": "succeeded"})
        == PlanState.PLAN_STATE_SUCCEEDED
    )
    assert (
        _nav_status_to_plan_state({"active": False, "state": "canceled"})
        == PlanState.PLAN_STATE_STOPPED
    )
    assert (
        _nav_status_to_plan_state({"active": False, "state": "failed"})
        == PlanState.PLAN_STATE_FAILED
    )
    assert (
        _nav_status_to_plan_state({"active": False, "state": "aborted"})
        == PlanState.PLAN_STATE_FAILED
    )
    assert (
        _nav_status_to_plan_state({"active": False, "state": "idle"})
        == PlanState.PLAN_STATE_UNSPECIFIED
    )


def test_ros_navigation_registers_as_motion():
    assert RosNavigation.API == Motion.API
    assert issubclass(RosNavigation, Motion)


def _configured_nav(*, nav_status=None, pose=None) -> tuple[RosNavigation, MagicMock]:
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
    mgr = MagicMock()
    mgr.navigate = MagicMock()
    mgr.cancel = MagicMock()
    mgr.nav_status = MagicMock(
        return_value=nav_status
        if nav_status is not None
        else {"active": True, "state": "idle"}
    )
    mgr.get_pose_in_map = MagicMock(
        return_value=pose if pose is not None else Pose2D(1.0, 2.0, 0.5)
    )
    mgr.nav2_diagnostics = MagicMock(return_value={})
    runtime = SimpleNamespace(manager=mgr, localization_check={})
    nav._resolve_runtime = MagicMock(return_value=runtime)  # type: ignore[method-assign]
    nav._base = MagicMock()
    return nav, mgr


def test_move_on_map_calls_navigate_with_meters_radians():
    nav, mgr = _configured_nav()
    dest = Pose(x=3500.0, y=-1000.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=90.0)
    expected = viam_pose_to_pose2d(dest.x, dest.y, dest.theta)

    execution_id = asyncio.run(
        nav.move_on_map(
            component_name="my-base",
            destination=dest,
            slam_service_name="slam",
        )
    )

    assert isinstance(execution_id, str) and execution_id
    mgr.navigate.assert_called_once()
    args = mgr.navigate.call_args[0]
    assert args[0] == pytest.approx(expected.x)
    assert args[1] == pytest.approx(expected.y)
    assert args[2] == pytest.approx(expected.theta)

    plan = asyncio.run(nav.get_plan("my-base", execution_id=execution_id))
    assert plan.current_plan_with_status.plan.execution_id == execution_id
    assert (
        plan.current_plan_with_status.status.state == PlanState.PLAN_STATE_IN_PROGRESS
    )


def test_stop_plan_cancels_and_marks_stopped():
    nav, mgr = _configured_nav()
    dest = Pose(x=1000.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    execution_id = asyncio.run(
        nav.move_on_map("my-base", dest, "slam")
    )
    asyncio.run(nav.stop_plan("my-base"))
    mgr.cancel.assert_called()
    plan = asyncio.run(nav.get_plan("my-base", execution_id=execution_id))
    assert plan.current_plan_with_status.status.state == PlanState.PLAN_STATE_STOPPED


def test_get_plan_syncs_succeeded_from_nav_status():
    nav, mgr = _configured_nav(
        nav_status={"active": False, "state": "succeeded"}
    )
    dest = Pose(x=500.0, y=500.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    # After navigate returns, sync will see succeeded.
    execution_id = asyncio.run(nav.move_on_map("my-base", dest, "slam"))
    plan = asyncio.run(nav.get_plan("my-base", execution_id=execution_id))
    assert plan.current_plan_with_status.status.state == PlanState.PLAN_STATE_SUCCEEDED


def test_list_plan_statuses_filters_active():
    nav, mgr = _configured_nav(nav_status={"active": True, "state": "idle"})
    dest = Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    asyncio.run(nav.move_on_map("my-base", dest, "slam"))
    all_statuses = asyncio.run(nav.list_plan_statuses(only_active_plans=False))
    active = asyncio.run(nav.list_plan_statuses(only_active_plans=True))
    assert len(all_statuses) >= 1
    assert len(active) == 1
    assert active[0].status.state == PlanState.PLAN_STATE_IN_PROGRESS


def test_get_pose_returns_map_frame():
    nav, _mgr = _configured_nav(pose=Pose2D(1.5, -0.5, 0.25))
    pose_in_frame = asyncio.run(nav.get_pose("my-base", "map"))
    assert isinstance(pose_in_frame, PoseInFrame)
    assert pose_in_frame.reference_frame == "map"
    assert pose_in_frame.pose.x == pytest.approx(1500.0)
    assert pose_in_frame.pose.y == pytest.approx(-500.0)


def test_move_and_move_on_globe_unimplemented():
    nav, _mgr = _configured_nav()

    with pytest.raises(GRPCError) as move_exc:
        asyncio.run(
            nav.move(
                "my-base",
                PoseInFrame(reference_frame="map", pose=Pose()),
            )
        )
    assert move_exc.value.status == Status.UNIMPLEMENTED

    with pytest.raises(GRPCError) as globe_exc:
        from viam.proto.common import GeoPoint

        asyncio.run(
            nav.move_on_globe(
                "my-base",
                GeoPoint(latitude=0.0, longitude=0.0),
                "gps",
            )
        )
    assert globe_exc.value.status == Status.UNIMPLEMENTED


def test_do_command_navigate_to_point_still_works():
    nav, mgr = _configured_nav()
    result = asyncio.run(
        nav.do_command({"command": "navigate_to_point", "x": 3.5, "y": -1.0, "theta": 0.1})
    )
    assert result["status"] == "navigating"
    mgr.navigate.assert_called_once_with(3.5, -1.0, 0.1)


def test_do_command_cancel_marks_motion_plan_stopped():
    nav, mgr = _configured_nav()
    dest = Pose(x=1000.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    execution_id = asyncio.run(nav.move_on_map("my-base", dest, "slam"))
    asyncio.run(nav.do_command({"command": "cancel"}))
    plan = asyncio.run(nav.get_plan("my-base", execution_id=execution_id))
    assert plan.current_plan_with_status.status.state == PlanState.PLAN_STATE_STOPPED
    mgr.cancel.assert_called()


def test_suspend_resume_move_on_map():
    nav, mgr = _configured_nav(
        nav_status={
            "active": True,
            "state": "active",
            "goal": {"x": 1.0, "y": 0.0, "theta": 0.0},
        }
    )
    dest = Pose(x=1000.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    execution_id = asyncio.run(nav.move_on_map("my-base", dest, "slam"))

    suspended = asyncio.run(
        nav.do_command({"command": "suspend", "reason": "safety"})
    )
    assert suspended["status"] == "suspended"
    assert suspended["goal"]["x"] == pytest.approx(1.0)
    assert suspended["goal"]["y"] == pytest.approx(0.0)
    assert suspended["goal"]["motion"] == "nav2"
    assert suspended["goal"]["reason"] == "safety"
    mgr.cancel.assert_called()
    plan = asyncio.run(nav.get_plan("my-base", execution_id=execution_id))
    assert plan.current_plan_with_status.status.state == PlanState.PLAN_STATE_STOPPED
    assert plan.current_plan_with_status.status.reason == "suspended"

    status = asyncio.run(nav.do_command({"command": "get_status"}))
    assert status["suspended"] is True
    assert status["suspended_goal"]["x"] == pytest.approx(1.0)

    mgr.navigate.reset_mock()
    mgr.nav_status.return_value = {"active": True, "state": "active", "goal": {"x": 1.0, "y": 0.0, "theta": 0.0}}
    resumed = asyncio.run(nav.do_command({"command": "resume"}))
    assert resumed["status"] == "navigating"
    assert resumed["resumed"] is True
    assert "execution_id" in resumed
    mgr.navigate.assert_called_once_with(1.0, 0.0, 0.0)

    status = asyncio.run(nav.do_command({"command": "get_status"}))
    assert status["suspended"] is False
    assert status["suspended_goal"] is None


def test_suspend_when_idle_raises():
    nav, _mgr = _configured_nav(nav_status={"active": False, "state": "idle"})
    with pytest.raises(ValueError, match="nothing to suspend"):
        asyncio.run(nav.do_command({"command": "suspend"}))


def test_resume_when_not_suspended_raises():
    nav, _mgr = _configured_nav()
    with pytest.raises(ValueError, match="nothing to resume"):
        asyncio.run(nav.do_command({"command": "resume"}))


def test_cancel_clears_suspended_goal():
    nav, mgr = _configured_nav(
        nav_status={
            "active": True,
            "state": "active",
            "goal": {"x": 2.0, "y": 3.0, "theta": 0.5},
        }
    )
    asyncio.run(
        nav.do_command({"command": "navigate_to_point", "x": 2.0, "y": 3.0, "theta": 0.5})
    )
    asyncio.run(nav.do_command({"command": "suspend"}))
    assert nav._suspended is not None
    asyncio.run(nav.do_command({"command": "cancel"}))
    assert nav._suspended is None
    with pytest.raises(ValueError, match="nothing to resume"):
        asyncio.run(nav.do_command({"command": "resume"}))


def test_suspend_already_suspended_is_idempotent():
    nav, mgr = _configured_nav(
        nav_status={
            "active": True,
            "state": "active",
            "goal": {"x": 1.5, "y": -0.5, "theta": 0.1},
        }
    )
    asyncio.run(nav.do_command({"command": "navigate_to_point", "x": 1.5, "y": -0.5, "theta": 0.1}))
    first = asyncio.run(nav.do_command({"command": "suspend", "reason": "a"}))
    mgr.nav_status.return_value = {"active": False, "state": "canceled"}
    second = asyncio.run(nav.do_command({"command": "suspend", "reason": "b"}))
    assert second["already_suspended"] is True
    assert second["goal"]["reason"] == "b"
    assert first["goal"]["x"] == pytest.approx(1.5)

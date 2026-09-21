"""Lightweight navigation model smoke tests (builtin path)."""
import asyncio

import pytest

pytest.importorskip("viam")

from src.models.navigation import NavigationService


def test_do_command_raises_when_unconfigured():
    nav = NavigationService("nav")
    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(nav.do_command({"command": "get_status"}))


def test_refresh_zone_masks_raises_when_unconfigured():
    nav = NavigationService("nav")
    with pytest.raises(RuntimeError, match="not configured"):
        nav._refresh_zone_masks()


@pytest.mark.asyncio
async def test_verify_pose_between_route_legs_calls_slam_check(monkeypatch):
    """Between waypoints, nav should ask SLAM to check_localization."""
    from unittest.mock import AsyncMock

    from src.config import BuiltinNavConfig, NavConfig
    from src.runtime import register_slam_service, unregister_slam_service

    nav = NavigationService("nav")
    nav._cfg = NavConfig(
        slam_service="slam-test-verify",
        base="base",
        builtin=BuiltinNavConfig(route_verify_pose=True),
    )
    slam = AsyncMock()
    slam.do_command = AsyncMock(
        return_value={"status": "ok", "corrected": False, "score": 0.8}
    )
    register_slam_service("slam-test-verify", slam)
    try:
        out = await nav._verify_pose_between_route_legs()
        assert out["status"] == "ok"
        slam.do_command.assert_awaited_with(
            {
                "command": "check_localization",
                "full_map_escalation": "still_bad",
            }
        )
    finally:
        unregister_slam_service("slam-test-verify")


@pytest.mark.asyncio
async def test_verify_pose_between_route_legs_can_disable():
    from src.config import BuiltinNavConfig, NavConfig

    nav = NavigationService("nav")
    nav._cfg = NavConfig(
        slam_service="slam",
        base="base",
        builtin=BuiltinNavConfig(route_verify_pose=False),
    )
    out = await nav._verify_pose_between_route_legs()
    assert out["status"] == "skipped"
    assert out["reason"] == "disabled"

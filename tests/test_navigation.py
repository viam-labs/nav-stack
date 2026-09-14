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

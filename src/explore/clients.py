"""Thin async clients over the SLAM and navigation services for exploration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple

from ..models.nav_api import _localized
from ..runtime import get_slam
from .grid import OccGrid


@dataclass
class Pose:
    x: float
    y: float
    theta: float


@dataclass
class NavStatus:
    active: bool
    state: str
    pose: Optional[Pose]
    localized: bool
    max_angular_rad_s: Optional[float]
    robot_radius_m: Optional[float]
    error: str


@dataclass
class SlamStatus:
    mode: str
    running: bool
    keyframes: int


@dataclass
class PaintHold:
    """SLAM's report that scans have stopped fitting the map.

    ``held_s`` is 0 while scans are being painted. ``verdict`` is what SLAM's
    own position search last concluded during this hold (empty before the
    first search). It is reported, not acted on: only painting again counts
    as being back on the map.
    """

    held_s: float = 0.0
    travel_m: float = 0.0
    verdict: str = ""


class NavClient(Protocol):
    async def status(self) -> NavStatus: ...

    async def plan_to_point(self, x: float, y: float) -> Tuple[bool, float]: ...

    async def navigate_to_point(self, x: float, y: float, theta: float) -> None: ...

    async def cancel(self) -> None: ...


class SlamClient(Protocol):
    async def occupancy(self) -> Optional[OccGrid]: ...

    async def status(self) -> SlamStatus: ...

    async def paint_hold(self) -> PaintHold: ...

    async def optimize(self) -> None: ...

    async def save_map(self) -> None: ...


def localized_from_status(raw: Mapping[str, Any]) -> bool:
    return _localized(raw)


def _pose_from(raw: Any) -> Optional[Pose]:
    if not isinstance(raw, Mapping) or "x" not in raw:
        return None
    return Pose(float(raw["x"]), float(raw["y"]), float(raw.get("theta", 0.0)))


def _optf(v: Any) -> Optional[float]:
    return None if v is None else float(v)


class ViamNavClient:
    def __init__(self, resource):
        self._res = resource

    async def status(self) -> NavStatus:
        raw = await self._res.do_command({"command": "get_status"})
        limits = raw.get("limits") if isinstance(raw.get("limits"), Mapping) else {}
        return NavStatus(
            active=bool(raw.get("active", False)),
            state=str(raw.get("state", "")),
            pose=_pose_from(raw.get("pose")),
            localized=localized_from_status(raw),
            max_angular_rad_s=_optf(limits.get("max_angular_rad_s")),
            robot_radius_m=_optf(raw.get("robot_radius_m")),
            error=str(raw.get("error_msg") or raw.get("error") or ""),
        )

    async def plan_to_point(self, x: float, y: float) -> Tuple[bool, float]:
        raw = await self._res.do_command({"command": "plan_to_point", "x": x, "y": y})
        return bool(raw.get("feasible", False)), float(raw.get("length_m") or 0.0)

    async def navigate_to_point(self, x: float, y: float, theta: float) -> None:
        await self._res.do_command(
            {"command": "navigate_to_point", "x": x, "y": y, "theta": theta}
        )

    async def cancel(self) -> None:
        await self._res.do_command({"command": "cancel"})


class ViamSlamClient:
    def __init__(self, resource, name: Optional[str] = None):
        self._res = resource
        self._name = name

    async def occupancy(self) -> Optional[OccGrid]:
        """The live map, read in-process when SLAM runs in this module.

        Falls back to the ``get_grid`` DoCommand (the whole grid, encoded) when
        the SLAM service lives in another process.
        """
        runtime = get_slam(self._name) if self._name else None
        node = getattr(getattr(runtime, "manager", None), "node", None)
        if node is not None and callable(getattr(node, "get_map", None)):
            return OccGrid.from_map_dict(await asyncio.to_thread(node.get_map))
        raw = await self._res.do_command({"command": "get_grid"})
        if not raw or raw.get("available") is False:
            return None
        return await asyncio.to_thread(OccGrid.from_get_grid, raw)

    async def status(self) -> SlamStatus:
        raw = await self._res.do_command(
            {"command": "get_status", "probe_sensors": False}
        )
        return SlamStatus(
            mode=str(raw.get("mode", "")),
            running=bool(raw.get("running", False)),
            keyframes=int(raw.get("keyframes") or 0),
        )

    async def paint_hold(self) -> PaintHold:
        """Whether scans have stopped fitting the map (builtin SLAM in this module).

        Read in-process: it is asked every tick, and ``get_status`` is a large
        reply. A SLAM service in another process reports no hold.
        """
        runtime = get_slam(self._name) if self._name else None
        node = getattr(getattr(runtime, "manager", None), "node", None)
        read = getattr(node, "paint_hold", None)
        if not callable(read):
            return PaintHold()
        raw = read() or {}
        return PaintHold(
            held_s=float(raw.get("held_s") or 0.0),
            travel_m=float(raw.get("travel_m") or 0.0),
            verdict=str(raw.get("verdict") or ""),
        )

    async def optimize(self) -> None:
        await self._res.do_command({"command": "optimize"})

    async def save_map(self) -> None:
        await self._res.do_command({"command": "save_map"})

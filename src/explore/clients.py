"""Thin async clients over the SLAM and navigation services for exploration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple

from ..models.nav_api import _localized


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
    simple_state: str = ""


@dataclass
class SlamStatus:
    mode: str
    running: bool
    keyframes: int
    graph_full: bool
    closures: int


class NavClient(Protocol):
    async def status(self) -> NavStatus: ...

    async def plan_to_point(self, x: float, y: float) -> Tuple[bool, float]: ...

    async def navigate_to_point(self, x: float, y: float, theta: float) -> None: ...

    async def go_to_point(self, x: float, y: float, theta: float, velocity_mps: float) -> None: ...

    async def cancel(self) -> None: ...


class SlamClient(Protocol):
    async def grid(self) -> Optional[Mapping]: ...

    async def status(self) -> SlamStatus: ...

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
        simple = raw.get("simple_nav") if isinstance(raw.get("simple_nav"), Mapping) else {}
        return NavStatus(
            active=bool(raw.get("active", False)),
            state=str(raw.get("state", "")),
            pose=_pose_from(raw.get("pose")),
            localized=localized_from_status(raw),
            max_angular_rad_s=_optf(limits.get("max_angular_rad_s")),
            robot_radius_m=_optf(raw.get("robot_radius_m")),
            error=str(raw.get("error_msg") or raw.get("error") or ""),
            simple_state=str(simple.get("state", "")),
        )

    async def plan_to_point(self, x: float, y: float) -> Tuple[bool, float]:
        raw = await self._res.do_command({"command": "plan_to_point", "x": x, "y": y})
        return bool(raw.get("feasible", False)), float(raw.get("length_m") or 0.0)

    async def navigate_to_point(self, x: float, y: float, theta: float) -> None:
        await self._res.do_command(
            {"command": "navigate_to_point", "x": x, "y": y, "theta": theta}
        )

    async def go_to_point(self, x: float, y: float, theta: float, velocity_mps: float) -> None:
        await self._res.do_command(
            {
                "command": "go_to_point",
                "x": x,
                "y": y,
                "theta": theta,
                "wait": False,
                "velocity_mps": velocity_mps,
            }
        )

    async def cancel(self) -> None:
        await self._res.do_command({"command": "cancel"})


class ViamSlamClient:
    def __init__(self, resource):
        self._res = resource

    async def grid(self) -> Optional[Mapping]:
        raw = await self._res.do_command({"command": "get_grid"})
        if not raw or raw.get("available") is False:
            return None
        return raw

    async def status(self) -> SlamStatus:
        raw = await self._res.do_command({"command": "get_status"})
        loop = raw.get("loop_closure") if isinstance(raw.get("loop_closure"), Mapping) else {}
        return SlamStatus(
            mode=str(raw.get("mode", "")),
            running=bool(raw.get("running", False)),
            keyframes=int(raw.get("keyframes") or 0),
            graph_full=bool(loop.get("graph_full", False)),
            closures=int(loop.get("closures") or 0),
        )

    async def optimize(self) -> None:
        await self._res.do_command({"command": "optimize"})

    async def save_map(self) -> None:
        await self._res.do_command({"command": "save_map"})

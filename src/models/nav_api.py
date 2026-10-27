"""Typed Navigation service methods (``rdk:service:navigation``).

Wire format follows the reworked Navigation API: local positions are
``common.Pose`` values in millimeters and degrees, zone geometry lengths are
millimeters, and speed limits are meters/sec. Internals stay meters and
radians. ``DoCommand`` is unchanged and still uses meters and radians.
"""
from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.proto.common import Pose
from viam.services.navigation import (
    GetCostmapResponse,
    GoalState,
    GoalStatus,
    Location,
    MapType,
    NavError,
    NavErrorCode,
    NavMode,
    Plan,
    Position,
    Progress,
    Properties,
    Route,
    RouteProgress,
    Status as NavStatus,
    Velocity,
    Waypoint,
    Zone,
)
from viam.utils import dict_to_struct

from ..geom import conversions as conv

_MM_PER_M = 1000.0
_UNTRUSTED_LOC = frozenset(
    {"refused_large", "nav_hold", "awaiting_confirm", "error", "ambiguous"}
)


def pose2d_to_position(pose: conv.Pose2D) -> Position:
    x_mm, y_mm, z_mm, o_x, o_y, o_z, theta_deg = conv.pose2d_to_viam_slam_pose(pose)
    return Position(
        pose=Pose(
            x=x_mm, y=y_mm, z=z_mm, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta_deg
        )
    )


def position_to_pose2d(position: Position) -> conv.Pose2D:
    if position.WhichOneof("value") != "pose":
        raise GRPCError(
            Status.INVALID_ARGUMENT,
            "this navigation service uses a local map; pass a Pose, not a GeoPose",
        )
    pose = position.pose
    return conv.viam_pose_to_pose2d(pose.x, pose.y, pose.theta)


def _scale_geometry(geometry: Mapping[str, Any], factor: float) -> dict:
    out = dict(geometry)
    if "center" in out:
        out["center"] = [float(v) * factor for v in out["center"]]
    if "radius" in out:
        out["radius"] = float(out["radius"]) * factor
    if "size" in out:
        out["size"] = [float(v) * factor for v in out["size"]]
    if "points" in out:
        out["points"] = [
            [float(p[0]) * factor, float(p[1]) * factor] for p in out["points"]
        ]
    return out


def geometry_mm_to_m(geometry: Mapping[str, Any]) -> dict:
    return _scale_geometry(geometry, 1.0 / _MM_PER_M)


def geometry_m_to_mm(geometry: Mapping[str, Any]) -> dict:
    return _scale_geometry(geometry, _MM_PER_M)


def _waypoint_names(waypoints: Optional[Sequence[Waypoint]]) -> List[str]:
    names: List[str] = []
    for wp in waypoints or []:
        if wp.WhichOneof("value") != "location" or not wp.location:
            raise GRPCError(
                Status.INVALID_ARGUMENT,
                "saved routes store location names; pass inline positions to NavigateRoute",
            )
        names.append(wp.location)
    return names


def _speed_pct(max_speed_mps: float, max_linear_mps: float) -> float:
    if max_linear_mps <= 0:
        raise GRPCError(
            Status.FAILED_PRECONDITION,
            "max_linear_velocity is not configured; cannot convert max_speed",
        )
    pct = 100.0 * float(max_speed_mps) / float(max_linear_mps)
    if not (0 < pct <= 100.0):
        raise GRPCError(
            Status.INVALID_ARGUMENT,
            f"max_speed {max_speed_mps} m/s is outside (0, {max_linear_mps}] "
            "configured max_linear_velocity",
        )
    return pct


def _error_code(message: str) -> NavErrorCode:
    text = message.lower()
    if "no feasible" in text or "no path" in text:
        return NavErrorCode.NAV_ERROR_CODE_NO_PATH
    if "timed out" in text or "timeout" in text:
        return NavErrorCode.NAV_ERROR_CODE_TIMEOUT
    if "localiz" in text or "pose" in text:
        return NavErrorCode.NAV_ERROR_CODE_LOCALIZATION_LOST
    if "block" in text:
        return NavErrorCode.NAV_ERROR_CODE_BLOCKED
    if "base" in text:
        return NavErrorCode.NAV_ERROR_CODE_BASE_ERROR
    if message:
        return NavErrorCode.NAV_ERROR_CODE_OTHER
    return NavErrorCode.NAV_ERROR_CODE_UNSPECIFIED


def _localized(status: Mapping[str, Any]) -> bool:
    pose = status.get("pose")
    if not isinstance(pose, dict) or "x" not in pose:
        return False
    progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
    if status.get("map_pose_untrusted") or progress.get("map_pose_untrusted"):
        return False
    check = status.get("localization_check")
    if isinstance(check, dict) and check.get("status") in _UNTRUSTED_LOC:
        return False
    return True


def _position_from_xy(raw: Mapping[str, Any]) -> Position:
    return pose2d_to_position(
        conv.Pose2D(
            float(raw["x"]),
            float(raw["y"]),
            float(raw.get("theta", 0.0)),
        )
    )


def _path_positions(points: Any) -> List[Position]:
    if not isinstance(points, list):
        return []
    out: List[Position] = []
    for point in points:
        if isinstance(point, dict) and "x" in point and "y" in point:
            out.append(_position_from_xy(point))
    return out


def _nav_mode(status: Mapping[str, Any]) -> NavMode:
    motion = str(status.get("motion") or "")
    route = status.get("route") if isinstance(status.get("route"), dict) else {}
    simple = status.get("simple_nav") if isinstance(status.get("simple_nav"), dict) else {}
    suspended = (
        status.get("suspended_goal")
        if isinstance(status.get("suspended_goal"), dict)
        else {}
    )
    if status.get("suspended") and suspended.get("route"):
        return NavMode.NAV_MODE_ROUTE
    if status.get("suspended") and suspended.get("motion") == "simple":
        return NavMode.NAV_MODE_DIRECT
    if motion == "route" or route.get("state") == "active":
        return NavMode.NAV_MODE_ROUTE
    if motion == "simple" or simple.get("state") == "active":
        return NavMode.NAV_MODE_DIRECT
    return NavMode.NAV_MODE_PLANNED


def _goal_state(status: Mapping[str, Any]) -> GoalState:
    if status.get("suspended"):
        return GoalState.GOAL_STATE_SUSPENDED
    if status.get("active"):
        return GoalState.GOAL_STATE_ACTIVE
    state = str(status.get("state") or "").lower()
    if state == "succeeded":
        return GoalState.GOAL_STATE_SUCCEEDED
    if state in ("failed", "aborted", "rejected"):
        return GoalState.GOAL_STATE_FAILED
    if state in ("canceled", "cancelled"):
        return GoalState.GOAL_STATE_CANCELED
    return GoalState.GOAL_STATE_UNSPECIFIED


def _route_progress(route: Mapping[str, Any]) -> RouteProgress:
    waypoints: List[Waypoint] = []
    for item in route.get("waypoints") or []:
        if not isinstance(item, dict):
            continue
        if item.get("location"):
            waypoints.append(Waypoint(location=str(item["location"])))
        elif "x" in item and "y" in item:
            waypoints.append(Waypoint(position=_position_from_xy(item)))
    name = str(route.get("name") or "")
    if name == "inline":
        name = ""
    return RouteProgress(
        name=name,
        waypoints=waypoints,
        index=int(route.get("index") or 0),
        loop=bool(route.get("loop")),
        lap=int(route.get("lap") or 0),
    )


def status_from_dict(status: Mapping[str, Any]) -> NavStatus:
    """Map a ``get_status`` payload onto a Navigation ``Status``."""
    from datetime import datetime, timezone

    from google.protobuf.timestamp_pb2 import Timestamp

    msg = NavStatus(localized=_localized(status), updated_at=Timestamp())
    msg.updated_at.FromDatetime(datetime.now(timezone.utc))

    pose = status.get("pose")
    if isinstance(pose, dict) and "x" in pose:
        msg.position.CopyFrom(_position_from_xy(pose))

    vx = float(status.get("cmd_vx_mps") or 0.0)
    vtheta = float(status.get("cmd_vtheta_rad_s") or 0.0)
    msg.velocity.CopyFrom(
        Velocity(
            linear_m_per_sec=vx,
            angular_deg_per_sec=math.degrees(vtheta),
        )
    )

    state = _goal_state(status)
    if state == GoalState.GOAL_STATE_UNSPECIFIED:
        extra = _status_extra(status)
        if extra is not None:
            msg.extra.CopyFrom(extra)
        return msg

    goal = GoalStatus(state=state, mode=_nav_mode(status))
    target = status.get("goal") if isinstance(status.get("goal"), dict) else None
    suspended = (
        status.get("suspended_goal")
        if isinstance(status.get("suspended_goal"), dict)
        else None
    )
    if target is None and suspended is not None and "x" in suspended:
        target = suspended
    if isinstance(target, dict) and "x" in target and "y" in target:
        goal.target.CopyFrom(_position_from_xy(target))
    name = None
    if isinstance(target, dict) and target.get("name"):
        name = str(target["name"])
    elif suspended and suspended.get("name"):
        name = str(suspended["name"])
    if name:
        goal.location = name

    if state == GoalState.GOAL_STATE_ACTIVE:
        progress = Progress(
            distance_remaining_m=float(status.get("distance_remaining_m") or 0.0),
            length_m=float(status.get("length_m") or 0.0),
            path=_path_positions(status.get("path")),
        )
        goal.progress.CopyFrom(progress)
        route = status.get("route") if isinstance(status.get("route"), dict) else {}
        if goal.mode == NavMode.NAV_MODE_ROUTE and route:
            goal.route.CopyFrom(_route_progress(route))
    if state == GoalState.GOAL_STATE_SUSPENDED:
        reason = (suspended or {}).get("reason") or ""
        if reason:
            goal.suspend_reason = str(reason)
        route = status.get("route") if isinstance(status.get("route"), dict) else {}
        if goal.mode == NavMode.NAV_MODE_ROUTE and (route or suspended):
            source = route if route.get("waypoints") else {
                "name": (suspended or {}).get("route") or "",
                "index": (suspended or {}).get("route_index") or 0,
                "loop": (suspended or {}).get("loop") or False,
                "waypoints": (suspended or {}).get("route_waypoints") or [],
            }
            goal.route.CopyFrom(_route_progress(source))
    if state == GoalState.GOAL_STATE_FAILED:
        message = str(status.get("error_msg") or "navigation failed")
        goal.error.CopyFrom(NavError(code=_error_code(message), message=message))

    msg.goal.CopyFrom(goal)
    extra = _status_extra(status)
    if extra is not None:
        msg.extra.CopyFrom(extra)
    return msg


def _status_extra(status: Mapping[str, Any]):
    extra: Dict[str, Any] = {}
    if status.get("obstacle"):
        extra["obstacle"] = status["obstacle"]
    check = status.get("localization_check")
    if isinstance(check, dict) and check.get("status"):
        extra["localization"] = str(check["status"])
    if not extra:
        return None
    return dict_to_struct(extra)


def plan_from_preview(preview: Mapping[str, Any], location: Optional[str] = None) -> Plan:
    if not preview.get("feasible"):
        raise GRPCError(
            Status.FAILED_PRECONDITION,
            str(preview.get("error_msg") or "no feasible path"),
        )
    points = preview.get("path") or preview.get("points") or []
    msg = Plan(
        path=_path_positions(points),
        length_m=float(preview.get("length_m") or 0.0),
    )
    if location:
        msg.location = location
    return msg


class NavigationApi:
    """Navigation RPCs implemented on top of the existing DoCommand handlers."""

    async def _nav_call(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return await self.do_command(command)  # type: ignore[attr-defined]
        except GRPCError:
            raise
        except KeyError as exc:
            raise GRPCError(Status.NOT_FOUND, str(exc).strip("'")) from exc
        except ValueError as exc:
            raise GRPCError(Status.INVALID_ARGUMENT, str(exc)) from exc
        except RuntimeError as exc:
            raise GRPCError(Status.FAILED_PRECONDITION, str(exc)) from exc

    def _max_linear_mps(self) -> float:
        cfg = self._require_cfg()  # type: ignore[attr-defined]
        return float(cfg.max_linear_velocity)

    async def _require_localized(self) -> None:
        status = await self.get_navigation_status()
        if not status.localized:
            raise GRPCError(
                Status.FAILED_PRECONDITION,
                "not localized; a new goal cannot start",
            )

    async def add_location(
        self,
        location_name: str,
        position: Optional[Position] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Location:
        command: Dict[str, Any] = {"command": "add_location", "name": location_name}
        if position is not None:
            pose = position_to_pose2d(position)
            command["x"] = pose.x
            command["y"] = pose.y
            command["theta"] = pose.theta
        result = await self._nav_call(command)
        loc = result["location"]
        return Location(
            name=str(loc["name"]),
            position=_position_from_xy(loc),
        )

    async def get_location(
        self,
        location_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Location:
        result = await self._nav_call({"command": "get_location", "name": location_name})
        loc = result["location"]
        return Location(name=str(loc["name"]), position=_position_from_xy(loc))

    async def list_locations(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Location]:
        result = await self._nav_call({"command": "list_locations"})
        return [
            Location(name=str(loc["name"]), position=_position_from_xy(loc))
            for loc in result.get("locations") or []
        ]

    async def update_location(
        self,
        location_name: str,
        position: Optional[Position] = None,
        new_name: Optional[str] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Location:
        command: Dict[str, Any] = {"command": "update_location", "name": location_name}
        if position is not None:
            pose = position_to_pose2d(position)
            command["x"] = pose.x
            command["y"] = pose.y
            command["theta"] = pose.theta
        if new_name is not None:
            command["new_name"] = new_name
        result = await self._nav_call(command)
        loc = result["location"]
        return Location(name=str(loc["name"]), position=_position_from_xy(loc))

    async def delete_location(
        self,
        location_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        await self._nav_call({"command": "delete_location", "name": location_name})

    async def add_route(
        self,
        route_name: str,
        waypoints: Optional[List[Waypoint]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Route:
        names = _waypoint_names(waypoints)
        result = await self._nav_call(
            {
                "command": "add_route",
                "name": route_name,
                "waypoints": [{"location": name} for name in names],
            }
        )
        route = result["route"]
        return Route(
            name=str(route["name"]),
            waypoints=[
                Waypoint(location=str(wp.get("location") or wp))
                if not isinstance(wp, dict) or wp.get("location")
                else Waypoint(position=_position_from_xy(wp))
                for wp in route.get("waypoints") or []
            ],
        )

    async def get_route(
        self,
        route_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Route:
        result = await self._nav_call({"command": "get_route", "name": route_name})
        route = result["route"]
        return Route(
            name=str(route["name"]),
            waypoints=[
                Waypoint(location=str(wp["location"]))
                if isinstance(wp, dict) and wp.get("location")
                else Waypoint(position=_position_from_xy(wp))
                for wp in route.get("waypoints") or []
                if isinstance(wp, dict)
            ],
        )

    async def list_routes(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Route]:
        result = await self._nav_call({"command": "list_routes"})
        routes = []
        for route in result.get("routes") or []:
            routes.append(
                Route(
                    name=str(route["name"]),
                    waypoints=[
                        Waypoint(location=str(wp["location"]))
                        for wp in route.get("waypoints") or []
                        if isinstance(wp, dict) and wp.get("location")
                    ],
                )
            )
        return routes

    async def update_route(
        self,
        route_name: str,
        new_name: Optional[str] = None,
        waypoints: Optional[List[Waypoint]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Route:
        current = route_name
        if new_name is not None:
            renamed = await self._nav_call(
                {"command": "update_route", "name": route_name, "new_name": new_name}
            )
            current = str(renamed["route"]["name"])
        if waypoints is not None:
            names = _waypoint_names(waypoints)
            await self._nav_call(
                {
                    "command": "set_waypoints",
                    "route": current,
                    "waypoints": [{"location": name} for name in names],
                }
            )
        return await self.get_route(current)

    async def delete_route(
        self,
        route_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        await self._nav_call({"command": "delete_route", "name": route_name})

    async def add_zone(
        self,
        zone_name: str,
        type: str,
        geometry: Dict[str, Any],
        max_speed: Optional[float] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Zone:
        command: Dict[str, Any] = {
            "command": "add_zone",
            "name": zone_name,
            "type": type,
            "geometry": geometry_mm_to_m(geometry),
        }
        if max_speed is not None:
            command["speed_pct"] = _speed_pct(max_speed, self._max_linear_mps())
        result = await self._nav_call(command)
        return _zone_from_dict(result["zone"], self._max_linear_mps())

    async def get_zone(
        self,
        zone_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Zone:
        result = await self._nav_call({"command": "get_zone", "name": zone_name})
        return _zone_from_dict(result["zone"], self._max_linear_mps())

    async def list_zones(
        self,
        type: Optional[str] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Zone]:
        command: Dict[str, Any] = {"command": "list_zones"}
        if type:
            command["type"] = type
        result = await self._nav_call(command)
        cap = self._max_linear_mps()
        return [_zone_from_dict(zone, cap) for zone in result.get("zones") or []]

    async def update_zone(
        self,
        zone_name: str,
        type: Optional[str] = None,
        geometry: Optional[Dict[str, Any]] = None,
        max_speed: Optional[float] = None,
        new_name: Optional[str] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Zone:
        command: Dict[str, Any] = {"command": "update_zone", "name": zone_name}
        if type is not None:
            command["type"] = type
        if geometry is not None:
            command["geometry"] = geometry_mm_to_m(geometry)
        if max_speed is not None:
            command["speed_pct"] = _speed_pct(max_speed, self._max_linear_mps())
        if new_name is not None:
            command["new_name"] = new_name
        result = await self._nav_call(command)
        return _zone_from_dict(result["zone"], self._max_linear_mps())

    async def delete_zone(
        self,
        zone_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        await self._nav_call({"command": "delete_zone", "name": zone_name})

    async def navigate_to_location(
        self,
        location_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._require_localized()
        await self._nav_call({"command": "navigate_to_location", "name": location_name})
        return await self.get_navigation_status()

    async def navigate_to_point(
        self,
        destination: Position,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._require_localized()
        pose = position_to_pose2d(destination)
        await self._nav_call(
            {
                "command": "navigate_to_point",
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
            }
        )
        return await self.get_navigation_status()

    async def navigate_route(
        self,
        route_name: Optional[str] = None,
        waypoints: Optional[List[Waypoint]] = None,
        start_index: int = 0,
        start_nearest: bool = False,
        loop: bool = False,
        wait: bool = False,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._require_localized()
        command: Dict[str, Any] = {
            "command": "navigate_route",
            "start_index": int(start_index),
            "start_nearest": bool(start_nearest),
            "loop": bool(loop),
            "wait": bool(wait),
        }
        if route_name:
            command["name"] = route_name
        if waypoints is not None:
            command["waypoints"] = [_waypoint_command(wp) for wp in waypoints]
        await self._nav_call(command)
        return await self.get_navigation_status()

    async def plan_to_point(
        self,
        destination: Position,
        start: Optional[Position] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Plan:
        pose = position_to_pose2d(destination)
        command: Dict[str, Any] = {
            "command": "plan_to_point",
            "x": pose.x,
            "y": pose.y,
            "theta": pose.theta,
        }
        if start is not None:
            origin = position_to_pose2d(start)
            command["start"] = {"x": origin.x, "y": origin.y, "theta": origin.theta}
        preview = await self._nav_call(command)
        return plan_from_preview(preview)

    async def plan_to_location(
        self,
        location_name: str,
        start: Optional[Position] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Plan:
        command: Dict[str, Any] = {
            "command": "plan_to_location",
            "name": location_name,
        }
        if start is not None:
            origin = position_to_pose2d(start)
            command["start"] = {"x": origin.x, "y": origin.y, "theta": origin.theta}
        preview = await self._nav_call(command)
        location = preview.get("location")
        name = location.get("name") if isinstance(location, dict) else location_name
        return plan_from_preview(preview, location=str(name) if name else None)

    async def execute_plan(
        self,
        plan: Plan,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        if not plan.path:
            raise GRPCError(Status.INVALID_ARGUMENT, "plan path is empty")
        await self._require_localized()
        dest = position_to_pose2d(plan.path[-1])
        await self._nav_call(
            {
                "command": "navigate_to_point",
                "x": dest.x,
                "y": dest.y,
                "theta": dest.theta,
            }
        )
        if plan.HasField("location") and plan.location:
            self._active_goal_name = plan.location  # type: ignore[attr-defined]
        return await self.get_navigation_status()

    async def go_to_location(
        self,
        location_name: str,
        wait: Optional[bool] = None,
        linear_m_per_sec: Optional[float] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._require_localized()
        command: Dict[str, Any] = {
            "command": "go_to_location",
            "name": location_name,
            "wait": True if wait is None else bool(wait),
        }
        if linear_m_per_sec is not None:
            command["velocity_mps"] = float(linear_m_per_sec)
        await self._nav_call(command)
        return await self.get_navigation_status()

    async def go_to_point(
        self,
        destination: Position,
        wait: Optional[bool] = None,
        linear_m_per_sec: Optional[float] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._require_localized()
        pose = position_to_pose2d(destination)
        command: Dict[str, Any] = {
            "command": "go_to_point",
            "x": pose.x,
            "y": pose.y,
            "theta": pose.theta,
            "wait": True if wait is None else bool(wait),
        }
        if linear_m_per_sec is not None:
            command["velocity_mps"] = float(linear_m_per_sec)
        await self._nav_call(command)
        return await self.get_navigation_status()

    async def suspend(
        self,
        reason: str = "",
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        command: Dict[str, Any] = {"command": "suspend"}
        if reason:
            command["reason"] = reason
        await self._nav_call(command)
        return await self.get_navigation_status()

    async def resume(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._nav_call({"command": "resume"})
        return await self.get_navigation_status()

    async def cancel(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        await self._nav_call({"command": "cancel"})
        return await self.get_navigation_status()

    async def get_position(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Position:
        status = await self.get_navigation_status()
        if status.position.WhichOneof("value") != "pose":
            raise GRPCError(Status.FAILED_PRECONDITION, "map pose unavailable")
        return status.position

    async def get_navigation_status(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> NavStatus:
        raw = await self._nav_call({"command": "get_status"})
        return status_from_dict(raw)

    async def get_status(
        self,
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, Any]:
        """Common ``GetStatus`` RPC. Same payload as DoCommand ``get_status``.

        ``GetNavigationStatus`` is the typed view of this payload.
        """
        return await self._nav_call({"command": "get_status"})

    async def get_properties(
        self,
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Properties:
        cfg = self._require_cfg()  # type: ignore[attr-defined]
        length = float(cfg.footprint_length_m or 0.0)
        width = float(cfg.footprint_width_m or 0.0)
        if not length or not width:
            diam = 2.0 * float(cfg.robot_radius)
            length = length or diam
            width = width or diam
        return Properties(
            map_type=MapType.MAP_TYPE_LOCAL,
            footprint_length_m=length,
            footprint_width_m=width,
            robot_radius_m=float(cfg.hard_clearance_radius_m()),
            planning_supported=True,
            costmap_supported=True,
            zones_supported=True,
            routes_supported=True,
        )

    async def get_costmap(
        self,
        layer: str = "",
        stride: int = 0,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> GetCostmapResponse:
        raw = await self._nav_call(
            {
                "command": "get_costmap",
                "layer": layer or "auto",
                "stride": int(stride) if stride else 2,
            }
        )
        if not raw.get("available"):
            return GetCostmapResponse(available=False, layer=str(raw.get("layer") or ""))
        data = base64.b64decode(raw.get("data_b64") or "")
        return GetCostmapResponse(
            available=True,
            layer=str(raw.get("layer") or ""),
            nav_backend=str(raw.get("nav_backend") or ""),
            origin_x=float(raw.get("origin_x") or 0.0),
            origin_y=float(raw.get("origin_y") or 0.0),
            resolution=float(raw.get("resolution") or 0.0),
            width=int(raw.get("width") or 0),
            height=int(raw.get("height") or 0),
            encoding=str(raw.get("encoding") or "uint8_row_major"),
            data=data,
        )


def _zone_from_dict(zone: Mapping[str, Any], max_linear_mps: float) -> Zone:
    msg = Zone(
        name=str(zone["name"]),
        type=str(zone["type"]),
        geometry=dict_to_struct(geometry_m_to_mm(zone["geometry"])),
    )
    if zone.get("speed_pct") is not None and max_linear_mps > 0:
        msg.max_speed = float(zone["speed_pct"]) / 100.0 * float(max_linear_mps)
    return msg


def _waypoint_command(wp: Waypoint) -> dict:
    kind = wp.WhichOneof("value")
    if kind == "location":
        return {"location": wp.location}
    if kind == "position":
        pose = position_to_pose2d(wp.position)
        return {"x": pose.x, "y": pose.y, "theta": pose.theta}
    raise GRPCError(Status.INVALID_ARGUMENT, "waypoint needs a location or a position")

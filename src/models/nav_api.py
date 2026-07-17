"""Shared Viam ``rdk:service:navigation`` API surface for the navigation models.

Both ``navigation`` (built-in slam_toolbox runtime) and ``navigation-external``
(arbitrary ``rdk:service:slam`` runtime) expose the standard Navigation API so a
client already speaking it (e.g. a webapp built against another
``rdk:service:navigation`` module) can drive either backend unchanged. The API
methods + waypoint queue live here; each model differs only in how it resolves
its ``SlamRuntime`` (``NavCoreMixin._resolve_runtime``).

Mixed in as ``(NavApiMixin, NavCoreMixin, Navigation)``: ``NavApiMixin`` provides
the Navigation methods, ``NavCoreMixin`` the Nav2 orchestration + ``do_command``,
``Navigation`` the service base — wired via cooperative ``super().__init__``.

Coordinate overload (matches the RTAB-Map module's documented convention so the
same webapp works): the map is not georeferenced — ``GeoPoint.latitude`` carries
map-frame **x** metres and ``GeoPoint.longitude`` carries map-frame **y** metres.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from viam.logging import getLogger
from viam.proto.common import (
    GeoGeometry,
    GeoPoint,
    Geometry,
    Pose,
    RectangularPrism,
    Vector3,
)
from viam.proto.service.navigation import MapType, Mode, Path, Waypoint

from ..nav import zones as zones_mod

LOGGER = getLogger(__name__)


# -- coordinate overload (latitude = x metres, longitude = y metres) ----------
def map_geopoint(x: float, y: float) -> GeoPoint:
    return GeoPoint(latitude=float(x), longitude=float(y))


def geopoint_to_xy(point: GeoPoint) -> tuple[float, float]:
    return float(point.latitude), float(point.longitude)


def waypoints_to_path(waypoints: List[Waypoint]) -> List[Path]:
    """Remaining waypoints as a single planned Path (coarse; the real Nav2 /plan
    is a follow-up). Empty when there are no waypoints."""
    if not waypoints:
        return []
    return [
        Path(
            destination_waypoint_id=waypoints[-1].id,
            geopoints=[w.location for w in waypoints],
        )
    ]


def zone_to_geo_geometry(zone) -> Optional[GeoGeometry]:
    """A keepout Zone -> GeoGeometry as an axis-aligned bounding box.

    Matches the RTAB-Map module: polygons are approximated by their AABB (the
    nearest primitive geo_geometry supports). Center uses the map overload
    (lat=x, lng=y); box dims are millimetres per the Viam geometry convention.
    """
    g = zone.geometry
    shape = g.get("type")
    try:
        if shape == "circle":
            cx, cy = g["center"]
            r = float(g["radius"])
            w = h = 2.0 * r
        elif shape == "box":
            cx, cy = g["center"]
            w, h = g["size"]  # AABB ignores rotation for the display approximation
        elif shape == "polygon":
            xs = [float(p[0]) for p in g["points"]]
            ys = [float(p[1]) for p in g["points"]]
            cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    geom = Geometry(
        center=Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        box=RectangularPrism(
            dims_mm=Vector3(x=float(w) * 1000.0, y=float(h) * 1000.0, z=100.0)
        ),
        label=zone.name,
    )
    return GeoGeometry(location=map_geopoint(cx, cy), geometries=[geom])


def _read_allow_unknown(cfg) -> bool:
    try:
        return bool(
            cfg.nav2_params["planner_server"]["ros__parameters"]["GridBased"][
                "allow_unknown"
            ]
        )
    except (KeyError, TypeError):
        return True  # Nav2 GridBased default


def _write_allow_unknown(cfg, value: bool) -> None:
    params = dict(cfg.nav2_params or {})
    ps = dict(params.get("planner_server") or {})
    rp = dict(ps.get("ros__parameters") or {})
    gb = dict(rp.get("GridBased") or {})
    gb["allow_unknown"] = bool(value)
    rp["GridBased"] = gb
    ps["ros__parameters"] = rp
    params["planner_server"] = ps
    cfg.nav2_params = params


class NavApiMixin:
    """Standard Navigation-API methods over a resolved ``SlamRuntime``.

    Requires the ``NavCoreMixin`` API (``_resolve_runtime``, ``_require_runtime``,
    ``_require_cfg``, ``_zones``) on the same instance.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._mode: int = Mode.MODE_MANUAL
        self._waypoints: List[Waypoint] = []
        self._wp_counter = 0
        self._wp_task: Optional[asyncio.Task] = None

    async def close(self) -> None:
        self._stop_waypoint_driver()
        parent_close = getattr(super(), "close", None)
        if parent_close is not None:
            await parent_close()

    def _reset_nav_state(self) -> None:
        """Clear mode + waypoint queue on (re)configure."""
        self._stop_waypoint_driver()
        self._mode = Mode.MODE_MANUAL
        self._waypoints = []

    # -- DoCommand extensions (webapp) ; rest delegate to NavCoreMixin -------
    async def do_command(self, command, *, timeout: Optional[float] = None, **kwargs):
        cmd = command.get("command")
        if cmd == "clear_waypoints":
            self._waypoints = []
            return {"cleared": True}
        if cmd == "replan":
            # Refresh annotation-derived masks + return the current polyline.
            await self._apply_annotations()
            pts = [
                [gp.latitude, gp.longitude]
                for p in waypoints_to_path(self._waypoints)
                for gp in p.geopoints
            ]
            return {"ok": True, "points": pts}
        if cmd == "get_motors_enabled":
            return {"enabled": self._require_runtime().manager.motors_enabled()}
        if cmd == "set_motors_enabled":
            enabled = bool(command["enabled"])
            self._require_runtime().manager.set_motors_enabled(enabled)
            return {"enabled": enabled}
        if cmd == "get_planner_config":
            cfg = self._require_cfg()
            return {
                "inflation_radius": cfg.inflation_radius,
                "allow_unknown": _read_allow_unknown(cfg),
            }
        if cmd == "set_planner_config":
            return await self._set_planner_config(command)
        return await super().do_command(command, timeout=timeout, **kwargs)

    async def _set_planner_config(self, command) -> dict:
        cfg = self._require_cfg()
        if "inflation_radius" in command:
            cfg.inflation_radius = float(command["inflation_radius"])
        if "allow_unknown" in command:
            _write_allow_unknown(cfg, bool(command["allow_unknown"]))
        runtime = self._require_runtime()
        runtime.manager.set_nav_config(cfg)
        params_path = self._write_nav2_params(cfg)

        def _restart():
            runtime.manager.stop_nav2()
            runtime.manager.ensure_nav2(cfg, params_path)

        await asyncio.to_thread(_restart)
        return {
            "inflation_radius": cfg.inflation_radius,
            "allow_unknown": _read_allow_unknown(cfg),
        }

    # -- Navigation API ------------------------------------------------------
    async def get_location(self, *, timeout: Optional[float] = None, **kwargs) -> GeoPoint:
        pose = self._require_runtime().manager.get_pose_in_map()
        if pose is None:
            raise RuntimeError("robot pose unavailable (SLAM not localized yet)")
        return map_geopoint(pose.x, pose.y)

    async def get_mode(self, *, timeout: Optional[float] = None, **kwargs) -> int:
        return self._mode

    async def set_mode(self, mode: int, *, timeout: Optional[float] = None, **kwargs) -> None:
        if mode == Mode.MODE_EXPLORE:
            raise ValueError("navigation does not support explore mode")
        if mode not in (Mode.MODE_MANUAL, Mode.MODE_WAYPOINT):
            raise ValueError(f"unsupported navigation mode: {mode}")
        self._mode = mode
        if mode == Mode.MODE_WAYPOINT:
            self._start_waypoint_driver()
        else:  # MANUAL: stop driving through waypoints and halt Nav2.
            self._stop_waypoint_driver()
            await asyncio.to_thread(self._require_runtime().manager.cancel)

    async def get_waypoints(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> List[Waypoint]:
        return list(self._waypoints)

    async def add_waypoint(
        self, point: GeoPoint, *, timeout: Optional[float] = None, **kwargs
    ) -> None:
        wp = Waypoint(id=str(self._wp_counter), location=point)
        self._wp_counter += 1
        self._waypoints.append(wp)

    async def remove_waypoint(
        self, id: str, *, timeout: Optional[float] = None, **kwargs
    ) -> None:
        self._waypoints = [w for w in self._waypoints if w.id != id]

    async def get_paths(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> List[Path]:
        return waypoints_to_path(self._waypoints)

    async def get_obstacles(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> List[GeoGeometry]:
        from types import SimpleNamespace

        from ..nav import annotations as ann

        obstacles: List[GeoGeometry] = []
        # Annotation no_go + slow_down regions (from the SLAM source).
        fc = await self._get_annotations()
        ann_polys = ann.no_go_polygons(fc) + [g for g, _ in ann.slow_down_regions(fc)]
        for i, geom in enumerate(ann_polys):
            gg = zone_to_geo_geometry(SimpleNamespace(name=f"ann_{i}", geometry=geom))
            if gg is not None:
                obstacles.append(gg)
        # Local keepout zones (DoCommand-authored).
        for z in self._zones().list(zones_mod.KEEPOUT):
            gg = zone_to_geo_geometry(z)
            if gg is not None:
                obstacles.append(gg)
        return obstacles

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> int:
        return MapType.MAP_TYPE_NONE  # local map, not georeferenced

    # -- waypoint driver -----------------------------------------------------
    def _start_waypoint_driver(self) -> None:
        if self._wp_task is None or self._wp_task.done():
            self._wp_task = asyncio.get_event_loop().create_task(self._drive_waypoints())

    def _stop_waypoint_driver(self) -> None:
        if self._wp_task is not None and not self._wp_task.done():
            self._wp_task.cancel()
        self._wp_task = None

    async def _drive_waypoints(self) -> None:
        """Navigate through the waypoint queue in order while in WAYPOINT mode."""
        try:
            while self._mode == Mode.MODE_WAYPOINT:
                wp = self._waypoints[0] if self._waypoints else None
                if wp is None:
                    await asyncio.sleep(0.5)
                    continue
                x, y = geopoint_to_xy(wp.location)
                await self._navigate(x, y, 0.0)  # refreshes annotation masks, then goals
                reached = await self._await_arrival()
                if self._mode != Mode.MODE_WAYPOINT:
                    break
                if reached:
                    self._waypoints = [w for w in self._waypoints if w.id != wp.id]
                else:
                    LOGGER.warning(
                        "navigation: waypoint %s did not complete; leaving waypoint mode",
                        wp.id,
                    )
                    self._mode = Mode.MODE_MANUAL
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - driver must not crash silently
            LOGGER.error("waypoint driver failed: %s", exc)

    async def _await_arrival(self, *, timeout_s: float = 600.0) -> bool:
        """Poll Nav2 status until the active goal finishes; True if it succeeded."""
        deadline = time.monotonic() + timeout_s
        saw_active = False
        while time.monotonic() < deadline and self._mode == Mode.MODE_WAYPOINT:
            st = await asyncio.to_thread(self._require_runtime().manager.nav_status)
            active = bool(st.get("active"))
            state = st.get("state")
            if active:
                saw_active = True
            elif saw_active:
                return state in ("succeeded", "reached")
            elif state in ("failed", "rejected", "canceled"):
                return False
            await asyncio.sleep(0.5)
        return False

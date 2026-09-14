"""Shared navigation-service core for the nav-stack models.

Both navigation models — ``viam-labs:nav-stack:navigation`` (borrows the SLAM
model's in-process runtime) and ``viam-labs:nav-stack:navigation-external``
(builds its own runtime around an arbitrary ``rdk:service:slam``) — share the
Motion API (``MoveOnMap`` / plan queries), DoCommand surface, and simple
closed-loop motion. That logic lives here in ``NavServiceBase``; the concrete
models differ only in how they obtain a ``SlamRuntime`` (``_resolve_runtime``)
and how they stand it up (``reconfigure``).
"""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Mapping, Optional, Sequence

from google.protobuf.timestamp_pb2 import Timestamp
from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.components.base import Base
from viam.logging import getLogger
from viam.proto.common import (
    GeoGeometry,
    GeoPoint,
    Geometry,
    Pose,
    PoseInFrame,
    Transform,
    Vector3,
    WorldState,
)
from viam.proto.service.motion import (
    ComponentState,
    Constraints,
    MotionConfiguration,
    Plan,
    PlanState,
    PlanStatus,
    PlanStatusWithID,
    PlanStep,
    PlanWithStatus,
)
from viam.services.motion import Motion
from viam.utils import ValueTypes

from ..config import NavConfig, body_twist_to_viam_set_velocity
from ..nav import zones as zones_mod
from ..nav.locations import LocationStore
from ..nav.maps import MapHandle
from ..nav.motion_summary import summarize_nav_motion
from ..nav.simple_motion import (
    ObstacleConfig,
    SimpleMotionCanceled,
    SimpleMotionError,
    config_from_nav,
    drive_to_pose,
)
from ..nav.zones import ZoneStore
from ..geom import conversions as conv

LOGGER = getLogger(__name__)

_TERMINAL_PLAN_STATES = frozenset(
    {
        PlanState.PLAN_STATE_SUCCEEDED,
        PlanState.PLAN_STATE_FAILED,
        PlanState.PLAN_STATE_STOPPED,
    }
)


def _nav_status_to_plan_state(status: Mapping) -> PlanState:
    """Map bridge ``nav_status()`` into a Motion ``PlanState``."""
    if status.get("active"):
        return PlanState.PLAN_STATE_IN_PROGRESS
    state = str(status.get("state") or "").lower()
    if state == "succeeded":
        return PlanState.PLAN_STATE_SUCCEEDED
    if state in ("canceled", "cancelled"):
        return PlanState.PLAN_STATE_STOPPED
    if state in ("failed", "aborted", "rejected"):
        return PlanState.PLAN_STATE_FAILED
    return PlanState.PLAN_STATE_UNSPECIFIED


def _utcnow_timestamp() -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(timezone.utc))
    return ts


def _pose2d_to_viam_pose_msg(pose: conv.Pose2D) -> Pose:
    x_mm, y_mm, z_mm, o_x, o_y, o_z, theta_deg = conv.pose2d_to_viam_slam_pose(pose)
    return Pose(x=x_mm, y=y_mm, z=z_mm, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta_deg)


@dataclass
class _PlanExecution:
    execution_id: str
    plan_id: str
    component_name: str
    destination: Pose
    state: PlanState = PlanState.PLAN_STATE_IN_PROGRESS
    reason: Optional[str] = None
    status_history: List[PlanStatus] = field(default_factory=list)


@dataclass
class _SuspendedNav:
    """Goal remembered by ``suspend`` so ``resume`` can re-issue navigate."""

    x: float
    y: float
    theta: float
    name: Optional[str] = None
    motion: str = "builtin"  # "builtin" | "simple"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        out = {
            "x": float(self.x),
            "y": float(self.y),
            "theta": float(self.theta),
            "motion": self.motion,
        }
        if self.name:
            out["name"] = self.name
        if self.reason:
            out["reason"] = self.reason
        return out


class NavServiceBase(Motion):
    """Runtime-agnostic navigation orchestration shared by the navigation models.

    Subclasses must implement :meth:`_resolve_runtime` (return the active
    ``SlamRuntime`` or ``None``) and :meth:`reconfigure`.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[NavConfig] = None
        self._base: Optional[Base] = None
        self._simple_nav_task: Optional[asyncio.Task] = None
        self._simple_nav_cancel: Optional[asyncio.Event] = None
        self._simple_nav_status: dict = {"state": "idle", "motion": "simple"}
        # Optional label (e.g. location name) merged into status["goal"].
        self._active_goal_name: Optional[str] = None
        self._plan_execution: Optional[_PlanExecution] = None
        self._plan_status_history: List[PlanStatusWithID] = []
        self._logged_motion_ignored: bool = False
        self._last_preview_plan: Optional[dict] = None
        # Set by ``suspend``; cleared by ``resume``, ``cancel``, ``stop_plan``,
        # or any new navigate / MoveOnMap / simple go.
        self._suspended: Optional[_SuspendedNav] = None
        # Builtin get_costmap cache (nav-stack-ui polls while Costmap is on).
        self._builtin_costmap_cache: Optional[dict] = None
        self._builtin_costmap_cache_at: float = 0.0

    # -- runtime resolution (subclass-specific) ------------------------------
    def _resolve_runtime(self):
        """Return the active ``SlamRuntime`` or ``None``.

        Built-in model looks it up in the in-process registry by the SLAM
        service name; the external model returns its locally-built runtime.
        """
        raise NotImplementedError

    def _require_cfg(self) -> NavConfig:
        if self._cfg is None:
            raise RuntimeError("navigation service not configured")
        return self._cfg

    def _require_runtime(self):
        # Surface "not configured" before touching the runtime source.
        self._require_cfg()
        runtime = self._resolve_runtime()
        if runtime is None:
            raise RuntimeError("SLAM runtime unavailable")
        return runtime

    # -- Motion API ----------------------------------------------------------
    async def move(
        self,
        component_name: str,
        destination: PoseInFrame,
        world_state: Optional[WorldState] = None,
        constraints: Optional[Constraints] = None,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        raise GRPCError(Status.UNIMPLEMENTED, "Move is not supported; use MoveOnMap")

    async def move_on_globe(
        self,
        component_name: str,
        destination: GeoPoint,
        movement_sensor_name: str,
        obstacles: Optional[Sequence[GeoGeometry]] = None,
        heading: Optional[float] = None,
        configuration: Optional[MotionConfiguration] = None,
        *,
        bounding_regions: Optional[Sequence[GeoGeometry]] = None,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        raise GRPCError(
            Status.UNIMPLEMENTED,
            "MoveOnGlobe is not supported; use MoveOnMap for map-frame navigation",
        )

    async def move_on_map(
        self,
        component_name: str,
        destination: Pose,
        slam_service_name: str,
        configuration: Optional[MotionConfiguration] = None,
        obstacles: Optional[Sequence[Geometry]] = None,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        cfg = self._require_cfg()
        if component_name and component_name != cfg.base:
            LOGGER.warning(
                f"MoveOnMap component_name={component_name!r} does not match "
                f"configured base={cfg.base!r}; navigating configured base anyway"
            )
        if slam_service_name and slam_service_name != cfg.slam_service:
            LOGGER.warning(
                f"MoveOnMap slam_service_name={slam_service_name!r} does not match "
                f"configured slam_service={cfg.slam_service!r}; continuing anyway"
            )
        if (obstacles or configuration) and not self._logged_motion_ignored:
            LOGGER.info(
                "MoveOnMap obstacles/configuration are ignored in v1 "
                "(costmaps still use live lidar)"
            )
            self._logged_motion_ignored = True

        pose2d = conv.viam_pose_to_pose2d(destination.x, destination.y, destination.theta)
        runtime = self._require_runtime()
        mgr = runtime.manager

        # Preview-only: plan a path, do not drive.
        extra = extra or {}
        if bool(extra.get("preview") or extra.get("plan_only")):
            preview = await asyncio.to_thread(
                mgr.compute_path, pose2d.x, pose2d.y, pose2d.theta
            )
            self._last_preview_plan = preview
            if not preview.get("feasible"):
                raise GRPCError(
                    Status.FAILED_PRECONDITION,
                    preview.get("error_msg")
                    or f"no feasible path (error_code={preview.get('error_code')})",
                )
            # Return a synthetic execution id so callers can correlate with DoCommand
            # get_last_plan; motion is not started.
            return f"preview-{uuid.uuid4()}"

        # A new MoveOnMap supersedes any in-flight plan.
        if (
            self._plan_execution is not None
            and self._plan_execution.state not in _TERMINAL_PLAN_STATES
        ):
            self._record_plan_state(
                self._plan_execution, PlanState.PLAN_STATE_STOPPED, reason="superseded"
            )

        self._suspended = None
        execution_id = str(uuid.uuid4())
        plan_id = str(uuid.uuid4())
        base_name = component_name or cfg.base
        execution = _PlanExecution(
            execution_id=execution_id,
            plan_id=plan_id,
            component_name=base_name,
            destination=Pose(
                x=destination.x,
                y=destination.y,
                z=destination.z,
                o_x=destination.o_x,
                o_y=destination.o_y,
                o_z=destination.o_z,
                theta=destination.theta,
            ),
            state=PlanState.PLAN_STATE_IN_PROGRESS,
        )
        execution.status_history.append(
            PlanStatus(
                state=PlanState.PLAN_STATE_IN_PROGRESS,
                timestamp=_utcnow_timestamp(),
            )
        )
        self._plan_execution = execution
        self._active_goal_name = None
        self._upsert_plan_status_history(execution)

        await self._cancel_simple_nav()
        await asyncio.to_thread(mgr.navigate, pose2d.x, pose2d.y, pose2d.theta)
        self._sync_plan_state_from_nav()
        return execution_id

    async def stop_plan(
        self,
        component_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        runtime = self._require_runtime()
        self._active_goal_name = None
        self._suspended = None
        await self._cancel_simple_nav()
        await asyncio.to_thread(runtime.manager.cancel)
        if self._plan_execution is not None:
            self._record_plan_state(
                self._plan_execution, PlanState.PLAN_STATE_STOPPED, reason="stop_plan"
            )

    async def get_plan(
        self,
        component_name: str,
        last_plan_only: bool = False,
        execution_id: Optional[str] = None,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Motion.Plan:
        self._sync_plan_state_from_nav()
        execution = self._plan_execution
        if execution is None:
            raise GRPCError(Status.NOT_FOUND, "no motion plan has been started")
        if execution_id and execution.execution_id != execution_id:
            raise GRPCError(
                Status.NOT_FOUND,
                f"no plan found for execution_id={execution_id!r}",
            )
        # last_plan_only: we only keep the current plan (no replan history yet).
        _ = last_plan_only
        _ = component_name
        return self._plan_to_response(execution)

    async def list_plan_statuses(
        self,
        only_active_plans: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Sequence[PlanStatusWithID]:
        self._sync_plan_state_from_nav()
        statuses = list(self._plan_status_history)
        if only_active_plans:
            statuses = [
                s
                for s in statuses
                if s.status.state == PlanState.PLAN_STATE_IN_PROGRESS
            ]
        return statuses

    async def get_pose(
        self,
        component_name: str,
        destination_frame: str,
        supplemental_transforms: Optional[Sequence[Transform]] = None,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> PoseInFrame:
        _ = supplemental_transforms
        cfg = self._require_cfg()
        if component_name and component_name != cfg.base:
            LOGGER.warning(
                f"get_pose component_name={component_name!r} does not match "
                f"configured base={cfg.base!r}"
            )
        frame = destination_frame or "map"
        if frame not in ("map", ""):
            raise GRPCError(
                Status.INVALID_ARGUMENT,
                f"only destination_frame='map' is supported (got {destination_frame!r})",
            )
        pose2d = await asyncio.to_thread(self._require_runtime().manager.get_pose_in_map)
        if pose2d is None:
            raise GRPCError(Status.FAILED_PRECONDITION, "current map pose unavailable")
        return PoseInFrame(reference_frame="map", pose=_pose2d_to_viam_pose_msg(pose2d))

    def _record_plan_state(
        self,
        execution: _PlanExecution,
        state: PlanState,
        *,
        reason: Optional[str] = None,
    ) -> None:
        if execution.state == state and (reason is None or execution.reason == reason):
            self._upsert_plan_status_history(execution)
            return
        execution.state = state
        execution.reason = reason
        status = PlanStatus(state=state, timestamp=_utcnow_timestamp())
        if reason:
            status.reason = reason
        execution.status_history.append(status)
        self._upsert_plan_status_history(execution)

    def _upsert_plan_status_history(self, execution: _PlanExecution) -> None:
        entry = PlanStatusWithID(
            plan_id=execution.plan_id,
            execution_id=execution.execution_id,
            component_name=execution.component_name,
            status=PlanStatus(
                state=execution.state,
                timestamp=_utcnow_timestamp(),
                reason=execution.reason or "",
            ),
        )
        updated: List[PlanStatusWithID] = []
        found = False
        for existing in self._plan_status_history:
            if existing.execution_id == execution.execution_id:
                updated.append(entry)
                found = True
            else:
                updated.append(existing)
        if not found:
            updated.append(entry)
        # Keep a bounded history of recent executions.
        self._plan_status_history = updated[-20:]

    def _sync_plan_state_from_nav(self) -> None:
        execution = self._plan_execution
        if execution is None:
            return
        if execution.state == PlanState.PLAN_STATE_STOPPED:
            # Explicit stop_plan / superseded — do not overwrite from live nav.
            self._upsert_plan_status_history(execution)
            return
        try:
            status = self._require_runtime().manager.nav_status()
        except Exception:  # noqa: BLE001 - plan queries should still return last known
            return
        mapped = _nav_status_to_plan_state(status)
        if mapped == PlanState.PLAN_STATE_UNSPECIFIED:
            if execution.state not in _TERMINAL_PLAN_STATES:
                mapped = PlanState.PLAN_STATE_IN_PROGRESS
            else:
                self._upsert_plan_status_history(execution)
                return
        reason = None
        if mapped == PlanState.PLAN_STATE_FAILED:
            reason = str(status.get("state") or "failed")
        self._record_plan_state(execution, mapped, reason=reason)

    def _plan_to_response(self, execution: _PlanExecution) -> Motion.Plan:
        component_key = str(Base.get_resource_name(execution.component_name))
        plan = Plan(
            id=execution.plan_id,
            execution_id=execution.execution_id,
            component_name=execution.component_name,
            steps=[
                PlanStep(
                    step={
                        component_key: ComponentState(pose=execution.destination),
                    }
                )
            ],
        )
        current = PlanStatus(
            state=execution.state,
            timestamp=_utcnow_timestamp(),
            reason=execution.reason or "",
        )
        return Motion.Plan(
            current_plan_with_status=PlanWithStatus(
                plan=plan,
                status=current,
                status_history=list(execution.status_history),
            )
        )

    # -- store helpers -------------------------------------------------------
    def _active_handle(self) -> MapHandle:
        runtime = self._require_runtime()
        handle = runtime.map_store.active_handle()
        if handle is None:
            raise RuntimeError("no active map; create/select one via the SLAM service")
        return handle

    def _locations(self) -> LocationStore:
        return LocationStore(self._active_handle().locations_path)

    def _zones(self) -> ZoneStore:
        return ZoneStore(self._active_handle().zones_path)

    def _refresh_zone_masks(self) -> None:
        runtime = self._require_runtime()
        node = getattr(runtime.manager, "node", None)
        if node is None:
            # Builtin host: no costmap-filter publisher.
            return
        grid = node.get_map() if node else None
        if not grid:
            LOGGER.warning("no map yet; zone masks will publish once a map is available")
            return
        h, w = grid["grid"].shape
        res = grid["resolution"]
        ox, oy = grid["origin_x"], grid["origin_y"]
        zone_list = self._zones().list()
        keepout = zones_mod.rasterize_zones(zone_list, zones_mod.KEEPOUT, w, h, res, ox, oy)
        speed = zones_mod.rasterize_zones(zone_list, zones_mod.SPEED_LIMIT, w, h, res, ox, oy)
        runtime.manager.publish_zone_masks(keepout, speed, res, ox, oy)

    # -- DoCommand -----------------------------------------------------------
    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        runtime = self._require_runtime()
        mgr = runtime.manager

        # -- locations CRUD --
        if cmd == "add_location":
            loc = self._add_location(command, runtime)
            return {"location": loc.to_dict()}
        if cmd == "get_location":
            return {"location": self._locations().get(str(command["name"])).to_dict()}
        if cmd == "list_locations":
            return {"locations": [l.to_dict() for l in self._locations().list()]}
        if cmd == "update_location":
            loc = self._locations().update(
                str(command["name"]),
                x=command.get("x"),
                y=command.get("y"),
                theta=command.get("theta"),
                new_name=command.get("new_name"),
            )
            return {"location": loc.to_dict()}
        if cmd in ("delete_location", "remove_location"):
            self._locations().delete(str(command["name"]))
            return {"status": "deleted"}
        if cmd == "delete_all_locations":
            self._locations().delete_all()
            return {"status": "deleted"}

        # -- zones CRUD --
        if cmd == "add_zone":
            zone = self._zones().add(
                str(command["name"]),
                str(command["type"]),
                dict(command["geometry"]),
                speed_pct=command.get("speed_pct"),
            )
            self._refresh_zone_masks()
            return {"zone": zone.to_dict()}
        if cmd == "get_zone":
            return {"zone": self._zones().get(str(command["name"])).to_dict()}
        if cmd == "list_zones":
            zlist = self._zones().list(command.get("type"))
            return {"zones": [z.to_dict() for z in zlist]}
        if cmd == "update_zone":
            zone = self._zones().update(
                str(command["name"]),
                type=command.get("type"),
                geometry=dict(command["geometry"]) if "geometry" in command else None,
                speed_pct=command.get("speed_pct"),
                new_name=command.get("new_name"),
            )
            self._refresh_zone_masks()
            return {"zone": zone.to_dict()}
        if cmd == "delete_zone":
            self._zones().delete(str(command["name"]))
            self._refresh_zone_masks()
            return {"status": "deleted"}
        if cmd == "delete_all_zones":
            self._zones().delete_all(command.get("type"))
            self._refresh_zone_masks()
            return {"status": "deleted"}

        # -- navigation --
        # navigate/cancel/status may block (planning / drive). Keep them off the
        # module event loop so Base.SetVelocity and sensor reads stay responsive.
        if cmd == "navigate_to_location":
            loc = self._locations().get(str(command["name"]))
            self._suspended = None
            self._active_goal_name = loc.name
            await self._cancel_simple_nav()
            await asyncio.to_thread(mgr.navigate, loc.x, loc.y, loc.theta)
            return {"status": "navigating", "target": loc.to_dict()}
        if cmd == "navigate_to_point":
            x = float(command["x"])
            y = float(command["y"])
            theta = float(command.get("theta", 0.0))
            self._suspended = None
            self._active_goal_name = None
            await self._cancel_simple_nav()
            await asyncio.to_thread(mgr.navigate, x, y, theta)
            return {"status": "navigating", "target": {"x": x, "y": y, "theta": theta}}
        if cmd in ("plan_to_point", "compute_path_to_point"):
            return await self._plan_preview(command, mgr)
        if cmd in ("plan_to_location", "compute_path_to_location"):
            loc = self._locations().get(str(command["name"]))
            payload = dict(command)
            payload["x"] = loc.x
            payload["y"] = loc.y
            payload["theta"] = loc.theta
            preview = await self._plan_preview(payload, mgr)
            preview["location"] = loc.to_dict()
            self._last_preview_plan = preview
            return preview
        if cmd in ("get_last_plan", "get_preview_plan"):
            preview = self._last_preview_plan or mgr.last_preview_plan()
            if not preview:
                raise ValueError("no preview plan; call plan_to_point first")
            return {"plan": preview}
        if cmd == "execute_plan":
            preview = self._last_preview_plan or mgr.last_preview_plan()
            if not preview:
                raise ValueError("no preview plan; call plan_to_point first")
            if not preview.get("feasible"):
                raise ValueError(
                    preview.get("error_msg")
                    or f"last preview was not feasible (error_code={preview.get('error_code')})"
                )
            goal = preview.get("goal") or {}
            x = float(goal["x"])
            y = float(goal["y"])
            theta = float(goal.get("theta", 0.0))
            name = preview.get("location", {}).get("name") if isinstance(preview.get("location"), dict) else None
            self._suspended = None
            self._active_goal_name = name
            await self._cancel_simple_nav()
            await asyncio.to_thread(mgr.navigate, x, y, theta)
            return {
                "status": "navigating",
                "target": {"x": x, "y": y, "theta": theta, **({"name": name} if name else {})},
                "from_preview": True,
                "length_m": preview.get("length_m"),
            }
        if cmd == "go_to_location":
            loc = self._locations().get(str(command["name"]))
            self._suspended = None
            self._active_goal_name = loc.name
            return await self._start_simple_go(loc.x, loc.y, loc.theta, command)
        if cmd == "go_to_point":
            x = float(command["x"])
            y = float(command["y"])
            theta = float(command.get("theta", 0.0))
            self._suspended = None
            self._active_goal_name = None
            return await self._start_simple_go(x, y, theta, command)
        if cmd in ("suspend", "pause_nav", "suspend_nav"):
            return await self._suspend_nav(command)
        if cmd in ("resume", "resume_nav"):
            return await self._resume_nav(command)
        if cmd == "cancel":
            self._active_goal_name = None
            self._suspended = None
            await self._cancel_simple_nav()
            await asyncio.to_thread(mgr.cancel)
            if self._plan_execution is not None:
                self._record_plan_state(
                    self._plan_execution,
                    PlanState.PLAN_STATE_STOPPED,
                    reason="cancel",
                )
            return {"status": "canceled"}
        if cmd == "test_drive":
            return await self._test_drive(command)
        if cmd in ("get_status", "describe_motion", "what_am_i_doing"):
            def _status():
                status = mgr.nav_status()
                simple = dict(self._simple_nav_status)
                status["simple_nav"] = simple
                if simple.get("state") == "active":
                    status["active"] = True
                    status["motion"] = "simple"
                    # Prefer simple-nav target when active (builtin goal may be stale).
                    target = simple.get("target")
                    if isinstance(target, dict):
                        status["goal"] = dict(target)
                goal = status.get("goal")
                if isinstance(goal, dict) and self._active_goal_name:
                    goal = dict(goal)
                    goal["name"] = self._active_goal_name
                    status["goal"] = goal
                status["localization_check"] = dict(runtime.localization_check)
                suspended = self._suspended
                status["suspended"] = suspended is not None
                status["suspended_goal"] = (
                    suspended.to_dict() if suspended is not None else None
                )
                return status

            status = await asyncio.to_thread(_status)
            if cmd == "get_status":
                return status
            cfg = self._require_cfg()
            return summarize_nav_motion(
                status,
                max_vel_x=float(cfg.max_vel_x),
                max_vel_theta=float(cfg.max_vel_theta),
            )
        if cmd == "get_costmap":
            # Inflated costmap for operator UIs (nav-stack-ui Costmap toggle).
            # ``layer``: ``auto`` (local while navigating, else global), ``local``,
            # or ``global``.
            from ..runtime import get_nav_view

            view = get_nav_view(self.name)
            if view is None:
                return {"available": False, "reason": "viz_unavailable"}

            cfg = self._require_cfg()
            layer_req = str(command.get("layer", "auto")).lower()
            if layer_req not in ("local", "global", "auto"):
                layer_req = "auto"

            def _fetch():
                import time as _time

                if hasattr(view, "enable_viz"):
                    view.enable_viz(8)
                snap = (
                    view.viz_snapshot()
                    if hasattr(view, "viz_snapshot")
                    else view.snapshot()
                )

                nav_active = False
                try:
                    nav_active = bool(runtime.manager.nav_status().get("active"))
                except Exception:  # noqa: BLE001
                    pass
                use_local = layer_req == "local" or (
                    layer_req == "auto" and nav_active
                )
                layer_used = "global"
                cm = None

                if use_local:
                    local_cm = snap.get("local_costmap")
                    if local_cm is not None and local_cm.get("grid") is not None:
                        cm = local_cm
                        layer_used = "local"

                if cm is None:
                    cm = snap.get("costmap")
                if layer_used == "global":
                    now = _time.monotonic()
                    cached = self._builtin_costmap_cache
                    if (
                        cached is not None
                        and now - self._builtin_costmap_cache_at < 1.0
                    ):
                        return cached, layer_used
                    # Always refresh from the live SLAM/world map — never prefer
                    # snap["map"], which is only updated when something else
                    # called world.get_map() (often stale throughout mapping).
                    mp = None
                    world = getattr(runtime.manager, "_world", None)
                    if world is None:
                        world = getattr(runtime.manager, "_builtin_world", None)
                    if world is not None and hasattr(world, "get_map"):
                        try:
                            mp = world.get_map()
                        except Exception:  # noqa: BLE001
                            mp = None
                    if mp is None or mp.get("grid") is None:
                        if hasattr(runtime.manager, "get_map"):
                            try:
                                mp = runtime.manager.get_map()
                            except Exception:  # noqa: BLE001
                                mp = None
                    if mp is None or mp.get("grid") is None:
                        if hasattr(view, "get_map"):
                            mp = view.get_map()
                    if mp is None or mp.get("grid") is None:
                        mp = snap.get("map")
                    if mp is not None and mp.get("grid") is not None:
                        from ..nav_builtin.costmap import (
                            build_costmap,
                            costmap_viz_dict,
                            occupancy_from_map_dict,
                        )

                        occ = occupancy_from_map_dict(mp)
                        costs = build_costmap(
                            occ,
                            inflation_radius_m=cfg.inflation_radius,
                            robot_radius_m=cfg.robot_radius,
                            cost_scaling_factor=float(
                                cfg.builtin.cost_scaling_factor
                            ),
                        )
                        cm = costmap_viz_dict(occ, costs)
                        self._builtin_costmap_cache = cm
                        self._builtin_costmap_cache_at = now
                        # Keep nav-camera / viz store in sync with the live map.
                        try:
                            if hasattr(view, "set_map"):
                                view.set_map(mp)
                            if hasattr(view, "set_costmap"):
                                view.set_costmap(cm)
                            elif hasattr(view, "_viz_lock"):
                                with view._viz_lock:  # noqa: SLF001
                                    view._viz_global_costmap = {  # noqa: SLF001
                                        "grid": cm["grid"],
                                        "resolution": cm["resolution"],
                                        "origin_x": cm["origin_x"],
                                        "origin_y": cm["origin_y"],
                                    }
                        except Exception:  # noqa: BLE001
                            pass
                if cm is None:
                    cm = snap.get("map")
                return cm, layer_used

            cm, layer_used = await asyncio.to_thread(_fetch)
            if not cm or cm.get("grid") is None:
                return {"available": False, "reason": "no_costmap"}

            import base64

            import numpy as np

            grid = np.asarray(cm["grid"])
            # Default stride=2 (~10 cm cells) keeps RPC payloads modest for UI poll.
            stride = max(1, int(command.get("stride", 2)))
            if stride > 1:
                grid = grid[::stride, ::stride]
            height, width = int(grid.shape[0]), int(grid.shape[1])
            resolution = float(cm["resolution"]) * stride
            # 255 = unknown (-1); 0 free; 1..100 cost.
            u8 = np.where(grid < 0, 255, np.clip(grid, 0, 100)).astype(np.uint8)
            return {
                "available": True,
                "layer": layer_used,
                "origin_x": float(cm["origin_x"]),
                "origin_y": float(cm["origin_y"]),
                "resolution": resolution,
                "width": width,
                "height": height,
                "encoding": "uint8_row_major",
                "unknown": 255,
                "data_b64": base64.b64encode(u8.tobytes()).decode("ascii"),
                "nav_backend": cfg.nav_backend,
            }

        raise ValueError(f"unknown command: {cmd!r}")

    def _add_location(self, command, runtime):
        store = self._locations()
        if "pose" in command or "x" in command:
            pose = command.get("pose", command)
            return store.add(
                str(command["name"]),
                float(pose["x"]),
                float(pose["y"]),
                float(pose.get("theta", 0.0)),
            )
        # Default to the robot's current pose in the map.
        mgr = runtime.manager
        cur = None
        getter = getattr(mgr, "get_pose_in_map", None)
        if callable(getter):
            cur = getter()
        if cur is None:
            node = getattr(mgr, "node", None)
            cur = node.get_pose_in_map() if node else None
        if cur is None:
            raise RuntimeError("current pose unavailable; provide an explicit pose")
        return store.add(str(command["name"]), cur.x, cur.y, cur.theta)

    # -- suspend / resume (cancel + remembered goal) -------------------------
    def _snapshot_active_goal(self) -> Optional[_SuspendedNav]:
        """Capture the in-flight builtin or simple-nav target, if any."""
        name = self._active_goal_name
        simple = self._simple_nav_status
        if simple.get("state") == "active":
            target = simple.get("target")
            if isinstance(target, Mapping):
                return _SuspendedNav(
                    x=float(target["x"]),
                    y=float(target["y"]),
                    theta=float(target.get("theta", 0.0)),
                    name=name,
                    motion="simple",
                )

        status: Mapping = {}
        try:
            status = self._require_runtime().manager.nav_status()
        except Exception:  # noqa: BLE001 - fall through to plan execution
            status = {}
        nav_active = bool(status.get("active")) or str(
            status.get("state") or ""
        ).lower() in ("active",)
        goal = status.get("goal") if isinstance(status.get("goal"), Mapping) else None
        if nav_active and goal is not None:
            return _SuspendedNav(
                x=float(goal["x"]),
                y=float(goal["y"]),
                theta=float(goal.get("theta", 0.0)),
                name=name,
                motion="builtin",
            )

        execution = self._plan_execution
        if execution is not None and execution.state not in _TERMINAL_PLAN_STATES:
            pose2d = conv.viam_pose_to_pose2d(
                execution.destination.x,
                execution.destination.y,
                execution.destination.theta,
            )
            return _SuspendedNav(
                x=pose2d.x,
                y=pose2d.y,
                theta=pose2d.theta,
                name=name,
                motion="builtin",
            )
        return None

    async def _suspend_nav(
        self, command: Mapping[str, ValueTypes]
    ) -> Mapping[str, ValueTypes]:
        """Cancel motion but remember the goal for a later ``resume``."""
        reason = command.get("reason")
        reason_s = str(reason) if reason is not None else None
        snapshot = self._snapshot_active_goal()
        if snapshot is None and self._suspended is not None:
            # Already suspended; refresh optional reason and return current.
            if reason_s:
                self._suspended.reason = reason_s
            return {
                "status": "suspended",
                "already_suspended": True,
                "goal": self._suspended.to_dict(),
            }
        if snapshot is None:
            raise ValueError("nothing to suspend (no active builtin or simple-nav goal)")

        snapshot.reason = reason_s
        runtime = self._require_runtime()
        # Keep the label on the suspended snapshot; clear live goal name.
        self._active_goal_name = None
        await self._cancel_simple_nav()
        await asyncio.to_thread(runtime.manager.cancel)
        if self._plan_execution is not None:
            self._record_plan_state(
                self._plan_execution,
                PlanState.PLAN_STATE_STOPPED,
                reason="suspended",
            )
        self._suspended = snapshot
        return {"status": "suspended", "goal": snapshot.to_dict()}

    async def _resume_nav(
        self, command: Mapping[str, ValueTypes]
    ) -> Mapping[str, ValueTypes]:
        """Re-issue the goal saved by ``suspend`` (replans from current pose)."""
        _ = command
        suspended = self._suspended
        if suspended is None:
            raise ValueError("nothing to resume (call suspend first)")

        self._suspended = None
        self._active_goal_name = suspended.name
        target = {
            "x": suspended.x,
            "y": suspended.y,
            "theta": suspended.theta,
            **({"name": suspended.name} if suspended.name else {}),
        }

        if suspended.motion == "simple":
            result = await self._start_simple_go(
                suspended.x,
                suspended.y,
                suspended.theta,
                {"wait": False},
            )
            out = dict(result)
            out["resumed"] = True
            out["target"] = target
            return out

        cfg = self._require_cfg()
        runtime = self._require_runtime()
        dest = _pose2d_to_viam_pose_msg(
            conv.Pose2D(suspended.x, suspended.y, suspended.theta)
        )
        execution_id = str(uuid.uuid4())
        plan_id = str(uuid.uuid4())
        execution = _PlanExecution(
            execution_id=execution_id,
            plan_id=plan_id,
            component_name=cfg.base,
            destination=dest,
            state=PlanState.PLAN_STATE_IN_PROGRESS,
        )
        execution.status_history.append(
            PlanStatus(
                state=PlanState.PLAN_STATE_IN_PROGRESS,
                timestamp=_utcnow_timestamp(),
            )
        )
        self._plan_execution = execution
        self._upsert_plan_status_history(execution)
        await self._cancel_simple_nav()
        await asyncio.to_thread(
            runtime.manager.navigate, suspended.x, suspended.y, suspended.theta
        )
        self._sync_plan_state_from_nav()
        return {
            "status": "navigating",
            "resumed": True,
            "target": target,
            "execution_id": execution_id,
        }

    # -- simple closed-loop navigation (map frame) ---------------------------
    async def _start_simple_go(
        self,
        x: float,
        y: float,
        theta: float,
        command: Mapping[str, ValueTypes],
    ) -> Mapping[str, ValueTypes]:
        wait = command.get("wait", True)
        velocity = command.get("velocity_mps")
        velocity_mps = float(velocity) if velocity is not None else None
        target = {"x": x, "y": y, "theta": theta}
        # Callers own cancellation of any previous simple-nav run: _simple_go_to
        # must never call _cancel_simple_nav itself, or a background run would
        # cancel its own task handle and drop it (making it uncancelable later).
        await self._cancel_simple_nav()
        if wait:
            await self._simple_go_to(x, y, theta, velocity_mps=velocity_mps)
            return {
                "status": self._simple_nav_status.get("state", "idle"),
                "motion": "simple",
                "target": target,
            }
        self._simple_nav_task = asyncio.create_task(
            self._simple_go_to(x, y, theta, velocity_mps=velocity_mps)
        )
        return {"status": "navigating", "motion": "simple", "target": target}

    async def _cancel_simple_nav(self) -> None:
        was_active = self._simple_nav_status.get("state") == "active"
        task = self._simple_nav_task
        had_running_task = task is not None and not task.done()
        if self._simple_nav_cancel is not None:
            self._simple_nav_cancel.set()
        if had_running_task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, SimpleMotionCanceled):
                pass
        self._simple_nav_task = None
        self._simple_nav_cancel = None
        # Only zero the base when simple nav was actually running. Always
        # stopping here races builtin SetVelocity and can wipe angular mid-turn.
        if was_active or had_running_task:
            await self._stop_base()
        if was_active:
            self._simple_nav_status = {"state": "canceled", "motion": "simple"}

    async def _stop_base(self) -> None:
        base = self._base
        if base is None:
            return
        try:
            node = self._require_runtime().manager.node
        except Exception:  # noqa: BLE001 - stop must still zero the base
            node = None
        if node is not None:
            node.record_cmd_vel(0.0, 0.0, 0.0, source="simple_stop")
        await base.set_velocity(
            linear=Vector3(x=0, y=0, z=0),
            angular=Vector3(x=0, y=0, z=0),
        )

    async def _test_drive(
        self, command: Mapping[str, ValueTypes]
    ) -> Mapping[str, ValueTypes]:
        """Send one body-frame cmd through the drive path, then stop."""
        runtime = self._require_runtime()
        node = getattr(runtime.manager, "node", None)
        io = node._io if node is not None else None

        vx = float(command.get("vx", command.get("body_vx_mps", command.get("ros_vx_mps", 0.0))))
        vy = float(command.get("vy", command.get("body_vy_mps", 0.0)))
        vtheta = float(command.get("vtheta", command.get("body_vtheta_rad_s", 0.0)))
        if (
            "angular_z_deg_s" in command
            and "vtheta" not in command
            and "body_vtheta_rad_s" not in command
        ):
            vtheta = math.radians(float(command["angular_z_deg_s"]))
        duration_s = max(0.1, min(float(command.get("duration_s", 1.5)), 5.0))

        lx, ly, ang_deg_s = body_twist_to_viam_set_velocity(
            vx, vy, vtheta, runtime.slam_cfg.base_velocity_convention
        )
        if io is not None:
            await io.drive_base(vx, vy, vtheta)
            await asyncio.sleep(duration_s)
            await io.stop_base()
        else:
            # Drive the Viam base directly.
            base = self._base
            if base is None:
                raise RuntimeError("base unavailable")
            await base.set_velocity(
                linear=Vector3(x=lx, y=ly, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=ang_deg_s),
            )
            await asyncio.sleep(duration_s)
            await base.set_velocity(
                linear=Vector3(x=0.0, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            )
        return {
            "status": "ok",
            "sent": {
                "body_vx_mps": vx,
                "body_vy_mps": vy,
                "body_vtheta_rad_s": vtheta,
                "viam_linear_x_mm_s": lx,
                "viam_linear_y_mm_s": ly,
                "viam_angular_z_deg_s": ang_deg_s,
            },
            "duration_s": duration_s,
        }

    async def _plan_preview(self, command: Mapping, mgr) -> dict:
        """Run path planning and cache the result for execute_plan."""
        from ..geom import conversions as conv

        x = float(command["x"])
        y = float(command["y"])
        theta = float(command.get("theta", 0.0))
        planner_id = str(command.get("planner_id", "GridBased"))
        timeout_s = float(command.get("timeout_s", 20.0))
        max_points = int(command.get("max_points", 400))
        start = None
        if "start" in command and isinstance(command["start"], Mapping):
            s = command["start"]
            start = conv.Pose2D(float(s["x"]), float(s["y"]), float(s.get("theta", 0.0)))
        preview = await asyncio.to_thread(
            mgr.compute_path,
            x,
            y,
            theta,
            planner_id=planner_id,
            start=start,
            timeout_s=timeout_s,
            max_points=max_points,
        )
        self._last_preview_plan = preview
        return {"status": "planned" if preview.get("feasible") else "infeasible", **preview}

    async def _simple_go_to(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        velocity_mps: Optional[float] = None,
    ) -> None:
        cfg = self._require_cfg()
        runtime = self._require_runtime()
        base = self._base
        if base is None:
            raise RuntimeError("navigation base dependency missing")

        # Stop any in-flight builtin goal; the caller has already canceled any
        # previous simple-nav run (see _start_simple_go).
        await asyncio.to_thread(runtime.manager.cancel)

        from ..geom import conversions as conv

        goal = conv.Pose2D(x, y, theta)
        motion_cfg = config_from_nav(
            max_vel_x=cfg.max_vel_x,
            max_vel_theta=cfg.max_vel_theta,
            yaw_tolerance_rad=cfg.builtin.yaw_goal_tolerance,
            min_linear_mps=cfg.min_cmd_vel_x,
            min_angular_rad_s=cfg.min_cmd_vel_theta,
        )
        cancel_event = asyncio.Event()
        self._simple_nav_cancel = cancel_event
        self._simple_nav_status = {
            "state": "active",
            "motion": "simple",
            "target": {"x": x, "y": y, "theta": theta},
        }

        convention = runtime.slam_cfg.base_velocity_convention

        async def _set_velocity(vx: float, vy: float, vtheta: float) -> None:
            node = getattr(runtime.manager, "node", None)
            if node is not None:
                node.record_cmd_vel(vx, vy, vtheta, source="simple")
            lx, ly, ang_deg_s = body_twist_to_viam_set_velocity(vx, vy, vtheta, convention)
            await base.set_velocity(
                linear=Vector3(x=lx, y=ly, z=0),
                angular=Vector3(x=0, y=0, z=ang_deg_s),
            )

        def _on_progress(progress: dict) -> None:
            prev = self._simple_nav_status.get("obstacle")
            self._simple_nav_status.update(progress)
            new_state = progress.get("obstacle")
            if new_state != prev and new_state in ("avoid", "slow", "no_scan"):
                clearance = progress.get("forward_clearance_m")
                if new_state == "no_scan":
                    LOGGER.warning(
                        "simple nav: no fresh lidar scan; suppressing forward motion"
                    )
                else:
                    LOGGER.info(
                        f"simple nav: obstacle {new_state} "
                        f"(forward clearance {clearance} m)"
                    )

        obstacle_cfg = ObstacleConfig(
            enabled=cfg.simple_avoid_obstacles,
            stop_distance_m=cfg.simple_stop_distance,
            slow_distance_m=cfg.simple_slow_distance,
            max_age_s=cfg.simple_scan_max_age,
        )

        def _get_scan():
            return runtime.manager.get_base_scan(obstacle_cfg.max_age_s)

        try:
            await drive_to_pose(
                goal=goal,
                get_pose=runtime.manager.get_pose_in_map,
                set_velocity=_set_velocity,
                stop=self._stop_base,
                cfg=motion_cfg,
                linear_mps=velocity_mps,
                cancel_event=cancel_event,
                on_progress=_on_progress,
                get_scan=_get_scan,
                obstacle=obstacle_cfg,
            )
            self._simple_nav_status = {
                "state": "succeeded",
                "motion": "simple",
                "target": {"x": x, "y": y, "theta": theta},
            }
        except SimpleMotionCanceled:
            self._simple_nav_status = {
                "state": "canceled",
                "motion": "simple",
                "target": {"x": x, "y": y, "theta": theta},
            }
            raise
        except SimpleMotionError as exc:
            self._simple_nav_status = {
                "state": "failed",
                "motion": "simple",
                "error": str(exc),
                "target": {"x": x, "y": y, "theta": theta},
            }
            raise RuntimeError(str(exc)) from exc
        finally:
            # Only clear state that still belongs to this run: a newer goal may
            # already have installed its own cancel event / task handle.
            if self._simple_nav_cancel is cancel_event:
                self._simple_nav_cancel = None
            if self._simple_nav_task is asyncio.current_task():
                self._simple_nav_task = None


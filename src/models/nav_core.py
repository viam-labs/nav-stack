"""Shared navigation-service core for the nav-stack Nav2 models.

Both navigation models — the built-in ``viam-labs:nav-stack:navigation`` (which
borrows the SLAM model's in-process ROS runtime) and the external-SLAM
``viam-labs:nav-stack:navigation-external`` (which builds its own runtime around
an arbitrary ``rdk:service:slam`` dependency) — share the Motion API
(``MoveOnMap`` / plan queries), the full DoCommand surface, Nav2 params
generation, and simple closed-loop motion. That logic lives here in
``NavServiceBase``; the concrete models differ only in how they obtain a
``SlamRuntime`` (``_resolve_runtime``) and how they stand it up (``reconfigure``).
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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

from ..config import OMNI, Nav2Config, NavConfig, ros_cmd_vel_to_viam_linear_mm_s
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
from ..ros import conversions as conv

LOGGER = getLogger(__name__)

_PARAMS_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "params" / "nav2_params.yaml"
_BT_FALLBACK = (
    Path(__file__).resolve().parent.parent.parent
    / "params"
    / "navigate_to_pose_w_replanning_and_recovery.xml"
)

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
    motion: str = "nav2"  # "nav2" | "simple"
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
    """Runtime-agnostic Nav2 orchestration shared by the navigation models.

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
        self._last_mppi_profile: dict = {}
        # Set by ``suspend``; cleared by ``resume``, ``cancel``, ``stop_plan``,
        # or any new navigate / MoveOnMap / simple go.
        self._suspended: Optional[_SuspendedNav] = None
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
                "(Nav2 costmaps still use live lidar)"
            )
            self._logged_motion_ignored = True

        pose2d = conv.viam_pose_to_pose2d(destination.x, destination.y, destination.theta)
        runtime = self._require_runtime()
        mgr = runtime.manager

        # Preview-only: plan with Nav2's ComputePathToPose, do not drive.
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
            # Explicit stop_plan / superseded — do not overwrite from Nav2.
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

    # -- nav2 params ---------------------------------------------------------
    def _write_nav2_params(self, cfg: NavConfig) -> Path:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML required to generate Nav2 params") from exc

        with open(_PARAMS_TEMPLATE) as fh:
            params = yaml.safe_load(fh)

        overrides = {
            "robot_radius": cfg.robot_radius,
            "inflation_radius": cfg.inflation_radius,
            "max_vel_x": cfg.max_vel_x,
            "max_vel_y": cfg.max_vel_y if cfg.kinematics == OMNI else 0.0,
            "max_vel_theta": cfg.max_vel_theta,
            "min_vel_x": -cfg.max_vel_x,
            "min_vel_y": -cfg.max_vel_y if cfg.kinematics == OMNI else 0.0,
            "acc_lim_x": cfg.acc_lim_x,
            "acc_lim_theta": cfg.acc_lim_theta,
            "holonomic_robot": cfg.kinematics == OMNI,
            **cfg.nav2.to_override_dict(),
        }
        _apply_nav2_tuning(params, overrides)
        _apply_velocity_limits(params, cfg)
        user_params: dict = {}
        if cfg.nav2_params:
            user_params = _normalize_nav2_user_params(dict(cfg.nav2_params), params)
            _deep_merge(params, user_params)
            # User may override FollowPath.vx_min after our wiring; keep the
            # smoother's reverse cap aligned so it cannot exceed MPPI again.
            _sync_smoother_reverse_to_mppi(params)
        # DiffDrive skid-steer cannot use stock MPPI — it converges to vx=0
        # micro-yaw. Swap to Regulated Pure Pursuit; user_params re-merge on
        # top so explicit nav2_params overrides survive the swap.
        _apply_diffdrive_controller(params, cfg, user_params)
        _apply_local_costmap_size(params, cfg.nav2)
        _sync_mppi_model_dt(params)

        runtime = self._resolve_runtime()
        _set_obstacle_sources(params, len(runtime.slam_cfg.lidars))
        runtime_dir = Path(runtime.slam_cfg.maps_dir).expanduser() / ".runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        # Small DiffDrive bases follow the path with a short lookahead, so raw
        # NavFn grid zigzag feeds curvature noise straight into cmd_vel; run
        # the path through smoother_server. Cart / omni BTs stay unchanged.
        # NavigateRecovery retries come from nav2.navigate_recovery_retries
        # (default 4) — BackUp/Spin/clear then replan when stuck near obstacles.
        bt_path = _write_nav2_bt_xml(
            runtime_dir,
            cfg.nav2,
            smooth_path=cfg.kinematics != OMNI and _is_small_base(cfg),
        )
        try:
            bt_params = params.setdefault("bt_navigator", {}).setdefault(
                "ros__parameters", {}
            )
            # Honor an explicit user BT path from nav2_params; otherwise point
            # at the tuned tree generated from replan_frequency / recovery knobs.
            if not bt_params.get("default_nav_to_pose_bt_xml"):
                bt_params["default_nav_to_pose_bt_xml"] = str(bt_path)
        except (AttributeError, TypeError):
            pass
        _validate_nav2_params_structure(params)
        out = runtime_dir / "nav2_params.yaml"
        with open(out, "w") as fh:
            yaml.safe_dump(params, fh, sort_keys=False)
        self._last_mppi_profile = _mppi_profile_snapshot(params)
        return out

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
        node = runtime.manager.node
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
        # navigate/cancel/status run ros2 CLI subprocesses and TF waits (seconds
        # on a Pi). They must not run on the module event loop: the bridge
        # marshals odom/lidar reads and cmd_vel onto this loop, so blocking here
        # stalls TF/scans (Nav2 extrapolation errors) and deadlocks stop_base.
        if cmd == "navigate_to_location":
            loc = self._locations().get(str(command["name"]))
            self._suspended = None
            self._active_goal_name = loc.name
            await asyncio.to_thread(mgr.navigate, loc.x, loc.y, loc.theta)
            return {"status": "navigating", "target": loc.to_dict()}
        if cmd == "navigate_to_point":
            x = float(command["x"])
            y = float(command["y"])
            theta = float(command.get("theta", 0.0))
            self._suspended = None
            self._active_goal_name = None
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
                status.update(mgr.nav2_diagnostics())
                simple = dict(self._simple_nav_status)
                status["simple_nav"] = simple
                if simple.get("state") == "active":
                    status["active"] = True
                    status["motion"] = "simple"
                    # Prefer simple-nav target when active (Nav2 goal may be stale).
                    target = simple.get("target")
                    if isinstance(target, dict):
                        status["goal"] = dict(target)
                goal = status.get("goal")
                if isinstance(goal, dict) and self._active_goal_name:
                    goal = dict(goal)
                    goal["name"] = self._active_goal_name
                    status["goal"] = goal
                status["localization_check"] = dict(runtime.localization_check)
                status["mppi_profile"] = dict(self._last_mppi_profile)
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
        if cmd == "start_nav2":
            cfg = self._require_cfg()
            runtime.manager.set_nav_config(cfg)
            params_path = self._write_nav2_params(cfg)
            await asyncio.to_thread(runtime.manager.ensure_nav2, cfg, params_path)
            return {"status": "nav2_started", **runtime.manager.nav2_diagnostics()}
        if cmd == "restart_nav2":
            # Unconditional stop + start: guarantees regenerated params are
            # loaded even when Nav2 currently looks healthy.
            cfg = self._require_cfg()
            runtime.manager.set_nav_config(cfg)
            params_path = self._write_nav2_params(cfg)

            def _restart():
                runtime.manager.stop_nav2()
                runtime.manager.ensure_nav2(cfg, params_path)

            await asyncio.to_thread(_restart)
            return {"status": "nav2_restarted", **runtime.manager.nav2_diagnostics()}

        if cmd == "get_costmap":
            # Live Nav2 global costmap (falls back to /map) as a compact uint8
            # occupancy grid in map frame — for operator UIs overlaying the SLAM map.
            from ..runtime import get_bridge

            bridge = get_bridge(self.name)
            if bridge is None:
                return {"available": False, "reason": "bridge_unavailable"}

            def _fetch():
                bridge.enable_viz(8)
                return bridge.viz_snapshot()

            snap = await asyncio.to_thread(_fetch)
            cm = snap.get("costmap") or snap.get("map")
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
            # 255 = unknown (-1); 0 free; 1..100 cost (Nav2 OccupancyGrid convention).
            u8 = np.where(grid < 0, 255, np.clip(grid, 0, 100)).astype(np.uint8)
            return {
                "available": True,
                "origin_x": float(cm["origin_x"]),
                "origin_y": float(cm["origin_y"]),
                "resolution": resolution,
                "width": width,
                "height": height,
                "encoding": "uint8_row_major",
                "unknown": 255,
                "data_b64": base64.b64encode(u8.tobytes()).decode("ascii"),
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
        node = runtime.manager.node
        cur = node.get_pose_in_map() if node else None
        if cur is None:
            raise RuntimeError("current pose unavailable; provide an explicit pose")
        return store.add(str(command["name"]), cur.x, cur.y, cur.theta)

    # -- suspend / resume (cancel + remembered goal) -------------------------
    def _snapshot_active_goal(self) -> Optional[_SuspendedNav]:
        """Capture the in-flight Nav2 or simple-nav target, if any."""
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
                motion="nav2",
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
                motion="nav2",
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
            raise ValueError("nothing to suspend (no active Nav2 or simple-nav goal)")

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

    # -- simple closed-loop navigation (map frame, no Nav2) ------------------
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
        if wait:
            await self._simple_go_to(x, y, theta, velocity_mps=velocity_mps)
            return {
                "status": self._simple_nav_status.get("state", "idle"),
                "motion": "simple",
                "target": target,
            }
        await self._cancel_simple_nav()
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
        # stopping here races Nav2 SetVelocity and can wipe angular mid-turn.
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
        """Send one ROS-body cmd through the same IO path Nav2 uses, then stop."""
        runtime = self._require_runtime()
        io = runtime.manager.node._io if runtime.manager.node else None
        if io is None:
            raise RuntimeError("bridge IO unavailable")

        vx = float(command.get("vx", command.get("ros_vx_mps", 0.0)))
        vy = float(command.get("vy", command.get("ros_vy_mps", 0.0)))
        vtheta = float(command.get("vtheta", command.get("ros_vtheta_rad_s", 0.0)))
        if (
            "angular_z_deg_s" in command
            and "vtheta" not in command
            and "ros_vtheta_rad_s" not in command
        ):
            vtheta = math.radians(float(command["angular_z_deg_s"]))
        duration_s = max(0.1, min(float(command.get("duration_s", 1.5)), 5.0))

        lx, ly = ros_cmd_vel_to_viam_linear_mm_s(
            vx, vy, runtime.slam_cfg.base_velocity_convention
        )
        await io.drive_base(vx, vy, vtheta)
        await asyncio.sleep(duration_s)
        await io.stop_base()
        return {
            "status": "ok",
            "sent": {
                "ros_vx_mps": vx,
                "ros_vy_mps": vy,
                "ros_vtheta_rad_s": vtheta,
                "viam_linear_x_mm_s": lx,
                "viam_linear_y_mm_s": ly,
                "viam_angular_z_deg_s": math.degrees(vtheta),
            },
            "duration_s": duration_s,
        }

    async def _plan_preview(self, command: Mapping, mgr) -> dict:
        """Run Nav2 ComputePathToPose and cache the result for execute_plan."""
        from ..ros import conversions as conv

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

        await self._cancel_simple_nav()
        await asyncio.to_thread(runtime.manager.cancel)

        from ..ros import conversions as conv

        goal = conv.Pose2D(x, y, theta)
        motion_cfg = config_from_nav(
            max_vel_x=cfg.max_vel_x,
            max_vel_theta=cfg.max_vel_theta,
            yaw_tolerance_rad=cfg.nav2.yaw_goal_tolerance,
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
            node = runtime.manager.node
            if node is not None:
                node.record_cmd_vel(vx, vy, vtheta, source="simple")
            lx, ly = ros_cmd_vel_to_viam_linear_mm_s(vx, vy, convention)
            await base.set_velocity(
                linear=Vector3(x=lx, y=ly, z=0),
                angular=Vector3(x=0, y=0, z=math.degrees(vtheta)),
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
            self._simple_nav_cancel = None
            self._simple_nav_task = None


# ---------------------------------------------------------------------------
# Nav2 params generation helpers (pure functions, shared by both models)
# ---------------------------------------------------------------------------
def _find_template_section_paths(template: Mapping, key: str) -> list:
    """Return paths to every nested mapping named ``key`` inside the template.

    Used to relocate plugin-section overrides (e.g. ``FollowPath``,
    ``inflation_layer``) that users put at the top level of ``nav2_params``.
    """
    paths: list = []

    def _walk(node: Mapping, path: tuple) -> None:
        for k, v in node.items():
            if not isinstance(v, Mapping):
                continue
            if k == key:
                paths.append(path + (k,))
            else:
                _walk(v, path + (k,))

    _walk(template, ())
    return paths


def _coerce_types_to_template(user_node: dict, template_node: Mapping) -> None:
    """Cast user override leaf types to match the template's declared types.

    Viam config attributes arrive through protobuf Structs, which turn every
    JSON number into a double. ROS 2 parameters are strictly typed, so a user
    writing ``"default_server_timeout": 200`` produces ``200.0`` in the merged
    YAML and the owning node dies at startup with "expected [integer] got
    [double]". Wherever the template declares an integer at the same path,
    whole-number floats are cast back to int (and ints are widened to float
    where the template declares a double). Bools and non-integral floats are
    left untouched; params absent from the template cannot be inferred.
    """
    for key, value in user_node.items():
        tmpl = template_node.get(key)
        if isinstance(value, dict) and isinstance(tmpl, Mapping):
            _coerce_types_to_template(value, tmpl)
            continue
        if isinstance(value, bool) or isinstance(tmpl, bool):
            continue
        if isinstance(tmpl, int) and isinstance(value, float) and value.is_integer():
            user_node[key] = int(value)
        elif isinstance(tmpl, float) and isinstance(value, int):
            user_node[key] = float(value)


def _normalize_nav2_user_params(user_params: dict, template: Mapping) -> dict:
    """Rewrite user ``nav2_params`` into the strict rcl-compatible structure.

    Handles the two natural-but-invalid forms config authors write:

    * node overrides missing the ``ros__parameters`` wrapper, e.g.
      ``{"controller_server": {"max_vel_x": 1}}``
    * plugin sections hoisted to the top level, e.g. ``{"FollowPath": {...}}``
      which really lives at ``controller_server/ros__parameters/FollowPath``

    Merging either form unfixed corrupts the generated params file and crashes
    every Nav2 node at startup (rcl: "Cannot have a value before ros__parameters").

    Leaf values are then type-coerced against the template (protobuf delivers
    all numbers as doubles; ROS 2 params are strictly typed).
    """
    normalized: dict = {}
    relocated: dict = {}
    for key, value in user_params.items():
        if not isinstance(value, Mapping):
            normalized[key] = value
            continue
        value = dict(value)
        template_node = template.get(key)
        if template_node is None:
            # Not a top-level node: relocate plugin sections (FollowPath,
            # inflation_layer, ...) to wherever they live in the template.
            section_paths = _find_template_section_paths(template, key)
            if section_paths:
                for path in section_paths:
                    cursor = relocated
                    for part in path[:-1]:
                        cursor = cursor.setdefault(part, {})
                    _deep_merge(
                        cursor.setdefault(path[-1], {}), value
                    )
                continue
            normalized[key] = value
            continue
        doubled = (
            isinstance(template_node, Mapping) and key in template_node
        )  # local_costmap/global_costmap use a doubled namespace
        if doubled and "ros__parameters" not in value and key not in value:
            value = {key: value}
        inner_template = template_node.get(key) if doubled else template_node
        target = value.get(key) if doubled and key in value else value
        if (
            isinstance(inner_template, Mapping)
            and "ros__parameters" in inner_template
            and isinstance(target, dict)
            and "ros__parameters" not in target
        ):
            wrapped = {"ros__parameters": target}
            if doubled:
                value = {key: wrapped}
            else:
                value = wrapped
        normalized[key] = value
    if relocated:
        _deep_merge(normalized, relocated)
    _coerce_types_to_template(normalized, template)
    return normalized


def _validate_nav2_params_structure(params: Mapping) -> None:
    """Reject param trees the rcl YAML parser would refuse to load.

    Every top-level node entry must nest all values under ``ros__parameters``
    (possibly through namespace dicts). Failing fast here surfaces the offending
    key instead of letting every Nav2 server die at launch with a parse error.
    """

    def _check(node, path: str) -> None:
        if not isinstance(node, Mapping):
            raise ValueError(
                f"invalid nav2_params: {path!r} must be a mapping that nests values "
                "under 'ros__parameters'"
            )
        if "ros__parameters" in node:
            extras = [k for k in node if k != "ros__parameters"]
            if extras:
                raise ValueError(
                    f"invalid nav2_params: {path!r} has keys {extras} outside "
                    "'ros__parameters'"
                )
            return
        for key, value in node.items():
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"invalid nav2_params: value {path + '/' + str(key)!r} appears "
                    "before 'ros__parameters'"
                )
            _check(value, path + "/" + str(key))

    for top, value in params.items():
        _check(value, str(top))


def _sync_mppi_model_dt(params: dict) -> None:
    """Keep MPPI ``model_dt`` >= the controller period (1/controller_frequency).

    MPPI's on_configure() raises "Controller period more then model dt" when
    1/controller_frequency > model_dt; that fails the controller_server
    lifecycle transition and aborts the whole Nav2 bringup. Lowering
    controller_frequency (e.g. 10 -> 5 Hz on a loaded Pi) without also raising
    model_dt trips this. Snap model_dt up to the controller period so the two
    can never drift out of sync, regardless of how the frequency was set
    (top-level nav2 config or raw nav2_params override).
    """
    try:
        cs = params["controller_server"]["ros__parameters"]
        freq = float(cs["controller_frequency"])
        fp = cs["FollowPath"]
    except (KeyError, TypeError, ValueError):
        return
    if not isinstance(fp, dict) or freq <= 0:
        return
    if "nav2_mppi_controller" not in str(fp.get("plugin", "")):
        return
    period = 1.0 / freq
    try:
        current = float(fp.get("model_dt", 0.0))
    except (TypeError, ValueError):
        current = 0.0
    if current < period:
        fp["model_dt"] = period


def _apply_local_costmap_size(params: dict, nav2_cfg: Nav2Config) -> None:
    """Set rolling local costmap dimensions (Jazzy requires integer width/height)."""
    try:
        lc = params["local_costmap"]["local_costmap"]["ros__parameters"]
    except (KeyError, TypeError):
        return
    lc["width"] = int(nav2_cfg.local_costmap_width)
    lc["height"] = int(nav2_cfg.local_costmap_height)


def _resolve_bt_template() -> Path:
    """Prefer the distro-installed Nav2 BT; fall back to the shipped Jazzy copy."""
    distro = os.environ.get("ROS_DISTRO", "").strip()
    if distro:
        candidate = Path(
            f"/opt/ros/{distro}/share/nav2_bt_navigator/behavior_trees"
            "/navigate_to_pose_w_replanning_and_recovery.xml"
        )
        if candidate.is_file():
            return candidate
    return _BT_FALLBACK


def _inject_smooth_path_bt(text: str) -> str:
    """Wrap the ``ComputePathToPose`` action with a ``SmoothPath`` step.

    ``smoothed_path`` overwrites ``{path}`` in place; ``ForceSuccess`` keeps a
    smoothing hiccup from failing navigation (the raw path is already on the
    blackboard). Works on both the distro-installed tree and the shipped
    fallback since both use a self-closing ``<ComputePathToPose .../>``.
    """
    if "<SmoothPath" in text:
        return text
    m = re.search(r"<ComputePathToPose\b[^>]*/>", text)
    if m is None:
        return text
    smooth = (
        '<SmoothPath unsmoothed_path="{path}" smoothed_path="{path}" '
        'smoother_id="simple_smoother" max_smoothing_duration="1.0" '
        'check_for_collisions="false"/>'
    )
    wrapped = (
        '<Sequence name="ComputeAndSmoothPath">'
        f"{m.group(0)}<ForceSuccess>{smooth}</ForceSuccess>"
        "</Sequence>"
    )
    return text[: m.start()] + wrapped + text[m.end():]


def _tune_nav2_bt_xml(
    text: str,
    *,
    replan_hz: float,
    navigate_recovery_retries: int,
    recovery_wait_duration: float,
    smooth_path: bool = False,
) -> str:
    """Rewrite replan rate / recovery patience fields in a Nav2 BT XML string."""
    hz = max(0.1, float(replan_hz))
    retries = max(0, int(navigate_recovery_retries))
    wait_s = max(0.0, float(recovery_wait_duration))
    if smooth_path:
        text = _inject_smooth_path_bt(text)
    text = re.sub(
        r'(RateController\s+hz=")[^"]+(")',
        rf"\g<1>{hz:.1f}\g<2>",
        text,
        count=1,
    )
    text = re.sub(
        r'(RecoveryNode\s+number_of_retries=")[^"]+("\s+name="NavigateRecovery")',
        rf"\g<1>{retries}\g<2>",
        text,
        count=1,
    )
    text = re.sub(
        r'(Wait\s+wait_duration=")[^"]+(")',
        rf"\g<1>{wait_s:.1f}\g<2>",
        text,
        count=1,
    )
    return text


def _write_nav2_bt_xml(
    runtime_dir: Path, nav2_cfg: Nav2Config, *, smooth_path: bool = False
) -> Path:
    """Write a navigate-to-pose BT tuned from ``nav2`` config into ``runtime_dir``."""
    template = _resolve_bt_template()
    text = template.read_text(encoding="utf-8")
    text = _tune_nav2_bt_xml(
        text,
        replan_hz=nav2_cfg.replan_frequency,
        navigate_recovery_retries=nav2_cfg.navigate_recovery_retries,
        recovery_wait_duration=nav2_cfg.recovery_wait_duration,
        smooth_path=smooth_path,
    )
    out = runtime_dir / "navigate_to_pose_w_replanning_and_recovery.xml"
    out.write_text(text, encoding="utf-8")
    return out


def _set_obstacle_sources(params: Mapping, n_lidars: int) -> None:
    """Replace the single ``scan`` obstacle source with one per lidar (``scan_0``..).

    Each lidar publishes ``/scan_<i>`` in its own frame, letting the costmap mark
    and clear obstacles from every lidar (including units at different heights).
    """
    if n_lidars <= 1:
        return
    names = [f"scan_{i}" for i in range(n_lidars)]
    for costmap_key in ("local_costmap", "global_costmap"):
        try:
            obstacle = params[costmap_key][costmap_key]["ros__parameters"]["obstacle_layer"]
        except (KeyError, TypeError):
            continue
        template = obstacle.pop("scan", {})
        obstacle["observation_sources"] = " ".join(names)
        for i, name in enumerate(names):
            entry = dict(template)
            entry["topic"] = f"/scan_{i}"
            obstacle[name] = entry


def _apply_velocity_limits(params: dict, cfg: NavConfig) -> None:
    """Wire top-level velocity/accel attributes into MPPI + velocity_smoother.

    The flat override pass only matches identical key names (``max_vel_x``),
    but MPPI uses ``vx_max``/``wz_max`` and the smoother uses arrays — so the
    user's configured speed limits silently never reached the controller.

    Reverse is intentionally capped (``<= 0.15 m/s``) for both MPPI and the
    velocity smoother. Allowing the smoother full ``-max_vel_x`` while MPPI is
    limited lets recoveries / stale cmd bursts command hard reverse that the
    controller never intended — dangerous on skid-steer / low-traction bases.
    """
    omni = cfg.kinematics == OMNI
    vy = cfg.max_vel_y if omni else 0.0
    # Modest reverse for small overshoots; diff-drive with vx_min=0 must spin
    # fully around instead. Cap magnitude so reverse never matches full forward.
    reverse_mps = -min(float(cfg.max_vel_x), 0.15)
    try:
        fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    except (KeyError, TypeError):
        fp = None
    if isinstance(fp, dict):
        fp["motion_model"] = "Omni" if omni else "DiffDrive"
        fp["vx_max"] = cfg.max_vel_x
        fp["vx_min"] = reverse_mps
        fp["vy_max"] = vy
        fp["wz_max"] = cfg.max_vel_theta
        fp["ax_max"] = cfg.acc_lim_x
        fp["ax_min"] = -cfg.acc_lim_x
        fp["az_max"] = cfg.acc_lim_theta
    try:
        vs = params["velocity_smoother"]["ros__parameters"]
    except (KeyError, TypeError):
        vs = None
    if isinstance(vs, dict):
        vs["max_velocity"] = [cfg.max_vel_x, vy, cfg.max_vel_theta]
        vs["min_velocity"] = [reverse_mps, -vy, -cfg.max_vel_theta]
        vs["max_accel"] = [cfg.acc_lim_x, cfg.acc_lim_x if omni else 0.0, cfg.acc_lim_theta]
        # Allow braking harder than accelerating (safety), but keep it bounded
        # so the smoother actually smooths instead of passing jerks through.
        vs["max_decel"] = [
            -1.5 * cfg.acc_lim_x,
            -1.5 * cfg.acc_lim_x if omni else 0.0,
            -1.5 * cfg.acc_lim_theta,
        ]


def _sync_smoother_reverse_to_mppi(params: dict) -> None:
    """Keep velocity_smoother min_velocity[0] aligned with controller reverse."""
    try:
        fp = params["controller_server"]["ros__parameters"]["FollowPath"]
        vs = params["velocity_smoother"]["ros__parameters"]
    except (KeyError, TypeError):
        return
    if not isinstance(fp, dict) or not isinstance(vs, dict):
        return
    if "vx_min" in fp:
        vx_min = float(fp["vx_min"])
    elif "min_linear_vel" in fp:
        vx_min = float(fp["min_linear_vel"])
    else:
        return
    min_vel = vs.get("min_velocity")
    if isinstance(min_vel, list) and min_vel:
        vs["min_velocity"] = [vx_min, *min_vel[1:]]
    elif isinstance(min_vel, tuple) and min_vel:
        vs["min_velocity"] = [vx_min, *list(min_vel[1:])]


# Bases at or below this radius get the "nimble" DiffDrive profile (Viam
# Rover class). Larger carts (MiR class) keep the carpet-tested values.
_SMALL_BASE_RADIUS_M = 0.15


def _is_small_base(cfg: NavConfig) -> bool:
    try:
        return float(cfg.robot_radius) <= _SMALL_BASE_RADIUS_M
    except (TypeError, ValueError):
        return False


def _user_subtree(user_params: Optional[Mapping], *path: str) -> Mapping:
    """Fetch a nested mapping out of normalized user nav2_params (or {})."""
    node: object = user_params or {}
    for key in path:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, Mapping) else {}


def _apply_diffdrive_controller(
    params: dict, cfg: NavConfig, user_params: Optional[Mapping] = None
) -> None:
    """Use Regulated Pure Pursuit for DiffDrive instead of MPPI.

    MPPI on this skid-steer repeatedly converges to ``vx=0`` with
    ``|ω|≈0.05–0.1``. RPP tracks a lookahead with forward motion. Also:
    open-loop smoother (CLOSED_LOOP fights carpet stiction), no RPP collision
    freeze, and a looser progress checker so weak first motion is not an
    instant abort.

    Two profiles, gated on ``robot_radius``:

    * cart (> 0.15 m, e.g. MiR): the carpet-tested values, unchanged.
      ``use_rotate_to_heading: false`` because stop-and-spin is useless there.
    * small (<= 0.15 m, e.g. Viam Rover): the cart geometry made these bases
      spin in place — ``regulated_linear_scaling_min_speed`` at half top speed
      inflated ω (RPP computes ω = v·curvature after flooring v), and the
      velocity-scaled lookahead collapsed to 0.3 m, where curvature = 2y/L²
      explodes. Longer minimum lookahead, a real speed floor, rotate-to-heading
      for large heading errors, and yaw accel the smoother can actually track.

    ``user_params`` (normalized ``nav2_params``) re-merges on top so explicit
    FollowPath / progress_checker / velocity_smoother overrides win. A user
    who sets ``FollowPath.plugin`` opts out of this profile entirely.
    """
    if cfg.kinematics == OMNI:
        return
    try:
        cs = params["controller_server"]["ros__parameters"]
    except (KeyError, TypeError):
        return
    if not isinstance(cs, dict):
        return
    user_cs = _user_subtree(user_params, "controller_server", "ros__parameters")
    user_fp = user_cs.get("FollowPath")
    if isinstance(user_fp, Mapping) and user_fp.get("plugin"):
        # Explicit controller choice: leave the merged template alone.
        return
    small = _is_small_base(cfg)
    reverse_mps = -min(float(cfg.max_vel_x), 0.15)
    if small:
        min_speed = min(0.10, 0.5 * float(cfg.max_vel_x))
        min_radius = 0.35
        lookahead, min_lookahead, max_lookahead = 0.55, 0.45, 0.9
        # Smoother slew must keep up with RPP's ω demand or pursuit limit-cycles
        # (lag distance approaching the lookahead distance = oscillation).
        yaw_accel = max(float(cfg.acc_lim_theta), 4.0 * float(cfg.max_vel_theta))
    else:
        min_speed = max(0.12, 0.5 * float(cfg.max_vel_x))
        min_radius = 0.9
        lookahead, min_lookahead, max_lookahead = 0.6, 0.3, 0.9
        yaw_accel = float(cfg.acc_lim_theta)
    # Keep both Jazzy (desired_linear_vel) and newer (max_linear_vel) names.
    cs["FollowPath"] = {
        "plugin": (
            "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
        ),
        "desired_linear_vel": cfg.max_vel_x,
        "max_linear_vel": cfg.max_vel_x,
        "min_linear_vel": reverse_mps,
        "max_angular_vel": cfg.max_vel_theta,
        "min_angular_vel": -cfg.max_vel_theta,
        "max_linear_accel": cfg.acc_lim_x,
        "max_linear_decel": -1.5 * cfg.acc_lim_x,
        "max_angular_accel": yaw_accel,
        "max_angular_decel": -1.5 * yaw_accel,
        "lookahead_dist": lookahead,
        "min_lookahead_dist": min_lookahead,
        "max_lookahead_dist": max_lookahead,
        "lookahead_time": 1.5,
        "rotate_to_heading_angular_vel": min(float(cfg.max_vel_theta), 1.0),
        "rotate_to_heading_min_angle": 0.785,
        "use_velocity_scaled_lookahead_dist": True,
        "min_approach_linear_velocity": max(0.08, min_speed * 0.5),
        "approach_velocity_scaling_dist": 0.6,
        # Global planner already avoids obstacles. Local RPP collision checks
        # often freeze the cart in inflation (cmd zero → progress fail → Spin).
        "use_collision_detection": False,
        "use_regulated_linear_velocity_scaling": True,
        "use_cost_regulated_linear_velocity_scaling": False,
        "regulated_linear_scaling_min_radius": min_radius,
        "regulated_linear_scaling_min_speed": min_speed,
        # Carts on carpet must never stop-and-spin; a small rover pivots
        # cleanly, and arcing through a >45° heading error is what spun it.
        "use_rotate_to_heading": small,
        "allow_reversing": False,
        "transform_tolerance": 10.0,
    }
    if isinstance(user_fp, Mapping) and user_fp:
        _deep_merge(cs["FollowPath"], user_fp)
    progress = cs.get("progress_checker")
    if isinstance(progress, dict):
        try:
            radius = float(progress.get("required_movement_radius", 0.25))
        except (TypeError, ValueError):
            radius = 0.25
        progress["required_movement_radius"] = min(radius, 0.15)
        try:
            allow = float(progress.get("movement_time_allowance", 10.0))
        except (TypeError, ValueError):
            allow = 10.0
        progress["movement_time_allowance"] = max(allow, 30.0)
        user_progress = user_cs.get("progress_checker")
        if isinstance(user_progress, Mapping) and user_progress:
            _deep_merge(progress, user_progress)
    try:
        vs = params["velocity_smoother"]["ros__parameters"]
    except (KeyError, TypeError):
        vs = None
    if isinstance(vs, dict):
        # CLOSED_LOOP + carpet stiction: smoother never sees motion and
        # collapses useful cmds. Open-loop ramps match test_drive.
        vs["feedback"] = "OPEN_LOOP"
        vs["deadband_velocity"] = [0.0, 0.0, 0.0]
        if small:
            accel = vs.get("max_accel")
            decel = vs.get("max_decel")
            if isinstance(accel, list) and len(accel) == 3:
                vs["max_accel"] = [accel[0], accel[1], yaw_accel]
            if isinstance(decel, list) and len(decel) == 3:
                vs["max_decel"] = [decel[0], decel[1], -1.5 * yaw_accel]
        user_vs = _user_subtree(user_params, "velocity_smoother", "ros__parameters")
        if user_vs:
            _deep_merge(vs, user_vs)
    _sync_smoother_reverse_to_mppi(params)


def _mppi_profile_snapshot(params: dict) -> dict:
    """Compact FollowPath settings for get_status diagnostics."""
    try:
        fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    except (KeyError, TypeError):
        return {}
    if not isinstance(fp, dict):
        return {}
    plugin = str(fp.get("plugin", ""))
    out: dict = {"plugin": plugin}
    if "regulated_pure_pursuit" in plugin:
        out.update(
            {
                "desired_linear_vel": fp.get(
                    "desired_linear_vel", fp.get("max_linear_vel")
                ),
                "max_angular_vel": fp.get("max_angular_vel"),
                "use_rotate_to_heading": fp.get("use_rotate_to_heading"),
                "allow_reversing": fp.get("allow_reversing"),
                "lookahead_dist": fp.get("lookahead_dist"),
                "min_lookahead_dist": fp.get("min_lookahead_dist"),
                "min_approach_linear_velocity": fp.get("min_approach_linear_velocity"),
                "use_collision_detection": fp.get("use_collision_detection"),
                "regulated_linear_scaling_min_radius": fp.get(
                    "regulated_linear_scaling_min_radius"
                ),
                "regulated_linear_scaling_min_speed": fp.get(
                    "regulated_linear_scaling_min_speed"
                ),
            }
        )
    else:
        critics = {}
        for name in (
            "PathFollowCritic",
            "PathAlignCritic",
            "PathAngleCritic",
            "PreferForwardCritic",
            "GoalCritic",
            "GoalAngleCritic",
            "VelocityDeadbandCritic",
        ):
            section = fp.get(name)
            if not isinstance(section, dict):
                continue
            critics[name] = {
                "enabled": bool(section.get("enabled", True)),
                "threshold_to_consider": section.get("threshold_to_consider"),
                "cost_weight": section.get("cost_weight"),
            }
        out.update(
            {
                "vx_max": fp.get("vx_max"),
                "vx_min": fp.get("vx_min"),
                "wz_max": fp.get("wz_max"),
                "motion_model": fp.get("motion_model"),
                "critics": critics,
            }
        )
    try:
        vs = params["velocity_smoother"]["ros__parameters"]
        if isinstance(vs, dict):
            out["smoother_feedback"] = vs.get("feedback")
            out["smoother_deadband"] = vs.get("deadband_velocity")
            out["smoother_min_velocity"] = vs.get("min_velocity")
    except (KeyError, TypeError):
        pass
    return out


def _deep_merge(obj: dict, overrides: Mapping) -> None:
    """Recursively merge ``overrides`` into nested Nav2 param dicts."""
    for key, value in overrides.items():
        if (
            key in obj
            and isinstance(obj[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(obj[key], value)
        else:
            obj[key] = value


def _apply_nav2_tuning(params: dict, overrides: Mapping) -> None:
    """Apply user tuning without clobbering unrelated ``tolerance`` keys."""
    leaf_overrides = dict(overrides)
    planner_tol = leaf_overrides.pop("tolerance", None)
    _apply_overrides(params, leaf_overrides)
    if planner_tol is not None:
        try:
            params["planner_server"]["ros__parameters"]["GridBased"]["tolerance"] = (
                planner_tol
            )
        except (KeyError, TypeError):
            pass


def _apply_overrides(obj, overrides: Mapping) -> None:
    """Recursively set any leaf key present in ``overrides`` (best-effort tuning)."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in overrides and not isinstance(value, (dict, list)):
                obj[key] = overrides[key]
            else:
                _apply_overrides(value, overrides)
    elif isinstance(obj, list):
        for item in obj:
            _apply_overrides(item, overrides)

"""Shared navigation-service core for the nav-stack Nav2 models.

Both navigation models — the built-in ``viam-labs:nav-stack:navigation`` (which
borrows the SLAM model's in-process ROS runtime) and the external-SLAM
``viam-labs:nav-stack:navigation-external`` (which builds its own runtime around
an arbitrary ``rdk:service:slam`` dependency) — share the entire DoCommand
surface, Nav2 params generation, and simple closed-loop motion. That logic lives
here in ``NavServiceBase``; the concrete models differ only in how they obtain a
``SlamRuntime`` (``_resolve_runtime``) and how they stand it up (``reconfigure``).

Keeping this API-agnostic (it operates on a ``SlamRuntime`` + ``NavConfig``, not
on a specific Viam service face) is what lets a future ``rdk:service:navigation``
model reuse it unchanged.
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Mapping, Optional

from viam.components.base import Base
from viam.logging import getLogger
from viam.proto.common import Vector3
from viam.services.generic import Generic
from viam.utils import ValueTypes

from ..config import OMNI, Nav2Config, NavConfig, ros_cmd_vel_to_viam_linear_mm_s
from ..nav import zones as zones_mod
from ..nav.locations import LocationStore
from ..nav.maps import MapHandle
from ..nav.simple_motion import (
    ObstacleConfig,
    SimpleMotionCanceled,
    SimpleMotionError,
    config_from_nav,
    drive_to_pose,
)
from ..nav.zones import ZoneStore

LOGGER = getLogger(__name__)

_PARAMS_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "params" / "nav2_params.yaml"


class NavServiceBase(Generic):
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
        if cfg.nav2_params:
            _deep_merge(
                params, _normalize_nav2_user_params(dict(cfg.nav2_params), params)
            )
        _apply_local_costmap_size(params, cfg.nav2)
        _sync_mppi_model_dt(params)

        runtime = self._resolve_runtime()
        _set_obstacle_sources(params, len(runtime.slam_cfg.lidars))
        _validate_nav2_params_structure(params)
        out = Path(runtime.slam_cfg.maps_dir).expanduser() / ".runtime" / "nav2_params.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            yaml.safe_dump(params, fh, sort_keys=False)
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
            await asyncio.to_thread(mgr.navigate, loc.x, loc.y, loc.theta)
            return {"status": "navigating", "target": loc.to_dict()}
        if cmd == "navigate_to_point":
            x = float(command["x"])
            y = float(command["y"])
            theta = float(command.get("theta", 0.0))
            await asyncio.to_thread(mgr.navigate, x, y, theta)
            return {"status": "navigating", "target": {"x": x, "y": y, "theta": theta}}
        if cmd == "go_to_location":
            loc = self._locations().get(str(command["name"]))
            return await self._start_simple_go(loc.x, loc.y, loc.theta, command)
        if cmd == "go_to_point":
            x = float(command["x"])
            y = float(command["y"])
            theta = float(command.get("theta", 0.0))
            return await self._start_simple_go(x, y, theta, command)
        if cmd == "cancel":
            await self._cancel_simple_nav()
            await asyncio.to_thread(mgr.cancel)
            return {"status": "canceled"}
        if cmd == "get_status":
            def _status():
                status = mgr.nav_status()
                status.update(mgr.nav2_diagnostics())
                simple = dict(self._simple_nav_status)
                status["simple_nav"] = simple
                if simple.get("state") == "active":
                    status["active"] = True
                    status["motion"] = "simple"
                status["localization_check"] = dict(runtime.localization_check)
                return status

            return await asyncio.to_thread(_status)
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
        if self._simple_nav_cancel is not None:
            self._simple_nav_cancel.set()
        task = self._simple_nav_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, SimpleMotionCanceled):
                pass
        self._simple_nav_task = None
        self._simple_nav_cancel = None
        await self._stop_base()
        if self._simple_nav_status.get("state") == "active":
            self._simple_nav_status = {"state": "canceled", "motion": "simple"}

    async def _stop_base(self) -> None:
        base = self._base
        if base is None:
            return
        await base.set_velocity(
            linear=Vector3(x=0, y=0, z=0),
            angular=Vector3(x=0, y=0, z=0),
        )

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


def _normalize_nav2_user_params(user_params: dict, template: Mapping) -> dict:
    """Rewrite user ``nav2_params`` into the strict rcl-compatible structure.

    Handles the two natural-but-invalid forms config authors write:

    * node overrides missing the ``ros__parameters`` wrapper, e.g.
      ``{"controller_server": {"max_vel_x": 1}}``
    * plugin sections hoisted to the top level, e.g. ``{"FollowPath": {...}}``
      which really lives at ``controller_server/ros__parameters/FollowPath``

    Merging either form unfixed corrupts the generated params file and crashes
    every Nav2 node at startup (rcl: "Cannot have a value before ros__parameters").
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
    """
    omni = cfg.kinematics == OMNI
    vy = cfg.max_vel_y if omni else 0.0
    try:
        fp = params["controller_server"]["ros__parameters"]["FollowPath"]
    except (KeyError, TypeError):
        fp = None
    if isinstance(fp, dict):
        fp["motion_model"] = "Omni" if omni else "DiffDrive"
        fp["vx_max"] = cfg.max_vel_x
        # Keep a modest reverse for goal corrections; diff-drive with no
        # reverse must rotate fully around to fix small overshoots.
        fp["vx_min"] = -min(cfg.max_vel_x, 0.15)
        fp["vy_max"] = vy
        fp["wz_max"] = cfg.max_vel_theta
        fp["ax_max"] = cfg.acc_lim_x
        fp["ax_min"] = -cfg.acc_lim_x
        fp["az_max"] = cfg.acc_lim_theta
    try:
        vs = params["velocity_smoother"]["ros__parameters"]
    except (KeyError, TypeError):
        return
    if isinstance(vs, dict):
        vs["max_velocity"] = [cfg.max_vel_x, vy, cfg.max_vel_theta]
        vs["min_velocity"] = [-cfg.max_vel_x, -vy, -cfg.max_vel_theta]
        vs["max_accel"] = [cfg.acc_lim_x, cfg.acc_lim_x if omni else 0.0, cfg.acc_lim_theta]
        # Allow braking harder than accelerating (safety), but keep it bounded
        # so the smoother actually smooths instead of passing jerks through.
        vs["max_decel"] = [
            -1.5 * cfg.acc_lim_x,
            -1.5 * cfg.acc_lim_x if omni else 0.0,
            -1.5 * cfg.acc_lim_theta,
        ]


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

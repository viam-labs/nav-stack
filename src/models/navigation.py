"""Navigation service model: ``viam-labs:nav-stack:navigation``.

A Viam generic service that wraps ROS2 Nav2. It launches Nav2 against the SLAM
service's shared ROS context and exposes, via ``DoCommand``:

* locations CRUD (named map-frame poses)
* zones CRUD (keepout + speed_limit virtual regions -> Nav2 costmap filters)
* navigation (to a named location or an arbitrary map point), cancel, and status

Physical obstacle avoidance is automatic via Nav2's costmaps (live ``/scan`` data).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar, Mapping, Optional, Sequence, cast

from typing_extensions import Self

from viam.components.base import Base
from viam.logging import getLogger
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import ValueTypes, struct_to_dict

from ..config import OMNI, Nav2Config, NavConfig
from ..nav import zones as zones_mod
from ..nav.locations import LocationStore
from ..nav.maps import MapHandle
from ..nav.zones import ZoneStore
from ..runtime import get_slam

LOGGER = getLogger(__name__)

_PARAMS_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "params" / "nav2_params.yaml"


class RosNavigation(Generic):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "nav-stack"), "navigation")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[NavConfig] = None
        self._base: Optional[Base] = None

    # -- registration --------------------------------------------------------
    @classmethod
    def new(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(
        cls, config: ServiceConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        cfg = NavConfig.from_dict(struct_to_dict(config.attributes))
        return cfg.required_dependencies(), []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = NavConfig.from_dict(struct_to_dict(config.attributes))
        self._cfg = cfg
        self._base = cast(Base, dependencies[Base.get_resource_name(cfg.base)])

        runtime = get_slam(cfg.slam_service)
        if runtime is None:
            raise RuntimeError(
                f"SLAM service {cfg.slam_service!r} not found; it must be configured "
                "and started before the navigation service"
            )
        runtime.manager.set_nav_config(cfg)
        params_path = self._write_nav2_params(cfg)
        runtime.manager.start_nav2(cfg, params_path)
        self._refresh_zone_masks()
        LOGGER.info(f"nav-stack navigation '{self.name}' configured ({cfg.kinematics})")

    def _require_cfg(self) -> NavConfig:
        if self._cfg is None:
            raise RuntimeError("navigation service not configured")
        return self._cfg

    def _require_runtime(self):
        cfg = self._require_cfg()
        runtime = get_slam(cfg.slam_service)
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
        _apply_overrides(params, overrides)
        if cfg.nav2_params:
            _deep_merge(params, dict(cfg.nav2_params))
        _apply_local_costmap_size(params, cfg.nav2)

        runtime = get_slam(cfg.slam_service)
        _set_obstacle_sources(params, len(runtime.slam_cfg.lidars))
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
        if cmd == "navigate_to_location":
            loc = self._locations().get(str(command["name"]))
            mgr.navigate(loc.x, loc.y, loc.theta)
            return {"status": "navigating", "target": loc.to_dict()}
        if cmd == "navigate_to_point":
            x = float(command["x"])
            y = float(command["y"])
            theta = float(command.get("theta", 0.0))
            mgr.navigate(x, y, theta)
            return {"status": "navigating", "target": {"x": x, "y": y, "theta": theta}}
        if cmd == "cancel":
            mgr.cancel()
            return {"status": "canceled"}
        if cmd == "get_status":
            status = mgr.nav_status()
            status.update(mgr.nav2_diagnostics())
            return status
        if cmd == "start_nav2":
            cfg = self._require_cfg()
            runtime.manager.set_nav_config(cfg)
            params_path = self._write_nav2_params(cfg)
            await asyncio.to_thread(runtime.manager.ensure_nav2, cfg, params_path)
            return {"status": "nav2_started", **runtime.manager.nav2_diagnostics()}

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


Registry.register_resource_creator(
    Generic.API,
    RosNavigation.MODEL,
    ResourceCreatorRegistration(RosNavigation.new, RosNavigation.validate_config),
)

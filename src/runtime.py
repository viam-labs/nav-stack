"""Process-global registry linking the SLAM and navigation models.

Both models live in the same module process. The SLAM model owns the map store
and builtin SLAM host; the navigation model (which ``depends_on`` the SLAM
service) looks the shared runtime up by the SLAM service's resource name.

Builtin nav registers a ``NavVizStore`` so nav-camera / get_costmap can render
without a ROS bridge.
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, Optional


class SlamRuntime:
    """Shared handle published by a SLAM model instance."""

    def __init__(
        self,
        manager,
        map_store,
        slam_cfg,
        localization_check: Optional[dict] = None,
        *,
        cameras: Optional[dict] = None,
        shm_lidar: Optional[object] = None,
        sim_sensors: Optional[object] = None,
    ):
        self.manager = manager
        self.map_store = map_store
        self.slam_cfg = slam_cfg
        self.localization_check = (
            localization_check if localization_check is not None else {"status": "idle"}
        )
        # Lidar camera resources (name -> Camera), for ViamWorldIO scan reads.
        self.cameras = dict(cameras or {})
        # Shared POSIX-shm PCD client (builtin nav).
        self.shm_lidar = shm_lidar
        # Optional SimSensors when ``sim.enabled`` (nav obstacle scans).
        self.sim_sensors = sim_sensors


_REGISTRY: Dict[str, SlamRuntime] = {}
_LOCK = Lock()


def register_slam(name: str, runtime: SlamRuntime) -> None:
    with _LOCK:
        _REGISTRY[name] = runtime


def unregister_slam(name: str) -> None:
    with _LOCK:
        _REGISTRY.pop(name, None)


def get_slam(name: str) -> Optional[SlamRuntime]:
    with _LOCK:
        return _REGISTRY.get(name)


# The live SLAM *service object* (RosSlam), keyed by resource name. Builtin nav
# uses this for sync ``get_position_pose2d`` when the motion dependency is a
# gRPC client stub — async GetPosition every control tick starves Base.SetVelocity
# on the shared module event loop.
_SLAM_SERVICES: Dict[str, object] = {}


def register_slam_service(name: str, service: object) -> None:
    with _LOCK:
        _SLAM_SERVICES[name] = service


def unregister_slam_service(name: str) -> None:
    with _LOCK:
        _SLAM_SERVICES.pop(name, None)


def get_slam_service(name: str) -> Optional[object]:
    with _LOCK:
        return _SLAM_SERVICES.get(name)


# Builtin-nav viz stores, keyed by navigation service name (nav-camera /
# get_costmap).
_NAV_VIZ: Dict[str, object] = {}


def register_nav_viz(nav_name: str, viz: object) -> None:
    with _LOCK:
        _NAV_VIZ[nav_name] = viz


def unregister_nav_viz(nav_name: str) -> None:
    with _LOCK:
        _NAV_VIZ.pop(nav_name, None)


def get_nav_viz(nav_name: str) -> Optional[object]:
    with _LOCK:
        return _NAV_VIZ.get(nav_name)


def get_nav_view(nav_name: str) -> Optional[object]:
    """Return the builtin viz store for nav-camera / get_costmap."""
    return get_nav_viz(nav_name)


# Navigation hosts, keyed by motion service name. SLAM uses this so
# ``_is_navigation_active`` works with BuiltinSlamHost.
_NAV_HOSTS: Dict[str, object] = {}


def register_nav_host(nav_name: str, host: object) -> None:
    with _LOCK:
        _NAV_HOSTS[nav_name] = host


def unregister_nav_host(nav_name: str) -> None:
    with _LOCK:
        _NAV_HOSTS.pop(nav_name, None)


def any_navigation_active() -> bool:
    """True if any registered navigation host reports an active goal."""
    with _LOCK:
        hosts = list(_NAV_HOSTS.values())
    for host in hosts:
        try:
            status = host.nav_status()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(status, dict) and status.get("active"):
            return True
    return False

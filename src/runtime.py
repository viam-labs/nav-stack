"""Process-global registry linking the SLAM and navigation models.

Both models live in the same module process. The SLAM model owns the ROS manager
and the map store; the navigation model (which ``depends_on`` the SLAM service)
looks the shared runtime up by the SLAM service's resource name so it can launch
Nav2 against the same rclpy context and read the active map's locations/zones.

Builtin nav (``nav_backend: builtin``) may also register a ``NavVizStore`` so
nav-camera / get_costmap work without a ROS bridge.
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
    ):
        self.manager = manager
        self.map_store = map_store
        self.slam_cfg = slam_cfg
        self.localization_check = (
            localization_check if localization_check is not None else {"status": "idle"}
        )
        # Lidar camera resources (name -> Camera), for ViamWorldIO scan reads.
        self.cameras = dict(cameras or {})
        # Shared POSIX-shm PCD client (bridge + builtin nav).
        self.shm_lidar = shm_lidar


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


# Live bridge nodes, keyed by the *navigation* service name that owns/drives
# them. Published so the ``nav-camera`` component can find the running
# ``BridgeNode`` in-process and read Nav2 costmap/plan/pose data for rendering,
# without a Viam RPC round-trip. Value is a ``ros.bridge.BridgeNode`` or a
# zero-arg callable returning the current node (so a SLAM restart that swaps
# the manager/node cannot leave the registry pointing at a dead node). Typed
# as ``object`` here to keep this module import-light and ROS-free.
_BRIDGES: Dict[str, object] = {}

# Builtin-nav viz stores (same key as navigation service name). Used when there
# is no ROS bridge (navigation-external + builtin) or when nav writes overlays
# via ViamWorldIO instead of Nav2 topics.
_NAV_VIZ: Dict[str, object] = {}


def register_bridge(nav_name: str, node_or_provider: object) -> None:
    with _LOCK:
        _BRIDGES[nav_name] = node_or_provider


def unregister_bridge(nav_name: str) -> None:
    with _LOCK:
        _BRIDGES.pop(nav_name, None)


def get_bridge(nav_name: str) -> Optional[object]:
    with _LOCK:
        entry = _BRIDGES.get(nav_name)
    if callable(entry):
        try:
            return entry()
        except Exception:  # noqa: BLE001 - a failing provider means no bridge
            return None
    return entry


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
    """Prefer builtin viz store, else ROS bridge (nav-camera / get_costmap)."""
    viz = get_nav_viz(nav_name)
    if viz is not None:
        return viz
    return get_bridge(nav_name)


# Builtin (and Nav2) navigation hosts, keyed by motion service name. SLAM uses
# this so ``_is_navigation_active`` works when the SLAM manager is
# BuiltinSlamHost (which always reports nav idle).
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

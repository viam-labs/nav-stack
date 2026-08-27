"""Process-global registry linking the SLAM and navigation models.

Both models live in the same module process. The SLAM model owns the ROS manager
and the map store; the navigation model (which ``depends_on`` the SLAM service)
looks the shared runtime up by the SLAM service's resource name so it can launch
Nav2 against the same rclpy context and read the active map's locations/zones.
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
    ):
        self.manager = manager
        self.map_store = map_store
        self.slam_cfg = slam_cfg
        self.localization_check = (
            localization_check if localization_check is not None else {"status": "idle"}
        )


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


# Live bridge nodes, keyed by the *navigation* service name that owns/drives
# them. Published so the ``nav-camera`` component can find the running
# ``BridgeNode`` in-process and read Nav2 costmap/plan/pose data for rendering,
# without a Viam RPC round-trip. Value is a ``ros.bridge.BridgeNode`` or a
# zero-arg callable returning the current node (so a SLAM restart that swaps
# the manager/node cannot leave the registry pointing at a dead node). Typed
# as ``object`` here to keep this module import-light and ROS-free.
_BRIDGES: Dict[str, object] = {}


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

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

    def __init__(self, manager, map_store, slam_cfg):
        self.manager = manager
        self.map_store = map_store
        self.slam_cfg = slam_cfg


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

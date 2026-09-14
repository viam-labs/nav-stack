"""Process-global SimWorld registry (shared by sim-base + SLAM sensors)."""
from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from .world import SimWorld

_WORLDS: Dict[str, SimWorld] = {}
_LOCK = Lock()


def register_sim_world(name: str, world: SimWorld) -> SimWorld:
    key = str(name or "default")
    with _LOCK:
        _WORLDS[key] = world
    return world


def get_sim_world(name: str = "default") -> Optional[SimWorld]:
    with _LOCK:
        return _WORLDS.get(str(name or "default"))


def require_sim_world(name: str = "default") -> SimWorld:
    world = get_sim_world(name)
    if world is None:
        raise RuntimeError(
            f"sim world {name!r} not registered; configure "
            "viam-labs:nav-stack:sim-base (or enable slam sim) first"
        )
    return world


def unregister_sim_world(name: str) -> None:
    with _LOCK:
        _WORLDS.pop(str(name or "default"), None)

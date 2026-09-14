"""Builtin simulation: raycast world, fake base, sensor facades."""

from .bootstrap import ensure_sim_world_from_slam_cfg
from .registry import (
    get_sim_world,
    register_sim_world,
    require_sim_world,
    unregister_sim_world,
)
from .sensors import SimSensors
from .world import SimMap, SimWorld, load_sim_map, make_builtin_corridor

__all__ = [
    "SimMap",
    "SimSensors",
    "SimWorld",
    "ensure_sim_world_from_slam_cfg",
    "get_sim_world",
    "load_sim_map",
    "make_builtin_corridor",
    "register_sim_world",
    "require_sim_world",
    "unregister_sim_world",
]

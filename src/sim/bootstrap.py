"""Create / reuse SimWorld from SlamConfig.sim."""
from __future__ import annotations

from ..ros import conversions as conv
from .registry import get_sim_world, register_sim_world
from .world import SimWorld, load_sim_map, make_builtin_corridor


def ensure_sim_world_from_slam_cfg(cfg) -> SimWorld:
    """Create or reuse the process-global SimWorld described by ``cfg.sim``."""
    sim = cfg.sim
    existing = get_sim_world(sim.world_name)
    seed = conv.Pose2D(sim.seed_x, sim.seed_y, sim.seed_theta)
    if existing is not None:
        existing.set_seed(seed)
        existing.scan_bins = int(sim.scan_bins)
        existing.range_min = float(sim.range_min)
        existing.range_max = float(sim.range_max)
        return existing
    sim_map = load_sim_map(sim.map_path) if sim.map_path else make_builtin_corridor()
    world = SimWorld(
        sim_map,
        seed_pose=seed,
        scan_bins=int(sim.scan_bins),
        range_min=float(sim.range_min),
        range_max=float(sim.range_max),
        name=sim.world_name,
    )
    register_sim_world(sim.world_name, world)
    return world

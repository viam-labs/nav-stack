"""BuiltinSensors-compatible facade over SimWorld."""
from __future__ import annotations

from typing import Optional

from ..geom import conversions as conv
from .world import SimWorld


class SimSensors:
    """Sync scan/odom reads for ``BuiltinSlamEngine`` (and nav scan_provider)."""

    def __init__(self, world: SimWorld):
        self._world = world

    @property
    def world(self) -> SimWorld:
        return self._world

    def get_scan(
        self, max_age_s: float = 2.0, *, fresh: bool = False
    ) -> Optional[conv.LaserScan2D]:
        del max_age_s, fresh
        return self._world.get_scan(capture_pose=True)

    def get_odom(self) -> Optional[conv.OdomReading]:
        return self._world.get_odom()

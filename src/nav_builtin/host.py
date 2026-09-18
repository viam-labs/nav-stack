"""Duck-typed nav host for builtin navigation."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import NAV_BACKEND_BUILTIN, NavConfig
from ..geom import conversions as conv
from .navigator import BuiltinNavigator
from .viz_store import NavVizStore
from .world_io import WorldIO


def make_builtin_navigator(
    world: WorldIO,
    nav_cfg: NavConfig,
    *,
    logger=None,
) -> BuiltinNavigator:
    if nav_cfg.inflation_is_noop() and logger is not None:
        inscribed = nav_cfg.inscribed_radius_m()
        logger.warn(
            f"inflation_radius={nav_cfg.inflation_radius:.2f} m is at or below the "
            f"footprint clearance radius ({inscribed:.2f} m), so it adds no soft "
            "inflation at all (it is measured from the obstacle, not added to the "
            "footprint). Set inflation_margin_m for a band past the footprint — "
            f"e.g. inflation_margin_m: 0.20 gives a soft ring out to "
            f"{inscribed + 0.20:.2f} m."
        )
    return BuiltinNavigator(world, nav_cfg, logger=logger)


class BuiltinNavHost:
    """Nav surface used by ``NavServiceBase`` for builtin navigation.

    Implements the methods ``nav_core`` calls on ``runtime.manager`` for
    navigate / plan / pose / scan / status. ``node`` is None (no zone filter
    publisher).
    """

    def __init__(
        self,
        navigator: BuiltinNavigator,
        world: WorldIO,
        viz: NavVizStore,
        *,
        nav_cfg: NavConfig,
    ):
        self._builtin_nav = navigator
        self._world = world
        self._viz = viz
        self._nav_cfg = nav_cfg
        self.node = None

    @property
    def viz(self) -> NavVizStore:
        return self._viz

    def set_nav_config(self, nav_cfg: NavConfig) -> None:
        self._nav_cfg = nav_cfg
        # Rebuild navigator with updated limits/planner if config changes.
        self._builtin_nav.cancel()
        self._builtin_nav = make_builtin_navigator(
            self._world, nav_cfg, logger=self._builtin_nav._logger  # noqa: SLF001
        )

    def nav_backend(self) -> str:
        return NAV_BACKEND_BUILTIN

    def navigate(self, x: float, y: float, theta: float) -> None:
        self._abort_slam_background_localize()
        self._builtin_nav.navigate(x, y, theta)

    def _abort_slam_background_localize(self) -> None:
        """Stop startup global_localize so SetVelocity is not starved."""
        name = getattr(self._nav_cfg, "slam_service", None)
        if not name:
            return
        try:
            from ..runtime import get_slam_service

            svc = get_slam_service(str(name))
        except Exception:  # noqa: BLE001
            return
        abort = getattr(svc, "abort_background_localize", None)
        if callable(abort):
            try:
                abort()
            except Exception:  # noqa: BLE001
                pass

    def compute_path(self, *args, **kwargs) -> Dict:
        return self._builtin_nav.compute_path(*args, **kwargs)

    def last_preview_plan(self) -> Optional[Dict]:
        return self._builtin_nav.last_preview_plan()

    def cancel(self) -> None:
        self._builtin_nav.cancel()

    def nav_status(self) -> Dict:
        status = self._builtin_nav.nav_status()
        status["nav_backend"] = NAV_BACKEND_BUILTIN
        last = getattr(self._world, "last_drive", None)
        if callable(last):
            drive = last()
            if drive is not None:
                status["last_drive"] = drive
        stats_fn = getattr(self._world, "drive_stats", None)
        if callable(stats_fn):
            try:
                status["drive"] = stats_fn()
            except Exception:  # noqa: BLE001
                pass
        ctrl = getattr(self._builtin_nav, "control_stats", None)
        if callable(ctrl):
            try:
                loop_stats = ctrl()
                if loop_stats is not None:
                    status["control_loop"] = loop_stats
            except Exception:  # noqa: BLE001
                pass
        src = getattr(self._world, "pose_source", None)
        if callable(src):
            status["pose_source"] = src()
        return status

    def get_pose_in_map(self) -> Optional[conv.Pose2D]:
        return self._world.get_pose()

    def get_base_scan(self, max_age_s: float = 1.0) -> Optional[conv.LaserScan2D]:
        return self._world.get_scan(max_age_s)

    def publish_zone_masks(
        self,
        keepout_mask: np.ndarray,
        speed_mask: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        # Builtin costmap does not consume keepout/speed zone masks yet.
        del keepout_mask, speed_mask, resolution, origin_x, origin_y

    def shutdown(self) -> None:
        try:
            self._builtin_nav.cancel()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._world.stop()
        except Exception:  # noqa: BLE001
            pass

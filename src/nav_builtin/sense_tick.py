"""Per-tick sensing for ``NavSupervisor.run_goal``.

Fetches scan / depth / above-cart, refreshes the local costmap, builds
``guard_pts``, and decides nose_clear / local_blocked. Later phases mutate
the same ``TickContext`` (cmd / progress / path / pose / corridor) instead
of re-threading those fields as kwargs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from ..geom import conversions as conv
from ..nav.simple_motion import DriveCommand, cone_min_range
from .above_cart import above_cart_frames
from .depth_memory import depth_frames
from .escape import EscapeSense
from .footprint_guard import obstacle_points, scan_to_world_points
from .types import Path2D, Pose2D

if TYPE_CHECKING:
    from .goal_episode import GoalEpisode
    from .local_costmap import LocalCostmapView
    from .supervisor import NavSupervisor

# Per-tick wall times (ms) for sense → costmap → guard → dwa → gate.
PHASE_MS_KEYS = (
    "sense_ms",
    "costmap_ms",
    "guard_ms",
    "dwa_ms",
    "gate_ms",
)


def elapsed_ms(started: float) -> float:
    """Milliseconds since ``perf_counter()`` mark, rounded to 0.1 ms."""
    return round((perf_counter() - started) * 1000.0, 1)


def _mono() -> float:
    from . import supervisor as _sup

    return _sup.time.monotonic()


@dataclass
class TickContext:
    """Per-tick inputs shared by every phase of ``run_goal``.

    Sense fields are set once; ``cmd`` / ``progress`` / ``path`` / ``pose`` /
    corridor / wait flags are updated in place as phases run.
    """

    now: float
    pose: Pose2D
    path: Path2D
    goal: Pose2D
    scan: Optional[conv.LaserScan2D]
    mem_pts: Any
    local_view: Optional["LocalCostmapView"]
    guard_pts: Any
    nose_clear: bool
    forward_clearance_m: Optional[float]
    path_ahead_cost: int
    pose_cost: int
    local_blocked: bool
    map_untrusted: bool
    collision_view: Optional["LocalCostmapView"]
    # Tick-local scan→world XY (shared by guard / corridor / path-follow).
    scan_world_pts: Any = None
    cmd: DriveCommand = field(
        default_factory=lambda: DriveCommand(0.0, 0.0, 0.0, False)
    )
    progress: Dict[str, Any] = field(default_factory=dict)
    # Wall times (ms) for sense / costmap / guard / dwa / gate this tick.
    phase_ms: Dict[str, float] = field(default_factory=dict)
    waiting_for_clear: bool = False
    corridor_yaw: Optional[float] = None
    loc_hold_motion: bool = False

    def escape_sense(
        self,
        nav: "NavSupervisor",
        ep: "GoalEpisode",
        *,
        prefer_yaw_sign: Optional[float] = None,
        corridor_active: bool = False,
    ) -> EscapeSense:
        """Build EscapeSense from this tick + nav/episode constants."""
        yaw = self.corridor_yaw
        if prefer_yaw_sign is None:
            prefer_yaw_sign = (
                math.copysign(1.0, yaw)
                if yaw is not None
                else ep.escape_m.yaw_sign
            )
        return EscapeSense(
            now=self.now,
            pose=self.pose,
            cmd=self.cmd,
            progress=self.progress,
            nose_clear=self.nose_clear,
            waiting=self.waiting_for_clear,
            failed_replans=ep.failed_replan_while_blocked,
            spin_blocked=bool(self.progress.get("spin_blocked")),
            heading_open=yaw is not None,
            bearing_rad=float(self.progress.get("bearing_error_rad") or 0.0),
            scan=self.scan,
            local_view=self.local_view,
            guard=nav._guard,  # noqa: SLF001
            guard_pts=self.guard_pts,
            robot_radius_m=nav._robot_radius,  # noqa: SLF001
            spin_radius_m=nav._spin_radius,  # noqa: SLF001
            follower=nav._follower,  # noqa: SLF001
            backup_dist_m=float(nav._backup_dist_m),  # noqa: SLF001
            backup_cooldown_s=float(nav._backup_cooldown_s),  # noqa: SLF001
            loc_hold=self.loc_hold_motion,
            corridor_active=corridor_active,
            prefer_yaw_sign=float(prefer_yaw_sign),
        )


def run_sense_tick(
    nav: "NavSupervisor",
    ep: "GoalEpisode",
    *,
    now: float,
    pose: Pose2D,
    path: Path2D,
    goal: Pose2D,
) -> TickContext:
    """Scan + memory + local costmap + nose / local_blocked detection."""
    need_scan = (
        nav._follower.obstacle is not None  # noqa: SLF001
        and nav._follower.obstacle.enabled  # noqa: SLF001
    ) or nav._local_costmap is not None  # noqa: SLF001
    refresh_local = nav._local_costmap is not None and (  # noqa: SLF001
        now - nav._local_view_at >= nav._local_update_period_s  # noqa: SLF001
        or nav._local_view_cache is None  # noqa: SLF001
    )

    t_sense = perf_counter()
    scan = None
    if need_scan:
        try:
            # Fused lidar + obstacles_only depth: reactive cone, local costmap
            # / DWA, and nose_clear. Loc refine still reads lidar-only.
            scan = nav._world.get_scan(nav._scan_max_age)  # noqa: SLF001
        except TimeoutError:
            scan = None
    mem_pts = None
    if nav._guard is not None:  # noqa: SLF001
        nav._depth_memory.update(  # noqa: SLF001
            depth_frames(nav._world), pose, _mono()  # noqa: SLF001
        )
        mem_pts = nav._depth_memory.points()  # noqa: SLF001
    if nav._above_memory is not None:  # noqa: SLF001
        nav._above_memory.update(  # noqa: SLF001
            above_cart_frames(nav._world), pose, _mono()  # noqa: SLF001
        )
        above_pts = nav._above_memory.points()  # noqa: SLF001
        if above_pts.size:
            mem_pts = (
                above_pts
                if mem_pts is None or not np.size(mem_pts)
                else np.vstack([np.asarray(mem_pts), above_pts])
            )
    sense_ms = elapsed_ms(t_sense)

    t_costmap = perf_counter()
    local_view = nav._local_view_cache  # noqa: SLF001
    if refresh_local:
        nav._kick_global_costmap_refresh(allow_inline=True)  # noqa: SLF001
        # Missing scan must not wipe live marks — that made path_blocked_local
        # flicker false under IO load while reactive avoid still saw the obstacle.
        if scan is None:
            local_view = nav._local_view_cache  # noqa: SLF001
        else:
            costmap_scan = scan
            if costmap_scan.capture_pose is None:
                costmap_scan = conv.LaserScan2D(
                    ranges=costmap_scan.ranges,
                    angle_min=costmap_scan.angle_min,
                    angle_increment=costmap_scan.angle_increment,
                    range_min=costmap_scan.range_min,
                    range_max=costmap_scan.range_max,
                    sensor_pose=costmap_scan.sensor_pose,
                    capture_pose=pose,
                )
            with nav._global_cache_lock:  # noqa: SLF001
                global_occ = nav._global_occ_cache  # noqa: SLF001
                global_costs = nav._global_costs_cache  # noqa: SLF001
            local_view = nav._local_costmap.update(  # noqa: SLF001
                pose,
                costmap_scan,
                global_occ=global_occ,
                global_costs=global_costs,
                extra_points=mem_pts,
                mapping=nav._slam_is_mapping(),  # noqa: SLF001
            )
            nav._local_view_cache = local_view  # noqa: SLF001
            nav._local_view_at = now  # noqa: SLF001
            nav._local_costmap_updates += 1  # noqa: SLF001
            if nav._local_costmap_enabled:  # noqa: SLF001
                from .costmap import local_view_viz_dict

                try:
                    nav._world.set_viz_local_costmap(  # noqa: SLF001
                        local_view_viz_dict(local_view)
                    )
                except Exception:  # noqa: BLE001 - viz is best-effort
                    pass
    costmap_ms = elapsed_ms(t_costmap)

    t_guard = perf_counter()
    path_ahead_cost = 0
    pose_cost = 0
    local_blocked = False
    nose_clear = True
    forward_clearance_m: Optional[float] = None
    guard_pts = None
    map_untrusted = nav._published_pose_misregistered()  # noqa: SLF001
    collision_view = None if map_untrusted else local_view
    scan_world_pts = scan_to_world_points(pose, scan) if scan is not None else None
    obs_cfg = nav._follower.obstacle  # noqa: SLF001
    if nav._guard is not None:  # noqa: SLF001
        guard_pts = obstacle_points(
            pose,
            scan,
            collision_view,
            radius_m=nav._guard.cfg.obstacle_radius_m,  # noqa: SLF001
            extra=mem_pts,
            scan_world_pts=scan_world_pts,
        )
        straight = nav._guard.free_distance(  # noqa: SLF001
            pose, 0.2, 0.0, guard_pts, 2.0
        )
        if math.isfinite(straight):
            forward_clearance_m = straight + nav._guard.cfg.length_m / 2.0  # noqa: SLF001
        nose_clear = straight >= nav._guard.cfg.alt_min_free_m or (  # noqa: SLF001
            nav._guard.best_forward_arc(  # noqa: SLF001
                pose, guard_pts, None, nav._guard.cfg.horizon_m, 0.0  # noqa: SLF001
            )
            is not None
        )
    elif obs_cfg is not None and obs_cfg.enabled and scan is not None:
        half = float(obs_cfg.front_cone_half_rad)
        nose_range = cone_min_range(scan, -half, half)
        if math.isfinite(nose_range):
            forward_clearance_m = float(nose_range)
        nose_clear = (
            forward_clearance_m is None
            or forward_clearance_m > obs_cfg.stop_distance_m
        )
    if local_view is not None:
        from .local_planner import path_cost_ahead as _path_cost_ahead

        path_ahead_cost = int(
            _path_cost_ahead(
                pose,
                path,
                local_view,
                lookahead_m=nav._local_planner.path_clearance_lookahead_m,  # noqa: SLF001
            )
        )
        pose_cost = int(local_view.cost_at_world(pose.x, pose.y))
        local_blocked = nav._path_locally_blocked(  # noqa: SLF001
            path_ahead_cost=path_ahead_cost,
            pose_cost=pose_cost,
            activate_cost=int(nav._local_planner_activate_cost),  # noqa: SLF001
            nose_clear=nose_clear,
            forward_clearance_m=forward_clearance_m,
        )
        if map_untrusted and nose_clear:
            local_blocked = False
    # Reactive avoid spinning with a clear-looking path still means the robot
    # cannot proceed — escalate to the blocked/replan path.
    if (
        ep.last_obstacle_state == "avoid"
        and ep.reactive_avoid_since is not None
        and now - ep.reactive_avoid_since >= 0.8
        and nav._path_locally_blocked(  # noqa: SLF001
            path_ahead_cost=path_ahead_cost,
            pose_cost=pose_cost,
            activate_cost=int(nav._local_planner_activate_cost),  # noqa: SLF001
            nose_clear=nose_clear,
            forward_clearance_m=forward_clearance_m,
        )
    ):
        local_blocked = True
        if ep.local_blocked_since is None:
            ep.local_blocked_since = ep.reactive_avoid_since
    if ep.last_obstacle_state == "in_lethal":
        local_blocked = True
        if ep.local_blocked_since is None:
            ep.local_blocked_since = now
    guard_ms = elapsed_ms(t_guard)

    return TickContext(
        now=now,
        pose=pose,
        path=path,
        goal=goal,
        scan=scan,
        mem_pts=mem_pts,
        local_view=local_view,
        guard_pts=guard_pts,
        nose_clear=nose_clear,
        forward_clearance_m=forward_clearance_m,
        path_ahead_cost=path_ahead_cost,
        pose_cost=pose_cost,
        local_blocked=local_blocked,
        map_untrusted=map_untrusted,
        collision_view=collision_view,
        scan_world_pts=scan_world_pts,
        phase_ms={
            "sense_ms": sense_ms,
            "costmap_ms": costmap_ms,
            "guard_ms": guard_ms,
        },
    )

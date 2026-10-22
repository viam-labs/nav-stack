"""Background occupancy SLAM loop: predict → match → update."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from ..config import (
    BASE_VELOCITY_Y_FORWARD,
    IMU_ODOM_ACCEL_ONLY,
    MODE_LOCALIZING,
    MODE_MAPPING,
    SlamConfig,
)
from ..nav.global_localize import OccupancyMap
from ..nav.maps import MapStore
from ..nav.pose_jump_gate import PoseJumpGate
from ..geom import conversions as conv
from . import occupancy as occ
from . import persistence
from . import scan_match
from . import loop_closure
from . import pose_graph as pg
from .io_sensors import BuiltinSensors
from .lidar_odom import LidarOdometry
from .types import LogOddsGrid


class BuiltinSlamEngine:
    """In-process 2D occupancy mapping + localization."""

    def __init__(
        self,
        cfg: SlamConfig,
        sensors: BuiltinSensors,
        map_store: MapStore,
        *,
        logger=None,
        rate_hz: float = 10.0,
        match_period_s: float = 0.3,
    ):
        self._cfg = cfg
        self._sensors = sensors
        self._map_store = map_store
        self._logger = logger
        self._period = 1.0 / max(rate_hz, 1.0)
        # Scan matching + map inserts are CPU-heavy; running them at the full
        # predict rate starved the module event loop (nav "Viam IO timed out").
        self._match_period_s = max(0.0, float(match_period_s))

        self._lock = threading.RLock()
        self._grid: LogOddsGrid = occ.empty_grid(
            resolution=cfg.map.resolution
        )
        self._pose = conv.Pose2D(0.0, 0.0, 0.0)
        self._last_odom_pose: Optional[conv.Pose2D] = None
        self._last_odom_pose_at: Optional[float] = None
        self._last_odom_time: Optional[float] = None
        self._last_odom_heading: Optional[float] = None
        self._mode = cfg.mode
        self._generation = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_match_score = float("nan")
        self._last_prior_score = float("nan")
        self._last_scan_age_s = float("nan")
        self._last_scan_obj: Optional[conv.LaserScan2D] = None
        self._last_tick_pose: Optional[conv.Pose2D] = None
        self._last_tick_at: Optional[float] = None
        self._last_yaw_rate = 0.0
        self._insert_skips_turning = 0
        self._ticks = 0
        self._updates = 0
        self._match_accepts = 0
        self._match_rejects = 0
        self._nav_track_applies = 0
        self._last_match_at = 0.0
        # Mapping inserts only after movement (or timeout).
        self._last_insert_pose: Optional[conv.Pose2D] = None
        self._last_insert_at = 0.0
        # Anti-oscillation: require two agreeing frames before applying a jump.
        self._pending_match: Optional[conv.Pose2D] = None
        self._lateral_pending_cross: Optional[float] = None
        self._lateral_pending_yaw: Optional[float] = None
        self._pending_count = 0
        # Cached ROS-style grid for scan matching (invalidated on map edits).
        self._occ_cache: Optional[OccupancyMap] = None
        self._occ_cache_epoch = -1
        self._match_epoch = 0
        self._occ_cache_known = 0.0
        # Localizing mode: do not odom-track from (0,0) until a full-map seed
        # succeeds. Without this the robot "assumes start" until the async
        # startup global_localize task eventually fires.
        self._seed_localize_pending = cfg.mode == MODE_LOCALIZING
        self._last_seed_attempt_at = 0.0
        self._seed_min_score = 0.40
        self._map_dir: Optional[Path] = None
        self._pose_restored_from_disk = False
        self._last_pose_save_at = 0.0
        self._graph = pg.PoseGraph(
            max_nodes=int(cfg.builtin_mapping_keyframe_max),
            node_spacing_m=float(cfg.builtin_loop_node_spacing_m),
            node_spacing_deg=float(cfg.builtin_loop_node_spacing_deg),
            scans_per_node=int(cfg.builtin_loop_scans_per_node),
        )
        # Map loaded when mapping resumed on it; renders start from it.
        self._graph_base: Optional[LogOddsGrid] = None
        # Manual erase/paint edits, replayed onto every re-rendered grid.
        self._grid_edits: list = []
        # One optimize + re-render at a time (worker, revisit, DoCommand).
        self._graph_job = threading.Lock()
        self._graph_revision = 0
        # Bumped whenever the pose jumps frame (correction / set_pose) so an
        # in-flight match computed in the old frame is not applied.
        self._pose_epoch = 0
        self._loop_params = loop_closure.LoopClosureParams(
            min_travel_gap_m=float(cfg.builtin_loop_min_travel_m),
            search_radius_m=float(cfg.builtin_loop_search_radius_m),
            window_m=float(cfg.builtin_loop_search_window_m),
            window_deg=float(cfg.builtin_loop_search_window_deg),
            min_score=float(cfg.builtin_loop_min_score),
        )
        self._loop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_last_attempt_at = 0.0
        self._loop_pending: Optional[tuple] = None
        self._loop_full_logged = False
        self._loop_stats: dict = {
            "attempts": 0,
            "closures": 0,
            "optimizations": 0,
            "rejected_inconsistent": 0,
            "reasons": {},
            "last": None,
            "last_optimize_ms": None,
            "last_render_ms": None,
        }
        self._keyframe_hook = None
        self._pose_listeners: list = []
        self._last_odom_twist = (0.0, 0.0, 0.0)
        # Forward speed (m/s) guessed from IMU accel when wheel velocity is absent.
        self._imu_forward_m_s = 0.0
        self._imu_ax_bias = None
        self._imu_yaw_bias = None
        # Lidar odometry (wheel-less carts): body velocity (forward, left) m/s
        # learned from successive scan matches.
        self._lo = LidarOdometry()
        self._lo_vel = (0.0, 0.0)
        self._lo_last_at: Optional[float] = None
        self._lo_reset_pending = False
        self._lo_map_updates: Optional[int] = None
        self._lo_fails = 0
        self._lo_accepts = 0
        self._lo_rejects = 0
        self._lo_resets = 0
        self._lo_inserts_held = 0
        self._lo_last_inlier = float("nan")
        self._lo_last_rms = float("nan")
        self._lo_prev_theta: Optional[float] = None
        self._lo_turn_rate = 0.0
        self._lo_match_ms = float("nan")
        # Signed yaw totals (rad) for comparing the gyro with the lidar on a spin.
        self._yaw_check_gyro = 0.0
        self._yaw_check_lidar = 0.0
        self._last_loop_rebuild_at = 0.0
        self._pose_jump_gate = PoseJumpGate(
            confirm_count=int(cfg.localize_jump_confirm_count),
            agree_m=float(cfg.localize_jump_agree_m),
            agree_deg=float(cfg.localize_jump_agree_deg),
            large_m=float(cfg.localize_jump_large_m),
            large_deg=float(cfg.localize_jump_large_deg),
        )

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger(msg)
            except Exception:  # noqa: BLE001
                pass

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="builtin-slam",
            daemon=True,
        )
        self._thread.start()
        self._loop_thread = threading.Thread(
            target=self._loop_closure_worker,
            name="builtin-slam-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self._log(f"builtin SLAM engine started ({self._mode})")

    def stop(self) -> None:
        self._persist_pose(force=True)
        self._running = False
        self._loop_event.set()
        for thr in (self._thread, self._loop_thread):
            if thr is not None and thr.is_alive():
                thr.join(timeout=2.0)
        self._thread = None
        self._loop_thread = None

    def configure_mode(self, mode: str, map_dir: Optional[Path] = None) -> None:
        with self._lock:
            self._mode = mode
            self._pose_restored_from_disk = False
            loaded = None
            if map_dir is not None:
                self._map_dir = Path(map_dir)
                loaded = persistence.load_log_odds(map_dir)
                if loaded is not None:
                    self._grid = loaded
                    self._invalidate_occ_cache()
                    self._generation += 1
                    self._log(f"loaded occupancy map from {map_dir}")
                elif mode == MODE_LOCALIZING:
                    raise RuntimeError(
                        f"cannot localize: no map.yaml/map.pgm under {map_dir}"
                    )
                elif mode == MODE_MAPPING:
                    # Fresh mapping session.
                    self._grid = occ.empty_grid(
                        resolution=self._cfg.map.resolution
                    )
                    self._pose = conv.Pose2D(0.0, 0.0, 0.0)
                    self._last_odom_pose = None
                    self._last_odom_heading = None
                    self._invalidate_occ_cache()
                    self._generation += 1
            self._reset_graph_locked()
            self._reset_lidar_odom_locked()
            if mode == MODE_LOCALIZING:
                self._seed_localize_pending = True
                self._last_seed_attempt_at = 0.0
                self._last_odom_pose = None
                self._last_odom_heading = None
                self._last_odom_time = None
                self._pending_match = None
                self._pending_count = 0
                restored = self._try_restore_last_pose_locked()
                if restored is not None:
                    self._pose = restored
                    self._seed_localize_pending = False
                    self._pose_restored_from_disk = True
                    self._log(
                        "restored last pose from disk "
                        f"({restored.x:.2f}, {restored.y:.2f}, "
                        f"{math.degrees(restored.theta):.1f} deg)"
                    )
                else:
                    self._log(
                        "builtin SLAM awaiting full-map seed localize "
                        "(will not assume start pose)"
                    )
            else:
                self._seed_localize_pending = False
                # Mapping with an existing map: resume near last pose if any.
                if map_dir is not None and loaded is not None:
                    restored = self._try_restore_last_pose_locked()
                    if restored is not None:
                        self._pose = restored
                        self._pose_restored_from_disk = True
                        self._last_odom_pose = None
                        self._last_odom_heading = None
                        self._last_odom_time = None
                        self._log(
                            "restored last pose from disk for mapping "
                            f"({restored.x:.2f}, {restored.y:.2f}, "
                            f"{math.degrees(restored.theta):.1f} deg)"
                        )
        # Notify outside the lock if we restored (listeners may call back).
        if self._pose_restored_from_disk:
            self._notify_pose_listeners(self.get_pose())

    def _try_restore_last_pose_locked(self) -> Optional[conv.Pose2D]:
        cfg = self._cfg
        if not bool(getattr(cfg, "persist_pose", True)):
            return None
        map_dir = self._map_dir
        if map_dir is None:
            return None
        max_age = float(getattr(cfg, "persist_pose_max_age_s", 0.0) or 0.0)
        return persistence.load_last_pose(map_dir, max_age_s=max_age)

    def _persist_pose(self, *, force: bool = False) -> None:
        cfg = self._cfg
        if not bool(getattr(cfg, "persist_pose", True)):
            return
        map_dir = self._map_dir
        if map_dir is None:
            return
        # Do not persist an unseeded origin placeholder in localizing mode.
        if self._mode == MODE_LOCALIZING and self._seed_localize_pending:
            return
        interval = float(getattr(cfg, "persist_pose_interval_s", 1.0) or 0.0)
        now = time.monotonic()
        if not force and interval > 0.0 and (now - self._last_pose_save_at) < interval:
            return
        if not force and interval <= 0.0:
            return
        with self._lock:
            pose = self._pose
        try:
            persistence.save_last_pose(map_dir, pose)
            self._last_pose_save_at = now
        except Exception as exc:  # noqa: BLE001
            self._log(f"persist last pose failed: {exc}")

    def pose_restored_from_disk(self) -> bool:
        return bool(self._pose_restored_from_disk)

    def reset_map(self) -> None:
        with self._lock:
            self._grid = occ.empty_grid(
                resolution=self._cfg.map.resolution
            )
            self._pose = conv.Pose2D(0.0, 0.0, 0.0)
            self._last_odom_pose = None
            self._last_odom_heading = None
            self._last_insert_pose = None
            self._reset_graph_locked()
            self._pose_epoch += 1
            self._invalidate_occ_cache()
            self._generation += 1
            self._pose_restored_from_disk = False
            self._reset_lidar_odom_locked()
            map_dir = self._map_dir
        if map_dir is not None:
            path = persistence.last_pose_path(map_dir)
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass

    def clear_obstacles(
        self, x_m: float, y_m: float, radius_m: float
    ) -> dict:
        """Erase occupied cells in a disk (map frame meters). Builtin only."""
        with self._lock:
            # Copy first: get_map reads the published array off this lock.
            updated = occ.copy_grid(self._grid)
            cleared = occ.clear_disk(updated, x_m, y_m, radius_m)
            if cleared:
                self._grid_edits.append(("clear", float(x_m), float(y_m), float(radius_m)))
                self._grid = updated
                self._invalidate_occ_cache()
                self._generation += 1
            return {
                "cleared_cells": int(cleared),
                "x": float(x_m),
                "y": float(y_m),
                "radius_m": float(radius_m),
                "generation": int(self._generation),
                "resolution": float(self._grid.resolution),
            }

    def add_obstacles(
        self, x_m: float, y_m: float, radius_m: float
    ) -> dict:
        """Paint occupied cells in a disk (map frame meters). Builtin only."""
        with self._lock:
            updated = occ.copy_grid(self._grid)
            marked = occ.mark_disk(updated, x_m, y_m, radius_m)
            if marked:
                self._grid_edits.append(("mark", float(x_m), float(y_m), float(radius_m)))
                self._grid = updated
                self._invalidate_occ_cache()
                self._generation += 1
            return {
                "marked_cells": int(marked),
                "x": float(x_m),
                "y": float(y_m),
                "radius_m": float(radius_m),
                "generation": int(self._generation),
                "resolution": float(self._grid.resolution),
            }

    def set_keyframe_hook(self, hook) -> None:
        self._keyframe_hook = hook

    def add_pose_listener(self, listener) -> None:
        """``listener(pose: Pose2D)`` after set_pose / apply_map_pose_correction."""
        self._pose_listeners.append(listener)

    def _notify_pose_listeners(self, pose: conv.Pose2D) -> None:
        for listener in list(self._pose_listeners):
            try:
                listener(pose)
            except Exception:  # noqa: BLE001
                pass

    def set_pose(self, pose: conv.Pose2D) -> None:
        with self._lock:
            if self._mode == MODE_MAPPING:
                self._graph.rebase(pg.world_correction(self._pose, pose))
            self._pose = pose
            self._pose_epoch += 1
            self._pending_match = None
            self._pending_count = 0
            self._lateral_pending_cross = None
            self._lateral_pending_yaw = None
            # External relocalize / startup global_localize counts as seeded.
            self._seed_localize_pending = False
            self._last_odom_pose = None
            self._last_odom_heading = None
            self._last_odom_time = None
            self._imu_forward_m_s = 0.0
            self._imu_ax_bias = None
            self._imu_yaw_bias = None
            self._reset_lidar_odom_locked()
        self._notify_pose_listeners(pose)
        self._persist_pose(force=True)

    def _reset_lidar_odom_locked(self) -> None:
        """Drop the local scan map and velocity; the loop thread rebuilds it."""
        self._lo_vel = (0.0, 0.0)
        self._lo_last_at = None
        self._lo_fails = 0
        self._lo_reset_pending = True
        self._lo_map_updates = None
        self._lo_prev_theta = None
        self._lo_turn_rate = 0.0

    def _set_corrected_pose_locked(self, pose: conv.Pose2D) -> None:
        self._pose = pose
        self._pose_epoch += 1
        self._pending_match = None
        self._pending_count = 0
        self._lateral_pending_cross = None
        self._lateral_pending_yaw = None
        self._seed_localize_pending = False
        self._last_odom_pose = None
        self._last_odom_heading = None
        self._last_odom_time = None
        self._last_insert_pose = pose
        self._last_tick_pose = pose
        # Keyscans sit at the pre-correction poses; keep velocity, rebuild map.
        self._lo_reset_pending = True
        self._lo_last_at = None
        self._lo_map_updates = None

    def _reset_graph_locked(self) -> None:
        self._graph.clear()
        self._grid_edits = []
        self._loop_pending = None
        self._loop_full_logged = False
        self._graph_revision += 1
        observed = bool(np.any(np.abs(self._grid.log_odds) > 0.05))
        self._graph_base = (
            occ.copy_grid(self._grid)
            if self._mode == MODE_MAPPING and observed
            else None
        )

    def _replay_grid_edits_locked(self, grid: LogOddsGrid) -> None:
        for kind, x, y, r in self._grid_edits:
            if kind == "clear":
                occ.clear_disk(grid, x, y, r)
            else:
                occ.mark_disk(grid, x, y, r)

    # Corrections up to this yaw are spread along the loop by the pose graph;
    # bigger ones (yaw flip, far relocalize) shift the trailing path rigidly.
    _GRAPH_CORRECTION_MAX_DEG = 30.0
    _REVISIT_EDGE_SIGMA_M = 0.05
    _REVISIT_EDGE_SIGMA_DEG = 1.0
    # A new loop edge still this far off after optimizing disagrees with the
    # rest of the graph (likely a false match) and is dropped.
    _LOOP_MAX_CHI2 = 25.0
    _LOOP_REBUILD_MIN_M = 0.08
    _LOOP_REBUILD_MIN_DEG = 1.0

    def apply_map_pose_correction(self, matched_pose: conv.Pose2D) -> dict:
        """Correct pose drift during mapping; re-render the grid from the graph."""
        cfg = self._cfg
        with self._lock:
            current = self._pose
            shift_m = math.hypot(
                matched_pose.x - current.x, matched_pose.y - current.y
            )
            shift_deg = abs(
                math.degrees(
                    conv.normalize_angle(matched_pose.theta - current.theta)
                )
            )
            scans_used = self._graph.scan_count()
            should_rebuild = (
                self._mode == MODE_MAPPING
                and cfg.builtin_rebuild_map_on_revisit
                and len(self._graph) > 0
                and (
                    shift_m >= float(cfg.mapping_revisit_min_shift_m)
                    or shift_deg >= float(cfg.mapping_revisit_min_shift_deg)
                )
            )
            if not should_rebuild:
                self._set_corrected_pose_locked(matched_pose)
        out = {
            "applied": True,
            "rebuilt": False,
            "keyframes": scans_used,
            "shift_m": round(shift_m, 3),
            "shift_deg": round(shift_deg, 2),
            "slam_backend": "builtin",
        }
        if not should_rebuild:
            self._notify_pose_listeners(matched_pose)
            self._persist_pose(force=True)
            return out

        with self._graph_job:
            with self._lock:
                corr = pg.world_correction(self._pose, matched_pose)
                cur = len(self._graph) - 1
                target = conv.compose_poses(corr, self._graph.nodes[cur].pose)
                anchor = self._graph.nearest_old_node(
                    target,
                    min_travel_gap_m=self._loop_params.min_travel_gap_m,
                    radius_m=3.0,
                )
                if anchor is None:
                    anchor = self._graph.earliest_node_near(target)
                plausible = (
                    shift_m <= float(cfg.mapping_revisit_max_shift_m)
                    and shift_deg <= self._GRAPH_CORRECTION_MAX_DEG
                )
                if plausible and anchor is not None and anchor < cur:
                    self._graph.add_edge(
                        pg.GraphEdge(
                            anchor,
                            cur,
                            conv.compose_poses(
                                conv.invert_pose(self._graph.nodes[anchor].pose), target
                            ),
                            pg.info_from_sigmas(
                                self._REVISIT_EDGE_SIGMA_M,
                                math.radians(self._REVISIT_EDGE_SIGMA_DEG),
                            ),
                            loop=True,
                        )
                    )
                    method, first_free, live = "pose_graph", anchor + 1, None
                else:
                    start = anchor + 1 if anchor is not None else 0
                    self._graph.apply_world_correction_from(start, corr)
                    method, first_free, live = "rigid", None, corr
            result = self._optimize_and_render(first_free, live_correction=live)
        out.update(result)
        out["method"] = method
        out["anchor"] = anchor
        out["rebuilt"] = result.get("status") == "rebuilt"
        self._log(
            f"builtin SLAM revisit correction ({method}, anchor={anchor}): "
            f"{result.get('status')} shift={shift_m:.2f} m {shift_deg:.1f} deg"
        )
        return out

    def optimize_pose_graph(self) -> dict:
        """Optimize the whole mapping pose graph and re-render the grid."""
        with self._lock:
            if self._mode != MODE_MAPPING or len(self._graph) < 2:
                return {"status": "skipped", "nodes": len(self._graph)}
        with self._graph_job:
            return self._optimize_and_render(1)

    def _optimize_and_render(
        self,
        first_free: Optional[int],
        *,
        new_edges: Optional[list] = None,
        live_correction: Optional[conv.Pose2D] = None,
    ) -> dict:
        """Optimize (unless ``first_free`` is None), re-render, swap. Holds ``_graph_job``."""
        with self._lock:
            snap = self._graph.snapshot()
            base = self._graph_base
            res = self._grid.resolution
            revision = self._graph_revision
        n = len(snap)
        poses = snap.poses
        info: dict = {"nodes": n, "edges": len(snap.edges)}
        if first_free is not None and n > 1:
            window = max(1, int(self._cfg.builtin_loop_max_window_nodes))
            ff = max(1, int(first_free), n - window)
            started = perf_counter()
            res_opt = pg.optimize(snap.poses, snap.edges, first_free=ff)
            ms = (perf_counter() - started) * 1000.0
            self._loop_stats["optimizations"] += 1
            self._loop_stats["last_optimize_ms"] = round(ms, 1)
            info.update(
                free_nodes=res_opt.free_nodes,
                iterations=res_opt.iterations,
                chi2_before=round(res_opt.chi2_before, 2),
                chi2_after=round(res_opt.chi2_after, 2),
                optimize_ms=round(ms, 1),
            )
            if new_edges:
                worst = max(pg.edge_chi2(res_opt.poses, e) for e in new_edges)
                info["loop_chi2"] = round(worst, 2)
                if worst > self._LOOP_MAX_CHI2:
                    with self._lock:
                        self._graph.edges = [
                            e for e in self._graph.edges
                            if not any(e is x for x in new_edges)
                        ]
                    self._loop_stats["rejected_inconsistent"] += 1
                    info["status"] = "rejected_inconsistent"
                    return info
            poses = res_opt.poses
        started = perf_counter()
        grid = pg.render_grid(snap, resolution=res, poses=poses, base=base)
        render_ms = (perf_counter() - started) * 1000.0
        self._loop_stats["last_render_ms"] = round(render_ms, 1)
        info["render_ms"] = round(render_ms, 1)
        with self._lock:
            if self._graph_revision != revision:
                info["status"] = "aborted"
                return info
            tail = self._graph.apply_poses(snap, poses)
            # Inserts that landed while optimizing/rendering.
            for pose, s in self._graph.scans_after(snap.last_seq):
                grid = pg.paint_scan(grid, pose, s)
            self._replay_grid_edits_locked(grid)
            self._grid = grid
            self._generation += 1
            self._invalidate_occ_cache()
            self._last_loop_rebuild_at = time.monotonic()
            corr = live_correction if live_correction is not None else tail
            new_pose = conv.compose_poses(corr, self._pose)
            self._set_corrected_pose_locked(new_pose)
            self._graph_revision += 1
        info["status"] = "rebuilt"
        info["live_correction"] = {
            "x": round(corr.x, 3),
            "y": round(corr.y, 3),
            "theta_deg": round(math.degrees(corr.theta), 2),
        }
        self._notify_pose_listeners(new_pose)
        self._persist_pose(force=True)
        return info

    # -- automatic loop closure ---------------------------------------------
    def _loop_closure_enabled(self) -> bool:
        return (
            bool(getattr(self._cfg, "builtin_loop_closure", True))
            and self._mode == MODE_MAPPING
            and getattr(self._sensors, "world", None) is None
        )

    def _loop_closure_worker(self) -> None:
        while self._running:
            self._loop_event.wait(timeout=1.0)
            self._loop_event.clear()
            if not self._running:
                break
            try:
                self.run_loop_closure()
            except Exception as exc:  # noqa: BLE001
                self._log(f"builtin SLAM loop closure failed: {exc}")

    @staticmethod
    def _loop_edge(
        snap: pg.GraphSnapshot, cur: int, match: loop_closure.LoopMatch
    ) -> pg.GraphEdge:
        # Match covariance is map-frame; edge errors are in the matched pose frame.
        c, s = math.cos(match.pose.theta), math.sin(match.pose.theta)
        rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        info = np.linalg.inv(rot @ match.cov @ rot.T)
        if match.anchor is None:
            return pg.GraphEdge(pg.PRIOR_NODE, cur, match.pose, info, loop=True)
        rel = conv.compose_poses(conv.invert_pose(snap.pose(match.anchor)), match.pose)
        return pg.GraphEdge(match.anchor, cur, rel, info, loop=True)

    @staticmethod
    def _corrections_agree(a: conv.Pose2D, b: conv.Pose2D, at: conv.Pose2D) -> bool:
        pa = conv.compose_poses(a, at)
        pb = conv.compose_poses(b, at)
        return (
            math.hypot(pa.x - pb.x, pa.y - pb.y) <= 0.15
            and abs(conv.normalize_angle(pa.theta - pb.theta)) <= math.radians(2.0)
        )

    def run_loop_closure(self) -> dict:
        """One detection pass on the newest keyframe; closes after two agree."""
        if not self._loop_closure_enabled():
            return {"status": "disabled"}
        if not self._graph_job.acquire(blocking=False):
            return {"status": "busy"}
        try:
            with self._lock:
                self._loop_last_attempt_at = time.monotonic()
                if self._graph.full:
                    if not self._loop_full_logged:
                        self._loop_full_logged = True
                        self._log(
                            "builtin SLAM loop closure off for this session: "
                            f"{len(self._graph)} keyframes "
                            "(builtin_mapping_keyframe_max)"
                        )
                    return {"status": "graph_full"}
                snap = self._graph.snapshot()
                base = self._graph_base
                res = self._grid.resolution
                revision = self._graph_revision
            match, reason = loop_closure.detect(
                snap, self._loop_params, resolution=res, base=base
            )
            stats = self._loop_stats
            stats["attempts"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            if reason != "ok" or match is None:
                self._loop_pending = None
                return {"status": reason}
            cur = len(snap) - 1
            prior = snap.pose(cur)
            corr = pg.world_correction(prior, match.pose)
            edge = self._loop_edge(snap, cur, match)
            pending = self._loop_pending
            if (
                pending is None
                or pending[0] != revision
                or pending[1] == cur
                or not self._corrections_agree(pending[2], corr, prior)
            ):
                self._loop_pending = (revision, cur, corr, edge)
                return {"status": "pending"}
            self._loop_pending = None
            edges = [pending[3], edge]
            with self._lock:
                if self._graph_revision != revision:
                    return {"status": "aborted"}
                for e in edges:
                    self._graph.add_edge(e)
            shift_m = math.hypot(match.pose.x - prior.x, match.pose.y - prior.y)
            shift_deg = abs(math.degrees(conv.normalize_angle(match.pose.theta - prior.theta)))
            stats["closures"] += 1
            stats["last"] = {
                "at": time.time(),
                "node": cur,
                "anchor": match.anchor,
                "gap_m": round(match.gap_m, 2),
                "score": round(match.fit, 3),
                "prior_score": round(match.prior_fit, 3),
                "overlap": round(match.overlap, 3),
                "shift_m": round(shift_m, 3),
                "shift_deg": round(shift_deg, 2),
            }
            if shift_m < self._LOOP_REBUILD_MIN_M and shift_deg < self._LOOP_REBUILD_MIN_DEG:
                # Matcher-quantization sized: keep the edge for the next
                # optimization but skip a full re-render now.
                return {"status": "constrained", **stats["last"]}
            anchors = [e.i for e in edges if e.i >= 0]
            first_free = min(anchors) + 1 if anchors else 1
            result = self._optimize_and_render(first_free, new_edges=edges)
            stats["last"]["result"] = result.get("status")
            self._log(
                "builtin SLAM loop closure: "
                f"{result.get('status')} node={cur} anchor={match.anchor} "
                f"gap={match.gap_m:.1f} m shift={shift_m:.2f} m {shift_deg:.1f} deg "
                f"score={match.fit:.2f} (prior {match.prior_fit:.2f}) "
                f"opt={result.get('optimize_ms')} ms render={result.get('render_ms')} ms"
            )
            return {**stats["last"], **result}
        finally:
            self._graph_job.release()

    def save_map(self, map_dir: Path) -> None:
        with self._lock:
            persistence.save_occupancy(map_dir, self._grid)
            self._map_dir = Path(map_dir)
            pose = self._pose
            seed_pending = self._seed_localize_pending
            mode = self._mode
        if bool(getattr(self._cfg, "persist_pose", True)) and not (
            mode == MODE_LOCALIZING and seed_pending
        ):
            try:
                persistence.save_last_pose(map_dir, pose)
                self._last_pose_save_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                self._log(f"persist last pose failed: {exc}")

    def _invalidate_occ_cache(self) -> None:
        self._match_epoch += 1
        self._occ_cache = None
        self._occ_cache_epoch = -1

    def _occupancy_for_match(self) -> OccupancyMap:
        """Scan-match grid. The int16 convert runs outside the pose lock."""
        with self._lock:
            if (
                self._occ_cache is not None
                and self._occ_cache_epoch == self._match_epoch
            ):
                return self._occ_cache
            grid = self._grid
            epoch = self._match_epoch
        int16 = occ.to_occupancy_int16(grid)
        occ_map = scan_match.occupancy_map_from_int16(
            int16,
            resolution=grid.resolution,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
        )
        known = float(np.mean(int16 >= 0))
        with self._lock:
            if self._grid is grid and self._match_epoch == epoch:
                self._occ_cache = occ_map
                self._occ_cache_epoch = epoch
                self._occ_cache_known = known
            elif self._occ_cache is None:
                self._occ_cache_known = known
        return occ_map

    # -- queries -------------------------------------------------------------
    def get_pose(self) -> conv.Pose2D:
        with self._lock:
            return self._pose

    def get_odom_twist(self) -> tuple[float, float, float]:
        with self._lock:
            return self._last_odom_twist

    def get_map(self) -> dict:
        with self._lock:
            grid = self._grid
            generation = self._generation
        # Convert off the lock. Published grids are replaced, not mutated, so
        # this snapshot stays stable while pose reads proceed.
        return {
            "grid": occ.to_occupancy_int16(grid),
            "resolution": grid.resolution,
            "origin_x": grid.origin_x,
            "origin_y": grid.origin_y,
            "generation": generation,
        }

    def diagnostics(self) -> dict:
        odom_status_fn = getattr(self._sensors, "odom_status", None)
        odom_status = None
        if callable(odom_status_fn):
            try:
                odom_status = odom_status_fn()
            except Exception:  # noqa: BLE001
                odom_status = None
        with self._lock:
            return {
                "odom": odom_status,
                "slam_backend": "builtin",
                "mode": self._mode,
                "running": self._running,
                "ticks": self._ticks,
                "map_updates": self._updates,
                "match_accepts": self._match_accepts,
                "match_rejects": self._match_rejects,
                "nav_track_applies": self._nav_track_applies,
                "pose": {
                    "x": self._pose.x,
                    "y": self._pose.y,
                    "theta": self._pose.theta,
                },
                "grid": {
                    "width": self._grid.width,
                    "height": self._grid.height,
                    "resolution": self._grid.resolution,
                    "origin_x": self._grid.origin_x,
                    "origin_y": self._grid.origin_y,
                },
                "last_match_score": self._last_match_score,
                "last_prior_score": self._last_prior_score,
                "last_scan_age_s": self._last_scan_age_s,
                "yaw_rate_deg_s": round(math.degrees(self._last_yaw_rate), 2),
                "insert_skips_turning": self._insert_skips_turning,
                "generation": self._generation,
                "seed_localize_pending": self._seed_localize_pending,
                "pose_restored_from_disk": self._pose_restored_from_disk,
                "keyframes": self._graph.scan_count(),
                "last_loop_rebuild_at": self._last_loop_rebuild_at,
                "loop_closure": {
                    "enabled": self._loop_closure_enabled(),
                    "nodes": len(self._graph),
                    "edges": len(self._graph.edges),
                    "loop_edges": self._graph.loop_edge_count(),
                    "graph_full": self._graph.full,
                    "pending": self._loop_pending is not None,
                    "attempts": self._loop_stats["attempts"],
                    "closures": self._loop_stats["closures"],
                    "optimizations": self._loop_stats["optimizations"],
                    "rejected_inconsistent": self._loop_stats["rejected_inconsistent"],
                    "reasons": dict(self._loop_stats["reasons"]),
                    "last": self._loop_stats["last"],
                    "last_optimize_ms": self._loop_stats["last_optimize_ms"],
                    "last_render_ms": self._loop_stats["last_render_ms"],
                },
                "lidar_odometry": {
                    "enabled": self._lidar_odom_enabled(),
                    "accepts": self._lo_accepts,
                    "rejects": self._lo_rejects,
                    "resets": self._lo_resets,
                    "inserts_held": self._lo_inserts_held,
                    "last_inlier_ratio": self._lo_last_inlier,
                    "last_rms_m": self._lo_last_rms,
                    "forward_m_s": round(self._lo_vel[0], 3),
                    "left_m_s": round(self._lo_vel[1], 3),
                    "turn_rate_deg_s": round(math.degrees(self._lo_turn_rate), 1),
                    "last_match_ms": round(self._lo_match_ms, 1),
                    # After a spin these should agree; a large ratio means the
                    # IMU's turn-rate units are off.
                    "yaw_check": {
                        "gyro_deg": round(math.degrees(self._yaw_check_gyro), 1),
                        "lidar_deg": round(math.degrees(self._yaw_check_lidar), 1),
                    },
                },
            }

    # -- loop ----------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                self._log(f"builtin SLAM tick failed: {exc}")
            else:
                self._persist_pose(force=False)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._period - elapsed))

    def _tick(self) -> None:
        # Matching/mapping against a stale scan biases the pose toward where
        # the robot was when the scan was taken; keep the age tight even if
        # the nav-facing scan_max_age_s is generous.
        scan_age = min(float(self._cfg.scan_max_age_s or 2.0), 0.75)
        get_scan = self._sensors.get_scan
        try:
            scan = get_scan(max_age_s=scan_age, fresh=True)
        except TypeError:  # test doubles without the ``fresh`` kwarg
            scan = get_scan(max_age_s=scan_age)
        odom = self._sensors.get_odom()
        now = time.monotonic()
        age_fn = getattr(self._sensors, "scan_age_s", None)
        if callable(age_fn):
            try:
                self._last_scan_age_s = float(age_fn())
            except Exception:  # noqa: BLE001
                pass
        # A scan object we already matched/inserted must not be re-used at a
        # newer pose: during a turn that paints the same wall at every heading.
        new_scan = scan is not None and scan is not self._last_scan_obj
        if scan is not None:
            self._last_scan_obj = scan

        # Builtin sim: body pose in SimWorld is ground truth (raycast + kinematics).
        # Track it directly so localize/set_pose cannot yank the body or desync
        # scans from the estimate. Real-robot path below is unchanged.
        world = getattr(self._sensors, "world", None)
        if world is not None:
            self._tick_sim_world(world, scan, odom, now, new_scan=new_scan)
            return

        with self._lock:
            seed_pending = (
                self._mode == MODE_LOCALIZING and self._seed_localize_pending
            )

        if seed_pending:
            # Hold absolute pose at whatever set_pose/origin is until a full-map
            # seed lands — do not odom-integrate from an assumed start point.
            if scan is None:
                return
            if now - self._last_seed_attempt_at < 2.0:
                return
            self._last_seed_attempt_at = now
            self._try_seed_global_localize(scan)
            return

        with self._lock:
            self._ticks += 1
            prev_pose, prev_at = self._last_tick_pose, self._last_tick_at
            predicted = self._predict(odom, now)
            self._pose = predicted
            self._last_tick_pose, self._last_tick_at = predicted, now
            yaw_rate = 0.0
            if prev_pose is not None and prev_at is not None and now > prev_at:
                yaw_rate = abs(
                    conv.normalize_angle(predicted.theta - prev_pose.theta)
                ) / (now - prev_at)
            self._last_yaw_rate = yaw_rate
            if scan is None or not new_scan:
                return
            use_lo = self._lidar_odom_active(odom)
            if self._lo_reset_pending:
                self._lo_reset_pending = False
                self._lo.reset()
            match_due = now - self._last_match_at >= self._match_period_s
            epoch = self._pose_epoch

        occ_map = None
        run_match = False
        if match_due:
            occ_map = self._occupancy_for_match()
            run_match = self._occ_cache_known > 0.02

        lo_ok = False
        turning = False
        if use_lo:
            gyro_rate = abs(float(odom.vtheta)) if odom is not None else 0.0
            lo_ok = self._lidar_odom_step(scan, now, gyro_rate=gyro_rate)
            turning = max(gyro_rate, self._lo_turn_rate) > self._LO_MAX_INSERT_YAW_RATE
            if lo_ok and self._mode == MODE_MAPPING:
                # The map is being built from these poses; the grid matcher
                # only re-checks them against themselves.
                run_match = False

        # Heavy work happens OUTSIDE the lock: matching while holding it
        # blocked every get_pose/get_map (nav planning + pose reads stalled).
        if run_match:
            # While driving, search far across the hall and only a short way
            # along it. A square window either misses a sideways drift or
            # slides into a lookalike stretch of the same corridor.
            nav_track = self._navigation_active() and bool(
                getattr(self._cfg, "nav_scan_track", True)
            )
            if nav_track:
                matched, score, prior_score = scan_match.refine_pose(
                    occ_map,
                    scan,
                    predicted,
                    along_half_m=self._NAV_TRACK_MAX_M,
                    cross_half_m=self._NAV_LATERAL_MAX_M,
                    yaw_half_deg=math.degrees(self._NAV_LATERAL_MAX_RAD),
                )
            else:
                matched, score, prior_score = scan_match.refine_pose(
                    occ_map, scan, predicted
                )
            with self._lock:
                self._last_match_at = now
                self._last_match_score = score
                self._last_prior_score = prior_score
                if self._pose_epoch != epoch:
                    # Pose moved frame (loop closure) mid-match.
                    return
                applied = self._apply_match(self._pose, matched, odom)
                if matched is not None and applied:
                    self._match_accepts += 1
                else:
                    self._match_rejects += 1

        if self._mode == MODE_MAPPING:
            with self._lock:
                if self._pose_epoch != epoch:
                    return
            if yaw_rate > self._MAX_INSERT_YAW_RATE:
                # One lidar revolution (~100 ms) is skewed by yaw_rate*0.1 s;
                # inserting it during a fast spin paints walls as arcs.
                self._insert_skips_turning += 1
            elif use_lo and turning and self._last_insert_pose is not None:
                self._insert_skips_turning += 1
            elif use_lo and not lo_ok and self._last_insert_pose is not None:
                # No scan match: the pose is a guess. Painting there doubles walls.
                self._lo_inserts_held += 1
            else:
                self._maybe_insert_scan(scan, now)

    # -- lidar odometry --------------------------------------------------------
    # Lidar velocity is trusted this long after the last match; after that the
    # IMU-only prediction takes over again.
    _LO_FRESH_S = 1.5
    # Velocity decays toward 0 once matches stop arriving.
    _LO_DECAY_AFTER_S = 0.3
    _LO_DECAY_TAU_S = 0.5
    _LO_GAIN = 0.5
    _LO_MAX_FORWARD_M_S = 1.2
    _LO_MAX_LATERAL_M_S = 0.3
    # Consecutive misses before the local map restarts from the current pose.
    _LO_MAX_FAILS = 10
    # Occupied cells (one hit is 0.85) also anchor matching; refreshed after
    # this many map inserts.
    _LO_MAP_OCC_LOG_ODDS = 1.5
    _LO_MAP_REFRESH_INSERTS = 10
    # One ~100 ms Livox frame taken while turning is smeared by rate * 0.1 s;
    # above this the scan still tracks the pose but is not drawn.
    _LO_MAX_INSERT_YAW_RATE = math.radians(15.0)

    def _lidar_odom_enabled(self) -> bool:
        return bool(getattr(self._cfg, "lidar_odometry", False))

    def _lidar_odom_active(self, odom: Optional[conv.OdomReading]) -> bool:
        """Only for samples without wheel speed or an odom pose (not the Tracer)."""
        if not self._lidar_odom_enabled():
            return False
        if odom is None:
            return True
        return (
            odom.pose is None
            and abs(float(odom.vx)) <= 1e-3
            and abs(float(odom.vy)) <= 1e-3
        )

    def _lidar_odom_fresh(self, now: float) -> bool:
        return self._lo_last_at is not None and now - self._lo_last_at <= self._LO_FRESH_S

    def _occupied_map_points_locked(self) -> np.ndarray:
        grid = self._grid
        rows, cols = np.nonzero(grid.log_odds > self._LO_MAP_OCC_LOG_ODDS)
        res = grid.resolution
        return np.stack(
            [grid.origin_x + (cols + 0.5) * res, grid.origin_y + (rows + 0.5) * res],
            axis=1,
        )

    def _lidar_odom_step(
        self, scan: conv.LaserScan2D, now: float, *, gyro_rate: float = 0.0
    ) -> bool:
        """ICP this scan against recent scans and the map. True when the pose came from it."""
        cloud = scan.cloud_xy
        scan_xy = cloud if cloud is not None and cloud.shape[0] else scan.to_points()
        with self._lock:
            guess = self._pose
            epoch = self._pose_epoch
            map_pts = None
            if (
                self._lo_map_updates is None
                or self._updates - self._lo_map_updates >= self._LO_MAP_REFRESH_INSERTS
            ):
                map_pts = self._occupied_map_points_locked()
                self._lo_map_updates = self._updates
        if map_pts is not None:
            self._lo.set_map_points(map_pts)
        if not self._lo.has_reference():
            self._lo.add_keyscan(scan_xy, guess, force=True)
            with self._lock:
                self._lo_last_at = now
                self._lo_prev_theta = guess.theta
            return self._last_insert_pose is None
        started = perf_counter()
        result = self._lo.match(scan_xy, guess)
        self._lo_match_ms = (perf_counter() - started) * 1000.0
        with self._lock:
            if self._pose_epoch != epoch:
                return False
            if result is None:
                self._lo_rejects += 1
                self._lo_fails += 1
                if self._lo_fails >= self._LO_MAX_FAILS:
                    self._lo_fails = 0
                    self._lo_resets += 1
                    self._lo_vel = (0.0, 0.0)
                    self._lo_prev_theta = self._pose.theta
                    self._lo_turn_rate = 0.0
                    self._lo.reset()
                    self._lo.add_keyscan(scan_xy, self._pose, force=True)
                    self._log("builtin SLAM: lidar odometry lost track; restarting local map")
                return False
            current = self._pose
            pose = result.pose
            if self._lo_last_at is not None:
                dt = now - self._lo_last_at
                if 0.0 < dt <= 1.0:
                    rx, ry = pose.x - current.x, pose.y - current.y
                    c, s = math.cos(current.theta), math.sin(current.theta)
                    fwd = self._lo_vel[0] + self._LO_GAIN * (c * rx + s * ry) / dt
                    left = self._lo_vel[1] + self._LO_GAIN * (-s * rx + c * ry) / dt
                    fwd = max(-self._LO_MAX_FORWARD_M_S, min(self._LO_MAX_FORWARD_M_S, fwd))
                    left = max(-self._LO_MAX_LATERAL_M_S, min(self._LO_MAX_LATERAL_M_S, left))
                    self._lo_vel = (fwd, left)
            if self._lo_prev_theta is not None:
                turned = conv.normalize_angle(pose.theta - self._lo_prev_theta)
                self._yaw_check_lidar += turned
                if self._lo_last_at is not None and now > self._lo_last_at:
                    self._lo_turn_rate = abs(turned) / (now - self._lo_last_at)
            self._lo_prev_theta = pose.theta
            self._pose = pose
            self._lo_last_at = now
            self._lo_fails = 0
            self._lo_accepts += 1
            self._lo_last_inlier = round(result.inlier_ratio, 3)
            self._lo_last_rms = round(result.rms_m, 4)
            self._imu_forward_m_s = 0.0
        if max(gyro_rate, self._lo_turn_rate) <= self._LO_MAX_INSERT_YAW_RATE:
            # A frame skewed by a turn would tilt every match that uses it.
            self._lo.add_keyscan(scan_xy, pose)
        return True

    def _lidar_odom_twist(self, now: float, dt: float) -> tuple[float, float]:
        """Body (forward, left) velocity for prediction, decayed when matches stop."""
        fwd, left = self._lo_vel
        if self._lo_last_at is not None and now - self._lo_last_at > self._LO_DECAY_AFTER_S:
            k = math.exp(-dt / self._LO_DECAY_TAU_S)
            fwd, left = fwd * k, left * k
            self._lo_vel = (fwd, left)
        return fwd, left

    def _tick_sim_world(
        self,
        world,
        scan,
        odom: Optional[conv.OdomReading],
        now: float,
        *,
        new_scan: bool,
    ) -> None:
        """SLAM pose follows SimWorld; match scores are diagnostic only."""
        pose = world.get_pose()
        with self._lock:
            prev_pose, prev_at = self._last_tick_pose, self._last_tick_at
            self._pose = pose
            # World pose is authoritative — never wait on seed global_localize.
            self._seed_localize_pending = False
            self._pending_match = None
            self._pending_count = 0
            self._ticks += 1
            self._last_tick_pose, self._last_tick_at = pose, now
            if odom is not None and odom.pose is not None:
                self._last_odom_pose = conv.Pose2D(
                    odom.pose.x, odom.pose.y, odom.pose.theta
                )
                self._last_odom_heading = odom.heading_rad
                self._last_odom_time = now
            yaw_rate = 0.0
            if prev_pose is not None and prev_at is not None and now > prev_at:
                yaw_rate = abs(
                    conv.normalize_angle(pose.theta - prev_pose.theta)
                ) / (now - prev_at)
            self._last_yaw_rate = yaw_rate
            if scan is None or not new_scan:
                return
            match_due = now - self._last_match_at >= self._match_period_s

        occ_map = None
        run_match = False
        if match_due:
            occ_map = self._occupancy_for_match()
            run_match = self._occ_cache_known > 0.02

        if run_match:
            matched, score, prior_score = scan_match.refine_pose(
                occ_map, scan, pose
            )
            with self._lock:
                self._last_match_at = now
                self._last_match_score = score
                self._last_prior_score = prior_score
                # Do not apply match to pose (would fight SimWorld). Count how
                # often the map agrees with ground truth for status.
                if matched is not None:
                    dist = math.hypot(matched.x - pose.x, matched.y - pose.y)
                    dyaw = abs(conv.normalize_angle(matched.theta - pose.theta))
                    if dist <= self._SMALL_CORRECTION_M and dyaw <= self._SMALL_CORRECTION_RAD:
                        self._match_accepts += 1
                    else:
                        self._match_rejects += 1
                else:
                    self._match_rejects += 1

        if self._mode == MODE_MAPPING:
            if yaw_rate > self._MAX_INSERT_YAW_RATE:
                self._insert_skips_turning += 1
            else:
                self._maybe_insert_scan(scan, now)

    # Above this yaw rate a revolution is skewed >~5 deg; keep matching but
    # do not bake the skewed scan into the map.
    _MAX_INSERT_YAW_RATE = math.radians(45.0)

    def _try_seed_global_localize(self, scan) -> bool:
        """Full-map match once scans are available; apply if score is trusted."""
        from ..nav.global_localize import global_localize_scan

        try:
            occ_map = self._occupancy_for_match()
        except Exception:  # noqa: BLE001
            return False
        known = self._occ_cache_known
        if known < 0.02:
            return False
        try:
            result = global_localize_scan(
                occ_map,
                scan,
                hint=None,
                full_map=True,
                coarse_position_step_m=0.6,
                coarse_yaw_step_deg=18.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"seed global localize failed: {exc}")
            return False
        score = float(result.score)
        if not math.isfinite(score) or score < self._seed_min_score:
            self._log(
                f"seed global localize weak (score={score:.3f} "
                f"ray_mae={result.ray_mae_m:.3f}); retrying"
            )
            return False
        pose = result.pose
        with self._lock:
            current = self._pose
        decision = self._pose_jump_gate.evaluate(current, pose, force=False)
        if not decision.should_apply:
            self._log(
                f"seed global localize awaiting confirm "
                f"{decision.confirm_count}/{decision.confirm_needed} "
                f"(shift={decision.shift_m:.2f} m score={score:.3f})"
            )
            return False
        with self._lock:
            self._pose = pose
            self._seed_localize_pending = False
            self._last_odom_pose = None
            self._last_odom_heading = None
            self._last_odom_time = None
            self._pending_match = None
            self._pending_count = 0
            self._last_match_score = score
        self._notify_pose_listeners(pose)
        self._log(
            f"seed global localize applied "
            f"x={pose.x:.2f} y={pose.y:.2f} th={pose.theta:.2f} "
            f"score={score:.3f} ray_mae={result.ray_mae_m:.3f} "
            f"jump={decision.status}"
        )
        return True

    def _maybe_insert_scan(self, scan, now: float) -> None:
        """Ray-cast into the grid only after movement or a timeout.

        The paint runs off the pose lock and in numpy, so GetPosition and the
        nav control loop are not stuck behind a Python Bresenham.
        """
        with self._lock:
            pose = self._pose
            last = self._last_insert_pose
            if last is not None:
                moved = math.hypot(pose.x - last.x, pose.y - last.y)
                turned = abs(conv.normalize_angle(pose.theta - last.theta))
                if (
                    moved < 0.03
                    and turned < math.radians(1.5)
                    and now - self._last_insert_at < 1.0
                ):
                    return
            src = self._grid
            pose_snap = conv.Pose2D(pose.x, pose.y, pose.theta)
        painted = occ.insert_scan(
            occ.copy_grid(src),
            pose_snap.x,
            pose_snap.y,
            pose_snap.theta,
            np.asarray(scan.ranges, dtype=float),
            float(scan.angle_min),
            float(scan.angle_increment),
            range_min=float(scan.range_min),
            range_max=float(scan.range_max)
            if math.isfinite(scan.range_max)
            else 30.0,
        )
        hook = None
        with self._lock:
            # Drop the paint if clear/reset/rebuild replaced the grid mid-ray.
            if self._grid is not src:
                return
            self._grid = painted
            self._last_insert_pose = pose_snap
            self._last_insert_at = now
            self._updates += 1
            if self._mode == MODE_MAPPING:
                hook = self._keyframe_hook
                nodes = len(self._graph)
                self._graph.add_scan(pose_snap, scan)
                if (
                    len(self._graph) > nodes
                    and self._loop_closure_enabled()
                    and now - self._loop_last_attempt_at
                    >= float(self._cfg.builtin_loop_check_interval_s)
                ):
                    self._loop_event.set()
            if self._updates % 20 == 0:
                self._generation += 1
                self._invalidate_occ_cache()
        if hook is not None:
            try:
                hook(scan, [], pose_snap)
            except Exception as exc:  # noqa: BLE001
                self._log(f"keyframe hook failed: {exc}")

    def _poses_agree(self, a: conv.Pose2D, b: conv.Pose2D) -> bool:
        return (
            math.hypot(a.x - b.x, a.y - b.y) <= 0.08
            and abs(conv.normalize_angle(a.theta - b.theta)) <= math.radians(5.0)
        )

    def _blend_pose(
        self,
        prior: conv.Pose2D,
        target: conv.Pose2D,
        *,
        alpha: float,
        max_xy: float,
        max_yaw: float,
    ) -> conv.Pose2D:
        """Move partially toward ``target`` and clamp the step size."""
        alpha = min(max(alpha, 0.0), 1.0)
        x = prior.x + alpha * (target.x - prior.x)
        y = prior.y + alpha * (target.y - prior.y)
        th = conv.normalize_angle(
            prior.theta + alpha * conv.normalize_angle(target.theta - prior.theta)
        )
        dx = x - prior.x
        dy = y - prior.y
        dist = math.hypot(dx, dy)
        if dist > max_xy and dist > 1e-9:
            scale = max_xy / dist
            x = prior.x + dx * scale
            y = prior.y + dy * scale
        dth = conv.normalize_angle(th - prior.theta)
        if abs(dth) > max_yaw:
            th = conv.normalize_angle(prior.theta + math.copysign(max_yaw, dth))
        return conv.Pose2D(x, y, th)

    # A correction this small is drift tracking, not a peak jump — apply it
    # immediately. Requiring confirmation + tiny clamps for these let odom
    # drift outrun the correction rate (pose "arrived" before the robot).
    _SMALL_CORRECTION_M = 0.15
    _SMALL_CORRECTION_RAD = math.radians(8.0)

    # Continuous local refine is only trusted as drift tracking when the
    # published pose already explains the scan. Below this, odom-only until
    # global_localize / manual seed — otherwise the ±0.3 m window walks
    # toward corridor twins (including while nav is suspended/paused).
    _PRIOR_TRUST_SCORE = 0.15

    # Mid-nav drift tracking (``nav_scan_track``). Stricter than idle: the
    # published pose must already fit the scan, the match must be good, the
    # base must not be spinning, and each step is small and gently blended.
    # Along-hall motion stays inside ``_NAV_TRACK_MAX_M``. Cross-track may be
    # larger (``_NAV_LATERAL_MAX_M``): the walls make that axis observable.
    # A skid-steer can slip about a meter and 15° before the bumper meets the
    # wall; a tighter window made that pose invisible and the heading error
    # vetoed the sideways step. Heading is bled a few degrees at a time.
    _NAV_TRACK_MAX_M = 0.15
    _NAV_TRACK_MAX_RAD = math.radians(3.0)
    _NAV_LATERAL_MAX_M = 1.20
    _NAV_LATERAL_MAX_RAD = math.radians(20.0)
    _NAV_LATERAL_STEP_M = 0.10
    _NAV_LATERAL_YAW_STEP = math.radians(3.0)
    _NAV_LATERAL_YAW_AGREE = math.radians(4.0)
    # Absolute score floor once the published pose no longer explains the scan.
    # The in-window leftover on a drifted hall is about 0.12; a real wall
    # alignment has to clear this.
    _NAV_LATERAL_MIN_SCORE = 0.20
    _NAV_TRACK_MIN_PRIOR = 0.25
    _NAV_TRACK_MIN_SCORE = 0.35
    _NAV_TRACK_MAX_YAW_RATE = 0.35
    _NAV_TRACK_MAX_SCAN_AGE_S = 0.25
    _NAV_TRACK_ALPHA = 0.3
    _NAV_TRACK_STEP_M = 0.05
    _NAV_TRACK_STEP_RAD = math.radians(1.0)

    def _nav_scan_calm(self) -> bool:
        if not bool(getattr(self._cfg, "nav_scan_track", True)):
            return False
        if self._last_yaw_rate > self._NAV_TRACK_MAX_YAW_RATE:
            return False
        age = float(self._last_scan_age_s)
        if math.isfinite(age) and age > self._NAV_TRACK_MAX_SCAN_AGE_S:
            return False
        return True

    def _nav_track_fresh(self) -> bool:
        if not self._nav_scan_calm():
            return False
        score = float(self._last_match_score)
        return math.isfinite(score) and score >= self._NAV_TRACK_MIN_SCORE

    def _nav_lateral_score_ok(self) -> bool:
        """Sideways recovery may use a lower score once the prior is lost."""
        if not self._nav_scan_calm():
            return False
        score = float(self._last_match_score)
        if not math.isfinite(score):
            return False
        prior = float(self._last_prior_score)
        if math.isfinite(prior) and prior >= self._NAV_TRACK_MIN_PRIOR:
            return score >= self._NAV_TRACK_MIN_SCORE
        return score >= self._NAV_LATERAL_MIN_SCORE

    def _nav_track_ok(self, predicted: conv.Pose2D, matched: conv.Pose2D) -> bool:
        """Small drift: the published pose already explains the scan."""
        if not self._nav_track_fresh():
            return False
        along, cross, dyaw = scan_match.shift_axes(predicted, matched)
        if abs(along) > self._NAV_TRACK_MAX_M or abs(cross) > self._NAV_TRACK_MAX_M:
            return False
        if abs(dyaw) > self._NAV_TRACK_MAX_RAD:
            return False
        prior = float(self._last_prior_score)
        return math.isfinite(prior) and prior >= self._NAV_TRACK_MIN_PRIOR

    def _nav_lateral_ok(self, predicted: conv.Pose2D, matched: conv.Pose2D) -> bool:
        """Sideways hallway correction, plus a small heading bleed.

        The prior is allowed to be bad: that is the drifted pose. Along-hall
        stays put. A heading error up to ``_NAV_LATERAL_MAX_RAD`` does not
        veto the step; each apply only turns a few degrees. One frame only
        arms the correction; the next agreeing frame takes a small step.
        """
        if not self._nav_lateral_score_ok():
            return False
        along, cross, dyaw = scan_match.shift_axes(predicted, matched)
        if abs(along) > self._NAV_TRACK_MAX_M or abs(dyaw) > self._NAV_LATERAL_MAX_RAD:
            return False
        if abs(cross) > self._NAV_LATERAL_MAX_M:
            return False
        # The small-drift path already covers a tight, well-scored pose.
        return (
            abs(cross) > self._NAV_TRACK_MAX_M or abs(dyaw) > self._NAV_TRACK_MAX_RAD
        )

    def _navigation_active(self) -> bool:
        """True when builtin nav has an in-flight goal (MoveOnMap / route)."""
        try:
            from ..runtime import any_navigation_active

            return bool(any_navigation_active())
        except Exception:  # noqa: BLE001 - registry may be mid-reconfigure
            return False

    def _apply_match(
        self,
        predicted: conv.Pose2D,
        matched: Optional[conv.Pose2D],
        odom: Optional[conv.OdomReading],
    ) -> bool:
        """Apply a blended match. Returns True if pose was corrected."""
        del odom  # prediction is delta-based; no map->odom re-anchor needed
        if matched is None:
            self._pending_match = None
            self._pending_count = 0
            self._pose = predicted
            return False

        # During MoveOnMap / routes: no jumps along the hall or in heading.
        # A sideways offset is applied in small steps once two scans agree,
        # so a drifted hallway pose can walk back to the walls without a
        # teleport into the next corridor.
        if self._navigation_active():
            self._pending_match = None
            self._pending_count = 0
            if self._nav_track_ok(predicted, matched):
                self._lateral_pending_cross = None
                self._lateral_pending_yaw = None
                self._pose = self._blend_pose(
                    predicted,
                    matched,
                    alpha=self._NAV_TRACK_ALPHA,
                    max_xy=self._NAV_TRACK_STEP_M,
                    max_yaw=self._NAV_TRACK_STEP_RAD,
                )
                self._nav_track_applies += 1
                return True
            if self._nav_lateral_ok(predicted, matched):
                _along, cross, dyaw = scan_match.shift_axes(predicted, matched)
                # Agree on the sideways offset and the heading error, not the
                # map point. The cart moves between scans, so the world target
                # moves with it.
                agreed = (
                    self._lateral_pending_cross is not None
                    and self._lateral_pending_yaw is not None
                    and abs(cross - self._lateral_pending_cross) <= 0.08
                    and abs(conv.normalize_angle(dyaw - self._lateral_pending_yaw))
                    <= self._NAV_LATERAL_YAW_AGREE
                )
                self._lateral_pending_cross = cross
                self._lateral_pending_yaw = dyaw
                if not agreed:
                    self._pose = predicted
                    return False
                if abs(cross) > self._NAV_TRACK_MAX_M:
                    target = scan_match.lateral_correction_pose(
                        predicted,
                        matched,
                        max_along_m=self._NAV_TRACK_MAX_M,
                        max_cross_m=self._NAV_LATERAL_MAX_M,
                        max_yaw_rad=self._NAV_LATERAL_MAX_RAD,
                        min_cross_m=self._NAV_TRACK_MAX_M,
                    )
                    if target is None:
                        self._pose = predicted
                        return False
                else:
                    target = predicted
                pose = self._blend_pose(
                    predicted,
                    target,
                    alpha=self._NAV_TRACK_ALPHA,
                    max_xy=self._NAV_LATERAL_STEP_M,
                    max_yaw=0.0,
                )
                yaw_step = max(
                    -self._NAV_LATERAL_YAW_STEP,
                    min(self._NAV_LATERAL_YAW_STEP, dyaw),
                )
                self._pose = conv.Pose2D(
                    pose.x,
                    pose.y,
                    conv.normalize_angle(predicted.theta + yaw_step),
                )
                self._nav_track_applies += 1
                return True
            self._lateral_pending_cross = None
            self._lateral_pending_yaw = None
            self._pose = predicted
            return False

        dist = math.hypot(matched.x - predicted.x, matched.y - predicted.y)
        dyaw = abs(conv.normalize_angle(matched.theta - predicted.theta))
        small = dist <= self._SMALL_CORRECTION_M and dyaw <= self._SMALL_CORRECTION_RAD
        prior = float(self._last_prior_score)
        prior_ok = math.isfinite(prior) and prior >= self._PRIOR_TRUST_SCORE

        if small:
            # Bad prior while idle/paused: never creep toward twins.
            if not prior_ok:
                self._pending_match = None
                self._pending_count = 0
                self._pose = predicted
                return False
            # Continuous drift correction: no confirmation needed.
            self._pending_match = None
            self._pending_count = 0
            self._pose = self._blend_pose(
                predicted,
                matched,
                alpha=0.6,
                max_xy=self._SMALL_CORRECTION_M,
                max_yaw=self._SMALL_CORRECTION_RAD,
            )
            return True

        # Large jump while idle: only with a trusted prior + two agreeing frames.
        if not prior_ok:
            self._pending_match = None
            self._pending_count = 0
            self._pose = predicted
            return False

        if self._pending_match is not None and self._poses_agree(
            matched, self._pending_match
        ):
            self._pending_count += 1
            self._pending_match = matched
        else:
            self._pending_match = matched
            self._pending_count = 1

        if self._pending_count < 2:
            self._pose = predicted
            return False

        self._pose = self._blend_pose(
            predicted,
            matched,
            alpha=0.6,
            max_xy=0.25,
            max_yaw=math.radians(12.0),
        )
        # Keep the streak so a persistent target keeps applying every match.
        self._pending_count = 2
        return True

    # Ignore accel smaller than this so gravity leftover and sensor noise do
    # not walk the pose while the cart is sitting still.
    _ACCEL_DEADBAND_M_S2 = 0.20
    _ACCEL_SPEED_MAX_M_S = 0.8
    # How fast a constant reading is absorbed into the baseline. A parked
    # tilt must not keep integrating; a real shove stays ahead of this.
    _ACCEL_BIAS_TAU_S = 1.5
    # A resting gyro faster than this is a real turn. Slower rates are the
    # IMU's zero (Wit already near 0, Livox a small bias) and do not turn.
    _YAW_STILL_RAD_S = math.radians(5.0)
    # Seed the resting yaw rate from the first sample only when it is modest.
    # A cart that powers on already spinning is not treated as bias.
    _YAW_BIAS_SEED_MAX_RAD_S = math.radians(30.0)

    def _imu_yaw_delta(self, dth: float, dt: float) -> float:
        """Yaw step for an IMU-only cart.

        The first modest gyro rate is that IMU's zero (a Wit sits near 0; a
        Livox can sit on a bias). Only a change above the still threshold
        turns the pose, so a parked cart does not rotate to a new heading.
        """
        rate = dth / dt if dt > 0.0 else 0.0
        if (
            self._imu_yaw_bias is None
            and abs(rate) <= self._YAW_BIAS_SEED_MAX_RAD_S
        ):
            self._imu_yaw_bias = rate
        bias = 0.0 if self._imu_yaw_bias is None else self._imu_yaw_bias
        residual = rate - bias
        if abs(residual) <= self._YAW_STILL_RAD_S:
            if self._imu_yaw_bias is not None:
                self._imu_yaw_bias += residual * (1.0 - math.exp(-dt / 2.0))
            return 0.0
        return residual * dt

    def _twist_with_accel_prior(
        self, odom: conv.OdomReading, dt: float
    ) -> tuple[float, float]:
        """Body twist for one predict step.

        Wheel velocity is used as reported. With ``imu_odom_mode=accel_only``
        and no wheel speed, a *change* in forward acceleration (``ax`` after
        the IMU mount yaw) is integrated into a short-lived forward speed.
        The steady reading, including gravity left over from a tilt, is the
        baseline and does not move the pose. When the reading returns to that
        baseline the speed is dropped, so stopping does not coast. Lateral
        accel is ignored.
        """
        vx, vy = float(odom.vx), float(odom.vy)
        if getattr(self._cfg, "imu_odom_mode", "") != IMU_ODOM_ACCEL_ONLY:
            return vx, vy
        if odom.ax is None or odom.ay is None:
            return vx, vy
        if abs(vx) > 1e-3 or abs(vy) > 1e-3:
            self._imu_forward_m_s = 0.0
            self._imu_ax_bias = None
            self._imu_yaw_bias = None
            return vx, vy
        ax = float(odom.ax)
        if self._imu_ax_bias is None:
            self._imu_ax_bias = ax
        residual = ax - self._imu_ax_bias
        # Always pull the baseline toward the reading. A resting offset bigger
        # than the deadband (Livox tilt) must be absorbed; otherwise it is
        # integrated forever and the parked cart slides. A short shove stays
        # ahead of this slow follow.
        self._imu_ax_bias += residual * (
            1.0 - math.exp(-dt / self._ACCEL_BIAS_TAU_S)
        )
        residual = ax - self._imu_ax_bias
        if abs(residual) < self._ACCEL_DEADBAND_M_S2:
            self._imu_forward_m_s = 0.0
            speed = 0.0
        else:
            speed = self._imu_forward_m_s + residual * dt
            speed = max(
                -self._ACCEL_SPEED_MAX_M_S, min(self._ACCEL_SPEED_MAX_M_S, speed)
            )
            self._imu_forward_m_s = speed
        if getattr(self._cfg, "base_velocity_convention", "viam") in BASE_VELOCITY_Y_FORWARD:
            return 0.0, speed
        return speed, 0.0

    # Matches ``TypedMovementSensorOdom.MAX_HOLD_GAP_S``. Past this the reader stops
    # inventing motion from the last twist, so a stalled read cannot justify
    # more yaw than that hold either.
    _ODOM_YAW_HOLD_MAX_S = 3.0

    def _predict(
        self, odom: Optional[conv.OdomReading], now: float
    ) -> conv.Pose2D:
        if odom is None:
            return self._pose

        if odom.pose is not None:
            # Absolute odom pose: apply the *relative* motion since the last
            # sample onto the current map pose. Composing map_to_odom with the
            # absolute pose let IMU/magnetometer heading snaps (merged into
            # odom pose theta) teleport the map pose every tick — the source
            # of correct↔wrong pose flapping.
            prev = self._last_odom_pose
            prev_at = self._last_odom_pose_at
            self._last_odom_pose = odom.pose
            self._last_odom_pose_at = now
            self._last_odom_time = now
            # Rate from the sample before this one: the turn already underway.
            # This sample's own vtheta can arrive with a heading snap.
            prev_yaw_rate = abs(float(self._last_odom_twist[2]))
            self._last_odom_twist = (odom.vx, odom.vy, odom.vtheta)
            self._imu_forward_m_s = 0.0
            self._imu_ax_bias = None
            self._imu_yaw_bias = None
            if prev is None:
                if odom.heading_rad is not None:
                    self._last_odom_heading = odom.heading_rad
                elif odom.pose is not None:
                    self._last_odom_heading = odom.pose.theta
                return self._pose
            delta = conv.compose_poses(conv.invert_pose(prev), odom.pose)
            # Prefer dedicated heading_sensor deltas for yaw so velocity-only
            # wheel odom does not dead-reckon heading from gyro vtheta alone.
            if odom.heading_rad is not None:
                if self._last_odom_heading is not None:
                    heading_delta = conv.normalize_angle(
                        odom.heading_rad - self._last_odom_heading
                    )
                    if abs(heading_delta) <= math.radians(40.0):
                        delta = conv.Pose2D(delta.x, delta.y, heading_delta)
                self._last_odom_heading = odom.heading_rad
            else:
                self._last_odom_heading = odom.pose.theta
            # Gate implausible jumps (heading snap, odom reset). Translation
            # scales with the gap because the reader zero-order-holds the last
            # twist across it. Yaw does not: the most a missed sample can
            # honestly contain is the turn already underway, and only for as
            # long as that hold lasts. A parked cart cannot absorb a
            # magnetometer flip just because the read was late.
            gap_s = (now - prev_at) if prev_at is not None else 0.0
            max_xy = max(0.5, 0.6 * gap_s)
            yaw_hold_s = min(max(gap_s, 0.0), self._ODOM_YAW_HOLD_MAX_S)
            max_yaw = max(math.radians(40.0), prev_yaw_rate * yaw_hold_s)
            if (
                math.hypot(delta.x, delta.y) > max_xy
                or abs(delta.theta) > max_yaw
            ):
                self._log(
                    "builtin SLAM: rejected odom jump "
                    f"dx={delta.x:.2f} dy={delta.y:.2f} "
                    f"dth={math.degrees(delta.theta):.0f}deg"
                )
                return self._pose
            return conv.compose_poses(self._pose, delta)

        # Twist integration in the *map* frame. Never snap map yaw to an
        # absolute IMU/magnetometer heading — that frame is not map-aligned
        # and was yanking localization after every tick.
        dt = 0.0
        if self._last_odom_time is not None:
            dt = max(0.0, min(now - self._last_odom_time, 0.5))
        self._last_odom_time = now
        if dt <= 0.0:
            if odom.heading_rad is not None and self._last_odom_heading is None:
                self._last_odom_heading = odom.heading_rad
            self._last_odom_twist = (odom.vx, odom.vy, odom.vtheta)
            return self._pose

        dth = odom.vtheta * dt
        if odom.heading_rad is not None:
            if self._last_odom_heading is not None:
                heading_delta = conv.normalize_angle(
                    odom.heading_rad - self._last_odom_heading
                )
                # Reject magnetometer jumps; fall back to gyro integration.
                if abs(heading_delta) <= math.radians(40.0):
                    dth = heading_delta
            self._last_odom_heading = odom.heading_rad

        if (
            getattr(self._cfg, "imu_odom_mode", "") == IMU_ODOM_ACCEL_ONLY
            and abs(float(odom.vx)) <= 1e-3
            and abs(float(odom.vy)) <= 1e-3
        ):
            dth = self._imu_yaw_delta(dth, dt)

        if self._lidar_odom_active(odom):
            self._yaw_check_gyro += float(odom.vtheta) * dt
        c = math.cos(self._pose.theta)
        s = math.sin(self._pose.theta)
        if self._lidar_odom_active(odom) and self._lidar_odom_fresh(now):
            fwd, left = self._lidar_odom_twist(now, dt)
            if getattr(self._cfg, "base_velocity_convention", "viam") in BASE_VELOCITY_Y_FORWARD:
                self._last_odom_twist = (-left, fwd, odom.vtheta)
            else:
                self._last_odom_twist = (fwd, left, odom.vtheta)
            return conv.Pose2D(
                self._pose.x + (c * fwd - s * left) * dt,
                self._pose.y + (s * fwd + c * left) * dt,
                conv.normalize_angle(self._pose.theta + dth),
            )

        vx, vy = self._twist_with_accel_prior(odom, dt)
        self._last_odom_twist = (vx, vy, odom.vtheta)
        if getattr(self._cfg, "base_velocity_convention", "viam") in BASE_VELOCITY_Y_FORWARD:
            # Sensor-native Y-forward / X-right → map (theta=0 faces +X).
            dx = (c * vy + s * vx) * dt
            dy = (s * vy - c * vx) * dt
        else:
            dx = (c * vx - s * vy) * dt
            dy = (s * vx + c * vy) * dt
        return conv.Pose2D(
            self._pose.x + dx,
            self._pose.y + dy,
            conv.normalize_angle(self._pose.theta + dth),
        )

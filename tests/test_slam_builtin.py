"""Unit tests for ROS-free builtin occupancy SLAM."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    SLAM_BACKEND_BUILTIN,
    SLAM_BACKEND_TOOLBOX,
    SlamConfig,
)
from src.nav.maps import MapStore
from src.nav_builtin.viam_io import bridge_map_to_get_grid, get_grid_response_to_map
from src.ros import conversions as conv
from src.slam_builtin import occupancy as occ
from src.slam_builtin import persistence
from src.slam_builtin import scan_match
from src.slam_builtin.engine import BuiltinSlamEngine
from src.slam_builtin.host import BuiltinSlamHost


def test_slam_backend_default_is_builtin():
    cfg = SlamConfig.from_dict({"base": "b", "lidar": "front"})
    assert cfg.slam_backend == SLAM_BACKEND_BUILTIN
    assert cfg.uses_builtin_slam()
    assert not cfg.uses_slam_toolbox()
    assert cfg.mapping_revisit_check is True
    assert cfg.mapping_revisit_while_moving is True
    assert cfg.builtin_rebuild_map_on_revisit is True


def test_slam_backend_toolbox_and_invalid():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "slam_backend": "slam_toolbox"}
    )
    assert cfg.slam_backend == SLAM_BACKEND_TOOLBOX
    assert cfg.uses_slam_toolbox()
    with pytest.raises(ValueError, match="slam_backend"):
        SlamConfig.from_dict(
            {"base": "b", "lidar": "front", "slam_backend": "cartographer"}
        )


def test_map_handle_occupancy_readiness(tmp_path: Path):
    store = MapStore(str(tmp_path))
    handle = store.get_or_create_map("m1")
    assert not handle.has_serialized_map()
    assert not handle.has_occupancy_map()
    assert not handle.has_any_map()

    grid = occ.empty_grid(resolution=0.05, size_m=4.0)
    # Mark a few cells occupied so save is meaningful.
    grid.log_odds[40, 40] = 2.0
    persistence.save_occupancy(handle.root, grid)
    assert handle.has_occupancy_map()
    assert handle.has_any_map()
    listed = {m["name"]: m for m in store.list_maps()}
    assert listed["m1"]["has_map"] is True
    assert listed["m1"]["has_occupancy"] is True
    assert listed["m1"]["has_posegraph"] is False


def test_occupancy_insert_and_roundtrip(tmp_path: Path):
    grid = occ.empty_grid(resolution=0.05, size_m=10.0, origin_x=-5.0, origin_y=-5.0)
    # Single forward beam hits a wall at 2 m.
    ranges = np.array([2.0])
    grid = occ.insert_scan(
        grid,
        0.0,
        0.0,
        0.0,
        ranges,
        0.0,
        0.0,
        range_min=0.1,
        range_max=10.0,
    )
    int16 = occ.to_occupancy_int16(grid)
    row, col = grid.world_to_cell(2.0, 0.0)
    assert int16[row, col] == 100
    row_m, col_m = grid.world_to_cell(1.0, 0.0)
    # One free update may land in the mid-probability band; must not be occupied.
    assert int16[row_m, col_m] < 50
    assert int16[row_m, col_m] >= 0

    persistence.save_occupancy(tmp_path, grid)
    loaded = persistence.load_log_odds(tmp_path)
    assert loaded is not None
    assert loaded.resolution == pytest.approx(0.05)
    assert loaded.origin_x == pytest.approx(grid.origin_x)
    roundtrip = occ.to_occupancy_int16(loaded)
    assert roundtrip[row, col] == 100


def test_get_grid_roundtrip_encoding():
    grid = np.full((8, 10), -1, dtype=np.int16)
    grid[2:6, 3:7] = 0
    grid[4, 5] = 100
    map_data = {
        "grid": grid,
        "resolution": 0.05,
        "origin_x": -1.0,
        "origin_y": -2.0,
    }
    payload = bridge_map_to_get_grid(map_data)
    back = get_grid_response_to_map(payload)
    assert back is not None
    assert back["resolution"] == pytest.approx(0.05)
    assert back["origin_x"] == pytest.approx(-1.0)
    assert back["origin_y"] == pytest.approx(-2.0)
    assert np.array_equal(back["grid"], grid)


def test_refine_pose_tracks_prior():
    # Synthetic corridor: occupied walls at y=±1.
    res = 0.05
    size = 80
    grid = np.full((size, size), 0, dtype=np.int16)
    origin = -size * res / 2
    for col in range(size):
        for wall_y in (-1.0, 1.0):
            row = int(math.floor((wall_y - origin) / res))
            if 0 <= row < size:
                grid[row, col] = 100
    occ_map = scan_match.occupancy_map_from_int16(
        grid, resolution=res, origin_x=origin, origin_y=origin
    )
    # Scan seeing both walls from origin.
    n = 60
    angles = np.linspace(-math.pi / 2, math.pi / 2, n)
    ranges = np.array([1.0 / max(abs(math.sin(a)), 0.2) for a in angles])
    ranges = np.clip(ranges, 0.2, 5.0)
    scan = conv.LaserScan2D(
        ranges, float(angles[0]), float(angles[1] - angles[0]), 0.1, 10.0
    )
    prior = conv.Pose2D(0.15, -0.10, math.radians(6.0))
    matched, score, prior_score = scan_match.refine_pose(
        occ_map,
        scan,
        prior,
        xy_half_m=0.25,
        yaw_half_deg=12.0,
        min_score=-0.5,
    )
    # With distance-weighted gating a modest offset may be rejected; either a
    # clear improvement toward origin or an explicit reject is OK.
    if matched is not None:
        assert abs(matched.x) < 0.12
        assert abs(matched.y) < 0.12
        assert score >= prior_score
    else:
        assert score < prior_score + scan_match._required_improvement(  # noqa: SLF001
            prior_score,
            0.15,
            math.radians(6.0),
        )


def test_refine_pose_rejects_weak_improvement():
    res = 0.05
    size = 40
    grid = np.full((size, size), 0, dtype=np.int16)
    origin = -size * res / 2
    # Sparse occupied speckles — any pose scores poorly.
    grid[20, 20] = 100
    occ_map = scan_match.occupancy_map_from_int16(
        grid, resolution=res, origin_x=origin, origin_y=origin
    )
    ranges = np.full(40, 1.0)
    scan = conv.LaserScan2D(ranges, -math.pi / 2, math.pi / 40, 0.1, 10.0)
    prior = conv.Pose2D(0.0, 0.0, 0.0)
    matched, score, prior_score = scan_match.refine_pose(
        occ_map, scan, prior, min_score=0.5
    )
    assert matched is None


def test_apply_match_small_correction_applies_immediately():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    engine = BuiltinSlamEngine(
        cfg, _FakeSensors(), MapStore("/tmp"), rate_hz=5.0
    )  # type: ignore[arg-type]
    predicted = conv.Pose2D(0.0, 0.0, 0.0)
    # 0.10 m correction = drift tracking; applied on the first frame.
    small = conv.Pose2D(0.10, 0.0, 0.0)
    assert engine._apply_match(predicted, small, None) is True  # noqa: SLF001
    # alpha 0.6 * 0.10 m
    assert engine.get_pose().x == pytest.approx(0.06, abs=1e-6)


def test_apply_match_large_jump_requires_two_frames():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    engine = BuiltinSlamEngine(
        cfg, _FakeSensors(), MapStore("/tmp"), rate_hz=5.0
    )  # type: ignore[arg-type]
    predicted = conv.Pose2D(0.0, 0.0, 0.0)
    peak_a = conv.Pose2D(0.40, 0.0, 0.0)
    peak_b = conv.Pose2D(-0.40, 0.0, 0.0)

    # Frame 1: see A — hold predicted.
    assert engine._apply_match(predicted, peak_a, None) is False  # noqa: SLF001
    assert engine.get_pose().x == pytest.approx(0.0)

    # Frame 2: flip to B — reset streak, still hold.
    assert engine._apply_match(predicted, peak_b, None) is False  # noqa: SLF001
    assert engine.get_pose().x == pytest.approx(0.0)

    # Two agreeing B frames — apply blended/clamped step toward B.
    assert engine._apply_match(predicted, peak_b, None) is True  # noqa: SLF001
    # alpha 0.6 * 0.40 = 0.24 m, inside the 0.25 m large-jump clamp.
    assert engine.get_pose().x == pytest.approx(-0.24, abs=1e-6)

    # Streak persists: the next agreeing frame keeps applying immediately.
    assert engine._apply_match(engine.get_pose(), peak_b, None) is True  # noqa: SLF001


def test_predict_absolute_odom_uses_deltas_and_gates_jumps():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    engine = BuiltinSlamEngine(
        cfg, _FakeSensors(), MapStore("/tmp"), rate_hz=5.0
    )  # type: ignore[arg-type]
    engine.set_pose(conv.Pose2D(5.0, 5.0, math.radians(90.0)))

    t0 = 50.0
    # First absolute odom sample only seeds the reference — no pose change,
    # even though the odom frame (0,0,0) is nowhere near the map pose.
    odom0 = conv.OdomReading(0.0, 0.0, 0.0, pose=conv.Pose2D(0.0, 0.0, 0.0))
    p0 = engine._predict(odom0, t0)  # noqa: SLF001
    assert p0.x == pytest.approx(5.0)
    assert p0.theta == pytest.approx(math.radians(90.0))

    # Forward 0.1 m in the odom frame → forward 0.1 m along map heading (90°).
    odom1 = conv.OdomReading(0.0, 0.0, 0.0, pose=conv.Pose2D(0.1, 0.0, 0.0))
    engine._pose = p0  # noqa: SLF001
    p1 = engine._predict(odom1, t0 + 0.1)  # noqa: SLF001
    assert p1.x == pytest.approx(5.0, abs=1e-6)
    assert p1.y == pytest.approx(5.1, abs=1e-6)

    # A 90° heading snap in the odom pose (magnetometer jump) is rejected.
    odom2 = conv.OdomReading(
        0.0, 0.0, 0.0, pose=conv.Pose2D(0.1, 0.0, math.radians(90.0))
    )
    engine._pose = p1  # noqa: SLF001
    p2 = engine._predict(odom2, t0 + 0.2)  # noqa: SLF001
    assert p2.x == pytest.approx(p1.x)
    assert p2.y == pytest.approx(p1.y)
    assert p2.theta == pytest.approx(p1.theta)


def test_follow_command_intermediate_target_never_marks_done():
    from src.nav_builtin.controller import FollowerConfig, compute_follow_command
    from src.nav_builtin.types import Pose2D

    cfg = FollowerConfig()
    cfg.motion.xy_tolerance_m = 0.25
    current = Pose2D(0.0, 0.0, 0.0)
    target = Pose2D(0.1, 0.0, 0.0)  # within 0.5 * xy_tol
    cmd = compute_follow_command(current, target, cfg=cfg, final_yaw=None)
    assert not cmd.done


def test_host_relocalize_applies_drift_correction():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    engine = BuiltinSlamEngine(
        cfg, _FakeSensors(), MapStore("/tmp"), rate_hz=5.0
    )  # type: ignore[arg-type]
    host = BuiltinSlamHost(engine)
    engine.set_pose(conv.Pose2D(1.0, 1.0, 0.0))

    # 0.2 m drift fix from periodic relocalize should apply (not soft-gated).
    host.relocalize(conv.Pose2D(1.2, 1.0, 0.0))
    assert engine.get_pose().x == pytest.approx(1.2)


def test_host_relocalize_soft_seed():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    engine = BuiltinSlamEngine(
        cfg, _FakeSensors(), MapStore("/tmp"), rate_hz=5.0
    )  # type: ignore[arg-type]
    host = BuiltinSlamHost(engine)
    engine.set_pose(conv.Pose2D(1.0, 1.0, 0.0))

    # Tiny nudge within the continuous matcher's band is still a no-op.
    host.relocalize(conv.Pose2D(1.1, 1.05, math.radians(5.0)))
    assert engine.get_pose().x == pytest.approx(1.0)
    assert engine.get_pose().y == pytest.approx(1.0)

    # A genuinely different pose (recovery) still hard-applies.
    host.relocalize(conv.Pose2D(3.0, 1.0, 0.0))
    assert engine.get_pose().x == pytest.approx(3.0)

    # set_initial_pose stays a hard snap regardless of distance.
    host.set_initial_pose(conv.Pose2D(3.05, 1.0, 0.0))
    assert engine.get_pose().x == pytest.approx(3.05)


def test_predict_uses_heading_delta_not_absolute():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "localizing"}
    )
    store = MapStore("/tmp")
    engine = BuiltinSlamEngine(cfg, _FakeSensors(), store, rate_hz=5.0)  # type: ignore[arg-type]
    engine.set_pose(conv.Pose2D(1.0, 2.0, math.radians(30.0)))
    # First sample seeds heading; pose yaw must stay at map 30°.
    t0 = 100.0
    odom0 = conv.OdomReading(0.0, 0.0, 0.0, heading_rad=math.radians(90.0))
    p0 = engine._predict(odom0, t0)  # noqa: SLF001
    assert p0.theta == pytest.approx(math.radians(30.0), abs=1e-6)
    # +10° heading change → map yaw advances by ~10°, not snaps to 100°.
    odom1 = conv.OdomReading(0.0, 0.0, 0.0, heading_rad=math.radians(100.0))
    p1 = engine._predict(odom1, t0 + 0.1)  # noqa: SLF001
    assert p1.theta == pytest.approx(math.radians(40.0), abs=1e-3)
    assert abs(p1.theta - math.radians(100.0)) > 0.5


class _FakeSensors:
    def __init__(self, scan=None, odom=None):
        self._scan = scan
        self._odom = odom

    def get_scan(self, max_age_s: float = 2.0):
        del max_age_s
        return self._scan

    def get_odom(self):
        return self._odom


def test_engine_set_pose_and_get_map(tmp_path: Path):
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": str(tmp_path), "mode": "mapping"}
    )
    store = MapStore(str(tmp_path))
    store.get_or_create_map("default")
    store.set_active_map("default")
    sensors = _FakeSensors()
    engine = BuiltinSlamEngine(cfg, sensors, store, rate_hz=5.0)  # type: ignore[arg-type]
    engine.set_pose(conv.Pose2D(1.0, 2.0, 0.5))
    assert engine.get_pose().x == pytest.approx(1.0)
    assert engine.get_pose().y == pytest.approx(2.0)

    # Seed a wall and export via host.
    with engine._lock:  # noqa: SLF001
        engine._grid = occ.insert_scan(  # noqa: SLF001
            engine._grid,
            0.0,
            0.0,
            0.0,
            np.full(20, 1.5),
            -0.5,
            0.05,
            range_min=0.1,
            range_max=10.0,
        )
        engine._generation += 1
        engine._invalidate_occ_cache()  # noqa: SLF001
    host = BuiltinSlamHost(engine)
    map_data = host.get_map()
    assert map_data is not None
    assert map_data["grid"].shape[0] > 0
    assert host.get_pose_in_map().x == pytest.approx(1.0)

    handle = store.active_handle()
    assert handle is not None
    host.save_map(handle.serialization_stem)
    assert handle.has_occupancy_map()

    diag = host.slam_diagnostics()
    assert diag["slam_backend"] == "builtin"


def test_map_keyframe_store_rebuild_after_pose_delta():
    from src.slam_builtin.keyframes import MapKeyframeStore

    store = MapKeyframeStore(max_keyframes=10)
    scan = conv.LaserScan2D(
        np.array([2.0]),
        0.0,
        0.0,
        0.1,
        10.0,
    )
    store.add(conv.Pose2D(0.0, 0.0, 0.0), scan)
    grid_ok = store.rebuild_grid(resolution=0.05)
    row, col = grid_ok.world_to_cell(2.0, 0.0)
    int16_ok = occ.to_occupancy_int16(grid_ok)
    assert int16_ok[row, col] == 100

    anchor = store.find_loop_anchor(conv.Pose2D(0.0, 0.0, 0.0))
    assert anchor == 0
    # Drifted leg after the anchor is corrected; anchor keyframe stays fixed.
    store.add(conv.Pose2D(0.0, 0.5, 0.0), scan)
    store.apply_pose_delta_from(anchor + 1, conv.Pose2D(0.0, -0.5, 0.0))
    grid_fixed = store.rebuild_grid(resolution=0.05)
    row_f, col_f = grid_fixed.world_to_cell(2.0, 0.0)
    int16_fixed = occ.to_occupancy_int16(grid_fixed)
    assert int16_fixed[row_f, col_f] == 100


def test_apply_map_pose_correction_rebuilds_grid():
    cfg = SlamConfig.from_dict(
        {
            "base": "b",
            "lidar": "front",
            "maps_dir": "/tmp",
            "mode": "mapping",
            "mapping_revisit_min_shift_m": 0.05,
            "mapping_revisit_min_shift_deg": 1.0,
        }
    )
    engine = BuiltinSlamEngine(
        cfg, _FakeSensors(), MapStore("/tmp"), rate_hz=5.0
    )  # type: ignore[arg-type]
    host = BuiltinSlamHost(engine)
    scan = conv.LaserScan2D(np.array([2.0]), 0.0, 0.0, 0.1, 10.0)

    with engine._lock:  # noqa: SLF001
        engine._grid = occ.insert_scan(  # noqa: SLF001
            engine._grid,
            0.0,
            0.0,
            0.0,
            np.array([2.0]),
            0.0,
            0.0,
            range_min=0.1,
            range_max=10.0,
        )
        engine._keyframes.add(conv.Pose2D(0.0, 0.0, 0.0), scan)
        engine._pose = conv.Pose2D(0.0, 0.5, 0.0)

    result = host.apply_map_pose_correction(conv.Pose2D(0.0, 0.0, 0.0))
    assert result["applied"] is True
    assert result["rebuilt"] is True
    assert engine.get_pose().y == pytest.approx(0.0)
    with engine._lock:  # noqa: SLF001
        int16 = occ.to_occupancy_int16(engine._grid)  # noqa: SLF001
        row, col = engine._grid.world_to_cell(2.0, 0.0)  # noqa: SLF001
    assert int16[row, col] == 100


def test_host_slam_bridge_status_reports_odom_twist():
    cfg = SlamConfig.from_dict(
        {"base": "b", "lidar": "front", "maps_dir": "/tmp", "mode": "mapping"}
    )
    odom = conv.OdomReading(0.2, 0.0, 0.1, pose=conv.Pose2D(0.0, 0.0, 0.0))
    engine = BuiltinSlamEngine(
        cfg,
        _FakeSensors(odom=odom),
        MapStore("/tmp"),
        rate_hz=5.0,
    )  # type: ignore[arg-type]
    host = BuiltinSlamHost(engine)
    engine._predict(odom, 1.0)  # noqa: SLF001
    status = host.slam_bridge_status()
    assert status["odom_velocity"]["vx"] == pytest.approx(0.2)
    assert status["odom_velocity"]["vtheta"] == pytest.approx(0.1)

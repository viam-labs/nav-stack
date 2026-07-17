import math
from pathlib import Path

import numpy as np
import pytest

from src.nav.global_localize import (
    OccupancyMap,
    global_localize_scan,
    load_occupancy_from_map_dir,
    pgm_to_occupancy_grid,
)
from src.ros import conversions as conv


def _asymmetric_map() -> OccupancyMap:
    """Rectangular room with a single interior pillar (breaks symmetry)."""
    grid = np.full((30, 30), -1, dtype=np.int16)
    grid[3:27, 3] = 100
    grid[3:27, 26] = 100
    grid[3, 3:27] = 100
    grid[26, 3:27] = 100
    grid[4:26, 4:26] = 0
    grid[12:14, 18:20] = 100  # pillar visible from most poses in the room
    return OccupancyMap(grid=grid, resolution=0.5, origin_x=0.0, origin_y=0.0)


def _raycast_scan(occ_map: OccupancyMap, pose: conv.Pose2D, num_bins: int = 360) -> conv.LaserScan2D:
    """Raycast the occupancy map to synthesize a scan at ``pose``."""
    angle_min = -math.pi
    angle_increment = 2 * math.pi / num_bins
    ranges = np.full(num_bins, np.inf, dtype=float)
    max_range = 30.0
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    for i in range(num_bins):
        angle = angle_min + i * angle_increment
        dx = math.cos(angle)
        dy = math.sin(angle)
        wx_dir = c * dx - s * dy
        wy_dir = s * dx + c * dy
        for step in np.arange(0.25, max_range, occ_map.resolution):
            wx = pose.x + wx_dir * step
            wy = pose.y + wy_dir * step
            row, col = occ_map.world_to_cell(wx, wy)
            if row < 0 or col < 0 or row >= occ_map.height or col >= occ_map.width:
                ranges[i] = step
                break
            val = int(occ_map.grid[row, col])
            if val >= 65:
                ranges[i] = step
                break
    return conv.LaserScan2D(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.1,
        range_max=max_range,
    )


def test_global_localize_finds_pose_near_hint():
    occ_map = _asymmetric_map()
    true_pose = conv.Pose2D(6.0, 8.0, 0.2)
    scan = _raycast_scan(occ_map, true_pose)
    wrong_hint = conv.Pose2D(7.5, 9.5, math.radians(30.0))

    result = global_localize_scan(
        occ_map,
        scan,
        hint=wrong_hint,
        search_radius_m=4.0,
        coarse_position_step_m=0.5,
        coarse_yaw_step_deg=12.0,
        local_yaw_window_deg=120.0,
    )

    assert result.score > 0.35
    assert result.scan_points_used <= 240
    assert result.in_map_points >= 40
    assert result.ray_score >= 0.0
    assert result.ray_mae_m >= 0.0
    assert abs(result.pose.x - true_pose.x) <= 1.0
    assert abs(result.pose.y - true_pose.y) <= 1.0
    assert abs(result.pose.theta - true_pose.theta) <= math.radians(25.0)


def test_global_localize_full_map_without_hint():
    occ_map = _asymmetric_map()
    true_pose = conv.Pose2D(6.0, 8.0, 0.2)
    scan = _raycast_scan(occ_map, true_pose)

    result = global_localize_scan(
        occ_map,
        scan,
        hint=None,
        full_map=True,
        coarse_position_step_m=0.5,
        coarse_yaw_step_deg=12.0,
    )

    assert result.score > 0.35
    assert result.scan_points_used <= 240
    assert result.ray_score >= 0.0
    assert abs(result.pose.x - true_pose.x) <= 1.0
    assert abs(result.pose.y - true_pose.y) <= 1.0


def test_global_localize_rejects_off_map_candidates():
    occ_map = _asymmetric_map()
    true_pose = conv.Pose2D(6.0, 8.0, 0.2)
    scan = _raycast_scan(occ_map, true_pose)
    with pytest.raises(RuntimeError, match="no valid candidate"):
        global_localize_scan(
            occ_map,
            scan,
            hint=conv.Pose2D(-20.0, -20.0, 0.0),
            search_radius_m=1.0,
            coarse_position_step_m=0.5,
        )


def test_pgm_to_occupancy_grid_nav2_values():
    pgm = np.array([[254, 0], [205, 254]], dtype=np.uint8)
    grid = pgm_to_occupancy_grid(pgm)
    assert grid[0, 0] == 0
    assert grid[0, 1] == 100
    assert grid[1, 0] == -1


def test_load_occupancy_from_map_dir(tmp_path: Path):
    pgm = np.full((4, 4), 254, dtype=np.uint8)
    pgm[1, 1] = 0
    pgm_path = tmp_path / "map.pgm"
    header = b"P5\n4 4\n255\n"
    pgm_path.write_bytes(header + pgm.tobytes())
    (tmp_path / "map.yaml").write_text(
        "\n".join(
            [
                "image: map.pgm",
                "resolution: 0.05",
                "origin: [-1.0, -2.0, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
            ]
        ),
        encoding="utf-8",
    )

    occ = load_occupancy_from_map_dir(tmp_path)
    assert occ is not None
    assert occ.resolution == 0.05
    assert occ.origin_x == -1.0
    occ_cells = np.argwhere(occ.grid == 100)
    assert occ_cells.shape[0] == 1
    # map_server image rows are top->bottom; loader flips into OccupancyGrid layout.
    assert tuple(occ_cells[0]) == (2, 1)


def test_global_localize_rejects_sparse_scan():
    occ_map = _asymmetric_map()
    empty = conv.LaserScan2D(
        ranges=np.full(10, np.inf),
        angle_min=-math.pi,
        angle_increment=0.1,
    )
    with pytest.raises(ValueError, match="too few"):
        global_localize_scan(occ_map, empty, hint=conv.Pose2D(0, 0, 0))


def _corridor_map() -> OccupancyMap:
    """Long N-S corridor — classic 180° scan ambiguity."""
    grid = np.full((40, 20), -1, dtype=np.int16)
    grid[2:38, 4] = 100
    grid[2:38, 15] = 100
    grid[2:38, 5:15] = 0
    return OccupancyMap(grid=grid, resolution=0.5, origin_x=0.0, origin_y=0.0)


def test_choose_yaw_or_flip_prefers_reference_on_near_tie():
    from src.nav.global_localize import choose_yaw_or_flip

    occ = _corridor_map()
    true = conv.Pose2D(5.0, 10.0, math.pi / 2)  # facing +Y along corridor
    scan = _raycast_scan(occ, true)
    # Pretend the matcher returned the opposite heading at the right XY.
    wrong = conv.Pose2D(true.x, true.y, true.theta + math.pi)
    choice = choose_yaw_or_flip(
        occ, scan, wrong, reference_theta=true.theta + 0.1
    )
    assert choice.flipped is True
    assert abs(_wrap(choice.pose.theta - true.theta)) < math.radians(5)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

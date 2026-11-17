import math
from pathlib import Path

import numpy as np
import pytest

from src.geom import conversions as conv
from src.nav.localize.global_localize import (
    OccupancyMap,
    global_localize_scan,
    load_occupancy_from_map_dir,
    pgm_to_occupancy_grid,
)


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


def _raycast_scan(
    occ_map: OccupancyMap, pose: conv.Pose2D, num_bins: int = 360
) -> conv.LaserScan2D:
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
    assert hasattr(result, "ambiguous")
    assert hasattr(result, "second_best_score")


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
    from src.nav.localize.global_localize import choose_yaw_or_flip

    occ = _corridor_map()
    true = conv.Pose2D(5.0, 10.0, math.pi / 2)  # facing +Y along corridor
    scan = _raycast_scan(occ, true)
    # Pretend the matcher returned the opposite heading at the right XY.
    wrong = conv.Pose2D(true.x, true.y, true.theta + math.pi)
    choice = choose_yaw_or_flip(occ, scan, wrong, reference_theta=true.theta + 0.1)
    assert choice.flipped is True
    assert abs(_wrap(choice.pose.theta - true.theta)) < math.radians(5)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# --- Search grid and ambiguity -------------------------------------------------


def _office_map(twin: bool, pillar: bool = False) -> OccupancyMap:
    """Two 5 m x 6 m rooms side by side at 5 cm cells.

    ``twin`` makes the right room an exact copy of the left one; otherwise it
    has a different alcove. ``pillar`` adds a post to the left room only.
    """
    grid = np.full((160, 260), -1, dtype=np.int16)
    for c0 in (20, 130):
        grid[20:141, c0 : c0 + 101] = 0
        grid[20, c0 : c0 + 101] = 100
        grid[140, c0 : c0 + 101] = 100
        grid[20:141, c0] = 100
        grid[20:141, c0 + 100] = 100
        grid[100:108, c0 + 60 : c0 + 100] = 100  # a counter along the right wall
    if not twin:
        grid[40:80, 150:158] = 100  # right room only
    if pillar:
        grid[60:66, 50:56] = 100  # left room only
    return OccupancyMap(grid=grid, resolution=0.05, origin_x=0.0, origin_y=0.0)


def test_batch_scorer_gives_the_same_scores_as_the_reference():
    from src.nav.localize.global_localize import (
        _BatchScorer,
        _inflate_occupied,
        _lattice_cells,
        _score_pose,
        scan_endpoints_base_link,
    )

    occ = _office_map(twin=False)
    truth = conv.Pose2D(3.37, 3.12, 0.4)
    xy = scan_endpoints_base_link(_raycast_scan(occ, truth, num_bins=180))
    lookup = _inflate_occupied(occ.grid >= 65, 2)
    scorer = _BatchScorer(occ, xy, lookup, min_in_map_points=40, min_in_map_ratio=0.35)
    rng = np.random.default_rng(0)

    xs, ys = rng.uniform(-1.0, 14.0, 200), rng.uniform(-1.0, 9.0, 200)
    score, inside, hit_rate, coverage = scorer.score_xy(xs, ys, 0.9)
    for i in range(200):
        ref = _score_pose(
            occ, xy, conv.Pose2D(xs[i], ys[i], 0.9), occupied_lookup=lookup
        )
        assert (score[i] == ref.score) or (
            not np.isfinite(score[i]) and not np.isfinite(ref.score)
        )
        assert inside[i] == ref.in_map_points
        assert hit_rate[i] == pytest.approx(ref.hit_rate) and coverage[
            i
        ] == pytest.approx(ref.coverage)

    rows, cols, frac_x, frac_y = _lattice_cells(occ, truth.x, truth.y, 3, radius_m=2.0)
    assert rows.size > 300
    score, _inside, _hit, _cov = scorer.score_lattice(rows, cols, frac_x, frac_y, -0.3)
    for i in range(0, rows.size, 7):
        pose = conv.Pose2D((cols[i] + frac_x) * 0.05, (rows[i] + frac_y) * 0.05, -0.3)
        ref = _score_pose(occ, xy, pose, occupied_lookup=lookup).score
        assert (score[i] == ref) or (not np.isfinite(score[i]) and not np.isfinite(ref))


def test_default_grid_is_sized_from_the_hit_tolerance():
    from src.nav.localize.global_localize import tolerance_grid_steps

    def ring(r):
        return np.column_stack(
            [r * np.cos(np.arange(90.0)), r * np.sin(np.arange(90.0))]
        )

    step_m, yaw_deg = tolerance_grid_steps(ring(3.0), 0.05, 2)
    assert step_m == pytest.approx(0.15)
    assert yaw_deg == pytest.approx(math.degrees(2 * 0.125 / 3.0))  # 4.8°
    # Near walls allow a wider heading step, far ones a narrower; both are capped.
    assert tolerance_grid_steps(ring(0.5), 0.05, 2)[1] == 12.0
    assert tolerance_grid_steps(ring(20.0), 0.05, 2)[1] == 2.0
    assert tolerance_grid_steps(ring(3.0), 0.05, 0)[0] == pytest.approx(0.05)


def test_whole_map_search_finds_a_pose_between_grid_samples():
    occ = _office_map(twin=False)
    # 12 cm and 7° off the nearest sample of the old 35 cm / 10° whole-map grid.
    truth = conv.Pose2D(3.37, 3.12, math.radians(27.0))
    result = global_localize_scan(occ, _raycast_scan(occ, truth), full_map=True)
    assert math.hypot(result.pose.x - truth.x, result.pose.y - truth.y) < 0.06
    assert abs(_wrap(result.pose.theta - truth.theta)) < math.radians(1.5)
    assert result.score > 0.85
    assert result.ambiguous is False


def test_ambiguous_means_another_place_fits_not_a_neighbouring_sample():
    truth = conv.Pose2D(3.37, 3.12, math.radians(27.0))

    twins = _office_map(twin=True)
    result = global_localize_scan(twins, _raycast_scan(twins, truth), full_map=True)
    # The same pose in either room explains the scan equally well.
    in_left = math.hypot(result.pose.x - truth.x, result.pose.y - truth.y) < 0.1
    in_right = (
        math.hypot(result.pose.x - (truth.x + 5.5), result.pose.y - truth.y) < 0.1
    )
    assert in_left or in_right
    assert result.ambiguous is True
    assert math.isfinite(result.second_best_score)

    # A post in the left room is enough to tell them apart.
    marked = _office_map(twin=True, pillar=True)
    result = global_localize_scan(marked, _raycast_scan(marked, truth), full_map=True)
    assert math.hypot(result.pose.x - truth.x, result.pose.y - truth.y) < 0.06
    assert result.ambiguous is False


def test_a_local_search_with_no_other_place_in_reach_is_not_ambiguous():
    occ = _office_map(twin=True)
    truth = conv.Pose2D(3.37, 3.12, math.radians(27.0))
    hint = conv.Pose2D(3.6, 2.9, math.radians(40.0))
    result = global_localize_scan(
        occ,
        _raycast_scan(occ, truth),
        hint=hint,
        search_radius_m=0.6,
        local_yaw_window_deg=60.0,
    )
    assert math.hypot(result.pose.x - truth.x, result.pose.y - truth.y) < 0.06
    assert result.second_best_score == float("-inf")
    assert result.ambiguous is False

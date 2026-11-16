"""Lidar scan matching against a saved nav-stack occupancy map.

Scores candidate map-frame poses by aligning live scan endpoints with occupied
cells. Used for global localization without MiR map pose or other external seeds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import numpy as np

from ...geom import conversions as conv


@dataclass(frozen=True)
class OccupancyMap:
    """Nav2-style occupancy grid (row-major, shape height x width)."""

    grid: np.ndarray  # int16: -1 unknown, 0 free, 100 occupied
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        col = int(math.floor((x_m - self.origin_x) / self.resolution))
        row = int(math.floor((y_m - self.origin_y) / self.resolution))
        return row, col

    def known_mask(self) -> np.ndarray:
        return self.grid >= 0


@dataclass(frozen=True)
class GlobalLocalizeResult:
    pose: conv.Pose2D
    score: float
    candidates_evaluated: int
    scan_points_used: int
    in_map_points: int
    hit_rate: float
    ray_score: float
    ray_mae_m: float
    # Combined (endpoint + ray) score of the best different place (over 1 m
    # away or facing over 45° differently); -inf when there is none. Used to
    # refuse applying a pose that a second place explains nearly as well.
    second_best_score: float = float("-inf")
    ambiguous: bool = False
    # The winner's own combined (endpoint + ray) score: the number to compare
    # with ``second_best_score``. ``score`` is the endpoint part alone.
    combined_score: float = float("-inf")
    # Fraction of the scan the map could judge at the winning pose.
    coverage: float = 0.0


@dataclass(frozen=True)
class _PoseScore:
    score: float
    in_map_points: int
    hit_rate: float
    # Fraction of the scan the map could judge (known cells).
    coverage: float = 0.0


# Below this judged fraction a whole-map candidate is penalised, so a pose
# that parks most of the scan in unknown or off-grid cells cannot outscore
# one that is actually inside the mapped area. Above it the score is the
# same fraction-explained measure everywhere on the map.
_COVERAGE_FLOOR = 0.6
_COVERAGE_PENALTY = 1.2


@dataclass(frozen=True)
class _ScoredPose:
    pose: conv.Pose2D
    endpoint_eval: _PoseScore


def load_occupancy_from_map_dict(map_data: Mapping) -> OccupancyMap:
    """Build an ``OccupancyMap`` from bridge ``get_map()`` output."""
    grid = np.asarray(map_data["grid"], dtype=np.int16)
    if grid.ndim != 2:
        raise ValueError("map grid must be 2D")
    return OccupancyMap(
        grid=grid,
        resolution=float(map_data["resolution"]),
        origin_x=float(map_data["origin_x"]),
        origin_y=float(map_data["origin_y"]),
    )


def _parse_simple_yaml(text: str) -> dict:
    """Parse the small Nav2 map.yaml files without requiring PyYAML."""
    data: dict = {}
    origin_re = re.compile(
        r"^origin\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]\s*$"
    )
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, raw_val = stripped.split(":", 1)
        key = key.strip()
        val = raw_val.strip()
        origin_match = origin_re.match(stripped)
        if origin_match:
            data["origin"] = [float(origin_match.group(i)) for i in range(1, 4)]
            continue
        if val.startswith("[") and val.endswith("]"):
            continue
        if val.lower() in {"true", "false"}:
            data[key] = val.lower() == "true"
        else:
            try:
                if "." in val or "e" in val.lower():
                    data[key] = float(val)
                else:
                    data[key] = int(val)
            except ValueError:
                data[key] = val.strip("'\"")
    return data


def _read_pgm(path: Path) -> np.ndarray:
    """Load a binary P5 PGM into a ``(height, width)`` uint8 array."""
    raw = path.read_bytes()
    if not raw.startswith(b"P5"):
        raise ValueError(f"unsupported PGM format in {path}")

    tokens: List[str] = []
    i = 0
    while i < len(raw) and len(tokens) < 4:
        if raw[i : i + 1] in b" \t\r\n":
            i += 1
            continue
        if raw[i : i + 1] == b"#":
            while i < len(raw) and raw[i : i + 1] != b"\n":
                i += 1
            continue
        start = i
        while i < len(raw) and raw[i : i + 1] not in b" \t\r\n":
            i += 1
        tokens.append(raw[start:i].decode("ascii"))

    if len(tokens) < 4 or tokens[0] != "P5":
        raise ValueError(f"invalid PGM header in {path}")
    width = int(tokens[1])
    height = int(tokens[2])
    maxval = int(tokens[3])
    # Skip whitespace after maxval.
    while i < len(raw) and raw[i : i + 1] in b" \t\r\n":
        i += 1
    body = raw[i : i + width * height]
    if len(body) != width * height:
        raise ValueError(f"truncated PGM payload in {path}")
    arr = np.frombuffer(body, dtype=np.uint8)
    if maxval != 255:
        arr = (arr.astype(np.float32) * (255.0 / maxval)).astype(np.uint8)
    return arr.reshape(height, width)


def pgm_to_occupancy_grid(
    pgm: np.ndarray,
    *,
    negate: int = 0,
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
) -> np.ndarray:
    """Convert a map_server PGM into ROS occupancy values."""
    pixels = pgm.astype(np.float32)
    if negate:
        occ_prob = pixels / 255.0
    else:
        occ_prob = 1.0 - (pixels / 255.0)
    grid = np.full(pixels.shape, -1, dtype=np.int16)
    grid[occ_prob > occupied_thresh] = 100
    grid[occ_prob < free_thresh] = 0
    return grid


def load_occupancy_from_map_dir(map_dir: Path) -> Optional[OccupancyMap]:
    """Load ``map.yaml`` + ``map.pgm`` from a nav-stack map directory."""
    yaml_path = map_dir / "map.yaml"
    if not yaml_path.exists():
        return None
    meta = _parse_simple_yaml(yaml_path.read_text(encoding="utf-8"))
    image_name = str(meta.get("image") or "map.pgm")
    image_path = map_dir / image_name
    if not image_path.is_file():
        image_path = map_dir / "map.pgm"
    if not image_path.is_file():
        return None
    pgm = _read_pgm(image_path)
    # map_server stores image rows top->bottom while OccupancyGrid is indexed from
    # map origin (bottom->top); flip vertically to recover OccupancyGrid layout.
    pgm = np.flipud(pgm)
    grid = pgm_to_occupancy_grid(
        pgm,
        negate=int(meta.get("negate", 0)),
        occupied_thresh=float(meta.get("occupied_thresh", 0.65)),
        free_thresh=float(meta.get("free_thresh", 0.196)),
    )
    origin = meta.get("origin") or [0.0, 0.0, 0.0]
    return OccupancyMap(
        grid=grid,
        resolution=float(meta.get("resolution", 0.05)),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
    )


def scan_endpoints_base_link(scan: conv.LaserScan2D) -> np.ndarray:
    """Return valid scan endpoints as ``(N, 2)`` points in base_link."""
    pts = scan.to_points()
    if pts.size == 0:
        return np.empty((0, 2))
    return pts


def _score_pose(
    occ_map: OccupancyMap,
    scan_xy: np.ndarray,
    pose: conv.Pose2D,
    *,
    occupied_lookup: np.ndarray,
    min_in_map_points: int = 40,
    min_in_map_ratio: float = 0.35,
) -> _PoseScore:
    """Higher is better. Rewards scan endpoints near occupied cells."""
    if scan_xy.size == 0:
        return _PoseScore(float("-inf"), 0, 0.0)

    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    xs = scan_xy[:, 0]
    ys = scan_xy[:, 1]
    wx = pose.x + c * xs - s * ys
    wy = pose.y + s * xs + c * ys

    grid = occ_map.grid
    height, width = grid.shape
    res = occ_map.resolution
    origin_x = occ_map.origin_x
    origin_y = occ_map.origin_y
    cols = np.floor((wx - origin_x) / res).astype(np.int32)
    rows = np.floor((wy - origin_y) / res).astype(np.int32)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    in_map_points = int(np.count_nonzero(inside))
    total_points = int(scan_xy.shape[0])
    min_points = max(min_in_map_points, int(total_points * min_in_map_ratio))
    if in_map_points < min_points:
        return _PoseScore(float("-inf"), in_map_points, 0.0, 0.0)

    rows_in = rows[inside]
    cols_in = cols[inside]
    hit_mask = occupied_lookup[rows_in, cols_in]
    # Endpoints landing in known free space are a mismatch. Unknown cells
    # and beams past the grid say nothing and are left out of the fraction,
    # so the score reads the same next to unexplored space as in a finished
    # room (see ``scan_match._candidate_scores_with_coverage``).
    cell_vals = grid[rows_in, cols_in]
    free_mask = (cell_vals >= 0) & (cell_vals < 50)
    known = int(np.count_nonzero(hit_mask | free_mask))
    if known < min_points:
        return _PoseScore(float("-inf"), in_map_points, 0.0, known / total_points)
    hits = int(np.count_nonzero(hit_mask))
    hit_rate = hits / float(known)
    free_rate = int(np.count_nonzero(free_mask)) / float(known)
    coverage = known / float(total_points)

    score = hit_rate - 0.7 * free_rate
    # A whole-map candidate that hides most of the scan in unmapped cells
    # cannot win on the few beams it does show.
    if coverage < _COVERAGE_FLOOR:
        score -= _COVERAGE_PENALTY * (_COVERAGE_FLOOR - coverage)
    return _PoseScore(score, in_map_points, hit_rate, coverage)


def _inflate_occupied(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate occupied cells for tolerant endpoint matching."""
    if radius_cells <= 0:
        return occupied
    padded = np.pad(occupied, radius_cells, mode="constant", constant_values=False)
    inflated = np.zeros_like(occupied, dtype=bool)
    height, width = occupied.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            rs = radius_cells + dr
            cs = radius_cells + dc
            inflated |= padded[rs : rs + height, cs : cs + width]
    return inflated


def _pose_in_known_map(occ_map: OccupancyMap, pose: conv.Pose2D) -> bool:
    """Quick center-pose gate to avoid obviously invalid candidates."""
    row, col = occ_map.world_to_cell(pose.x, pose.y)
    if row < 0 or col < 0 or row >= occ_map.height or col >= occ_map.width:
        return False
    return int(occ_map.grid[row, col]) >= 0


def _sample_scan_beams(
    scan: conv.LaserScan2D, max_beams: int
) -> Tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(scan.ranges, dtype=float)
    if ranges.size == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    angles = scan.angle_min + np.arange(ranges.size, dtype=float) * scan.angle_increment
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan.range_min))
        & (ranges <= float(scan.range_max))
    )
    if not np.any(valid):
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    valid_idx = np.flatnonzero(valid)
    if max_beams > 0 and valid_idx.size > max_beams:
        pick = np.linspace(0, valid_idx.size - 1, max_beams, dtype=np.int32)
        valid_idx = valid_idx[pick]
    return angles[valid_idx], ranges[valid_idx]


def _raycast_distance(
    occ_map: OccupancyMap,
    x_m: float,
    y_m: float,
    heading_rad: float,
    *,
    max_range_m: float,
    step_m: float,
) -> float:
    step = max(step_m, occ_map.resolution * 0.5)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    dist = step
    while dist <= max_range_m:
        sx = x_m + cos_h * dist
        sy = y_m + sin_h * dist
        row, col = occ_map.world_to_cell(sx, sy)
        if row < 0 or col < 0 or row >= occ_map.height or col >= occ_map.width:
            return dist
        if int(occ_map.grid[row, col]) >= 65:
            return dist
        dist += step
    return max_range_m


def _ray_alignment(
    occ_map: OccupancyMap,
    pose: conv.Pose2D,
    beam_angles: np.ndarray,
    beam_ranges: np.ndarray,
    *,
    range_max_m: float,
    step_m: float,
) -> Tuple[float, float]:
    if beam_angles.size == 0:
        return 0.0, float("inf")
    errors: List[float] = []
    for angle, observed in zip(beam_angles, beam_ranges):
        predicted = _raycast_distance(
            occ_map,
            pose.x,
            pose.y,
            pose.theta + float(angle),
            max_range_m=range_max_m,
            step_m=step_m,
        )
        errors.append(min(4.0, abs(predicted - float(observed))))
    mae = float(np.mean(errors))
    # 0..1 quality (1 is perfect): 0m->1.0, 2m->0.0
    quality = max(0.0, 1.0 - (mae / 2.0))
    return quality, mae


def score_pose_with_rays(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    pose: conv.Pose2D,
    *,
    max_scan_points: int = 240,
    min_in_map_points: int = 40,
    min_in_map_ratio: float = 0.35,
    hit_radius_cells: int = 2,
    ray_refine_beams: int = 64,
    ray_step_m: float = 0.08,
) -> Tuple[float, float]:
    """Score a fixed pose against ``scan`` (endpoint score, ray MAE).

    Used to ask "does the *previous* pose still explain this scan?" when a
    large jump candidate appears during navigation.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 8:
        return float("-inf"), float("inf")
    if max_scan_points > 0 and scan_xy.shape[0] > max_scan_points:
        idx = np.linspace(0, scan_xy.shape[0] - 1, max_scan_points, dtype=np.int32)
        scan_xy = scan_xy[idx]
    occupied_lookup = _inflate_occupied(
        occ_map.grid >= 65, max(0, int(hit_radius_cells))
    )
    endpoint = _score_pose(
        occ_map,
        scan_xy,
        pose,
        occupied_lookup=occupied_lookup,
        min_in_map_points=min_in_map_points,
        min_in_map_ratio=min_in_map_ratio,
    )
    ray_angles, ray_ranges = _sample_scan_beams(scan, max(0, int(ray_refine_beams)))
    range_max = float(scan.range_max) if math.isfinite(scan.range_max) else 25.0
    _ray_q, ray_mae = _ray_alignment(
        occ_map,
        pose,
        ray_angles,
        ray_ranges,
        range_max_m=range_max,
        step_m=float(ray_step_m),
    )
    return float(endpoint.score), float(ray_mae)


def _map_search_bounds(occ_map: OccupancyMap) -> Tuple[float, float, float, float]:
    known = occ_map.known_mask()
    if not np.any(known):
        return (
            occ_map.origin_x,
            occ_map.origin_y,
            occ_map.origin_x + occ_map.width * occ_map.resolution,
            occ_map.origin_y + occ_map.height * occ_map.resolution,
        )
    rows, cols = np.where(known)
    min_col = int(cols.min())
    max_col = int(cols.max())
    min_row = int(rows.min())
    max_row = int(rows.max())
    min_x = occ_map.origin_x + min_col * occ_map.resolution
    max_x = occ_map.origin_x + (max_col + 1) * occ_map.resolution
    min_y = occ_map.origin_y + min_row * occ_map.resolution
    max_y = occ_map.origin_y + (max_row + 1) * occ_map.resolution
    return min_x, min_y, max_x, max_y


# The coarse grid may be no coarser than the match tolerance supports. A scan
# endpoint counts as a hit within ``hit_radius_cells`` of a wall, so a sample
# farther than that from the true pose scores like a wrong pose and is thrown
# away before the fine search sees it. On an office map (5 cm cells, 10 cm
# tolerance) the score kept about 70% of its peak 10 cm and 2° off and about
# 30% at 20 cm and 5°; a 35 cm / 10° grid lost the true pose in 46-88% of
# whole-map searches.
_YAW_STEP_RANGE_PERCENTILE = 90.0
_MIN_COARSE_YAW_STEP_DEG = 2.0
_MAX_COARSE_YAW_STEP_DEG = 12.0
# Two answers this far apart, or facing this differently, are different
# places. ``second_best_score`` and ``ambiguous`` compare the winner with the
# best different place, never with a neighbouring sample of the same peak.
_RIVAL_DISTANCE_M = 1.0
_RIVAL_YAW_DEG = 45.0
# The winner must lead the best different place by this much on the combined
# score. Over 90 whole-map searches per office run the lead had a median of
# 0.26-0.54 when the answer was right and 0.05 when it was wrong; 0.06 let
# through 1-3% wrong placements in changed scenes and 0.15 none, for 0-4
# points fewer right ones.
DEFAULT_AMBIGUITY_MARGIN = 0.15
# Different places refined by the fine search before one is chosen.
_REFINE_PLACES = 3
# Fine candidates per place that get the ray check.
_FINE_RAY_CANDIDATES = 12
_COARSE_KEEP_PER_YAW = 64
# Quarter fine steps either side of the fine winner (position and heading).
_POLISH_STEPS = np.arange(-2, 3)
_BATCH_POSES = 16384


def tolerance_grid_steps(
    scan_xy: np.ndarray, resolution: float, hit_radius_cells: int
) -> Tuple[float, float]:
    """Coarsest ``(position step m, heading step deg)`` the hit tolerance supports.

    Neighbouring position samples are ``hit_radius_cells + 1`` cells apart, so
    the true pose is never more than the tolerance from one of them along an
    axis. The heading step moves the far scan points (90th percentile range)
    by no more than the tolerance either side of a sample.
    """
    radius = max(0, int(hit_radius_cells))
    position_step = (radius + 1) * float(resolution)
    tolerance_m = (radius + 0.5) * float(resolution)
    ranges = np.hypot(scan_xy[:, 0], scan_xy[:, 1]) if scan_xy.size else np.zeros(0)
    far = float(np.percentile(ranges, _YAW_STEP_RANGE_PERCENTILE)) if ranges.size else 1.0
    yaw_step = math.degrees(2.0 * tolerance_m / max(far, 1e-3))
    return position_step, min(_MAX_COARSE_YAW_STEP_DEG, max(_MIN_COARSE_YAW_STEP_DEG, yaw_step))


class _BatchScorer:
    """Score many candidate poses at once. Same numbers as ``_score_pose``."""

    def __init__(
        self,
        occ_map: OccupancyMap,
        scan_xy: np.ndarray,
        occupied_lookup: np.ndarray,
        *,
        min_in_map_points: int,
        min_in_map_ratio: float,
    ) -> None:
        grid = occ_map.grid
        height, width = grid.shape
        self._map = occ_map
        self._xs = np.asarray(scan_xy[:, 0], dtype=float)
        self._ys = np.asarray(scan_xy[:, 1], dtype=float)
        self._total = int(scan_xy.shape[0])
        self._min_points = max(int(min_in_map_points), int(self._total * min_in_map_ratio))
        # One byte per cell: bit 0 hit (near a wall), bit 1 known free,
        # bit 2 inside the grid. The border of zeros stands for off-grid.
        code = occupied_lookup.astype(np.uint8)
        code |= ((grid >= 0) & (grid < 50)).astype(np.uint8) << 1
        code |= 4
        reach = float(np.hypot(self._xs, self._ys).max()) if self._total else 0.0
        pad = int(min(math.ceil(reach / occ_map.resolution) + 2, max(height, width) + 2))
        padded = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.uint8)
        padded[pad : pad + height, pad : pad + width] = code
        self._flat = padded.ravel()
        self._pad = pad
        self._padded_width = width + 2 * pad

    def _rotated(self, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(theta), math.sin(theta)
        return c * self._xs - s * self._ys, s * self._xs + c * self._ys

    def _scores(self, codes: np.ndarray) -> Tuple[np.ndarray, ...]:
        inside = np.count_nonzero(codes & 4, axis=1)
        hits = np.count_nonzero(codes & 1, axis=1)
        free = np.count_nonzero(codes & 2, axis=1)
        known = np.count_nonzero(codes & 3, axis=1)
        enough = (inside >= self._min_points) & (known >= self._min_points)
        denom = np.maximum(known, 1).astype(float)
        hit_rate = hits / denom
        coverage = known / float(max(1, self._total))
        score = hit_rate - 0.7 * (free / denom)
        score = np.where(
            coverage < _COVERAGE_FLOOR,
            score - _COVERAGE_PENALTY * (_COVERAGE_FLOOR - coverage),
            score,
        )
        score = np.where(enough, score, -np.inf)
        hit_rate = np.where(enough, hit_rate, 0.0)
        coverage = np.where(inside >= self._min_points, coverage, 0.0)
        return score, inside, hit_rate, coverage

    def score_lattice(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        frac_x: float,
        frac_y: float,
        theta: float,
    ) -> Tuple[np.ndarray, ...]:
        """Candidates at in-grid cells ``(rows, cols)`` sharing one sub-cell offset.

        With a common offset each scan point shifts every candidate by the
        same whole number of cells, so the lookup is one integer add.
        """
        res = self._map.resolution
        rx, ry = self._rotated(theta)
        dcol = np.clip(np.floor(frac_x + rx / res), -self._pad, self._pad).astype(np.int64)
        drow = np.clip(np.floor(frac_y + ry / res), -self._pad, self._pad).astype(np.int64)
        offsets = drow * self._padded_width + dcol
        base = (rows.astype(np.int64) + self._pad) * self._padded_width + (
            cols.astype(np.int64) + self._pad
        )
        parts = [
            self._scores(self._flat[base[i : i + _BATCH_POSES, None] + offsets[None, :]])
            for i in range(0, len(base), _BATCH_POSES)
        ]
        return tuple(np.concatenate(col) for col in zip(*parts))

    def score_xy(self, xs: np.ndarray, ys: np.ndarray, theta: float) -> Tuple[np.ndarray, ...]:
        """Candidates at arbitrary positions, one heading."""
        res = self._map.resolution
        height, width = self._map.grid.shape
        rx, ry = self._rotated(theta)
        cols = np.floor((xs[:, None] + rx[None, :] - self._map.origin_x) / res).astype(np.int64)
        rows = np.floor((ys[:, None] + ry[None, :] - self._map.origin_y) / res).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        # Index 0 is in the zero border: an off-grid point reads as off-grid.
        index = np.where(
            inside, (rows + self._pad) * self._padded_width + cols + self._pad, 0
        )
        return self._scores(self._flat[index])


def _lattice_cells(
    occ_map: OccupancyMap,
    anchor_x: float,
    anchor_y: float,
    stride: int,
    *,
    radius_m: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Known cells on a square lattice through the anchor.

    ``radius_m`` limits the lattice to a disc around the anchor; ``None``
    covers the known part of the map. Returns ``(rows, cols, frac_x, frac_y)``
    where a candidate sits at ``origin + (col + frac_x) * resolution``.
    """
    res = occ_map.resolution
    height, width = occ_map.grid.shape
    ax = (anchor_x - occ_map.origin_x) / res
    ay = (anchor_y - occ_map.origin_y) / res
    col0, row0 = math.floor(ax), math.floor(ay)
    frac_x, frac_y = ax - col0, ay - row0
    if radius_m is None:
        known = occ_map.known_mask()
        if np.any(known):
            known_rows, known_cols = np.where(known)
            lo_r, hi_r = int(known_rows.min()), int(known_rows.max())
            lo_c, hi_c = int(known_cols.min()), int(known_cols.max())
        else:
            lo_r, hi_r, lo_c, hi_c = 0, height - 1, 0, width - 1
        row_steps = np.arange(-((row0 - lo_r) // stride), (hi_r - row0) // stride + 1)
        col_steps = np.arange(-((col0 - lo_c) // stride), (hi_c - col0) // stride + 1)
    else:
        n = max(1, int(math.ceil(radius_m / (stride * res))))
        row_steps = col_steps = np.arange(-n, n + 1)
    jj, ii = np.meshgrid(row_steps, col_steps, indexing="ij")
    if radius_m is not None:
        keep = (ii * ii + jj * jj) * (stride * res) ** 2 <= radius_m * radius_m + 1e-9
        ii, jj = ii[keep], jj[keep]
    rows = (row0 + jj * stride).ravel()
    cols = (col0 + ii * stride).ravel()
    ok = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows, cols = rows[ok], cols[ok]
    ok = occ_map.grid[rows, cols] >= 0
    return rows[ok], cols[ok], float(frac_x), float(frac_y)


def _different_place(a: conv.Pose2D, b: conv.Pose2D) -> bool:
    return math.hypot(a.x - b.x, a.y - b.y) > _RIVAL_DISTANCE_M or abs(
        math.degrees(conv.normalize_angle(a.theta - b.theta))
    ) > _RIVAL_YAW_DEG


def global_localize_scan(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    *,
    hint: Optional[conv.Pose2D] = None,
    full_map: bool = False,
    search_radius_m: float = 8.0,
    coarse_position_step_m: Optional[float] = None,
    coarse_yaw_step_deg: Optional[float] = None,
    local_yaw_window_deg: float = 180.0,
    fine_position_step_m: float = 0.08,
    fine_yaw_step_deg: float = 2.0,
    fine_window_m: float = 0.6,
    fine_yaw_window_deg: float = 18.0,
    max_scan_points: int = 240,
    min_in_map_points: int = 40,
    min_in_map_ratio: float = 0.35,
    hit_radius_cells: int = 2,
    ray_refine_candidates: int = 24,
    ray_refine_beams: int = 64,
    ray_step_m: float = 0.08,
    ray_weight: float = 0.35,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> GlobalLocalizeResult:
    """Find the map-frame pose that best explains ``scan`` against ``occ_map``.

    The coarse grid defaults to the coarsest spacing the hit tolerance
    supports (``tolerance_grid_steps``). A caller on a tight time budget may
    pass a coarser one and accept that the true pose can be missed.

    The strongest distinct places from the coarse grid get the ray check, the
    best few different places are each refined, and the best refined pose
    wins. ``second_best_score`` is the best combined (endpoint + ray) score of
    a different place, and ``ambiguous`` is set when the winner leads it by
    less than ``ambiguity_margin`` — callers should not auto-apply then.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 8:
        raise ValueError(
            f"scan has too few valid returns ({scan_xy.shape[0]}); need merged lidar"
        )
    if max_scan_points > 0 and scan_xy.shape[0] > max_scan_points:
        idx = np.linspace(0, scan_xy.shape[0] - 1, max_scan_points, dtype=np.int32)
        scan_xy = scan_xy[idx]

    whole_map = bool(full_map) or hint is None
    # Whole-map: a stronger ray voice so repetitive corridors are less likely
    # to win on endpoint hits alone. Only the legacy defaults are rewritten.
    if whole_map:
        if abs(ray_weight - 0.35) < 1e-9:
            ray_weight = 0.55
        if ray_refine_candidates == 24:
            ray_refine_candidates = 48

    res = float(occ_map.resolution)
    hit_radius_cells = max(0, int(hit_radius_cells))
    default_step_m, default_yaw_deg = tolerance_grid_steps(scan_xy, res, hit_radius_cells)
    coarse_step_m = default_step_m if coarse_position_step_m is None else float(coarse_position_step_m)
    coarse_yaw_deg = default_yaw_deg if coarse_yaw_step_deg is None else float(coarse_yaw_step_deg)
    stride = max(1, int(round(coarse_step_m / res)))
    coarse_yaw = math.radians(max(1e-3, coarse_yaw_deg))

    occupied_lookup = _inflate_occupied(occ_map.grid >= 65, hit_radius_cells)
    scorer = _BatchScorer(
        occ_map,
        scan_xy,
        occupied_lookup,
        min_in_map_points=min_in_map_points,
        min_in_map_ratio=min_in_map_ratio,
    )
    ray_angles, ray_ranges = _sample_scan_beams(scan, max(0, int(ray_refine_beams)))
    range_max = float(scan.range_max) if math.isfinite(scan.range_max) else 25.0
    evaluated = 0

    def with_rays(pose: conv.Pose2D, endpoint: _PoseScore) -> Tuple[float, float, float]:
        quality, mae = _ray_alignment(
            occ_map, pose, ray_angles, ray_ranges, range_max_m=range_max, step_m=ray_step_m
        )
        return endpoint.score + ray_weight * ((2.0 * quality) - 1.0), quality, mae

    # --- Coarse grid -------------------------------------------------------
    if whole_map:
        min_x, min_y, _max_x, _max_y = _map_search_bounds(occ_map)
        rows, cols, frac_x, frac_y = _lattice_cells(
            occ_map, min_x + 0.5 * res, min_y + 0.5 * res, stride, radius_m=None
        )
        yaws = np.arange(-math.pi, math.pi - 1e-9, coarse_yaw)
    else:
        rows, cols, frac_x, frac_y = _lattice_cells(
            occ_map, hint.x, hint.y, stride, radius_m=float(search_radius_m)
        )
        half = math.radians(local_yaw_window_deg / 2.0)
        yaw_steps = max(1, int(math.ceil(half / coarse_yaw)))
        yaws = hint.theta + np.arange(-yaw_steps, yaw_steps + 1) * coarse_yaw

    coarse: List[Tuple[float, conv.Pose2D, _PoseScore]] = []
    if rows.size:
        cell_x = occ_map.origin_x + (cols + frac_x) * res
        cell_y = occ_map.origin_y + (rows + frac_y) * res
        for theta in yaws:
            theta = conv.normalize_angle(float(theta))
            score, inside, hit_rate, coverage = scorer.score_lattice(
                rows, cols, frac_x, frac_y, theta
            )
            evaluated += int(rows.size)
            take = min(_COARSE_KEEP_PER_YAW, score.size)
            best = np.argpartition(-score, take - 1)[:take] if take < score.size else np.arange(score.size)
            for i in best:
                if math.isfinite(score[i]):
                    coarse.append(
                        (
                            float(score[i]),
                            conv.Pose2D(float(cell_x[i]), float(cell_y[i]), theta),
                            _PoseScore(
                                float(score[i]), int(inside[i]), float(hit_rate[i]), float(coverage[i])
                            ),
                        )
                    )
    if not coarse:
        raise RuntimeError("global scan match failed; no valid candidate in map bounds")
    coarse.sort(key=lambda item: item[0], reverse=True)

    # Neighbouring samples of one peak would fill the shortlist; keep one each.
    nms_m = max(2.0 * stride * res, 0.3)
    nms_yaw = max(2.0 * coarse_yaw, math.radians(8.0))
    peaks: List[Tuple[conv.Pose2D, _PoseScore]] = []
    for _score, pose, endpoint in coarse:
        if all(
            math.hypot(pose.x - kept.x, pose.y - kept.y) > nms_m
            or abs(conv.normalize_angle(pose.theta - kept.theta)) > nms_yaw
            for kept, _ in peaks
        ):
            peaks.append((pose, endpoint))
            if len(peaks) >= max(1, int(ray_refine_candidates)):
                break

    ranked = sorted(
        ((with_rays(pose, endpoint)[0], pose, endpoint) for pose, endpoint in peaks),
        key=lambda item: item[0],
        reverse=True,
    )
    places: List[Tuple[float, conv.Pose2D, _PoseScore]] = []
    for item in ranked:
        if all(_different_place(item[1], chosen[1]) for chosen in places):
            places.append(item)
            if len(places) >= _REFINE_PLACES:
                break

    # --- Fine search around each place --------------------------------------
    fine_step = max(1e-3, float(fine_position_step_m))
    fine_yaw = math.radians(max(1e-3, float(fine_yaw_step_deg)))
    pos_steps = max(1, int(math.ceil(fine_window_m / fine_step)))
    offs = np.arange(-pos_steps, pos_steps + 1) * fine_step
    gx, gy = np.meshgrid(offs, offs, indexing="ij")
    in_disc = gx * gx + gy * gy <= fine_window_m * fine_window_m + 1e-9
    gx, gy = gx[in_disc], gy[in_disc]
    yaw_count = max(1, int(math.ceil(math.radians(fine_yaw_window_deg) / fine_yaw)))

    def refine(center: conv.Pose2D) -> Optional[Tuple[float, conv.Pose2D, _PoseScore, float, float]]:
        nonlocal evaluated
        xs, ys = center.x + gx, center.y + gy
        cell_cols = np.floor((xs - occ_map.origin_x) / res).astype(np.int64)
        cell_rows = np.floor((ys - occ_map.origin_y) / res).astype(np.int64)
        ok = (
            (cell_rows >= 0)
            & (cell_rows < occ_map.height)
            & (cell_cols >= 0)
            & (cell_cols < occ_map.width)
        )
        ok[ok] = occ_map.grid[cell_rows[ok], cell_cols[ok]] >= 0
        xs, ys = xs[ok], ys[ok]
        if xs.size == 0:
            return None
        found: List[Tuple[float, conv.Pose2D, _PoseScore]] = []
        for k in range(-yaw_count, yaw_count + 1):
            theta = conv.normalize_angle(center.theta + k * fine_yaw)
            score, inside, hit_rate, coverage = scorer.score_xy(xs, ys, theta)
            evaluated += int(xs.size)
            take = min(_FINE_RAY_CANDIDATES, score.size)
            best = np.argpartition(-score, take - 1)[:take] if take < score.size else np.arange(score.size)
            for i in best:
                if math.isfinite(score[i]):
                    found.append(
                        (
                            float(score[i]),
                            conv.Pose2D(float(xs[i]), float(ys[i]), theta),
                            _PoseScore(
                                float(score[i]), int(inside[i]), float(hit_rate[i]), float(coverage[i])
                            ),
                        )
                    )
        if not found:
            return None
        found.sort(key=lambda item: item[0], reverse=True)
        best_item = None
        for _score, pose, endpoint in found[:_FINE_RAY_CANDIDATES]:
            combined, quality, mae = with_rays(pose, endpoint)
            if best_item is None or combined > best_item[0]:
                best_item = (combined, pose, endpoint, quality, mae)
        # Polish between the fine samples. The peak is narrower than the fine
        # step, so without this the reported score depends on where the grid
        # happened to fall (0.80 against 0.93 for one office pose).
        anchor = best_item[1]
        px, py = np.meshgrid(_POLISH_STEPS * (fine_step / 4.0), _POLISH_STEPS * (fine_step / 4.0))
        px, py = anchor.x + px.ravel(), anchor.y + py.ravel()
        polished: List[Tuple[float, conv.Pose2D, _PoseScore]] = []
        for k in _POLISH_STEPS:
            theta = conv.normalize_angle(anchor.theta + float(k) * (fine_yaw / 4.0))
            score, inside, hit_rate, coverage = scorer.score_xy(px, py, theta)
            evaluated += int(px.size)
            i = int(np.argmax(score))
            if math.isfinite(score[i]):
                polished.append(
                    (
                        float(score[i]),
                        conv.Pose2D(float(px[i]), float(py[i]), theta),
                        _PoseScore(
                            float(score[i]), int(inside[i]), float(hit_rate[i]), float(coverage[i])
                        ),
                    )
                )
        polished.sort(key=lambda item: item[0], reverse=True)
        for _score, pose, endpoint in polished[:3]:
            combined, quality, mae = with_rays(pose, endpoint)
            if combined > best_item[0]:
                best_item = (combined, pose, endpoint, quality, mae)
        return best_item

    refined = [item for item in (refine(place[1]) for place in places) if item is not None]
    if not refined:
        raise RuntimeError("global scan match failed; no valid candidate in map bounds")
    refined.sort(key=lambda item: item[0], reverse=True)
    best_combined, best_pose, best_eval, best_ray_score, best_ray_mae = refined[0]

    # Best different place: a refined rival, or one the fine search never
    # reached (its coarse score is a lower bound on what refining would find).
    second_best = float("-inf")
    for combined, pose, *_rest in refined[1:]:
        if _different_place(pose, best_pose):
            second_best = max(second_best, combined)
    for combined, pose, _endpoint in ranked:
        if _different_place(pose, best_pose) and all(
            _different_place(pose, item[1]) for item in refined
        ):
            second_best = max(second_best, combined)
            break

    # Corridor 180° check on the winner (cheap; breaks common false peaks).
    yaw_choice = choose_yaw_or_flip(
        occ_map,
        scan,
        best_pose,
        reference_theta=hint.theta if hint is not None else None,
        hit_radius_cells=hit_radius_cells,
        max_scan_points=max_scan_points,
        ray_beams=ray_refine_beams,
        ray_step_m=ray_step_m,
        ray_weight=ray_weight,
    )
    if yaw_choice.flipped:
        best_pose = yaw_choice.pose
        best_eval = _PoseScore(
            yaw_choice.score,
            best_eval.in_map_points,
            best_eval.hit_rate,
            best_eval.coverage,
        )
        # Refresh ray quality at the flipped heading for status reporting.
        best_ray_score, best_ray_mae = _ray_alignment(
            occ_map,
            best_pose,
            ray_angles,
            ray_ranges,
            range_max_m=range_max,
            step_m=ray_step_m,
        )
        best_combined = best_eval.score + ray_weight * ((2.0 * best_ray_score) - 1.0)

    ambiguous = (
        math.isfinite(second_best)
        and math.isfinite(best_combined)
        and (best_combined - second_best) < float(ambiguity_margin)
    )

    return GlobalLocalizeResult(
        pose=best_pose,
        score=best_eval.score,
        candidates_evaluated=evaluated,
        scan_points_used=int(scan_xy.shape[0]),
        in_map_points=best_eval.in_map_points,
        hit_rate=best_eval.hit_rate,
        ray_score=best_ray_score,
        ray_mae_m=best_ray_mae,
        second_best_score=second_best,
        ambiguous=ambiguous,
        coverage=best_eval.coverage,
        combined_score=float(best_combined),
    )


@dataclass(frozen=True)
class YawFlipChoice:
    """Result of comparing ``pose`` against the same XY with yaw + π."""

    pose: conv.Pose2D
    flipped: bool
    score: float
    ray_mae_m: float
    alt_pose: conv.Pose2D
    alt_score: float
    alt_ray_mae_m: float


def choose_yaw_or_flip(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    pose: conv.Pose2D,
    *,
    reference_theta: Optional[float] = None,
    score_margin: float = 0.05,
    ray_mae_slack_m: float = 0.2,
    hit_radius_cells: int = 2,
    max_scan_points: int = 240,
    ray_beams: int = 64,
    ray_step_m: float = 0.08,
    ray_weight: float = 0.35,
) -> YawFlipChoice:
    """Break corridor 180° ambiguity by scoring ``pose`` vs the same XY at yaw+π.

    Corridors often score similarly both ways; when the two are within
    ``score_margin`` / ``ray_mae_slack_m``, prefer the yaw closer to
    ``reference_theta`` (usually the current IMU/map heading). Otherwise take
    the measurably better score.
    """
    scan_xy = scan_endpoints_base_link(scan)
    if max_scan_points > 0 and scan_xy.shape[0] > max_scan_points:
        idx = np.linspace(0, scan_xy.shape[0] - 1, max_scan_points, dtype=np.int32)
        scan_xy = scan_xy[idx]
    occupied_lookup = _inflate_occupied(
        occ_map.grid >= 65, max(0, int(hit_radius_cells))
    )
    ray_angles, ray_ranges = _sample_scan_beams(scan, max(0, int(ray_beams)))
    range_max = float(scan.range_max) if math.isfinite(scan.range_max) else 25.0

    def _eval(candidate: conv.Pose2D) -> Tuple[float, float, float]:
        ep = _score_pose(occ_map, scan_xy, candidate, occupied_lookup=occupied_lookup)
        ray_q, ray_mae = _ray_alignment(
            occ_map,
            candidate,
            ray_angles,
            ray_ranges,
            range_max_m=range_max,
            step_m=ray_step_m,
        )
        combined = float(ep.score) + ray_weight * ((2.0 * ray_q) - 1.0)
        return combined, float(ep.score), float(ray_mae)

    flip = conv.Pose2D(pose.x, pose.y, conv.normalize_angle(pose.theta + math.pi))
    c0, s0, m0 = _eval(pose)
    c1, s1, m1 = _eval(flip)

    near_tie = abs(c0 - c1) <= score_margin and abs(m0 - m1) <= ray_mae_slack_m
    prefer_flip = c1 > c0
    if near_tie and reference_theta is not None:
        d0 = abs(conv.normalize_angle(pose.theta - reference_theta))
        d1 = abs(conv.normalize_angle(flip.theta - reference_theta))
        prefer_flip = d1 < d0
    elif abs(c0 - c1) <= score_margin:
        # Score tie but MAE differs — trust the tighter ray fit.
        prefer_flip = m1 < m0

    if prefer_flip:
        return YawFlipChoice(
            pose=flip,
            flipped=True,
            score=s1,
            ray_mae_m=m1,
            alt_pose=pose,
            alt_score=s0,
            alt_ray_mae_m=m0,
        )
    return YawFlipChoice(
        pose=pose,
        flipped=False,
        score=s0,
        ray_mae_m=m0,
        alt_pose=flip,
        alt_score=s1,
        alt_ray_mae_m=m1,
    )

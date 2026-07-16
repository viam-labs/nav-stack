"""Lidar scan matching against a saved nav-stack occupancy map.

Scores candidate map-frame poses by aligning live scan endpoints with occupied
cells. Used for global localization without MiR map pose or other external seeds.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ..ros import conversions as conv


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

    def occupied_mask(self, threshold: int = 65) -> np.ndarray:
        return self.grid >= threshold

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


@dataclass(frozen=True)
class _PoseScore:
    score: float
    in_map_points: int
    hit_rate: float


@dataclass(frozen=True)
class _ScoredPose:
    pose: conv.Pose2D
    endpoint_eval: _PoseScore


def load_occupancy_from_bridge_map(map_data: Mapping) -> OccupancyMap:
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

    header_end = 0
    parts: List[bytes] = []
    for _ in range(3):
        while header_end < len(raw):
            if raw[header_end : header_end + 1] in b" \t\r\n":
                header_end += 1
                continue
            if raw[header_end : header_end + 1] == b"#":
                while header_end < len(raw) and raw[header_end : header_end + 1] != b"\n":
                    header_end += 1
                continue
            start = header_end
            while header_end < len(raw) and raw[header_end : header_end + 1] not in b" \t\r\n":
                header_end += 1
            parts.append(raw[start:header_end])
            break

    # Re-scan header more robustly.
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


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


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
        return _PoseScore(float("-inf"), in_map_points, 0.0)

    rows_in = rows[inside]
    cols_in = cols[inside]
    hit_mask = occupied_lookup[rows_in, cols_in]
    hit_rate = float(np.mean(hit_mask))

    # Endpoints landing in known free space are a mismatch.
    cell_vals = grid[rows_in, cols_in]
    free_rate = float(np.mean(cell_vals == 0))
    out_of_map_rate = 1.0 - (in_map_points / total_points)

    # Weight out-of-map heavily so edge/off-map poses cannot win.
    score = hit_rate - 0.7 * free_rate - 1.2 * out_of_map_rate
    return _PoseScore(score, in_map_points, hit_rate)


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


def _iter_pose_grid(
    *,
    center_x: float,
    center_y: float,
    center_theta: float,
    radius_m: float,
    position_step_m: float,
    yaw_step_rad: float,
    yaw_range_rad: float,
) -> Iterable[conv.Pose2D]:
    pos_steps = max(1, int(math.ceil(radius_m / position_step_m)))
    yaw_steps = max(1, int(math.ceil(yaw_range_rad / yaw_step_rad)))
    for ix in range(-pos_steps, pos_steps + 1):
        x = center_x + ix * position_step_m
        for iy in range(-pos_steps, pos_steps + 1):
            y = center_y + iy * position_step_m
            if (x - center_x) ** 2 + (y - center_y) ** 2 > radius_m**2 + 1e-9:
                continue
            for it in range(-yaw_steps, yaw_steps + 1):
                theta = _normalize_angle(center_theta + it * yaw_step_rad)
                yield conv.Pose2D(x, y, theta)


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


def _iter_full_map_poses(
    occ_map: OccupancyMap,
    *,
    position_step_m: float,
    yaw_step_rad: float,
) -> Iterable[conv.Pose2D]:
    min_x, min_y, max_x, max_y = _map_search_bounds(occ_map)
    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            theta = -math.pi
            while theta < math.pi:
                yield conv.Pose2D(x, y, theta)
                theta += yaw_step_rad
            y += position_step_m
        x += position_step_m


def global_localize_scan(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    *,
    hint: Optional[conv.Pose2D] = None,
    full_map: bool = False,
    search_radius_m: float = 8.0,
    coarse_position_step_m: float = 0.4,
    coarse_yaw_step_deg: float = 12.0,
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
) -> GlobalLocalizeResult:
    """Find the map-frame pose that best explains ``scan`` against ``occ_map``."""
    scan_xy = scan_endpoints_base_link(scan)
    if scan_xy.shape[0] < 8:
        raise ValueError(
            f"scan has too few valid returns ({scan_xy.shape[0]}); need merged lidar"
        )
    if max_scan_points > 0 and scan_xy.shape[0] > max_scan_points:
        idx = np.linspace(0, scan_xy.shape[0] - 1, max_scan_points, dtype=np.int32)
        scan_xy = scan_xy[idx]

    coarse_yaw = math.radians(coarse_yaw_step_deg)
    fine_yaw = math.radians(fine_yaw_step_deg)
    occupied_lookup = _inflate_occupied(
        occ_map.grid >= 65, max(0, int(hit_radius_cells))
    )
    ray_angles, ray_ranges = _sample_scan_beams(scan, max(0, int(ray_refine_beams)))
    range_max = float(scan.range_max) if math.isfinite(scan.range_max) else 25.0
    evaluated = 0

    def search_candidates(
        candidates: Iterable[conv.Pose2D], keep: int
    ) -> List[_ScoredPose]:
        nonlocal evaluated
        scored: List[_ScoredPose] = []
        for pose in candidates:
            if not _pose_in_known_map(occ_map, pose):
                continue
            score_eval = _score_pose(
                occ_map,
                scan_xy,
                pose,
                occupied_lookup=occupied_lookup,
                min_in_map_points=min_in_map_points,
                min_in_map_ratio=min_in_map_ratio,
            )
            evaluated += 1
            if math.isfinite(score_eval.score):
                scored.append(_ScoredPose(pose=pose, endpoint_eval=score_eval))
        scored.sort(key=lambda item: item.endpoint_eval.score, reverse=True)
        return scored[: max(1, keep)]

    def rerank_with_rays(
        scored: List[_ScoredPose],
    ) -> Tuple[conv.Pose2D, _PoseScore, float, float]:
        if not scored:
            return conv.Pose2D(0.0, 0.0, 0.0), _PoseScore(float("-inf"), 0, 0.0), 0.0, float("inf")
        best_pose = scored[0].pose
        best_eval = scored[0].endpoint_eval
        best_ray_score = 0.0
        best_ray_mae = float("inf")
        best_combined = float("-inf")
        limit = min(len(scored), max(1, int(ray_refine_candidates)))
        for item in scored[:limit]:
            ray_quality, ray_mae = _ray_alignment(
                occ_map,
                item.pose,
                ray_angles,
                ray_ranges,
                range_max_m=range_max,
                step_m=ray_step_m,
            )
            combined = item.endpoint_eval.score + ray_weight * ((2.0 * ray_quality) - 1.0)
            if combined > best_combined:
                best_combined = combined
                best_pose = item.pose
                best_eval = item.endpoint_eval
                best_ray_score = ray_quality
                best_ray_mae = ray_mae
        return best_pose, best_eval, best_ray_score, best_ray_mae

    if full_map or hint is None:
        coarse_candidates = _iter_full_map_poses(
            occ_map,
            position_step_m=coarse_position_step_m,
            yaw_step_rad=coarse_yaw,
        )
    else:
        coarse_candidates = _iter_pose_grid(
            center_x=hint.x,
            center_y=hint.y,
            center_theta=hint.theta,
            radius_m=search_radius_m,
            position_step_m=coarse_position_step_m,
            yaw_step_rad=coarse_yaw,
            yaw_range_rad=math.radians(local_yaw_window_deg / 2.0),
        )

    coarse_scored = search_candidates(
        coarse_candidates, keep=max(8, int(ray_refine_candidates))
    )
    best_pose, best_eval, _, _ = rerank_with_rays(coarse_scored)

    fine_yaw_window = math.radians(fine_yaw_window_deg)
    fine_candidates = _iter_pose_grid(
        center_x=best_pose.x,
        center_y=best_pose.y,
        center_theta=best_pose.theta,
        radius_m=fine_window_m,
        position_step_m=fine_position_step_m,
        yaw_step_rad=fine_yaw,
        yaw_range_rad=fine_yaw_window,
    )
    fine_scored = search_candidates(
        fine_candidates, keep=max(12, int(ray_refine_candidates))
    )
    best_pose, best_eval, best_ray_score, best_ray_mae = rerank_with_rays(fine_scored)

    if not math.isfinite(best_eval.score):
        raise RuntimeError(
            "global scan match failed; no valid candidate in map bounds"
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

    flip = conv.Pose2D(pose.x, pose.y, _normalize_angle(pose.theta + math.pi))
    c0, s0, m0 = _eval(pose)
    c1, s1, m1 = _eval(flip)

    near_tie = abs(c0 - c1) <= score_margin and abs(m0 - m1) <= ray_mae_slack_m
    prefer_flip = c1 > c0
    if near_tie and reference_theta is not None:
        d0 = abs(_normalize_angle(pose.theta - reference_theta))
        d1 = abs(_normalize_angle(flip.theta - reference_theta))
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

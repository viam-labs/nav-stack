"""Scan-vs-map consistency at the published pose (mid-nav loc refine)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Union

import numpy as np

from ..geom import conversions as conv
from .costmap import occupancy_from_map_dict
from .types import OccupancyGrid, Pose2D

# Body-frame labels for diagnostics: +X forward, +Y left.
_SECTORS = (
    ("forward", 0.0),
    ("left", math.pi / 2.0),
    ("right", -math.pi / 2.0),
)
_SECTOR_ASSIGN_HALF_WIDTH_RAD = math.radians(45.0)
_BEAM_STEP_RAD = math.radians(5.0)
_OCCUPIED_THRESH = 50


@dataclass(frozen=True)
class SectorClearance:
    name: str
    lidar_m: Optional[float]
    map_m: Optional[float]
    gap_m: Optional[float]
    disagree: bool
    compared: int = 0
    disagree_beams: int = 0


@dataclass(frozen=True)
class LocDisagreement:
    """Whether the published pose is a poor explanation of the current scan."""

    disagree: bool
    reason: str = ""
    compared_beams: int = 0
    disagree_beams: int = 0
    disagree_frac: float = 0.0
    worst: Optional[SectorClearance] = None
    sectors: tuple[SectorClearance, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        worst = self.worst
        return {
            "disagree": self.disagree,
            "reason": self.reason,
            "compared_beams": self.compared_beams,
            "disagree_beams": self.disagree_beams,
            "disagree_frac": round(self.disagree_frac, 3),
            "sector": None if worst is None else worst.name,
            "lidar_m": None if worst is None else worst.lidar_m,
            "map_m": None if worst is None else worst.map_m,
            "gap_m": None if worst is None else worst.gap_m,
            "sectors": {
                s.name: {
                    "lidar_m": s.lidar_m,
                    "map_m": s.map_m,
                    "gap_m": s.gap_m,
                    "disagree": s.disagree,
                    "compared": s.compared,
                    "disagree_beams": s.disagree_beams,
                }
                for s in self.sectors
            },
        }


def residual_is_lost(verdict: LocDisagreement) -> bool:
    """True when the leftover residual means the pose is really unexplained.

    A thin leftover (a few far beams, one side still matching) is not enough
    to abort a goal after a failed refine.
    """
    if not verdict.disagree:
        return False
    if verdict.compared_beams >= 10 and verdict.disagree_frac >= 0.55:
        return True
    return verdict.compared_beams >= 16 and verdict.disagree_frac >= 0.45


def small_local_match_worth_applying(result: Optional[Mapping]) -> bool:
    """A local rematch that clearly beats the published pose and is a small shift.

    Used to force-apply when ``good_match`` is just shy (e.g. 0.47 vs 0.50)
    but the prior is already bad. Never for large / ambiguous / full-map peaks.
    """
    if not result or result.get("corrected"):
        return False
    if result.get("ambiguous") or result.get("large_jump"):
        return False
    mode = str(result.get("match_mode") or "local")
    if mode and not mode.startswith("local"):
        return False
    try:
        shift_m = abs(float(result.get("shift_m") or 0.0))
        shift_deg = abs(float(result.get("shift_deg") or 0.0))
        score = float(result["score"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(score):
        return False
    if shift_m > 0.6 or shift_deg > 30.0:
        return False
    if shift_m < 0.08 and shift_deg < 6.0:
        return False
    prev = result.get("previous_score")
    try:
        prev_f = float(prev) if prev is not None else None
    except (TypeError, ValueError):
        prev_f = None
    if prev_f is not None and math.isfinite(prev_f):
        return score >= prev_f + 0.15 or (prev_f < 0.20 and score >= 0.35)
    return score >= 0.40


def occupancy_for_consistency(
    map_or_occ: Union[OccupancyGrid, Mapping, None],
) -> Optional[OccupancyGrid]:
    if map_or_occ is None:
        return None
    if isinstance(map_or_occ, OccupancyGrid):
        return map_or_occ
    if not isinstance(map_or_occ, Mapping) or map_or_occ.get("grid") is None:
        return None
    try:
        return occupancy_from_map_dict(map_or_occ)
    except (KeyError, TypeError, ValueError):
        return None


def localization_looks_bad(
    pose: Pose2D,
    scan: Optional[conv.LaserScan2D],
    map_or_occ: Union[OccupancyGrid, Mapping, None],
    *,
    margin_m: float = 0.8,
    map_max_m: float = 2.5,
    min_frac: float = 0.30,
    min_beams: int = 6,
) -> LocDisagreement:
    """True when the map near the published pose does not match the lidar.

    Samples beams around the robot. A beam counts only when occupancy claims a
    wall within ``map_max_m``. That beam votes "bad loc" when lidar is open or
    much farther (map closer by ≥ ``margin_m``). Live obstacles (lidar closer
    than the map) are ignored — those are not localization.

    Triggers when enough such map-claimed beams disagree
    (``disagree_beams / compared_beams ≥ min_frac`` and ``compared ≥ min_beams``).
    This is an overall residual, not a three-sector special case, and not a
    SLAM match score (hallway scores are often mediocre while the pose is fine).
    """
    occ = occupancy_for_consistency(map_or_occ)
    if occ is None or scan is None or scan.ranges.size == 0:
        return LocDisagreement(disagree=False)
    ranges = np.asarray(scan.ranges, dtype=float)
    inc = float(scan.angle_increment)
    if inc <= 1e-6:
        return LocDisagreement(disagree=False)
    step = max(1, int(round(_BEAM_STEP_RAD / inc)))
    compared = 0
    bad = 0
    sector_stats: dict[str, dict] = {
        name: {
            "compared": 0,
            "bad": 0,
            "lidar_open": None,
            "map_near": None,
            "gap": None,
        }
        for name, _ in _SECTORS
    }

    for i in range(0, len(ranges), step):
        raw = float(ranges[i])
        lidar_m = (
            raw
            if math.isfinite(raw) and raw >= scan.range_min
            else float("inf")
        )
        body_ang = scan.angle_min + i * inc
        map_m = _raycast_occupied_m(
            occ,
            pose,
            body_ang,
            max_m=float(map_max_m),
            occupied_thresh=_OCCUPIED_THRESH,
        )
        if map_m is None:
            continue
        compared += 1
        gap = lidar_m - map_m
        is_bad = gap >= float(margin_m)
        if is_bad:
            bad += 1
        label = _sector_label(body_ang)
        if label is None:
            continue
        st = sector_stats[label]
        st["compared"] += 1
        if is_bad:
            st["bad"] += 1
        if st["map_near"] is None or map_m < st["map_near"]:
            st["map_near"] = map_m
            st["gap"] = gap if math.isfinite(gap) else None
            st["lidar_open"] = None if not math.isfinite(lidar_m) else lidar_m

    frac = (float(bad) / float(compared)) if compared else 0.0
    trigger = (
        compared >= int(min_beams)
        and frac >= float(min_frac)
    )
    sectors: list[SectorClearance] = []
    worst: Optional[SectorClearance] = None
    for name, _center in _SECTORS:
        st = sector_stats[name]
        lidar_m = st["lidar_open"]
        map_m = st["map_near"]
        gap = st["gap"]
        sector = SectorClearance(
            name=name,
            lidar_m=None if lidar_m is None else round(float(lidar_m), 3),
            map_m=None if map_m is None else round(float(map_m), 3),
            gap_m=None if gap is None else round(float(gap), 3),
            disagree=st["compared"] > 0 and st["bad"] / st["compared"] >= float(min_frac),
            compared=int(st["compared"]),
            disagree_beams=int(st["bad"]),
        )
        sectors.append(sector)
        if sector.disagree_beams and (
            worst is None or sector.disagree_beams > worst.disagree_beams
        ):
            worst = sector
    return LocDisagreement(
        disagree=trigger,
        reason="scan_map" if trigger else "",
        compared_beams=compared,
        disagree_beams=bad,
        disagree_frac=frac,
        worst=worst,
        sectors=tuple(sectors),
    )


def lidar_map_clearance_disagree(
    pose: Pose2D,
    scan: Optional[conv.LaserScan2D],
    map_or_occ: Union[OccupancyGrid, Mapping, None],
    **kwargs,
) -> LocDisagreement:
    """Alias kept for callers; same overall scan-vs-map check."""
    kwargs.pop("lidar_min_m", None)
    return localization_looks_bad(pose, scan, map_or_occ, **kwargs)


def _sector_label(body_ang: float) -> Optional[str]:
    wrapped = (body_ang + math.pi) % (2.0 * math.pi) - math.pi
    best_name = None
    best_abs = _SECTOR_ASSIGN_HALF_WIDTH_RAD
    for name, center in _SECTORS:
        delta = abs((wrapped - center + math.pi) % (2.0 * math.pi) - math.pi)
        if delta <= best_abs:
            best_abs = delta
            best_name = name
    return best_name


def _raycast_occupied_m(
    occ: OccupancyGrid,
    pose: Pose2D,
    body_bearing_rad: float,
    *,
    max_m: float,
    occupied_thresh: int,
) -> Optional[float]:
    if max_m <= 0.0 or occ.resolution <= 0.0:
        return None
    world_theta = pose.theta + body_bearing_rad
    dx = math.cos(world_theta)
    dy = math.sin(world_theta)
    step = float(occ.resolution)
    dist = 0.0
    while dist <= max_m + 1e-9:
        row, col = occ.world_to_cell(pose.x + dx * dist, pose.y + dy * dist)
        if not occ.in_bounds(row, col):
            return None
        val = int(occ.grid[row, col])
        if val >= occupied_thresh:
            return float(dist)
        dist += step
    return None

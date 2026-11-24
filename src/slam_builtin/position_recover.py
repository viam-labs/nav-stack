"""Find where the robot is on the map it has already drawn.

While mapping, a scan that does not fit the map is not painted, so the map
stays clean when the pose is wrong. The tick can put the pose back when it is
off by a turn (any heading) or by up to about half a meter. Past that it has
no answer: every scan misses, nothing is painted, and the pose keeps drifting
on wheel odometry alone.

On one robot the pose slid 2.2 m with the heading still right. For six
minutes no scan was painted. The mapping revisit check searched with both
lidars merged into one scan, where the furniture the low lidar sees hides the
walls, scored the right place 0.10 and proposed a place 12 m away instead.
Searching with each lidar height by itself found the right place in 130 ms,
all heights within 10 cm of each other.

:func:`find_position` is that search with the checks that make its answer
safe to apply while mapping. It only says where the robot is. Nothing in the
map or the pose graph moves: while paint is held no keyframe is added, so
the error is in the live pose alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..geom import conversions as conv
from ..nav.localize.global_localize import OccupancyMap, global_localize_scan
from . import scan_match

# Rings searched around the current pose, nearest first. A slip larger than
# the last ring is left for a person: the wider the search, the more places
# in an office look alike.
RADII_M = (3.0, 6.0)
# Total heading window. Heading comes from the gyro and was right when the
# position was 2.2 m off; a wrong heading alone is recovered by the tick.
YAW_WINDOW_DEG = 60.0
# The deciding height must sit on walls this well at the new pose, and this
# much better than at the old one. 0.5 is seven returns in ten on a wall.
MIN_SCORE = 0.5
MIN_GAIN = 0.4
# With one height there is no second opinion, so it has to fit better.
MIN_SCORE_ONE_HEIGHT = 0.6
# Every other height has to agree that the new pose is better.
OTHER_HEIGHT_MIN_GAIN = 0.2
OTHER_HEIGHT_MIN_COVERAGE = 0.5
# Closer than this the tick's own matcher can reach it.
MIN_SHIFT_M = 0.3
MIN_SHIFT_RAD = math.radians(6.0)
# A height votes only with this share of the fullest height's returns, as in
# ``scan_match.height_votes``.
_MIN_RETURN_SHARE = 0.5


@dataclass(frozen=True)
class Recovery:
    """Outcome of one search. ``pose`` is set only when it is safe to apply."""

    pose: Optional[conv.Pose2D]
    reason: str
    shift_m: float = 0.0
    shift_deg: float = 0.0
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    radius_m: float = 0.0

    def as_dict(self) -> dict:
        out = {
            "recovered": self.pose is not None,
            "reason": self.reason,
            "shift_m": round(self.shift_m, 3),
            "shift_deg": round(self.shift_deg, 1),
            "radius_m": self.radius_m,
        }
        if self.score_before is not None:
            out["score_before"] = round(self.score_before, 3)
        if self.score_after is not None:
            out["score_after"] = round(self.score_after, 3)
        return out


def _agreement(
    occ_map: OccupancyMap, scan: conv.LaserScan2D, pose: conv.Pose2D
) -> tuple[Optional[float], int]:
    score, judged = scan_match.wall_agreement(occ_map, scan, pose)
    if judged <= 0 or not math.isfinite(score):
        return None, 0
    return float(score), int(judged)


def _settle(
    occ_map: OccupancyMap, scan: conv.LaserScan2D, pose: conv.Pose2D
) -> conv.Pose2D:
    """Slide a search result onto the walls.

    The search steps in 8 cm and scores a return two cells from a wall as a
    hit. The checks below count a return only within one cell, so the last
    few centimeters decide whether a right answer passes.
    """
    for window in (
        {"xy_half_m": 0.25, "xy_step_m": 0.05, "yaw_half_deg": 4.0},
        {},
    ):
        found, _score, _edge_gain = scan_match.polish_on_walls(
            occ_map, scan, pose, **window
        )
        if found is not None:
            pose = found
    return pose


def _judges(scan: conv.LaserScan2D, bands: Mapping[int, conv.LaserScan2D]) -> dict:
    """Scans that each get a say: one per height, or the merged scan alone."""
    counts = {
        band: int(scan_match.scan_endpoints_base_link(band_scan).shape[0])
        for band, band_scan in bands.items()
    }
    if counts:
        floor = _MIN_RETURN_SHARE * max(counts.values())
        usable = {band: bands[band] for band, n in counts.items() if n >= floor}
        if len(usable) >= 2:
            return usable
    return {None: scan}


def find_position(
    occ_map: OccupancyMap,
    scan: conv.LaserScan2D,
    bands: Mapping[int, conv.LaserScan2D],
    pose: conv.Pose2D,
    *,
    radii_m: Sequence[float] = RADII_M,
) -> Recovery:
    """Search around ``pose`` for the place ``scan`` was taken from.

    ``scan`` is the merged scan and ``bands`` the per-height scans of the
    same lidar read (empty with lidars at one height); ``pose`` is where the
    robot believes that read was taken.

    Each height searches by itself. The best-fitting answer is kept only if
    its search found no second place nearly as good, it sits on the walls
    much better than ``pose`` does, and every other height also fits better
    there than at ``pose``. Lidars bolted to one robot move together, so a
    wrong place that happens to suit one height does not suit the others.
    """
    judges = _judges(scan, bands)
    min_score = MIN_SCORE_ONE_HEIGHT if len(judges) == 1 else MIN_SCORE
    before = {band: _agreement(occ_map, s, pose) for band, s in judges.items()}
    # ``no_better_place``: nowhere nearby explains the scan clearly better
    # than ``pose`` does, so the pose is not what is wrong (the room changed,
    # or the map is). Any other reason: a clearly better place exists and it
    # failed a check, so the robot is probably lost and this cannot say where.
    unsure: Optional[Recovery] = None
    for radius in radii_m:
        better = []
        for band, band_scan in judges.items():
            try:
                found = global_localize_scan(
                    occ_map,
                    band_scan,
                    hint=pose,
                    search_radius_m=float(radius),
                    local_yaw_window_deg=YAW_WINDOW_DEG,
                )
            except ValueError:
                continue
            settled = _settle(occ_map, band_scan, found.pose)
            score, _judged = _agreement(occ_map, band_scan, settled)
            if score is None:
                continue
            shift_m = math.hypot(settled.x - pose.x, settled.y - pose.y)
            shift_rad = abs(conv.normalize_angle(settled.theta - pose.theta))
            score_before = before[band][0]
            if shift_m < MIN_SHIFT_M and shift_rad < MIN_SHIFT_RAD:
                continue
            if score - (score_before if score_before is not None else 0.0) < MIN_GAIN:
                continue
            if found.ambiguous:
                reason = "ambiguous"
            elif score < min_score:
                reason = "weak_fit"
            elif not _others_agree(occ_map, judges, before, band, settled):
                reason = "heights_disagree"
            else:
                reason = "recovered"
            better.append(
                Recovery(
                    settled if reason == "recovered" else None,
                    reason,
                    shift_m=shift_m,
                    shift_deg=math.degrees(shift_rad),
                    score_before=score_before,
                    score_after=score,
                    radius_m=float(radius),
                )
            )
        safe = [r for r in better if r.pose is not None]
        if safe:
            return max(safe, key=lambda r: r.score_after)
        if better:
            unsure = max(better, key=lambda r: r.score_after)
    if unsure is not None:
        return unsure
    return Recovery(None, "no_better_place", radius_m=float(radii_m[-1]))


def _others_agree(
    occ_map: OccupancyMap,
    judges: Mapping,
    before: Mapping,
    decided_by,
    pose: conv.Pose2D,
) -> bool:
    for band, band_scan in judges.items():
        if band == decided_by:
            continue
        score_before, judged_before = before[band]
        if score_before is None:
            continue
        score, judged = _agreement(occ_map, band_scan, pose)
        if score is None or judged < OTHER_HEIGHT_MIN_COVERAGE * judged_before:
            return False
        if score < score_before + OTHER_HEIGHT_MIN_GAIN:
            return False
    return True

"""Scan keyframes for builtin SLAM loop-closure map rebuild."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..ros import conversions as conv
from . import occupancy as occ
from .types import LogOddsGrid


@dataclass
class MapKeyframe:
    pose: conv.Pose2D
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float


class MapKeyframeStore:
    """Session store of map inserts for pose-correcting grid rebuild."""

    def __init__(
        self,
        *,
        max_keyframes: int = 500,
        max_beams: int = 180,
    ):
        self.max_keyframes = int(max_keyframes)
        self.max_beams = int(max_beams)
        self._frames: List[MapKeyframe] = []

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def add(self, pose: conv.Pose2D, scan: conv.LaserScan2D) -> bool:
        ranges = np.asarray(scan.ranges, dtype=float)
        if ranges.size == 0:
            return False
        if self.max_beams > 0 and ranges.size > self.max_beams:
            pick = np.linspace(0, ranges.size - 1, self.max_beams, dtype=np.int32)
            ranges = ranges[pick]
            angle_min = float(scan.angle_min)
            angle_increment = float(scan.angle_increment)
            if self.max_beams > 1:
                angle_increment = float(
                    scan.angle_increment * (pick[-1] - pick[0]) / (self.max_beams - 1)
                )
        else:
            angle_min = float(scan.angle_min)
            angle_increment = float(scan.angle_increment)
        range_max = float(scan.range_max)
        if not math.isfinite(range_max):
            range_max = 30.0
        self._frames.append(
            MapKeyframe(
                pose=conv.Pose2D(pose.x, pose.y, pose.theta),
                ranges=ranges.copy(),
                angle_min=angle_min,
                angle_increment=angle_increment,
                range_min=float(scan.range_min),
                range_max=range_max,
            )
        )
        if len(self._frames) > self.max_keyframes:
            self._frames = self._frames[-self.max_keyframes :]
        return True

    def apply_pose_delta(self, delta: conv.Pose2D) -> None:
        """Shift every stored pose by ``delta`` (map frame)."""
        self.apply_pose_delta_from(0, delta)

    def find_loop_anchor(
        self, matched: conv.Pose2D, *, radius_m: float = 1.5
    ) -> int:
        """Index of the earliest keyframe near a revisit match (loop anchor)."""
        anchor: Optional[int] = None
        for i, kf in enumerate(self._frames):
            dist = math.hypot(kf.pose.x - matched.x, kf.pose.y - matched.y)
            if dist <= radius_m:
                if anchor is None:
                    anchor = i
        return anchor if anchor is not None else 0

    def apply_pose_delta_from(self, start_index: int, delta: conv.Pose2D) -> None:
        """Shift keyframes ``start_index`` onward by ``delta``."""
        start = max(0, int(start_index))
        for i in range(start, len(self._frames)):
            kf = self._frames[i]
            self._frames[i] = MapKeyframe(
                pose=conv.compose_poses(kf.pose, delta),
                ranges=kf.ranges,
                angle_min=kf.angle_min,
                angle_increment=kf.angle_increment,
                range_min=kf.range_min,
                range_max=kf.range_max,
            )

    def rebuild_grid(
        self,
        *,
        resolution: float,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> LogOddsGrid:
        """Ray-cast all keyframes into a fresh log-odds grid."""
        grid = occ.empty_grid(resolution=float(resolution))
        total = len(self._frames)
        for i, kf in enumerate(self._frames):
            grid = occ.insert_scan(
                grid,
                kf.pose.x,
                kf.pose.y,
                kf.pose.theta,
                kf.ranges,
                kf.angle_min,
                kf.angle_increment,
                range_min=kf.range_min,
                range_max=kf.range_max,
                max_beams=0,
            )
            if progress is not None:
                progress(i + 1, total)
        return grid
